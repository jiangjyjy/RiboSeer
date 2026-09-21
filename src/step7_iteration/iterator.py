"""Step 7 main logic — iterative accept / refine / restart loop (Section 3.8 of the paper).

Pipeline per call to ``run_iteration_loop``
-------------------------------------------
For one sample, given the initial step-4 / step-5 / step-6 outputs:

  1. Iteration 0: ask LLM what to do given the current PocketQA score.
     - ``accept``  → record the iteration, terminate (reason="accepted").
     - ``refine``  → drop the named tool from the prediction set, re-fuse
       (step 5), re-score (step 6), record the new score.
     - ``restart`` → drop the *worst* tool by average per-residue
       confidence, re-fuse, re-score.
  2. Iterate. Termination rules:
     - LLM picks accept                       → reason="accepted"
     - |score_after - score_before| < conv    → reason="converged" (one
       extra synthesised accept record is appended so the schema's
       "last action == accept" invariant holds)
     - we hit the iteration cap               → reason="max_iterations"
       (last slot's action is forced to accept; no extra synthesised
       record so total stays at the cap)
  3. Result: ``IterationResult`` with the full trajectory.

Modes
-----
- ``lightweight`` (default; MVP): refine / restart never re-run tools;
  they manipulate the existing ``tool_predictions`` list and call step 5
  + step 6 with ``client=None`` (equal-weights fusion, no LLM tokens
  burned in the inner re-fuse). Fast and deterministic.
- ``full``: re-runs the tool. Not implemented in the MVP — raises
  ``NotImplementedError`` so we don't silently fall back to lightweight.

Three-layer error protection (mirrors step 5)
---------------------------------------------
- Transport retries  : ``LLMClient.call`` already handles network/429/5xx.
- JSON-parse retries : regex-extract a ``{...}`` block (markdown-fence
  aware via ``extract_json_object``).
- Schema retries     : on ``ValidationError`` we append an
  error-correction turn and call once more.
- Final fallback     : synthesise an accept action with rationale
  "LLM unavailable" so the loop still produces a valid IterationResult.

``run_iteration_loop`` NEVER raises — every code path produces an
``IterationResult`` so a batch run can't be killed by a single bad
sample (matches step 2/3/5 discipline).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from pydantic import ValidationError

from step2_target_char.llm_client import (
    LLMClient, LLMError, extract_content, extract_json_object,
)
from step4_tool_adapters.schemas import ToolPrediction
from step5_fusion.fusion import fuse_predictions
from step5_fusion.schemas import CompositeResult
from step6_pocket_qa.scorer import score_prediction
from step6_pocket_qa.schemas import PocketQAResult

from .prompts import build_error_correction_messages, build_messages
from .schemas import (
    IterationAction, IterationRecord, IterationResult, TERMINATION_VALUES,
)


# ---------- bookkeeping helpers --------------------------------------------


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sum_usage(*responses: Optional[dict]) -> dict:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for r in responses:
        if not isinstance(r, dict):
            continue
        u = r.get("usage") or {}
        for k in totals:
            v = u.get(k)
            if isinstance(v, int):
                totals[k] += v
    return totals


def _safe_extract_content(response: dict) -> Optional[str]:
    try:
        return extract_content(response)
    except LLMError:
        return None


# ---------- LLM call → IterationAction (with retry) ------------------------


def _parse_and_validate(content: str) -> tuple[Optional[IterationAction], Optional[str]]:
    """Best-effort decode of the LLM response into an IterationAction."""
    if not content or not content.strip():
        return None, "response content is empty"
    parsed = extract_json_object(content)
    if parsed is None:
        return None, "response is not valid JSON (no {...} block found)"
    try:
        action = IterationAction.model_validate(parsed)
    except ValidationError as e:
        msg = str(e)
        if len(msg) > 800:
            msg = msg[:800] + " ... (truncated)"
        return None, msg
    return action, None


def _ask_llm(
    *,
    sample_id: str,
    target_char: Optional[dict],
    qa_result: PocketQAResult,
    composite_result: CompositeResult,
    history: list[IterationRecord],
    iteration_index: int,
    max_iterations: int,
    client: LLMClient,
    api_cfg: dict,
) -> tuple[Optional[IterationAction], dict, Optional[str]]:
    """One LLM round-trip with one error-correction retry.

    Returns ``(action, api_usage, error_message)``. ``action`` is None
    only when both attempts failed; ``error_message`` then carries the
    last failure reason (used by the fallback synthesiser).
    """
    initial = build_messages(
        sample_id=sample_id,
        target_char=target_char,
        qa_result=qa_result,
        composite_result=composite_result,
        history=history,
        iteration_index=iteration_index,
        max_iterations=max_iterations,
    )

    temperature = float(api_cfg.get("temperature", 0.1))
    use_json_mode = bool(api_cfg.get("use_json_mode", False))
    call_kwargs: dict[str, Any] = {"temperature": temperature}
    if use_json_mode:
        call_kwargs["response_format"] = {"type": "json_object"}

    # ---- attempt 1 -------------------------------------------------------
    try:
        r1 = client.call(initial, **call_kwargs)
    except LLMError as e:
        return None, _sum_usage(), f"API error (attempt 1): {e}"

    content1 = _safe_extract_content(r1) or ""
    action, err = _parse_and_validate(content1)
    if action is not None:
        return action, _sum_usage(r1), None

    # ---- attempt 2: error correction ------------------------------------
    correction = build_error_correction_messages(initial, content1, err or "empty response")
    try:
        r2 = client.call(correction, **call_kwargs)
    except LLMError as e:
        return None, _sum_usage(r1), f"API error on correction retry: {e}"

    content2 = _safe_extract_content(r2) or ""
    action, err2 = _parse_and_validate(content2)
    if action is not None:
        return action, _sum_usage(r1, r2), None

    return None, _sum_usage(r1, r2), (
        f"schema validation failed on retry: {err2 or 'empty response'}"
    )


def _synth_accept(rationale: str, *, confidence: float = 0.5) -> IterationAction:
    """Build a synthesised accept action — used by every fallback / forced path."""
    # Schema requires rationale length >= 10. Defensive pad.
    if len(rationale) < 10:
        rationale = (rationale + " (loop terminated)")[:4000]
    return IterationAction(
        action="accept",
        rationale=rationale[:4000],
        confidence=max(0.0, min(1.0, confidence)),
    )


# ---------- action application: lightweight mode ---------------------------


def _avg_per_residue(pred: ToolPrediction) -> float:
    """Mean per-residue confidence used to rank tools for restart's drop."""
    prc = getattr(pred, "per_residue_confidence", None) or {}
    if not prc:
        return 0.0
    vals = list(prc.values())
    if not vals:
        return 0.0
    avg = sum(vals) / len(vals)
    # Cat A reports pLDDT (0-100); normalise so all units share one scale.
    if getattr(pred, "category", "") == "A":
        avg = avg / 100.0
    return float(avg)


def _select_worst_tool(predictions: list[ToolPrediction]) -> Optional[str]:
    """Pick the lowest-avg-confidence successful tool — restart's drop target.

    Failed tools are NOT eligible (already excluded from fusion, dropping
    them would be a no-op). Ties broken by tool_id for determinism.
    """
    eligible = [
        (p.tool_id, _avg_per_residue(p))
        for p in predictions
        if getattr(p, "success", False)
    ]
    if not eligible:
        return None
    eligible.sort(key=lambda x: (x[1], x[0]))
    return eligible[0][0]


def _refuse_and_score(
    *,
    sample_id: str,
    sample_json: dict,
    target_char: Optional[dict],
    surviving: list[ToolPrediction],
    fusion_config: dict,
    qa_config: dict,
) -> tuple[CompositeResult, PocketQAResult]:
    """Lightweight re-fuse + re-score on the supplied tool subset.

    ``client=None`` is passed to ``fuse_predictions`` deliberately:
    lightweight mode runs equal-weights noisy-OR (no LLM call inside
    the inner re-fuse). The outer loop still pays for one LLM call per
    iteration — this just keeps the inner cost zero so a refine/restart
    is cheap and deterministic.
    """
    composite = fuse_predictions(
        sample_json=sample_json,
        target_char=target_char or {},
        tool_predictions=surviving,
        client=None,
        weight_tensor=None,
        config=fusion_config,
    )
    qa = score_prediction(
        sample_json=sample_json,
        composite_result=composite,
        tool_predictions=surviving,
        config=qa_config,
    )
    return composite, qa


def _apply_action(
    action: IterationAction,
    *,
    sample_id: str,
    sample_json: dict,
    target_char: Optional[dict],
    current_predictions: list[ToolPrediction],
    fusion_config: dict,
    qa_config: dict,
    mode: str,
) -> tuple[Optional[CompositeResult], Optional[PocketQAResult],
           list[ToolPrediction], Optional[str]]:
    """Apply a non-accept action and return the new state.

    Returns ``(new_composite, new_qa, new_predictions, degrade_reason)``.
    When the action degenerates (refine of an absent tool, restart with
    nothing left to drop), ``new_*`` are None and ``degrade_reason``
    explains why the caller should record an accept instead.
    """
    if mode != "lightweight":
        # We could plug in a real "full" mode here in the future; for now
        # raise loudly so a config typo doesn't silently downgrade.
        raise NotImplementedError(
            f"iteration mode {mode!r} is not implemented; only "
            f"'lightweight' is supported in the MVP"
        )

    if action.action == "refine":
        target = action.refine_tool or ""
        present = [p.tool_id for p in current_predictions
                   if getattr(p, "success", False)]
        if target not in present:
            return None, None, current_predictions, (
                f"refine_tool {target!r} is not in the current successful "
                f"prediction set {present}; cannot refine — degraded to accept"
            )
        new_preds = [p for p in current_predictions
                     if not (getattr(p, "tool_id", None) == target
                             and getattr(p, "success", False))]
        # If dropping leaves zero tools, refine cannot help.
        if not any(getattr(p, "success", False) for p in new_preds):
            return None, None, current_predictions, (
                f"dropping {target!r} would leave zero successful tools; "
                f"degraded to accept"
            )
    elif action.action == "restart":
        worst = _select_worst_tool(current_predictions)
        if worst is None:
            return None, None, current_predictions, (
                "no eligible tool to drop for restart — degraded to accept"
            )
        new_preds = [p for p in current_predictions
                     if not (getattr(p, "tool_id", None) == worst
                             and getattr(p, "success", False))]
        if not any(getattr(p, "success", False) for p in new_preds):
            return None, None, current_predictions, (
                f"dropping worst tool {worst!r} would leave zero successful "
                f"tools; degraded to accept"
            )
    else:
        raise AssertionError(
            f"_apply_action called with action={action.action!r}; "
            f"only refine/restart should reach here"
        )

    composite, qa = _refuse_and_score(
        sample_id=sample_id,
        sample_json=sample_json,
        target_char=target_char,
        surviving=new_preds,
        fusion_config=fusion_config,
        qa_config=qa_config,
    )
    return composite, qa, new_preds, None


# ---------- main entry point -----------------------------------------------


def run_iteration_loop(
    sample_json: dict,
    target_char: Optional[dict],
    tool_predictions: list[ToolPrediction],
    composite_result: CompositeResult,
    qa_result: PocketQAResult,
    client: Optional[LLMClient],
    config: dict,
    fusion_config: Optional[dict] = None,
    qa_config: Optional[dict] = None,
) -> IterationResult:
    """Run the accept/refine/restart loop for one sample.

    Parameters
    ----------
    sample_json
        Step-1 sample dict (provides ``sample_id`` + sequences for
        re-scoring).
    target_char
        Step-2 target_char dict (for prompt + q5 category).
    tool_predictions
        Step-4 ``ToolPrediction`` list (initial set; refine/restart may
        prune it).
    composite_result
        Step-5 ``CompositeResult`` for the initial state.
    qa_result
        Step-6 ``PocketQAResult`` for the initial state — its
        ``total_score`` seeds the trajectory.
    client
        ``LLMClient`` for action decisions. Pass ``None`` to force the
        fallback path (synthesised accept on iteration 0); useful for
        offline smoke runs.
    config
        Loaded ``step7_config.yaml``. Reads
        ``iteration.{max_iterations, convergence_threshold, mode}`` and
        ``api.{temperature, use_json_mode}``.
    fusion_config / qa_config
        Loaded ``step5_config.yaml`` / ``step6_config.yaml``. Required
        when refine/restart actually trigger an inner re-fuse + re-score.
        Default ``{}`` — the inner functions tolerate empty configs.

    Never raises. Bad inputs / API failures surface as a synthesised
    accept iteration with a descriptive rationale.
    """
    fusion_config = fusion_config or {}
    qa_config = qa_config or {}

    iter_cfg = (config.get("iteration") or {})
    api_cfg = (config.get("api") or {})
    max_iter = int(iter_cfg.get("max_iterations", 3))
    if max_iter < 1:
        max_iter = 1
    conv_thr = float(iter_cfg.get("convergence_threshold", 0.02))
    mode = str(iter_cfg.get("mode", "lightweight"))

    sample_id = composite_result.sample_id

    # Mutable per-iteration state.
    current_preds: list[ToolPrediction] = list(tool_predictions)
    current_comp: CompositeResult = composite_result
    current_qa: PocketQAResult = qa_result
    current_score: float = float(qa_result.total_score)

    records: list[IterationRecord] = []
    trajectory: list[float] = []
    termination: Optional[str] = None

    for iter_idx in range(max_iter):
        is_last_slot = (iter_idx == max_iter - 1)

        # ----- decide ------------------------------------------------------
        if client is None:
            action: IterationAction = _synth_accept(
                "LLMClient is None (offline mode); accepting current prediction "
                "without further iteration."
            )
            api_usage = {"status": "no_client"}
        else:
            llm_action, api_usage, err = _ask_llm(
                sample_id=sample_id,
                target_char=target_char,
                qa_result=current_qa,
                composite_result=current_comp,
                history=records,
                iteration_index=iter_idx,
                max_iterations=max_iter,
                client=client,
                api_cfg=api_cfg,
            )
            if llm_action is None:
                action = _synth_accept(
                    f"LLM action could not be parsed; accepting current "
                    f"prediction. Reason: {err or 'unknown error'}"
                )
                api_usage = {**api_usage, "status": "fallback",
                             "failure_reason": err or "unknown"}
            else:
                action = llm_action
                api_usage = {**api_usage, "status": "ok"}

        # ----- act ---------------------------------------------------------
        if action.action == "accept":
            records.append(IterationRecord(
                iteration=iter_idx,
                action=action,
                score_before=current_score,
                score_after=current_score,
                delta=0.0,
                api_usage=api_usage,
                timestamp=_timestamp(),
            ))
            trajectory.append(current_score)
            termination = "accepted"
            break

        # refine / restart on the last slot: don't actually mutate state;
        # synthesise an accept with the LLM's reasoning preserved in the
        # record's ``api_usage.suppressed_action`` field for traceability.
        if is_last_slot:
            forced = _synth_accept(
                f"Iteration cap ({max_iter}) reached; forcing accept of the "
                f"current prediction. (LLM proposed {action.action!r} but "
                f"there is no slot left to apply it.)",
                confidence=action.confidence,
            )
            records.append(IterationRecord(
                iteration=iter_idx,
                action=forced,
                score_before=current_score,
                score_after=current_score,
                delta=0.0,
                api_usage={**api_usage, "suppressed_action": action.action,
                           "suppressed_reason": (
                               action.refine_reason or action.restart_reason
                               or "")[:300]},
                timestamp=_timestamp(),
            ))
            trajectory.append(current_score)
            termination = "max_iterations"
            break

        # Apply refine / restart in lightweight mode.
        try:
            new_comp, new_qa, new_preds, degrade = _apply_action(
                action,
                sample_id=sample_id,
                sample_json=sample_json,
                target_char=target_char,
                current_predictions=current_preds,
                fusion_config=fusion_config,
                qa_config=qa_config,
                mode=mode,
            )
        except Exception as e:  # noqa: BLE001 — never let inner crashes kill the loop
            forced = _synth_accept(
                f"Action {action.action!r} crashed during application "
                f"({type(e).__name__}: {e}); accepting current prediction.",
            )
            records.append(IterationRecord(
                iteration=iter_idx,
                action=forced,
                score_before=current_score,
                score_after=current_score,
                delta=0.0,
                api_usage={**api_usage, "status": "apply_error",
                           "error": f"{type(e).__name__}: {e}"[:300]},
                timestamp=_timestamp(),
            ))
            trajectory.append(current_score)
            termination = "accepted"
            break

        if degrade is not None:
            # Couldn't apply (e.g. refine_tool not in current set). Record
            # this slot as accept with the degrade reason — the LLM's
            # original action is preserved in api_usage for traceability.
            forced = _synth_accept(
                f"Could not apply {action.action!r}: {degrade}",
                confidence=action.confidence,
            )
            records.append(IterationRecord(
                iteration=iter_idx,
                action=forced,
                score_before=current_score,
                score_after=current_score,
                delta=0.0,
                api_usage={**api_usage, "status": "degraded",
                           "suppressed_action": action.action,
                           "degrade_reason": degrade[:300]},
                timestamp=_timestamp(),
            ))
            trajectory.append(current_score)
            termination = "accepted"
            break

        # Successful refine / restart.
        new_score = float(new_qa.total_score)
        delta = new_score - current_score

        records.append(IterationRecord(
            iteration=iter_idx,
            action=action,
            score_before=current_score,
            score_after=new_score,
            delta=delta,
            api_usage=api_usage,
            timestamp=_timestamp(),
        ))
        trajectory.append(new_score)

        # Promote new state for the next iteration.
        current_preds = new_preds
        current_comp = new_comp
        current_qa = new_qa
        current_score = new_score

        # ----- convergence check ------------------------------------------
        if abs(delta) < conv_thr:
            # Synthesise a final accept record so the schema's
            # "last action == accept" invariant holds.
            synth = _synth_accept(
                f"Convergence reached: |delta|={abs(delta):.4f} < threshold "
                f"{conv_thr:.4f}; further iteration is unlikely to help.",
            )
            records.append(IterationRecord(
                iteration=iter_idx + 1,
                action=synth,
                score_before=new_score,
                score_after=new_score,
                delta=0.0,
                api_usage={"status": "convergence_synth"},
                timestamp=_timestamp(),
            ))
            trajectory.append(new_score)
            termination = "converged"
            break

    # Should never happen — every loop branch sets ``termination`` and
    # appends at least one record. Defensive synthesis in case the loop
    # body is changed without updating this guard.
    if termination is None:  # pragma: no cover
        synth = _synth_accept(
            "Loop ended without explicit termination; defensive accept synthesised."
        )
        records.append(IterationRecord(
            iteration=0, action=synth,
            score_before=current_score, score_after=current_score, delta=0.0,
            api_usage={"status": "defensive_synth"},
            timestamp=_timestamp(),
        ))
        trajectory.append(current_score)
        termination = "accepted"

    assert termination in TERMINATION_VALUES, termination

    return IterationResult(
        sample_id=sample_id,
        final_action="accept",
        total_iterations=len(records),
        final_score=trajectory[-1] if trajectory else current_score,
        score_trajectory=trajectory,
        iterations=records,
        termination_reason=termination,
        final_binding_protein_residues=list(current_comp.binding_protein_residues),
        final_binding_rna_nucleotides=list(current_comp.binding_rna_nucleotides),
        timestamp=_timestamp(),
    )
