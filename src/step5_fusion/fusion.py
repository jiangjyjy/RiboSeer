"""Step 5 main logic — fuse one sample's tool predictions via LLM + noisy-OR.

Pipeline (Section 3.6, Phase 1)
-------------------------------
  1. Filter out ``success=False`` ToolPrediction objects.
  2. 0 surviving tools → return an empty CompositeResult (sample skipped
     downstream).
  3. 1 surviving tool → no LLM call, no fusion: c_k = 1.0, the tool's own
     prediction is the result.
  4. Multiple surviving tools → call LLM to assign per-tool weights c_k
     and threshold tau (ToolWeightAssignment).
  5. Run noisy-OR over the surviving predictions with those weights.
  6. Return a CompositeResult.

Three-layer error protection (mirrors step 2/3)
-----------------------------------------------
  - Transport retries  : ``LLMClient.call`` handles network / 429 / 5xx
    with exponential backoff.
  - JSON-parse retries : regex-extract a ``{...}`` block if the model
    wraps its output in markdown.
  - Schema retries     : on ``ValidationError`` we append an
    error-correction turn and call once more.
  - Final fallback     : equal weights c_k = ``fallback_weight`` (default
    0.5) for every surviving tool and threshold = 0.5. This still uses
    noisy-OR; the LLM rationale is replaced by a "fallback" string.

``fuse_predictions`` NEVER raises — every code path produces a
``CompositeResult`` so a batch run can't be killed by a single bad
sample (matches step 2/3 discipline).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from pydantic import ValidationError

from step2_target_char.llm_client import (
    LLMClient, LLMError, extract_content, extract_json_object,
)
from step3_tool_selection.weight_tensor import WeightTensor
from step4_tool_adapters.schemas import ToolPrediction

from .noisy_or import equal_weights, noisy_or_fusion
from .prompts import build_error_correction_messages, build_messages
from .schemas import CompositeResult, ToolWeightAssignment


# shared extractor in llm_client (markdown-fence aware).
_try_extract_json = extract_json_object


def _parse_and_validate(
    content: str,
    expected_tool_ids: list[str],
) -> tuple[Optional[ToolWeightAssignment], Optional[str]]:
    """Return ``(validated assignment, None)`` or ``(None, error msg)``.

    Beyond schema validation we also enforce the "every tool_id from the
    user prompt MUST be a key in weights" rule because the LLM sometimes
    drops a tool silently and noisy_or_fusion would then assign it 0.0
    without the user noticing.
    """
    if not content or not content.strip():
        return None, "response content is empty"
    parsed = _try_extract_json(content)
    if parsed is None:
        return None, "response is not valid JSON (no {...} block found)"
    try:
        assignment = ToolWeightAssignment.model_validate(parsed)
    except ValidationError as e:
        msg = str(e)
        if len(msg) > 800:
            msg = msg[:800] + " ... (truncated)"
        return None, msg

    missing = [tid for tid in expected_tool_ids if tid not in assignment.weights]
    if missing:
        return None, (
            f"weights is missing required tool_id(s): {missing}. "
            f"Every tool_id listed in the user prompt MUST be a key "
            f"(use 0.0 if you want to exclude a tool)."
        )
    return assignment, None


def _safe_extract_content(response: dict) -> Optional[str]:
    try:
        return extract_content(response)
    except LLMError:
        return None


# ---------- bookkeeping helpers --------------------------------------------


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sum_usage(*responses: Optional[dict]) -> dict:
    """Sum token usage across attempts (None entries skipped)."""
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


# ---------- core: build CompositeResult given weights ---------------------


def _build_composite(
    sample_id: str,
    surviving: list[ToolPrediction],
    assignment_weights: dict[str, float],
    threshold: float,
    rationale: str,
    confidence: float,
    api_usage: dict,
    *,
    extra_meta: Optional[dict] = None,
) -> CompositeResult:
    """Run noisy-OR with the supplied weights and pack everything into a
    CompositeResult. Helper extracted so the LLM-success and fallback
    paths share the same construction logic.
    """
    fused = noisy_or_fusion(surviving, weights=assignment_weights, threshold=threshold)

    per_nuc: Optional[dict[int, float]] = fused.get("per_nucleotide_probability") or None
    if per_nuc is not None and not per_nuc:
        # Empty dict isn't useful for downstream consumers; emit None when
        # no tool produced an RNA-side prediction (mirrors schema's
        # "optional when nothing on RNA side" semantics).
        per_nuc = None

    api_usage_full = dict(api_usage or {})
    if extra_meta:
        api_usage_full.update(extra_meta)

    return CompositeResult(
        sample_id=sample_id,
        tool_weights={tid: float(c) for tid, c in assignment_weights.items()},
        binding_protein_residues=fused["binding_protein_residues"],
        binding_rna_nucleotides=fused["binding_rna_nucleotides"],
        per_residue_probability=fused["per_residue_probability"],
        per_nucleotide_probability=per_nuc,
        threshold=float(threshold),
        fusion_rationale=rationale,
        confidence=float(confidence),
        tools_fused=[p.tool_id for p in surviving],
        api_usage=api_usage_full,
        timestamp=_timestamp(),
    )


def _empty_result(sample_id: str, reason: str) -> CompositeResult:
    """0-tool short-circuit: no tools to fuse."""
    return CompositeResult(
        sample_id=sample_id,
        tool_weights={},
        binding_protein_residues=[],
        binding_rna_nucleotides=[],
        per_residue_probability={},
        per_nucleotide_probability=None,
        threshold=0.5,
        fusion_rationale=reason,
        confidence=0.0,
        tools_fused=[],
        api_usage={"status": "empty", "reason": reason},
        timestamp=_timestamp(),
    )


def _single_tool_result(
    sample_id: str,
    pred: ToolPrediction,
) -> CompositeResult:
    """1-tool short-circuit: skip LLM, c_k = 1.0, result == that tool's prediction."""
    return _build_composite(
        sample_id=sample_id,
        surviving=[pred],
        assignment_weights={pred.tool_id: 1.0},
        threshold=0.5,
        rationale=(
            f"Only one tool ({pred.tool_id}) produced a successful "
            f"prediction; no fusion needed (c_k = 1.0)."
        ),
        confidence=0.5,
        api_usage={},
        extra_meta={"status": "single_tool", "tools_fused": [pred.tool_id]},
    )


def _fallback_result(
    sample_id: str,
    surviving: list[ToolPrediction],
    fallback_weight: float,
    *,
    reason: str,
    retries: int,
    responses: tuple = (),
) -> CompositeResult:
    """Equal-weight fallback when the LLM path can't produce a valid
    ToolWeightAssignment after the error-correction retry.

    ``responses`` is a tuple of LLM responses (each may be ``None``)
    whose ``usage`` blocks are summed into the result's ``api_usage``.
    Keyword-only after ``fallback_weight`` to avoid ambiguity between
    "another response dict" and "reason/retries".
    """
    weights = equal_weights(
        (p.tool_id for p in surviving), value=float(fallback_weight),
    )
    return _build_composite(
        sample_id=sample_id,
        surviving=surviving,
        assignment_weights=weights,
        threshold=0.5,
        rationale=(
            f"LLM weight assignment failed; falling back to equal weights "
            f"c_k = {fallback_weight:.2f} for every surviving tool. "
            f"Reason: {reason[:300]}"
        ),
        confidence=0.0,
        api_usage=_sum_usage(*responses),
        extra_meta={
            "status": "fallback",
            "failure_reason": reason,
            "retries": retries,
        },
    )


# ---------- main entry point ------------------------------------------------


def fuse_predictions(
    sample_json: dict,
    target_char: dict,
    tool_predictions: Iterable[ToolPrediction],
    client: Optional[LLMClient],
    weight_tensor: Optional[WeightTensor],
    config: dict,
    history: Any = None,
) -> CompositeResult:
    """Fuse one sample's ToolPrediction set into a single CompositeResult.

    Parameters
    ----------
    sample_json
        Raw step 1 sample dict (used only for ``sample_id``).
    target_char
        Step 2 ``TargetCharOutput`` as dict (category / analysis / etc.).
        Passed through to the LLM prompt; not strictly required for the
        fusion math.
    tool_predictions
        Iterable of step 4 ``ToolPrediction`` objects for the same sample.
    client
        LLMClient. Pass ``None`` to force the equal-weights fallback path
        (useful for offline / mock testing). Not consulted when 0 or 1
        tools survive.
    weight_tensor
        Optional WeightTensor — used to render the "Tool reliability"
        block in the user prompt. Pass ``None`` to skip that section.
    config
        Loaded ``step5_config.yaml``. Reads ``fusion.default_threshold``,
        ``fusion.fallback_weight``, ``fusion.min_tools_for_llm``,
        ``api.temperature``.
    history
        Optional ``PredictionHistory``. ``get_summary()`` is called and
        the result injected into the system prompt (best-effort — any
        exception just suppresses history).

    Never raises. Always returns a ``CompositeResult``.
    """
    sample_id = sample_json.get("sample_id") or "unknown"

    fusion_cfg = (config.get("fusion") or {})
    fallback_weight = float(fusion_cfg.get("fallback_weight", 0.5))
    min_for_llm = int(fusion_cfg.get("min_tools_for_llm", 2))
    default_threshold = float(fusion_cfg.get("default_threshold", 0.5))

    # 1. Filter success=False
    surviving = [p for p in tool_predictions if getattr(p, "success", False)]

    # 2. 0 tools
    if not surviving:
        return _empty_result(sample_id, "no successful tool predictions to fuse")

    # 3. 1 tool (or below the LLM threshold) → short-circuit
    if len(surviving) < max(min_for_llm, 1) or len(surviving) == 1:
        # Always handle single tool with the dedicated path (c=1.0); for
        # 0 < n < min_for_llm we still pick the single path if n == 1.
        if len(surviving) == 1:
            return _single_tool_result(sample_id, surviving[0])
        # n surviving but min_for_llm > n > 1 → unusual config; treat as
        # equal-weights fallback so we still produce a result.
        return _fallback_result(
            sample_id, surviving, fallback_weight,
            reason=(
                f"only {len(surviving)} tools survived but config requires "
                f"min_tools_for_llm={min_for_llm}; using equal weights"
            ),
            retries=0,
        )

    # 4. Multi-tool path: call LLM
    if client is None:
        return _fallback_result(
            sample_id, surviving, fallback_weight,
            reason="LLMClient is None (offline mode); equal weights",
            retries=0,
        )

    weight_summary = None
    if weight_tensor is not None:
        category = (target_char or {}).get("category") or "novel_fold_x_unstructured"
        try:
            weight_summary = weight_tensor.get_category_summary(category)
        except Exception:
            weight_summary = None

    history_summary = None
    if history is not None:
        try:
            history_summary = history.get_summary()
        except Exception:
            history_summary = None

    initial_messages = build_messages(
        sample_id=sample_id,
        target_char=target_char,
        predictions=surviving,
        weight_summary=weight_summary,
        history_summary=history_summary,
    )

    api_cfg = config.get("api") or {}
    temperature = float(api_cfg.get("temperature", 0.1))
    # see target_char._run_llm for rationale. Default off; if
    # config flips it on, LLMClient still falls back gracefully on a 400.
    use_json_mode = bool(api_cfg.get("use_json_mode", False))
    call_kwargs: dict[str, Any] = {"temperature": temperature}
    if use_json_mode:
        call_kwargs["response_format"] = {"type": "json_object"}

    expected_tool_ids = [p.tool_id for p in surviving]

    # --- attempt 1 --------------------------------------------------------
    try:
        r1 = client.call(initial_messages, **call_kwargs)
    except LLMError as e:
        return _fallback_result(
            sample_id, surviving, fallback_weight,
            reason=f"API error (attempt 1): {e}",
            retries=0,
        )

    content1 = _safe_extract_content(r1)
    assignment, err = _parse_and_validate(content1 or "", expected_tool_ids)
    if assignment is not None:
        return _build_composite(
            sample_id=sample_id,
            surviving=surviving,
            assignment_weights=assignment.weights,
            threshold=assignment.threshold,
            rationale=assignment.rationale,
            confidence=assignment.confidence,
            api_usage=_sum_usage(r1),
            extra_meta={"status": "ok", "retries": 0},
        )

    # --- attempt 2: error correction --------------------------------------
    correction = build_error_correction_messages(
        initial_messages, content1 or "", err or "empty response",
    )
    try:
        r2 = client.call(correction, **call_kwargs)
    except LLMError as e:
        return _fallback_result(
            sample_id, surviving, fallback_weight,
            reason=f"API error on correction retry: {e}",
            retries=1,
            responses=(r1,),
        )

    content2 = _safe_extract_content(r2)
    assignment, err2 = _parse_and_validate(content2 or "", expected_tool_ids)
    if assignment is not None:
        return _build_composite(
            sample_id=sample_id,
            surviving=surviving,
            assignment_weights=assignment.weights,
            threshold=assignment.threshold,
            rationale=assignment.rationale,
            confidence=assignment.confidence,
            api_usage=_sum_usage(r1, r2),
            extra_meta={"status": "ok", "retries": 1},
        )

    return _fallback_result(
        sample_id, surviving, fallback_weight,
        reason=(
            f"schema validation failed on retry: {err2 or 'empty response'}"
        ),
        retries=1,
        responses=(r1, r2),
    )


# ---------- ground-truth comparison ---------------------------------------


def evaluate_against_ground_truth(
    composite: CompositeResult,
    sample_json: dict,
) -> dict:
    """Compare the fused binding set to step 1 ground truth.

    Returns a dict with ``ground_truth_protein``, ``precision``, ``recall``,
    ``f1``. Empty-or-missing ground truth produces zeros (and logs nothing
    — the caller can decide whether to surface that).
    """
    interaction = (sample_json.get("interaction") or {})
    gt = interaction.get("binding_protein_residues") or []
    gt_set = set(int(i) for i in gt)
    pred_set = set(composite.binding_protein_residues or [])

    if not gt_set and not pred_set:
        return {
            "ground_truth_protein": [],
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
        }

    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "ground_truth_protein": sorted(gt_set),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ---------- output record builder -----------------------------------------


def build_output_record(
    composite: CompositeResult,
    sample_json: dict,
    target_char: dict,
) -> dict:
    """Flatten ``CompositeResult`` + step 2 cat + ground-truth metrics into
    the JSONL record format documented in the step 5 spec.
    """
    cd = composite.model_dump(mode="json")

    # JSON keys must be strings; per_residue_probability has int keys.
    if cd.get("per_residue_probability"):
        cd["per_residue_probability"] = {
            str(k): v for k, v in cd["per_residue_probability"].items()
        }
    if cd.get("per_nucleotide_probability"):
        cd["per_nucleotide_probability"] = {
            str(k): v for k, v in cd["per_nucleotide_probability"].items()
        }

    record = {
        "sample_id": composite.sample_id,
        "step2_category": (target_char or {}).get("category"),
        "tools_fused": composite.tools_fused,
        "tool_weights": cd["tool_weights"],
        "threshold": composite.threshold,
        "binding_protein_residues": composite.binding_protein_residues,
        "binding_rna_nucleotides": composite.binding_rna_nucleotides,
        "per_residue_probability": cd["per_residue_probability"],
        "per_nucleotide_probability": cd.get("per_nucleotide_probability"),
        "fusion_rationale": composite.fusion_rationale,
        "confidence": composite.confidence,
        "api_usage": composite.api_usage,
        "timestamp": composite.timestamp,
    }

    record.update(evaluate_against_ground_truth(composite, sample_json))
    return record
