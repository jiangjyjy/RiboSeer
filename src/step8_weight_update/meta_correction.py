"""Step 8 LLM meta-correction (Section 3.6, Phase 2, of the paper; formulas 15-16).

After the EMA pass moves W toward this sample's PocketQA scores, we
optionally ask LLM for a per-tool multiplicative correction factor
γ_k ∈ [0.5, 1.5] (formula 16):

    W''[k, m, j] = γ_k · W'[k, m, j]

The LLM looks at the recent history (across many samples, same
category) and the current W slice. It either boosts (γ > 1), dampens
(γ < 1) or leaves alone (γ = 1.0) each tool. The correction factor is
a per-TOOL constant — every metric for that tool is scaled by the
same γ_k. (Formula 15 of the paper specifies γ_k, not γ_{k,m}.)

Public surface
--------------
- ``meta_correct(...)``  → ``MetaCorrectionResult`` (or ``None``).
- ``apply_correction_factors(...)``  → applies γ in-place on
  ``WeightTensor`` and returns updated weights_after snapshot.
- ``MetaCorrectionResult``  → dataclass carrying factors / rationale /
  api_usage. Used by ``run.py`` to populate ``WeightUpdateResult``.

Short-circuit rules (return None — caller treats as "no correction"):
- ``enable_meta_correction`` config flag is False.
- ``len(history) < min_history``.
- ``client is None`` (offline mode).
- The LLM call fails persistently (initial + correction retry).
- The LLM returns no factors / all-1.0 factors (no signal — same as
  not applying).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from step2_target_char.llm_client import (
    LLMClient, LLMError, extract_content, extract_json_object,
)
from step3_tool_selection.tool_registry import get_all_tool_ids
from step3_tool_selection.weight_tensor import METRICS, WeightTensor

from .prompts import build_error_correction_messages, build_messages
from .schemas import META_CORRECTION_MAX, META_CORRECTION_MIN


# ---------- response schema (LLM output validation) -----------------------


class _LLMResponse(BaseModel):
    """Strict schema for the LLM's correction response.

    Internal-only: it's the wire-format Pydantic model we use to
    validate what the LLM returns. The Public ``WeightUpdateResult``
    schema (in ``schemas.py``) wraps the validated result with extra
    snapshot / api_usage bookkeeping.
    """
    model_config = ConfigDict(extra="forbid")

    correction_factors: dict[str, float] = Field(default_factory=dict)
    rationale: str = Field(default="", max_length=4000)

    @field_validator("correction_factors")
    @classmethod
    def _factors_in_range(cls, v: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for tid, gamma in (v or {}).items():
            if not isinstance(tid, str) or not tid:
                raise ValueError("tool_id must be a non-empty string")
            try:
                g = float(gamma)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"correction_factor for {tid!r} = {gamma!r} not a number"
                ) from e
            if not (META_CORRECTION_MIN <= g <= META_CORRECTION_MAX):
                raise ValueError(
                    f"correction_factor for {tid!r} = {g} outside "
                    f"[{META_CORRECTION_MIN}, {META_CORRECTION_MAX}]"
                )
            out[tid] = g
        return out


@dataclass
class MetaCorrectionResult:
    """Outcome of one meta-correction pass.

    ``factors`` is None when the call short-circuited (disabled / not
    enough history / offline / LLM failed). When non-None, it's the
    full ``{tool_id: γ_k}`` dict the caller should multiply into W.
    ``status`` is a short tag for ``api_usage`` so step 8's record
    explains why correction was (or wasn't) applied.
    """
    factors: Optional[dict[str, float]] = None
    rationale: Optional[str] = None
    status: str = "skipped"
    api_usage: dict = field(default_factory=dict)


# ---------- helpers --------------------------------------------------------


def _safe_extract_content(response: dict) -> Optional[str]:
    try:
        return extract_content(response)
    except LLMError:
        return None


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


def _parse_response(content: str) -> tuple[Optional[_LLMResponse], Optional[str]]:
    if not content or not content.strip():
        return None, "response content is empty"
    parsed = extract_json_object(content)
    if parsed is None:
        return None, "response is not valid JSON (no {...} block found)"
    try:
        validated = _LLMResponse.model_validate(parsed)
    except ValidationError as e:
        msg = str(e)
        if len(msg) > 800:
            msg = msg[:800] + " ... (truncated)"
        return None, msg
    return validated, None


def _filter_to_known_tools(
    factors: dict[str, float],
    *,
    allowed: set[str],
) -> dict[str, float]:
    """Drop any tool_id the LLM hallucinated (not in the registry).

    Hallucinated tool ids would silently NOT apply (apply_correction_factors
    only touches W cells that exist) but would also fail the
    WeightUpdateResult schema's "factors keys ⊆ tools_updated" check
    downstream. Filter early so the rest of the pipeline sees a clean
    factor dict.
    """
    return {k: v for k, v in factors.items() if k in allowed}


# ---------- main entry point ----------------------------------------------


def meta_correct(
    weight_tensor: WeightTensor,
    category: str,
    history: Iterable[dict],
    client: Optional[LLMClient],
    config: dict,
) -> MetaCorrectionResult:
    """Optionally ask LLM for per-tool correction factors γ_k.

    Parameters
    ----------
    weight_tensor
        The W tensor *after* the EMA pass (so the LLM sees the same
        thing the next prediction will read from).
    category
        Target category j.
    history
        Iterable of recent history records (dicts) — typically loaded
        from ``data/history/prediction_history.jsonl`` and filtered to
        the same category before this call (the prompt re-filters as a
        safety net).
    client
        LLMClient. ``None`` short-circuits to the no-correction path.
    config
        Loaded ``step8_config.yaml``. Reads
        ``weight_update.{enable_meta_correction, meta_correction.*}``
        and ``api.{temperature, use_json_mode}``.

    Returns
    -------
    ``MetaCorrectionResult`` — never raises. ``.factors is None`` means
    "no correction applied" (the caller leaves W unchanged).
    """
    wu_cfg = (config.get("weight_update") or {})
    api_cfg = (config.get("api") or {})

    enabled = bool(wu_cfg.get("enable_meta_correction", False))
    if not enabled:
        return MetaCorrectionResult(status="disabled_by_config")

    mc_cfg = (wu_cfg.get("meta_correction") or {})
    min_history = int(mc_cfg.get("min_history", 5))
    history_list = list(history or [])
    same_cat = [r for r in history_list if r.get("category") == category]
    if len(same_cat) < min_history:
        return MetaCorrectionResult(
            status="insufficient_history",
            api_usage={"history_count": len(same_cat),
                       "required": min_history},
        )

    if client is None:
        return MetaCorrectionResult(status="no_client")

    # Build messages.
    try:
        tool_ids = get_all_tool_ids()
    except Exception:  # pragma: no cover — defensive
        tool_ids = []
    msgs = build_messages(
        category=category,
        history=history_list,
        weight_tensor=weight_tensor,
        tool_ids=tool_ids,
    )

    temperature = float(api_cfg.get("temperature", 0.1))
    use_json_mode = bool(api_cfg.get("use_json_mode", False))
    call_kwargs: dict[str, Any] = {"temperature": temperature}
    if use_json_mode:
        call_kwargs["response_format"] = {"type": "json_object"}

    # ---- attempt 1 -----------------------------------------------------
    try:
        r1 = client.call(msgs, **call_kwargs)
    except LLMError as e:
        return MetaCorrectionResult(
            status="api_error",
            api_usage={"failure_reason": f"API error (attempt 1): {e}",
                       **_sum_usage()},
        )
    content1 = _safe_extract_content(r1) or ""
    parsed, err = _parse_response(content1)
    if parsed is None:
        # ---- attempt 2: error correction --------------------------------
        correction = build_error_correction_messages(
            msgs, content1, err or "empty response",
        )
        try:
            r2 = client.call(correction, **call_kwargs)
        except LLMError as e:
            return MetaCorrectionResult(
                status="api_error",
                api_usage={"failure_reason": f"API error on retry: {e}",
                           **_sum_usage(r1)},
            )
        content2 = _safe_extract_content(r2) or ""
        parsed, err2 = _parse_response(content2)
        if parsed is None:
            return MetaCorrectionResult(
                status="schema_validation_failed",
                api_usage={"failure_reason": err2 or "empty response",
                           **_sum_usage(r1, r2)},
            )
        usage = _sum_usage(r1, r2)
        usage["retries"] = 1
    else:
        usage = _sum_usage(r1)
        usage["retries"] = 0

    # Filter hallucinated tool ids; they'd fail downstream anyway.
    allowed = set(get_all_tool_ids()) if tool_ids else set()
    factors = _filter_to_known_tools(
        dict(parsed.correction_factors), allowed=allowed,
    )

    if not factors:
        # Either LLM returned empty or every tool got filtered out.
        return MetaCorrectionResult(
            status="no_factors_returned",
            api_usage={**usage, "rationale": parsed.rationale},
        )

    # All-1.0 factors are equivalent to "no correction" — flag and skip
    # the in-place mutation to keep the audit trail honest.
    if all(abs(g - 1.0) < 1e-9 for g in factors.values()):
        return MetaCorrectionResult(
            status="all_neutral",
            rationale=parsed.rationale,
            api_usage={**usage, "factors_proposed": factors},
        )

    return MetaCorrectionResult(
        factors=factors,
        rationale=parsed.rationale,
        status="ok",
        api_usage=usage,
    )


# ---------- application ----------------------------------------------------


def apply_correction_factors(
    weight_tensor: WeightTensor,
    category: str,
    factors: dict[str, float],
    *,
    tool_ids: Optional[Iterable[str]] = None,
) -> dict[str, dict[str, float]]:
    """Multiply γ_k into W[k, *, j] in-place; return the new snapshot.

    Only cells that already have an explicit value (i.e. were touched
    by the EMA pass or a prior write) get multiplied — γ_k applied to
    cold-start defaults would mass-move tools that haven't been
    evaluated yet, which is not what formula 16 intends.

    Returns the updated W slice as ``{tool_id: {metric: weight_after}}``
    so the caller can store the snapshot in ``WeightUpdateResult``.
    """
    affected_tools = list(tool_ids) if tool_ids is not None else list(factors.keys())
    snap_after: dict[str, dict[str, float]] = {}
    for tid, gamma in factors.items():
        try:
            g = float(gamma)
        except (TypeError, ValueError):
            continue
        if not (META_CORRECTION_MIN <= g <= META_CORRECTION_MAX):
            # Defensive: schema already enforces this, but a buggy
            # caller shouldn't be able to silently push W out of range.
            continue
        # Read current cells through the public API. This walks the 5
        # canonical metrics so we don't blow up if W has extras.
        for metric in METRICS:
            cell = (
                weight_tensor._data.get(tid, {}).get(metric, {})  # noqa: SLF001
                if hasattr(weight_tensor, "_data") else {}
            )
            if category not in cell:
                # Cell has never been written → cold-start default.
                # Per formula 16 we do NOT multiply cold-start defaults
                # (would unfairly move untouched tools). Skip.
                continue
            old = cell[category]
            new = max(0.0, min(1.0, old * g))
            weight_tensor.update(tid, category, metric, new)

    for tid in affected_tools:
        snap_after[tid] = {
            m: weight_tensor.get_weight(tid, m, category) for m in METRICS
        }
    return snap_after
