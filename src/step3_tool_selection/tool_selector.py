"""Step 3 main logic: select tools for one sample via LLM.

Pipeline
--------
  1. Compute UCB utility scores from WeightTensor for this category
  2. Build weight summary text
  3. Assemble ToolSelectionInput
  4. build_messages → system+user
  5. client.call (JSON mode) → raw response
  6. Parse JSON + ToolPlan.model_validate
  7. Validation failure → error-correction retry (1 attempt)
  8. Persistent failure → fallback plan (all Category C, cascade)

Like step 2, `select_tools` NEVER raises. Every code path produces a
JSONL-serializable result dict.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import ValidationError

from step2_target_char.llm_client import (
    LLMClient, LLMError, extract_content, extract_json_object,
)
from .prompts import build_error_correction_messages, build_messages
from .schemas import ToolPlan, ToolSelectionInput
from .tool_registry import MANDATORY_TOOLS, is_available
from .weight_tensor import WeightTensor


# The MANDATORY guardrail is the 7-tool core from ``tool_registry`` (paper
# Table 1, chosen by forward greedy search on the training set). Every sample's
# plan is widened to include it — we append, never remove, so the LLM's own
# picks are preserved. ``is_available`` is checked at append time, so disabling
# a tool in the registry transparently drops it from the mandatory set.


def _enforce_mandatory_tools(selected: list[str]) -> list[str]:
    """Return ``selected`` with every available MANDATORY tool appended
    if not already present. Order: LLM picks first (preserved), then
    missing mandatory tools in MANDATORY_TOOLS order. De-dupes
    defensively so the ToolPlan no-duplicate invariant holds."""
    out: list[str] = []
    seen: set[str] = set()
    for t in selected:
        if t not in seen:
            out.append(t)
            seen.add(t)
    for tool in MANDATORY_TOOLS:
        if tool not in seen and is_available(tool):
            out.append(tool)
            seen.add(tool)
    return out


# Local alias preserved because tests import ``_try_extract_json`` by name.
_try_extract_json = extract_json_object


def _parse_and_validate(content: str) -> tuple[Optional[ToolPlan], Optional[str]]:
    if not content or not content.strip():
        return None, "response content is empty"
    parsed = _try_extract_json(content)
    if parsed is None:
        return None, "response is not valid JSON (no {...} block found)"
    try:
        return ToolPlan.model_validate(parsed), None
    except ValidationError as e:
        msg = str(e)
        if len(msg) > 800:
            msg = msg[:800] + " ... (truncated)"
        return None, msg


def _safe_extract_content(response: dict) -> Optional[str]:
    try:
        return extract_content(response)
    except LLMError:
        return None


# ---------- result record builders -----------------------------------------


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


def _success_record(
    sample_id: str,
    target_char: dict,
    plan: ToolPlan,
    utility_scores: dict,
    *responses: Optional[dict],
    retries: int,
) -> dict:
    # Widen the plan with the MANDATORY core before serialising.
    # validate_assignment is off on ToolPlan, so mutating the list
    # attribute won't re-trigger validators — appending only widens the
    # selected set, so the param_override ⊆ selected_tools invariant
    # still holds.
    plan.selected_tools = _enforce_mandatory_tools(plan.selected_tools)
    return {
        "sample_id": sample_id,
        "step2_category": target_char.get("category"),
        "step2_confidence": target_char.get("confidence"),
        "tool_plan": plan.model_dump(mode="json"),
        "utility_scores": utility_scores,
        "api_usage": _sum_usage(*responses),
        "timestamp": _timestamp(),
        "success": True,
        "retries": retries,
    }


def _fallback_record(
    sample_id: str,
    target_char: dict,
    config: dict,
    utility_scores: dict,
    *responses: Optional[dict],
    reason: str,
    retries: int,
) -> dict:
    fb = config.get("fallback") or {}
    plan = {
        "selected_tools": _enforce_mandatory_tools(
            list(fb.get("tools", ["equipnas", "p2rank"]))),
        "execution_strategy": fb.get("strategy", "cascade"),
        "param_overrides": [],
        "early_stop_threshold": None,
        "rationale": f"LLM tool selection failed; fallback to Category C. Reason: {reason[:200]}",
        "confidence": float(fb.get("confidence", 0.0)),
    }
    return {
        "sample_id": sample_id,
        "step2_category": target_char.get("category"),
        "step2_confidence": target_char.get("confidence"),
        "tool_plan": plan,
        "utility_scores": utility_scores,
        "api_usage": _sum_usage(*responses),
        "timestamp": _timestamp(),
        "success": False,
        "retries": retries,
        "failure_reason": reason,
    }


# ---------- main entry point ------------------------------------------------


def select_tools(
    sample_id: str,
    target_char: dict,
    target_features: dict,
    client: LLMClient,
    weight_tensor: WeightTensor,
    config: dict,
    history: Any = None,
) -> dict:
    """Run step 3 for one sample. Returns a result dict, never raises."""
    category = target_char.get("category", "novel_fold_x_unstructured")

    # 1. UCB scores
    utility_scores = weight_tensor.compute_utility_scores(category)

    # 2. Weight summary
    weight_summary = weight_tensor.get_category_summary(category)

    # 3. Build input
    inp = ToolSelectionInput(
        sample_id=sample_id,
        target_char=target_char,
        target_features=target_features,
        weight_summary=weight_summary,
        utility_scores=utility_scores,
        available_tools=list(utility_scores.keys()),
    )

    # History
    history_summary = None
    if history is not None:
        try:
            history_summary = history.get_summary()
        except Exception:
            history_summary = None

    selection_cfg = config.get("selection") or {}
    max_tools = int(selection_cfg.get("max_tools", 6))
    initial_messages = build_messages(inp, max_tools=max_tools,
                                      history_summary=history_summary)

    api_cfg = config.get("api") or {}
    temperature = float(api_cfg.get("temperature", 0.1))
    # response_format=json_object is not used by default; see
    # target_char._run_llm for the rationale.
    use_json_mode = bool(api_cfg.get("use_json_mode", False))
    call_kwargs: dict[str, Any] = {"temperature": temperature}
    if use_json_mode:
        call_kwargs["response_format"] = {"type": "json_object"}

    # --- attempt 1 --------------------------------------------------------
    try:
        r1 = client.call(initial_messages, **call_kwargs)
    except LLMError as e:
        return _fallback_record(
            sample_id, target_char, config, utility_scores,
            reason=f"API error (attempt 1): {e}", retries=0,
        )

    content1 = _safe_extract_content(r1)
    plan, err = _parse_and_validate(content1 or "")
    if plan is not None:
        return _success_record(
            sample_id, target_char, plan, utility_scores, r1, retries=0,
        )

    # --- attempt 2: error correction --------------------------------------
    correction = build_error_correction_messages(
        initial_messages, content1 or "", err or "empty response",
    )
    try:
        r2 = client.call(correction, **call_kwargs)
    except LLMError as e:
        return _fallback_record(
            sample_id, target_char, config, utility_scores, r1,
            reason=f"API error on correction retry: {e}", retries=1,
        )

    content2 = _safe_extract_content(r2)
    plan, err2 = _parse_and_validate(content2 or "")
    if plan is not None:
        return _success_record(
            sample_id, target_char, plan, utility_scores, r1, r2, retries=1,
        )

    return _fallback_record(
        sample_id, target_char, config, utility_scores, r1, r2,
        reason=f"schema validation failed on retry: {err2 or 'empty response'}",
        retries=1,
    )
