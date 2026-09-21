"""Step 2 main logic: characterize one sample via LLM.

Pipeline
--------
  1. extract_target_features(sample_json)  → TargetFeatures
  2. build_messages(features)              → system+user messages
  3. client.call(messages, json_mode)      → raw response
  4. extract_content(response) + try_parse_json  → dict
  5. TargetCharOutput.model_validate(dict) → structured output
  6. On JSON/validation failure: one error-correction retry
  7. On persistent failure: return a well-formed fallback record

Error discipline: `characterize_target` NEVER raises. Every code path produces
a JSONL-serializable result dict so a batch run over 45k samples can't be
killed by a single bad sample.

Retry budget
------------
- Transport retries (network / 429 / 5xx): 3 inside `LLMClient.call`, default
  exponential backoff. If those exhaust, `LLMClient` raises `LLMError` and
  we go straight to fallback — no point double-retrying transport failures.
- Schema retries (JSON parse / pydantic validation): 1 error-correction turn
  appended to the conversation. The correction prompt quotes the specific
  validation error verbatim so the model can target the fix.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import ValidationError

from .feature_adapter import extract_target_features
from .llm_client import (
    LLMClient, LLMError, extract_content, extract_json_object,
)
from .prompts import build_error_correction_messages, build_messages
from .schemas import TargetCharOutput, TargetFeatures


# shared extractor lives in llm_client.extract_json_object so
# step 2 / 3 / 5 stay byte-for-byte consistent on JSON parsing (including
# markdown-fence handling for the no-json_object-mode endpoint). Local
# alias kept because tests import this private name.
_try_extract_json = extract_json_object


def _parse_and_validate(content: str) -> tuple[Optional[TargetCharOutput], Optional[str]]:
    """Return (validated output, None) or (None, error_message)."""
    if not content or not content.strip():
        return None, "response content is empty"
    parsed = _try_extract_json(content)
    if parsed is None:
        return None, "response is not valid JSON (no {...} block found)"
    try:
        return TargetCharOutput.model_validate(parsed), None
    except ValidationError as e:
        # pydantic v2's str(e) is already verbose enough for the correction
        # prompt; trim to keep the correction message short.
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
    """Sum usage across attempts. Unknown fields stay 0, not None, so the
    record is always JSON-clean and downstream cost accounting works."""
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


def _base_record(features: TargetFeatures) -> dict:
    return {
        "sample_id": features.sample_id,
        "input_features": features.model_dump(),
        "timestamp": _timestamp(),
    }


def _success_record(
    features: TargetFeatures,
    output: TargetCharOutput,
    *responses: Optional[dict],
    retries: int,
) -> dict:
    rec = _base_record(features)
    rec.update({
        "output": output.model_dump(mode="json"),
        "api_usage": _sum_usage(*responses),
        "success": True,
        "retries": retries,
    })
    return rec


def _fallback_record(
    features: TargetFeatures,
    config: dict,
    *responses: Optional[dict],
    reason: str,
    retries: int,
) -> dict:
    """Emit a well-formed record when the LLM path failed.

    Uses the config's `fallback` section for category + confidence + notes,
    so the user can tune the fallback without editing source.
    """
    fb = config.get("fallback") or {}
    domain = fb.get("protein_domain", "novel_fold")
    rna_struct = fb.get("rna_structure", "unstructured")
    category = f"{domain}_x_{rna_struct}"
    notes_fmt = fb.get("notes") or "LLM characterization failed after retries"
    output = {
        "analysis": (
            "LLM characterization failed after retries; fallback category emitted. "
            f"Failure reason: {reason[:200]}"
        ),
        "protein_domain": domain,
        "rna_structure": rna_struct,
        "category": category,
        "confidence": float(fb.get("confidence", 0.0)),
        "notes": f"{notes_fmt}; cause: {reason[:200]}",
    }
    rec = _base_record(features)
    rec.update({
        "output": output,
        "api_usage": _sum_usage(*responses),
        "success": False,
        "retries": retries,
        "failure_reason": reason,
    })
    return rec


# ---------- main entry point ------------------------------------------------


def characterize_target(
    sample_json: dict,
    client: LLMClient,
    config: dict,
    history: Optional[Any] = None,
) -> dict:
    """Run step 2 for one sample. Returns a result dict, never raises.

    `history` is a `PredictionHistory`-like object with `get_summary()`
    (stage 2.4); None disables the historical-context block.
    """
    features = extract_target_features(sample_json)

    history_summary = None
    if history is not None:
        try:
            history_summary = history.get_summary()
        except Exception:
            # History is an advisory input; never let it kill the request.
            history_summary = None

    initial_messages = build_messages(features, history_summary=history_summary)

    api_cfg = config.get("api") or {}
    temperature = float(api_cfg.get("temperature", 0.1))
    # ``response_format=json_object`` is rejected by the
    # coding-plan endpoint that ships llm-model. Default off so we don't
    # waste a round trip on the LLMClient fallback. The system prompt
    # already requires JSON-only output, and extract_json_object handles
    # markdown fences if the model adds any.
    use_json_mode = bool(api_cfg.get("use_json_mode", False))
    call_kwargs: dict[str, Any] = {"temperature": temperature}
    if use_json_mode:
        call_kwargs["response_format"] = {"type": "json_object"}

    # --- attempt 1 --------------------------------------------------------
    try:
        r1 = client.call(initial_messages, **call_kwargs)
    except LLMError as e:
        return _fallback_record(
            features, config, reason=f"API error (attempt 1): {e}", retries=0,
        )

    content1 = _safe_extract_content(r1)
    output, err = _parse_and_validate(content1 or "")
    if output is not None:
        return _success_record(features, output, r1, retries=0)

    # --- attempt 2: error-correction turn --------------------------------
    correction = build_error_correction_messages(
        initial_messages, content1 or "", err or "empty response",
    )
    try:
        r2 = client.call(correction, **call_kwargs)
    except LLMError as e:
        return _fallback_record(
            features, config, r1,
            reason=f"API error on correction retry: {e}", retries=1,
        )

    content2 = _safe_extract_content(r2)
    output, err2 = _parse_and_validate(content2 or "")
    if output is not None:
        return _success_record(features, output, r1, r2, retries=1)

    return _fallback_record(
        features, config, r1, r2,
        reason=f"schema validation failed on retry: {err2 or 'empty response'}",
        retries=1,
    )
