"""Prompts for step 8 LLM meta-correction (Section 3.6, Phase 2, of the paper; formulas 15-16).

Same LLM endpoint conventions as steps 2 / 3 / 5 / 7:
``temperature=0.1``, ``use_json_mode=false`` (the deployed llm-model
endpoint rejects ``response_format=json_object``), so the schema is
spelled out in the system prompt and the LLM emits one bare JSON
object.

Three builders:
  - ``build_messages(...)`` — initial call (system + user)
  - ``build_error_correction_messages(...)`` — appends one
    assistant-bad-output + user-correction turn (step 2/3/5/7 pattern)
  - ``format_user_prompt(...)`` — internal helper, exposed for tests

Goal of the LLM call:
  Given the current category-slice of W and the recent history of
  PocketQA scores per tool, return a per-tool correction factor
  γ_k ∈ [0.5, 1.5]. γ > 1 boosts a tool that's been consistently
  outperforming its current weight; γ < 1 dampens a tool that's been
  unreliable. γ = 1.0 means "leave it alone".

Token budget
------------
System ≈ 800 tokens, user ≈ 400-1500 tokens depending on history
length. Total well under 3000 even with 20 history records (limit_n
in summarise_history is the ceiling).
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

# Defensive import: keep prompts usable in unit tests that don't have
# the full registry on PYTHONPATH (matches step 5 / step 7 pattern).
try:
    from step3_tool_selection.tool_registry import (  # type: ignore
        get_all_tool_ids,
    )
except Exception:  # pragma: no cover
    def get_all_tool_ids(*, include_unavailable: bool = False) -> list[str]:
        return ["boltz2", "chai1", "p2rank", "equipnas", "haddock3"]


from .schemas import META_CORRECTION_MAX, META_CORRECTION_MIN


# ---------- system prompt ---------------------------------------------------

SYSTEM_PROMPT = """\
You are a tool-reliability calibration expert for an RNA-protein \
binding-site prediction pipeline. After each prediction, the system \
maintains a weight tensor W[tool, metric, category] that tracks how \
reliable each tool is on each category. An EMA pass has already \
moved the weights toward this sample's PocketQA scores. Your job is \
to apply a multiplicative correction γ_k per tool, based on the \
RECENT HISTORY across many samples in the same category.

# When to boost (γ > 1)

- The tool's PocketQA scores have been consistently HIGH over the \
last N runs in this category, but its weight is still low (cold \
start, or recent EMA hasn't caught up).
- The tool's predictions have driven repeated successful (high \
final_score) iterations.

# When to dampen (γ < 1)

- The tool's PocketQA scores have been consistently LOW over the \
last N runs in this category.
- The tool was named in step-7 refines that DID NOT improve the \
score (delta <= 0) — past refines failed.
- The tool has been failing (success=False) more often than it \
succeeds on this category.

# When to leave alone (γ = 1.0)

- Not enough history to judge (the system already filters this case \
out before calling you, but if the data looks ambiguous, return 1.0).
- EMA has clearly already caught up; further multiplicative tweak \
would be noise.

# Constraints

- γ_k MUST satisfy %(MIN).2f <= γ_k <= %(MAX).2f. Values outside this \
range will be rejected and you will be asked to retry.
- Return a γ for EVERY tool in the "Available tools" list, even if \
the value is 1.0. Tools you don't return are treated as 1.0 by the \
caller, but explicit is better.
- Use the rationale field to explain your reasoning briefly (2-4 \
sentences). Reference specific tools and trends, not platitudes.

# Available tools

%(TOOL_IDS_LINE)s

# Output JSON schema (strict)

Return ONE JSON object with exactly these keys:

{
  "correction_factors": {
    "<tool_id>": <number in [%(MIN).2f, %(MAX).2f]>,
    ...
  },
  "rationale": <string, English, 2-4 sentences explaining the \
correction trends across tools>
}

Rules:
- Output ONLY the JSON object. No markdown, no code fences, no \
commentary before or after.
- `correction_factors` keys MUST be a subset of the "Available \
tools" list above.
- All `correction_factors` values MUST be numbers in \
[%(MIN).2f, %(MAX).2f].
- `rationale` MUST be in English.
"""


def _system_prompt() -> str:
    """Inject the live tool registry + the canonical clamp range."""
    tool_ids = get_all_tool_ids()
    return SYSTEM_PROMPT % {
        "TOOL_IDS_LINE": ", ".join(tool_ids),
        "MIN": META_CORRECTION_MIN,
        "MAX": META_CORRECTION_MAX,
    }


# ---------- snapshot helpers (consumed by the prompt) ----------------------


def _fmt_score(v: Any) -> str:
    if v is None:
        return "--"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def summarise_current_weights(
    weight_tensor: Any,
    category: str,
    *,
    tool_ids: Optional[Iterable[str]] = None,
) -> str:
    """Render the W[*, *, j] slice as a per-tool summary block.

    Accepts either a real ``WeightTensor`` (uses ``get_weights`` /
    ``get_count``) or a duck-typed mock that exposes the same methods.
    Returns one line per tool: ``id  count=N  metrics: q1=.., q2=..``.
    """
    if tool_ids is None:
        try:
            tool_ids = get_all_tool_ids()
        except Exception:  # pragma: no cover
            tool_ids = []
    lines = []
    for tid in tool_ids:
        try:
            w = weight_tensor.get_weights(tid, category)
            n = weight_tensor.get_count(tid, category)
        except Exception:
            continue
        # 5 metrics, abbreviated to q1..q5 in display order to keep the
        # block compact (the names are spelled out in the system prompt).
        order = (
            ("q1", "structural_plausibility"),
            ("q2", "physicochemical_complementarity"),
            ("q3", "evolutionary_conservation"),
            ("q4", "cross_tool_consensus"),
            ("q5", "known_motif_consistency"),
        )
        cells = ", ".join(f"{label}={_fmt_score(w.get(name))}"
                          for label, name in order)
        lines.append(f"  {tid}  evals={n}  [{cells}]")
    if not lines:
        return "  (no weight data available)"
    return "\n".join(lines)


def summarise_history(
    history: Iterable[dict],
    category: Optional[str] = None,
    *,
    limit: int = 15,
) -> str:
    """Render recent history records as one line per past run.

    Each record (dict) is expected to carry at least:
      sample_id, category, scores (q1..q5 + total), tools_used,
      final_action, final_score, timestamp
    Missing keys render as ``?``. Records are filtered by ``category``
    when given (we want recent behaviour on the SAME category, not
    cross-category noise).
    """
    items = list(history or [])
    if category:
        items = [r for r in items if r.get("category") == category]
    if not items:
        return "(no recent history available)"
    items = items[-limit:]
    lines = [f"Recent {len(items)} run(s) on category '{category or 'any'}':"]
    for r in items:
        sid = r.get("sample_id", "?")
        final = _fmt_score(r.get("final_score"))
        action = r.get("final_action", "?")
        scores = r.get("scores") or {}
        tools = r.get("tools_used") or []
        if isinstance(tools, list):
            tools_str = ",".join(str(t) for t in tools[:6])
        else:
            tools_str = str(tools)
        # Compact per-q render — 5 numbers max, useful for spotting
        # trends without dumping the full info dict.
        per_q = " ".join(
            f"q{i+1}={_fmt_score(scores.get(name))}"
            for i, name in enumerate((
                "structural_plausibility",
                "physicochemical_complementarity",
                "evolutionary_conservation",
                "cross_tool_consensus",
                "known_motif_consistency",
            ))
        )
        lines.append(
            f"  - {sid}: final={final} action={action}  "
            f"tools=[{tools_str}]  {per_q}"
        )
    return "\n".join(lines)


# ---------- user prompt + message builders ---------------------------------


def format_user_prompt(
    category: str,
    history: Iterable[dict],
    weight_tensor: Any,
    *,
    tool_ids: Optional[Iterable[str]] = None,
    history_limit: int = 15,
) -> str:
    """Build the user-prompt body."""
    sections = [
        f"Target category: {category}",
        "",
        "# Current weight slice (W[*, *, this category])",
        summarise_current_weights(
            weight_tensor, category, tool_ids=tool_ids,
        ),
        "",
        "# Recent history (per-sample summary)",
        summarise_history(history, category=category, limit=history_limit),
        "",
        "Decide a γ_k correction factor for each tool and return the "
        "JSON object as specified in the system prompt.",
    ]
    return "\n".join(sections)


def build_messages(
    category: str,
    history: Iterable[dict],
    weight_tensor: Any,
    *,
    tool_ids: Optional[Iterable[str]] = None,
    history_limit: int = 15,
) -> list[dict]:
    """Assemble system + user messages for the LLM chat API."""
    user = format_user_prompt(
        category=category,
        history=history,
        weight_tensor=weight_tensor,
        tool_ids=tool_ids,
        history_limit=history_limit,
    )
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user},
    ]


def build_error_correction_messages(
    prev_messages: list[dict],
    bad_output: str,
    error_message: str,
) -> list[dict]:
    """Append assistant's bad output + user correction turn (steps 2/3/5/7 pattern)."""
    correction = (
        "Your previous response failed validation.\n\n"
        f"Validation error:\n{error_message.strip()}\n\n"
        "Return ONE corrected JSON object with these keys:\n"
        "  - `correction_factors`: {tool_id: number}, each value in "
        f"[{META_CORRECTION_MIN:.2f}, {META_CORRECTION_MAX:.2f}]\n"
        "  - `rationale`: English, 2-4 sentences\n"
        "Tool ids MUST be from the 'Available tools' list. "
        "No markdown, no code fences."
    )
    return [
        *prev_messages,
        {"role": "assistant", "content": bad_output},
        {"role": "user", "content": correction},
    ]


# ---------- token estimate -------------------------------------------------


def estimate_prompt_tokens(messages: list[dict]) -> int:
    """Rough char/4 estimate (matches step 5 / 7 heuristic)."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return total_chars // 4
