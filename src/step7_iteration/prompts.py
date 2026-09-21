"""Prompt templates for Step 7 — accept / refine / restart decision (Section 3.8 of the paper).

Design
------
Same LLM endpoint and conventions as steps 2 / 3 / 5: ``temperature=0.1``,
``use_json_mode=false`` (the deployed llm-model endpoint rejects
``response_format=json_object``), so the schema is spelled out in the
system prompt and the LLM is asked to emit one bare JSON object.

Three builders, mirroring step 5:
  - ``build_messages(...)`` — initial call (system + user)
  - ``build_error_correction_messages(...)`` — appends one
    assistant-bad-output + user-correction turn (step 2/3/5 pattern)
  - ``format_user_prompt(...)`` — internal helper, exposed for
    tests / debugging / golden files

Two analysis helpers consumed by the prompt:
  - ``summarize_qa(qa_result)`` — render the 5 sub-scores + most
    diagnostic ``info`` keys per metric; the LLM uses these to pick
    *which* metric to refine on, not just whether the total is low.
  - ``summarize_history(iterations)`` — compact one-line-per-iter
    trajectory; lets the LLM avoid repeating a refine that already
    failed.

Token budget
------------
System ≈ 1100 tokens, user ≈ 400-700 tokens. Total well under the
2000-token budget set by the spec; the slack absorbs long histories
and verbose ``info`` dicts.

Null / empty handling
---------------------
- Missing sub-scores (q_m=None, e.g. consensus abstain when no tools
  succeeded) render as ``q_m: -- (abstained: <reason>)`` so the LLM
  treats them as "no signal" rather than "score=0".
- Iteration 0 (no prior history) renders the history block as
  ``(this is the first iteration; no history yet)``.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

# Lazy / defensive imports so unit tests can pass plain dicts and
# duck-typed mocks without dragging in the upstream packages. Same
# pattern as step 5's prompts.py.
try:
    from step3_tool_selection.tool_registry import (  # type: ignore
        get_all_tool_ids,
        get_tool,
    )
except Exception:  # pragma: no cover
    def get_all_tool_ids(*, include_unavailable: bool = False) -> list[str]:
        return ["boltz2", "chai1", "p2rank", "equipnas", "haddock3"]

    def get_tool(tool_id: str):  # type: ignore
        raise KeyError(tool_id)


# ---------- system prompt ---------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert reviewer of RNA-protein binding-site predictions. \
For one sample, you are given the current PocketQA quality scores \
(Section 3.7 of the paper) and the iteration history so far. Decide what to do next: \
accept the prediction, refine one tool, or restart with a fresh tool plan.

# Action menu

- accept   -- the prediction is good enough; stop iterating.
- refine   -- a single sub-score is the bottleneck and re-running ONE \
specific tool is likely to fix it.
- restart  -- multiple sub-scores are low and the current tool set is \
mismatched to the target; a fresh tool plan is needed.

# When to pick each action

Lean toward `accept` when ANY of these holds:
  - total_score > 0.80
  - all 5 sub-scores > 0.6 (consistent quality across criteria)
  - a refine in a previous iteration failed to improve the score \
(delta <= 0); refining again is unlikely to help

Pick `refine` when ONE sub-score is clearly the bottleneck (others are \
acceptable):
  - q1 (structural_plausibility) low  → refine a Cat A tool \
(boltz2 / chai1): re-predicting the structure can fix \
non-compact / non-contiguous interfaces.
  - q2 (physicochemical_complementarity) low → DO NOT refine; this \
score reflects sequence-intrinsic chemistry and re-running tools will \
not change it. Prefer accept.
  - q3 (evolutionary_conservation) low → DO NOT refine; same reason \
(sequence-intrinsic). Prefer accept.
  - q4 (cross_tool_consensus) low → refine the disagreeing tool. The \
qa info dict shows pairwise Jaccard and per-tool vote ratios; pick the \
tool with the LOWEST agreement and refine it.
  - q5 (known_motif_consistency) low → refine the Cat C tool \
(equipnas) — its threshold / parameters may need tuning to capture the \
expected motif positions.

Pick `restart` when MULTIPLE sub-scores are low (≥ 3 of 5 below 0.4) \
AND no single tool is responsible. This signals that the chosen tool \
plan does not fit the target.

# History rules (avoid wasted work)

- If a prior iteration refined tool X and the score did not improve \
(delta <= 0.02), do NOT refine tool X again this turn. Either pick a \
different action, refine a different tool, or accept.
- If two prior refines both failed to improve the score, lean strongly \
toward `accept` -- further refinement is unlikely to help.
- Restart is expensive; only choose it on iteration 0 or after refines \
have repeatedly failed.

# Available tool_ids (for the `refine_tool` field)

%(TOOL_IDS_LINE)s

The `refine_tool` field MUST be one of the above ids. You may pick a \
tool that is currently not in the prediction set (it will be added on \
the next iteration); but in lightweight mode the loop will just drop \
the named tool and re-fuse without it, so prefer naming a tool that IS \
present.

# Output JSON schema (strict)

Return ONE JSON object with exactly these keys:

{
  "action":         <"accept" | "refine" | "restart">,
  "refine_tool":    <string tool_id; REQUIRED iff action == "refine"; \
omit or set to null otherwise>,
  "refine_reason":  <string, one sentence; REQUIRED iff action == \
"refine"; omit or set to null otherwise>,
  "restart_reason": <string, one sentence; REQUIRED iff action == \
"restart"; omit or set to null otherwise>,
  "rationale":      <string, English, 2-4 sentences explaining your \
decision in terms of the sub-scores and history>,
  "confidence":     <number in [0.0, 1.0]; YOUR self-assessment of the \
decision's quality>
}

Rules:
- Output ONLY the JSON object. No markdown, no code fences, no \
commentary before or after.
- `rationale` must be in English.
- For `accept`, all three of `refine_tool` / `refine_reason` / \
`restart_reason` MUST be omitted or null.
- For `refine`, BOTH `refine_tool` and `refine_reason` MUST be set; \
`restart_reason` MUST be omitted or null.
- For `restart`, `restart_reason` MUST be set; `refine_tool` and \
`refine_reason` MUST be omitted or null.
"""


def _system_prompt() -> str:
    """Inject the live tool registry into the system prompt."""
    tool_ids = get_all_tool_ids()
    return SYSTEM_PROMPT % {"TOOL_IDS_LINE": ", ".join(tool_ids)}


# ---------- QA summary -----------------------------------------------------


# Per-metric whitelist of the most decision-relevant ``info`` keys. We
# only surface these in the prompt -- the rest of the dict is debug
# noise that wastes tokens.
_INFO_KEYS_BY_METRIC: dict[str, tuple[str, ...]] = {
    "structural_plausibility": (
        "n_clusters", "interface_ratio", "rg", "expected_rg",
    ),
    "physicochemical_complementarity": (
        "positive_ratio", "aromatic_ratio", "polar_ratio", "gp_ratio",
    ),
    "evolutionary_conservation": (
        "rare_ratio", "terminal_share", "pI",
    ),
    "cross_tool_consensus": (
        "n_active_tools", "avg_vote_ratio", "avg_jaccard",
        "pairwise_jaccard",
    ),
    "known_motif_consistency": (
        "domain", "n_motifs_found", "coverage_ratio",
    ),
}

_METRIC_ORDER: tuple[str, ...] = (
    "structural_plausibility",
    "physicochemical_complementarity",
    "evolutionary_conservation",
    "cross_tool_consensus",
    "known_motif_consistency",
)


def _fmt_score(v: Any) -> str:
    if v is None:
        return "--"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_info(info: dict, allowed: tuple[str, ...]) -> str:
    parts = []
    for k in allowed:
        if k in info and info[k] is not None:
            v = info[k]
            if isinstance(v, dict):
                # pairwise_jaccard is a small dict; render compact.
                inner = ", ".join(f"{kk}={vv}" for kk, vv in list(v.items())[:6])
                if len(v) > 6:
                    inner += f", +{len(v) - 6} more"
                parts.append(f"{k}={{{inner}}}")
            elif isinstance(v, float):
                parts.append(f"{k}={v:.3f}")
            else:
                parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else "(no detail)"


def summarize_qa(qa_result: Any) -> str:
    """Render the step-6 PocketQAResult for the user prompt.

    Accepts either a ``PocketQAResult`` Pydantic model or its dict form
    (so the prompt builder can be called from CLI, tests, and the
    iterator without re-validating).
    """
    qa = qa_result.model_dump() if hasattr(qa_result, "model_dump") else dict(qa_result)
    total = qa.get("total_score", 0.0)
    n = qa.get("n_metrics_computed", 0)
    weights = qa.get("weights_used", {}) or {}

    lines = [
        f"total_score = {_fmt_score(total)}  "
        f"({n}/5 metrics computed)",
    ]
    for name in _METRIC_ORDER:
        score = qa.get(name)
        detail = (qa.get("details") or {}).get(name) or {}
        info = detail.get("info") or {}
        if not detail.get("computed", score is not None):
            reason = (
                detail.get("error")
                or info.get("reason")
                or "no signal"
            )
            lines.append(f"  {name}: -- (abstained: {reason})")
            continue
        weight = weights.get(name)
        weight_str = f", weight={weight:.2f}" if weight is not None else ""
        keys = _INFO_KEYS_BY_METRIC.get(name, ())
        info_str = _fmt_info(info, keys)
        lines.append(
            f"  {name}: {_fmt_score(score)}{weight_str}  [{info_str}]"
        )
    return "\n".join(lines)


# ---------- prediction summary ---------------------------------------------


def summarize_prediction(composite_result: Any) -> str:
    """One-block summary of step-5 fused output for the prompt."""
    cr = (
        composite_result.model_dump()
        if hasattr(composite_result, "model_dump")
        else dict(composite_result)
    )
    bps = cr.get("binding_protein_residues") or []
    brn = cr.get("binding_rna_nucleotides") or []
    weights = cr.get("tool_weights") or {}
    threshold = cr.get("threshold")
    confidence = cr.get("confidence")

    head_n = 20
    bps_str = (
        f"{bps[:head_n]}{' ...' if len(bps) > head_n else ''}"
    )
    lines = [
        f"binding(protein): {len(bps)} residues -- {bps_str}",
        f"binding(rna)    : {len(brn)} nucleotides",
        f"threshold (tau) : {_fmt_score(threshold)}",
        f"fusion confidence: {_fmt_score(confidence)}",
    ]
    if weights:
        w_str = ", ".join(
            f"{k}={v:.2f}" for k, v in sorted(weights.items())
        )
        lines.append(f"tool_weights    : {w_str}")
    return "\n".join(lines)


# ---------- history summary ------------------------------------------------


def summarize_history(history: Optional[Iterable]) -> str:
    """Render iteration history as one line per past iteration.

    Accepts either an iterable of ``IterationRecord`` Pydantic models
    or dicts. Returns a placeholder string when history is empty so the
    prompt stays well-formed on iteration 0.
    """
    items = list(history or [])
    if not items:
        return "(this is the first iteration; no history yet)"

    lines = []
    for rec in items:
        d = rec.model_dump() if hasattr(rec, "model_dump") else dict(rec)
        action = d.get("action") or {}
        if hasattr(action, "model_dump"):
            action = action.model_dump()
        a_name = action.get("action", "?")
        before = _fmt_score(d.get("score_before"))
        after = _fmt_score(d.get("score_after"))
        delta = d.get("delta")
        delta_str = (
            "n/a" if delta is None
            else f"{delta:+.3f}"
        )
        extras = []
        if a_name == "refine":
            tool = action.get("refine_tool") or "?"
            extras.append(f"tool={tool}")
            reason = action.get("refine_reason")
            if reason:
                extras.append(f"reason={reason[:80]}")
        elif a_name == "restart":
            reason = action.get("restart_reason")
            if reason:
                extras.append(f"reason={reason[:80]}")
        extras_str = (" -- " + "; ".join(extras)) if extras else ""
        lines.append(
            f"  iter {d.get('iteration', '?')}: action={a_name}  "
            f"score: {before} -> {after}  delta={delta_str}{extras_str}"
        )
    return "\n".join(lines)


# ---------- target-char rendering (compact) --------------------------------


def _fmt_target_char(tc: Optional[dict]) -> str:
    if not tc:
        return "(step 2 characterization not available)"
    cat = tc.get("category", "unknown")
    conf = tc.get("confidence", "?")
    return f"category: {cat}  (confidence={conf})"


# ---------- user prompt + message builders ---------------------------------


def format_user_prompt(
    sample_id: str,
    target_char: Optional[dict],
    qa_result: Any,
    composite_result: Any,
    history: Optional[Iterable] = None,
    iteration_index: int = 0,
    max_iterations: int = 3,
) -> str:
    """Build the user-prompt body from the iterator's per-turn inputs."""
    sections = [
        f"Sample ID: {sample_id}",
        f"Iteration: {iteration_index} of max {max_iterations}",
        "",
        "# Target",
        _fmt_target_char(target_char),
        "",
        "# Current PocketQA scores (step 6)",
        summarize_qa(qa_result),
        "",
        "# Current prediction (step 5)",
        summarize_prediction(composite_result),
        "",
        "# Iteration history",
        summarize_history(history),
        "",
        "Decide the next action and return the JSON object as specified "
        "in the system prompt.",
    ]
    return "\n".join(sections)


def build_messages(
    sample_id: str,
    target_char: Optional[dict],
    qa_result: Any,
    composite_result: Any,
    history: Optional[Iterable] = None,
    iteration_index: int = 0,
    max_iterations: int = 3,
) -> list[dict]:
    """Assemble system + user messages for the LLM chat API."""
    user = format_user_prompt(
        sample_id=sample_id,
        target_char=target_char,
        qa_result=qa_result,
        composite_result=composite_result,
        history=history,
        iteration_index=iteration_index,
        max_iterations=max_iterations,
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
    """Append assistant's bad output + user correction turn (step 2/3/5 pattern).

    The correction echoes the strict per-action requirements so the
    model can localise its mistake even without re-reading the system
    prompt.
    """
    correction = (
        "Your previous response failed validation.\n\n"
        f"Validation error:\n{error_message.strip()}\n\n"
        "Return ONE corrected JSON object with these keys:\n"
        "  - `action`: one of \"accept\" / \"refine\" / \"restart\"\n"
        "  - `refine_tool`: string tool_id (REQUIRED iff action == \"refine\")\n"
        "  - `refine_reason`: string (REQUIRED iff action == \"refine\")\n"
        "  - `restart_reason`: string (REQUIRED iff action == \"restart\")\n"
        "  - `rationale`: English, 2-4 sentences\n"
        "  - `confidence`: number in [0, 1]\n"
        "Action-specific fields not required by the chosen action MUST "
        "be omitted or null. No markdown, no code fences."
    )
    return [
        *prev_messages,
        {"role": "assistant", "content": bad_output},
        {"role": "user", "content": correction},
    ]


# ---------- token estimate -------------------------------------------------


def estimate_prompt_tokens(messages: list[dict]) -> int:
    """Rough char/4 estimate (matches step 5's heuristic)."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return total_chars // 4


# ---------- demo -----------------------------------------------------------


def _demo_main() -> None:
    """Render messages for a synthetic 2-iteration scenario without API."""
    import sys
    from pathlib import Path

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    REPO = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO / "src"))

    sample_id = "1un6_B_F"

    target_char = {
        "category": "RRM_x_stem_loop",
        "confidence": 0.9,
    }

    # Mock step-5 composite result (dict form to keep demo dependency-free).
    composite = {
        "sample_id": sample_id,
        "binding_protein_residues": [10, 11, 12, 23, 24, 25],
        "binding_rna_nucleotides": [3, 4, 5, 6],
        "tool_weights": {"boltz2": 0.7, "p2rank": 0.4, "equipnas": 0.6},
        "threshold": 0.5,
        "confidence": 0.72,
    }

    # Mock step-6 QA result with 4/5 metrics computed and q4 abstained
    # (consensus dropped because only one tool was active in this scenario).
    qa_result = {
        "sample_id": sample_id,
        "structural_plausibility": 0.82,
        "physicochemical_complementarity": 0.65,
        "evolutionary_conservation": 0.71,
        "cross_tool_consensus": 0.42,
        "known_motif_consistency": 0.55,
        "total_score": 0.62,
        "n_metrics_computed": 5,
        "weights_used": {
            "structural_plausibility": 0.25,
            "physicochemical_complementarity": 0.20,
            "evolutionary_conservation": 0.15,
            "cross_tool_consensus": 0.25,
            "known_motif_consistency": 0.15,
        },
        "details": {
            "structural_plausibility": {
                "computed": True, "score": 0.82,
                "info": {"n_clusters": 2, "interface_ratio": 0.21,
                         "rg": 8.4, "expected_rg": 9.3},
            },
            "physicochemical_complementarity": {
                "computed": True, "score": 0.65,
                "info": {"positive_ratio": 0.33, "aromatic_ratio": 0.08,
                         "polar_ratio": 0.25, "gp_ratio": 0.10},
            },
            "evolutionary_conservation": {
                "computed": True, "score": 0.71,
                "info": {"rare_ratio": 0.22, "terminal_share": 0.0,
                         "pI": 8.87},
            },
            "cross_tool_consensus": {
                "computed": True, "score": 0.42,
                "info": {"n_active_tools": 3, "avg_vote_ratio": 0.55,
                         "avg_jaccard": 0.21,
                         "pairwise_jaccard": {
                             "boltz2|p2rank": 0.18,
                             "boltz2|equipnas": 0.32,
                             "p2rank|equipnas": 0.13,
                         }},
            },
            "known_motif_consistency": {
                "computed": True, "score": 0.55,
                "info": {"domain": "RRM", "n_motifs_found": 1,
                         "coverage_ratio": 0.55},
            },
        },
    }

    # Mock prior iteration: tried to refine equipnas, score barely changed.
    history = [
        {
            "iteration": 0,
            "action": {
                "action": "refine",
                "refine_tool": "equipnas",
                "refine_reason": "low motif coverage",
                "rationale": "q5 was the bottleneck on the first pass.",
                "confidence": 0.7,
            },
            "score_before": 0.61,
            "score_after": 0.62,
            "delta": 0.01,
            "api_usage": {"total_tokens": 850},
            "timestamp": "2026-05-04T00:00:00Z",
        },
    ]

    messages = build_messages(
        sample_id=sample_id,
        target_char=target_char,
        qa_result=qa_result,
        composite_result=composite,
        history=history,
        iteration_index=1,
        max_iterations=3,
    )
    est = estimate_prompt_tokens(messages)

    print("=" * 72)
    print(f"sample_id: {sample_id}")
    print(f"iteration: 1 / 3")
    print(f"estimated prompt tokens: ~{est}")
    print("=" * 72)
    for m in messages:
        print(f"\n--- role: {m['role']} ---")
        print(m["content"])


if __name__ == "__main__":
    _demo_main()
