"""Prompt templates for Step 3 — tool selection with LLM.

Design
------
Like step 2, we use JSON mode (`response_format={"type":"json_object"}`).
The system prompt contains:
  - role definition (tool-dispatch expert)
  - descriptions for the *available* (deployed) tools only — the full
    registry holds more, but we don't ask the LLM to choose tools we can't
    actually execute (see tool_registry.format_tool_descriptions)
  - cascade strategy explanation
  - domain-knowledge tool-selection heuristics
  - complete JSON output schema with enum values
  - missing-info / fallback guidance

The user prompt contains:
  - step 2 characterization result (category + analysis + confidence)
  - key target features (protein length, RNA length, pI, structure composition)
  - weight tensor summary (UCB scores for this category)
  - explicit instruction to return JSON

Prompt budget: system ≈ 2200 tokens, user ≈ 500-800 tokens. Total < 3000.
"""
from __future__ import annotations

from typing import Optional

from .schemas import ToolSelectionInput
from .tool_registry import format_tool_descriptions


# ---------- system prompt ---------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert in RNA-protein structural biology tool orchestration. \
Given a target characterization (from a prior analysis step) and tool \
reliability data, select the optimal subset of computational tools and \
plan their execution.

# Available tools

{tool_descriptions}
# Default cascade strategy

Run tools in order: **C → B → D → A** (fastest first).
- Category C (seconds): always worth running as a fast baseline.
- Category B (minutes): pocket detection on protein surface; requires \
3D structure.
- Category D (minutes–hours): molecular docking; requires both RNA and \
protein structures.
- Category A (hours): end-to-end structure prediction; slowest but most \
comprehensive.

If intermediate results from earlier categories are already high-quality \
(PocketQA score above the `early_stop_threshold`), later categories can \
be skipped to save compute.

# Tool selection heuristics (treat as priors, not hard rules)

- Only the tools listed above are deployed; do NOT propose any tool_id \
that is not in the list.
- Category C tools are almost always worth including — they're fast, \
cheap, and provide a baseline. Only exclude if there's a strong reason.
- If the target category involves a well-characterized domain (RRM, KH, \
dsRBD), EquiPNAS (the available Category C tool) is especially reliable \
— prioritize it. EquiPNAS uses E(3) equivariant GNN with ESM2 protein \
language model embeddings and does NOT require MSA/PSSM (advantage for \
novel proteins).
- If the protein has no available 3D structure (only sequence), the \
currently deployed structure-requiring tools (EquiPNAS, P2Rank) cannot \
be used. The deployed Category A tools (Boltz-2, Chai-1) accept \
sequence input and remain viable.
- If RNA length > 200 nt, Category A tools may time out or run out of \
memory (samples with rna_length > 200 are now filtered out of the \
training set; if one slips through, prefer the faster C/B tools). \
For `protein_length` > 1000 aa, drop Category A entirely.
- For `novel_fold` protein domains, residue predictors (Category C) \
trained on known folds may be less reliable. Lean more on Category A \
(de novo structure) tools.
- For `junction` RNA structures (large, multi-stem), prefer tools that \
handle complex RNA topology — Category A tools are generally more robust.
- P2Rank is the deployed Category B tool (ML, gradient boosting on \
surface features). Fpocket is registered but currently disabled \
(Pearson R ≈ 0.04 on the test set, ~0 importance in learned fusion).
- Category D (molecular docking) is back online with HADDOCK 3 — \
data-driven protein-RNA docking that outputs a ranked complex plus \
interface residues from rigidbody → flexref → emref. Useful when a \
3D structure is available and the agent wants an independent signal \
that doesn't share noise with the Cat A predictors. Slower than \
Cat C/B but typically finishes within minutes.
- If step 2 confidence is low (< 0.5), the category assignment may be \
unreliable. In that case, diversify tool selection across categories \
rather than specializing.
- `interface_ratio_protein < 0.05` suggests a small, focal binding site \
— EquiPNAS plus a Category A model can give complementary signals here. \
EquiPNAS-predicted residues can also be used as restraints for downstream \
docking (when Category D tools become available).
- `mean_bfactor > 150` or `resolution > 4.0 Å` indicates structural \
uncertainty. Note this but don't automatically exclude structure-based \
tools — they may still add value.

# UCB utility scores

The user prompt includes UCB scores for each tool on this category. These \
scores combine historical reliability weights with an exploration bonus. \
Use them as a ranking prior, but you may override the ranking based on \
biological reasoning. Explain any significant deviations in your rationale.

# Output JSON schema (strict)

Return ONE JSON object with exactly these keys:

{{
  "selected_tools":       <list of tool_id strings, ordered by execution \
sequence; at least 1, at most {max_tools}>,
  "execution_strategy":   <one of: "cascade" | "parallel" | "staged">,
  "param_overrides":      <list of {{"tool_id": str, "param_name": str, \
"param_value": any, "reason": str}}; empty list [] if no overrides>,
  "early_stop_threshold": <number in [0.0, 1.0] or null; PocketQA score \
above which later cascade stages can be skipped>,
  "rationale":            <string, English, 2-6 sentences; explain why \
these tools were chosen and in this order>,
  "confidence":           <number in [0.0, 1.0]>
}}

Rules:
- Output ONLY the JSON object. No markdown, no code fences.
- Every `tool_id` must be one of the registered IDs listed above.
- `param_overrides[].tool_id` must be in `selected_tools`.
- Order `selected_tools` by intended execution sequence.
- `rationale` must be in English.
"""


# ---------- user prompt builder ---------------------------------------------


def _fmt_target_char(tc: dict) -> str:
    """Render step 2 output compactly for the user prompt."""
    lines = [
        f"Category:   {tc.get('category', 'unknown')}",
        f"Confidence: {tc.get('confidence', '?')}",
    ]
    analysis = tc.get("analysis")
    if analysis:
        lines.append(f"Analysis:   {analysis}")
    notes = tc.get("notes")
    if notes:
        lines.append(f"Notes:      {notes}")
    return "\n".join(lines)


def _fmt_key_features(tf: dict) -> str:
    """Extract the most decision-relevant features from TargetFeatures dict."""
    lines = []

    def _add(label: str, value, fmt: str = "{}"):
        if value is not None:
            lines.append(f"- {label}: {fmt.format(value)}")
        else:
            lines.append(f"- {label}: (unknown)")

    _add("protein length", tf.get("protein_length"), "{} aa")
    _add("RNA length", tf.get("rna_length"), "{} nt")
    _add("pI", tf.get("protein_pI"), "{:.2f}")
    _add("RNA GC content", tf.get("rna_gc_content"), "{:.3f}")
    _add("interface ratio (protein)", tf.get("interface_ratio_protein"), "{:.4f}")
    _add("interface ratio (RNA)", tf.get("interface_ratio_rna"), "{:.4f}")
    _add("quality tier", tf.get("quality_tier"))
    _add("resolution", tf.get("resolution"), "{:.2f} Å")
    _add("experimental method", tf.get("experimental_method"))
    _add("RNA has modification", tf.get("rna_has_modification"))

    sc = tf.get("rna_structure_composition")
    if sc and isinstance(sc, dict):
        comp = ", ".join(f"{k}={v:.2f}" for k, v in sc.items())
        lines.append(f"- RNA structure composition: {comp}")
    else:
        lines.append("- RNA structure composition: (unknown)")

    top3 = tf.get("protein_top3_aa")
    if top3:
        aa_str = ", ".join(f"{aa}={frac:.3f}" for aa, frac in top3)
        lines.append(f"- protein top-3 amino acids: {aa_str}")
    else:
        lines.append("- protein top-3 amino acids: (unknown)")

    return "\n".join(lines)


def _fmt_utility_scores(scores: dict[str, float]) -> str:
    """Render UCB scores as a ranked list."""
    lines = ["UCB utility scores (descending):"]
    for i, (tid, score) in enumerate(scores.items(), 1):
        lines.append(f"  {i:2d}. {tid:<20s} {score:.3f}")
    return "\n".join(lines)


def format_user_prompt(inp: ToolSelectionInput) -> str:
    """Build the user-prompt text from a ToolSelectionInput."""
    return f"""\
Sample ID: {inp.sample_id}

# Step 2 Target Characterization
{_fmt_target_char(inp.target_char)}

# Key Target Features
{_fmt_key_features(inp.target_features)}

# Tool Reliability (weight tensor for this category)
{inp.weight_summary}

# {_fmt_utility_scores(inp.utility_scores)}

Return the JSON object as specified.\
"""


# ---------- message builders -----------------------------------------------


def build_messages(
    inp: ToolSelectionInput,
    max_tools: int = 6,
    history_summary: Optional[str] = None,
) -> list[dict]:
    """Assemble system + user messages for the LLM chat API."""
    system = SYSTEM_PROMPT.format(
        tool_descriptions=format_tool_descriptions(),
        max_tools=max_tools,
    )
    if history_summary:
        system += (
            "\n\n# Historical context (from prior predictions)\n"
            + history_summary.strip()
            + "\nUse this as a weak prior; each sample is evaluated on its own merits."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": format_user_prompt(inp)},
    ]


def build_error_correction_messages(
    prev_messages: list[dict],
    bad_output: str,
    error_message: str,
) -> list[dict]:
    """Append assistant's bad output + user correction turn."""
    correction = (
        "Your previous response failed schema validation.\n\n"
        f"Validation error:\n{error_message.strip()}\n\n"
        "Return ONE corrected JSON object. "
        "Every `tool_id` must be from the registered tool list. "
        "`param_overrides[].tool_id` must be in `selected_tools`. "
        "`execution_strategy` must be one of: cascade, parallel, staged. "
        "`confidence` is a number in [0,1]. "
        "No markdown, no code fences."
    )
    return [
        *prev_messages,
        {"role": "assistant", "content": bad_output},
        {"role": "user", "content": correction},
    ]


# ---------- token estimate --------------------------------------------------


def estimate_prompt_tokens(messages: list[dict]) -> int:
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return total_chars // 4


# ---------- demo / CLI ------------------------------------------------------


def _demo_main() -> None:
    import json
    import sys
    from pathlib import Path

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from .schemas import ToolSelectionInput
    from .weight_tensor import WeightTensor

    # Build a mock ToolSelectionInput from the step 2 e2e output for 1un6_B_F
    mock_tc = {
        "analysis": "The protein length of 87 aa aligns with a canonical RRM domain. "
                    "The basic pI (8.87) with high Lysine content supports RNA-binding. "
                    "RNA is short (61 nt) with high paired fraction (0.75), "
                    "indicating a structured stem-loop.",
        "protein_domain": "RRM",
        "rna_structure": "stem_loop",
        "category": "RRM_x_stem_loop",
        "confidence": 0.9,
        "notes": None,
    }
    mock_tf = {
        "sample_id": "1un6_B_F",
        "rna_length": 61, "rna_gc_content": 0.656,
        "rna_ss_status": "done",
        "rna_structure_composition": {
            "paired_frac": 0.75, "hairpin_frac": 0.11,
            "interior_frac": 0.07, "multiloop_frac": 0.07, "external_frac": 0.0,
        },
        "rna_has_modification": False,
        "protein_length": 87, "protein_pI": 8.87,
        "protein_mean_bfactor": 54.6,
        "protein_top3_aa": [("K", 0.126), ("H", 0.103), ("C", 0.081)],
        "n_binding_protein_residues": 17, "n_binding_rna_nucleotides": 14,
        "interface_ratio_protein": 0.1954, "interface_ratio_rna": 0.2295,
        "quality_tier": "strict", "resolution": 3.1,
        "experimental_method": "X-RAY",
    }

    wt = WeightTensor()
    category = mock_tc["category"]
    scores = wt.compute_utility_scores(category)
    summary = wt.get_category_summary(category)

    inp = ToolSelectionInput(
        sample_id="1un6_B_F",
        target_char=mock_tc,
        target_features=mock_tf,
        weight_summary=summary,
        utility_scores=scores,
        available_tools=list(scores.keys()),
    )

    messages = build_messages(inp)
    est = estimate_prompt_tokens(messages)

    print("=" * 72)
    print(f"sample_id: 1un6_B_F  (mock step2 output)")
    print(f"estimated prompt tokens: ~{est}")
    print("=" * 72)
    for m in messages:
        print(f"\n--- role: {m['role']} ---")
        print(m["content"])


if __name__ == "__main__":
    _demo_main()
