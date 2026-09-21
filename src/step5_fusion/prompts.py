"""Prompt templates for Step 5 — fusion-weight assignment with LLM.

Design
------
LLM JSON mode is supported (probed in step 2). We use
``response_format={"type":"json_object"}`` and still spell out the schema
in the system prompt because JSON mode alone doesn't constrain keys.

Three builders:
  - ``build_messages(...)`` — initial call (system + user)
  - ``build_error_correction_messages(...)`` — appends 1 assistant+user
    turn asking the model to fix a validation error (matches step 2/3
    pattern)
  - ``format_user_prompt(...)`` — internal helper, exposed for
    tests / debugging / golden files

Two analysis helpers consumed by the prompt:
  - ``compute_consensus(predictions, side)`` — count how many tools
    voted for each residue / nucleotide.
  - ``summarize_predictions(predictions)`` — render a compact
    per-tool + cross-tool block for the user prompt.

Prompt budget: system ≈ 1700 tokens, user ≈ 600-1200 tokens. Total < 3000.

Null / empty handling
---------------------
A ``ToolPrediction`` with ``success=False`` is rendered as a single
"FAILED — <error>" line so the LLM sees the failure but doesn't waste
tokens on empty fields. Tools with no per-residue confidence get an
``avg_conf=(unknown)`` marker rather than a fake zero.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

# step4 ToolPrediction is the canonical shape; import lazily so unit tests
# can also pass duck-typed mocks. Same pattern as noisy_or.py.
try:
    from step4_tool_adapters.schemas import ToolPrediction  # type: ignore
except Exception:  # pragma: no cover
    ToolPrediction = object  # type: ignore


# ---------- system prompt ---------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert in RNA-protein structural biology, specializing in \
critical assessment of computational predictions. Given the predictions \
from multiple binding-site tools on the same target, assign each tool a \
fusion weight c_k in [0, 1] reflecting how much its prediction should \
contribute to the final consensus.

The downstream fusion is noisy-OR over per-residue probabilities:
  b_hat(i) = 1 - prod_k (1 - c_k * b_k(i))
  B_p_hat  = {i : b_hat(i) > tau}

Your job is to set the c_k values and the threshold tau.

# Tool categories (as registered in this pipeline)

- Cat A -- end-to-end complex structure prediction (Boltz-2, Chai-1; \
RoseTTAFold2NA is registered but currently unavailable due to MSA cost). \
Reports per-residue pLDDT (0-100), interface pTM (ipTM, 0-1), and PAE.
- Cat B -- pocket detection on protein surface (P2Rank; Fpocket is \
registered but currently disabled — see tool_registry.py). P2Rank \
(ML, gradient boosting on surface features) reports pockets with \
per-residue ligandability score (0-1). RNA-agnostic — it finds \
generic ligand-binding cavities, not RNA-specific interfaces.
- Cat C -- RNA-binding residue predictor (EquiPNAS). Per-residue \
probability (0-1) on the protein side only. Does NOT predict RNA-side \
nucleotides.
- Cat D -- molecular docking (HADDOCK 3). Outputs a ranked docked \
complex; the adapter extracts heavy-atom interface residues plus a \
distance-derived per-residue probability on the same axis Boltz-2 / \
Chai-1 use. Independent signal from the Cat A predictors — useful for \
diversifying the ensemble.

# Weight-assignment heuristics (treat as priors, not hard rules)

- Two Cat A tools agreeing on the interface (large residue overlap, both \
with high pLDDT > 80 and low pAE < 10) -> both deserve high weight \
(>= 0.8). If they disagree, the higher-confidence one (better pLDDT/ipTM) \
gets the higher weight; downweight the laggard.
- A Cat A tool with pLDDT < 60 or ipTM < 0.4 is unreliable on the \
interface -- drop its weight to 0.2-0.4 even if it predicts many residues.
- Cat C (EquiPNAS) predicts ONLY protein-side residues. Its RNA-side \
contribution is by construction zero; this is reflected by it providing \
no `binding_rna_nucleotides`. Do not penalize the Cat C weight for that.
- Cat B (P2Rank) detects pockets without knowing whether they bind RNA \
specifically. Use it as supporting evidence (typical c_k = 0.3-0.5) when \
its top pocket residues overlap the Cat A/C predictions; downweight \
further (<= 0.2) when no overlap.
- Multi-tool consensus on a residue (predicted by 2 or more tools) is a \
strong positive signal. If most of one tool's predictions are also \
predicted by the others, raise its weight; if a tool's predictions are \
mostly singletons (no other tool agrees), lower its weight.
- If step 2 category is `novel_fold`, Cat C tools were trained on \
known-fold residue patterns and may generalise poorly -- modestly lower \
their weight (multiply by ~0.7-0.8).
- If only one tool succeeded (others failed), set that tool's weight to \
1.0 (no actual fusion happens -- the result is that tool alone).

# Threshold (tau) guidance

- Default 0.5. Use this when tool agreement is mixed.
- Raise to 0.6-0.7 when ALL contributing tools have high confidence and \
high cross-tool overlap (reduces false positives).
- Lower to 0.3-0.4 when tools disagree heavily or per-residue confidences \
are uniformly low (preserves recall in low-signal regimes).

# Output JSON schema (strict)

Return ONE JSON object with exactly these keys:

{
  "weights":    <object mapping tool_id (string) to c_k (number in \
[0.0, 1.0]); MUST include every tool_id listed in the user prompt; \
do not invent tool_ids>,
  "threshold":  <number in [0.0, 1.0]; default 0.5>,
  "rationale":  <string, English, 2-6 sentences explaining the weight \
choices and any threshold deviation from 0.5>,
  "confidence": <number in [0.0, 1.0]; YOUR self-assessment of the fused \
result quality given the tool agreement and category context>
}

Rules:
- Output ONLY the JSON object. No markdown, no code fences, no commentary \
before or after.
- `rationale` must be in English.
- Every tool_id listed in the user prompt MUST appear as a key in `weights`. \
Set weight to 0.0 (not omit) if you want to exclude a tool from the fusion.
- Failed tools (marked FAILED in the user prompt) should still get a \
weight entry; conventionally 0.0 -- they will be filtered upstream anyway.
"""


# ---------- consensus + summary helpers ------------------------------------


def compute_consensus(
    predictions: Iterable, side: str = "protein",
) -> dict[int, int]:
    """Count how many tools voted for each residue / nucleotide.

    Parameters
    ----------
    predictions
        Iterable of ``ToolPrediction``-like objects. Failed predictions
        (``success=False``) are skipped.
    side
        "protein" → use ``binding_protein_residues``;
        "rna"     -> use ``binding_rna_nucleotides``.

    Returns
    -------
    ``{residue_or_nucleotide_id: tool_count}``. Empty dict if no tool
    voted for anything on that side.
    """
    if side not in ("protein", "rna"):
        raise ValueError(f"side must be 'protein' or 'rna', got {side!r}")

    counts: Counter[int] = Counter()
    for p in predictions:
        if not getattr(p, "success", False):
            continue
        if side == "protein":
            ids = getattr(p, "binding_protein_residues", None) or []
        else:
            ids = getattr(p, "binding_rna_nucleotides", None) or []
        for r in ids:
            counts[int(r)] += 1
    return dict(counts)


def _avg_per_residue_confidence(pred) -> Optional[float]:
    """Mean of ``per_residue_confidence`` values, or None if missing."""
    prc = getattr(pred, "per_residue_confidence", None)
    if not prc:
        return None
    vals = list(prc.values())
    if not vals:
        return None
    return sum(vals) / len(vals)


def _format_pocket_summary(pred) -> Optional[str]:
    """One-line pocket summary for Cat B tools (rank, score, residue count)."""
    pockets = getattr(pred, "pockets", None)
    if not pockets:
        return None
    parts = []
    for pk in pockets[:3]:  # at most top 3 to control tokens
        parts.append(
            f"#{pk.rank} score={pk.score:.2f} residues={len(pk.residues)}"
        )
    suffix = f" (+{len(pockets) - 3} more)" if len(pockets) > 3 else ""
    return ", ".join(parts) + suffix


def _format_one_tool(pred) -> str:
    """Render a per-tool block — one-line header + indented details."""
    tool_id = getattr(pred, "tool_id", "?")
    cat = getattr(pred, "category", "?")
    success = getattr(pred, "success", False)

    header = f"- {tool_id} (Cat {cat})"
    if not success:
        err = getattr(pred, "error_message", None) or "no detail"
        return f"{header}: FAILED — {err[:200]}"

    bps = getattr(pred, "binding_protein_residues", None) or []
    brn = getattr(pred, "binding_rna_nucleotides", None) or []
    avg_c = _avg_per_residue_confidence(pred)

    detail_lines = [
        f"  protein residues predicted: {len(bps)}",
    ]

    if avg_c is None:
        detail_lines.append("  per-residue confidence: (none)")
    else:
        # Cat A is pLDDT (0-100), others are [0, 1] — make units explicit.
        unit = "pLDDT" if cat == "A" else "prob"
        detail_lines.append(
            f"  per-residue confidence: avg {unit} = {avg_c:.3f}"
        )

    detail_lines.append(f"  RNA nucleotides predicted: {len(brn)}")

    plddt = getattr(pred, "plddt_mean", None)
    iptm = getattr(pred, "iptm_score", None)
    pae = getattr(pred, "pae_mean", None)
    if plddt is not None or iptm is not None or pae is not None:
        struct_parts = []
        if plddt is not None:
            struct_parts.append(f"pLDDT={plddt:.1f}")
        if iptm is not None:
            struct_parts.append(f"ipTM={iptm:.3f}")
        if pae is not None:
            struct_parts.append(f"pAE={pae:.2f}")
        detail_lines.append(f"  structure metrics: {', '.join(struct_parts)}")

    pocket_summary = _format_pocket_summary(pred)
    if pocket_summary:
        detail_lines.append(f"  pockets: {pocket_summary}")

    runtime = getattr(pred, "runtime_seconds", None)
    if runtime is not None:
        detail_lines.append(f"  runtime: {runtime:.1f}s")

    return "\n".join([header, *detail_lines])


def _format_consensus_block(predictions: Iterable) -> str:
    """Render cross-tool overlap summary (protein side + RNA side)."""
    preds = list(predictions)

    p_consensus = compute_consensus(preds, side="protein")
    r_consensus = compute_consensus(preds, side="rna")

    lines = []

    # Protein side
    if p_consensus:
        max_count = max(p_consensus.values())
        # Bucket by vote count
        by_count: dict[int, list[int]] = {}
        for r, c in p_consensus.items():
            by_count.setdefault(c, []).append(r)
        lines.append(
            f"Protein-side consensus (max overlap = {max_count} tools):"
        )
        for k in sorted(by_count.keys(), reverse=True):
            ids = sorted(by_count[k])
            shown = ids[:25]
            extra = f" ... (+{len(ids) - 25} more)" if len(ids) > 25 else ""
            lines.append(
                f"  predicted by {k} tool(s): {len(ids)} residues -- "
                f"{shown}{extra}"
            )
    else:
        lines.append("Protein-side consensus: (no successful protein-side predictions)")

    # RNA side
    if r_consensus:
        max_count = max(r_consensus.values())
        by_count: dict[int, list[int]] = {}
        for r, c in r_consensus.items():
            by_count.setdefault(c, []).append(r)
        lines.append(f"RNA-side consensus (max overlap = {max_count} tools):")
        for k in sorted(by_count.keys(), reverse=True):
            ids = sorted(by_count[k])
            shown = ids[:25]
            extra = f" ... (+{len(ids) - 25} more)" if len(ids) > 25 else ""
            lines.append(
                f"  predicted by {k} tool(s): {len(ids)} nucleotides -- "
                f"{shown}{extra}"
            )
    else:
        lines.append("RNA-side consensus: (no successful RNA-side predictions)")

    return "\n".join(lines)


def summarize_predictions(predictions: Iterable) -> str:
    """Compact human-readable summary of a list of ToolPrediction objects.

    Layout:
      ``# Per-tool predictions``
        one block per tool (id, category, residue counts, conf, structure
        metrics, pockets, runtime — or 'FAILED' line)

      ``# Cross-tool consensus``
        protein-side + RNA-side overlap summary (residues bucketed by
        how many tools voted for them).
    """
    preds = list(predictions)
    blocks: list[str] = ["# Per-tool predictions"]
    if not preds:
        blocks.append("(no predictions provided)")
    else:
        for p in preds:
            blocks.append(_format_one_tool(p))

    blocks.append("")
    blocks.append("# Cross-tool consensus")
    blocks.append(_format_consensus_block(preds))

    return "\n".join(blocks)


# ---------- target-char rendering ------------------------------------------


def _fmt_target_char(tc: dict) -> str:
    """Render step 2 output for the user prompt (single block, compact)."""
    if not tc:
        return "(step 2 characterization not available)"
    lines = [
        f"Category:   {tc.get('category', 'unknown')}",
        f"Confidence: {tc.get('confidence', '?')}",
    ]
    analysis = tc.get("analysis")
    if analysis:
        # Trim very long analyses to keep prompt tight (rare, but possible).
        if len(analysis) > 1500:
            analysis = analysis[:1500].rstrip() + " ..."
        lines.append(f"Analysis:   {analysis}")
    notes = tc.get("notes")
    if notes:
        lines.append(f"Notes:      {notes}")
    return "\n".join(lines)


# ---------- user prompt + message builders --------------------------------


def format_user_prompt(
    sample_id: str,
    target_char: dict,
    predictions: Iterable,
    weight_summary: Optional[str] = None,
) -> str:
    """Build the user-prompt text from step-5 inputs."""
    pred_summary = summarize_predictions(predictions)

    sections = [
        f"Sample ID: {sample_id}",
        "",
        "# Step 2 Target Characterization",
        _fmt_target_char(target_char),
        "",
        pred_summary,
    ]

    if weight_summary and weight_summary.strip():
        sections.extend([
            "",
            "# Tool reliability (weight tensor for this category)",
            weight_summary.strip(),
        ])

    # Collect tool_ids so the LLM has an explicit list to populate.
    preds = list(predictions)
    tool_ids = [getattr(p, "tool_id", "?") for p in preds]
    sections.extend([
        "",
        "# Required tool_ids in `weights`",
        ", ".join(tool_ids) if tool_ids else "(no tools)",
        "",
        "Return the JSON object as specified in the system prompt.",
    ])

    return "\n".join(sections)


def build_messages(
    sample_id: str,
    target_char: dict,
    predictions: Iterable,
    weight_summary: Optional[str] = None,
    history_summary: Optional[str] = None,
) -> list[dict]:
    """Assemble system + user messages for the LLM chat API."""
    system = SYSTEM_PROMPT
    if history_summary and history_summary.strip():
        system = (
            system
            + "\n\n# Historical context (from prior fusions)\n"
            + history_summary.strip()
            + "\nUse this as a weak prior; weight each sample on its own merits."
        )

    user = format_user_prompt(
        sample_id=sample_id,
        target_char=target_char,
        predictions=predictions,
        weight_summary=weight_summary,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_error_correction_messages(
    prev_messages: list[dict],
    bad_output: str,
    error_message: str,
) -> list[dict]:
    """Append assistant's bad output + user correction turn (step 2/3 pattern)."""
    correction = (
        "Your previous response failed schema validation.\n\n"
        f"Validation error:\n{error_message.strip()}\n\n"
        "Return ONE corrected JSON object with these keys: "
        "`weights` (object mapping tool_id to a number in [0, 1]; "
        "every tool_id from the user prompt MUST be a key), "
        "`threshold` (number in [0, 1]), "
        "`rationale` (English, 2-6 sentences), "
        "`confidence` (number in [0, 1]). "
        "No markdown, no code fences."
    )
    return [
        *prev_messages,
        {"role": "assistant", "content": bad_output},
        {"role": "user", "content": correction},
    ]


# ---------- token estimate -------------------------------------------------


def estimate_prompt_tokens(messages: list[dict]) -> int:
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return total_chars // 4


# ---------- demo / CLI -----------------------------------------------------


def _demo_main() -> None:
    """Render messages for a synthetic 3-tool scenario without calling the API."""
    import sys
    from pathlib import Path

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    REPO = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO / "src"))

    from step4_tool_adapters.schemas import Pocket, ToolPrediction
    from step3_tool_selection.weight_tensor import WeightTensor

    sample_id = "1un6_B_F"

    # Mock step 2 target characterization (matches the 1un6_B_F demo used in
    # step 3 prompts._demo_main).
    target_char = {
        "category": "RRM_x_stem_loop",
        "confidence": 0.9,
        "analysis": (
            "RRM domain (~87 aa) with basic pI 8.87 and high Lys/His content "
            "binding a stem-loop RNA (61 nt, paired_frac 0.75)."
        ),
        "notes": None,
    }

    # 3 mock tool predictions: boltz2 (Cat A), p2rank (Cat B), equipnas (Cat C)
    boltz2 = ToolPrediction(
        tool_id="boltz2", category="A", sample_id=sample_id,
        success=True,
        binding_protein_residues=[10, 11, 12, 23, 24],
        binding_rna_nucleotides=[3, 4, 5],
        per_residue_confidence={10: 88.0, 11: 92.0, 12: 87.0, 23: 75.0, 24: 70.0},
        predicted_structure_path="/tmp/boltz2/model_0.cif",
        plddt_mean=82.5, iptm_score=0.71, pae_mean=8.4,
        runtime_seconds=312.0,
    )
    p2rank = ToolPrediction(
        tool_id="p2rank", category="B", sample_id=sample_id,
        success=True,
        binding_protein_residues=[11, 12, 23, 50, 51, 52],
        per_residue_confidence={11: 0.65, 12: 0.62, 23: 0.55, 50: 0.40, 51: 0.41, 52: 0.39},
        pockets=[
            Pocket(rank=1, score=12.4, residues=[11, 12, 23, 50, 51, 52]),
            Pocket(rank=2, score=6.8, residues=[80, 81, 82]),
        ],
        runtime_seconds=14.0,
    )
    equipnas = ToolPrediction(
        tool_id="equipnas", category="C", sample_id=sample_id,
        success=True,
        binding_protein_residues=[10, 11, 23, 24, 25],
        per_residue_confidence={10: 0.85, 11: 0.91, 23: 0.78, 24: 0.66, 25: 0.55},
        runtime_seconds=58.0,
    )
    predictions = [boltz2, p2rank, equipnas]

    # Pull a weight summary from a fresh tensor (cold-start defaults).
    wt = WeightTensor()
    weight_summary = wt.get_category_summary(target_char["category"])

    messages = build_messages(
        sample_id=sample_id,
        target_char=target_char,
        predictions=predictions,
        weight_summary=weight_summary,
    )
    est = estimate_prompt_tokens(messages)

    print("=" * 72)
    print(f"sample_id: {sample_id}")
    print(f"estimated prompt tokens: ~{est}")
    print(f"tools in prompt: {[p.tool_id for p in predictions]}")
    print("=" * 72)
    for m in messages:
        print(f"\n--- role: {m['role']} ---")
        print(m["content"])


if __name__ == "__main__":
    _demo_main()
