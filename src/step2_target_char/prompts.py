"""Prompt templates for Step 2 — target characterization with LLM.

Design
------
LLM endpoint accepts `response_format={"type":"json_object"}` (verified in
stage 2.0 probe), so we rely on JSON mode for structural output instead of
soft "please return JSON" hinting in the prompt body. The prompt still spells
out the exact JSON schema because JSON mode alone doesn't constrain keys.

Three builders:
  - `build_messages(features, history_summary=None)` — initial call
  - `build_error_correction_messages(prev_messages, bad_output, error)` —
    appends 1 assistant+user turn asking the model to fix a validation error
  - `format_features_for_prompt(features)` — internal helper, exposed for
    tests / debugging / golden files

Prompt budget: system ≈ 1400 tokens, user ≈ 250-500 tokens. Total < 2000 tokens
per call before any history summary.

Null handling
-------------
`TargetFeatures` preserves None for fields upstream couldn't compute (pI for
all-X proteins, gc_content for poly-inosine, ss_status before stage 1.3b,
etc.). The user prompt renders these as `(unknown — not computed)` so the
LLM sees the *reason* for missingness rather than a silent zero.
"""
from __future__ import annotations

from typing import Optional

from .schemas import TargetFeatures


# ---------- system prompt ---------------------------------------------------

SYSTEM_PROMPT = """\
You are a structural biology expert specialized in RNA-protein interactions. \
Given a compact feature summary of an RNA-protein binding complex, classify \
the binding mode along two axes and produce a short English analysis.

# Pocket category axes

## Protein domain (choose exactly one)
- RRM         : ~80-90 aa; β1-α1-β2-β3-α2-β4 fold; binds ssRNA via aromatic \
stacking (Phe/Tyr) and electrostatics (Lys/Arg); common in splicing factors.
- KH          : ~70 aa; GXXG loop motif; binds ~4 nt ssRNA in a cleft; \
Gly/Ile/Leu/Val enriched.
- zinc_finger : short (~30-60 aa) Cys/His-coordinated Zn domains, often \
tandem; CCCH / CCHC variants bind RNA; look for high Cys+His composition.
- dsRBD       : ~65-70 aa; α1-β1-β2-β3-α2; binds dsRNA minor groove; \
Lys/Arg-rich binding face; prefers paired RNA.
- PUF         : ~350-400 aa; 8 Puf repeats; sequence-specific ssRNA \
recognition, 1 nt per repeat.
- DEAD_box    : ~400-500 aa; helicase core with DEAD motif; ATP-dependent \
RNA remodeling; common in spliceosome / ribosome biogenesis.
- multi_domain: clearly composite (>300 aa) with multiple recognizable RBDs \
or mixed signals that don't fit one category.
- novel_fold  : evidence doesn't fit any known category, or information is \
too sparse to commit — explain in `notes`.

## RNA structure (choose exactly one)
- single_stranded : little to no base pairing (paired_frac ≲ 0.2); binding \
region is accessible ssRNA.
- stem_loop       : one or a few hairpins; moderate pairing (0.4-0.7) with \
high hairpin_frac relative to multiloop_frac.
- internal_loop   : stems interrupted by internal loops / bulges; \
interior_frac is non-trivial, often alongside hairpin_frac.
- junction        : multi-way junction (≥3 stems meeting); high multiloop_frac; \
typical of rRNA, riboswitches, large structured RNAs.
- g_quadruplex    : G-rich RNA forming tetrads; canonical MFE fold does NOT \
capture this — infer from sequence composition and length hints only; be \
conservative, prefer only when features point strongly here.
- unstructured    : disordered / no reliable secondary structure; very low \
paired_frac and no hairpin signature.

# Domain knowledge hints (treat as priors, not hard rules)
- pI > 9 AND Lys/Arg in top-3 amino acids → basic RNA-binding protein; \
strongly compatible with RRM / KH / dsRBD / zinc_finger.
- protein_length < 120 → likely a single RBD (RRM ~90, KH ~70, ZnF ~30-60).
- protein_length > 500 → lean toward multi_domain or DEAD_box (helicase core).
- High Cys+His composition AND short protein → consider zinc_finger.
- paired_frac > 0.6 → stem_loop or internal_loop.
- paired_frac < 0.2 AND hairpin_frac < 0.1 → single_stranded or unstructured.
- multiloop_frac > 0.1 → junction character.
- rna_length > 1000 → large RNA (ribosomal / long mRNA); likely junction.
- interface_ratio_protein < 0.05 → small surface pocket (focal contact); \
consistent with specific ssRNA recognition (PUF, single KH).
- mean_bfactor > 150 → cryo-EM low-resolution / flexible region; reduce \
confidence and note this.
- resolution > 4.0 Å → architectural-only; don't over-commit on side-chain \
identity; prefer lower confidence.

# Missing information
Some features may be marked `(unknown — not computed)`. This means the value \
is genuinely unavailable, NOT zero. When key features are missing, lower the \
confidence and mention in `notes` which missing fields affected the call. \
When evidence is genuinely insufficient, prefer `novel_fold` for the protein \
axis and/or `unstructured` for the RNA axis over guessing.

# Output JSON schema (strict)
Return ONE JSON object with exactly these keys, no extras, no trailing prose:

{
  "analysis":        <string, English, 2-6 sentences; structural-biology \
reasoning that walks through the key features and justifies both axis choices>,
  "protein_domain":  <one of: "RRM" | "KH" | "zinc_finger" | "dsRBD" | "PUF" | \
"DEAD_box" | "multi_domain" | "novel_fold">,
  "rna_structure":   <one of: "single_stranded" | "stem_loop" | \
"internal_loop" | "junction" | "g_quadruplex" | "unstructured">,
  "category":        <string, MUST equal "{protein_domain}_x_{rna_structure}">,
  "confidence":      <number in [0.0, 1.0]; 0.0=pure guess, 1.0=textbook case>,
  "notes":           <string or null; cite missing fields, unusual signals, \
or low-resolution caveats here>
}

Rules:
- Output ONLY the JSON object. No markdown, no code fences, no commentary \
before or after.
- `analysis` MUST be in English.
- `category` MUST be the underscore-joined label, e.g. "RRM_x_stem_loop".
- Enum values are case-sensitive and must match exactly.
"""


# ---------- user prompt -----------------------------------------------------


def _fmt_opt(value, fmt: str = "{}", unknown: str = "unknown — not computed") -> str:
    """Render value with fmt if present, else the unknown marker."""
    if value is None:
        return f"({unknown})"
    return fmt.format(value)


def _fmt_top3(top3: Optional[list]) -> str:
    if not top3:
        return "(unknown — not computed)"
    return ", ".join(f"{aa}={frac:.3f}" for aa, frac in top3)


def _fmt_structure_composition(sc) -> str:
    if sc is None:
        return "(unknown — stage 1.3b not yet run for this sample)"
    # Compact single-line: paired=0.52, hairpin=0.12, interior=0.08, ...
    return (
        f"paired={sc.paired_frac:.2f}, "
        f"hairpin={sc.hairpin_frac:.2f}, "
        f"interior={sc.interior_frac:.2f}, "
        f"multiloop={sc.multiloop_frac:.2f}, "
        f"external={sc.external_frac:.2f}"
    )


def format_features_for_prompt(features: TargetFeatures) -> str:
    """Render TargetFeatures as a labeled text block for the user prompt.

    Nulls are shown as `(unknown — …)` with a hint about *why* so the LLM can
    judge whether to downgrade confidence on that axis.
    """
    f = features
    return f"""\
Sample ID: {f.sample_id}

# RNA
- length:                 {f.rna_length} nt
- GC content:             {_fmt_opt(f.rna_gc_content, '{:.3f}')}
- has modification:       {str(f.rna_has_modification).lower()}
- secondary-structure status: {_fmt_opt(f.rna_ss_status)}
- structure composition:  {_fmt_structure_composition(f.rna_structure_composition)}

# Protein
- length:                 {f.protein_length} aa
- isoelectric point (pI): {_fmt_opt(f.protein_pI, '{:.2f}', unknown='unknown — all-X or unparseable sequence')}
- mean B-factor:          {_fmt_opt(f.protein_mean_bfactor, '{:.1f}', unknown='unknown — cryo-EM B=0 placeholder or unparseable')}
- top-3 amino acids:      {_fmt_top3(f.protein_top3_aa)}

# Interaction
- binding protein residues: {f.n_binding_protein_residues}
- binding RNA nucleotides:  {f.n_binding_rna_nucleotides}
- interface ratio (protein): {f.interface_ratio_protein:.4f}
- interface ratio (RNA):     {f.interface_ratio_rna:.4f}

# Data provenance
- quality tier:         {f.quality_tier}
- resolution:           {_fmt_opt(f.resolution, '{:.2f} Å')}
- experimental method:  {_fmt_opt(f.experimental_method)}

Return the JSON object as specified. Remember: analysis in English, \
`category` must be `"{{protein_domain}}_x_{{rna_structure}}"`.\
"""


# ---------- message builders -----------------------------------------------


def build_messages(
    features: TargetFeatures,
    history_summary: Optional[str] = None,
) -> list[dict]:
    """Assemble the messages list for the LLM chat API.

    `history_summary` is reserved for stage 2.4 (PredictionHistory); when
    provided, it's appended to the system prompt as a contextual block so
    the LLM can condition on past category distribution / confidence trends.
    """
    system = SYSTEM_PROMPT
    if history_summary:
        system = (
            SYSTEM_PROMPT
            + "\n\n# Historical context (from prior predictions)\n"
            + history_summary.strip()
            + "\nUse this as a weak prior; each sample is evaluated on its own features."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": format_features_for_prompt(features)},
    ]


def build_error_correction_messages(
    prev_messages: list[dict],
    bad_output: str,
    error_message: str,
) -> list[dict]:
    """Append assistant's bad output + a user correction turn.

    The correction turn quotes the exact validation error and asks for a
    fresh JSON object. We keep the original system+user intact so the model
    still has all the feature context.
    """
    correction = (
        "Your previous response failed schema validation.\n\n"
        f"Validation error:\n{error_message.strip()}\n\n"
        "Return ONE corrected JSON object matching the schema exactly. "
        "Do not include explanations, markdown, or code fences. "
        "Remember: `category` must equal `{protein_domain}_x_{rna_structure}`, "
        "`confidence` is a number in [0,1], and `analysis` is 2-6 sentences in English."
    )
    return [
        *prev_messages,
        {"role": "assistant", "content": bad_output},
        {"role": "user", "content": correction},
    ]


# ---------- token budget guard (rough estimate) ----------------------------


def estimate_prompt_tokens(messages: list[dict]) -> int:
    """Very rough char-based token estimate (~4 chars/token for English/ASCII).

    Not used for any hard limit — just for catching accidental prompt bloat
    in tests and demos. LLM tokenizer isn't exposed client-side.
    """
    total_chars = sum(len(m["content"]) for m in messages if "content" in m)
    return total_chars // 4


# ---------- demo / CLI ------------------------------------------------------


def _demo_main() -> None:
    import argparse
    import csv
    import json
    import random
    import sys
    from pathlib import Path

    from .feature_adapter import extract_target_features, sid_to_filename

    # Force UTF-8 stdout so Å / ≲ / other unicode in prompts print on
    # Windows' GBK console. Safe no-op on Linux (already utf-8).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(
        description="Print the full messages list for 1 random sample.",
    )
    ap.add_argument("--processed-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-id", type=str, default=None)
    args = ap.parse_args()

    samples_dir = args.processed_dir / "samples"
    index_csv = args.processed_dir / "index.csv"
    with index_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.sample_id:
        rows = [r for r in rows if r["sample_id"] == args.sample_id]
        if not rows:
            raise SystemExit(f"sample_id {args.sample_id!r} not in index.csv")
    else:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.n]

    for r in rows:
        sid = r["sample_id"]
        path = samples_dir / (sid_to_filename(sid) + ".json")
        with path.open("r", encoding="utf-8") as f:
            sample = json.load(f)
        feats = extract_target_features(sample)
        messages = build_messages(feats)
        print("=" * 72)
        print(f"sample_id: {sid}")
        print(f"estimated prompt tokens: ~{estimate_prompt_tokens(messages)}")
        print("=" * 72)
        for m in messages:
            print(f"\n--- role: {m['role']} ---")
            print(m["content"])


if __name__ == "__main__":
    _demo_main()
