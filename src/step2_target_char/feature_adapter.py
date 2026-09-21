"""Extract `TargetFeatures` from a step-1 sample JSON.

Responsibilities:
  - pick out the handful of fields the LLM actually looks at
  - compute derived features (interface ratios, top-3 amino acids)
  - tolerate missing/null upstream fields without crashing — the prompt
    downstream shows "unknown" so the LLM can reason about missingness

Null tolerance matters because several stable edge cases from step 1 leave
fields null on purpose:
  - protein.features.pI / aa_composition: null for the 122 all-X proteins
  - protein.features.mean_bfactor: null for 434 cryo-EM samples with B=0
  - rna.features.gc_content: null for 29 poly-inosine chains
  - rna.features.{ss_status, secondary_structure_pred, structure_composition}:
    null on local copies (stage 1.3b is server-side)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Optional

from .schemas import StructureComposition, TargetFeatures


# ---------- filename encoding (same as step1_server/step1_local) ------------
# Duplicated here so step2 is self-contained (no transitive `import gemmi`
# via step1_local._common). Must stay in sync with the originals — unit test
# below guards the round-trip.


def _encode_case_marked(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalpha() and ch.islower():
            out.append("-")
        out.append(ch)
    return "".join(out)


def sid_to_filename(sample_id: str) -> str:
    pdb, _, tail = sample_id.partition("_")
    if not tail:
        return _encode_case_marked(sample_id)
    return f"{pdb}_{_encode_case_marked(tail)}"


# ---------- extractor -------------------------------------------------------


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Return numerator/denominator, clamped to >= 0. Denominator 0 → 0.0."""
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _top_n_aa(aa_composition: Optional[dict], n: int = 3) -> Optional[list[tuple[str, float]]]:
    """Return top-N (aa, frac) pairs sorted by frac desc, ties broken lexicographically."""
    if not aa_composition:
        return None
    # Filter out zero-fraction entries so edge cases don't pollute the top-3.
    items = [(aa, float(frac)) for aa, frac in aa_composition.items() if frac]
    if not items:
        return None
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return [(aa, round(frac, 4)) for aa, frac in items[:n]]


def _build_structure_composition(raw: Any) -> Optional[StructureComposition]:
    """Wrap the dict {paired_frac, hairpin_frac, ...} if present."""
    if raw is None:
        return None
    if isinstance(raw, StructureComposition):
        return raw
    if not isinstance(raw, dict):
        return None
    # Only construct when all 5 keys exist — partial dicts indicate upstream bug.
    required = {"paired_frac", "hairpin_frac", "interior_frac",
                "multiloop_frac", "external_frac"}
    if not required.issubset(raw.keys()):
        return None
    return StructureComposition(**{k: raw[k] for k in required})


def extract_target_features(sample_json: dict) -> TargetFeatures:
    """Produce a `TargetFeatures` from a raw sample JSON.

    Tolerant of missing subtrees: a pathological sample with only `sample_id`
    still returns a well-formed object (with zeros / Nones in the rest).
    """
    sid = sample_json["sample_id"]

    protein = sample_json.get("protein") or {}
    rna = sample_json.get("rna") or {}
    interaction = sample_json.get("interaction") or {}
    data_avail = sample_json.get("data_availability") or {}

    prot_feat = protein.get("features") or {}
    rna_feat = rna.get("features") or {}

    # Lengths (defensive — fall back to sequence length if explicit length missing)
    prot_len = protein.get("length")
    if prot_len is None:
        prot_len = len(protein.get("sequence") or "")
    rna_len = rna.get("length")
    if rna_len is None:
        rna_len = len(rna.get("sequence") or "")

    n_bind_prot = len(interaction.get("binding_protein_residues") or [])
    n_bind_rna = len(interaction.get("binding_rna_nucleotides") or [])

    return TargetFeatures(
        sample_id=sid,
        rna_length=int(rna_len or 0),
        rna_gc_content=rna_feat.get("gc_content"),
        rna_ss_status=rna_feat.get("ss_status"),
        rna_structure_composition=_build_structure_composition(
            rna_feat.get("structure_composition")
        ),
        rna_has_modification=bool(rna.get("has_modification", False)),
        protein_length=int(prot_len or 0),
        protein_pI=prot_feat.get("pI"),
        protein_mean_bfactor=prot_feat.get("mean_bfactor"),
        protein_top3_aa=_top_n_aa(prot_feat.get("aa_composition"), n=3),
        n_binding_protein_residues=n_bind_prot,
        n_binding_rna_nucleotides=n_bind_rna,
        interface_ratio_protein=round(_safe_ratio(n_bind_prot, int(prot_len or 0)), 4),
        interface_ratio_rna=round(_safe_ratio(n_bind_rna, int(rna_len or 0)), 4),
        quality_tier=data_avail.get("quality_tier") or "unknown",
        resolution=data_avail.get("resolution"),
        experimental_method=data_avail.get("experimental_method"),
    )


# ---------- demo / CLI ------------------------------------------------------


def _demo_main() -> None:
    ap = argparse.ArgumentParser(
        description="Demo: extract TargetFeatures from N random samples",
    )
    ap.add_argument("--processed-dir", type=Path, required=True,
                    help="root containing samples/*.json and index.csv")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-id", type=str, default=None,
                    help="if given, extract this specific sample instead of random")
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
        print("-" * 60)
        print(f"sample_id: {sid}")
        print(feats.model_dump_json(indent=2))


if __name__ == "__main__":
    _demo_main()
