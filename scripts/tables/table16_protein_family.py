"""Table 17 — protein domain-family stratified performance analysis.

Same machinery as Table 16 (``table15_rna_length``) — the per-sample,
resolved-residue per-method Pearson R is reused verbatim — but the
grouping dimension is the **protein domain family** instead of RNA length.

Group source (per sample):
  1. the SCOPE LLM profile's ``protein_family`` field
     (``--scope-profiles/<sample_id>.json``);
  2. failing that, the heuristic ``auto_scope_profile`` family read off the
     sample JSON.

Families (paper Table 3); raw values are normalised onto:

    RRM     RNA Recognition Motif
    KH      K Homology
    ZnF     Zinc Finger
    dsRBD   double-stranded RNA Binding Domain
    Other   multi-domain (Multi), novel folds, and anything unrecognised

Per the shared note, a sparse group (n < 5) is still shown and its Pearson
R still computed (flagged by the printed group size — small-n means are
noisy), rather than auto-merged away.

Methods (same as Table 16): Boltz-2, EquiPNAS, Chai-1, P2Rank, RF2NA,
RiboSeer. No retraining — step4 scores + the pre-trained RiboSeer bundle.

Usage
-----
::

    python scripts/tables/table16_protein_family.py \\
        --step4-dir          data/batch_test_v7/step4/ \\
        --processed-dir      data/processed_quality \\
        --sample-list        data/processed_quality/splits_tmscore_035/test.txt \\
        --enriched-model-dir data/enriched_v7_lgbm \\
        --scope-profiles     data/batch_test_v7/scope_profiles_llm/ \\
        --output             data/batch_test_v7/table17_protein_family.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scripts.riboseer.ablation_fusion_method import (  # noqa: E402
    collect_sample_data,
)
from scripts.tables.table15_rna_length import (  # noqa: E402
    METHOD_ORDER, group_counts, load_profiles, method_corrs, print_table,
    profile_labels, resolve_riboseer_source, stratify_by, write_csv,
    _load_sample_ids,
)

# Column order (Other last — the catch-all).
FAMILY_GROUPS = ["RRM", "KH", "ZnF", "dsRBD", "Other"]

# Exact canonical values (SCOPE_FAMILIES, lower-cased) → bucket. Multi /
# Novel fold into Other.
_CANON_FAMILY = {
    "rrm": "RRM", "kh": "KH", "znf": "ZnF", "dsrbd": "dsRBD",
    "multi": "Other", "novel": "Other",
}


def normalize_family(raw: Optional[str]) -> str:
    """Map a raw ``protein_family`` string onto a ``FAMILY_GROUPS`` bucket.
    Exact canonical match first, then a light substring fallback for LLM
    re-wordings; everything else → ``Other``."""
    s = (raw or "").strip().lower()
    if not s:
        return "Other"
    if s in _CANON_FAMILY:
        return _CANON_FAMILY[s]
    if "recognition motif" in s or "rrm" in s:
        return "RRM"
    if "k homology" in s or s.startswith("kh"):
        return "KH"
    if "zinc" in s or "znf" in s or "zn finger" in s:
        return "ZnF"
    if "double-stranded rna binding" in s or "dsrb" in s:
        return "dsRBD"
    return "Other"


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step4-dir", type=Path, required=True,
                   help="test-split step4 JSONL dir")
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--sample-list", type=Path, required=True)
    p.add_argument("--enriched-model-dir", type=Path, default=None,
                   help="pre-trained RiboSeer EnrichedFusion bundle; not "
                        "needed when --predictions-dir is given")
    p.add_argument("--predictions-dir", type=Path, default=None,
                   help="pre-computed full-system (on/on/on) predictions "
                        "<sid>.json; when set RiboSeer reads these instead "
                        "of loading the model")
    p.add_argument("--scope-profiles", type=Path, default=None,
                   help="dir of SCOPE profile JSONs (protein_family source); "
                        "falls back to the auto profile per sample")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for d in (args.step4_dir, args.processed_dir):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 1
    try:
        sample_ids = _load_sample_ids(args.sample_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    try:
        model, preds = resolve_riboseer_source(
            args.predictions_dir, args.enriched_model_dir)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    samples = collect_sample_data(
        args.step4_dir, args.processed_dir, sample_ids,
        feature_set=(model.feature_set if model else "full"),
        use_context=(model.use_context if model else True))
    if not samples:
        print("ERROR: no usable test samples", file=sys.stderr)
        return 1
    print(f"usable test samples: {len(samples)}")

    profiles = load_profiles(args.scope_profiles)
    labels = profile_labels(args.processed_dir, [s.sid for s in samples],
                            profiles, "protein_family", normalize_family)
    corrs = method_corrs(samples, model, preds)
    table = stratify_by(corrs, labels, FAMILY_GROUPS)
    counts = group_counts(labels, {s.sid for s in samples}, FAMILY_GROUPS)

    write_csv(args.output, table, FAMILY_GROUPS)
    print(f"wrote {args.output}  "
          f"({len(METHOD_ORDER)} methods × {len(FAMILY_GROUPS)} families)")
    print()
    print_table(table, counts, FAMILY_GROUPS, dropped_label="unmapped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
