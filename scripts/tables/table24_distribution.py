"""Table 25 — per-sample Pearson R distribution.

For each method, computes the per-sample Pearson R (resolved-residue
subset, same convention as Tables 4 / 21 / 22) over the test split and
reports its distribution as percentiles: 10% / 25% / Median / 75% / 90%.

Methods: Boltz-2, EquiPNAS, RiboSeer (the paper's Table-25 rows). Reuses
the same data pipeline as Tables 21/22:
  * single tools → ``_tool_raw_vector`` (step4 per-residue score);
  * RiboSeer → the loaded ``EnrichedFusion`` bundle;
  * GT + resolved mask → ``collect_sample_data``
    (``interaction.binding_protein_residues`` / ``protein.resolved_residues``).

A sample where a tool produced no score (constant vector) is undefined and
drops out of that method's distribution (so per-method ``n`` can differ).

Run
---
::

    python scripts/tables/table24_distribution.py \\
        --data-dir   data/processed_quality \\
        --step4-dir  data/batch_test_v7/step4 \\
        --model-dir  data/enriched_v7_lgbm \\
        --split-file data/processed_quality/splits_tmscore_035/test.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from step5_fusion.enriched_fusion import EnrichedFusion  # noqa: E402
from scripts.riboseer.ablation_fusion_method import (  # noqa: E402
    SampleData, collect_sample_data, per_sample_corr, _tool_raw_vector,
)
from scripts.tables.table15_rna_length import riboseer_predict  # noqa: E402
from step5_fusion.prediction_io import (  # noqa: E402
    load_predictions_dir, predictions_to_vector,
)

TOOL_METHODS: list[tuple[str, str]] = [
    ("Boltz-2", "boltz2"),
    ("EquiPNAS", "equipnas"),
]
RIBOSEER = "RiboSeer (ours)"
METHODS: list[str] = [n for n, _ in TOOL_METHODS] + [RIBOSEER]
_PCTS = [10, 25, 50, 75, 90]
_PCT_LABELS = ["10%", "25%", "Median", "75%", "90%"]


def per_sample_pearson(samples: list[SampleData], model: Optional[EnrichedFusion],
                       fs_preds: Optional[dict[str, dict[int, float]]] = None
                       ) -> dict[str, list[float]]:
    """``{method: [per-sample Pearson R]}`` over the resolved subset.

    RiboSeer's vector comes from ``fs_preds`` (pre-computed full-system
    predictions) when supplied, else from the loaded ``model``."""
    out: dict[str, list[float]] = {m: [] for m in METHODS}
    for s in samples:
        mask = s.eval_mask
        y = s.y[mask]
        preds: dict[str, np.ndarray] = {}
        for name, tid in TOOL_METHODS:
            if tid in s.raw_scores_by_tool:
                preds[name] = _tool_raw_vector(s, tid)
        if fs_preds is not None:
            preds[RIBOSEER] = predictions_to_vector(
                fs_preds.get(s.sid), s.residue_ids)
        else:
            preds[RIBOSEER] = riboseer_predict(model, s.X)
        for name, vec in preds.items():
            corr = per_sample_corr(np.asarray(vec, dtype=np.float64)[mask], y)
            if corr is not None and corr["pearson_r"] is not None:
                out[name].append(corr["pearson_r"])
    return out


def percentiles(values: list[float]) -> list[Optional[float]]:
    if not values:
        return [None] * len(_PCTS)
    qs = np.percentile(np.asarray(values, dtype=np.float64), _PCTS)
    return [round(float(q), 4) for q in qs]


def print_table(dist: dict[str, list[float]]) -> None:
    print("=== Table 25: Per-Sample Pearson R Distribution ===")
    hdr = (f"{'Method':<17s} {'n':>4s} " +
           " ".join(f"{lbl:>8s}" for lbl in _PCT_LABELS))
    print(hdr)
    print("-" * len(hdr))
    for m in METHODS:
        vals = dist[m]
        qs = percentiles(vals)
        cells = " ".join(f"{(f'{q:.3f}' if q is not None else '–'):>8s}"
                         for q in qs)
        print(f"{m:<17s} {len(vals):>4d} {cells}")


def _load_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"split file not found: {path}")
    return [ln.split()[0] for ln in
            path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--step4-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=None,
                   help="RiboSeer EnrichedFusion bundle; not needed when "
                        "--predictions-dir is given")
    p.add_argument("--predictions-dir", type=Path, default=None,
                   help="pre-computed full-system (on/on/on) predictions "
                        "<sid>.json; when set RiboSeer reads these instead "
                        "of loading the model")
    p.add_argument("--split-file", type=Path, required=True)
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for d in (args.data_dir, args.step4_dir):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 1
    try:
        ids = _load_split(args.split_file)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    fs_preds = None
    model = None
    if args.predictions_dir is not None:
        fs_preds = load_predictions_dir(args.predictions_dir)
        if not fs_preds:
            print(f"ERROR: no predictions under {args.predictions_dir}",
                  file=sys.stderr)
            return 1
        print(f"loaded {len(fs_preds)} full-system predictions from "
              f"{args.predictions_dir}")
    else:
        if args.model_dir is None:
            print("ERROR: pass either --predictions-dir or --model-dir",
                  file=sys.stderr)
            return 1
        try:
            model = EnrichedFusion.load(args.model_dir)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: failed to load model from {args.model_dir}: {e}",
                  file=sys.stderr)
            return 1

    samples = collect_sample_data(
        args.step4_dir, args.data_dir, ids,
        feature_set=(model.feature_set if model else "full"),
        use_context=(model.use_context if model else True))
    if not samples:
        print("ERROR: no usable test samples", file=sys.stderr)
        return 1
    print(f"usable test samples: {len(samples)}")
    print()
    print_table(per_sample_pearson(samples, model, fs_preds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
