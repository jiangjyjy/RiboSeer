"""Table 22 — per-sample AUROC / AUPRC.

For each test sample and method, the per-residue ROC-AUC and
Precision-Recall AUC (average precision) are computed on the
resolved-residue subset (same evaluation domain as Tables 4 / 16-21),
then averaged across the valid samples of each method.

Data sources (real layout — reuses the same helpers as the other table
scripts):

* GT ``y*`` + resolved-residue mask from
  ``ablation_fusion_method.collect_sample_data``
  (``interaction.binding_protein_residues`` + ``protein.resolved_residues``
  + step4 JSONL).
* Each tool's per-residue score is ``_tool_raw_vector`` (per_residue_pae_score
  → per_residue_confidence).
* RiboSeer is the loaded ``EnrichedFusion`` bundle's prediction.

A sample is *valid* for a method when the tool produced a score for it AND
the resolved subset has both a positive and a negative residue (AUROC is
undefined otherwise); otherwise it's skipped and not counted in ``n``.

Run
---
::

    python scripts/tables/table21_auroc_auprc.py \\
        --data-dir   data/processed_quality \\
        --step4-dir  data/batch_test_v7/step4 \\
        --model-dir  data/enriched_v7_lgbm \\
        --split-file data/processed_quality/splits_tmscore_035/test.txt
"""
from __future__ import annotations

import argparse
import statistics
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from step5_fusion.enriched_fusion import EnrichedFusion  # noqa: E402
from scripts.riboseer.ablation_fusion_method import (  # noqa: E402
    SampleData, collect_sample_data, _tool_raw_vector,
)
from scripts.tables.table15_rna_length import riboseer_predict  # noqa: E402
from step5_fusion.prediction_io import (  # noqa: E402
    load_predictions_dir, predictions_to_vector,
)

# (display name, step4 tool_id). RiboSeer (fusion) handled separately.
TOOL_METHODS: list[tuple[str, str]] = [
    ("Boltz-2", "boltz2"),
    ("EquiPNAS", "equipnas"),
    ("Chai-1", "chai1"),
    ("P2Rank", "p2rank"),
    ("RoseTTAFold2NA", "rosettafold2na"),
]
RIBOSEER = "RiboSeer (ours)"
METHODS: list[str] = [n for n, _ in TOOL_METHODS] + [RIBOSEER]
_METRIC_KEYS = ["auroc", "auprc"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_auc_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray
                        ) -> Optional[tuple[float, float]]:
    """(AUROC, AUPRC) or None when the sample has only one class."""
    n_pos = int(np.sum(y_true))
    if n_pos == 0 or n_pos == len(y_true):
        return None
    auroc = float(roc_auc_score(y_true, y_pred_prob))
    auprc = float(average_precision_score(y_true, y_pred_prob))
    return auroc, auprc


# ---------------------------------------------------------------------------
# Per-sample method predictions (masked to resolved residues)
# ---------------------------------------------------------------------------


def method_predictions(s: SampleData, model: Optional[EnrichedFusion],
                       preds: Optional[dict[str, dict[int, float]]] = None
                       ) -> dict[str, np.ndarray]:
    """``{method: pred_vector}`` over the FULL residue range for every
    method whose tool produced a score on this sample (+ RiboSeer).

    RiboSeer's vector comes from ``preds`` (pre-computed full-system
    predictions) when supplied, else from the loaded ``model``."""
    out: dict[str, np.ndarray] = {}
    for name, tid in TOOL_METHODS:
        if tid in s.raw_scores_by_tool:
            out[name] = _tool_raw_vector(s, tid)
    if preds is not None:
        out[RIBOSEER] = predictions_to_vector(preds.get(s.sid), s.residue_ids)
    else:
        out[RIBOSEER] = riboseer_predict(model, s.X)
    return out


def evaluate(samples: list[SampleData], model: Optional[EnrichedFusion],
             preds: Optional[dict[str, dict[int, float]]] = None
             ) -> dict[str, list[dict]]:
    """``{method: [per-sample metric dict]}``."""
    bucket: dict[str, list[dict]] = {m: [] for m in METHODS}
    for s in samples:
        mask = s.eval_mask
        y = s.y[mask].astype(int)
        n_pos = int(y.sum())
        if n_pos == 0 or n_pos == y.size:        # single class → undefined
            continue
        for name, pred_full in method_predictions(s, model, preds).items():
            pred = np.asarray(pred_full, dtype=np.float64)[mask]
            res = compute_auc_metrics(y, pred)
            if res is None:
                continue
            auroc, auprc = res
            bucket[name].append({"auroc": auroc, "auprc": auprc})
    return bucket


def _mean(vs):
    return statistics.fmean(vs) if vs else None


def aggregate(bucket: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for m in METHODS:
        rs = bucket[m]
        row = {"method": m, "n": len(rs)}
        for key in _METRIC_KEYS:
            row[key] = _mean([r[key] for r in rs])
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Output (stdout only — no LaTeX file)
# ---------------------------------------------------------------------------


def print_table(rows: list[dict]) -> None:
    print("=== Table 22: AUROC / AUPRC (per-sample mean) ===")
    hdr = f"{'Method':<17s} {'n':>4s} {'AUROC':>7s} {'AUPRC':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def fmt(v):
            return f"{v:.3f}" if v is not None else "  –  "
        print(f"{r['method']:<17s} {r['n']:>4d} "
              f"{fmt(r['auroc']):>7s} {fmt(r['auprc']):>7s}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _load_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"split file not found: {path}")
    return [ln.split()[0] for ln in
            path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def _dry_run(data_dir: Path, step4_dir: Path, model_dir: Path,
             ids: list[str]) -> int:
    print("[dry-run] would read:")
    print(f"  model bundle : {model_dir}")
    print(f"  split        : {len(ids)} sample ids")
    print(f"  samples      : {data_dir}/samples/<sid>.json  "
          f"(e.g. {data_dir}/samples/{ids[0]}.json)")
    print(f"  step4        : {step4_dir}/<sid>.jsonl  "
          f"(e.g. {step4_dir}/{ids[0]}.jsonl)")
    print(f"  tools        : {', '.join(t for _, t in TOOL_METHODS)} + "
          f"RiboSeer(fusion)")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="processed dir (has samples/<sid>.json)")
    p.add_argument("--step4-dir", type=Path, required=True,
                   help="test-split step4 JSONL dir")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="RiboSeer EnrichedFusion bundle (enriched_v7_lgbm); "
                        "not needed when --predictions-dir is given")
    p.add_argument("--predictions-dir", type=Path, default=None,
                   help="pre-computed full-system (on/on/on) predictions "
                        "<sid>.json; when set RiboSeer reads these instead "
                        "of loading the model")
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true",
                   help="print the files that would be read, then exit")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    try:
        ids = _load_split(args.split_file)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        return _dry_run(args.data_dir, args.step4_dir, args.model_dir, ids)

    for d in (args.data_dir, args.step4_dir):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 1

    preds = None
    model = None
    if args.predictions_dir is not None:
        preds = load_predictions_dir(args.predictions_dir)
        if not preds:
            print(f"ERROR: no predictions under {args.predictions_dir}",
                  file=sys.stderr)
            return 1
        print(f"loaded {len(preds)} full-system predictions from "
              f"{args.predictions_dir}")
    else:
        if args.model_dir is None:
            print("ERROR: pass either --predictions-dir or --model-dir",
                  file=sys.stderr)
            return 1
        try:
            model = EnrichedFusion.load(args.model_dir)
        except Exception as e:  # noqa: BLE001 — surface load failure clearly
            print(f"ERROR: failed to load model from {args.model_dir}: {e}",
                  file=sys.stderr)
            return 1

    samples = collect_sample_data(
        args.step4_dir, args.data_dir, ids,
        feature_set=(model.feature_set if model else "full"),
        use_context=(model.use_context if model else True))
    if not samples:
        print("ERROR: no usable test samples (check --data-dir/--step4-dir)",
              file=sys.stderr)
        return 1
    print(f"usable test samples: {len(samples)}")

    rows = aggregate(evaluate(samples, model, preds))
    print()
    print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
