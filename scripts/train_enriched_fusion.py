"""Train the enriched multi-metric + cross-tool fusion model.

The 5-tool design (boltz2, chai1, equipnas, p2rank, fpocket) with the
48-d feature set (28 base multi-metric + 10 score cross-terms + 10 gate
cross-terms). See ``src/step5_fusion/enriched_fusion.py`` for the
column contract.

Outputs land under ``--output-dir``:

  - ``model.json``            XGBoost native JSON (or Ridge weights)
  - ``enriched_meta.json``    model_type / feature_set / feature_names
  - ``training_report.json``  n_rows, pos_rate, train Pearson/Spearman/R²,
                              full ranked feature importances

Examples
--------
::

    python scripts/train_enriched_fusion.py \\
        --step4-dir    data/batch_train_v4/step4/ \\
        --processed-dir data/processed_filtered \\
        --sample-list  data/processed_filtered/splits/train_200.txt \\
        --output-dir   data/enriched_fusion_model/ \\
        --model xgboost

Ablation (base 28-d only)::

    python scripts/train_enriched_fusion.py ... \\
        --feature-set base --output-dir data/enriched_fusion_model_base/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from step5_fusion.enriched_fusion import EnrichedFusion  # noqa: E402


def _load_yaml(path: Optional[Path]) -> dict:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if isinstance(data, dict) and isinstance(
            data.get("enriched_fusion"), dict):
        return data["enriched_fusion"]
    return data if isinstance(data, dict) else {}


def _load_sample_ids(path: Optional[Path]) -> Optional[list[str]]:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"sample-list not found: {path}")
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step4-dir", type=Path, required=True,
                   help="batch_predict's step4/ dir (one JSONL/sample).")
    p.add_argument("--processed-dir", type=Path, required=True,
                   help="step1 output dir (samples/<id>.json with GT).")
    p.add_argument("--sample-list", type=Path, default=None,
                   help="one-id-per-line text file restricting training "
                        "to a subset (typical: the train split).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="dir where the model bundle is written.")
    p.add_argument("--model",
                   choices=("xgboost", "ridge", "lightgbm"),
                   default=None, help="learner (default: xgboost; "
                   "falls back to YAML 'model' then xgboost). "
                   "lightgbm requires `pip install lightgbm`.")
    p.add_argument("--feature-set", choices=("full", "base"),
                   default=None,
                   help="full (default), base=28 (cross-term "
                        "ablation). Falls back to YAML then 'full'.")
    p.add_argument("--no-context", action="store_true",
                   help="disable the 13-col ±5 neighbourhood block "
                        "(full→48 instead of 61); for the context "
                        "ablation row. Ignored when feature-set=base.")
    p.add_argument("--config", type=Path, default=None,
                   help="optional YAML with hyperparameters.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cfg = _load_yaml(args.config)
    if args.model:
        cfg["model"] = args.model
    if args.feature_set:
        cfg["feature_set"] = args.feature_set
    if args.no_context:
        cfg["use_context"] = False
    cfg.setdefault("model", "xgboost")
    cfg.setdefault("feature_set", "full")
    cfg.setdefault("use_context", True)

    if not args.step4_dir.is_dir():
        print(f"ERROR: --step4-dir not a directory: {args.step4_dir}",
              file=sys.stderr)
        return 1
    if not (args.processed_dir / "samples").is_dir():
        print(f"ERROR: samples/ not under --processed-dir: "
              f"{args.processed_dir}", file=sys.stderr)
        return 1
    try:
        sample_ids = _load_sample_ids(args.sample_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    fusion = EnrichedFusion(cfg)
    try:
        report = fusion.train(
            step4_dir=args.step4_dir,
            processed_dir=args.processed_dir,
            sample_ids=sample_ids,
            verbose=not args.quiet,
        )
    except (RuntimeError, ImportError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out_dir = fusion.save(args.output_dir)
    print()
    print(f"wrote enriched fusion bundle to {out_dir}")
    print(f"  model.json            ({report['model_type']}, "
          f"feature_set={report['feature_set']}, "
          f"context={report['use_context']}, d={report['n_features']})")
    print(f"  enriched_meta.json")
    print(f"  training_report.json  (n_rows={report['n_rows']}, "
          f"Pearson R={report['pearson_r']:.4f}, "
          f"R²={report['r2']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
