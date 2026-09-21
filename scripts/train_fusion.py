"""Train the learned-fusion bundle (standardiser + Ridge optimiser).

Reads the per-residue tool scores already produced by ``batch_predict.py``
on the train split — no need to re-run the structure tools — and fits

  - ``ScoreStandardizer``: per-tool z-score → sigmoid
  - ``WeightOptimizer`` (Ridge): tool / residue-type / optional cross
    features → binary binding label

Outputs land under ``--output``:

  - ``standardizer.json``       per-tool {mean, std, n}
  - ``optimizer.json``          weights + bias + feature names
  - ``feature_names.json``      tool_order + score_field per tool + config
  - ``training_report.json``    n_rows, RMSE, R², per-tool counts, weights

Use ``scripts/evaluate_learned_fusion.py`` to load the bundle and score
a held-out test split. The learned fusion is purely a post-hoc model
on top of step 4's outputs — it does NOT replace step 5's noisy-OR
fusion in the live pipeline; both can be compared in the eval script.

Example
-------
    python scripts/train_fusion.py \\
        --train-step4-dir data/batch_train/step4/ \\
        --processed-dir   data/processed_filtered \\
        --output          data/fusion_model/ \\
        --config          configs/learned_fusion_config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from step5_fusion.learned_fusion import LearnedFusion  # noqa: E402


def _load_yaml(path: Optional[Path]) -> dict:
    """Load a YAML config or return ``{}`` for missing path."""
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    # Allow both ``learned_fusion: {...}`` wrapping and a flat dict.
    if "learned_fusion" in data and isinstance(data["learned_fusion"], dict):
        return data["learned_fusion"]
    return data


def _load_sample_ids(path: Optional[Path]) -> Optional[list[str]]:
    """Optional sample-list filter — same one-id-per-line format
    batch_predict.py uses."""
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
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-step4-dir", type=Path, required=True,
                   help="batch_predict's step4/ dir on the train split.")
    p.add_argument("--processed-dir", type=Path, required=True,
                   help="step 1 output dir (samples/<id>.json with GT).")
    p.add_argument("--sample-list", type=Path, default=None,
                   help="optional one-id-per-line text file restricting "
                        "training to a subset (typical: data/splits/"
                        "train.txt).")
    p.add_argument("--output", type=Path, required=True,
                   help="dir where the learned-fusion bundle is written.")
    p.add_argument("--config", type=Path, default=None,
                   help="learned_fusion_config.yaml; missing file → use "
                        "defaults (use_residue_type=true, no cross terms, "
                        "λ=0.01, default per-tool score_fields).")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-feature weight printout")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    cfg = _load_yaml(args.config)
    if not args.train_step4_dir.is_dir():
        print(f"ERROR: --train-step4-dir not a directory: "
              f"{args.train_step4_dir}", file=sys.stderr)
        return 1
    samples_dir = args.processed_dir / "samples"
    if not samples_dir.is_dir():
        print(f"ERROR: samples/ not under --processed-dir: {samples_dir}",
              file=sys.stderr)
        return 1

    try:
        sample_ids = _load_sample_ids(args.sample_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    fusion = LearnedFusion(cfg)
    try:
        report = fusion.train(
            step4_dir=args.train_step4_dir,
            processed_dir=args.processed_dir,
            sample_ids=sample_ids,
            verbose=not args.quiet,
        )
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out_dir = fusion.save(args.output)

    print()
    print(f"wrote learned-fusion bundle to {out_dir}")
    print(f"  standardizer.json   ({len(fusion.standardizer.params)} tools)")
    print(f"  optimizer.json      "
          f"(d={report['n_features']}, λ={report['lambda']})")
    print(f"  feature_names.json")
    print(f"  training_report.json (n_rows={report['n_rows']}, "
          f"RMSE={report['rmse']:.4f}, R²={report['r2']:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
