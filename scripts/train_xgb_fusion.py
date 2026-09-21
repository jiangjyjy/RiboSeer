"""Train an XGBoost-based learned-fusion bundle.

Two modes:

  - ``--mode standalone`` (default): :class:`XGBFusion` — the model
    REPLACES the agent's noisy-OR fusion. Inputs are step4 + GT.
  - ``--mode calibrated``: :class:`AgentCalibratedXGBFusion` — the
    agent's noisy-OR output (from step5) is fed in as a feature and
    the small XGB model only learns to RE-RANK / re-calibrate the
    agent's per-residue probability. Inputs are step4 + step5 + GT;
    pass ``--train-step5-dir``. This matches the user's "joint reference"
    spec — the agent stays as the primary predictor.

The two modes share data collection (``step5_fusion.data_collector``)
and feature builder (``WeightOptimizer``); the calibrated mode just
prepends two columns ``[agent_fusion_prob, agent_in_binding]`` and
fits a smaller, more conservative XGBoost (``max_depth=3``,
``n_estimators=30``, ``min_child_weight=20``).

Outputs land under ``--output``:

  - ``standardizer.json``    per-tool {mean, std, n}
  - ``xgb_model.json``       XGBoost native JSON
  - ``xgb_meta.json``        layout + ``model_type``
                             (``xgboost`` or ``xgboost_calibrated``)
                             + ``calibrated: bool``
  - ``training_report.json`` n_rows, pos_rate, train Pearson / R²,
                             top-10 feature importances

``scripts/evaluate_learned_fusion.py`` auto-detects via
``xgb_meta.json``'s ``model_type`` / ``calibrated`` flag.

Examples
--------
Standalone (replaces agent fusion)::

    python scripts/train_xgb_fusion.py \\
        --train-step4-dir data/batch_train/step4/ \\
        --processed-dir   data/processed_filtered \\
        --output          data/xgb_model/ \\
        --config          configs/xgb_fusion_config.yaml

Calibrated (XGBoost on top of agent fusion)::

    python scripts/train_xgb_fusion.py \\
        --train-step4-dir data/batch_train/step4/ \\
        --train-step5-dir data/batch_train/step5/ \\
        --processed-dir   data/processed_filtered \\
        --output          data/xgb_calibrated/ \\
        --config          configs/xgb_calibrated_config.yaml \\
        --mode calibrated
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from step5_fusion.xgb_fusion import (  # noqa: E402
    AgentCalibratedXGBFusion, XGB_OK, XGBFusion,
)


_MODES = ("standalone", "calibrated")


def _load_yaml(path: Optional[Path]) -> dict:
    """Load a YAML config; accept either flat or ``xgb_fusion: {...}`` wrapped."""
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    if "xgb_fusion" in data and isinstance(data["xgb_fusion"], dict):
        return data["xgb_fusion"]
    return data


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
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-step4-dir", type=Path, required=True,
                   help="batch_predict's step4/ dir on the train split.")
    p.add_argument("--train-step5-dir", type=Path, default=None,
                   help="batch_predict's step5/ dir — REQUIRED when "
                        "--mode calibrated; ignored in standalone mode.")
    p.add_argument("--processed-dir", type=Path, required=True,
                   help="step 1 output dir (samples/<id>.json with GT).")
    p.add_argument("--sample-list", type=Path, default=None,
                   help="optional one-id-per-line text file restricting "
                        "training to a subset (typical: data/splits/"
                        "train.txt).")
    p.add_argument("--output", type=Path, required=True,
                   help="dir where the XGBoost bundle is written.")
    p.add_argument("--config", type=Path, default=None,
                   help="xgb_fusion_config.yaml; missing file → use "
                        "default hyperparameters for the selected mode.")
    p.add_argument("--mode", choices=_MODES, default=None,
                   help="standalone (default) replaces the agent "
                        "fusion; calibrated layers XGBoost on TOP of "
                        "the agent's noisy-OR output (needs "
                        "--train-step5-dir). When omitted, falls back "
                        "to the YAML's 'mode' field, then 'standalone'.")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-feature importance printout")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not XGB_OK:
        print(
            "ERROR: xgboost not importable in this environment. "
            "Install it with `conda run -n riboseer pip install xgboost`.",
            file=sys.stderr,
        )
        return 1

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

    # Mode resolution: CLI > YAML > default ('standalone'). Keeping it
    # explicit so a config-less invocation still trains something.
    mode = args.mode or cfg.get("mode") or "standalone"
    if mode not in _MODES:
        print(f"ERROR: unknown --mode {mode!r}; expected one of {_MODES}",
              file=sys.stderr)
        return 1

    if mode == "calibrated":
        if args.train_step5_dir is None:
            print("ERROR: --mode calibrated requires --train-step5-dir "
                  "(the agent's noisy-OR fusion output is the model's "
                  "primary feature).", file=sys.stderr)
            return 1
        if not args.train_step5_dir.is_dir():
            print(f"ERROR: --train-step5-dir not a directory: "
                  f"{args.train_step5_dir}", file=sys.stderr)
            return 1
        fusion = AgentCalibratedXGBFusion(cfg)
        try:
            report = fusion.train(
                step4_dir=args.train_step4_dir,
                step5_dir=args.train_step5_dir,
                processed_dir=args.processed_dir,
                sample_ids=sample_ids,
                verbose=not args.quiet,
            )
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    else:
        fusion = XGBFusion(cfg)
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
    label = (
        "calibrated XGBoost (on top of agent fusion)"
        if mode == "calibrated"
        else "standalone XGBoost"
    )
    print(f"wrote {label} bundle to {out_dir}")
    print(f"  standardizer.json      "
          f"({len(fusion.standardizer.params)} tools)")
    print(f"  xgb_model.json         "
          f"(d={report['n_features']}, "
          f"n_trees={fusion.xgb_params.get('n_estimators')})")
    print(f"  xgb_meta.json")
    print(f"  training_report.json   (n_rows={report['n_rows']}, "
          f"Pearson R={report['pearson_r']}, R²={report['r2']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
