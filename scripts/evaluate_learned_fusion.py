"""Evaluate a learned-fusion bundle against the test split.

Loads the model written by ``train_fusion.py`` (Ridge) OR
``train_xgb_fusion.py`` (XGBoost), applies it to every sample in
``--test-step4-dir``, and computes per-residue Pearson R / Spearman R
/ R² against ground truth. The model class is auto-detected from the
bundle dir's contents — ``xgb_model.json`` ⇒ XGBFusion,
``optimizer.json`` ⇒ LearnedFusion. Override with ``--model-type
{auto,ridge,xgboost}``.

When ``--test-step5-dir`` is also passed, the script emits a
side-by-side comparison against the existing noisy-OR fusion baseline
so the paper table can quote the delta directly.

Outputs (under ``--output``):

  - ``learned_fusion_predictions.jsonl``  one record per sample with
    ``{sample_id, per_residue_probability, binding_protein_residues}``;
    the latter is thresholded at ``--binding-threshold`` (default 0.5).
  - ``learned_fusion_metrics.json``       per-sample and aggregate
    Pearson / Spearman / R² for the learned method, plus the baseline
    when available.
  - ``learned_fusion_metrics.csv``        per-sample flat view.
  - ``method_comparison.csv``             aggregate row per method
    (paper-table shape).

Example
-------
    python scripts/evaluate_learned_fusion.py \\
        --test-step4-dir  data/batch_test/step4/ \\
        --test-step5-dir  data/batch_test/step5/ \\
        --processed-dir   data/processed_filtered \\
        --fusion-model    data/fusion_model/ \\
        --output          data/evaluation_learned/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from step5_fusion.learned_fusion import (  # noqa: E402
    LearnedFusion, _load_sample_json, _per_residue_to_int_dict,
    _read_jsonl_record,
)
from step5_fusion.xgb_fusion import (  # noqa: E402
    AgentCalibratedXGBFusion, XGB_OK, XGBFusion,
)


# Auto-detect choices for ``--model-type``. ``auto`` picks based on
# file presence + xgb_meta.json's calibrated flag; the explicit
# values force a specific loader.
_MODEL_TYPES = ("auto", "ridge", "xgboost", "xgboost_calibrated")


def _detect_model_type(bundle_dir: Path) -> str:
    """Return one of ``"ridge"`` / ``"xgboost"`` / ``"xgboost_calibrated"``.

    Detection order:
      1. ``xgb_model.json`` present → XGBoost path. Peek at
         ``xgb_meta.json``; if it carries ``model_type ==
         "xgboost_calibrated"`` or ``calibrated: true``, route to the
         calibrated class. Otherwise standalone.
      2. ``optimizer.json`` present → Ridge.
      3. Neither → bundle is not a learned-fusion artefact.
    """
    if (bundle_dir / "xgb_model.json").is_file():
        meta_path = bundle_dir / "xgb_meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
            if isinstance(meta, dict):
                if (meta.get("model_type") == "xgboost_calibrated"
                        or bool(meta.get("calibrated"))):
                    return "xgboost_calibrated"
        return "xgboost"
    if (bundle_dir / "optimizer.json").is_file():
        return "ridge"
    raise FileNotFoundError(
        f"--fusion-model {bundle_dir} contains neither xgb_model.json "
        f"nor optimizer.json — not a learned-fusion bundle. Re-run "
        f"scripts/train_fusion.py or scripts/train_xgb_fusion.py to "
        f"produce one."
    )


def _load_fusion_model(bundle_dir: Path, model_type: str):
    """Load the appropriate fusion class for ``bundle_dir``.

    Returns ``(instance, resolved_type)`` where ``instance`` is one of
    ``LearnedFusion`` / ``XGBFusion`` / ``AgentCalibratedXGBFusion``.
    The eval loop branches on the resolved type only when the
    inference signature differs (calibrated needs the agent input).
    """
    if model_type == "auto":
        model_type = _detect_model_type(bundle_dir)
    if model_type == "ridge":
        return LearnedFusion.load(bundle_dir), "ridge"
    if model_type == "xgboost":
        if not XGB_OK:
            raise ImportError(
                "xgboost is required to load this bundle but failed "
                "to import. Install with `conda run -n riboseer pip "
                "install xgboost`."
            )
        return XGBFusion.load(bundle_dir), "xgboost"
    if model_type == "xgboost_calibrated":
        if not XGB_OK:
            raise ImportError(
                "xgboost is required to load this bundle but failed "
                "to import. Install with `conda run -n riboseer pip "
                "install xgboost`."
            )
        return AgentCalibratedXGBFusion.load(bundle_dir), "xgboost_calibrated"
    raise ValueError(
        f"unknown --model-type {model_type!r}; expected one of {_MODEL_TYPES}"
    )


# ---- correlation helpers (mirror scripts/evaluate.py — stdlib only) ------


def _pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx2 = sum((x - mx) ** 2 for x in xs)
    sy2 = sum((y - my) ** 2 for y in ys)
    if sx2 == 0 or sy2 == 0:
        return None
    return round(cov / math.sqrt(sx2 * sy2), 6)


def _ranks(vs: list[float]) -> list[float]:
    """Average-tied ranks (1-based)."""
    order = sorted(range(len(vs)), key=lambda i: vs[i])
    ranks = [0.0] * len(vs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _agg(values: list[float]) -> dict:
    """Mean / std / median, rounded for human-readable JSON."""
    if not values:
        return {"mean": None, "std": None, "median": None, "n": 0}
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) >= 2 else 0.0
    return {
        "mean":   round(mean, 6),
        "std":    round(std, 6),
        "median": round(statistics.median(values), 6),
        "n":      len(values),
    }


# ---- per-sample evaluator ------------------------------------------------


def _vectors(
    per_res: dict[int, float], gt: list[int], protein_length: int,
) -> tuple[list[float], list[float]]:
    """Build dense [pred, gt] vectors of length ``protein_length``."""
    gt_set = set(gt)
    pred = [0.0] * protein_length
    label = [0.0] * protein_length
    for i in range(1, protein_length + 1):
        if i in gt_set:
            label[i - 1] = 1.0
        v = per_res.get(i)
        if v is None:
            continue
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        pred[i - 1] = float(v)
    return pred, label


def _correlation(pred: list[float], label: list[float]) -> Optional[dict]:
    if not pred or not label:
        return None
    if len(set(label)) < 2:
        return None
    if max(pred) - min(pred) < 1e-12:
        return None
    pr = _pearson(pred, label)
    if pr is None:
        return None
    sr = _spearman(pred, label)
    return {
        "pearson_r": pr,
        "spearman_r": sr,
        "r_squared": round(pr * pr, 6),
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-step4-dir", type=Path, required=True,
                   help="batch_predict's step4/ dir on the test split.")
    p.add_argument("--test-step5-dir", type=Path, default=None,
                   help="optional batch_predict step5/ dir; when given, "
                        "the script also computes per-residue correlations "
                        "for the noisy-OR baseline so they can be diffed.")
    p.add_argument("--processed-dir", type=Path, required=True,
                   help="step 1 output dir (samples/<id>.json with GT).")
    p.add_argument("--sample-list", type=Path, default=None,
                   help="optional one-id-per-line filter (typical: "
                        "data/splits/test.txt).")
    p.add_argument("--fusion-model", type=Path, required=True,
                   help="dir produced by train_fusion.py (Ridge) or "
                        "train_xgb_fusion.py (XGBoost).")
    p.add_argument("--model-type", choices=_MODEL_TYPES, default="auto",
                   help="auto = detect by file presence "
                        "(xgb_model.json → xgboost, optimizer.json → "
                        "ridge); pass ridge / xgboost to force.")
    p.add_argument("--output", type=Path, required=True,
                   help="dir to write the prediction JSONL + metrics.")
    p.add_argument("--binding-threshold", type=float, default=0.5,
                   help="prob > THRESHOLD → residue lands in "
                        "binding_protein_residues (default 0.5).")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not args.test_step4_dir.is_dir():
        print(f"ERROR: --test-step4-dir not a directory: "
              f"{args.test_step4_dir}", file=sys.stderr)
        return 1
    samples_dir = args.processed_dir / "samples"
    if not samples_dir.is_dir():
        print(f"ERROR: samples/ not under --processed-dir: {samples_dir}",
              file=sys.stderr)
        return 1

    try:
        fusion, resolved_type = _load_fusion_model(
            args.fusion_model, args.model_type,
        )
    except (FileNotFoundError, ImportError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"loaded {resolved_type} fusion model from {args.fusion_model}")

    # Calibrated bundles need the agent's noisy-OR output at inference,
    # so the test-side step5 dir is required (not just used for the
    # noisy-OR baseline comparison).
    if resolved_type == "xgboost_calibrated":
        if args.test_step5_dir is None:
            print("ERROR: a calibrated fusion bundle needs the agent's "
                  "step5 output at inference. Pass --test-step5-dir.",
                  file=sys.stderr)
            return 1
        if not args.test_step5_dir.is_dir():
            print(f"ERROR: --test-step5-dir not a directory: "
                  f"{args.test_step5_dir}", file=sys.stderr)
            return 1

    if args.sample_list is not None:
        sids = [
            line.strip() for line in args.sample_list.read_text(
                encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        sids = sorted(p.stem for p in args.test_step4_dir.glob("*.jsonl"))

    args.output.mkdir(parents=True, exist_ok=True)
    pred_path = args.output / "learned_fusion_predictions.jsonl"

    learned_corr: list[dict] = []
    baseline_corr: list[dict] = []
    n_samples_used = 0
    n_samples_skipped = 0

    with pred_path.open("w", encoding="utf-8") as out_f:
        for sid in sids:
            s4 = _read_jsonl_record(args.test_step4_dir / f"{sid}.jsonl")
            sample = _load_sample_json(args.processed_dir, sid)
            if s4 is None or sample is None:
                n_samples_skipped += 1
                continue
            seq = (sample.get("protein") or {}).get("sequence") or ""
            length = (sample.get("protein") or {}).get("length") or len(seq)
            gt = (
                (sample.get("interaction") or {}).get(
                    "binding_protein_residues") or []
            )
            if not length:
                n_samples_skipped += 1
                continue

            # Read step5 once if available — its agent output is used
            # by both the calibrated predict path AND the noisy-OR
            # baseline correlation below.
            s5 = (
                _read_jsonl_record(args.test_step5_dir / f"{sid}.jsonl")
                if args.test_step5_dir is not None else None
            )

            preds = (s4.get("predictions") or [])
            try:
                if resolved_type == "xgboost_calibrated":
                    if s5 is None:
                        # Calibrated needs the agent input; skip cleanly
                        # and tally the miss for the report.
                        n_samples_skipped += 1
                        continue
                    agent_probs = _per_residue_to_int_dict(
                        s5.get("per_residue_probability"))
                    agent_binding_set = set(
                        s5.get("binding_protein_residues") or []
                    )
                    per_res = fusion.predict_sample(
                        preds, seq, int(length),
                        agent_probs=agent_probs,
                        agent_binding_set=agent_binding_set,
                    )
                else:
                    per_res = fusion.predict_sample(preds, seq, int(length))
            except (RuntimeError, ValueError) as e:
                print(f"WARN: predict_sample failed for {sid}: {e}",
                      file=sys.stderr)
                n_samples_skipped += 1
                continue

            binding = sorted(
                k for k, v in per_res.items()
                if v > args.binding_threshold
            )
            out_f.write(json.dumps({
                "sample_id": sid,
                "per_residue_probability": {str(k): v
                                            for k, v in per_res.items()},
                "binding_protein_residues": binding,
                "threshold": args.binding_threshold,
            }, ensure_ascii=False) + "\n")
            n_samples_used += 1

            # Correlations (only when GT exists, else skip — undefined).
            if not gt:
                continue
            pred_vec, label_vec = _vectors(per_res, gt, int(length))
            corr = _correlation(pred_vec, label_vec)
            if corr is not None:
                learned_corr.append({"sample_id": sid, **corr})

            if s5 is not None:
                base_per = _per_residue_to_int_dict(
                    s5.get("per_residue_probability"))
                pred_b, _ = _vectors(base_per, gt, int(length))
                base_corr = _correlation(pred_b, label_vec)
                if base_corr is not None:
                    baseline_corr.append({
                        "sample_id": sid, **base_corr,
                    })

    # ---- aggregate -------------------------------------------------------
    methods: dict[str, list[dict]] = {"learned_fusion": learned_corr}
    if args.test_step5_dir is not None:
        methods["noisy_or_baseline"] = baseline_corr

    aggregate = {
        method: {
            "pearson_r":  _agg([r["pearson_r"]  for r in rows
                                if r["pearson_r"]  is not None]),
            "spearman_r": _agg([r["spearman_r"] for r in rows
                                if r["spearman_r"] is not None]),
            "r_squared":  _agg([r["r_squared"]  for r in rows
                                if r["r_squared"]  is not None]),
            "n_samples":  len(rows),
        }
        for method, rows in methods.items()
    }

    metrics_json = {
        "n_samples_used":    n_samples_used,
        "n_samples_skipped": n_samples_skipped,
        "binding_threshold": args.binding_threshold,
        "fusion_model":      str(args.fusion_model),
        "model_type":        resolved_type,
        "methods":           aggregate,
    }
    (args.output / "learned_fusion_metrics.json").write_text(
        json.dumps(metrics_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Per-sample CSV (learned only — baseline rows would clutter).
    csv_rows = [
        {"sample_id": r["sample_id"],
         "pearson_r": r["pearson_r"],
         "spearman_r": r["spearman_r"],
         "r_squared": r["r_squared"]}
        for r in learned_corr
    ]
    with (args.output / "learned_fusion_metrics.csv").open(
        "w", encoding="utf-8", newline="",
    ) as f:
        w = csv.DictWriter(
            f, fieldnames=["sample_id", "pearson_r", "spearman_r", "r_squared"],
        )
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)

    # Method-comparison CSV (paper table shape).
    with (args.output / "method_comparison.csv").open(
        "w", encoding="utf-8", newline="",
    ) as f:
        w = csv.DictWriter(f, fieldnames=[
            "method", "n_samples",
            "pearson_r_mean", "pearson_r_std", "pearson_r_median",
            "spearman_r_mean", "spearman_r_std", "spearman_r_median",
            "r2_mean", "r2_std", "r2_median",
        ])
        w.writeheader()
        for method, agg in aggregate.items():
            w.writerow({
                "method": method, "n_samples": agg["n_samples"],
                "pearson_r_mean":   agg["pearson_r"]["mean"],
                "pearson_r_std":    agg["pearson_r"]["std"],
                "pearson_r_median": agg["pearson_r"]["median"],
                "spearman_r_mean":   agg["spearman_r"]["mean"],
                "spearman_r_std":    agg["spearman_r"]["std"],
                "spearman_r_median": agg["spearman_r"]["median"],
                "r2_mean":   agg["r_squared"]["mean"],
                "r2_std":    agg["r_squared"]["std"],
                "r2_median": agg["r_squared"]["median"],
            })

    print(f"wrote {n_samples_used} predictions to {pred_path}")
    print(f"  skipped {n_samples_skipped} samples (missing JSONL / sample / "
          f"empty GT / predict failure)")
    for method, agg in aggregate.items():
        print(f"  [{method:20s}] n={agg['n_samples']:>4}  "
              f"Pearson R = {agg['pearson_r']['mean']}  "
              f"Spearman R = {agg['spearman_r']['mean']}  "
              f"R² = {agg['r_squared']['mean']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
