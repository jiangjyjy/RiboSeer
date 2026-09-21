#!/usr/bin/env python3
"""Table 7 (v2) — HARMONY fusion-method ablation on the new headline space.

Why a v2
--------
The original ``ablation_fusion_method`` compared 8 fusion strategies on the
**old** 79-D enriched features (fixed 5/6-tool ``TOOL_ORDER``). The headline
system is now MANDATORY optimal-7 ∪ MAESTRO selection + SCOPE profile →
``features_15tool.build_15tool_features`` (**154-D**), so Table 7 must be
re-run on that exact feature space for every method to be compared fairly.

This v2 reuses the headline machinery verbatim (``table09_llm_modules``):
per sample the tool set is ``K_LLM ∪ config.MANDATORY_TOOLS`` and the 154-D
matrix is ``build_sample_matrix(s, scope_on=True, maestro_on=True, …,
mandatory=MANDATORY_TOOLS)`` — i.e. exactly the Table 9 on/on row. **POLISH
is never applied** (Table 7 compares pure fusion methods, no post-edit), so
the LightGBM row equals the Table 9 SCOPE+MAESTRO, POLISH-off number
(≈ 0.593) rather than the 0.588 headline.

Eight methods (output order)
----------------------------
No training — operate on the raw per-residue scores
(``per_residue_pae_score`` → ``per_residue_confidence`` fallback) of the
SELECTED tools that are present/successful in step4:

1. ``mean_raw``  — per-residue mean over selected tools
2. ``max_raw``   — per-residue max
3. ``noisy_or``  — ``1 − ∏_k (1 − c·s_k(i))``, uniform ``c = 1/K``

Train on the 154-D feature matrix (concat over train samples), predict per
test sample:

4. ``logistic_regression`` — sklearn LR (class_weight='balanced')
5. ``random_forest``       — sklearn RandomForestRegressor (500×6)
6. ``mlp_3x256``           — sklearn MLPRegressor (256×256×256)
7. ``xgboost``             — XGBRegressor on the 154-D matrix
8. ``lightgbm``            — LGBMRegressor (= headline recipe; the "ours" row)

NaN handling: an inactive (un-selected / absent) tool's columns are NaN.
LightGBM / XGBoost split on NaN natively; the sklearn rows (LR / RF / MLP)
cannot, so their matrices are imputed NaN→0 (deterministic, train and test
identically). The SCOPE G4 block is never NaN.

Metrics: per-sample Pearson / Spearman / R² on the resolved-residue subset
(``eval_mask``), mean across samples — same convention as Table 9 / 10.

Usage
-----
::

    python scripts/tables/table07_fusion_method.py \\
        --processed-dir      data/processed_quality \\
        --train-step4-dir    data/batch_train_v7/step4 \\
        --test-step4-dir     data/batch_test_v7/step4 \\
        --train-list         data/processed_quality/splits_tmscore_035/train.txt \\
        --test-list          data/processed_quality/splits_tmscore_035/test.txt \\
        --scope-dir-train    data/batch_train_v7/scope_profiles_llm \\
        --scope-dir-test     data/batch_test_v7/scope_profiles_llm \\
        --maestro-dir-train  data/batch_train_v7/maestro_selections_llm_v4 \\
        --maestro-dir-test   data/batch_test_v7/maestro_selections_llm_v4 \\
        --output             data/batch_test_v7/table7_v2.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from step5_fusion.data_collector import per_residue_to_int_dict  # noqa: E402
# Headline 154-D machinery (MANDATORY-aware): tool resolution + feature build.
from step5_fusion.lightgbm_fusion import (  # noqa: E402
    LGBM_OK,
    SampleT9,
    _load_json_dir,
    build_sample_matrix,
    collect_samples,
    resolve_selected_tools,
)
from scripts.tables.table09_llm_modules import (  # noqa: E402
    _load_sample_ids,
)
from scripts.riboseer.config import MANDATORY_TOOLS  # noqa: E402
from step5_fusion.features_15tool import (  # noqa: E402
    _index_predictions,
)
# Reuse the v1 estimators + metric verbatim so rows stay comparable.
from scripts.riboseer.ablation_fusion_method import (  # noqa: E402
    TrainablePredictor, _make_logistic_regression, _make_lightgbm,
    _make_mlp, _make_random_forest, _tool_raw_scores, per_sample_corr,
)


# ---------------------------------------------------------------------------
# Per-sample selected-tool raw scores (for the no-train methods)
# ---------------------------------------------------------------------------


def selected_raw_scores(sample: SampleT9, selections: dict[str, dict]
                        ) -> dict[str, dict[int, float]]:
    """``{tool: {res_id: score}}`` for the tools in this sample's headline
    selection (``K_LLM ∪ MANDATORY_TOOLS``) that actually produced a score
    in step4. Same raw signal (pae→confidence) the v1 Table-7 mean/max/
    noisy-OR rows used, restricted to the selected set."""
    selected = set(resolve_selected_tools(
        sample, True, selections, mandatory=MANDATORY_TOOLS))
    out: dict[str, dict[int, float]] = {}
    for tid, pred in _index_predictions(sample.step4_data).items():
        if tid not in selected:
            continue
        scores = _tool_raw_scores(pred)
        if scores:
            out[tid] = scores
    return out


def _stack(raw: dict[str, dict[int, float]],
           residue_ids: list[int]) -> np.ndarray:
    if not raw:
        return np.zeros((0, len(residue_ids)))
    return np.vstack([
        np.fromiter((float(sc.get(r, 0.0)) for r in residue_ids),
                    dtype=np.float64, count=len(residue_ids))
        for sc in raw.values()])


def predict_mean(raw, residue_ids) -> np.ndarray:
    stack = _stack(raw, residue_ids)
    return stack.mean(axis=0) if stack.size else np.zeros(len(residue_ids))


def predict_max(raw, residue_ids) -> np.ndarray:
    stack = _stack(raw, residue_ids)
    return stack.max(axis=0) if stack.size else np.zeros(len(residue_ids))


def predict_noisy_or(raw, residue_ids) -> np.ndarray:
    """``1 − ∏_k (1 − c·s_k(i))`` with uniform ``c = 1/K`` over the selected
    tools that scored this sample; raw scores clamped to [0, 1]."""
    n = len(residue_ids)
    if not raw:
        return np.zeros(n)
    c = 1.0 / len(raw)
    one_minus = np.ones(n, dtype=np.float64)
    for sc in raw.values():
        s = np.fromiter((float(sc.get(r, 0.0)) for r in residue_ids),
                        dtype=np.float64, count=n)
        np.clip(s, 0.0, 1.0, out=s)
        one_minus *= (1.0 - c * s)
    return 1.0 - one_minus


NO_TRAIN: dict[str, Callable[[dict, list[int]], np.ndarray]] = {
    "mean_raw": predict_mean,
    "max_raw": predict_max,
    "noisy_or": predict_noisy_or,
}


# ---------------------------------------------------------------------------
# Trainable estimators on the 154-D feature space
# ---------------------------------------------------------------------------


def _make_xgboost() -> Optional[TrainablePredictor]:
    """XGBRegressor on the 154-D matrix (NaN-native, like LightGBM). Recipe
    mirrors the LightGBM row so the two tree methods are directly
    comparable. ``None`` when xgboost isn't installed."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return None
    return TrainablePredictor(
        "xgboost",
        XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            random_state=42, verbosity=0),
        use_proba=False)


# (key, label, factory, impute_nan).  impute_nan=True → NaN→0 before fit/
# predict (sklearn methods); tree methods keep NaN (split on missing).
TRAINABLE: list[tuple[str, str, Callable[[], Optional[TrainablePredictor]],
                      bool]] = [
    ("logistic_regression", "Logistic Regression",
     _make_logistic_regression, True),
    ("random_forest", "Random Forest", _make_random_forest, True),
    ("mlp_3x256", "MLP (3 layers, 256 hidden)", _make_mlp, True),
    ("xgboost", "XGBoost", _make_xgboost, False),
    ("lightgbm", "LightGBM (ours)", _make_lightgbm, False),
]

# Display labels for the no-train rows.
NO_TRAIN_LABELS = {
    "mean_raw": "Mean of raw scores",
    "max_raw": "Max of raw scores",
    "noisy_or": "Noisy OR (prototype)",
}

# Output order (8 rows).
METHOD_ORDER = ["mean_raw", "max_raw", "noisy_or",
                "logistic_regression", "random_forest", "mlp_3x256",
                "xgboost", "lightgbm"]
LABELS = {**NO_TRAIN_LABELS, **{k: lbl for k, lbl, _f, _i in TRAINABLE}}


# ---------------------------------------------------------------------------
# Feature matrices
# ---------------------------------------------------------------------------


def build_matrix(sample: SampleT9, profiles: dict[str, dict],
                 selections: dict[str, dict]) -> np.ndarray:
    """Headline 154-D matrix for one sample (SCOPE on, MAESTRO ∪ MANDATORY)."""
    return build_sample_matrix(sample, True, True, profiles, selections,
                               mandatory=MANDATORY_TOOLS)


def _corr_vec(vec: np.ndarray, sample: SampleT9) -> Optional[dict]:
    """Per-sample corr of a full-length residue vector on the eval subset."""
    mask = sample.eval_mask
    if mask.size != vec.size:
        return None
    return per_sample_corr(vec[mask], sample.y[mask])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def eval_no_train(key: str, fn, test: list[SampleT9],
                  selections_test: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for s in test:
        raw = selected_raw_scores(s, selections_test)
        corr = _corr_vec(fn(raw, s.residue_ids), s)
        if corr is not None:
            rows.append({"sample_id": s.sid, "method": key, **corr})
    return rows


def eval_trainable(predictor: TrainablePredictor, impute: bool,
                   train: list[SampleT9], test: list[SampleT9],
                   profiles_train, profiles_test,
                   selections_train, selections_test) -> list[dict]:
    X_train = np.vstack([build_matrix(s, profiles_train, selections_train)
                         for s in train])
    y_train = np.concatenate([s.y for s in train])
    if impute:
        X_train = np.nan_to_num(X_train, nan=0.0)
    predictor.train(X_train, y_train)

    rows: list[dict] = []
    for s in test:
        X = build_matrix(s, profiles_test, selections_test)
        if impute:
            X = np.nan_to_num(X, nan=0.0)
        vec = np.asarray(predictor.predict(X), dtype=np.float64)
        corr = _corr_vec(vec, s)
        if corr is not None:
            rows.append({"sample_id": s.sid, "method": predictor.name, **corr})
    return rows


def run_ablation(
    *, train: list[SampleT9], test: list[SampleT9],
    profiles_train, profiles_test, selections_train, selections_test,
    skip: Optional[set[str]] = None,
) -> dict[str, list[dict]]:
    skip = skip or set()
    bucket: dict[str, list[dict]] = {}

    for key, fn in NO_TRAIN.items():
        if key in skip:
            continue
        bucket[key] = eval_no_train(key, fn, test, selections_test)
        print(f"  {LABELS[key]:30s} n={len(bucket[key])}")

    for key, _label, factory, impute in TRAINABLE:
        if key in skip:
            continue
        predictor = factory()
        if predictor is None:
            print(f"  WARN: {LABELS[key]} unavailable (missing dep), skipped",
                  file=sys.stderr)
            bucket[key] = []
            continue
        bucket[key] = eval_trainable(
            predictor, impute, train, test,
            profiles_train, profiles_test, selections_train, selections_test)
        print(f"  {LABELS[key]:30s} n={len(bucket[key])}"
              f"{' (NaN→0 imputed)' if impute else ' (NaN-native)'}")
    return bucket


# ---------------------------------------------------------------------------
# Aggregation + output
# ---------------------------------------------------------------------------


def _mean(vs): return round(statistics.fmean(vs), 4) if vs else None


def aggregate(bucket: dict[str, list[dict]]) -> list[dict]:
    rows: list[dict] = []
    for key in METHOD_ORDER:
        rs = bucket.get(key, [])
        prs = [r["pearson_r"] for r in rs if r["pearson_r"] is not None]
        srs = [r["spearman_r"] for r in rs if r["spearman_r"] is not None]
        r2s = [r["r_squared"] for r in rs if r["r_squared"] is not None]
        rows.append({
            "method": key, "label": LABELS[key], "n_samples": len(rs),
            "pearson_r": _mean(prs), "spearman_r": _mean(srs),
            "r_squared": _mean(r2s)})
    return rows


_COLUMNS = ["method", "label", "n_samples",
            "pearson_r", "spearman_r", "r_squared"]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in _COLUMNS})


def write_per_sample_csv(path: Path, bucket: dict[str, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["sample_id", "method", "pearson_r", "spearman_r", "r_squared"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for key in METHOD_ORDER:
            for r in bucket.get(key, []):
                w.writerow({c: r.get(c) for c in cols})


def _fnum(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def print_table(rows: list[dict]) -> None:
    print("=== Table 7: HARMONY Fusion Method Ablation ===")
    hdr = f"{'Fusion method':<30s} {'PearsonR':>8s} {'SpearmanR':>10s} {'R2':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:<30s} {_fnum(r['pearson_r']):>8s} "
              f"{_fnum(r['spearman_r']):>10s} {_fnum(r['r_squared']):>8s}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--train-step4-dir", type=Path, required=True)
    p.add_argument("--test-step4-dir", type=Path, required=True)
    p.add_argument("--train-list", type=Path, required=True)
    p.add_argument("--test-list", type=Path, required=True)
    p.add_argument("--scope-dir-train", type=Path, default=None)
    p.add_argument("--scope-dir-test", type=Path, default=None)
    p.add_argument("--maestro-dir-train", type=Path, default=None)
    p.add_argument("--maestro-dir-test", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--skip", action="append", default=[],
                   choices=METHOD_ORDER, help="skip a method (repeatable).")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for d in (args.processed_dir, args.train_step4_dir, args.test_step4_dir):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 1
    try:
        train_ids = _load_sample_ids(args.train_list)
        test_ids = _load_sample_ids(args.test_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    train = collect_samples(args.train_step4_dir, args.processed_dir,
                            train_ids)
    test = collect_samples(args.test_step4_dir, args.processed_dir, test_ids)
    print(f"train usable: {len(train)}, test usable: {len(test)}")
    if not train or not test:
        print("ERROR: no usable train/test samples", file=sys.stderr)
        return 1

    profiles_train = _load_json_dir(args.scope_dir_train)
    profiles_test = _load_json_dir(args.scope_dir_test)
    selections_train = _load_json_dir(args.maestro_dir_train)
    selections_test = _load_json_dir(args.maestro_dir_test)
    print(f"loaded SCOPE: train={len(profiles_train)} test={len(profiles_test)}"
          f"  MAESTRO: train={len(selections_train)} "
          f"test={len(selections_test)}  (POLISH not applied)")

    bucket = run_ablation(
        train=train, test=test,
        profiles_train=profiles_train, profiles_test=profiles_test,
        selections_train=selections_train, selections_test=selections_test,
        skip=set(args.skip))
    rows = aggregate(bucket)
    write_csv(args.output, rows)
    ps_path = args.output.with_name(args.output.stem + "_per_sample.csv")
    write_per_sample_csv(ps_path, bucket)
    print(f"\nwrote {args.output}  ({len(rows)} methods)")
    print(f"wrote {ps_path}\n")
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
