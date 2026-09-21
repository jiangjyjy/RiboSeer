"""Fusion-method ablation for paper Table 7.

Compares 8 fusion strategies on the same feature space / train+test
split. XGBoost is our default (Pearson R = 0.524); this script supplies
the other 7 rows.

The metrics are per-sample Pearson / Spearman / R² (same convention as
``scripts/tables/table04_main_results.py``): for each test sample we score every
residue, correlate against the binary GT vector, then mean across
samples. Samples whose correlation is undefined (constant predictions
or all-zero GT) are skipped.

Methods
-------

No training (operate on raw per-residue scores, ``per_residue_pae_score``
→ ``per_residue_confidence`` fallback, same priority as
``table04_main_results.py``):

* ``mean_raw``   — average across all tools (missing tool → 0)
* ``max_raw``    — max across all tools
* ``noisy_or``   — 1 − ∏_k (1 − c·s_k(i)) with uniform c = 1/K

Train on ``--train-list`` (concatenated rows over all train samples,
79-D enriched features), predict on ``--test-list``:

* ``logistic_regression``  — sklearn LR (class_weight='balanced',
                             predict_proba)
* ``random_forest``        — sklearn RandomForestRegressor
                             (n_estimators=500, max_depth=6)
* ``mlp_3x256``            — sklearn MLPRegressor (256×3, ReLU,
                             lr=1e-3, 100 epochs)
* ``lightgbm``             — LightGBM regressor (optional dep; row
                             skipped with a one-line warning when not
                             installed)
* ``xgboost``              — re-uses ``EnrichedFusion`` on the same
                             train split if ``--enriched-model-dir``
                             is not given, otherwise loads the bundle

Category-E naive ensembles (paper Table 4 — operate on the SAME raw
per-residue tool scores as Mean/Max/Noisy-OR, so they share Cat E's
"5 tools" set, derived from the data — see ``present_tools``):

* ``weighted_mean``        — performance-based ("public") weighting:
                             w_k = the tool's mean per-sample Pearson R
                             on the TRAIN split (the very number Table 4's
                             per-tool rows report). Per residue:
                             Σ_k w_k·s_k / Σ_k w_k over the tools present
                             at that sample; negative weights clamped to 0.
* ``stacked_lr``           — sklearn LogisticRegression on the 5-D raw
                             tool-score vector ONLY (one column per tool),
                             NOT the 79-D enriched features. Deliberately
                             distinct from the Table 7 ``logistic_regression``
                             row (79-D); class_weight='balanced' matches it.

Usage
-----
::

    python scripts/riboseer/ablation_fusion_method.py \\
        --train-step4-dir data/batch_train_v7/step4/ \\
        --test-step4-dir  data/batch_test_v7/step4/ \\
        --processed-dir   data/processed_quality \\
        --train-list      data/processed_quality/splits/train.txt \\
        --test-list       data/processed_quality/splits/test.txt \\
        --output          data/batch_test_v7/ablation_fusion_method.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from step5_fusion.data_collector import (  # noqa: E402
    load_sample_json, per_residue_to_int_dict, read_jsonl_record,
)
from step5_fusion.metrics import per_sample_corr  # noqa: E402,F401
from step5_fusion.enriched_fusion import (  # noqa: E402
    EnrichedFusion, TOOL_ORDER, _pearson, _spearman,
    build_enriched_features,
)


# ---- optional LightGBM --------------------------------------------------

try:
    import lightgbm as lgb  # type: ignore
    LGBM_OK = True
except ImportError:
    lgb = None  # type: ignore
    LGBM_OK = False


# ---- aggregation helpers ------------------------------------------------


def _mean(vs): return round(statistics.fmean(vs), 4) if vs else None
def _median(vs): return round(statistics.median(vs), 4) if vs else None
def _std(vs):
    if len(vs) < 2:
        return 0.0 if vs else None
    return round(statistics.pstdev(vs), 4)


# ---- sample fixture -----------------------------------------------------


@dataclass
class SampleData:
    """Per-sample bundle the ablation operates on.

    Training and evaluation use different residue domains:

    - **Training** ranges over ``range(1, length+1)`` so every position
      gets a feature row — matches ``EnrichedFusion._collect``'s
      convention so the in-script XGBoost training stays
      bit-comparable to the saved bundle.
    - **Evaluation** masks down to ``resolved_residues`` (positions
      with experimental coordinates) — matches
      ``scripts/tables/table04_main_results.py``'s convention so the XGBoost
      row in Table 7 reproduces Table 4's number for the same model.

    Fields
    ------
    - ``X`` / ``y``: enriched feature matrix (n_residues × D) and the
      binary GT vector, both ordered by ``residue_ids``.
    - ``residue_ids``: 1-based residue index in ``X`` row order
      (``range(1, length+1)``).
    - ``eval_mask``: boolean array of length ``n_residues``; ``True`` at
      positions whose residue id is in ``resolved_residues``. Empty
      ``resolved_residues`` → all-True (fallback that mirrors
      ``table04_main_results.py``'s degenerate-case behaviour).
    - ``raw_scores_by_tool``: ``{tool_id: {res_id: score}}`` for every
      *successful* tool record. Used by the no-train methods
      (mean / max / noisy_or); empty dicts are filtered out.
    """
    sid: str
    X: np.ndarray
    y: np.ndarray
    residue_ids: list[int]
    eval_mask: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=bool))
    raw_scores_by_tool: dict[str, dict[int, float]] = field(
        default_factory=dict)

    def __post_init__(self) -> None:
        # Default mask = all positions, so legacy callers / fixtures
        # that don't supply one still work and don't silently drop
        # everything.
        if self.eval_mask.size == 0 and self.residue_ids:
            self.eval_mask = np.ones(len(self.residue_ids), dtype=bool)


def _tool_raw_scores(pred: dict) -> dict[int, float]:
    """Prefer ``per_residue_pae_score`` (Cat A's PAE-derived [0,1]
    interface affinity) → fall back to ``per_residue_confidence``.
    Same priority table04_main_results.py uses, so per-residue R and the
    Mean/Max/Noisy-OR rows here run off the same raw signal."""
    pae = per_residue_to_int_dict(pred.get("per_residue_pae_score"))
    if pae:
        return pae
    return per_residue_to_int_dict(pred.get("per_residue_confidence"))


def collect_sample_data(
    step4_dir: Path,
    processed_dir: Path,
    sample_ids: list[str],
    *,
    feature_set: str = "full",
    use_context: bool = True,
) -> list[SampleData]:
    """Walk a split → list of SampleData (one per usable sample).

    Skip reasons mirror EnrichedFusion._collect: missing step4 JSONL,
    missing sample JSON, zero GT residues, or no successful
    in-TOOL_ORDER prediction. Skipped samples DO NOT enter the returned
    list (so the caller's `len(samples)` is the usable count)."""
    out: list[SampleData] = []
    for sid in sample_ids:
        s4 = read_jsonl_record(step4_dir / f"{sid}.jsonl")
        if s4 is None:
            continue
        sample = load_sample_json(processed_dir, sid)
        if sample is None:
            continue
        prot = sample.get("protein") or {}
        length = prot.get("length") or len(prot.get("sequence") or "")
        gt = set((sample.get("interaction") or {})
                 .get("binding_protein_residues") or [])
        if not length or not gt:
            continue
        predictions = s4.get("predictions") or []
        if not [p for p in predictions
                if p.get("success") and p.get("tool_id") in TOOL_ORDER]:
            continue

        residue_ids = list(range(1, int(length) + 1))
        X = build_enriched_features(s4, residue_ids,
                                    feature_set=feature_set,
                                    use_context=use_context)
        y = np.fromiter(
            (1.0 if r in gt else 0.0 for r in residue_ids),
            dtype=np.float64, count=len(residue_ids))

        # Eval domain = resolved_residues (positions with experimental
        # coordinates). Mirrors table04_main_results.py:165–168 so the
        # XGBoost row here equals Table 4's number on the same model.
        # Empty resolved → fall back to "all positions" (the same
        # degenerate-case fallback table04_main_results.py uses).
        resolved = prot.get("resolved_residues") or []
        if resolved:
            resolved_set = {int(r) for r in resolved}
            eval_mask = np.fromiter(
                (r in resolved_set for r in residue_ids),
                dtype=bool, count=len(residue_ids))
        else:
            eval_mask = np.ones(len(residue_ids), dtype=bool)

        raw: dict[str, dict[int, float]] = {}
        for p in predictions:
            if not p.get("success"):
                continue
            tid = p.get("tool_id")
            if tid not in TOOL_ORDER:
                continue
            scores = _tool_raw_scores(p)
            if scores:
                raw[tid] = scores

        out.append(SampleData(
            sid=sid, X=X, y=y,
            residue_ids=residue_ids,
            eval_mask=eval_mask,
            raw_scores_by_tool=raw))
    return out


# ---- per-sample evaluation ---------------------------------------------




def corr_on_eval_subset(pred: np.ndarray,
                        sample: SampleData) -> Optional[dict]:
    """``per_sample_corr`` applied to ``sample.eval_mask``'d
    predictions vs the same-masked GT. The full-length ``pred`` is what
    every method emits (mean/max/noisy-OR over ``residue_ids``,
    sklearn/XGBoost ``predict`` over ``X``); the mask projects it onto
    the resolved-residue subset that the paper's Table 4 evaluator
    uses. This is what makes the XGBoost row here equal Table 4."""
    mask = sample.eval_mask
    if mask.size != pred.size:
        # Defensive: a method built a pred vector of a different length
        # than the feature/y matrix. Can't safely correlate.
        return None
    return per_sample_corr(pred[mask], sample.y[mask])


# ---- no-train methods --------------------------------------------------


def _stacked_raw(sample: SampleData) -> np.ndarray:
    """Stack every successful tool's raw-score vector into an
    ``(n_tools_present, n_residues)`` array. Tools that didn't score a
    residue contribute 0 there. Returns empty array if no tool scored
    anything for the sample (caller treats as skip)."""
    if not sample.raw_scores_by_tool:
        return np.zeros((0, len(sample.residue_ids)))
    rows = []
    for _tid, scores in sample.raw_scores_by_tool.items():
        rows.append(np.fromiter(
            (float(scores.get(r, 0.0)) for r in sample.residue_ids),
            dtype=np.float64, count=len(sample.residue_ids)))
    return np.vstack(rows)


def predict_mean(sample: SampleData) -> np.ndarray:
    """Per-residue mean of every successful tool's raw score. Missing
    tools (no score for a residue) effectively contribute 0 to the
    sum — uniform-denominator mean over the *present* tool set, which
    matches the user's "missing tools filled with 0" spec for this baseline."""
    stack = _stacked_raw(sample)
    if stack.size == 0:
        return np.zeros(len(sample.residue_ids))
    return stack.mean(axis=0)


def predict_max(sample: SampleData) -> np.ndarray:
    """Per-residue max of every successful tool's raw score."""
    stack = _stacked_raw(sample)
    if stack.size == 0:
        return np.zeros(len(sample.residue_ids))
    return stack.max(axis=0)


def predict_noisy_or(sample: SampleData) -> np.ndarray:
    """``1 − ∏_k (1 − c · s_k(i))`` with uniform ``c = 1/K`` over the
    set of tools that produced a score on this sample.

    Raw scores are clamped to [0, 1] before entering the product:
    ``per_residue_pae_score`` is already in [0, 1] for Cat A; Cat B/C
    confidences are too. The clamp catches any out-of-range pLDDT that
    slipped through the pae→confidence fallback."""
    if not sample.raw_scores_by_tool:
        return np.zeros(len(sample.residue_ids))
    k = len(sample.raw_scores_by_tool)
    c = 1.0 / k
    one_minus = np.ones(len(sample.residue_ids), dtype=np.float64)
    for scores in sample.raw_scores_by_tool.values():
        s = np.fromiter(
            (float(scores.get(r, 0.0)) for r in sample.residue_ids),
            dtype=np.float64, count=len(sample.residue_ids))
        np.clip(s, 0.0, 1.0, out=s)
        one_minus *= (1.0 - c * s)
    return 1.0 - one_minus


NO_TRAIN_METHODS: dict[str, Callable[[SampleData], np.ndarray]] = {
    "mean_raw": predict_mean,
    "max_raw": predict_max,
    "noisy_or": predict_noisy_or,
}


# ---- sklearn-style trainable wrapper ------------------------------------


class TrainablePredictor:
    """Thin shim around any sklearn-style estimator.

    Lets us treat ``LogisticRegression`` (``predict_proba``),
    ``RandomForestRegressor`` (``predict``), MLP, LightGBM and XGBoost
    behind one ``train(X, y)`` / ``predict(X)`` interface so the driver
    loop stays one block of code."""

    def __init__(self, name: str, model, *,
                 use_proba: bool = False) -> None:
        self.name = name
        self.model = model
        self.use_proba = use_proba

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        # Classifier APIs in sklearn want integer labels; regressors
        # take any float. Cast on the way in so both shapes work.
        if self.use_proba:
            self.model.fit(X, y.astype(int))
        else:
            self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.use_proba:
            proba = self.model.predict_proba(X)
            # In the degenerate case where the training set was
            # single-class (one fold containing only negatives), sklearn
            # returns a (n,1) array — handle gracefully.
            if proba.shape[1] == 1:
                return proba[:, 0]
            return proba[:, 1]
        return self.model.predict(X)


def _make_logistic_regression() -> TrainablePredictor:
    from sklearn.linear_model import LogisticRegression
    # ``saga`` handles large feature counts and L2 with class weights
    # cleanly; max_iter bumped because 79-D fits converge slowly with
    # severe class imbalance.
    return TrainablePredictor(
        "logistic_regression",
        LogisticRegression(
            class_weight="balanced", solver="lbfgs",
            max_iter=1000, n_jobs=None),
        use_proba=True)


def _make_random_forest() -> TrainablePredictor:
    from sklearn.ensemble import RandomForestRegressor
    return TrainablePredictor(
        "random_forest",
        RandomForestRegressor(
            n_estimators=500, max_depth=6, random_state=42,
            n_jobs=-1))


def _make_mlp() -> TrainablePredictor:
    from sklearn.neural_network import MLPRegressor
    return TrainablePredictor(
        "mlp_3x256",
        MLPRegressor(
            hidden_layer_sizes=(256, 256, 256),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=100,
            random_state=42))


def _make_lightgbm() -> Optional[TrainablePredictor]:
    if not LGBM_OK:
        return None
    return TrainablePredictor(
        "lightgbm",
        lgb.LGBMRegressor(  # type: ignore[union-attr]
            n_estimators=100, max_depth=4, learning_rate=0.1,
            min_child_weight=5, subsample=0.8,
            colsample_bytree=0.8, random_state=42, verbose=-1))


# Exposed so tests can override (e.g. swap MLP to a 1-iter dummy).
TRAINABLE_FACTORIES: dict[str, Callable[[], Optional[TrainablePredictor]]] = {
    "logistic_regression": _make_logistic_regression,
    "random_forest": _make_random_forest,
    "mlp_3x256": _make_mlp,
    "lightgbm": _make_lightgbm,
}


# ---- driver -------------------------------------------------------------


def _stack_xy(samples: list[SampleData]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate per-sample (X, y) into a single big (X, y) for
    sklearn-style training."""
    if not samples:
        return np.zeros((0, 0)), np.zeros((0,))
    X = np.vstack([s.X for s in samples])
    y = np.concatenate([s.y for s in samples])
    return X, y


def evaluate_no_train(
    method_name: str,
    predict_fn: Callable[[SampleData], np.ndarray],
    test_samples: list[SampleData],
) -> list[dict]:
    """Apply a no-train method to every test sample, collect per-sample
    correlations on the resolved-residue subset. Samples with undefined
    correlation drop out."""
    out: list[dict] = []
    for s in test_samples:
        pred = predict_fn(s)
        corr = corr_on_eval_subset(pred, s)
        if corr is None:
            continue
        out.append({"sample_id": s.sid, "method": method_name, **corr})
    return out


def evaluate_trainable(
    predictor: TrainablePredictor,
    train_samples: list[SampleData],
    test_samples: list[SampleData],
) -> list[dict]:
    """Train on concat-of-train, predict per-test-sample, correlate on
    the resolved-residue subset only (matches table04_main_results.py)."""
    X_train, y_train = _stack_xy(train_samples)
    if X_train.size == 0:
        return []
    predictor.train(X_train, y_train)
    out: list[dict] = []
    for s in test_samples:
        pred = np.asarray(predictor.predict(s.X))
        corr = corr_on_eval_subset(pred, s)
        if corr is None:
            continue
        out.append({"sample_id": s.sid, "method": predictor.name, **corr})
    return out


def evaluate_xgboost(
    train_samples: list[SampleData],
    test_samples: list[SampleData],
    enriched_model: Optional[EnrichedFusion],
) -> list[dict]:
    """If ``enriched_model`` is given, use its predictions on the test
    set. Otherwise train a fresh EnrichedFusion (xgboost) on the
    concatenated train samples and predict on test.

    Either path runs through ``EnrichedFusion.predict_sample`` so the
    XGBoost row is byte-identical to what the rest of the eval suite
    sees from ``data/enriched_v7_model/``."""
    model = enriched_model
    if model is None:
        # The class's _collect path re-reads JSONLs from disk, but we
        # already have features in memory — bypass _collect by fitting
        # directly on the stacked matrix.
        model = EnrichedFusion({"model": "xgboost"})
        X_train, y_train = _stack_xy(train_samples)
        if X_train.size == 0:
            return []
        _fit_xgboost_in_memory(model, X_train, y_train)
    out: list[dict] = []
    for s in test_samples:
        # EnrichedFusion.predict_sample reconstructs features from
        # tool_predictions; we already have S.X so call the model
        # directly on it for an apples-to-apples comparison with the
        # other trainable methods (skips the redundant rebuild).
        probs = _xgboost_predict_on_matrix(model, s.X)
        corr = corr_on_eval_subset(probs, s)
        if corr is None:
            continue
        out.append({"sample_id": s.sid, "method": "xgboost", **corr})
    return out


def _fit_xgboost_in_memory(model: EnrichedFusion,
                           X: np.ndarray, y: np.ndarray) -> None:
    """Mirror EnrichedFusion.train's balancing trick without going
    through its disk-reading _collect path."""
    import xgboost as xgb  # noqa: F401 — imported for side-effects
    pos = int((y > 0.5).sum())
    neg = int(y.size - pos)
    spw = float(neg / max(pos, 1))
    model.model = __import__("xgboost").XGBClassifier(
        **model.xgb_params, scale_pos_weight=spw)
    model.model.fit(X, y, verbose=False)


def _xgboost_predict_on_matrix(model: EnrichedFusion,
                               X: np.ndarray) -> np.ndarray:
    if model.model is None:
        raise RuntimeError("XGBoost model not trained / loaded")
    return model.model.predict_proba(X)[:, 1]


# ---- Category-E naive ensembles (Table 4) -------------------------------
#
# Both methods read ``raw_scores_by_tool`` (per_residue_pae_score →
# per_residue_confidence fallback) — the identical raw signal the
# Mean/Max/Noisy-OR rows use — so their tool set is whatever actually
# emitted a score (``present_tools``), which is why the paper counts 5
# even though TOOL_ORDER lists 6.


def _tool_raw_vector(sample: SampleData, tool_id: str) -> np.ndarray:
    """Raw per-residue score vector for one tool over ``residue_ids``
    (0 where the tool didn't score that residue / isn't present)."""
    scores = sample.raw_scores_by_tool.get(tool_id, {})
    return np.fromiter(
        (float(scores.get(r, 0.0)) for r in sample.residue_ids),
        dtype=np.float64, count=len(sample.residue_ids))


def present_tools(samples: list[SampleData]) -> list[str]:
    """Tools (in TOOL_ORDER column order) that emitted a raw score in at
    least one of ``samples`` — Cat E's effective tool set. Returned in
    TOOL_ORDER order so the stacked-LR feature columns are stable."""
    seen: set[str] = set()
    for s in samples:
        seen.update(s.raw_scores_by_tool.keys())
    return [t for t in TOOL_ORDER if t in seen]


def compute_tool_weights(
    train_samples: list[SampleData], tools: list[str],
) -> dict[str, float]:
    """w_k = mean per-sample TRAIN Pearson R for each tool, evaluated on
    the resolved-residue subset (``corr_on_eval_subset``). Same machinery
    as table04_main_results.py, so these equal the per-tool R values Table 4
    reports for Cat A/B/C. Tools with no defined train correlation → 0."""
    weights: dict[str, float] = {}
    for t in tools:
        prs: list[float] = []
        for s in train_samples:
            corr = corr_on_eval_subset(_tool_raw_vector(s, t), s)
            if corr is not None and corr["pearson_r"] is not None:
                prs.append(corr["pearson_r"])
        weights[t] = statistics.fmean(prs) if prs else 0.0
    return weights


def predict_weighted_mean(
    sample: SampleData, weights: dict[str, float],
) -> np.ndarray:
    """``Σ_k w_k·s_k(i) / Σ_k w_k`` over the tools PRESENT at this sample
    (mirrors mean_raw's present-tool denominator), with
    ``w_k = max(train Pearson R, 0)``. Clamping negatives keeps the
    combination convex so an anti-correlated tool can't push the fused
    score below zero. All-zero / no-positive-weight tools → zeros."""
    n = len(sample.residue_ids)
    num = np.zeros(n, dtype=np.float64)
    den = 0.0
    for tid in sample.raw_scores_by_tool:
        w = max(weights.get(tid, 0.0), 0.0)
        if w <= 0.0:
            continue
        num += w * _tool_raw_vector(sample, tid)
        den += w
    return num / den if den > 0.0 else num


def evaluate_weighted_mean(
    weights: dict[str, float], test_samples: list[SampleData],
) -> list[dict]:
    """Apply weighted_mean to every test sample, correlate on the
    resolved-residue subset (undefined-corr samples drop out)."""
    out: list[dict] = []
    for s in test_samples:
        corr = corr_on_eval_subset(predict_weighted_mean(s, weights), s)
        if corr is None:
            continue
        out.append({"sample_id": s.sid, "method": "weighted_mean", **corr})
    return out


def _stack_tool_features(
    sample: SampleData, tools: list[str],
) -> np.ndarray:
    """``(n_residues × n_tools)`` raw-score matrix in fixed ``tools``
    column order — the (≈5-D) feature space for stacked LR."""
    if not tools:
        return np.zeros((len(sample.residue_ids), 0))
    return np.column_stack(
        [_tool_raw_vector(sample, t) for t in tools])


def evaluate_stacked_lr(
    train_samples: list[SampleData],
    test_samples: list[SampleData],
    tools: list[str],
) -> list[dict]:
    """Train sklearn LogisticRegression on the raw tool-score features
    (NOT the 79-D enriched features), predict per test sample, correlate
    on the resolved-residue subset. ``class_weight='balanced'`` matches
    the Table 7 LR so the feature space is the only difference."""
    from sklearn.linear_model import LogisticRegression
    if not tools or not train_samples:
        return []
    X_train = np.vstack(
        [_stack_tool_features(s, tools) for s in train_samples])
    y_train = np.concatenate([s.y for s in train_samples]).astype(int)
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs", max_iter=1000)
    clf.fit(X_train, y_train)
    out: list[dict] = []
    for s in test_samples:
        proba = clf.predict_proba(_stack_tool_features(s, tools))
        # Single-class training fold → (n, 1); fall back to col 0.
        pred = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        corr = corr_on_eval_subset(pred, s)
        if corr is None:
            continue
        out.append({"sample_id": s.sid, "method": "stacked_lr", **corr})
    return out


# ---- aggregation --------------------------------------------------------


METHOD_ORDER = [
    "mean_raw", "max_raw", "noisy_or",
    "logistic_regression", "random_forest", "mlp_3x256",
    "lightgbm", "xgboost",
    # Category-E naive ensembles (Table 4).
    "weighted_mean", "stacked_lr",
]


def aggregate(bucket: dict[str, list[dict]]) -> list[dict]:
    """One row per method, in METHOD_ORDER. Methods with zero usable
    rows are still emitted (with None means) so the CSV row count is
    predictable across runs."""
    rows: list[dict] = []
    for m in METHOD_ORDER:
        rs = bucket.get(m, [])
        prs = [r["pearson_r"] for r in rs if r["pearson_r"] is not None]
        srs = [r["spearman_r"] for r in rs if r["spearman_r"] is not None]
        r2s = [r["r_squared"] for r in rs if r["r_squared"] is not None]
        rows.append({
            "method": m,
            "n_samples": len(rs),
            "pearson_r_mean": _mean(prs),
            "pearson_r_std": _std(prs),
            "pearson_r_median": _median(prs),
            "spearman_r_mean": _mean(srs),
            "spearman_r_median": _median(srs),
            "r2_mean": _mean(r2s),
            "r2_median": _median(r2s),
        })
    return rows


_COLUMNS = [
    "method", "n_samples",
    "pearson_r_mean", "pearson_r_std", "pearson_r_median",
    "spearman_r_mean", "spearman_r_median",
    "r2_mean", "r2_median",
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in _COLUMNS})


def _write_per_sample_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["sample_id", "method", "pearson_r", "spearman_r", "r_squared"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def _print_table(rows: list[dict]) -> None:
    hdr = (f"{'method':22s} {'n':>4s} "
           f"{'PearsonR':>9s} {'SpearmanR':>10s} {'R2':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['method']:22s} {r['n_samples']:>4d} "
              f"{str(r['pearson_r_mean']):>9s} "
              f"{str(r['spearman_r_mean']):>10s} "
              f"{str(r['r2_mean']):>8s}")


# ---- top-level orchestration -------------------------------------------


def run_ablation(
    *,
    train_step4_dir: Path,
    test_step4_dir: Path,
    processed_dir: Path,
    train_ids: list[str],
    test_ids: list[str],
    enriched_model: Optional[EnrichedFusion] = None,
    skip_methods: Optional[set[str]] = None,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Returns (bucket: {method → [per-sample corr]}, flat per-sample
    rows)."""
    skip_methods = skip_methods or set()
    train_samples = collect_sample_data(
        train_step4_dir, processed_dir, train_ids)
    test_samples = collect_sample_data(
        test_step4_dir, processed_dir, test_ids)

    bucket: dict[str, list[dict]] = defaultdict(list)
    per_sample: list[dict] = []

    print(f"train usable: {len(train_samples)}, "
          f"test usable: {len(test_samples)}")

    # No-train methods first (test-only, cheapest).
    for name, fn in NO_TRAIN_METHODS.items():
        if name in skip_methods:
            continue
        rows = evaluate_no_train(name, fn, test_samples)
        bucket[name].extend(rows)
        per_sample.extend(rows)
        print(f"  {name:22s} n={len(rows)}")

    # Trainable methods.
    for name, ctor in TRAINABLE_FACTORIES.items():
        if name in skip_methods:
            continue
        predictor = ctor()
        if predictor is None:
            # LightGBM not installed — warn and move on.
            print(f"  WARN: {name} unavailable (e.g. missing dep), skipped",
                  file=sys.stderr)
            continue
        rows = evaluate_trainable(predictor, train_samples, test_samples)
        bucket[name].extend(rows)
        per_sample.extend(rows)
        print(f"  {name:22s} n={len(rows)}")

    if "xgboost" not in skip_methods:
        rows = evaluate_xgboost(
            train_samples, test_samples, enriched_model)
        bucket["xgboost"].extend(rows)
        per_sample.extend(rows)
        print(f"  {'xgboost':22s} n={len(rows)}")

    # ---- Category-E naive ensembles (Table 4) --------------------------
    # Effective tool set derived from the data so "5 tools" stays honest;
    # weighted_mean weights come from the TRAIN split's per-tool R.
    e_tools = present_tools(train_samples) or present_tools(test_samples)

    if "weighted_mean" not in skip_methods:
        weights = compute_tool_weights(train_samples, e_tools)
        norm = sum(max(w, 0.0) for w in weights.values()) or 1.0
        print(f"  Cat-E tool set ({len(e_tools)}): "
              f"{', '.join(e_tools) or '(none)'}")
        print("  weighted_mean weights (raw train R / normalised w):")
        for t in e_tools:
            print(f"    {t:16s} R={weights[t]:+.4f}  "
                  f"w={max(weights[t], 0.0) / norm:.4f}")
        rows = evaluate_weighted_mean(weights, test_samples)
        bucket["weighted_mean"].extend(rows)
        per_sample.extend(rows)
        print(f"  {'weighted_mean':22s} n={len(rows)}")

    if "stacked_lr" not in skip_methods:
        print(f"  stacked_lr feature dim = {len(e_tools)} "
              f"(tools: {', '.join(e_tools) or '(none)'})")
        rows = evaluate_stacked_lr(train_samples, test_samples, e_tools)
        bucket["stacked_lr"].extend(rows)
        per_sample.extend(rows)
        print(f"  {'stacked_lr':22s} n={len(rows)}")

    return bucket, per_sample


def _load_sample_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"sample list not found: {path}")
    return [ln.split()[0] for ln in path.read_text(encoding="utf-8")
            .splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-step4-dir", type=Path, required=True)
    p.add_argument("--test-step4-dir", type=Path, required=True)
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--train-list", type=Path, required=True)
    p.add_argument("--test-list", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--enriched-model-dir", type=Path, default=None,
                   help="optional pre-trained EnrichedFusion bundle; "
                        "skips XGBoost retraining when provided.")
    p.add_argument("--skip", action="append", default=[],
                   choices=METHOD_ORDER,
                   help="skip a specific method (repeatable).")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for d in (args.train_step4_dir, args.test_step4_dir,
              args.processed_dir):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 1
    try:
        train_ids = _load_sample_ids(args.train_list)
        test_ids = _load_sample_ids(args.test_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    enriched_model = None
    if args.enriched_model_dir is not None:
        try:
            enriched_model = EnrichedFusion.load(args.enriched_model_dir)
        except (OSError, ValueError, ImportError,
                json.JSONDecodeError) as e:
            print(f"ERROR: --enriched-model-dir load failed: {e}",
                  file=sys.stderr)
            return 1

    bucket, per_sample = run_ablation(
        train_step4_dir=args.train_step4_dir,
        test_step4_dir=args.test_step4_dir,
        processed_dir=args.processed_dir,
        train_ids=train_ids,
        test_ids=test_ids,
        enriched_model=enriched_model,
        skip_methods=set(args.skip),
    )
    rows = aggregate(bucket)
    _write_csv(args.output, rows)
    ps_path = args.output.with_name(
        args.output.stem + "_per_sample.csv")
    _write_per_sample_csv(ps_path, per_sample)

    print(f"wrote {args.output}  ({len(rows)} methods)")
    print(f"wrote {ps_path}  ({len(per_sample)} sample-method rows)")
    print()
    _print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
