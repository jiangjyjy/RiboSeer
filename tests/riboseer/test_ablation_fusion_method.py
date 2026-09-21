"""Mock tests for scripts/riboseer/ablation_fusion_method.py.

The expensive bits (real sklearn / xgboost training, structure I/O)
are kept out of these unit tests. Pure functions are exercised against
fixtures with known answers; the driver is exercised with monkey-
patched factories so each "method" returns a deterministic prediction
vector — that way the test stays fast AND covers the wiring.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import scripts.riboseer.ablation_fusion_method as afm  # noqa: E402


# ---- fixtures ------------------------------------------------------------


def _sample(sid, residue_ids, gt, raw_per_tool, feature_dim=4):
    """Build a SampleData with arbitrary feature dim (tests don't care
    about the 79-D enriched layout; they only need *some* X / y pair)."""
    n = len(residue_ids)
    X = np.zeros((n, feature_dim))
    # Inject a weak signal correlated with GT so trainable methods can
    # learn something on tiny mocks.
    for i, r in enumerate(residue_ids):
        X[i, 0] = 1.0 if r in gt else 0.0
        X[i, 1] = i / max(n - 1, 1)
    y = np.fromiter(
        (1.0 if r in gt else 0.0 for r in residue_ids),
        dtype=np.float64, count=n)
    raw = {tid: dict(scores) for tid, scores in raw_per_tool.items()}
    return afm.SampleData(
        sid=sid, X=X, y=y,
        residue_ids=list(residue_ids),
        raw_scores_by_tool=raw)


# ---- pure prediction helpers --------------------------------------------


class TestNoTrainPredictors(unittest.TestCase):

    def test_mean_raw_averages_present_tools(self):
        s = _sample(
            "s1", [1, 2, 3], gt={2, 3},
            raw_per_tool={
                "boltz2": {1: 0.0, 2: 1.0, 3: 0.5},
                "p2rank": {1: 0.2, 2: 0.8, 3: 0.0},
            })
        pred = afm.predict_mean(s)
        # Both tools cover all 3 residues → element-wise mean.
        np.testing.assert_allclose(pred, [0.1, 0.9, 0.25])

    def test_mean_raw_missing_residue_is_zero(self):
        s = _sample(
            "s1", [1, 2, 3], gt={2},
            raw_per_tool={"boltz2": {2: 0.6}})  # only scores residue 2
        pred = afm.predict_mean(s)
        # 1 tool → mean = the tool itself; residue 1/3 → 0.
        np.testing.assert_allclose(pred, [0.0, 0.6, 0.0])

    def test_max_raw_picks_highest(self):
        s = _sample(
            "s1", [1, 2], gt={2},
            raw_per_tool={
                "boltz2": {1: 0.2, 2: 0.4},
                "p2rank": {1: 0.5, 2: 0.1},
            })
        pred = afm.predict_max(s)
        np.testing.assert_allclose(pred, [0.5, 0.4])

    def test_noisy_or_formula(self):
        # Two tools, uniform c = 1/2:
        #  residue 1: s = (1.0, 0.0) → 1 - (1 - 0.5*1)(1 - 0.5*0) = 0.5
        #  residue 2: s = (1.0, 1.0) → 1 - (0.5)(0.5)             = 0.75
        s = _sample(
            "s1", [1, 2], gt={1, 2},
            raw_per_tool={
                "boltz2": {1: 1.0, 2: 1.0},
                "p2rank": {1: 0.0, 2: 1.0},
            })
        pred = afm.predict_noisy_or(s)
        np.testing.assert_allclose(pred, [0.5, 0.75])

    def test_noisy_or_clamps_out_of_range(self):
        # A buggy adapter giving 50.0 (pLDDT-style) shouldn't blow up
        # the formula — clamp to 1.0 first.
        s = _sample(
            "s1", [1], gt={1},
            raw_per_tool={"boltz2": {1: 50.0}})  # out of [0,1]
        pred = afm.predict_noisy_or(s)
        # 1 tool, c=1, s=1 (clamped) → 1 - (1 - 1*1) = 1
        np.testing.assert_allclose(pred, [1.0])

    def test_no_raw_scores_returns_zeros(self):
        s = _sample("s1", [1, 2, 3], gt={1}, raw_per_tool={})
        for fn in (afm.predict_mean, afm.predict_max,
                   afm.predict_noisy_or):
            np.testing.assert_array_equal(fn(s), np.zeros(3))


# ---- Category-E naive ensembles -----------------------------------------


class TestCatENaiveEnsembles(unittest.TestCase):

    def test_present_tools_in_tool_order(self):
        s1 = _sample("s1", [1, 2], gt={1},
                     raw_per_tool={"p2rank": {1: 0.5}, "boltz2": {1: 0.6}})
        s2 = _sample("s2", [1, 2], gt={1},
                     raw_per_tool={"equipnas": {1: 0.4}})
        # Returned in TOOL_ORDER order (boltz2 before p2rank before
        # equipnas), de-duplicated across samples.
        self.assertEqual(afm.present_tools([s1, s2]),
                         ["boltz2", "equipnas", "p2rank"])

    def test_compute_tool_weights_is_mean_train_pearson(self):
        # One tool, perfectly correlated with GT on both train samples →
        # weight ≈ 1.0.
        train = [
            _sample("t1", [1, 2, 3, 4], gt={2, 4},
                    raw_per_tool={"boltz2": {1: 0.0, 2: 0.9, 3: 0.1,
                                             4: 0.8}}),
            _sample("t2", [1, 2, 3, 4], gt={2, 4},
                    raw_per_tool={"boltz2": {1: 0.1, 2: 0.85, 3: 0.0,
                                             4: 0.95}}),
        ]
        w = afm.compute_tool_weights(train, ["boltz2"])
        self.assertIn("boltz2", w)
        self.assertGreater(w["boltz2"], 0.9)

    def test_compute_tool_weights_zero_when_no_corr(self):
        # Tool never scored → no defined correlation → weight 0.0.
        train = [_sample("t1", [1, 2, 3], gt={2},
                         raw_per_tool={"boltz2": {2: 0.9}})]
        w = afm.compute_tool_weights(train, ["p2rank"])
        self.assertEqual(w["p2rank"], 0.0)

    def test_predict_weighted_mean_formula(self):
        # Two tools present, weights 0.8 / 0.2 → per residue
        #   (0.8*s_b + 0.2*s_p) / (0.8 + 0.2).
        s = _sample("s1", [1, 2], gt={2},
                    raw_per_tool={"boltz2": {1: 0.0, 2: 1.0},
                                  "p2rank": {1: 1.0, 2: 0.0}})
        pred = afm.predict_weighted_mean(
            s, {"boltz2": 0.8, "p2rank": 0.2})
        np.testing.assert_allclose(pred, [0.2, 0.8])

    def test_predict_weighted_mean_clamps_negative_weights(self):
        # Negative-R tool is dropped (weight clamped to 0); only the
        # positive tool contributes.
        s = _sample("s1", [1, 2], gt={2},
                    raw_per_tool={"boltz2": {1: 0.0, 2: 1.0},
                                  "p2rank": {1: 0.9, 2: 0.1}})
        pred = afm.predict_weighted_mean(
            s, {"boltz2": 0.5, "p2rank": -0.3})
        np.testing.assert_allclose(pred, [0.0, 1.0])

    def test_predict_weighted_mean_all_nonpositive_returns_zeros(self):
        s = _sample("s1", [1, 2], gt={2},
                    raw_per_tool={"boltz2": {1: 0.5, 2: 0.5}})
        pred = afm.predict_weighted_mean(s, {"boltz2": -1.0})
        np.testing.assert_array_equal(pred, np.zeros(2))

    def test_stack_tool_features_fixed_column_order(self):
        s = _sample("s1", [1, 2, 3], gt={2},
                    raw_per_tool={"boltz2": {1: 0.1, 2: 0.2, 3: 0.3},
                                  "p2rank": {2: 0.7}})
        X = afm._stack_tool_features(s, ["boltz2", "p2rank"])
        self.assertEqual(X.shape, (3, 2))
        np.testing.assert_allclose(X[:, 0], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(X[:, 1], [0.0, 0.7, 0.0])

    def test_evaluate_stacked_lr_learns_signal(self):
        # boltz2 raw score tracks GT on train → LR on the 1-D feature
        # should produce a positively-correlated prediction on test.
        train = [
            _sample("t1", [1, 2, 3, 4], gt={2, 4},
                    raw_per_tool={"boltz2": {1: 0.0, 2: 0.9, 3: 0.1,
                                             4: 0.85}}),
            _sample("t2", [1, 2, 3, 4], gt={2, 4},
                    raw_per_tool={"boltz2": {1: 0.05, 2: 0.95, 3: 0.0,
                                             4: 0.8}}),
        ]
        test = [
            _sample("e1", [1, 2, 3, 4], gt={2, 4},
                    raw_per_tool={"boltz2": {1: 0.1, 2: 0.88, 3: 0.05,
                                             4: 0.9}}),
        ]
        rows = afm.evaluate_stacked_lr(train, test, ["boltz2"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["method"], "stacked_lr")
        self.assertGreater(rows[0]["pearson_r"], 0.5)

    def test_evaluate_stacked_lr_empty_tools_returns_empty(self):
        train = [_sample("t1", [1, 2], gt={1},
                         raw_per_tool={"boltz2": {1: 0.5}})]
        self.assertEqual(afm.evaluate_stacked_lr(train, train, []), [])


# ---- per_sample_corr ----------------------------------------------------


class TestPerSampleCorr(unittest.TestCase):

    def test_perfect_pred_pearson_one(self):
        pred = np.array([0.1, 0.9, 0.2, 0.8])
        gt = np.array([0.0, 1.0, 0.0, 1.0])
        c = afm.per_sample_corr(pred, gt)
        self.assertGreater(c["pearson_r"], 0.95)

    def test_constant_pred_returns_none(self):
        self.assertIsNone(afm.per_sample_corr(
            np.array([0.5, 0.5, 0.5]),
            np.array([0.0, 1.0, 0.0])))

    def test_constant_gt_returns_none(self):
        self.assertIsNone(afm.per_sample_corr(
            np.array([0.1, 0.2, 0.3]),
            np.array([1.0, 1.0, 1.0])))


# ---- TrainablePredictor wiring ------------------------------------------


class _FakeRegressor:
    """Records fit/predict calls; predict just returns the first column."""

    def __init__(self):
        self.fit_calls = 0
        self.last_X = None
        self.last_y = None

    def fit(self, X, y):
        self.fit_calls += 1
        self.last_X = X
        self.last_y = y

    def predict(self, X):
        return X[:, 0]


class _FakeClassifier:
    def __init__(self):
        self.fit_calls = 0

    def fit(self, X, y):
        self.fit_calls += 1

    def predict_proba(self, X):
        # 2-class proba: probability of "1" = first feature column,
        # probability of "0" = 1 - first column. Stack as (n, 2).
        p = X[:, 0].clip(0, 1)
        return np.column_stack([1 - p, p])


class TestTrainablePredictor(unittest.TestCase):

    def test_regressor_routes_through_predict(self):
        tp = afm.TrainablePredictor("foo", _FakeRegressor(),
                                    use_proba=False)
        X = np.array([[0.1], [0.9]])
        tp.train(X, np.array([0.0, 1.0]))
        self.assertEqual(tp.model.fit_calls, 1)
        np.testing.assert_allclose(tp.predict(X), [0.1, 0.9])

    def test_classifier_routes_through_predict_proba(self):
        tp = afm.TrainablePredictor("foo", _FakeClassifier(),
                                    use_proba=True)
        X = np.array([[0.2], [0.9]])
        tp.train(X, np.array([0.0, 1.0]))
        self.assertEqual(tp.model.fit_calls, 1)
        np.testing.assert_allclose(tp.predict(X), [0.2, 0.9])

    def test_classifier_single_class_proba_shape(self):
        # sklearn returns (n, 1) when training had only one class —
        # our wrapper should fall back to column 0 instead of crashing.
        class OneClass:
            def fit(self, X, y): pass
            def predict_proba(self, X):
                return np.full((X.shape[0], 1), 0.5)
        tp = afm.TrainablePredictor("foo", OneClass(), use_proba=True)
        tp.train(np.zeros((3, 1)), np.zeros(3))
        np.testing.assert_allclose(
            tp.predict(np.zeros((3, 1))), [0.5, 0.5, 0.5])


# ---- evaluate_trainable / evaluate_no_train ---------------------------


class TestEvaluateFunctions(unittest.TestCase):

    def _split(self):
        train = [
            _sample("t1", [1, 2, 3, 4], gt={2, 4},
                    raw_per_tool={"boltz2": {1: 0.1, 2: 0.9, 3: 0.1,
                                             4: 0.8}}),
            _sample("t2", [1, 2, 3, 4], gt={2, 4},
                    raw_per_tool={"boltz2": {1: 0.0, 2: 0.85, 3: 0.05,
                                             4: 0.9}}),
        ]
        test = [
            _sample("te1", [1, 2, 3, 4], gt={2, 4},
                    raw_per_tool={"boltz2": {1: 0.1, 2: 0.8, 3: 0.1,
                                             4: 0.9},
                                  "p2rank": {1: 0.2, 2: 0.7, 3: 0.0,
                                             4: 0.95}}),
        ]
        return train, test

    def test_evaluate_no_train_collects_corrs(self):
        _, test = self._split()
        rows = afm.evaluate_no_train(
            "mean_raw", afm.predict_mean, test)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], "te1")
        self.assertEqual(rows[0]["method"], "mean_raw")
        self.assertGreater(rows[0]["pearson_r"], 0.0)

    def test_evaluate_trainable_runs_fit_then_per_sample(self):
        train, test = self._split()
        fake = _FakeRegressor()
        tp = afm.TrainablePredictor("custom", fake, use_proba=False)
        rows = afm.evaluate_trainable(tp, train, test)
        # FakeRegressor uses first feature → highly correlated with GT.
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["method"], "custom")
        self.assertEqual(fake.fit_calls, 1)
        # Concatenated train should pass through with 4+4 = 8 rows.
        self.assertEqual(fake.last_X.shape[0], 8)

    def test_evaluate_trainable_skips_undefined_corr(self):
        # Build a "test sample" where GT is all zeros → corr undefined,
        # method should drop the row.
        train, _ = self._split()
        flat = _sample("te0", [1, 2, 3], gt=set(),
                       raw_per_tool={"boltz2": {1: 0.0, 2: 0.0,
                                                3: 0.0}})
        # zero-binding sample → y all zeros → constant GT → skip.
        tp = afm.TrainablePredictor("custom", _FakeRegressor(),
                                    use_proba=False)
        rows = afm.evaluate_trainable(tp, train, [flat])
        self.assertEqual(rows, [])


# ---- collect_sample_data ------------------------------------------------


def _write_sample_json(proc: Path, sid: str, *, length, gt, seq=None):
    (proc / "samples").mkdir(parents=True, exist_ok=True)
    seq = seq or "M" * length
    (proc / "samples" / f"{sid}.json").write_text(json.dumps({
        "sample_id": sid,
        "protein": {"sequence": seq, "length": length,
                    "resolved_residues": list(range(1, length + 1))},
        "rna": {"sequence": "ACGU", "length": 4},
        "interaction": {"binding_protein_residues": list(gt)},
    }), encoding="utf-8")


def _write_step4(step4: Path, sid: str, preds):
    step4.mkdir(parents=True, exist_ok=True)
    (step4 / f"{sid}.jsonl").write_text(
        json.dumps({"sample_id": sid, "predictions": preds}) + "\n",
        encoding="utf-8")


def _pred(tool_id, *, success=True, binding=None, pae=None, conf=None):
    d = {"tool_id": tool_id, "category": "A", "success": success}
    if binding is not None:
        d["binding_protein_residues"] = binding
    if pae is not None:
        d["per_residue_pae_score"] = pae
    if conf is not None:
        d["per_residue_confidence"] = conf
    return d


class TestEvalMask(unittest.TestCase):
    """Regression tests for the residue-domain fix.

    Issue: ablation_fusion_method.py originally computed correlations
    over ``range(1, length+1)`` while table04_main_results.py uses the
    ``resolved_residues`` subset. That mismatch produced 0.524 (Table 4)
    vs 0.536 (Table 7) for the same XGBoost model on the same data.
    The fix is an ``eval_mask`` on SampleData; these tests pin the
    new behaviour."""

    def test_corr_on_eval_subset_uses_mask(self):
        # 6 residues: predictions correlate perfectly with GT on the
        # resolved subset {1, 3, 5}, but predictions on residues 2/4/6
        # are anti-correlated with GT. If the mask is honoured we
        # should see Pearson=+1 on the subset; if ignored we'd see
        # ~0 over all 6.
        residue_ids = [1, 2, 3, 4, 5, 6]
        gt =   np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        pred = np.array([0.0, 0.9, 0.5, 0.1, 1.0, 0.8])  # masked off: 2,4,6
        eval_mask = np.array([True, False, True, False, True, False])
        s = afm.SampleData(
            sid="m1", X=np.zeros((6, 1)), y=gt,
            residue_ids=residue_ids, eval_mask=eval_mask)
        c = afm.corr_on_eval_subset(pred, s)
        # On the subset [(0,0), (0.5, 0), (1.0, 1)] vs [0,0,1] → r=+
        self.assertIsNotNone(c)
        self.assertGreater(c["pearson_r"], 0.85)

    def test_corr_on_eval_subset_length_mismatch_returns_none(self):
        s = afm.SampleData(
            sid="m1", X=np.zeros((4, 1)),
            y=np.array([0.0, 1.0, 0.0, 1.0]),
            residue_ids=[1, 2, 3, 4],
            eval_mask=np.array([True, True, True, True]))
        # Pred only covers 3 residues — defensively return None.
        self.assertIsNone(afm.corr_on_eval_subset(
            np.array([0.1, 0.9, 0.5]), s))

    def test_collect_sample_data_builds_eval_mask_from_resolved(self):
        # length=6, resolved_residues=[1,2,5,6] (positions 3,4 missing)
        # → eval_mask = [T, T, F, F, T, T].
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp) / "proc"
            step4 = Path(tmp) / "step4"
            (proc / "samples").mkdir(parents=True)
            (proc / "samples" / "s1.json").write_text(json.dumps({
                "sample_id": "s1",
                "protein": {"sequence": "MMMMMM", "length": 6,
                            "resolved_residues": [1, 2, 5, 6]},
                "rna": {"sequence": "AC", "length": 2},
                "interaction": {"binding_protein_residues": [2, 5]},
            }), encoding="utf-8")
            step4.mkdir()
            (step4 / "s1.jsonl").write_text(json.dumps({
                "sample_id": "s1",
                "predictions": [_pred(
                    "boltz2", binding=[2, 5],
                    pae={"2": 0.9, "5": 0.8})]
            }) + "\n", encoding="utf-8")
            out = afm.collect_sample_data(step4, proc, ["s1"])
        self.assertEqual(len(out), 1)
        np.testing.assert_array_equal(
            out[0].eval_mask,
            np.array([True, True, False, False, True, True]))
        # Sanity: X / y still cover all 6 residues (training domain
        # unchanged).
        self.assertEqual(out[0].X.shape[0], 6)
        self.assertEqual(out[0].y.shape[0], 6)

    def test_collect_sample_data_no_resolved_defaults_to_all_true(self):
        # Empty resolved_residues → fallback all-True (mirrors
        # table04_main_results.py:166–168).
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp) / "proc"
            step4 = Path(tmp) / "step4"
            (proc / "samples").mkdir(parents=True)
            (proc / "samples" / "s1.json").write_text(json.dumps({
                "sample_id": "s1",
                "protein": {"sequence": "MMMM", "length": 4,
                            "resolved_residues": []},
                "rna": {"sequence": "AC", "length": 2},
                "interaction": {"binding_protein_residues": [2]},
            }), encoding="utf-8")
            step4.mkdir()
            (step4 / "s1.jsonl").write_text(json.dumps({
                "sample_id": "s1",
                "predictions": [_pred(
                    "boltz2", binding=[2], pae={"2": 0.9})]
            }) + "\n", encoding="utf-8")
            out = afm.collect_sample_data(step4, proc, ["s1"])
        self.assertEqual(len(out), 1)
        self.assertTrue(bool(out[0].eval_mask.all()))

    def test_evaluate_no_train_respects_mask(self):
        # Same setup as test_corr_on_eval_subset_uses_mask but driven
        # via the public evaluate_no_train path: the masked subset has
        # Mean=Max correlation +1, but the full-length domain would
        # give a much weaker correlation. We assert we see the masked
        # answer (= positive), confirming the wiring.
        residue_ids = [1, 2, 3, 4]
        # GT: residue 2 and 4 binding.
        gt = np.array([0.0, 1.0, 0.0, 1.0])
        # Raw scores: tool gets it right on resolved {1, 2}, wrong
        # everywhere else. Mask {1, 2} → perfect; mask {1..4} → meh.
        raw = {"boltz2": {1: 0.0, 2: 0.9, 3: 0.95, 4: 0.0}}
        eval_mask = np.array([True, True, False, False])
        s = afm.SampleData(
            sid="m1", X=np.zeros((4, 1)), y=gt,
            residue_ids=residue_ids, eval_mask=eval_mask,
            raw_scores_by_tool=raw)
        rows = afm.evaluate_no_train("mean_raw", afm.predict_mean, [s])
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["pearson_r"], 0.99)


class TestCollectSampleData(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4 = self.tmp / "step4"
        self.proc = self.tmp / "proc"

    def test_skips_missing_step4(self):
        _write_sample_json(self.proc, "s1", length=4, gt={1, 2})
        out = afm.collect_sample_data(self.step4, self.proc, ["s1"])
        self.assertEqual(out, [])

    def test_skips_zero_gt(self):
        _write_sample_json(self.proc, "s1", length=4, gt=[])
        _write_step4(self.step4, "s1",
                     [_pred("boltz2", binding=[1], pae={"1": 0.5})])
        self.assertEqual(
            afm.collect_sample_data(self.step4, self.proc, ["s1"]),
            [])

    def test_populates_raw_scores_from_pae(self):
        _write_sample_json(self.proc, "s1", length=4, gt={2, 3})
        _write_step4(self.step4, "s1", [
            _pred("boltz2", binding=[2, 3],
                  pae={"2": 0.9, "3": 0.7},
                  conf={"2": 80.0, "3": 70.0}),
            _pred("p2rank", binding=[2],
                  conf={"2": 0.6}),
            _pred("equipnas", success=False),
        ])
        out = afm.collect_sample_data(self.step4, self.proc, ["s1"])
        self.assertEqual(len(out), 1)
        s = out[0]
        # pae preferred over confidence for boltz2; equipnas failed →
        # absent from raw_scores_by_tool.
        self.assertEqual(s.raw_scores_by_tool["boltz2"],
                         {2: 0.9, 3: 0.7})
        self.assertEqual(s.raw_scores_by_tool["p2rank"], {2: 0.6})
        self.assertNotIn("equipnas", s.raw_scores_by_tool)
        # X/y row count = protein length.
        self.assertEqual(s.X.shape[0], 4)
        self.assertEqual(s.y.tolist(), [0.0, 1.0, 1.0, 0.0])


# ---- aggregation --------------------------------------------------------


class TestAggregateAndCsv(unittest.TestCase):

    def test_aggregate_emits_all_methods_in_order(self):
        bucket = {
            "mean_raw": [{"pearson_r": 0.5, "spearman_r": 0.4,
                          "r_squared": 0.25}] * 3,
            "xgboost": [{"pearson_r": 0.8, "spearman_r": 0.7,
                         "r_squared": 0.64}] * 5,
        }
        rows = afm.aggregate(bucket)
        # Every method appears, in METHOD_ORDER.
        self.assertEqual([r["method"] for r in rows], afm.METHOD_ORDER)
        # Missing methods → n=0, all means None.
        for r in rows:
            if r["method"] in ("mean_raw", "xgboost"):
                self.assertGreater(r["n_samples"], 0)
                self.assertIsNotNone(r["pearson_r_mean"])
            else:
                self.assertEqual(r["n_samples"], 0)
                self.assertIsNone(r["pearson_r_mean"])

    def test_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "agg.csv"
            afm._write_csv(out, [
                {"method": "mean_raw", "n_samples": 107,
                 "pearson_r_mean": 0.42, "pearson_r_std": 0.1,
                 "pearson_r_median": 0.45, "spearman_r_mean": 0.40,
                 "spearman_r_median": 0.41, "r2_mean": 0.18,
                 "r2_median": 0.20},
            ])
            with out.open(encoding="utf-8") as f:
                row = next(iter(csv.DictReader(f)))
        self.assertEqual(row["method"], "mean_raw")
        self.assertEqual(int(row["n_samples"]), 107)


# ---- driver + CLI (with patched factories) -----------------------------


class TestRunAblationAndCli(unittest.TestCase):
    """End-to-end smoke: build a tiny dataset, swap every trainable
    factory for a deterministic fake so tests don't import xgboost /
    sklearn slow paths, then run the full driver + CLI."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.train_step4 = self.tmp / "train_step4"
        self.test_step4 = self.tmp / "test_step4"
        self.proc = self.tmp / "proc"
        # 2 train + 2 test samples, all with usable GT + 1 tool record.
        self.train_ids = []
        for i in range(2):
            sid = f"t{i}"
            self.train_ids.append(sid)
            _write_sample_json(self.proc, sid, length=6, gt={2, 4})
            _write_step4(self.train_step4, sid, [
                _pred("boltz2", binding=[2, 4],
                      pae={"2": 0.9, "4": 0.8})])
        self.test_ids = []
        for i in range(2):
            sid = f"e{i}"
            self.test_ids.append(sid)
            _write_sample_json(self.proc, sid, length=6, gt={2, 4})
            _write_step4(self.test_step4, sid, [
                _pred("boltz2", binding=[2, 4],
                      pae={"2": 0.85, "4": 0.75})])

    def _patch_trainables(self):
        """Swap every trainable factory for a wrapper around the fake
        regressor (predicts feature column 0). Keeps the test fast and
        deterministic."""
        return mock.patch.dict(
            afm.TRAINABLE_FACTORIES,
            {
                "logistic_regression": lambda: afm.TrainablePredictor(
                    "logistic_regression", _FakeRegressor(),
                    use_proba=False),
                "random_forest": lambda: afm.TrainablePredictor(
                    "random_forest", _FakeRegressor(),
                    use_proba=False),
                "mlp_3x256": lambda: afm.TrainablePredictor(
                    "mlp_3x256", _FakeRegressor(),
                    use_proba=False),
                "lightgbm": lambda: afm.TrainablePredictor(
                    "lightgbm", _FakeRegressor(), use_proba=False),
            },
            clear=True,
        )

    def test_run_ablation_smoke(self):
        with self._patch_trainables(), \
             mock.patch.object(afm, "evaluate_xgboost",
                               return_value=[
                                   {"sample_id": "e0", "method": "xgboost",
                                    "pearson_r": 0.9,
                                    "spearman_r": 0.85,
                                    "r_squared": 0.81}]):
            bucket, per_sample = afm.run_ablation(
                train_step4_dir=self.train_step4,
                test_step4_dir=self.test_step4,
                processed_dir=self.proc,
                train_ids=self.train_ids, test_ids=self.test_ids)
        # Each no-train method runs on the 2 test samples; trainable
        # methods + xgboost likewise.
        for m in ("mean_raw", "max_raw", "noisy_or",
                  "logistic_regression", "random_forest",
                  "mlp_3x256", "lightgbm"):
            self.assertGreater(len(bucket[m]), 0, f"{m} produced 0 rows")
        self.assertEqual(len(bucket["xgboost"]), 1)

    def test_skip_method(self):
        with self._patch_trainables(), \
             mock.patch.object(afm, "evaluate_xgboost",
                               return_value=[]):
            bucket, _ = afm.run_ablation(
                train_step4_dir=self.train_step4,
                test_step4_dir=self.test_step4,
                processed_dir=self.proc,
                train_ids=self.train_ids, test_ids=self.test_ids,
                skip_methods={"logistic_regression",
                              "random_forest", "mlp_3x256"})
        # Skipped methods absent from bucket; others still present.
        for m in ("logistic_regression", "random_forest", "mlp_3x256"):
            self.assertEqual(len(bucket.get(m, [])), 0)
        for m in ("mean_raw", "max_raw", "noisy_or", "lightgbm"):
            self.assertGreater(len(bucket[m]), 0)

    def test_cli_writes_csvs(self):
        out = self.tmp / "ablation.csv"
        with self._patch_trainables(), \
             mock.patch.object(afm, "evaluate_xgboost",
                               return_value=[
                                   {"sample_id": "e0", "method": "xgboost",
                                    "pearson_r": 0.9,
                                    "spearman_r": 0.85,
                                    "r_squared": 0.81}]):
            train_list = self.tmp / "train.txt"
            test_list = self.tmp / "test.txt"
            train_list.write_text("\n".join(self.train_ids),
                                  encoding="utf-8")
            test_list.write_text("\n".join(self.test_ids),
                                 encoding="utf-8")
            rc = afm.main([
                "--train-step4-dir", str(self.train_step4),
                "--test-step4-dir", str(self.test_step4),
                "--processed-dir", str(self.proc),
                "--train-list", str(train_list),
                "--test-list", str(test_list),
                "--output", str(out),
            ])
        self.assertEqual(rc, 0)
        self.assertTrue(out.is_file())
        ps = out.with_name("ablation_per_sample.csv")
        self.assertTrue(ps.is_file())
        with out.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # Every method emitted (8 rows total, in METHOD_ORDER).
        self.assertEqual([r["method"] for r in rows], afm.METHOD_ORDER)


# ---- LightGBM optional dep ----------------------------------------------


class TestLightgbmFactory(unittest.TestCase):

    def test_returns_none_when_lgbm_missing(self):
        with mock.patch.object(afm, "LGBM_OK", False):
            self.assertIsNone(afm._make_lightgbm())


if __name__ == "__main__":
    unittest.main()
