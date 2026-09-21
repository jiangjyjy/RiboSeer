"""Mock tests for the learned-fusion bundle (standardizer + Ridge
optimiser + LearnedFusion orchestrator + train / evaluate CLIs).

The fusion model is data-driven, numpy-only, and never touches the
existing noisy-OR fusion — these tests verify each layer in isolation
plus the end-to-end CLI round-trip on a synthetic step4 dir.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from step5_fusion.learned_fusion import (  # noqa: E402
    DEFAULT_SCORE_FIELDS, LearnedFusion,
)
from step5_fusion.standardizer import ScoreStandardizer  # noqa: E402
from step5_fusion.weight_optimizer import (  # noqa: E402
    AA20, WeightOptimizer, _sigmoid_array,
)


# ============================================================
# 1. ScoreStandardizer
# ============================================================


class TestScoreStandardizer(unittest.TestCase):
    def test_fit_records_per_tool_moments(self):
        std = ScoreStandardizer()
        std.fit({
            "boltz2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "p2rank": [0.0, 0.0, 0.0, 1.0, 1.0],
        })
        self.assertIn("boltz2", std.params)
        self.assertAlmostEqual(std.params["boltz2"]["mean"], 0.3, places=6)
        # Population std of [0.1..0.5] = 0.1414...
        self.assertAlmostEqual(
            std.params["boltz2"]["std"], np.std([0.1, 0.2, 0.3, 0.4, 0.5]),
            places=6,
        )
        self.assertEqual(std.params["boltz2"]["n"], 5)

    def test_transform_at_mean_returns_half(self):
        std = ScoreStandardizer()
        std.fit({"x": [0.0, 1.0]})  # mean=0.5
        # σ(0) = 0.5
        self.assertAlmostEqual(std.transform("x", 0.5), 0.5, places=6)

    def test_transform_unknown_tool_clamps_passthrough(self):
        std = ScoreStandardizer()
        std.fit({"x": [0.0, 1.0]})
        # Unknown tool: just clamp into [0, 1].
        self.assertEqual(std.transform("unknown", 0.7), 0.7)
        self.assertEqual(std.transform("unknown", -1.0), 0.0)
        self.assertEqual(std.transform("unknown", 5.0), 1.0)

    def test_transform_handles_constant_distribution(self):
        # All-equal training scores → std clamped to 1e-8 → very large
        # z for any deviation. We just need transform to NOT divide by
        # zero or NaN.
        std = ScoreStandardizer()
        std.fit({"x": [0.5, 0.5, 0.5]})
        v = std.transform("x", 0.5)
        self.assertEqual(v, 0.5)  # σ(0) = 0.5
        v_high = std.transform("x", 1.0)
        self.assertGreater(v_high, 0.999)

    def test_drops_non_finite(self):
        std = ScoreStandardizer()
        std.fit({"x": [0.1, float("nan"), 0.3, float("inf")]})
        # Only 0.1 and 0.3 contribute → mean = 0.2
        self.assertAlmostEqual(std.params["x"]["mean"], 0.2, places=6)
        self.assertEqual(std.params["x"]["n"], 2)

    def test_save_load_round_trip(self):
        tmp = Path(tempfile.mkdtemp())
        std = ScoreStandardizer()
        std.fit({"a": [0.1, 0.2, 0.3], "b": [0.0, 1.0]})
        path = std.save(tmp / "std.json")
        loaded = ScoreStandardizer.load(path)
        self.assertEqual(set(loaded.params), set(std.params))
        for tid in std.params:
            self.assertAlmostEqual(
                loaded.params[tid]["mean"], std.params[tid]["mean"], places=6,
            )
            self.assertAlmostEqual(
                loaded.params[tid]["std"], std.params[tid]["std"], places=6,
            )

    def test_transform_many(self):
        std = ScoreStandardizer()
        std.fit({"x": [0.0, 1.0]})
        out = std.transform_many("x", [0.5, 0.5, 0.5])
        self.assertEqual(len(out), 3)
        for v in out:
            self.assertAlmostEqual(v, 0.5, places=6)


# ============================================================
# 2. WeightOptimizer (Ridge regression)
# ============================================================


class TestWeightOptimizer(unittest.TestCase):
    def test_sigmoid_is_numerically_stable(self):
        z = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
        out = _sigmoid_array(z)
        # No NaN / inf, monotonic, σ(0)=0.5.
        self.assertFalse(np.any(np.isnan(out)))
        self.assertFalse(np.any(np.isinf(out)))
        self.assertTrue(np.all(np.diff(out) >= 0))
        self.assertAlmostEqual(out[2], 0.5, places=6)

    def test_build_features_default_layout(self):
        opt = WeightOptimizer(
            use_residue_type=True, use_gating=False, use_cross_terms=False,
        )
        row = opt.build_features({"a": 0.5, "b": 0.7}, "K")
        # 2 tools + 20 AAs = 22 features.
        self.assertEqual(row.shape, (22,))
        # Tools sorted alphabetically → a then b.
        self.assertEqual(opt.tool_order, ["a", "b"])
        self.assertAlmostEqual(row[0], 0.5)
        self.assertAlmostEqual(row[1], 0.7)
        # K is at index 8 in AA20 (ACDEFGHIKLMNPQRSTVWY: A=0..K=8).
        self.assertEqual(AA20[8], "K")
        self.assertEqual(row[2 + 8], 1.0)
        # All other AA slots are 0.
        aa_block = row[2:22].tolist()
        self.assertEqual(sum(aa_block), 1.0)

    def test_build_features_cross_terms(self):
        opt = WeightOptimizer(
            use_residue_type=False, use_gating=False, use_cross_terms=True,
        )
        row = opt.build_features({"a": 2.0, "b": 3.0, "c": 5.0}, "X")
        # 3 tools + C(3,2)=3 cross terms = 6 features.
        self.assertEqual(row.shape, (6,))
        # Cross terms: a*b=6, a*c=10, b*c=15.
        self.assertAlmostEqual(row[3], 6.0)
        self.assertAlmostEqual(row[4], 10.0)
        self.assertAlmostEqual(row[5], 15.0)

    def test_build_features_unknown_aa_zero_block(self):
        opt = WeightOptimizer(use_residue_type=True, use_cross_terms=False)
        row = opt.build_features({"a": 0.5}, "X")  # X not in AA20
        aa_block = row[1:21].tolist()
        self.assertEqual(sum(aa_block), 0.0)

    def test_build_features_missing_tool_is_zero(self):
        opt = WeightOptimizer(
            use_residue_type=False, use_gating=False, use_cross_terms=False,
        )
        # Pre-pin tool_order so the second call doesn't re-derive.
        opt.tool_order = ["a", "b", "c"]
        row = opt.build_features({"a": 0.5, "c": 0.7}, "K")
        self.assertEqual(row.tolist(), [0.5, 0.0, 0.7])

    def test_fit_recovers_known_ridge_solution_2x2(self):
        """Hand-checked Ridge on a tiny design with λ=1.

        X = [[1, 0], [0, 1], [1, 1]], y = [1, 0, 1].
        With λ=1, bias unregularised, augmented X̃ = X with ones col.
        Closed form via numpy gives weights ≈ [0.6, 0.0667], bias ≈ 0.4.
        Verify the optimiser matches the same numbers (small precision).
        """
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = np.array([1.0, 0.0, 1.0])
        opt = WeightOptimizer(
            use_residue_type=False, use_gating=False, use_cross_terms=False,
            regularization=1.0,
        )
        # Bypass build_features by feeding X directly + tool_order.
        opt.fit(X, y, tool_order=["a", "b"], verbose=False)
        # Independent reference: solve (X̃ᵀ X̃ + λIʹ) w = X̃ᵀ y manually.
        Xb = np.hstack([X, np.ones((3, 1))])
        I = np.eye(3); I[-1, -1] = 0
        ref = np.linalg.solve(Xb.T @ Xb + 1.0 * I, Xb.T @ y)
        np.testing.assert_allclose(opt.weights, ref[:-1], atol=1e-10)
        self.assertAlmostEqual(opt.bias, ref[-1])

    def test_predict_returns_probabilities(self):
        opt = WeightOptimizer(
            use_residue_type=False, use_gating=False, use_cross_terms=False,
            regularization=1.0,
        )
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        opt.fit(X, np.array([1.0, 0.0, 1.0]),
                tool_order=["a", "b"], verbose=False)
        probs = opt.predict(X)
        self.assertEqual(probs.shape, (3,))
        for p in probs:
            self.assertTrue(0.0 <= p <= 1.0)

    def test_lambda_zero_is_plain_ols(self):
        opt = WeightOptimizer(
            use_residue_type=False, use_gating=False, use_cross_terms=False,
            regularization=0.0,
        )
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = np.array([1.0, 0.0, 1.0])
        opt.fit(X, y, tool_order=["a", "b"], verbose=False)
        # With λ=0, OLS exactly fits 3 points with 3 unknowns
        # ([w_a, w_b, bias]) → residual ≈ 0.
        Xb = np.hstack([X, np.ones((3, 1))])
        residual = y - (Xb @ np.append(opt.weights, opt.bias))
        np.testing.assert_allclose(residual, np.zeros(3), atol=1e-10)

    def test_save_load_round_trip(self):
        tmp = Path(tempfile.mkdtemp())
        opt = WeightOptimizer(
            use_residue_type=False, use_gating=False, use_cross_terms=False,
            regularization=0.5,
        )
        opt.fit(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
                np.array([1.0, 0.0, 1.0]),
                tool_order=["a", "b"], verbose=False)
        path = opt.save(tmp / "opt.json")
        loaded = WeightOptimizer.load(path)
        np.testing.assert_array_equal(loaded.weights, opt.weights)
        self.assertAlmostEqual(loaded.bias, opt.bias)
        self.assertEqual(loaded.tool_order, opt.tool_order)
        self.assertEqual(loaded.feature_names, opt.feature_names)

    def test_fit_validates_feature_count(self):
        opt = WeightOptimizer(
            use_residue_type=True, use_gating=False, use_cross_terms=False,
        )
        # tool_order has 2 → expect 2 + 20 = 22 features. Pass 5.
        with self.assertRaises(ValueError):
            opt.fit(
                np.zeros((10, 5)), np.zeros(10),
                tool_order=["a", "b"], verbose=False,
            )

    def test_predict_before_fit_raises(self):
        opt = WeightOptimizer()
        with self.assertRaises(RuntimeError):
            opt.predict(np.zeros((1, 5)))


# ============================================================
# 2b. Gating features (the noisy-OR-style binding-list signal)
# ============================================================


class TestGatingFeatures(unittest.TestCase):
    def test_gating_layout(self):
        opt = WeightOptimizer(
            use_residue_type=False, use_gating=True, use_cross_terms=False,
        )
        opt.tool_order = ["a", "b", "c"]
        row = opt.build_features(
            {"a": 0.5, "b": 0.7, "c": 0.3},
            "K",
            {"a": True, "b": False, "c": True},
        )
        # 3 tools + 3 gates + 1 vote_count = 7 features.
        self.assertEqual(row.shape, (7,))
        # Tool scores first.
        self.assertEqual(list(row[:3]), [0.5, 0.7, 0.3])
        # Then gating flags in tool_order.
        self.assertEqual(list(row[3:6]), [1.0, 0.0, 1.0])
        # vote_count = 2/3 ≈ 0.667
        self.assertAlmostEqual(row[6], 2.0 / 3.0, places=6)

    def test_gating_feature_names(self):
        opt = WeightOptimizer(
            use_residue_type=False, use_gating=True, use_cross_terms=False,
        )
        names = opt._feature_names(["a", "b"])
        self.assertEqual(
            names, ["tool:a", "tool:b", "gate:a", "gate:b", "vote_count"],
        )

    def test_gating_with_residue_type_and_cross_terms(self):
        # Full design: 2 tools + 2 gates + 1 vote + 20 AA + 1 cross = 26.
        opt = WeightOptimizer(
            use_residue_type=True, use_gating=True, use_cross_terms=True,
        )
        opt.tool_order = ["a", "b"]
        row = opt.build_features(
            {"a": 2.0, "b": 3.0},
            "A",
            {"a": True, "b": False},
        )
        self.assertEqual(row.shape, (26,))
        names = opt._feature_names(opt.tool_order)
        # AA letter "A" sits at index 0 in AA20 → first AA slot is 1.
        aa_block_start = 5  # 2 tool + 2 gate + 1 vote
        self.assertEqual(row[aa_block_start], 1.0)  # A
        self.assertEqual(sum(row[aa_block_start:aa_block_start + 20]), 1.0)
        # Cross-term block at the end: a*b = 2*3 = 6.
        self.assertEqual(row[-1], 6.0)
        # Sanity: feature_names length matches.
        self.assertEqual(len(names), 26)

    def test_gating_missing_flags_dict_treats_all_false(self):
        opt = WeightOptimizer(
            use_residue_type=False, use_gating=True, use_cross_terms=False,
        )
        opt.tool_order = ["a", "b"]
        # tool_binding_flags=None ⇒ flags default to False, vote=0.
        row = opt.build_features({"a": 0.5, "b": 0.7}, "K", None)
        self.assertEqual(row.shape, (5,))
        self.assertEqual(list(row[2:4]), [0.0, 0.0])
        self.assertEqual(row[4], 0.0)

    def test_gating_missing_tool_in_flags_dict_is_false(self):
        # Flag dict has only a; b should default to False.
        opt = WeightOptimizer(
            use_residue_type=False, use_gating=True, use_cross_terms=False,
        )
        opt.tool_order = ["a", "b"]
        row = opt.build_features(
            {"a": 0.5, "b": 0.7}, "K", {"a": True},
        )
        self.assertEqual(list(row[2:4]), [1.0, 0.0])
        self.assertAlmostEqual(row[4], 0.5)  # vote = 1/2


# ============================================================
# 2c. Weighted Ridge — math + sample-weight plumbing
# ============================================================


class TestWeightedRidge(unittest.TestCase):
    def test_uniform_weights_match_unweighted(self):
        # Fitting with sample_weights=ones should give the same weights
        # as the no-weights path (modulo float drift).
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
        y = np.array([1.0, 0.0, 1.0, 0.0])
        opt_a = WeightOptimizer(
            use_residue_type=False, use_gating=False, use_cross_terms=False,
            regularization=0.5,
        )
        opt_a.fit(X, y, tool_order=["a", "b"], verbose=False)
        opt_b = WeightOptimizer(
            use_residue_type=False, use_gating=False, use_cross_terms=False,
            regularization=0.5,
        )
        opt_b.fit(
            X, y, sample_weights=np.ones(4),
            tool_order=["a", "b"], verbose=False,
        )
        np.testing.assert_allclose(
            opt_a.weights, opt_b.weights, atol=1e-10,
        )
        self.assertAlmostEqual(opt_a.bias, opt_b.bias, places=10)

    def test_high_pos_weight_pulls_predictions_toward_positives(self):
        # 100 negatives, 5 positives — plain Ridge predicts ≈ pos_rate
        # everywhere. Upweighting the positives should push the
        # prediction at "positive-feature" rows visibly higher.
        rng = np.random.default_rng(0)
        n_neg, n_pos = 100, 5
        # Feature is the label + small noise so the Bayes-optimal
        # boundary exists; with severe imbalance + small λ the
        # unweighted fit still mostly predicts low.
        X_neg = rng.normal(0.0, 0.1, size=(n_neg, 1))
        X_pos = rng.normal(1.0, 0.1, size=(n_pos, 1))
        X = np.vstack([X_neg, X_pos])
        y = np.array([0.0] * n_neg + [1.0] * n_pos)

        cfg = dict(
            use_residue_type=False, use_gating=False, use_cross_terms=False,
            regularization=0.01,
        )
        plain = WeightOptimizer(**cfg)
        plain.fit(X, y, tool_order=["a"], verbose=False)
        weighted = WeightOptimizer(**cfg)
        weighted.fit(
            X, y, sample_weights=np.where(y > 0.5, n_neg / n_pos, 1.0),
            tool_order=["a"], verbose=False,
        )
        # Predict on a positive-style row (feature ≈ 1.0).
        x_pos = np.array([[1.0]])
        plain_p = float(plain.predict(x_pos)[0])
        weighted_p = float(weighted.predict(x_pos)[0])
        self.assertGreater(weighted_p, plain_p,
                           "balanced weighting should pull a "
                           "positive-feature prediction higher")
        # And the weighted fit should put a non-trivial probability on it.
        self.assertGreater(weighted_p, 0.4)

    def test_negative_sample_weights_rejected(self):
        opt = WeightOptimizer(
            use_residue_type=False, use_gating=False, use_cross_terms=False,
        )
        with self.assertRaises(ValueError):
            opt.fit(
                np.zeros((2, 2)), np.zeros(2),
                sample_weights=np.array([1.0, -1.0]),
                tool_order=["a", "b"], verbose=False,
            )

    def test_sample_weights_length_mismatch_rejected(self):
        opt = WeightOptimizer(
            use_residue_type=False, use_gating=False, use_cross_terms=False,
        )
        with self.assertRaises(ValueError):
            opt.fit(
                np.zeros((3, 2)), np.zeros(3),
                sample_weights=np.array([1.0, 2.0]),  # wrong length
                tool_order=["a", "b"], verbose=False,
            )

    def test_report_records_weighted_flag(self):
        opt = WeightOptimizer(
            use_residue_type=False, use_gating=False, use_cross_terms=False,
        )
        report = opt.fit(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([1.0, 0.0]),
            sample_weights=np.array([2.0, 1.0]),
            tool_order=["a", "b"], verbose=False,
        )
        self.assertTrue(report["weighted"])

        opt2 = WeightOptimizer(
            use_residue_type=False, use_gating=False, use_cross_terms=False,
        )
        report2 = opt2.fit(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([1.0, 0.0]),
            tool_order=["a", "b"], verbose=False,
        )
        self.assertFalse(report2["weighted"])


# ============================================================
# 2d. class_weight resolver
# ============================================================


class TestClassWeightResolver(unittest.TestCase):
    def test_none_returns_no_weights(self):
        y = np.array([0, 0, 1, 0, 1])
        for value in (None, "none", "NONE", " None ", ""):
            with self.subTest(value=value):
                w, info = LearnedFusion._resolve_class_weight(value, y)
                self.assertIsNone(w)
                self.assertEqual(info["mode"], "none")

    def test_balanced_uses_neg_over_pos(self):
        y = np.array([0.0] * 95 + [1.0] * 5)  # 5 % positive
        w, info = LearnedFusion._resolve_class_weight("balanced", y)
        self.assertEqual(info["mode"], "balanced")
        self.assertAlmostEqual(info["pos_weight"], 95 / 5)
        # Negatives get 1.0, positives get pos_weight.
        np.testing.assert_array_equal(w[:95], np.ones(95))
        np.testing.assert_allclose(w[95:], np.full(5, 95 / 5))

    def test_numeric_uses_value_directly(self):
        y = np.array([0.0, 0.0, 1.0])
        w, info = LearnedFusion._resolve_class_weight(7.5, y)
        self.assertEqual(info["mode"], "manual")
        self.assertAlmostEqual(info["pos_weight"], 7.5)
        np.testing.assert_allclose(w, [1.0, 1.0, 7.5])

    def test_numeric_string_accepted(self):
        y = np.array([0.0, 1.0])
        w, info = LearnedFusion._resolve_class_weight("3.5", y)
        self.assertEqual(info["mode"], "manual")
        self.assertAlmostEqual(info["pos_weight"], 3.5)

    def test_unknown_string_raises(self):
        y = np.array([0.0, 1.0])
        with self.assertRaises(ValueError):
            LearnedFusion._resolve_class_weight("garbage", y)

    def test_balanced_no_positives_safe(self):
        # max(pos_count, 1) divisor → no ZeroDivisionError even when
        # the batch happened to land in a single class.
        y = np.zeros(10)
        w, info = LearnedFusion._resolve_class_weight("balanced", y)
        self.assertEqual(info["pos_weight"], 10.0)  # 10 negatives / max(0,1)
        np.testing.assert_array_equal(w, np.ones(10))


# ============================================================
# 2e. End-to-end with gating + balanced weighting
# ============================================================


class TestLearnedFusionWithGating(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4, self.processed, self.sids = \
            _make_synthetic_dataset(self.tmp)

    def test_train_includes_gating_features_and_class_weight(self):
        fusion = LearnedFusion({
            "use_residue_type": False,   # keep design small
            "use_gating": True,
            "use_cross_terms": False,
            "regularization": 0.01,
            "class_weight": "balanced",
        })
        report = fusion.train(
            step4_dir=self.step4, processed_dir=self.processed,
            verbose=False,
        )
        # 3 tools + 3 gates + 1 vote = 7 features.
        self.assertEqual(report["n_features"], 7)
        # Class-weight info was recorded.
        cw = report["class_weight"]
        self.assertEqual(cw["mode"], "balanced")
        self.assertGreater(cw["pos_weight"], 1.0)
        self.assertTrue(report["weighted"])
        # Feature names include gate columns.
        self.assertTrue(any("gate:" in n for n in fusion.optimizer.feature_names))
        self.assertIn("vote_count", fusion.optimizer.feature_names)

    def test_gating_produces_positive_binding_gap(self):
        # On clean synthetic data the score channel and the gating
        # channel are highly correlated (binding residues have both
        # high tool scores AND tool flags set), so we can't reliably
        # claim "gating strictly widens the gap" — Ridge spreads
        # weight across redundant features. The robust claim is
        # the practical one: with gating on + balanced class weight,
        # binding residues clearly outscore non-binding residues.
        fusion = LearnedFusion({
            "use_residue_type": False, "use_gating": True,
            "use_cross_terms": False, "regularization": 0.01,
            "class_weight": "balanced",
        })
        fusion.train(step4_dir=self.step4, processed_dir=self.processed,
                     verbose=False)
        sample = json.loads(
            (self.processed / "samples" / f"{self.sids[0]}.json")
            .read_text(encoding="utf-8")
        )
        s4 = json.loads(
            (self.step4 / f"{self.sids[0]}.jsonl")
            .read_text(encoding="utf-8")
        )
        per_res = fusion.predict_sample(
            s4["predictions"], sample["protein"]["sequence"],
            sample["protein"]["length"],
        )
        gt = set(sample["interaction"]["binding_protein_residues"])
        binding_mean = sum(per_res[r] for r in gt) / len(gt)
        nonbinding = [r for r in per_res if r not in gt]
        non_mean = sum(per_res[r] for r in nonbinding) / len(nonbinding)
        self.assertGreater(binding_mean - non_mean, 0.1)

    def test_gating_recovers_signal_when_scores_are_noise(self):
        # Construct a dataset where the per-tool SCORE channel is pure
        # noise but the BINDING-LIST channel still encodes ground
        # truth. A no-gating model has nothing to learn from
        # (Pearson ≈ 0); a gating model recovers the signal entirely
        # via the gate / vote-count columns.
        #
        # 4 samples × 6 residues; GT = {1, 3, 5} every sample;
        # tool scores ∈ U(0.4, 0.6) (no class signal); tool binding
        # lists = GT verbatim for all three tools.
        rng = np.random.default_rng(42)
        n_samples, length = 4, 6
        gt = [1, 3, 5]
        for i in range(n_samples):
            sid = f"n{i:03d}"
            seq = "MKRYAV"[:length]
            _write_sample(self.processed, sid, length=length, sequence=seq,
                          gt=list(gt))
            # Score channel: pure noise around 0.5 — no class info.
            preds = []
            for tool_id in ("boltz2", "equipnas", "p2rank"):
                noisy = {
                    str(r): float(round(rng.uniform(0.4, 0.6), 4))
                    for r in range(1, length + 1)
                }
                preds.append({
                    "tool_id": tool_id, "category": "A",
                    "sample_id": sid, "success": True,
                    "binding_protein_residues": list(gt),  # the only signal
                    "per_residue_pae_score": noisy if tool_id != "equipnas" else None,
                    "per_residue_confidence": None if tool_id != "equipnas" else noisy,
                })
            _write_step4_record(self.step4, sid, preds)

        sample_ids = [f"n{i:03d}" for i in range(n_samples)]

        def _gap(use_gating: bool) -> float:
            f = LearnedFusion({
                "use_residue_type": False,
                "use_gating": use_gating,
                "use_cross_terms": False,
                "regularization": 0.01,
                "class_weight": "balanced",
            })
            f.train(step4_dir=self.step4, processed_dir=self.processed,
                    sample_ids=sample_ids, verbose=False)
            sid = sample_ids[0]
            sample = json.loads(
                (self.processed / "samples" / f"{sid}.json")
                .read_text(encoding="utf-8")
            )
            s4 = json.loads(
                (self.step4 / f"{sid}.jsonl").read_text(encoding="utf-8")
            )
            per_res = f.predict_sample(
                s4["predictions"], sample["protein"]["sequence"],
                sample["protein"]["length"],
            )
            gt_set = set(sample["interaction"]["binding_protein_residues"])
            b = sum(per_res[r] for r in gt_set) / len(gt_set)
            nb_keys = [r for r in per_res if r not in gt_set]
            n = sum(per_res[r] for r in nb_keys) / len(nb_keys)
            return b - n

        # No-gating: only noisy scores → near-zero gap (the score
        # channel is unbiased noise, so binding ≈ non-binding mean).
        gap_no = _gap(use_gating=False)
        # With-gating: vote_count column perfectly aligns with the
        # 0/1 label, so Ridge fits ≈ wᵥ * vote_count + b. After the
        # output sigmoid the maximum achievable gap caps around
        # σ(1) - σ(0) ≈ 0.23 — that's the linear-then-sigmoid ceiling
        # for a binary label / binary feature, not a model defect.
        gap_yes = _gap(use_gating=True)
        # The substantive claim: gating must visibly beat the
        # noise-only design. We check both an absolute gap (binding
        # genuinely ranks above non-binding) and the relative
        # superiority over the no-gating run.
        self.assertGreater(gap_yes, 0.15,
                           "gating should give a clear binding gap "
                           "even when scores are noise")
        self.assertGreater(gap_yes, gap_no + 0.05,
                           "gating must beat the no-gating run by "
                           "more than float drift")
        # Sanity: with no signal, the no-gating run should be near 0.
        self.assertLess(abs(gap_no), 0.1)

    def test_save_load_round_trip_with_gating(self):
        fusion = LearnedFusion({
            "use_residue_type": False, "use_gating": True,
            "use_cross_terms": False, "regularization": 0.01,
            "class_weight": "balanced",
        })
        fusion.train(step4_dir=self.step4, processed_dir=self.processed,
                     verbose=False)
        bundle_dir = self.tmp / "bundle_gating"
        fusion.save(bundle_dir)
        # Bundle config records the new knobs.
        meta = json.loads(
            (bundle_dir / "feature_names.json").read_text(encoding="utf-8")
        )
        self.assertTrue(meta["config"]["use_gating"])
        self.assertEqual(meta["config"]["class_weight"], "balanced")

        # Reload + predict — must match pre-save numbers.
        sample = json.loads(
            (self.processed / "samples" / f"{self.sids[0]}.json")
            .read_text(encoding="utf-8")
        )
        s4 = json.loads(
            (self.step4 / f"{self.sids[0]}.jsonl")
            .read_text(encoding="utf-8")
        )
        before = fusion.predict_sample(
            s4["predictions"], sample["protein"]["sequence"],
            sample["protein"]["length"],
        )
        fusion2 = LearnedFusion.load(bundle_dir)
        # Loaded bundle preserves gating config.
        self.assertTrue(fusion2.optimizer.use_gating)
        after = fusion2.predict_sample(
            s4["predictions"], sample["protein"]["sequence"],
            sample["protein"]["length"],
        )
        for r in before:
            self.assertAlmostEqual(before[r], after[r], places=5)

    def test_old_bundle_loads_with_gating_off(self):
        # Hand-craft a v1-style optimizer.json that has no use_gating.
        # Loading it must default use_gating=False so the saved
        # weights still match the design they were fit with.
        d = self.tmp / "old_bundle"
        d.mkdir()
        (d / "standardizer.json").write_text(json.dumps({
            "version": 1, "method": "z-score-sigmoid", "min_std": 1e-8,
            "params": {},
        }), encoding="utf-8")
        (d / "optimizer.json").write_text(json.dumps({
            "version": 1, "method": "ridge",
            "config": {  # no use_gating key
                "use_residue_type": False,
                "use_cross_terms": False,
                "regularization": 0.01,
            },
            "tool_order": ["a", "b"],
            "feature_names": ["tool:a", "tool:b"],
            "weights": [0.5, 0.5],
            "bias": 0.0,
            "train_metrics": {},
        }), encoding="utf-8")
        (d / "feature_names.json").write_text(json.dumps({
            "tool_order": ["a", "b"],
            "feature_names": ["tool:a", "tool:b"],
            "score_fields": {},
            "config": {  # no use_gating key
                "use_residue_type": False,
                "use_cross_terms": False,
                "regularization": 0.01,
            },
        }), encoding="utf-8")
        f = LearnedFusion.load(d)
        self.assertFalse(f.optimizer.use_gating)


# ============================================================
# 3. LearnedFusion — orchestrator + end-to-end on synthetic data
# ============================================================


def _write_step4_record(step4_dir: Path, sid: str, predictions: list[dict]) -> None:
    step4_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "sample_id": sid,
        "tools_run": [p["tool_id"] for p in predictions],
        "predictions": predictions,
    }
    (step4_dir / f"{sid}.jsonl").write_text(
        json.dumps(rec) + "\n", encoding="utf-8",
    )


def _write_sample(processed: Path, sid: str, *, length: int,
                  sequence: str, gt: list[int]) -> None:
    samples = processed / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    sample = {
        "sample_id": sid,
        "protein": {"length": length, "sequence": sequence},
        "rna": {"length": 5, "sequence": "GCGCG"},
        "interaction": {"binding_protein_residues": gt},
    }
    (samples / f"{sid}.json").write_text(
        json.dumps(sample), encoding="utf-8",
    )


def _make_synthetic_dataset(
    tmp: Path, *, n_samples: int = 6, length: int = 8,
) -> tuple[Path, Path, list[str]]:
    """Build a synthetic train+test split: tools that AGREE with GT."""
    step4 = tmp / "step4"
    processed = tmp / "processed"
    sids = []
    for i in range(n_samples):
        sid = f"s{i:03d}"
        sids.append(sid)
        # GT: residues 1, 4, 7 bind.
        gt = [1, 4, 7]
        seq = "MKRYAVCDEFGHIKLM"[:length]
        _write_sample(processed, sid, length=length, sequence=seq, gt=gt)

        # Tool boltz2: high PAE-score on binding residues + noise.
        pae = {}
        for r in range(1, length + 1):
            base = 0.85 if r in gt else 0.10
            pae[str(r)] = round(base + (i * 0.01) % 0.05, 4)
        # Tool equipnas: confidence aligned with GT.
        conf = {}
        for r in range(1, length + 1):
            base = 0.7 if r in gt else 0.05
            conf[str(r)] = round(base + (i * 0.005) % 0.03, 4)
        # Tool p2rank: noisier — half the binding residues only.
        p2 = {}
        for r in range(1, length + 1):
            base = 0.6 if (r in gt and r % 2 == 1) else 0.15
            p2[str(r)] = round(base + (i * 0.01) % 0.04, 4)
        _write_step4_record(step4, sid, [
            {"tool_id": "boltz2", "category": "A", "sample_id": sid,
             "success": True, "binding_protein_residues": list(gt),
             "per_residue_pae_score": pae},
            {"tool_id": "equipnas", "category": "C", "sample_id": sid,
             "success": True, "binding_protein_residues": list(gt),
             "per_residue_confidence": conf},
            {"tool_id": "p2rank", "category": "B", "sample_id": sid,
             "success": True,
             "binding_protein_residues": [r for r in gt if r % 2 == 1],
             "per_residue_confidence": p2},
        ])
    return step4, processed, sids


class TestLearnedFusionTrainPredict(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4, self.processed, self.sids = \
            _make_synthetic_dataset(self.tmp)

    def test_train_and_predict_recover_signal(self):
        fusion = LearnedFusion({
            "use_residue_type": False,   # keep design small for the test
            "use_gating": False,
            "use_cross_terms": False,
            "regularization": 0.01,
        })
        report = fusion.train(
            step4_dir=self.step4, processed_dir=self.processed,
            verbose=False,
        )
        # Report shape.
        self.assertEqual(report["n_features"], 3)  # boltz2, equipnas, p2rank
        self.assertGreater(report["n_rows"], 0)
        # GT is residues 1/4/7 out of 8 → pos_rate ≈ 0.375.
        self.assertAlmostEqual(report["pos_rate"], 3 / 8, places=2)
        # Tool order pinned alphabetically.
        self.assertEqual(fusion.tool_order, ["boltz2", "equipnas", "p2rank"])
        # Prediction sanity: binding residues should outscore non-binding.
        sample = json.loads(
            (self.processed / "samples" / f"{self.sids[0]}.json")
            .read_text(encoding="utf-8")
        )
        s4 = json.loads(
            (self.step4 / f"{self.sids[0]}.jsonl")
            .read_text(encoding="utf-8")
        )
        per_res = fusion.predict_sample(
            s4["predictions"], sample["protein"]["sequence"],
            sample["protein"]["length"],
        )
        # Mean prob on binding residues > mean on non-binding.
        gt = set(sample["interaction"]["binding_protein_residues"])
        binding_mean = sum(per_res[r] for r in gt) / len(gt)
        nonbinding = [r for r in per_res if r not in gt]
        non_mean = sum(per_res[r] for r in nonbinding) / len(nonbinding)
        self.assertGreater(binding_mean, non_mean)

    def test_save_load_bundle_round_trip(self):
        fusion = LearnedFusion({
            "use_residue_type": False, "use_gating": False,
            "use_cross_terms": False, "regularization": 0.01,
        })
        fusion.train(
            step4_dir=self.step4, processed_dir=self.processed,
            verbose=False,
        )
        bundle_dir = self.tmp / "bundle"
        fusion.save(bundle_dir)
        # All four files present.
        for name in ("standardizer.json", "optimizer.json",
                     "feature_names.json", "training_report.json"):
            self.assertTrue((bundle_dir / name).is_file(), f"missing {name}")
        # Reload and predict — must produce identical output to pre-save.
        s4 = json.loads(
            (self.step4 / f"{self.sids[0]}.jsonl")
            .read_text(encoding="utf-8")
        )
        sample = json.loads(
            (self.processed / "samples" / f"{self.sids[0]}.json")
            .read_text(encoding="utf-8")
        )
        before = fusion.predict_sample(
            s4["predictions"], sample["protein"]["sequence"],
            sample["protein"]["length"],
        )
        fusion2 = LearnedFusion.load(bundle_dir)
        after = fusion2.predict_sample(
            s4["predictions"], sample["protein"]["sequence"],
            sample["protein"]["length"],
        )
        self.assertEqual(set(before.keys()), set(after.keys()))
        for r in before:
            self.assertAlmostEqual(before[r], after[r], places=4)

    def test_score_field_resolution_default(self):
        fusion = LearnedFusion({})
        for tool, field in DEFAULT_SCORE_FIELDS.items():
            self.assertEqual(fusion.score_fields[tool], field)

    def test_score_field_resolution_nested_override(self):
        fusion = LearnedFusion({
            "tools": {
                "boltz2": {"score_field": "per_residue_confidence"},
            },
        })
        self.assertEqual(
            fusion.score_fields["boltz2"], "per_residue_confidence",
        )

    def test_collect_skips_samples_without_gt(self):
        # Add a no-GT sample on top of the synthetic set.
        _write_sample(self.processed, "no_gt",
                      length=5, sequence="MKRYK", gt=[])
        _write_step4_record(self.step4, "no_gt", [{
            "tool_id": "boltz2", "category": "A", "sample_id": "no_gt",
            "success": True, "binding_protein_residues": [],
            "per_residue_pae_score": {"1": 0.5},
        }])
        fusion = LearnedFusion({"use_residue_type": False,
                                "use_gating": False})
        bundle = fusion.collect_training_data(
            step4_dir=self.step4, processed_dir=self.processed,
        )
        # All synthetic samples used; no_gt skipped.
        self.assertEqual(bundle["n_samples_used"], len(self.sids))
        self.assertGreaterEqual(bundle["n_samples_skipped"], 1)


# ============================================================
# 4. Train-fusion CLI round-trip
# ============================================================


import scripts.train_fusion as train_fusion_cli  # noqa: E402


class TestTrainFusionCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4, self.processed, self.sids = \
            _make_synthetic_dataset(self.tmp)

    def test_writes_bundle_files(self):
        out_dir = self.tmp / "model"
        rc = train_fusion_cli.main([
            "--train-step4-dir", str(self.step4),
            "--processed-dir", str(self.processed),
            "--output", str(out_dir),
            "--quiet",
        ])
        self.assertEqual(rc, 0)
        for name in ("standardizer.json", "optimizer.json",
                     "feature_names.json", "training_report.json"):
            self.assertTrue((out_dir / name).is_file(), f"missing {name}")
        # training_report.json carries the pos_rate field for the
        # synthetic dataset.
        report = json.loads(
            (out_dir / "training_report.json").read_text(encoding="utf-8")
        )
        self.assertIn("pos_rate", report)
        self.assertIn("rmse", report)
        self.assertIn("r2", report)

    def test_missing_step4_dir_returns_1(self):
        rc = train_fusion_cli.main([
            "--train-step4-dir", str(self.tmp / "no_such"),
            "--processed-dir", str(self.processed),
            "--output", str(self.tmp / "out"),
            "--quiet",
        ])
        self.assertEqual(rc, 1)

    def test_sample_list_filter(self):
        # Restrict training to first 2 samples only.
        list_path = self.tmp / "subset.txt"
        list_path.write_text("\n".join(self.sids[:2]) + "\n",
                             encoding="utf-8")
        out_dir = self.tmp / "model_sub"
        rc = train_fusion_cli.main([
            "--train-step4-dir", str(self.step4),
            "--processed-dir", str(self.processed),
            "--sample-list", str(list_path),
            "--output", str(out_dir),
            "--quiet",
        ])
        self.assertEqual(rc, 0)
        report = json.loads(
            (out_dir / "training_report.json").read_text(encoding="utf-8")
        )
        # Only 2 samples were considered.
        self.assertEqual(report["n_samples_used"], 2)


# ============================================================
# 5. Evaluate-learned-fusion CLI round-trip
# ============================================================


import scripts.evaluate_learned_fusion as eval_cli  # noqa: E402


class TestEvaluateLearnedFusionCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4, self.processed, self.sids = \
            _make_synthetic_dataset(self.tmp)
        # Train a model first (no separate test split — re-use the
        # train data; we're verifying the CLI plumbing, not generalisation).
        self.bundle = self.tmp / "model"
        train_fusion_cli.main([
            "--train-step4-dir", str(self.step4),
            "--processed-dir", str(self.processed),
            "--output", str(self.bundle),
            "--quiet",
        ])

    def test_emits_predictions_and_metrics(self):
        out = self.tmp / "eval_out"
        rc = eval_cli.main([
            "--test-step4-dir", str(self.step4),
            "--processed-dir", str(self.processed),
            "--fusion-model", str(self.bundle),
            "--output", str(out),
        ])
        self.assertEqual(rc, 0)
        for name in ("learned_fusion_predictions.jsonl",
                     "learned_fusion_metrics.json",
                     "learned_fusion_metrics.csv",
                     "method_comparison.csv"):
            self.assertTrue((out / name).is_file(), f"missing {name}")
        # JSONL: one record per sample.
        lines = (out / "learned_fusion_predictions.jsonl") \
            .read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), len(self.sids))
        for line in lines:
            rec = json.loads(line)
            self.assertIn("sample_id", rec)
            self.assertIn("per_residue_probability", rec)
            self.assertIn("binding_protein_residues", rec)

    def test_baseline_comparison_when_step5_provided(self):
        # Synthesise a step5 dir that emits a noisy-OR-style fused dict.
        step5 = self.tmp / "step5"
        step5.mkdir()
        for sid in self.sids:
            (step5 / f"{sid}.jsonl").write_text(json.dumps({
                "sample_id": sid,
                "binding_protein_residues": [1, 4, 7],
                "per_residue_probability": {"1": 0.9, "4": 0.85, "7": 0.8},
                "threshold": 0.5,
            }) + "\n", encoding="utf-8")
        out = self.tmp / "eval_with_baseline"
        rc = eval_cli.main([
            "--test-step4-dir", str(self.step4),
            "--test-step5-dir", str(step5),
            "--processed-dir", str(self.processed),
            "--fusion-model", str(self.bundle),
            "--output", str(out),
        ])
        self.assertEqual(rc, 0)
        metrics = json.loads(
            (out / "learned_fusion_metrics.json").read_text(encoding="utf-8")
        )
        self.assertIn("learned_fusion", metrics["methods"])
        self.assertIn("noisy_or_baseline", metrics["methods"])
        # method_comparison.csv has both rows.
        with (out / "method_comparison.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        method_names = {r["method"] for r in rows}
        self.assertEqual(
            method_names, {"learned_fusion", "noisy_or_baseline"},
        )


# ============================================================
# 6. Existing fusion modules untouched (defensive)
# ============================================================


class TestNoisyOrUntouched(unittest.TestCase):
    def test_noisy_or_does_not_import_learned_fusion(self):
        from step5_fusion import noisy_or
        src = Path(noisy_or.__file__).read_text(encoding="utf-8")
        self.assertNotIn("learned_fusion", src)
        self.assertNotIn("WeightOptimizer", src)
        self.assertNotIn("ScoreStandardizer", src)

    def test_fusion_does_not_import_learned_fusion(self):
        from step5_fusion import fusion
        src = Path(fusion.__file__).read_text(encoding="utf-8")
        self.assertNotIn("learned_fusion", src)
        self.assertNotIn("WeightOptimizer", src)


if __name__ == "__main__":
    unittest.main()
