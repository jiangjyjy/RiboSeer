"""Mock tests for the enriched 5-tool cross-tool fusion.

Covers the spec's required cases:
  - build_enriched_features layout (34 / 64 / 79) + column names + RF2NA
  - missing-tool ⇒ all-zero columns (gate / rank / zscore included)
  - compute_rank / compute_zscore correctness
  - cross-term (score & gate) computation
  - end-to-end train → save → load → predict
  - train_enriched_fusion / evaluate_enriched_fusion CLIs
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from step5_fusion.enriched_fusion import (  # noqa: E402
    CONTEXT_FEATURE_NAMES, FEATURE_NAMES, LGBM_OK, N_BASE, N_CONTEXT,
    N_FULL, TOOL_ORDER, XGB_OK, EnrichedFusion, binding_streak,
    build_enriched_features, compute_rank, compute_zscore,
    feature_names, window_mean,
)
from step5_fusion import enriched_fusion as ef  # noqa: E402 — for monkeypatching


# ---- transforms ---------------------------------------------------------


class TestTransforms(unittest.TestCase):
    def test_compute_rank_monotone_normalized(self):
        r = compute_rank({1: 0.1, 2: 0.9, 3: 0.5})
        self.assertEqual(r[1], 0.0)      # lowest score → rank 0
        self.assertEqual(r[2], 1.0)      # highest → rank 1
        self.assertAlmostEqual(r[3], 0.5)
        self.assertEqual(compute_rank({}), {})
        self.assertEqual(compute_rank({7: 3.0}), {7: 0.0})  # single

    def test_compute_zscore(self):
        z = compute_zscore({1: 1.0, 2: 2.0, 3: 3.0})
        self.assertAlmostEqual(sum(z.values()), 0.0, places=9)
        self.assertAlmostEqual(z[3], 1.224744871, places=6)
        # zero variance → all zeros
        self.assertEqual(compute_zscore({1: 5.0, 2: 5.0}), {1: 0.0, 2: 0.0})
        self.assertEqual(compute_zscore({}), {})


class TestWindowHelpers(unittest.TestCase):
    def test_window_mean_interior_and_bounds(self):
        v = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        # full ±2 window at center 3: idx 1..5 → mean(1,2,3,4,5)=3
        self.assertAlmostEqual(window_mean(v, 3, 2), 3.0)
        # left edge clipped: center 0, ±2 → idx 0..2 → mean(0,1,2)=1
        self.assertAlmostEqual(window_mean(v, 0, 2), 1.0)
        # right edge clipped: center 5, ±2 → idx 3..5 → mean(3,4,5)=4
        self.assertAlmostEqual(window_mean(v, 5, 2), 4.0)
        self.assertEqual(window_mean([], 0, 5), 0.0)

    def test_binding_streak(self):
        # contiguous run of votes around center
        vc = [0, 0, 1, 2, 1, 0, 3, 0]
        self.assertEqual(binding_streak(vc, 3), 3)   # idx 2,3,4
        self.assertEqual(binding_streak(vc, 2), 3)
        self.assertEqual(binding_streak(vc, 0), 0)   # centre has 0 votes
        self.assertEqual(binding_streak(vc, 6), 1)   # isolated single
        self.assertEqual(binding_streak([1, 1, 1, 1], 1), 4)  # all run


# ---- feature builder ----------------------------------------------------


def _pred(tool_id, category, *, success=True, binding=None,
          pae=None, conf=None, plddt_mean=None, iptm=None):
    d = {"tool_id": tool_id, "category": category, "success": success}
    if binding is not None:
        d["binding_protein_residues"] = binding
    if pae is not None:
        d["per_residue_pae_score"] = pae
    if conf is not None:
        d["per_residue_confidence"] = conf
    if plddt_mean is not None:
        d["plddt_mean"] = plddt_mean
    if iptm is not None:
        d["iptm_score"] = iptm
    return d


class TestBuildEnrichedFeatures(unittest.TestCase):
    def setUp(self):
        self.residues = [1, 2, 3]
        self.step4 = {"predictions": [
            _pred("boltz2", "A", binding=[2, 3],
                  pae={1: 0.1, 2: 0.8, 3: 0.6},
                  conf={1: 70.0, 2: 90.0, 3: 80.0},
                  plddt_mean=80.0, iptm=0.55),
            _pred("chai1", "A", binding=[3],
                  pae={1: 0.2, 2: 0.3, 3: 0.9},
                  conf={1: 60.0, 2: 65.0, 3: 88.0},
                  plddt_mean=71.0),
            _pred("equipnas", "C", binding=[2],
                  conf={1: 0.2, 2: 0.7, 3: 0.4}),
            _pred("p2rank", "B", binding=[3],
                  conf={1: 0.1, 2: 0.2, 3: 0.6}),
            # fpocket intentionally absent → all-zero block.
        ]}

    def test_shapes_and_names(self):
        # 6 tools: base 34, +cross 64, +context 79.
        self.assertEqual(N_FULL, 64)
        self.assertEqual(N_BASE, 34)
        self.assertEqual(N_CONTEXT, 15)
        self.assertEqual(len(FEATURE_NAMES), 64)
        self.assertEqual(len(CONTEXT_FEATURE_NAMES), 15)
        # default = full + context = 79
        X = build_enriched_features(self.step4, self.residues)
        self.assertEqual(X.shape, (3, 79))
        self.assertEqual(len(feature_names("full", True)), 79)
        # full, no context = 64
        Xf = build_enriched_features(self.step4, self.residues,
                                     use_context=False)
        self.assertEqual(Xf.shape, (3, 64))
        self.assertEqual(len(feature_names("full", False)), 64)
        # base = 34 (context ignored even if requested)
        Xb = build_enriched_features(self.step4, self.residues,
                                     feature_set="base")
        self.assertEqual(Xb.shape, (3, 34))
        self.assertEqual(len(feature_names("base", True)), 34)
        # strict prefix relationship across all three variants.
        np.testing.assert_array_equal(Xb, X[:, :34])
        np.testing.assert_array_equal(Xf, X[:, :64])

    def test_missing_tool_is_all_zero(self):
        names = feature_names("full", True)  # 79
        X = build_enriched_features(self.step4, self.residues)
        idx = {n: i for i, n in enumerate(names)}
        # every column whose name mentions fpocket — base score/gate/
        # rank/zscore, cross-terms AND the two window cols — is zero
        # because fpocket is absent from self.step4.
        for col in names:
            if "fpocket" in col:
                self.assertTrue(np.all(X[:, idx[col]] == 0.0), col)
        self.assertIn("fpocket_score_win5", idx)
        self.assertIn("fpocket_gate_density5", idx)

    def test_gate_and_cross_gate(self):
        X = build_enriched_features(self.step4, self.residues)
        idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
        # residue 3 (row index 2): boltz2 gate=1, chai1 gate=1,
        # equipnas gate=0, p2rank gate=1.
        self.assertEqual(X[2, idx["boltz2_gate"]], 1.0)
        self.assertEqual(X[2, idx["chai1_gate"]], 1.0)
        self.assertEqual(X[2, idx["equipnas_gate"]], 0.0)
        self.assertEqual(X[2, idx["p2rank_gate"]], 1.0)
        # catA_agree = (≥2 of boltz2/chai1/rf2na gates). At residue 3
        # boltz2=1, chai1=1, rf2na=0 → sum 2 → 1.0.
        self.assertEqual(X[2, idx["catA_agree"]], 1.0)
        # residue 2: only boltz2 gate=1 (chai1/rf2na 0) → sum 1 → 0.0.
        self.assertEqual(X[1, idx["catA_agree"]], 0.0)
        # vote_count at residue 3 = 1+1+0+1+0 = 3 (rf2na/fpocket absent).
        self.assertEqual(X[2, idx["vote_count"]], 3.0)
        # gate cross term boltz2*chai1 = 1 at residue 3, 0 at residue 1.
        self.assertEqual(X[2, idx["gate_boltz2_chai1"]], 1.0)
        self.assertEqual(X[0, idx["gate_boltz2_chai1"]], 0.0)
        # n_tools_available = 4 (rf2na + fpocket missing), shared.
        self.assertTrue(np.all(X[:, idx["n_tools_available"]] == 4.0))

    def test_cross_score_is_product_of_main_scores(self):
        X = build_enriched_features(self.step4, self.residues)
        idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
        # residue 2 (row 1): boltz2 main(pae)=0.8, chai1 main=0.3.
        self.assertAlmostEqual(X[1, idx["boltz2_dist_score"]], 0.8)
        self.assertAlmostEqual(X[1, idx["chai1_dist_score"]], 0.3)
        self.assertAlmostEqual(
            X[1, idx["cross_boltz2_chai1"]], 0.8 * 0.3)
        # boltz2 × equipnas at residue 2: 0.8 * 0.7.
        self.assertAlmostEqual(
            X[1, idx["cross_boltz2_equipnas"]], 0.8 * 0.7)

    def test_rank_and_zscore_columns(self):
        X = build_enriched_features(self.step4, self.residues)
        idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
        # equipnas conf {1:.2, 2:.7, 3:.4} → rank: r1=0, r3=.5, r2=1.
        self.assertEqual(X[0, idx["equipnas_conf_rank"]], 0.0)
        self.assertEqual(X[1, idx["equipnas_conf_rank"]], 1.0)
        self.assertAlmostEqual(X[2, idx["equipnas_conf_rank"]], 0.5)
        # zscore column matches compute_zscore.
        z = compute_zscore({1: 0.2, 2: 0.7, 3: 0.4})
        self.assertAlmostEqual(X[1, idx["equipnas_conf_zscore"]], z[2])
        # boltz2 global_plddt = plddt_mean = 80 shared across rows.
        self.assertTrue(np.all(X[:, idx["boltz2_global_plddt"]] == 80.0))
        self.assertTrue(np.all(X[:, idx["boltz2_iptm"]] == 0.55))
        # chai1 global_plddt = its plddt_mean = 71 (no iptm column).
        self.assertTrue(np.all(X[:, idx["chai1_global_plddt"]] == 71.0))

    def test_failed_tool_treated_as_missing(self):
        s4 = {"predictions": [
            _pred("boltz2", "A", success=False, binding=[1]),
            _pred("equipnas", "C", binding=[2],
                  conf={1: 0.2, 2: 0.9}),
        ]}
        X = build_enriched_features(s4, [1, 2])
        idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
        self.assertTrue(np.all(X[:, idx["boltz2_gate"]] == 0.0))
        self.assertTrue(np.all(X[:, idx["boltz2_dist_score"]] == 0.0))
        self.assertTrue(np.all(X[:, idx["n_tools_available"]] == 1.0))

    def test_rf2na_catA_features_and_alias(self):
        # RF2NA via the ``rf2na`` alias (step4 may use either id). It's
        # a Cat A tool → same 6 cols as Chai-1 (no ipTM).
        s4 = {"predictions": [
            _pred("boltz2", "A", binding=[1, 2],
                  pae={1: 0.7, 2: 0.6, 3: 0.1},
                  conf={1: 80, 2: 85, 3: 60}, plddt_mean=75.0),
            _pred("rf2na", "A", binding=[2, 3],
                  pae={1: 0.2, 2: 0.9, 3: 0.8},
                  conf={1: 55, 2: 70, 3: 88}, plddt_mean=66.0),
        ]}
        X = build_enriched_features(s4, [1, 2, 3])
        idx = {n: i for i, n in enumerate(feature_names("full", True))}
        # all six rf2na base cols exist and are populated.
        for c in ("rf2na_dist_score", "rf2na_plddt", "rf2na_gate",
                  "rf2na_plddt_rank", "rf2na_dist_rank",
                  "rf2na_global_plddt"):
            self.assertIn(c, idx, c)
        self.assertNotIn("rf2na_iptm", idx)        # only boltz2 has ipTM
        # alias resolved → rf2na present, n_tools_available = 2.
        self.assertTrue(np.all(X[:, idx["n_tools_available"]] == 2.0))
        # main score = per_residue_pae_score (Cat A): res2 → 0.9.
        self.assertAlmostEqual(X[1, idx["rf2na_dist_score"]], 0.9)
        # plddt = per_residue_confidence: res3 → 88.
        self.assertEqual(X[2, idx["rf2na_plddt"]], 88.0)
        self.assertTrue(np.all(X[:, idx["rf2na_global_plddt"]] == 66.0))
        # gate from binding list [2,3].
        self.assertEqual(X[1, idx["rf2na_gate"]], 1.0)
        self.assertEqual(X[0, idx["rf2na_gate"]], 0.0)
        # catA_agree: res2 boltz2=1 & rf2na=1 (chai1 absent) → sum 2 → 1.
        self.assertEqual(X[1, idx["catA_agree"]], 1.0)
        # res1: boltz2=1, rf2na=0 → sum 1 → 0.
        self.assertEqual(X[0, idx["catA_agree"]], 0.0)
        # cross-term boltz2 × rf2na = product of main scores (res2).
        self.assertAlmostEqual(
            X[1, idx["cross_boltz2_rf2na"]], 0.6 * 0.9)
        self.assertEqual(X[1, idx["gate_boltz2_rf2na"]], 1.0)
        # context cols for rf2na exist.
        self.assertIn("rf2na_dist_score_win5", idx)
        self.assertIn("rf2na_gate_density5", idx)

    def test_missing_rf2na_zeroes_its_columns(self):
        # self.step4 (setUp) has no rf2na → every rf2na col == 0.
        names = feature_names("full", True)
        X = build_enriched_features(self.step4, self.residues)
        idx = {n: i for i, n in enumerate(names)}
        for col in names:
            if col.startswith("rf2na") or "_rf2na" in col:
                self.assertTrue(np.all(X[:, idx[col]] == 0.0), col)


class TestContextFeatures(unittest.TestCase):
    def setUp(self):
        # 6 residues, only equipnas present so the windows are easy to
        # hand-check. equipnas conf = res value/10; binding (gate) on
        # residues 2,3,4 → a contiguous run.
        self.res = [1, 2, 3, 4, 5, 6]
        conf = {1: 0.1, 2: 0.9, 3: 0.8, 4: 0.7, 5: 0.2, 6: 0.1}
        self.step4 = {"predictions": [
            _pred("equipnas", "C", binding=[2, 3, 4], conf=conf)]}
        self.names = feature_names("full", True)
        self.idx = {n: i for i, n in enumerate(self.names)}

    def test_window_mean_column_matches_helper(self):
        X = build_enriched_features(self.step4, self.res)
        vals = [0.1, 0.9, 0.8, 0.7, 0.2, 0.1]  # equipnas conf, seq order
        col = self.idx["equipnas_conf_win5"]
        for k in range(6):
            self.assertAlmostEqual(
                X[k, col], window_mean(vals, k, 5), places=9)

    def test_gate_density_and_vote_window(self):
        X = build_enriched_features(self.step4, self.res)
        gd = self.idx["equipnas_gate_density5"]
        # gate vector = [0,1,1,1,0,0]; ±5 window covers all 6 → mean=0.5
        for k in range(6):
            self.assertAlmostEqual(X[k, gd], 3.0 / 6.0, places=9)
        # vote_count_win5 == equipnas_gate_density5 here (only 1 tool).
        self.assertTrue(np.allclose(
            X[:, self.idx["vote_count_win5"]], X[:, gd]))

    def test_binding_streak_column(self):
        X = build_enriched_features(self.step4, self.res)
        bs = self.idx["binding_streak"]
        # votes = [0,1,1,1,0,0]: res 2,3,4 in the run of length 3;
        # res 1,5,6 have 0 votes → streak 0.
        got = {r: X[i, bs] for i, r in enumerate(self.res)}
        self.assertEqual(got[1], 0.0)
        self.assertEqual(got[2], 3.0)
        self.assertEqual(got[3], 3.0)
        self.assertEqual(got[4], 3.0)
        self.assertEqual(got[5], 0.0)

    def test_max_score_win5(self):
        X = build_enriched_features(self.step4, self.res)
        # only equipnas → max_tool_score == equipnas conf; window max.
        ms = self.idx["max_score_win5"]
        # every ±5 window over 6 residues spans them all → global max .9
        self.assertTrue(np.all(X[:, ms] == 0.9))

    def test_context_unaffected_by_input_order(self):
        # Shuffled residue_ids must give the SAME per-residue context
        # (windows computed in sequence order, scattered back).
        X1 = build_enriched_features(self.step4, [1, 2, 3, 4, 5, 6])
        shuffled = [4, 1, 6, 3, 5, 2]
        X2 = build_enriched_features(self.step4, shuffled)
        col = self.idx["equipnas_conf_win5"]
        for row, rid in enumerate(shuffled):
            self.assertAlmostEqual(
                X2[row, col], X1[rid - 1, col], places=9)

    def test_missing_tool_window_cols_zero(self):
        X = build_enriched_features(self.step4, self.res)
        for col in ("boltz2_dist_score_win5", "boltz2_gate_density5",
                    "fpocket_score_win5", "p2rank_gate_density5"):
            self.assertTrue(np.all(X[:, self.idx[col]] == 0.0), col)


# ---- synthetic end-to-end dataset --------------------------------------


def _make_dataset(tmp: Path, n: int = 8, length: int = 12):
    step4 = tmp / "step4"
    processed = tmp / "processed"
    (step4).mkdir(parents=True, exist_ok=True)
    (processed / "samples").mkdir(parents=True, exist_ok=True)
    sids = []
    gt = [2, 5, 8, 11]
    seq = "MKRYAVCDEFGHIKLMNPQR"[:length]
    for i in range(n):
        sid = f"s{i:03d}"
        sids.append(sid)
        (processed / "samples" / f"{sid}.json").write_text(json.dumps({
            "sample_id": sid,
            "protein": {"length": length, "sequence": seq},
            "interaction": {"binding_protein_residues": gt},
        }), encoding="utf-8")
        pae = {r: round((0.85 if r in gt else 0.1)
                        + (i * 0.01) % 0.05, 4)
               for r in range(1, length + 1)}
        conf_a = {r: round((88.0 if r in gt else 60.0)
                           + (i % 3), 4)
                  for r in range(1, length + 1)}
        eq = {r: round((0.8 if r in gt else 0.15)
                       + (i * 0.005) % 0.03, 4)
              for r in range(1, length + 1)}
        p2 = {r: round((0.6 if (r in gt and r % 2 == 0) else 0.12)
                       + (i * 0.01) % 0.04, 4)
              for r in range(1, length + 1)}
        fp = {r: round((0.5 if r in gt else 0.1), 4)
              for r in range(1, length + 1)}
        rec = {"sample_id": sid, "predictions": [
            _pred("boltz2", "A", binding=gt, pae=pae, conf=conf_a,
                  plddt_mean=80.0, iptm=0.5),
            _pred("chai1", "A", binding=gt, pae=pae, conf=conf_a,
                  plddt_mean=75.0),
            # RF2NA via the alias id (step4 may emit either).
            _pred("rf2na", "A", binding=gt, pae=pae, conf=conf_a,
                  plddt_mean=70.0),
            _pred("equipnas", "C", binding=gt, conf=eq),
            _pred("p2rank", "B",
                  binding=[r for r in gt if r % 2 == 0], conf=p2),
            _pred("fpocket", "B", binding=gt, conf=fp),
        ]}
        (step4 / f"{sid}.jsonl").write_text(
            json.dumps(rec) + "\n", encoding="utf-8")
    return step4, processed, sids


def _cfg(extra=None):
    c = {"model": "xgboost", "feature_set": "full",
         "n_estimators": 30, "max_depth": 3, "learning_rate": 0.3,
         "min_child_weight": 1, "seed": 0}
    if extra:
        c.update(extra)
    return c


@unittest.skipUnless(XGB_OK, "xgboost not installed")
class TestEnrichedFusionXGB(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4, self.processed, self.sids = _make_dataset(self.tmp)

    def test_train_separates_classes(self):
        f = EnrichedFusion(_cfg())  # default → full + context = 79
        rep = f.train(step4_dir=self.step4,
                      processed_dir=self.processed, verbose=False)
        self.assertEqual(rep["n_features"], 79)
        self.assertTrue(rep["use_context"])
        self.assertGreater(rep["pearson_r"], 0.3)
        self.assertEqual(len(rep["feature_importances"]), 79)

    def test_no_context_is_64(self):
        f = EnrichedFusion(_cfg({"use_context": False}))
        rep = f.train(step4_dir=self.step4,
                      processed_dir=self.processed, verbose=False)
        self.assertEqual(rep["n_features"], 64)
        self.assertFalse(rep["use_context"])
        self.assertEqual(len(rep["feature_importances"]), 64)

    def test_base_feature_set_is_34(self):
        f = EnrichedFusion(_cfg({"feature_set": "base"}))
        rep = f.train(step4_dir=self.step4,
                      processed_dir=self.processed, verbose=False)
        self.assertEqual(rep["n_features"], 34)
        self.assertEqual(len(rep["feature_importances"]), 34)

    def test_predict_ranks_binding_higher(self):
        f = EnrichedFusion(_cfg())
        f.train(step4_dir=self.step4,
                processed_dir=self.processed, verbose=False)
        s4 = json.loads((self.step4 / f"{self.sids[0]}.jsonl")
                         .read_text(encoding="utf-8"))
        pr = f.predict_sample(s4["predictions"], "", 12)
        gt = {2, 5, 8, 11}
        bind = sum(pr[r] for r in gt) / len(gt)
        non = sum(pr[r] for r in pr if r not in gt) / (len(pr) - len(gt))
        self.assertGreater(bind, non)
        for v in pr.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_save_load_round_trip(self):
        f = EnrichedFusion(_cfg())
        f.train(step4_dir=self.step4,
                processed_dir=self.processed, verbose=False)
        bundle = self.tmp / "bundle"
        f.save(bundle)
        for name in ("model.json", "enriched_meta.json",
                     "training_report.json"):
            self.assertTrue((bundle / name).is_file(), name)
        s4 = json.loads((self.step4 / f"{self.sids[0]}.jsonl")
                         .read_text(encoding="utf-8"))
        before = f.predict_sample(s4["predictions"], "", 12)
        after = EnrichedFusion.load(bundle).predict_sample(
            s4["predictions"], "", 12)
        for r in before:
            self.assertAlmostEqual(before[r], after[r], places=5)

    def test_train_no_data_raises(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        with self.assertRaises(RuntimeError):
            EnrichedFusion(_cfg()).train(
                step4_dir=empty, processed_dir=self.processed,
                verbose=False)


class _FakeLGBMRegressor:
    """Mock LightGBM regressor — kept at module top-level so pickle
    can resolve it during save/load round-trip tests.

    Mimics the sklearn-style surface ``EnrichedFusion`` touches:
    constructor swallows arbitrary kwargs, ``fit(X, y)`` learns the
    'first column dominates' rule we plant in the fixture, and
    ``predict(X)`` returns a 1-D array of MSE-style values.

    NOTE: previously a Classifier with ``predict_proba``; the path was
    switched to ``LGBMRegressor`` after Table 7 vs Table 4 calibration
    showed Pearson R 0.555 (regressor) vs 0.530 (classifier) on the
    same data. Pickling works because the class lives at module scope
    (no nested-class / lambda capture)."""

    def __init__(self, **params):
        self.params = dict(params)
        self.n_features_in_: int = 0
        self.feature_importances_: np.ndarray = np.empty(0)
        self._fit_done = False

    def fit(self, X, y, **_fit_kwargs):
        X = np.asarray(X)
        y = np.asarray(y)
        self.n_features_in_ = int(X.shape[1])
        # Importance: variance of each column, so the fixture's
        # binding-correlated columns dominate.
        self.feature_importances_ = X.var(axis=0).astype(np.float64)
        self._fit_done = True
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        # MSE-style raw output: mean of the first 2 columns, no
        # sigmoid squash. Unlike a classifier this CAN go outside
        # [0, 1] in general — but for our fixture (binary-ish
        # features) it stays close to the target range.
        return X[:, :min(2, X.shape[1])].mean(axis=1)


class _FakeLgbModule:
    """Stand-in for the ``lgb`` module attribute on ``ef``. Holds just
    the one symbol EnrichedFusion calls — ``LGBMRegressor``."""

    LGBMRegressor = _FakeLGBMRegressor


class TestEnrichedFusionLGBM(unittest.TestCase):
    """LightGBM path with the lgb module mocked out. Tests construction
    + train/predict/save/load + the no-lgbm error surface."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4, self.processed, self.sids = _make_dataset(self.tmp)
        # Force EnrichedFusion to "see" lightgbm installed and route
        # through the fake module — applies to every test in this class.
        self._patches = [
            unittest.mock.patch.object(ef, "LGBM_OK", True),
            unittest.mock.patch.object(ef, "lgb", _FakeLgbModule),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def test_init_accepts_lightgbm_and_builds_params(self):
        f = EnrichedFusion({"model": "lightgbm",
                            "n_estimators": 50, "max_depth": 3,
                            "learning_rate": 0.07})
        self.assertEqual(f.model_type, "lightgbm")
        # Overrides land in lgbm_params; defaults stay otherwise.
        self.assertEqual(f.lgbm_params["n_estimators"], 50)
        self.assertEqual(f.lgbm_params["max_depth"], 3)
        self.assertAlmostEqual(f.lgbm_params["learning_rate"], 0.07)
        # Untouched defaults survive — regressor recipe (objective
        # "regression", metric "rmse"). If this flips back to "binary"
        # we've silently reverted to the classifier path.
        self.assertEqual(f.lgbm_params["objective"], "regression")
        self.assertEqual(f.lgbm_params["metric"], "rmse")

    def test_init_rejects_unknown_model(self):
        with self.assertRaises(ValueError):
            EnrichedFusion({"model": "catboost"})

    def test_train_records_lgbm_params_in_report(self):
        f = EnrichedFusion({"model": "lightgbm",
                            "n_estimators": 30, "max_depth": 3})
        rep = f.train(step4_dir=self.step4,
                      processed_dir=self.processed, verbose=False)
        self.assertEqual(rep["model_type"], "lightgbm")
        # n_features matches the active feature design.
        self.assertEqual(rep["n_features"], 79)
        # lgbm_params block populated, xgb / ridge blocks left None.
        self.assertIsNotNone(rep["lgbm_params"])
        self.assertIsNone(rep["xgb_params"])
        self.assertIsNone(rep["ridge_lambda"])
        # Regressor recipe → scale_pos_weight is N/A.
        self.assertIsNone(rep["scale_pos_weight"])
        # Fake fit produced the right importance shape.
        self.assertEqual(len(rep["feature_importances"]), 79)
        # The mock regressor is what got attached.
        self.assertIsInstance(f.model, _FakeLGBMRegressor)
        self.assertTrue(f.model._fit_done)

    def test_predict_sample_uses_lgbm_predict(self):
        # Drives the regressor path: EnrichedFusion calls
        # model.predict(X), NOT predict_proba. The fake doesn't even
        # define predict_proba, so an accidental switch back to
        # the classifier path would raise AttributeError here.
        f = EnrichedFusion({"model": "lightgbm"})
        f.train(step4_dir=self.step4,
                processed_dir=self.processed, verbose=False)
        s4 = json.loads((self.step4 / f"{self.sids[0]}.jsonl")
                         .read_text(encoding="utf-8"))
        pr = f.predict_sample(s4["predictions"], "", 12)
        self.assertEqual(len(pr), 12)
        # Regressor output is raw MSE-fit, not clipped — just sanity
        # that it's finite per residue.
        for v in pr.values():
            self.assertTrue(np.isfinite(v))

    def test_predict_sample_does_not_call_predict_proba(self):
        # Explicit regression guard for the classifier→regressor
        # switch. Sub-class the fake to record any predict_proba call;
        # the production path must not touch it.
        class _NoProbaRegressor(_FakeLGBMRegressor):
            def __init__(self, **p):
                super().__init__(**p)
                self.proba_calls = 0

            def predict_proba(self, X):  # would be unused
                self.proba_calls += 1
                return super().predict(X)

        fake_module = type("L", (), {"LGBMRegressor": _NoProbaRegressor})
        with unittest.mock.patch.object(ef, "lgb", fake_module):
            f = EnrichedFusion({"model": "lightgbm"})
            f.train(step4_dir=self.step4,
                    processed_dir=self.processed, verbose=False)
            s4 = json.loads((self.step4 / f"{self.sids[0]}.jsonl")
                             .read_text(encoding="utf-8"))
            f.predict_sample(s4["predictions"], "", 12)
            self.assertEqual(f.model.proba_calls, 0)

    def test_save_load_round_trip(self):
        f = EnrichedFusion({"model": "lightgbm"})
        f.train(step4_dir=self.step4,
                processed_dir=self.processed, verbose=False)
        bundle = self.tmp / "lgbm_bundle"
        f.save(bundle)
        # LightGBM saves to model.pkl, NOT model.json (xgb's slot).
        self.assertTrue((bundle / "model.pkl").is_file())
        self.assertFalse((bundle / "model.json").is_file())
        # Meta records the LightGBM block alongside the model_type.
        meta = json.loads(
            (bundle / "enriched_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["model_type"], "lightgbm")
        self.assertIn("lgbm_params", meta)
        self.assertEqual(meta["lgbm_params"]["objective"], "regression")
        # Round-trip predictions match.
        s4 = json.loads((self.step4 / f"{self.sids[0]}.jsonl")
                         .read_text(encoding="utf-8"))
        before = f.predict_sample(s4["predictions"], "", 12)
        after = EnrichedFusion.load(bundle).predict_sample(
            s4["predictions"], "", 12)
        for r in before:
            self.assertAlmostEqual(before[r], after[r], places=6)


class TestEnrichedFusionLGBMMissing(unittest.TestCase):
    """``model='lightgbm'`` on a host without the package: train() and
    load() both raise a clear ImportError. No fixture needed — the
    error fires before training touches the dataset."""

    def setUp(self):
        self._patch = unittest.mock.patch.object(ef, "LGBM_OK", False)
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def test_train_raises_clear_message(self):
        # Build a 1-sample dataset just enough to reach the train()
        # body — the LGBM_OK gate fires before lgb is actually touched.
        tmp = Path(tempfile.mkdtemp())
        step4, processed, _ = _make_dataset(tmp, n=1, length=12)
        f = EnrichedFusion({"model": "lightgbm"})
        with self.assertRaises(ImportError) as cm:
            f.train(step4_dir=step4, processed_dir=processed,
                    verbose=False)
        self.assertIn("lightgbm", str(cm.exception).lower())
        self.assertIn("pip install lightgbm",
                      str(cm.exception).lower())

    def test_load_raises_clear_message(self):
        # Build a bundle while pretending lgbm IS installed, then
        # toggle the flag off and re-load. Pickle file is touched so
        # the load reaches the LGBM_OK gate.
        tmp = Path(tempfile.mkdtemp())
        step4, processed, _ = _make_dataset(tmp, n=2, length=12)
        with unittest.mock.patch.object(ef, "LGBM_OK", True), \
             unittest.mock.patch.object(ef, "lgb", _FakeLgbModule):
            f = EnrichedFusion({"model": "lightgbm"})
            f.train(step4_dir=step4, processed_dir=processed,
                    verbose=False)
            bundle = tmp / "bundle"
            f.save(bundle)
        # Patch back to "missing" (the setUp patch is already active).
        with self.assertRaises(ImportError) as cm:
            EnrichedFusion.load(bundle)
        self.assertIn("lightgbm", str(cm.exception).lower())


class TestEnrichedFusionRidge(unittest.TestCase):
    """Ridge path needs no xgboost — always runs."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4, self.processed, self.sids = _make_dataset(self.tmp)

    def test_ridge_train_predict_save_load(self):
        f = EnrichedFusion({"model": "ridge", "ridge_lambda": 1.0})
        rep = f.train(step4_dir=self.step4,
                      processed_dir=self.processed, verbose=False)
        self.assertEqual(rep["model_type"], "ridge")
        self.assertEqual(rep["n_features"], 79)  # full + context
        bundle = self.tmp / "ridge_bundle"
        f.save(bundle)
        s4 = json.loads((self.step4 / f"{self.sids[0]}.jsonl")
                         .read_text(encoding="utf-8"))
        before = f.predict_sample(s4["predictions"], "", 12)
        after = EnrichedFusion.load(bundle).predict_sample(
            s4["predictions"], "", 12)
        for r in before:
            self.assertAlmostEqual(before[r], after[r], places=6)


# ---- CLIs ---------------------------------------------------------------


import scripts.train_enriched_fusion as train_cli  # noqa: E402
import scripts.evaluate_enriched_fusion as eval_cli  # noqa: E402


@unittest.skipUnless(XGB_OK, "xgboost not installed")
class TestCLIs(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4, self.processed, self.sids = _make_dataset(self.tmp)
        self.cfg = self.tmp / "cfg.yaml"
        self.cfg.write_text(
            "enriched_fusion:\n"
            "  n_estimators: 30\n  max_depth: 3\n"
            "  learning_rate: 0.3\n  min_child_weight: 1\n  seed: 0\n",
            encoding="utf-8")

    def test_train_then_evaluate_four_row_ablation(self):
        model_dir = self.tmp / "model"          # 79: full + context
        noctx_dir = self.tmp / "model_noctx"    # 64: full, no context
        base_dir = self.tmp / "model_base"      # 34: base
        rc = train_cli.main([
            "--step4-dir", str(self.step4),
            "--processed-dir", str(self.processed),
            "--output-dir", str(model_dir),
            "--config", str(self.cfg),
            "--model", "xgboost", "--feature-set", "full", "--quiet"])
        self.assertEqual(rc, 0)
        self.assertTrue((model_dir / "model.json").is_file())
        meta = json.loads((model_dir / "enriched_meta.json")
                          .read_text(encoding="utf-8"))
        self.assertTrue(meta["use_context"])
        self.assertEqual(len(meta["feature_names"]), 79)

        rc = train_cli.main([
            "--step4-dir", str(self.step4),
            "--processed-dir", str(self.processed),
            "--output-dir", str(noctx_dir),
            "--config", str(self.cfg),
            "--feature-set", "full", "--no-context", "--quiet"])
        self.assertEqual(rc, 0)
        meta = json.loads((noctx_dir / "enriched_meta.json")
                          .read_text(encoding="utf-8"))
        self.assertFalse(meta["use_context"])
        self.assertEqual(len(meta["feature_names"]), 64)

        rc = train_cli.main([
            "--step4-dir", str(self.step4),
            "--processed-dir", str(self.processed),
            "--output-dir", str(base_dir),
            "--config", str(self.cfg),
            "--feature-set", "base", "--quiet"])
        self.assertEqual(rc, 0)

        out = self.tmp / "eval"
        rc = eval_cli.main([
            "--step4-dir", str(self.step4),
            "--processed-dir", str(self.processed),
            "--model-dir", str(model_dir),
            "--no-context-model-dir", str(noctx_dir),
            "--base-model-dir", str(base_dir),
            "--output", str(out)])
        self.assertEqual(rc, 0)
        for name in ("per_residue_correlation.csv",
                     "feature_importance.csv",
                     "per_sample_correlation.csv"):
            self.assertTrue((out / name).is_file(), name)
        rows = (out / "per_residue_correlation.csv").read_text(
            encoding="utf-8").splitlines()
        methods = [r.split(",")[0] for r in rows[1:]]
        self.assertEqual(methods, [
            "enriched_fusion", "enriched_fusion_no_context",
            "enriched_fusion_base", "noisy_or_baseline"])
        # feature_importance.csv has the 79-d model's full ranking.
        fi = (out / "feature_importance.csv").read_text(
            encoding="utf-8").splitlines()
        self.assertEqual(len(fi) - 1, 79)

    def test_train_missing_step4_returns_1(self):
        rc = train_cli.main([
            "--step4-dir", str(self.tmp / "nope"),
            "--processed-dir", str(self.processed),
            "--output-dir", str(self.tmp / "x"), "--quiet"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
