"""Mock tests for the Table 27 (feature importance) and Table 28
(failure-mode analysis) scripts.

Pure detectors / classifiers are exercised against hand-built fixtures
with known answers. The Table-27 gain extraction is checked end-to-end
against a tiny *real* XGBoost ``EnrichedFusion`` (xgboost is available on
this box; lightgbm may not be — the gain path is model-type-agnostic so
the xgboost check covers the wiring).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import scripts.tables.table26_feature_importance as t27  # noqa: E402
import scripts.tables.table27_failure_modes as t28  # noqa: E402
from scripts.riboseer.ablation_fusion_method import SampleData  # noqa: E402


# ===========================================================================
# Table 27
# ===========================================================================


class TestFeatureGroup(unittest.TestCase):

    def test_g1_per_tool(self):
        for n in ("boltz2_dist_score", "chai1_plddt", "boltz2_gate",
                  "equipnas_conf_rank", "rf2na_global_plddt", "boltz2_iptm"):
            self.assertEqual(t27.feature_group(n), "G1", n)

    def test_g2_cross_and_summary(self):
        for n in ("cross_boltz2_chai1", "gate_boltz2_p2rank",
                  "vote_count", "catA_agree", "n_tools_available"):
            self.assertEqual(t27.feature_group(n), "G2", n)

    def test_g3_context_beats_g2(self):
        # vote_count_win5 contains 'vote_count' (a G2 token) but is context.
        for n in ("boltz2_dist_score_win5", "boltz2_gate_density5",
                  "binding_streak", "vote_count_win5", "max_score_win5"):
            self.assertEqual(t27.feature_group(n), "G3", n)

    def test_g4_scope(self):
        for n in ("scope_family_RRM", "scope_rna_junction",
                  "scope_difficulty"):
            self.assertEqual(t27.feature_group(n), "G4", n)


class TestRankNormalized(unittest.TestCase):

    def test_top1_is_one_and_sorted(self):
        names = ["a_win5", "boltz2_dist_score", "cross_boltz2_chai1"]
        gains = [10.0, 40.0, 20.0]
        rows = t27.rank_normalized(names, gains, k=3)
        self.assertEqual([r["feature"] for r in rows],
                         ["boltz2_dist_score", "cross_boltz2_chai1",
                          "a_win5"])
        self.assertAlmostEqual(rows[0]["norm_importance"], 1.0)
        self.assertAlmostEqual(rows[1]["norm_importance"], 0.5)
        self.assertAlmostEqual(rows[2]["norm_importance"], 0.25)
        self.assertEqual([r["group"] for r in rows], ["G1", "G2", "G3"])

    def test_k_truncates(self):
        rows = t27.rank_normalized(["a", "b", "c"], [1.0, 2.0, 3.0], k=2)
        self.assertEqual(len(rows), 2)

    def test_all_zero_gains_no_div0(self):
        rows = t27.rank_normalized(["a", "b"], [0.0, 0.0], k=2)
        self.assertEqual(rows[0]["norm_importance"], 0.0)


class TestExtractGainXGB(unittest.TestCase):
    """End-to-end: tiny real XGBoost EnrichedFusion → gain extraction."""

    def _toy_model(self):
        from step5_fusion.enriched_fusion import EnrichedFusion
        from scripts.riboseer.ablation_fusion_method import (
            _fit_xgboost_in_memory)
        model = EnrichedFusion({"model": "xgboost"})
        d = len(model.feature_names)  # 79
        rng = np.random.default_rng(0)
        X = rng.standard_normal((300, d))
        # Make a couple of columns genuinely predictive.
        y = (X[:, 0] + 0.5 * X[:, 5] + rng.standard_normal(300) * 0.1
             > 0).astype(float)
        _fit_xgboost_in_memory(model, X, y)
        return model

    def test_extract_and_rank(self):
        try:
            import xgboost  # noqa: F401
        except ImportError:
            self.skipTest("xgboost not installed")
        model = self._toy_model()
        names, gains = t27.extract_gain(model)
        self.assertEqual(len(names), len(model.feature_names))
        self.assertEqual(len(gains), len(names))
        self.assertTrue(any(g > 0 for g in gains))
        rows = t27.rank_normalized(names, gains, 10)
        self.assertEqual(rows[0]["norm_importance"], 1.0)
        self.assertLessEqual(len(rows), 10)

    def test_main_runs(self):
        try:
            import xgboost  # noqa: F401
        except ImportError:
            self.skipTest("xgboost not installed")
        model = self._toy_model()
        with tempfile.TemporaryDirectory() as td:
            model.save(td)
            rc = t27.main(["--model-dir", td, "--top-k", "5"])
        self.assertEqual(rc, 0)


# ===========================================================================
# Table 28
# ===========================================================================


def _sample(sid, residue_ids, gt):
    n = len(residue_ids)
    y = np.fromiter((1.0 if r in gt else 0.0 for r in residue_ids),
                    dtype=np.float64, count=n)
    X = np.zeros((n, 4))
    return SampleData(sid=sid, X=X, y=y, residue_ids=list(residue_ids))


def _write_step4(path, sid, tool_preds):
    """tool_preds: list of (tool_id, success, plddt_mean, conf_map, gate)."""
    preds = []
    for tid, ok, plddt, conf, gate in tool_preds:
        rec = {"tool_id": tid, "success": ok}
        if plddt is not None:
            rec["plddt_mean"] = plddt
        if conf is not None:
            rec["per_residue_confidence"] = {str(k): v
                                             for k, v in conf.items()}
        if gate is not None:
            rec["binding_protein_residues"] = list(gate)
        preds.append(rec)
    line = {"sample_id": sid, "predictions": preds}
    (path / f"{sid}.jsonl").write_text(json.dumps(line), encoding="utf-8")


class TestJaccard(unittest.TestCase):

    def test_mean_pairwise(self):
        self.assertAlmostEqual(
            t28.mean_pairwise_jaccard([{1, 2}, {2, 3}]), 1 / 3)

    def test_lt2_none(self):
        self.assertIsNone(t28.mean_pairwise_jaccard([{1}]))

    def test_disjoint_zero(self):
        self.assertEqual(t28.mean_pairwise_jaccard([{1}, {2}]), 0.0)


class TestDetectors(unittest.TestCase):

    def test_low_plddt(self):
        low = {"boltz2": {"plddt": 50.0, "has_plddt": True, "gate": set()},
               "chai1": {"plddt": 40.0, "has_plddt": True, "gate": set()}}
        self.assertTrue(t28.detect_low_plddt(low))
        high = {"boltz2": {"plddt": 80.0, "has_plddt": True, "gate": set()}}
        self.assertFalse(t28.detect_low_plddt(high))

    def test_low_plddt_none_when_no_catA(self):
        only_p2 = {"p2rank": {"plddt": 0.0, "has_plddt": False,
                              "gate": {1}}}
        self.assertIsNone(t28.detect_low_plddt(only_p2))

    def test_disagreement(self):
        tools = {"boltz2": {"gate": {1, 2, 3}, "plddt": 0, "has_plddt": 0},
                 "chai1": {"gate": {10, 11}, "plddt": 0, "has_plddt": 0}}
        self.assertTrue(t28.detect_disagreement(tools))  # jaccard 0
        agree = {"a": {"gate": {1, 2, 3}}, "b": {"gate": {1, 2, 3}}}
        self.assertFalse(t28.detect_disagreement(agree))  # jaccard 1

    def test_multi_domain_keyword(self):
        self.assertTrue(t28.detect_multi_domain(
            {"protein_family": "Multi"}, [1, 2, 3]))

    def test_multi_domain_gap_fallback(self):
        # two segments split by a >50 gap
        self.assertTrue(t28.detect_multi_domain(None, [1, 2, 3, 80, 81]))
        self.assertFalse(t28.detect_multi_domain(None, [1, 2, 3, 4, 5]))

    def test_rna_struct(self):
        self.assertTrue(t28.detect_rna_struct({"rna_context": "junction"}))
        self.assertTrue(t28.detect_rna_struct(
            {"rna_context": "G-quadruplex"}))
        self.assertFalse(t28.detect_rna_struct({"rna_context": "stem-loop"}))

    def test_polish_relocate_single(self):
        self.assertTrue(t28.detect_polish_relocate({"action": "relocate"}))
        self.assertFalse(t28.detect_polish_relocate({"action": "mask"}))

    def test_polish_relocate_rounds(self):
        self.assertTrue(t28.detect_polish_relocate(
            {"rounds": [{"action": "mask"}, {"action": "relocate"}]}))


class TestStep4Read(unittest.TestCase):

    def test_read_tools(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _write_step4(d, "x1", [
                ("boltz2", True, 55.0, {1: 50.0, 2: 60.0}, [1, 2]),
                ("rf2na", True, None, {1: 30.0, 2: 40.0}, [2]),
                ("p2rank", False, None, None, None),  # dropped (failure)
            ])
            tools = t28.read_step4_tools(d, "x1")
        self.assertIn("boltz2", tools)
        self.assertIn("rosettafold2na", tools)  # rf2na alias canonicalised
        self.assertNotIn("p2rank", tools)
        self.assertEqual(tools["boltz2"]["plddt"], 55.0)
        self.assertEqual(tools["boltz2"]["gate"], {1, 2})
        # rf2na has no plddt_mean → falls back to mean(conf)=35
        self.assertAlmostEqual(tools["rosettafold2na"]["plddt"], 35.0)


class TestBottomDecile(unittest.TestCase):

    def test_threshold_and_selection(self):
        per_r = {f"s{i}": i / 10.0 for i in range(11)}  # 0.0 .. 1.0
        thr, failed = t28.bottom_decile(per_r)
        self.assertAlmostEqual(thr, 1.0, delta=1.0)  # 10th pct of 0..1
        # at least the worst sample is included and is first
        self.assertEqual(failed[0], "s0")


class TestAnalyseEndToEnd(unittest.TestCase):

    def test_counts_and_modes(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            # f1: low plddt + disagreement
            _write_step4(d, "f1", [
                ("boltz2", True, 40.0, {1: 40.0}, {1, 2, 3}),
                ("chai1", True, 45.0, {1: 45.0}, {50, 51}),
            ])
            # f2: high plddt, agreeing tools, relocate → polish only
            _write_step4(d, "f2", [
                ("boltz2", True, 90.0, {1: 90.0}, {1, 2}),
                ("chai1", True, 88.0, {1: 88.0}, {1, 2}),
            ])
            # f3: nothing → other
            _write_step4(d, "f3", [
                ("boltz2", True, 90.0, {1: 90.0}, {1, 2}),
                ("chai1", True, 88.0, {1: 88.0}, {1, 2}),
            ])
            samples = {
                "f1": _sample("f1", range(1, 60), {1, 2, 3}),
                "f2": _sample("f2", range(1, 10), {1, 2}),
                "f3": _sample("f3", range(1, 10), {1, 2}),
            }
            scope = {"f1": {"protein_family": "Multi",
                            "rna_context": "junction"}}
            polish = {"f2": {"action": "relocate"}}
            counts, detail = t28.analyse(
                ["f1", "f2", "f3"], samples, d, scope, polish,
                have_scope=True, have_polish=True)

        self.assertEqual(counts["low_plddt"], 1)        # f1
        self.assertEqual(counts["disagreement"], 1)     # f1
        self.assertEqual(counts["multi_domain"], 1)     # f1 (Multi)
        self.assertEqual(counts["rna_struct"], 1)       # f1 (junction)
        self.assertEqual(counts["polish_relocate"], 1)  # f2
        self.assertEqual(counts["other"], 1)            # f3
        modes = {d_["sid"]: set(d_["modes"]) for d_ in detail}
        self.assertEqual(modes["f3"], {"other"})
        self.assertIn("multi_domain", modes["f1"])


if __name__ == "__main__":
    unittest.main()
