"""Mock tests for scripts/riboseer/compute_weight_tensor.py."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.riboseer import compute_weight_tensor as cwt  # noqa: E402
from step5_fusion.lightgbm_fusion import (  # noqa: E402
    SampleT9,
)
from step5_fusion.features_15tool import ALL_KNOWN_TOOLS  # noqa: E402
from step3_tool_selection.weight_tensor import (  # noqa: E402
    METRICS, WeightTensor,
)


def _sample(sid, length, binding, *, preds):
    rids = list(range(1, length + 1))
    y = np.fromiter((1.0 if r in binding else 0.0 for r in rids),
                    dtype=np.float64, count=length)
    return SampleT9(
        sid=sid, step4_data={"predictions": preds}, sample={},
        residue_ids=rids, y=y, eval_mask=np.ones(length, dtype=bool),
        protein_len=length, rna_len=40,
        available_tools=[p["tool_id"] for p in preds])


def _pred(tool_id, conf):
    return {"tool_id": tool_id, "success": True,
            "per_residue_pae_score": {}, "per_residue_confidence": conf,
            "binding_protein_residues": []}


class TestPerToolPearson(unittest.TestCase):
    def test_mu_and_n(self):
        # boltz2 perfectly correlates with GT on both samples; equipnas on one.
        gt_like = {1: 0.9, 2: 0.8, 3: 0.1, 4: 0.7, 5: 0.2}
        s1 = _sample("a", 5, binding={1, 2, 4},
                     preds=[_pred("boltz2", gt_like),
                            _pred("equipnas", gt_like)])
        s2 = _sample("b", 5, binding={1, 2, 4},
                     preds=[_pred("boltz2", gt_like)])
        mu, n = cwt.per_tool_pearson([s1, s2])
        self.assertEqual(n["boltz2"], 2)
        self.assertEqual(n["equipnas"], 1)
        self.assertGreater(mu["boltz2"], 0.5)   # strong positive corr

    def test_constant_prediction_skipped(self):
        # A flat prediction → undefined correlation → not counted.
        s = _sample("a", 5, binding={1, 2},
                    preds=[_pred("boltz2", {1: 0.5, 2: 0.5, 3: 0.5,
                                            4: 0.5, 5: 0.5})])
        mu, n = cwt.per_tool_pearson([s])
        self.assertNotIn("boltz2", n)


class TestAssemble(unittest.TestCase):
    def test_all_15_tools_with_priority(self):
        mu_train = {"boltz2": 0.5, "equipnas": 0.3}
        n_train = {"boltz2": 200, "equipnas": 120}
        rows = cwt.assemble_tool_stats(mu_train, n_train)
        self.assertEqual(len(rows), len(ALL_KNOWN_TOOLS))
        by = {r["tool_id"]: r for r in rows}
        # training estimate
        self.assertEqual(by["boltz2"]["source"], "train")
        self.assertEqual(by["boltz2"]["n"], 200)
        # hard-coded test estimates (fixed in code, not from train)
        self.assertEqual(by["alphafold3"]["source"], "test_estimate")
        self.assertAlmostEqual(by["alphafold3"]["mu_hat"], 0.246)
        self.assertEqual(by["alphafold3"]["n"], 100)
        self.assertAlmostEqual(by["bindup"]["mu_hat"], 0.250)
        self.assertEqual(by["bindup"]["n"], 100)
        # tool with no data anywhere → category prior, N=0
        self.assertEqual(by["chai1"]["source"], "prior")
        self.assertEqual(by["chai1"]["n"], 0)

    def test_test_estimate_overrides_train(self):
        # Even if af3/bindup somehow had train rows, the fixed values win.
        rows = cwt.assemble_tool_stats(
            {"alphafold3": 0.9, "bindup": 0.9},
            {"alphafold3": 50, "bindup": 50})
        by = {r["tool_id"]: r for r in rows}
        self.assertAlmostEqual(by["alphafold3"]["mu_hat"], 0.246)
        self.assertEqual(by["alphafold3"]["n"], 100)


class TestBuildTensor(unittest.TestCase):
    def _rows(self):
        return cwt.assemble_tool_stats({"boltz2": 0.6, "equipnas": 0.3},
                                       {"boltz2": 200, "equipnas": 120})

    def test_encodes_mu_into_all_metrics_and_counts(self):
        wt = cwt.build_weight_tensor(self._rows())
        for m in METRICS:
            self.assertAlmostEqual(
                wt.get_weight("boltz2", m, cwt.STORE_CATEGORY), 0.6)
        self.assertEqual(wt.get_count("boltz2", cwt.STORE_CATEGORY), 200)
        # weighted sum over metrics ≈ μ̂ (alpha sums to 1)
        util = wt.compute_utility("boltz2", cwt.STORE_CATEGORY)
        self.assertGreater(util, 0.6)             # μ̂ + exploration bonus

    def test_any_category_returns_mu(self):
        # THE FIX: μ̂ is stored in the global cell, so ANY pocket category a
        # sample resolves to returns the real μ̂ (not 0 / a letter prior).
        wt = cwt.build_weight_tensor(self._rows())
        for cat in ("novel_x_stem-loop", "RRM_x_hairpin", "whatever"):
            self.assertAlmostEqual(
                wt.get_weight("boltz2", METRICS[0], cat), 0.6)
            self.assertGreater(wt.compute_utility("boltz2", cat), 0.6)
            # count + exploration bonus also resolve via the global cell
            self.assertEqual(wt.get_count("boltz2", cat), 200)

    def test_higher_mu_ranks_higher_for_arbitrary_category(self):
        wt = cwt.build_weight_tensor(self._rows())
        self.assertGreater(
            wt.compute_utility("boltz2", "novel_x_stem-loop"),
            wt.compute_utility("equipnas", "novel_x_stem-loop"))

    def test_data_driven_letter_defaults(self):
        # Letter defaults are still data-driven (mean μ̂ of A-tools with
        # evidence) as a secondary fallback for tools lacking a global cell.
        rows = self._rows()
        wt = cwt.build_weight_tensor(rows)
        a_vals = [r["mu_hat"] for r in rows
                  if r["category"] == "A" and r["n"] > 0]
        expected = round(sum(a_vals) / len(a_vals), 6)
        self.assertAlmostEqual(wt._defaults["A"], expected)
        # a tool with NO global cell falls through to the letter default
        bare = WeightTensor(default_weights=wt._defaults)
        self.assertAlmostEqual(
            bare.get_weight("boltz2", METRICS[0], "x"), expected)

    def test_round_trips_through_save_load(self):
        wt = cwt.build_weight_tensor(self._rows())
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "W.json"
            wt.save(out)
            obj = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("data", obj)
            self.assertIn("counts", obj)
            reloaded = WeightTensor.load(out)
        # global cell survives the round trip → any category still resolves
        self.assertAlmostEqual(
            reloaded.get_weight("boltz2", METRICS[0], "novel_x_stem-loop"),
            0.6)


if __name__ == "__main__":
    unittest.main()
