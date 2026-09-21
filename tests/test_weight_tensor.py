"""Unit tests for weight_tensor.py."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step3_tool_selection.weight_tensor import (  # noqa: E402
    WeightTensor, METRICS, _DEFAULT_CATEGORY_WEIGHTS,
)
from step3_tool_selection.tool_registry import get_all_tool_ids  # noqa: E402


class TestColdStart(unittest.TestCase):
    def setUp(self):
        self.wt = WeightTensor()

    def test_default_weight_by_category(self):
        # Defaults are looked up via tool metadata which still resolves for
        # inactive tools (alphafold3 / hdock) — covers both active and
        # inactive cases so all four category defaults are exercised.
        self.assertEqual(self.wt.get_weight("alphafold3", "structural_plausibility", "RRM_x_stem_loop"), 0.7)
        self.assertEqual(self.wt.get_weight("boltz2", "structural_plausibility", "RRM_x_stem_loop"), 0.7)
        self.assertEqual(self.wt.get_weight("p2rank", "structural_plausibility", "RRM_x_stem_loop"), 0.5)
        self.assertEqual(self.wt.get_weight("equipnas", "structural_plausibility", "RRM_x_stem_loop"), 0.6)
        self.assertEqual(self.wt.get_weight("hdock", "structural_plausibility", "RRM_x_stem_loop"), 0.6)

    def test_all_metrics_present(self):
        w = self.wt.get_weights("boltz2", "RRM_x_stem_loop")
        self.assertEqual(set(w.keys()), set(METRICS))

    def test_count_is_zero(self):
        self.assertEqual(self.wt.get_count("boltz2", "RRM_x_stem_loop"), 0)


class TestUpdate(unittest.TestCase):
    def test_update_single_cell(self):
        wt = WeightTensor()
        wt.update("equipnas", "RRM_x_stem_loop", "structural_plausibility", 0.85)
        self.assertEqual(
            wt.get_weight("equipnas", "structural_plausibility", "RRM_x_stem_loop"),
            0.85,
        )
        # other metrics still default
        self.assertEqual(
            wt.get_weight("equipnas", "evolutionary_conservation", "RRM_x_stem_loop"),
            0.6,
        )

    def test_increment_count(self):
        wt = WeightTensor()
        wt.increment_count("equipnas", "RRM_x_stem_loop")
        wt.increment_count("equipnas", "RRM_x_stem_loop")
        self.assertEqual(wt.get_count("equipnas", "RRM_x_stem_loop"), 2)
        # different category untouched
        self.assertEqual(wt.get_count("equipnas", "KH_x_junction"), 0)


class TestUCB(unittest.TestCase):
    def test_cold_start_exploration_bonus_is_beta(self):
        wt = WeightTensor(beta=2.0)
        bonus = wt._exploration_bonus("equipnas", "RRM_x_stem_loop")
        self.assertEqual(bonus, 2.0)

    def test_cold_start_utility_equals_weighted_sum_plus_beta(self):
        alpha = {m: 0.2 for m in METRICS}
        wt = WeightTensor(alpha=alpha, beta=1.5)
        # Category C default = 0.6; weighted sum = 5 * 0.2 * 0.6 = 0.6
        u = wt.compute_utility("equipnas", "RRM_x_stem_loop")
        self.assertAlmostEqual(u, 0.6 + 1.5, places=4)

    def test_utility_ranking_reflects_category_defaults(self):
        wt = WeightTensor(beta=0.0)  # no exploration
        scores = wt.compute_utility_scores("RRM_x_stem_loop")
        # Category A tools (0.7) should rank above C (0.6) above B (0.5).
        # Pick representatives from the active set.
        a_score = scores["boltz2"]
        c_score = scores["equipnas"]
        b_score = scores["p2rank"]
        self.assertGreater(a_score, c_score)
        self.assertGreater(c_score, b_score)

    def test_exploration_bonus_decreases_with_count(self):
        wt = WeightTensor(beta=1.0)
        # Simulate: tool A evaluated 10 times, tool B evaluated 1 time
        for _ in range(10):
            wt.increment_count("equipnas", "X_x_Y")
        wt.increment_count("p2rank", "X_x_Y")
        bonus_a = wt._exploration_bonus("equipnas", "X_x_Y")
        bonus_b = wt._exploration_bonus("p2rank", "X_x_Y")
        self.assertLess(bonus_a, bonus_b)

    def test_compute_utility_scores_sorted_desc(self):
        wt = WeightTensor()
        scores = wt.compute_utility_scores("RRM_x_stem_loop")
        values = list(scores.values())
        self.assertEqual(values, sorted(values, reverse=True))

    def test_all_tools_in_scores(self):
        wt = WeightTensor()
        scores = wt.compute_utility_scores("RRM_x_stem_loop")
        self.assertEqual(set(scores.keys()), set(get_all_tool_ids()))


class TestCategorySummary(unittest.TestCase):
    def test_contains_all_tools(self):
        wt = WeightTensor()
        summary = wt.get_category_summary("RRM_x_stem_loop")
        for tid in get_all_tool_ids():
            self.assertIn(tid, summary)

    def test_contains_ucb_scores(self):
        wt = WeightTensor()
        summary = wt.get_category_summary("RRM_x_stem_loop")
        self.assertIn("UCB=", summary)
        self.assertIn("evaluations=", summary)


class TestSaveLoad(unittest.TestCase):
    def test_roundtrip(self):
        wt = WeightTensor(beta=2.5)
        wt.update("equipnas", "RRM_x_stem_loop", "structural_plausibility", 0.9)
        wt.increment_count("equipnas", "RRM_x_stem_loop")

        tmp = Path(tempfile.mkdtemp()) / "wt.json"
        wt.save(tmp)

        wt2 = WeightTensor.load(tmp)
        self.assertEqual(
            wt2.get_weight("equipnas", "structural_plausibility", "RRM_x_stem_loop"),
            0.9,
        )
        self.assertEqual(wt2.get_count("equipnas", "RRM_x_stem_loop"), 1)
        self.assertEqual(wt2.beta, 2.5)

    def test_load_missing_returns_default(self):
        wt = WeightTensor.load(Path("/nonexistent/path.json"))
        self.assertEqual(wt.get_count("equipnas", "X_x_Y"), 0)

    def test_save_creates_dirs(self):
        tmp = Path(tempfile.mkdtemp()) / "sub" / "wt.json"
        wt = WeightTensor()
        wt.save(tmp)
        self.assertTrue(tmp.exists())


class TestFromConfig(unittest.TestCase):
    def test_loads_params(self):
        config = {
            "default_weights": {"A": 0.8, "B": 0.4, "C": 0.7, "D": 0.5},
            "ucb": {
                "beta": 2.0,
                "alpha": {"structural_plausibility": 0.5, "physicochemical_complementarity": 0.5},
            },
        }
        wt = WeightTensor.from_config(config)
        self.assertEqual(wt.beta, 2.0)
        # Category A default now 0.8
        self.assertEqual(wt.get_weight("alphafold3", "structural_plausibility", "X_x_Y"), 0.8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
