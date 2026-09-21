"""Mock tests for src/step7_iteration/polish_ops.py."""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from step7_iteration import polish_ops as po  # noqa: E402


class TestApply(unittest.TestCase):
    def setUp(self):
        self.prob = {1: 0.9, 2: 0.8, 3: 0.1, 4: 0.7, 5: 0.2}

    def test_accept_is_noop(self):
        out = po.apply_polish_to_probability(self.prob, {"action": "accept"})
        self.assertEqual(out, self.prob)

    def test_unknown_action_is_noop(self):
        out = po.apply_polish_to_probability(self.prob, {"action": "zzz"})
        self.assertEqual(out, self.prob)

    def test_mask_scales_down_by_default(self):
        # No confidence → default 0.2× (soft suppress, not hard zero).
        out = po.apply_polish_to_probability(
            self.prob, {"action": "mask", "residues": [1, 4]})
        self.assertAlmostEqual(out[1], 0.9 * 0.2)
        self.assertAlmostEqual(out[4], 0.7 * 0.2)
        self.assertEqual(out[2], 0.8)  # untouched

    def test_mask_confidence_modulates_strength(self):
        # factor = 1 - 0.8*confidence
        full = po.apply_polish_to_probability(
            self.prob, {"action": "mask", "residues": [1], "confidence": 1.0})
        self.assertAlmostEqual(full[1], 0.9 * 0.2)        # 1-0.8 = 0.2
        half = po.apply_polish_to_probability(
            self.prob, {"action": "mask", "residues": [1], "confidence": 0.5})
        self.assertAlmostEqual(half[1], 0.9 * 0.6)        # 1-0.4 = 0.6
        none = po.apply_polish_to_probability(
            self.prob, {"action": "mask", "residues": [1], "confidence": 0.0})
        self.assertAlmostEqual(none[1], 0.9)              # factor 1.0 = no-op

    def test_does_not_mutate_input(self):
        po.apply_polish_to_probability(
            self.prob, {"action": "mask", "residues": [1]})
        self.assertEqual(self.prob[1], 0.9)

    def test_extend_lifts_to_neighbour_mean_fraction(self):
        # residue 3 extended: ±2 binding neighbours 1(0.9),2(0.8),4(0.7),5(0.2)
        # mean = 0.65, lifted to max(current 0.1, 0.8*0.65 = 0.52).
        out = po.apply_polish_to_probability(
            self.prob, {"action": "extend", "residues": [3]},
            binding_set=[1, 2, 4, 5])
        self.assertAlmostEqual(out[3], 0.8 * (0.9 + 0.8 + 0.7 + 0.2) / 4)

    def test_extend_never_lowers(self):
        # current already above 0.8*neighbour_mean → unchanged.
        out = po.apply_polish_to_probability(
            {1: 0.9, 2: 0.9, 3: 0.95}, {"action": "extend", "residues": [3]},
            binding_set=[1, 2])
        self.assertAlmostEqual(out[3], 0.95)

    def test_extend_fallback_when_no_binding_neighbour(self):
        # fallback mean 0.5 → lift to max(0.1, 0.8*0.5 = 0.4).
        out = po.apply_polish_to_probability(
            {10: 0.1, 11: 0.2}, {"action": "extend", "residues": [10]},
            binding_set=[], extend_fallback=0.5)
        self.assertAlmostEqual(out[10], 0.4)

    def test_relocate(self):
        out = po.apply_polish_to_probability(
            self.prob,
            {"action": "relocate", "residues": [1, 2],
             "target_residues": [4, 5]})
        self.assertAlmostEqual(out[1], 0.9 * 0.2)   # old region soft-suppressed
        self.assertAlmostEqual(out[2], 0.8 * 0.2)
        # new region lifted to 0.8*within-region max (max(0.7,0.2)=0.7 → 0.56)
        self.assertAlmostEqual(out[4], 0.7)         # max(0.7, 0.56)
        self.assertAlmostEqual(out[5], 0.56)        # max(0.2, 0.56)

    def test_apply_actions_sequential(self):
        out = po.apply_actions(
            self.prob,
            [{"action": "mask", "residues": [1]},
             {"action": "mask", "residues": [2]}])
        self.assertAlmostEqual(out[1], 0.9 * 0.2)
        self.assertAlmostEqual(out[2], 0.8 * 0.2)


class TestAutoAction(unittest.TestCase):
    def test_too_few_binders_accept(self):
        act = po.auto_polish_action({1: 0.9, 2: 0.8, 3: 0.1})
        self.assertEqual(act["action"], "accept")
        self.assertEqual(act["residues"], [])

    def test_masks_weakest_quartile(self):
        # 8 binders; weakest 25% = 2 residues (the lowest smoothed).
        prob = {i: 0.9 for i in range(1, 9)}
        prob[7] = 0.55  # weak but above threshold
        prob[8] = 0.51
        act = po.auto_polish_action(prob, weak_quantile=0.25)
        self.assertEqual(act["action"], "mask")
        self.assertEqual(len(act["residues"]), 2)
        self.assertIn(8, act["residues"])
        self.assertIn(7, act["residues"])

    def test_deterministic(self):
        prob = {i: 0.6 + 0.01 * i for i in range(1, 9)}
        a1 = po.auto_polish_action(prob)
        a2 = po.auto_polish_action(prob)
        self.assertEqual(a1, a2)

    def test_empty(self):
        act = po.auto_polish_action({})
        self.assertEqual(act["action"], "accept")


if __name__ == "__main__":
    unittest.main()
