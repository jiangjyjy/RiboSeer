"""Unit tests for step6 physicochemical_complementarity (q2).

Each scenario states the expected ratios + sub-scores in its docstring
so the math can be checked by inspection without re-running.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step6_pocket_qa.metrics.physicochemical import (  # noqa: E402
    physicochemical_complementarity,
)


def _default_cfg(**overrides) -> dict:
    base = {
        "positive_threshold": 0.3,
        "aromatic_threshold": 0.1,
        "polar_threshold": 0.2,
        "gp_threshold": 0.3,
    }
    base.update(overrides)
    return base


# ============================================================================
# Abstain (None) cases
# ============================================================================


class TestAbstain(unittest.TestCase):
    def test_empty_binding_returns_none(self):
        score, info = physicochemical_complementarity(
            [], "MKRRKKKK", "AUCG", None, _default_cfg(),
        )
        self.assertIsNone(score)
        self.assertEqual(info["n_binding_input"], 0)
        self.assertIn("empty", info["reason"])

    def test_all_out_of_range_returns_none(self):
        # Sequence length 5, all residues > 5 → no standard AAs survive.
        score, info = physicochemical_complementarity(
            [10, 20, 30], "MKRRK", "AU", None, _default_cfg(),
        )
        self.assertIsNone(score)
        self.assertEqual(info["n_standard_aa"], 0)
        self.assertEqual(info["n_out_of_range"], 3)

    def test_all_non_standard_returns_none(self):
        # Sequence is all X's (non-canonical) — no standard AAs.
        score, info = physicochemical_complementarity(
            [1, 2, 3], "XXX", "AU", None, _default_cfg(),
        )
        self.assertIsNone(score)
        self.assertEqual(info["n_standard_aa"], 0)
        self.assertEqual(info["n_non_standard"], 3)

    def test_index_zero_treated_as_out_of_range(self):
        # Indices are 1-based; res=0 should be out of range, not aa[-1].
        score, info = physicochemical_complementarity(
            [0], "MKR", "A", None, _default_cfg(),
        )
        self.assertIsNone(score)
        self.assertEqual(info["n_out_of_range"], 1)


# ============================================================================
# Charge sub-score
# ============================================================================


class TestCharge(unittest.TestCase):
    def test_all_positive_residues_max_charge(self):
        """All K/R/H → pos_ratio = 1.0 → score_charge = 1.0."""
        # Residues 1-4 = K, R, K, R; binding all four.
        score, info = physicochemical_complementarity(
            [1, 2, 3, 4], "KRKR", "A", None, _default_cfg(),
        )
        self.assertEqual(info["counts"]["positive"], 4)
        self.assertEqual(info["sub_scores"]["charge"], 1.0)

    def test_all_hydrophobic_zero_charge(self):
        """All A/L/V → pos_ratio = 0 → score_charge = 0."""
        score, info = physicochemical_complementarity(
            [1, 2, 3, 4], "ALVI", "A", None, _default_cfg(),
        )
        self.assertEqual(info["counts"]["positive"], 0)
        self.assertEqual(info["sub_scores"]["charge"], 0.0)

    def test_charge_at_threshold_gives_full_credit(self):
        """positive_ratio == positive_threshold → score == 1.0 (min clamp)."""
        # 3 K out of 10 residues = 0.3, equal to default threshold.
        # Sequence: K K K A A A A A A A
        score, info = physicochemical_complementarity(
            list(range(1, 11)), "KKKAAAAAAA", "A", None, _default_cfg(),
        )
        self.assertEqual(info["ratios"]["positive"], 0.3)
        self.assertEqual(info["sub_scores"]["charge"], 1.0)

    def test_charge_below_threshold_partial_credit(self):
        """0.15 / 0.3 = 0.5 → score_charge = 0.5."""
        # 3 K out of 20 residues = 0.15.
        seq = "KKK" + "A" * 17
        score, info = physicochemical_complementarity(
            list(range(1, 21)), seq, "A", None, _default_cfg(),
        )
        self.assertEqual(info["ratios"]["positive"], 0.15)
        self.assertAlmostEqual(info["sub_scores"]["charge"], 0.5, places=4)


# ============================================================================
# Aromatic sub-score
# ============================================================================


class TestAromatic(unittest.TestCase):
    def test_all_aromatic_max_score(self):
        score, info = physicochemical_complementarity(
            [1, 2, 3], "FYW", "A", None, _default_cfg(),
        )
        self.assertEqual(info["counts"]["aromatic"], 3)
        self.assertEqual(info["sub_scores"]["aromatic"], 1.0)

    def test_aromatic_threshold_low_default(self):
        """1 aromatic in 10 residues = 0.1 → equals default threshold → 1.0."""
        seq = "F" + "A" * 9
        score, info = physicochemical_complementarity(
            list(range(1, 11)), seq, "A", None, _default_cfg(),
        )
        self.assertEqual(info["sub_scores"]["aromatic"], 1.0)

    def test_no_aromatic_zero_score(self):
        score, info = physicochemical_complementarity(
            [1, 2, 3], "AAA", "A", None, _default_cfg(),
        )
        self.assertEqual(info["counts"]["aromatic"], 0)
        self.assertEqual(info["sub_scores"]["aromatic"], 0.0)


# ============================================================================
# Polar sub-score
# ============================================================================


class TestPolar(unittest.TestCase):
    def test_all_polar_max_score(self):
        # All from {S,T,N,Q,D,E}.
        score, info = physicochemical_complementarity(
            [1, 2, 3, 4, 5, 6], "STNQDE", "A", None, _default_cfg(),
        )
        self.assertEqual(info["counts"]["polar"], 6)
        self.assertEqual(info["sub_scores"]["polar"], 1.0)

    def test_no_polar_zero_score(self):
        # All G/A — no polar residues.
        score, info = physicochemical_complementarity(
            [1, 2, 3, 4], "GAGA", "A", None, _default_cfg(),
        )
        self.assertEqual(info["counts"]["polar"], 0)
        self.assertEqual(info["sub_scores"]["polar"], 0.0)


# ============================================================================
# G/P sub-score (penalty above threshold)
# ============================================================================


class TestGlycineProline(unittest.TestCase):
    def test_no_gp_full_credit(self):
        score, info = physicochemical_complementarity(
            [1, 2, 3], "KRR", "A", None, _default_cfg(),
        )
        self.assertEqual(info["counts"]["glycine_proline"], 0)
        self.assertEqual(info["sub_scores"]["gp"], 1.0)

    def test_gp_below_threshold_full_credit(self):
        """gp_ratio = 0.2 < 0.3 default → score = 1.0."""
        # 1 G out of 5 = 0.2.
        score, info = physicochemical_complementarity(
            [1, 2, 3, 4, 5], "GAAAA", "A", None, _default_cfg(),
        )
        self.assertEqual(info["ratios"]["glycine_proline"], 0.2)
        self.assertEqual(info["sub_scores"]["gp"], 1.0)

    def test_gp_above_threshold_partial_penalty(self):
        """gp_ratio = 0.6 → penalty = (0.6-0.3)/0.3 = 1.0 → score = 0.0."""
        seq = "GGGPPPAAAA"  # 6 G/P out of 10 = 0.6
        score, info = physicochemical_complementarity(
            list(range(1, 11)), seq, "A", None, _default_cfg(),
        )
        self.assertEqual(info["ratios"]["glycine_proline"], 0.6)
        self.assertEqual(info["sub_scores"]["gp"], 0.0)

    def test_gp_at_threshold_full_credit(self):
        """gp_ratio == threshold → linear formula = 1.0 (boundary inclusive)."""
        # 3 G/P out of 10 = 0.3, exactly the threshold.
        seq = "GGGAAAAAAA"
        score, info = physicochemical_complementarity(
            list(range(1, 11)), seq, "A", None, _default_cfg(),
        )
        self.assertEqual(info["ratios"]["glycine_proline"], 0.3)
        # ratio < thr is False; (0.3-0.3)/0.3 = 0 → 1.0 - 0 = 1.0.
        self.assertEqual(info["sub_scores"]["gp"], 1.0)

    def test_gp_extreme_clamps_to_zero(self):
        """gp_ratio = 1.0 → 1 - 0.7/0.3 = -1.33 → clamp to 0."""
        score, info = physicochemical_complementarity(
            [1, 2, 3], "GGP", "A", None, _default_cfg(),
        )
        self.assertEqual(info["sub_scores"]["gp"], 0.0)


# ============================================================================
# Combined / overall score
# ============================================================================


class TestOverall(unittest.TestCase):
    def test_classic_rna_binding_interface(self):
        """Mixed ideal interface: 4K, 2Y, 2S, 2A in 10 residues.

        positive=4/10=0.4 → score_charge = 1.0 (>= 0.3)
        aromatic=2/10=0.2 → score_aromatic = 1.0 (>= 0.1)
        polar=2/10=0.2 → score_polar = 1.0 (>= 0.2)
        gp=0/10=0 → score_gp = 1.0
        q2 = (1+1+1+1)/4 = 1.0
        """
        seq = "KKKKYYSSAA"
        score, info = physicochemical_complementarity(
            list(range(1, 11)), seq, "AUCG", None, _default_cfg(),
        )
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_pure_hydrophobic_interface_low_score(self):
        """All hydrophobic A/L/V/I → all sub-scores 0 except gp=1.

        q2 = (0 + 0 + 0 + 1) / 4 = 0.25
        """
        seq = "ALVIALVI"
        score, info = physicochemical_complementarity(
            list(range(1, 9)), seq, "A", None, _default_cfg(),
        )
        self.assertAlmostEqual(score, 0.25, places=6)

    def test_score_in_unit_interval_random_sample(self):
        """Always in [0, 1] for arbitrary sequences."""
        seq = "MKRRGGGYHEEEFGPSTQQQNNDD"
        score, info = physicochemical_complementarity(
            list(range(1, len(seq) + 1)), seq, "AUCG", None, _default_cfg(),
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


# ============================================================================
# Non-standard / out-of-range handling
# ============================================================================


class TestFiltering(unittest.TestCase):
    def test_non_standard_aa_skipped_not_counted(self):
        """X is non-standard; should be excluded from numerator AND denominator.

        Binding residues = [1, 2, 3, 4]; sequence = K X R A
        n_standard = 3 (K, R, A); positive = 2 (K, R)
        pos_ratio = 2/3 ≈ 0.667 → score_charge clamps to 1.0 (>= 0.3).
        info should record n_non_standard=1.
        """
        score, info = physicochemical_complementarity(
            [1, 2, 3, 4], "KXRA", "A", None, _default_cfg(),
        )
        self.assertEqual(info["n_non_standard"], 1)
        self.assertEqual(info["n_standard_aa"], 3)
        self.assertEqual(info["counts"]["positive"], 2)
        self.assertAlmostEqual(info["ratios"]["positive"], 0.6667, places=4)

    def test_out_of_range_residues_recorded(self):
        """Residue 100 in a length-5 protein → out of range, skipped."""
        score, info = physicochemical_complementarity(
            [1, 2, 100], "KRRAA", "A", None, _default_cfg(),
        )
        self.assertEqual(info["n_out_of_range"], 1)
        self.assertEqual(info["n_standard_aa"], 2)
        # 2 K/R out of 2 std → pos_ratio = 1.0.
        self.assertEqual(info["ratios"]["positive"], 1.0)

    def test_lowercase_treated_as_non_standard(self):
        # Lowercase letters are not in STANDARD_AA frozenset.
        score, info = physicochemical_complementarity(
            [1, 2, 3], "krr", "A", None, _default_cfg(),
        )
        self.assertIsNone(score)
        self.assertEqual(info["n_non_standard"], 3)


# ============================================================================
# Config wiring
# ============================================================================


class TestConfigWiring(unittest.TestCase):
    def test_custom_thresholds_change_score(self):
        """Halving positive_threshold doubles the effective ratio score."""
        # 1 K out of 10 = 0.1.
        seq = "KAAAAAAAAA"
        cfg_default = _default_cfg()  # pos_thr = 0.3 → score = 0.1/0.3 ≈ 0.333
        cfg_strict = _default_cfg(positive_threshold=0.6)  # → 0.1/0.6 ≈ 0.167
        s1, i1 = physicochemical_complementarity(
            list(range(1, 11)), seq, "A", None, cfg_default,
        )
        s2, i2 = physicochemical_complementarity(
            list(range(1, 11)), seq, "A", None, cfg_strict,
        )
        self.assertAlmostEqual(i1["sub_scores"]["charge"], 0.3333, places=3)
        self.assertAlmostEqual(i2["sub_scores"]["charge"], 0.1667, places=3)

    def test_empty_config_uses_defaults(self):
        score, info = physicochemical_complementarity(
            [1, 2, 3, 4], "KRKR", "A", None, {},
        )
        # Same as default — all K/R, full charge credit.
        self.assertEqual(info["sub_scores"]["charge"], 1.0)
        self.assertEqual(info["thresholds"]["positive"], 0.3)

    def test_zero_threshold_safe(self):
        """positive_threshold=0 should not divide-by-zero."""
        cfg = _default_cfg(positive_threshold=0.0)
        # With any positive residue → score = 1.0.
        score, info = physicochemical_complementarity(
            [1, 2], "KA", "A", None, cfg,
        )
        self.assertEqual(info["sub_scores"]["charge"], 1.0)
        # Zero positives → score = 0.0.
        score2, info2 = physicochemical_complementarity(
            [1, 2], "AA", "A", None, cfg,
        )
        self.assertEqual(info2["sub_scores"]["charge"], 0.0)


# ============================================================================
# RNA-side info (not in score, but should pass through info)
# ============================================================================


class TestRnaInfoPassthrough(unittest.TestCase):
    def test_rna_length_in_info(self):
        score, info = physicochemical_complementarity(
            [1, 2, 3], "KRR", "AUCGAUCG", [1, 2], _default_cfg(),
        )
        self.assertEqual(info["rna_length"], 8)
        self.assertEqual(info["n_binding_nucleotides"], 2)

    def test_none_rna_inputs_handled(self):
        # binding_nucleotides=None is allowed (RNA-side may be missing).
        score, info = physicochemical_complementarity(
            [1, 2, 3], "KRR", "", None, _default_cfg(),
        )
        self.assertEqual(info["rna_length"], 0)
        self.assertEqual(info["n_binding_nucleotides"], 0)


if __name__ == "__main__":
    unittest.main()
