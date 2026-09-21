"""Unit tests for step6 evolutionary_conservation (q3).

Each scenario states the expected ratios + sub-scores in its docstring
so the math can be checked by inspection without re-running.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step6_pocket_qa.metrics.conservation import (  # noqa: E402
    evolutionary_conservation,
)


def _default_cfg(**overrides) -> dict:
    base = {
        "rare_aa": "CWMH",
        "rare_aa_threshold": 0.15,
        "terminal_fraction": 0.1,
        "positive_pi_threshold": 9.0,
        "acidic_pi_threshold": 6.0,
        "charge_target": 0.3,
    }
    base.update(overrides)
    return base


def _sample_with_pi(pi):
    return {"protein": {"features": {"pI": pi}}}


# ============================================================================
# Abstain (None) cases
# ============================================================================


class TestAbstain(unittest.TestCase):
    def test_empty_binding_returns_none(self):
        score, info = evolutionary_conservation(
            [], "MKRRKKKK", _sample_with_pi(8.5), _default_cfg(),
        )
        self.assertIsNone(score)
        self.assertIn("empty", info["reason"])

    def test_empty_sequence_returns_none(self):
        score, info = evolutionary_conservation(
            [1, 2], "", _sample_with_pi(7.0), _default_cfg(),
        )
        self.assertIsNone(score)
        self.assertIn("protein_sequence", info["reason"])

    def test_all_out_of_range_returns_none(self):
        score, info = evolutionary_conservation(
            [50, 60], "MKR", _sample_with_pi(7.0), _default_cfg(),
        )
        self.assertIsNone(score)
        self.assertEqual(info["n_standard_aa"], 0)
        self.assertEqual(info["n_out_of_range"], 2)

    def test_all_non_standard_returns_none(self):
        score, info = evolutionary_conservation(
            [1, 2, 3], "XXX", _sample_with_pi(7.0), _default_cfg(),
        )
        self.assertIsNone(score)
        self.assertEqual(info["n_standard_aa"], 0)
        self.assertEqual(info["n_non_standard"], 3)


# ============================================================================
# Sub-score 1: rare-AA enrichment
# ============================================================================


class TestRareAA(unittest.TestCase):
    def test_zero_rare_residues(self):
        """All hydrophobic non-rare → score_rare = 0.0."""
        score, info = evolutionary_conservation(
            [1, 2, 3, 4], "ALVI", {}, _default_cfg(),
        )
        self.assertEqual(info["counts"]["rare"], 0)
        self.assertEqual(info["sub_scores"]["rare_aa"], 0.0)

    def test_full_rare_residues(self):
        """All CWMH → rare_ratio = 1.0 → score_rare = 1.0 (min clamp)."""
        score, info = evolutionary_conservation(
            [1, 2, 3, 4], "CWMH", {}, _default_cfg(),
        )
        self.assertEqual(info["counts"]["rare"], 4)
        self.assertEqual(info["sub_scores"]["rare_aa"], 1.0)

    def test_rare_at_threshold_gives_full_credit(self):
        """3 rare out of 20 = 0.15 == threshold → score = 1.0."""
        seq = "CWM" + "A" * 17
        score, info = evolutionary_conservation(
            list(range(1, 21)), seq, {}, _default_cfg(),
        )
        self.assertEqual(info["ratios"]["rare"], 0.15)
        self.assertEqual(info["sub_scores"]["rare_aa"], 1.0)

    def test_rare_half_threshold(self):
        """1 rare out of 20 = 0.05 → 0.05 / 0.15 = 0.333..."""
        seq = "C" + "A" * 19
        score, info = evolutionary_conservation(
            list(range(1, 21)), seq, {}, _default_cfg(),
        )
        self.assertEqual(info["counts"]["rare"], 1)
        self.assertAlmostEqual(info["sub_scores"]["rare_aa"], 1 / 3, places=3)

    def test_zero_threshold_full_credit_when_any_rare(self):
        """rare_threshold=0 → any rare residue gives 1.0."""
        seq = "C" + "A" * 19
        score, info = evolutionary_conservation(
            list(range(1, 21)), seq, {}, _default_cfg(rare_aa_threshold=0.0),
        )
        self.assertEqual(info["sub_scores"]["rare_aa"], 1.0)

    def test_zero_threshold_zero_when_no_rare(self):
        """rare_threshold=0 + no rare residues → 0.0."""
        score, info = evolutionary_conservation(
            [1, 2, 3], "AAA", {}, _default_cfg(rare_aa_threshold=0.0),
        )
        self.assertEqual(info["sub_scores"]["rare_aa"], 0.0)


# ============================================================================
# Sub-score 2: positional (terminal penalty)
# ============================================================================


class TestPosition(unittest.TestCase):
    def test_all_middle_full_credit(self):
        """seq_len=100, terminal=10; binding=[40, 50, 60] all in middle."""
        seq = "A" * 100
        score, info = evolutionary_conservation(
            [40, 50, 60], seq, {}, _default_cfg(),
        )
        self.assertEqual(info["terminal_size"], 10)
        self.assertEqual(info["n_terminal_binding"], 0)
        self.assertEqual(info["sub_scores"]["position"], 1.0)

    def test_all_terminal_zero_credit(self):
        """All binding in N/C termini → score_position = 0.0."""
        seq = "A" * 100
        # First 5 residues + last 5 residues; terminal_size=10 covers both.
        score, info = evolutionary_conservation(
            [1, 2, 3, 96, 97], seq, {}, _default_cfg(),
        )
        self.assertEqual(info["n_terminal_binding"], 5)
        self.assertEqual(info["sub_scores"]["position"], 0.0)

    def test_half_terminal_half_middle(self):
        """seq_len=100, terminal=10; 2 of 4 in N-term → score = 0.5."""
        seq = "A" * 100
        score, info = evolutionary_conservation(
            [3, 8, 50, 60], seq, {}, _default_cfg(),
        )
        self.assertEqual(info["n_terminal_binding"], 2)
        self.assertAlmostEqual(info["sub_scores"]["position"], 0.5, places=4)

    def test_terminal_fraction_zero_falls_back_to_minimum_one(self):
        """terminal_fraction=0 still uses size=1 (min) so the rule fires."""
        seq = "A" * 100
        # binding=[1, 2, 50] — only res 1 is in (N=[1..1], C=[100..100])
        score, info = evolutionary_conservation(
            [1, 2, 50], seq, {}, _default_cfg(terminal_fraction=0.0),
        )
        self.assertEqual(info["terminal_size"], 1)
        self.assertEqual(info["n_terminal_binding"], 1)
        self.assertAlmostEqual(
            info["sub_scores"]["position"], 1 - 1 / 3, places=4,
        )

    def test_short_sequence_terminal_capped_at_half(self):
        """seq_len=4 → half=2; round(0.4)=0 but clamp lifts to 1."""
        seq = "AAAA"
        score, info = evolutionary_conservation(
            [1, 2, 3, 4], seq, {}, _default_cfg(terminal_fraction=0.1),
        )
        # seq_len=4, half=2, round(4*0.1)=round(0.4)=0 → clamp to 1.
        self.assertEqual(info["terminal_size"], 1)
        self.assertEqual(info["n_terminal_binding"], 2)  # res 1 and res 4


# ============================================================================
# Sub-score 3: pI consistency
# ============================================================================


class TestPiConsistency(unittest.TestCase):
    def test_basic_protein_with_positive_interface(self):
        """pI=10 > 9; binding=KKK → pos_ratio=1.0 → pi_score=1.0."""
        score, info = evolutionary_conservation(
            [1, 2, 3], "KKK", _sample_with_pi(10.0), _default_cfg(),
        )
        self.assertEqual(info["sub_scores"]["pi_consistency"], 1.0)

    def test_basic_protein_no_positive_interface(self):
        """pI=10 > 9; binding all hydrophobic → pos_ratio=0 → pi_score=0."""
        score, info = evolutionary_conservation(
            [1, 2, 3], "AAA", _sample_with_pi(10.0), _default_cfg(),
        )
        self.assertEqual(info["sub_scores"]["pi_consistency"], 0.0)

    def test_basic_protein_partial_positive(self):
        """pI=10 > 9; pos_ratio=0.15 → 0.15/0.3 = 0.5."""
        seq = "KKK" + "A" * 17
        score, info = evolutionary_conservation(
            list(range(1, 21)), seq, _sample_with_pi(10.0), _default_cfg(),
        )
        self.assertAlmostEqual(
            info["sub_scores"]["pi_consistency"], 0.5, places=4,
        )

    def test_acidic_protein_no_acidic_interface(self):
        """pI=4 < 6; binding=KKK → acidic_ratio=0 → pi_score=1.0."""
        score, info = evolutionary_conservation(
            [1, 2, 3], "KKK", _sample_with_pi(4.0), _default_cfg(),
        )
        self.assertEqual(info["sub_scores"]["pi_consistency"], 1.0)

    def test_acidic_protein_with_acidic_interface(self):
        """pI=4 < 6; binding=DDD → acidic_ratio=1 → pi_score=0."""
        score, info = evolutionary_conservation(
            [1, 2, 3], "DDD", _sample_with_pi(4.0), _default_cfg(),
        )
        self.assertEqual(info["sub_scores"]["pi_consistency"], 0.0)

    def test_acidic_protein_partial_acidic(self):
        """pI=4 < 6; acidic_ratio=0.15 → 1 - 0.15/0.3 = 0.5."""
        seq = "DDD" + "A" * 17
        score, info = evolutionary_conservation(
            list(range(1, 21)), seq, _sample_with_pi(4.0), _default_cfg(),
        )
        self.assertAlmostEqual(
            info["sub_scores"]["pi_consistency"], 0.5, places=4,
        )

    def test_neutral_pi_returns_half(self):
        """pI=7 in [6, 9] → pi_score = 0.5."""
        score, info = evolutionary_conservation(
            [1, 2, 3], "KKK", _sample_with_pi(7.0), _default_cfg(),
        )
        self.assertEqual(info["sub_scores"]["pi_consistency"], 0.5)

    def test_pi_missing_skipped_from_average(self):
        """pI absent → pi sub-score is None and avg is over 2 axes."""
        score, info = evolutionary_conservation(
            [1, 2, 3, 4, 5], "CWMHK",
            {"protein": {"features": {}}}, _default_cfg(),
        )
        self.assertIsNone(info["sub_scores"]["pi_consistency"])
        self.assertEqual(info["n_sub_scores_computed"], 2)

    def test_pi_non_numeric_skipped(self):
        """pI non-numeric (e.g. None as string) → treated as missing."""
        score, info = evolutionary_conservation(
            [1, 2, 3], "CWM",
            {"protein": {"features": {"pI": "not a number"}}},
            _default_cfg(),
        )
        self.assertIsNone(info["sub_scores"]["pi_consistency"])

    def test_pi_threshold_boundary_basic(self):
        """pI exactly at positive_pi_threshold (9.0) → falls in neutral band."""
        # Strict > so pI=9.0 not "basic" → neutral 0.5.
        score, info = evolutionary_conservation(
            [1, 2, 3], "KKK", _sample_with_pi(9.0), _default_cfg(),
        )
        self.assertEqual(info["sub_scores"]["pi_consistency"], 0.5)


# ============================================================================
# Overall q3 average + clamps
# ============================================================================


class TestOverall(unittest.TestCase):
    def test_q3_with_three_axes_computed(self):
        """All 3 sub-scores computable → mean of three."""
        # binding 4 res in middle, all rare CWMH, basic protein
        seq = "A" * 20 + "CWMH" + "A" * 76  # length 100
        score, info = evolutionary_conservation(
            [21, 22, 23, 24], seq, _sample_with_pi(10.5), _default_cfg(),
        )
        self.assertEqual(info["n_sub_scores_computed"], 3)
        # rare=1.0, position=1.0 (all middle), pi=H is positive → ratio 1/4 → 0.25/0.3 ≈ 0.833
        self.assertEqual(info["sub_scores"]["rare_aa"], 1.0)
        self.assertEqual(info["sub_scores"]["position"], 1.0)
        # H is in POSITIVE_AA → 1/4 = 0.25 → 0.25/0.3 ≈ 0.833
        self.assertAlmostEqual(
            info["sub_scores"]["pi_consistency"], 0.25 / 0.3, places=3,
        )
        expected = (1.0 + 1.0 + 0.25 / 0.3) / 3
        self.assertAlmostEqual(score, expected, places=4)

    def test_q3_with_two_axes(self):
        """pI missing → q3 = mean(rare, position)."""
        # binding all in middle of long protein, no rare residues
        seq = "A" * 100
        score, info = evolutionary_conservation(
            [40, 50, 60], seq, {"protein": {"features": {}}}, _default_cfg(),
        )
        self.assertEqual(info["n_sub_scores_computed"], 2)
        # rare=0, position=1.0 → mean=0.5
        self.assertAlmostEqual(score, 0.5, places=4)

    def test_score_in_unit_interval(self):
        """Random configurations stay within [0, 1]."""
        seq = "MKDEAAAAAAAA" + "C" * 5 + "AAAAAA" + "K" * 5
        for residues in (
            [1, 2, 3], [10, 11, 12], [13, 14, 15],
            [25, 26, 27], list(range(1, len(seq) + 1)),
        ):
            score, info = evolutionary_conservation(
                residues, seq, _sample_with_pi(8.0), _default_cfg(),
            )
            if score is not None:
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)


# ============================================================================
# Filtering & config wiring
# ============================================================================


class TestFiltering(unittest.TestCase):
    def test_non_standard_excluded_from_numerator_and_denominator(self):
        """seq KXKX, binding=[1,2,3,4]: 2 standard letters, both K (positive)."""
        score, info = evolutionary_conservation(
            [1, 2, 3, 4], "KXKX", _sample_with_pi(10.5), _default_cfg(),
        )
        self.assertEqual(info["n_standard_aa"], 2)
        self.assertEqual(info["n_non_standard"], 2)
        self.assertEqual(info["counts"]["positive"], 2)
        # pos_ratio 2/2 = 1.0 → pi_consistency = 1.0
        self.assertEqual(info["sub_scores"]["pi_consistency"], 1.0)

    def test_index_zero_treated_as_out_of_range(self):
        score, info = evolutionary_conservation(
            [0, 1, 2], "MKR", _sample_with_pi(10.5), _default_cfg(),
        )
        self.assertEqual(info["n_out_of_range"], 1)
        self.assertEqual(info["n_standard_aa"], 2)


class TestConfigWiring(unittest.TestCase):
    def test_custom_rare_aa_set(self):
        """Override rare_aa to detect P only."""
        seq = "PPP" + "A" * 17
        score, info = evolutionary_conservation(
            list(range(1, 21)), seq, {},
            _default_cfg(rare_aa="P", rare_aa_threshold=0.15),
        )
        self.assertEqual(info["counts"]["rare"], 3)
        self.assertEqual(info["ratios"]["rare"], 0.15)
        self.assertEqual(info["sub_scores"]["rare_aa"], 1.0)

    def test_custom_pi_thresholds(self):
        """Lower positive_pi_threshold → pI=8 now triggers basic branch."""
        score, info = evolutionary_conservation(
            [1, 2, 3], "KKK", _sample_with_pi(8.0),
            _default_cfg(positive_pi_threshold=7.5),
        )
        self.assertEqual(info["sub_scores"]["pi_consistency"], 1.0)

    def test_empty_config_uses_defaults(self):
        """Passing {} → defaults still work."""
        score, info = evolutionary_conservation(
            [1, 2, 3, 4], "CWMH", {}, {},
        )
        self.assertEqual(info["sub_scores"]["rare_aa"], 1.0)
        self.assertEqual(info["config"]["rare_aa_threshold"], 0.15)
        self.assertEqual(info["config"]["positive_pi_threshold"], 9.0)

    def test_custom_charge_target(self):
        """charge_target=0.5 → pos_ratio 0.25 → score 0.5."""
        seq = "K" + "A" * 3  # 1 K, 3 A; pos_ratio=0.25
        score, info = evolutionary_conservation(
            [1, 2, 3, 4], seq, _sample_with_pi(10.5),
            _default_cfg(charge_target=0.5),
        )
        self.assertAlmostEqual(
            info["sub_scores"]["pi_consistency"], 0.5, places=4,
        )


# ============================================================================
# Scorer integration smoke
# ============================================================================


class TestScorerIntegration(unittest.TestCase):
    """Sanity check that the scorer's _safe_metric_call wraps q3 correctly."""

    def test_metric_detail_via_scorer(self):
        from step6_pocket_qa.scorer import _safe_metric_call

        detail = _safe_metric_call(
            evolutionary_conservation,
            binding_residues=[1, 2, 3, 4],
            protein_sequence="CWMH",
            sample_json=_sample_with_pi(10.0),
            config=_default_cfg(),
        )
        self.assertTrue(detail.computed)
        self.assertIsNotNone(detail.score)
        self.assertGreaterEqual(detail.score, 0.0)
        self.assertLessEqual(detail.score, 1.0)
        self.assertIsNone(detail.error)

    def test_abstain_via_scorer(self):
        from step6_pocket_qa.scorer import _safe_metric_call

        detail = _safe_metric_call(
            evolutionary_conservation,
            binding_residues=[],
            protein_sequence="MKR",
            sample_json={},
            config=_default_cfg(),
        )
        self.assertFalse(detail.computed)
        self.assertIsNone(detail.score)
        self.assertIsNone(detail.error)
        self.assertIn("empty", detail.info["reason"])


if __name__ == "__main__":
    unittest.main()
