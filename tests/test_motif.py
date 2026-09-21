"""Unit tests for step6 known_motif_consistency (q5).

Each scenario states the expected hits + coverage in its docstring so
the math can be checked by inspection without re-running.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step6_pocket_qa.metrics.motif import (  # noqa: E402
    MOTIFS,
    known_motif_consistency,
)


def _default_cfg(**overrides) -> dict:
    base = {"neutral_score": 0.5}
    base.update(overrides)
    return base


# ============================================================================
# Abstain (None) cases
# ============================================================================


class TestAbstain(unittest.TestCase):
    def test_empty_binding_returns_none(self):
        score, info = known_motif_consistency(
            [], "MKRKGFGFV", "RRM_x_stem_loop", {}, _default_cfg(),
        )
        self.assertIsNone(score)
        self.assertIn("empty", info["reason"])

    def test_empty_sequence_returns_none(self):
        score, info = known_motif_consistency(
            [1, 2], "", "RRM_x_stem_loop", {}, _default_cfg(),
        )
        self.assertIsNone(score)
        self.assertIn("protein_sequence", info["reason"])

    def test_all_out_of_range_returns_none(self):
        score, info = known_motif_consistency(
            [50, 60], "MKR", "RRM_x_stem_loop", {}, _default_cfg(),
        )
        self.assertIsNone(score)
        self.assertEqual(info["n_binding_in_range"], 0)
        self.assertEqual(info["n_out_of_range"], 2)


# ============================================================================
# Neutral domains: novel_fold / unstructured / unknown
# ============================================================================


class TestNeutralDomains(unittest.TestCase):
    def test_novel_fold_returns_neutral(self):
        score, info = known_motif_consistency(
            [1, 2, 3], "KKK", "novel_fold_x_unstructured", {},
            _default_cfg(),
        )
        self.assertEqual(score, 0.5)
        self.assertEqual(info["protein_domain"], "novel_fold")
        self.assertEqual(info["n_motifs_found"], 0)

    def test_unstructured_protein_domain_returns_neutral(self):
        # Bare domain "unstructured" (rna_structure side, but accept it
        # as a domain label too — defensive parsing).
        score, info = known_motif_consistency(
            [1, 2, 3], "KKK", "unstructured", {}, _default_cfg(),
        )
        self.assertEqual(score, 0.5)

    def test_empty_category_returns_neutral(self):
        score, info = known_motif_consistency(
            [1, 2, 3], "KKK", "", {}, _default_cfg(),
        )
        self.assertEqual(score, 0.5)

    def test_unrecognised_domain_returns_neutral(self):
        score, info = known_motif_consistency(
            [1, 2, 3], "KKK", "Made_Up_Domain_x_stem_loop", {},
            _default_cfg(),
        )
        self.assertEqual(score, 0.5)
        self.assertIn("unrecognised", info["reason"])

    def test_custom_neutral_score(self):
        """Neutral score can be configured."""
        score, info = known_motif_consistency(
            [1, 2, 3], "KKK", "novel_fold_x_unstructured", {},
            _default_cfg(neutral_score=0.3),
        )
        self.assertEqual(score, 0.3)

    def test_neutral_score_clamped(self):
        """Neutral > 1 clamps to 1, < 0 clamps to 0."""
        score, _ = known_motif_consistency(
            [1, 2, 3], "KKK", "", {}, _default_cfg(neutral_score=2.5),
        )
        self.assertEqual(score, 1.0)
        score, _ = known_motif_consistency(
            [1, 2, 3], "KKK", "", {}, _default_cfg(neutral_score=-1.0),
        )
        self.assertEqual(score, 0.0)


# ============================================================================
# RRM motifs (RNP1 + RNP2)
# ============================================================================


class TestRRM(unittest.TestCase):
    def test_rnp1_full_coverage(self):
        """KGFGFVKF matches RNP1 exactly; binding covers all 8 positions."""
        # Embed RNP1 'KGFGFVKF' at positions 5-12 in a 20-aa sequence.
        seq = "AAAA" + "KGFGFVKF" + "AAAAAAAA"  # length 20
        # Verify the chosen substring matches our regex.
        # RNP1 = [KR]G[FY][GA][FY][VILM].[FY] — KGFGFVKF: K, G, F, G, F, V, K, F → ✓
        score, info = known_motif_consistency(
            list(range(5, 13)), seq, "RRM_x_stem_loop", {},
            _default_cfg(),
        )
        self.assertEqual(score, 1.0)
        self.assertEqual(info["n_motifs_found"], 1)
        self.assertEqual(info["motif_residues_in_binding"], 8)
        self.assertEqual(info["n_motifs_total_residues"], 8)

    def test_rnp1_zero_coverage(self):
        """Motif present but binding covers other residues."""
        seq = "AAAA" + "KGFGFVKF" + "AAAAAAAA"
        score, info = known_motif_consistency(
            [1, 2, 3, 18, 19, 20], seq, "RRM_x_stem_loop", {},
            _default_cfg(),
        )
        self.assertEqual(score, 0.0)
        self.assertEqual(info["motif_residues_in_binding"], 0)

    def test_rnp1_partial_coverage(self):
        """4 of 8 motif residues covered → score = 0.5."""
        seq = "AAAA" + "KGFGFVKF" + "AAAAAAAA"
        # Cover positions 5,6,7,8 → 4 out of 8.
        score, info = known_motif_consistency(
            [5, 6, 7, 8], seq, "RRM_x_stem_loop", {},
            _default_cfg(),
        )
        self.assertEqual(score, 0.5)

    def test_rnp2_alone(self):
        """Only RNP2 matches (no RNP1) → score depends on overlap."""
        # RNP2 = [ILV][FY][ILV].NL — 'IFINL' has I-F-I-N-L wait need 6 chars
        # [ILV][FY][ILV] x N L → I F I X N L (6 residues)
        # Use IFIANL at positions 3-8.
        seq = "AA" + "IFIANL" + "AA"  # length 10
        # Only RNP2 at 3-8 matches.
        score, info = known_motif_consistency(
            list(range(3, 9)), seq, "RRM_x_stem_loop", {},
            _default_cfg(),
        )
        self.assertEqual(score, 1.0)
        # Should be just RNP2 hit.
        names = [h["name"] for h in info["motif_hits"]]
        self.assertIn("RNP2", names)

    def test_no_rrm_motif_returns_neutral(self):
        """RRM declared but no motif present → neutral 0.5."""
        seq = "AAAAAAAAAAAAAAAA"  # 16 aa, none match
        score, info = known_motif_consistency(
            [1, 2, 3], seq, "RRM_x_stem_loop", {},
            _default_cfg(),
        )
        self.assertEqual(score, 0.5)
        self.assertEqual(info["n_motifs_found"], 0)


# ============================================================================
# KH domain (GXXG loop)
# ============================================================================


class TestKH(unittest.TestCase):
    def test_kh_gxxg_full_coverage(self):
        """GAAG at positions 5-8; all in binding → score = 1.0."""
        seq = "AAAA" + "GAAG" + "AAAA"  # length 12
        score, info = known_motif_consistency(
            [5, 6, 7, 8], seq, "KH_x_stem_loop", {},
            _default_cfg(),
        )
        self.assertEqual(score, 1.0)
        self.assertEqual(info["n_motifs_found"], 1)

    def test_kh_partial_coverage(self):
        """2 of 4 GXXG residues in binding → 0.5."""
        seq = "AAAA" + "GAAG" + "AAAA"
        score, info = known_motif_consistency(
            [5, 6], seq, "KH_x_stem_loop", {}, _default_cfg(),
        )
        self.assertEqual(score, 0.5)

    def test_kh_no_motif_returns_neutral(self):
        score, info = known_motif_consistency(
            [1, 2], "AAAAAAAA", "KH_x_stem_loop", {}, _default_cfg(),
        )
        self.assertEqual(score, 0.5)
        self.assertEqual(info["n_motifs_found"], 0)


# ============================================================================
# Zinc finger (C2H2)
# ============================================================================


class TestZincFinger(unittest.TestCase):
    def test_c2h2_match(self):
        """Synthetic C2H2: C-AA-C-(12x A)-H-AAA-H spans 21 residues.

        Positions: C@1, C@4, H@17, H@21 → length 21.
        Verify pattern matches and we score the overlap correctly.
        """
        # Pattern requires C.{2,4}C.{12}H.{3,5}H
        # Build: C AA C AAAAAAAAAAAA H AAA H = 1+2+1+12+1+3+1 = 21 chars
        seq = "C" + "AA" + "C" + "A" * 12 + "H" + "AAA" + "H"
        self.assertEqual(len(seq), 21)
        # Cover all 21 → score=1.0
        score, info = known_motif_consistency(
            list(range(1, 22)), seq, "zinc_finger_x_single_stranded", {},
            _default_cfg(),
        )
        self.assertEqual(info["n_motifs_found"], 1)
        self.assertEqual(score, 1.0)

    def test_c2h2_no_match(self):
        """Sequence with isolated C/H but not in C2H2 spacing → no match."""
        seq = "C" + "A" * 50 + "H"
        score, info = known_motif_consistency(
            [1, 52], seq, "zinc_finger_x_single_stranded", {},
            _default_cfg(),
        )
        self.assertEqual(score, 0.5)  # neutral — pattern doesn't match
        self.assertEqual(info["n_motifs_found"], 0)


# ============================================================================
# DEAD-box motif
# ============================================================================


class TestDeadBox(unittest.TestCase):
    def test_dead_motif_match(self):
        seq = "AAAA" + "DEAD" + "AAAA"  # length 12
        score, info = known_motif_consistency(
            [5, 6, 7, 8], seq, "DEAD_box_x_single_stranded", {},
            _default_cfg(),
        )
        self.assertEqual(score, 1.0)
        self.assertEqual(info["n_motifs_found"], 1)

    def test_deah_variant_match(self):
        seq = "AAAA" + "DEAH" + "AAAA"
        score, info = known_motif_consistency(
            [5, 6, 7, 8], seq, "DEAD_box_x_single_stranded", {},
            _default_cfg(),
        )
        self.assertEqual(score, 1.0)


# ============================================================================
# Multi-domain
# ============================================================================


class TestMultiDomain(unittest.TestCase):
    def test_multi_domain_scans_all_motif_sets(self):
        """multi_domain protein with both an RNP1 and a GXXG loop."""
        # RNP1 at 1-8 (KGFGFVKF), GAAG at 12-15.
        seq = "KGFGFVKF" + "AAA" + "GAAG" + "AAA"
        # Length: 8 + 3 + 4 + 3 = 18
        # Bind all 8 of RNP1 + all 4 of GXXG → 12 / 12 = 1.0
        score, info = known_motif_consistency(
            list(range(1, 9)) + [12, 13, 14, 15], seq,
            "multi_domain_x_stem_loop", {}, _default_cfg(),
        )
        self.assertEqual(score, 1.0)
        names = sorted(set(h["name"] for h in info["motif_hits"]))
        self.assertIn("RNP1", names)
        self.assertIn("GXXG", names)

    def test_multi_domain_partial_coverage(self):
        """Cover RNP1 fully but skip GXXG → coverage = 8/12 ≈ 0.667."""
        seq = "KGFGFVKF" + "AAA" + "GAAG" + "AAA"
        score, info = known_motif_consistency(
            list(range(1, 9)), seq,
            "multi_domain_x_stem_loop", {}, _default_cfg(),
        )
        self.assertAlmostEqual(score, 8 / 12, places=4)


# ============================================================================
# Bare domain & out-of-range filtering
# ============================================================================


class TestParsing(unittest.TestCase):
    def test_bare_domain_label_accepted(self):
        """Category 'RRM' (no _x_) is parsed as domain RRM."""
        seq = "AAAA" + "KGFGFVKF" + "AAAA"
        score, info = known_motif_consistency(
            list(range(5, 13)), seq, "RRM", {}, _default_cfg(),
        )
        self.assertEqual(info["protein_domain"], "RRM")
        self.assertEqual(score, 1.0)

    def test_out_of_range_dropped_before_overlap(self):
        """Indices > len(seq) excluded; overlap counted on in-range only."""
        seq = "AAAA" + "KGFGFVKF" + "AAAA"  # length 16
        # 100 is out of range; 5..12 cover the whole motif.
        score, info = known_motif_consistency(
            list(range(5, 13)) + [100], seq, "RRM_x_stem_loop", {},
            _default_cfg(),
        )
        self.assertEqual(info["n_out_of_range"], 1)
        self.assertEqual(score, 1.0)


# ============================================================================
# MOTIFS table sanity
# ============================================================================


class TestMotifsTable(unittest.TestCase):
    def test_all_known_domains_have_at_least_one_motif(self):
        for domain in ("RRM", "KH", "zinc_finger", "dsRBD", "PUF", "DEAD_box"):
            self.assertIn(domain, MOTIFS)
            self.assertGreaterEqual(len(MOTIFS[domain]), 1)

    def test_all_motif_patterns_compile(self):
        import re
        for domain, pats in MOTIFS.items():
            for name, pattern in pats:
                # Must compile without raising.
                re.compile(pattern)


# ============================================================================
# Scorer integration
# ============================================================================


class TestScorerIntegration(unittest.TestCase):
    def test_metric_detail_via_scorer(self):
        from step6_pocket_qa.scorer import _safe_metric_call

        seq = "AAAA" + "KGFGFVKF" + "AAAA"
        detail = _safe_metric_call(
            known_motif_consistency,
            binding_residues=list(range(5, 13)),
            protein_sequence=seq,
            target_category="RRM_x_stem_loop",
            sample_json={},
            config=_default_cfg(),
        )
        self.assertTrue(detail.computed)
        self.assertEqual(detail.score, 1.0)
        self.assertIsNone(detail.error)

    def test_neutral_via_scorer(self):
        from step6_pocket_qa.scorer import _safe_metric_call

        detail = _safe_metric_call(
            known_motif_consistency,
            binding_residues=[1, 2, 3],
            protein_sequence="KKK",
            target_category="novel_fold_x_unstructured",
            sample_json={},
            config=_default_cfg(),
        )
        self.assertTrue(detail.computed)
        self.assertEqual(detail.score, 0.5)

    def test_abstain_via_scorer(self):
        from step6_pocket_qa.scorer import _safe_metric_call

        detail = _safe_metric_call(
            known_motif_consistency,
            binding_residues=[],
            protein_sequence="MKR",
            target_category="RRM_x_stem_loop",
            sample_json={},
            config=_default_cfg(),
        )
        self.assertFalse(detail.computed)
        self.assertIsNone(detail.score)


if __name__ == "__main__":
    unittest.main()
