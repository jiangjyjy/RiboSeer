"""Unit tests for step6 structural_plausibility (q1).

Hand-calculated reference values are stated in each docstring so the
math can be checked by inspection. Compactness tests build a tiny
synthetic PDB in a tempdir so the suite stays self-contained.
"""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step6_pocket_qa.metrics.structural import (  # noqa: E402
    _CA_BOND_ANGSTROM,
    _compactness_score,
    _continuity_score,
    _radius_of_gyration,
    _ratio_score,
    count_sequence_clusters,
    structural_plausibility,
)


# --------------------------- helpers ----------------------------------------


def _default_cfg(**overrides) -> dict:
    base = {
        "expected_interface_ratio": [0.1, 0.4],
        "max_clusters": 3,
        "cluster_gap": 5,
    }
    base.update(overrides)
    return base


def _atom_record(
    serial: int, name: str, res_name: str, chain: str, res_seq: int,
    x: float, y: float, z: float, element: str,
) -> str:
    """Format a single PDB v3.3 ATOM line."""
    return (
        f"ATOM  {serial:5d}  {name:<3s} {res_name:<3s} {chain:1s}"
        f"{res_seq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00 50.00          {element:>2s}\n"
    )


def _write_pdb(
    path: Path,
    ca_coords: dict[int, tuple[float, float, float]],
    chain: str = "A",
) -> None:
    """Write a PDB containing one CA per residue at the given coords.

    Residues are emitted as ALA so gemmi classifies the chain as
    protein. Author seq numbers start at the smallest key; gemmi's
    ``assign_label_seq_id(True)`` then aligns ``label_seq`` to the
    polymer index (1-based, gap-free), which is what step 5's binding
    indices use.
    """
    lines = ["HEADER    TEST\n"]
    serial = 1
    for res_seq in sorted(ca_coords):
        x, y, z = ca_coords[res_seq]
        # Need at least N + CA + C so gemmi treats it as a polymer
        # residue and assigns a label_seq.
        lines.append(_atom_record(
            serial,     "N",  "ALA", chain, res_seq, x - 1.0, y, z, "N",
        ))
        lines.append(_atom_record(
            serial + 1, "CA", "ALA", chain, res_seq, x,       y, z, "C",
        ))
        lines.append(_atom_record(
            serial + 2, "C",  "ALA", chain, res_seq, x + 1.0, y, z, "C",
        ))
        serial += 3
    lines.append("TER\n")
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


# ============================================================================
# count_sequence_clusters / continuity sub-score
# ============================================================================


class TestContinuity(unittest.TestCase):
    def test_single_run_one_cluster(self):
        """Consecutive residues with no gaps → 1 cluster."""
        self.assertEqual(count_sequence_clusters([10, 11, 12, 13]), 1)

    def test_gap_below_threshold_still_one_cluster(self):
        """Gap of exactly 5 (= threshold) keeps the cluster intact."""
        self.assertEqual(count_sequence_clusters([10, 15], gap_threshold=5), 1)

    def test_gap_above_threshold_splits(self):
        """Gap of 6 > threshold of 5 → two clusters."""
        self.assertEqual(count_sequence_clusters([10, 16], gap_threshold=5), 2)

    def test_scattered_residues_many_clusters(self):
        """Spread-out residues split into one cluster each."""
        self.assertEqual(count_sequence_clusters([1, 20, 50, 80]), 4)

    def test_unsorted_and_duplicate_input(self):
        """Function sorts and dedups internally."""
        self.assertEqual(
            count_sequence_clusters([13, 10, 11, 11, 12]),
            1,
        )

    def test_empty_returns_zero(self):
        self.assertEqual(count_sequence_clusters([]), 0)

    def test_continuity_score_under_max(self):
        """1, 2, 3 clusters all give 1.0 under default max_clusters=3."""
        self.assertEqual(_continuity_score(1, 3), 1.0)
        self.assertEqual(_continuity_score(2, 3), 1.0)
        self.assertEqual(_continuity_score(3, 3), 1.0)

    def test_continuity_score_decays(self):
        """4 clusters → 1/(1+1)=0.5; 6 clusters → 1/(1+3)=0.25."""
        self.assertAlmostEqual(_continuity_score(4, 3), 0.5, places=6)
        self.assertAlmostEqual(_continuity_score(5, 3), 1/3, places=6)
        self.assertAlmostEqual(_continuity_score(6, 3), 0.25, places=6)


# ============================================================================
# _ratio_score
# ============================================================================


class TestRatioScore(unittest.TestCase):
    def test_inside_band_full_credit(self):
        # Default band [0.1, 0.4]: 0.2 squarely inside.
        self.assertEqual(_ratio_score(0.2, 0.1, 0.4), 1.0)

    def test_band_endpoints_full_credit(self):
        self.assertEqual(_ratio_score(0.1, 0.1, 0.4), 1.0)
        self.assertEqual(_ratio_score(0.4, 0.1, 0.4), 1.0)

    def test_outside_below_falloff(self):
        """ratio=0.05; centre=0.25, scale=0.25 →
        score = 1 - |0.05-0.25|/0.25 = 1 - 0.8 = 0.2.
        """
        self.assertAlmostEqual(_ratio_score(0.05, 0.1, 0.4), 0.2, places=6)

    def test_outside_above_clamped_to_zero(self):
        """ratio=0.9; |0.9-0.25|/0.25 = 2.6 → max(0, 1-2.6)=0."""
        self.assertEqual(_ratio_score(0.9, 0.1, 0.4), 0.0)

    def test_zero_ratio(self):
        """ratio=0 with default centre 0.25 → 1 - 1.0 = 0.0."""
        self.assertEqual(_ratio_score(0.0, 0.1, 0.4), 0.0)

    def test_custom_band(self):
        """Band [0.2, 0.6]: centre=0.4, scale=0.4.
        ratio=0.1 → 1 - 0.3/0.4 = 0.25.
        """
        self.assertAlmostEqual(_ratio_score(0.1, 0.2, 0.6), 0.25, places=6)


# ============================================================================
# Radius of gyration / compactness sub-score
# ============================================================================


class TestRgAndCompactness(unittest.TestCase):
    def test_rg_single_point(self):
        self.assertEqual(_radius_of_gyration([(1.0, 2.0, 3.0)]), 0.0)

    def test_rg_square_4_points(self):
        """4 CAs at corners of a 4 Å square in xy plane.

        Centroid = (2,2,0); each squared distance = 4+4 = 8;
        msd = 8; Rg = sqrt(8) ≈ 2.828427.
        """
        coords = [(0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0)]
        self.assertAlmostEqual(
            _radius_of_gyration(coords), math.sqrt(8.0), places=6,
        )

    def test_compactness_at_expected_is_half(self):
        """Rg = expected → score = 1/(1+1) = 0.5."""
        n = 9  # expected = 3 * 3.8 = 11.4
        self.assertAlmostEqual(_compactness_score(11.4, n), 0.5, places=6)

    def test_compactness_zero_rg_full_score(self):
        self.assertEqual(_compactness_score(0.0, 5), 1.0)

    def test_compactness_score_for_square(self):
        """4-point square: Rg≈2.828, expected=sqrt(4)*3.8=7.6
        score = 1 / (1 + 2.828/7.6) ≈ 1 / 1.3722 ≈ 0.7290.
        """
        rg = math.sqrt(8.0)
        s = _compactness_score(rg, 4)
        self.assertAlmostEqual(s, 1.0 / (1.0 + rg / (2.0 * 3.8)), places=6)


# ============================================================================
# Abstain (None) cases
# ============================================================================


class TestAbstain(unittest.TestCase):
    def test_empty_binding_returns_none(self):
        score, info = structural_plausibility([], None, 100, _default_cfg())
        self.assertIsNone(score)
        self.assertIn("empty", info["reason"])

    def test_zero_protein_length_returns_none(self):
        score, info = structural_plausibility(
            [10, 11], None, 0, _default_cfg(),
        )
        self.assertIsNone(score)
        self.assertIn("protein_length", info["reason"])

    def test_negative_protein_length_returns_none(self):
        score, info = structural_plausibility(
            [10], None, -5, _default_cfg(),
        )
        self.assertIsNone(score)

    def test_all_residues_out_of_range_returns_none(self):
        # protein_length=10, but every binding residue > 10.
        score, info = structural_plausibility(
            [50, 60, 70], None, 10, _default_cfg(),
        )
        self.assertIsNone(score)
        self.assertEqual(info["n_out_of_range"], 3)
        self.assertEqual(info["n_in_range"], 0)

    def test_residue_zero_treated_as_out_of_range(self):
        """1-based indexing: residue 0 must NOT match position 0 in the
        sequence array. With only res=0 → all out of range → abstain.
        """
        score, info = structural_plausibility(
            [0], None, 100, _default_cfg(),
        )
        self.assertIsNone(score)
        self.assertEqual(info["n_out_of_range"], 1)


# ============================================================================
# Sequence-only path (structure_path=None)
# ============================================================================


class TestSequenceOnly(unittest.TestCase):
    def test_perfect_continuity_and_ratio(self):
        """One contiguous cluster of 25 in a 100-aa protein.

        n_clusters=1 → continuity=1.0
        ratio=25/100=0.25 → in [0.1,0.4] → 1.0
        compactness skipped (no structure)
        q1 = (1.0 + 1.0) / 2 = 1.0
        """
        binding = list(range(40, 65))
        score, info = structural_plausibility(
            binding, None, 100, _default_cfg(),
        )
        self.assertAlmostEqual(score, 1.0, places=6)
        self.assertEqual(info["n_clusters"], 1)
        self.assertEqual(info["interface_ratio"], 0.25)
        self.assertIsNone(info["sub_scores"]["compactness"])
        self.assertEqual(
            info["compactness_skip_reason"], "no structure_path provided",
        )
        self.assertEqual(info["n_sub_scores_computed"], 2)

    def test_scattered_residues_low_continuity(self):
        """4 isolated residues spread across a 100-aa protein.

        n_clusters=4 → continuity = 1/(1+1)=0.5
        ratio=4/100=0.04 (outside) → 1 - |0.04-0.25|/0.25 = 0.16
        q1 = (0.5 + 0.16) / 2 = 0.33
        """
        score, info = structural_plausibility(
            [1, 30, 60, 90], None, 100, _default_cfg(),
        )
        self.assertEqual(info["n_clusters"], 4)
        self.assertAlmostEqual(info["sub_scores"]["continuity"], 0.5, places=6)
        self.assertAlmostEqual(info["sub_scores"]["ratio"], 0.16, places=6)
        self.assertAlmostEqual(score, (0.5 + 0.16) / 2, places=6)

    def test_extreme_high_ratio(self):
        """Binding covers 90 % of the protein → ratio falls off to 0."""
        binding = list(range(1, 91))
        score, info = structural_plausibility(
            binding, None, 100, _default_cfg(),
        )
        self.assertEqual(info["sub_scores"]["ratio"], 0.0)
        # 1 cluster → continuity 1.0; (1+0)/2 = 0.5.
        self.assertAlmostEqual(score, 0.5, places=6)


# ============================================================================
# Compactness via PDB
# ============================================================================


class TestCompactnessFromPdb(unittest.TestCase):
    def test_pdb_with_known_coords(self):
        """4 residues at the corners of a 4 Å square + matching binding.

        Rg = sqrt(8) ≈ 2.828; expected = sqrt(4)*3.8 = 7.6
        score_compactness = 1 / (1 + 2.828/7.6) ≈ 0.7290
        n_clusters = 1 (residues 10-13 contiguous), continuity=1.0
        ratio = 4/100 = 0.04 → outside → 0.16
        q1 = (1.0 + 0.7290 + 0.16) / 3 ≈ 0.6297
        """
        with tempfile.TemporaryDirectory() as td:
            pdb = Path(td) / "square.pdb"
            _write_pdb(pdb, {
                10: (0.0, 0.0, 0.0),
                11: (4.0, 0.0, 0.0),
                12: (4.0, 4.0, 0.0),
                13: (0.0, 4.0, 0.0),
            })
            score, info = structural_plausibility(
                [10, 11, 12, 13], str(pdb), 100, _default_cfg(),
            )
        self.assertEqual(info["n_ca_found"], 4)
        rg = math.sqrt(8.0)
        expected_compact = 1.0 / (1.0 + rg / (math.sqrt(4) * 3.8))
        self.assertAlmostEqual(
            info["sub_scores"]["compactness"], round(expected_compact, 4),
            places=4,
        )
        # Three sub-scores should now be averaged.
        self.assertEqual(info["n_sub_scores_computed"], 3)
        # Hand-calc total: (1.0 + expected_compact + 0.16) / 3.
        expected_q = (1.0 + expected_compact + 0.16) / 3.0
        self.assertAlmostEqual(score, round(expected_q, 6), places=4)

    def test_missing_binding_residue_skipped_not_failed(self):
        """Binding asks for residues 10, 11, 12 but PDB only has 10, 11.

        n_ca_found=2 (still >= 2) → compactness still computed.
        n_ca_missing should record the unfound residue.
        """
        with tempfile.TemporaryDirectory() as td:
            pdb = Path(td) / "two.pdb"
            _write_pdb(pdb, {
                10: (0.0, 0.0, 0.0),
                11: (3.8, 0.0, 0.0),
            })
            score, info = structural_plausibility(
                [10, 11, 12], str(pdb), 100, _default_cfg(),
            )
        self.assertEqual(info["n_ca_found"], 2)
        self.assertEqual(info["n_ca_missing"], 1)
        self.assertIn(12, info["missing_residues"])
        self.assertIsNotNone(info["sub_scores"]["compactness"])

    def test_too_few_ca_skips_compactness(self):
        """Only 1 binding residue maps to a CA → compactness skipped.

        Note: the file *does* have CAs for residues 10 and 11, but
        binding only asks for residue 10. Continuity + ratio still
        compute, so q1 is the 2-axis mean.
        """
        with tempfile.TemporaryDirectory() as td:
            pdb = Path(td) / "two.pdb"
            _write_pdb(pdb, {
                10: (0.0, 0.0, 0.0),
                11: (3.8, 0.0, 0.0),
            })
            score, info = structural_plausibility(
                [10], str(pdb), 100, _default_cfg(),
            )
        self.assertIsNotNone(score)
        self.assertEqual(info["n_ca_found"], 1)
        self.assertIsNone(info["sub_scores"]["compactness"])
        self.assertIn("need >= 2 CA", info["compactness_skip_reason"])
        self.assertEqual(info["n_sub_scores_computed"], 2)

    def test_missing_file_still_returns_score(self):
        """Bad path → compactness skipped with parse_error in info,
        continuity + ratio still produce a score (no abstain).
        """
        score, info = structural_plausibility(
            [10, 11, 12, 13], "/no/such/file.pdb", 100, _default_cfg(),
        )
        self.assertIsNotNone(score)
        self.assertIsNone(info["sub_scores"]["compactness"])
        self.assertIn("file not found", info["compactness_skip_reason"])


# ============================================================================
# Config wiring
# ============================================================================


class TestConfigWiring(unittest.TestCase):
    def test_custom_max_clusters(self):
        """With max_clusters=1, even 2 clusters costs score.

        n_clusters=2 → continuity = 1/(1+1) = 0.5
        ratio=2/100 = 0.02 → outside → 1 - 0.23/0.25 = 0.08
        q1 = (0.5 + 0.08)/2 = 0.29
        """
        score, info = structural_plausibility(
            [10, 50], None, 100, _default_cfg(max_clusters=1),
        )
        self.assertEqual(info["n_clusters"], 2)
        self.assertAlmostEqual(info["sub_scores"]["continuity"], 0.5, places=6)
        self.assertAlmostEqual(score, (0.5 + 0.08) / 2, places=6)

    def test_custom_cluster_gap(self):
        """gap=10 keeps [10, 20] in one cluster (gap=10 == threshold)."""
        score, info = structural_plausibility(
            [10, 20], None, 100, _default_cfg(cluster_gap=10),
        )
        self.assertEqual(info["n_clusters"], 1)

    def test_custom_interface_band(self):
        """Custom band [0.05, 0.5] makes ratio=0.04 outside but close.

        centre=0.275, scale=0.275; |0.04-0.275|/0.275 = 0.8545
        score_ratio = 1 - 0.8545 = 0.1455
        """
        cfg = _default_cfg(expected_interface_ratio=[0.05, 0.5])
        score, info = structural_plausibility(
            [1, 2, 3, 4], None, 100, cfg,
        )
        self.assertAlmostEqual(
            info["sub_scores"]["ratio"], 0.1455, places=3,
        )

    def test_empty_config_uses_defaults(self):
        """Pass {} → defaults [0.1,0.4] / max=3 / gap=5 still apply."""
        score, info = structural_plausibility(
            list(range(20, 41)), None, 100, {},
        )
        # 21 residues out of 100 → 0.21, inside default band.
        self.assertEqual(info["interface_ratio"], 0.21)
        self.assertEqual(info["sub_scores"]["ratio"], 1.0)
        # 1 cluster → continuity 1.0.
        self.assertAlmostEqual(score, 1.0, places=6)


# ============================================================================
# Scorer integration smoke
# ============================================================================


class TestScorerIntegration(unittest.TestCase):
    def test_safe_metric_call_wraps_score(self):
        """Run via the scorer's safe wrapper to confirm MetricDetail
        is populated correctly (computed=True, score in [0,1]).
        """
        from step6_pocket_qa.scorer import _safe_metric_call

        detail = _safe_metric_call(
            structural_plausibility,
            binding_residues=list(range(40, 65)),
            structure_path=None,
            protein_length=100,
            config=_default_cfg(),
        )
        self.assertTrue(detail.computed)
        self.assertAlmostEqual(detail.score, 1.0, places=6)
        self.assertIsNone(detail.error)

    def test_safe_metric_call_handles_abstain(self):
        from step6_pocket_qa.scorer import _safe_metric_call

        detail = _safe_metric_call(
            structural_plausibility,
            binding_residues=[],
            structure_path=None,
            protein_length=100,
            config=_default_cfg(),
        )
        self.assertFalse(detail.computed)
        self.assertIsNone(detail.score)
        self.assertIsNone(detail.error)
        self.assertIn("empty", detail.info["reason"])


if __name__ == "__main__":
    unittest.main()
