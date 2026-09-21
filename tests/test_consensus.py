"""Unit tests for step6 cross_tool_consensus (q4).

Hand-calculation reference values are stated in each docstring so a
future reader can verify the math by inspection without re-running.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step4_tool_adapters.schemas import ToolPrediction  # noqa: E402
from step6_pocket_qa.metrics.consensus import cross_tool_consensus  # noqa: E402


# --------------------------- helpers ----------------------------------------


def _pred(
    tool_id: str,
    *,
    category: str = "C",
    success: bool = True,
    binding_protein: list[int] | None = None,
    per_residue: dict[int, float] | None = None,
) -> ToolPrediction:
    if not success:
        return ToolPrediction(
            tool_id=tool_id, category=category, sample_id="s",
            success=False, error_message=f"{tool_id} synthetic failure",
        )
    # Need at least one prediction field populated for success=True.
    return ToolPrediction(
        tool_id=tool_id, category=category, sample_id="s",
        success=True,
        binding_protein_residues=binding_protein,
        per_residue_confidence=(per_residue
                                or ({i: 0.8 for i in (binding_protein or [])}
                                    or {1: 0.5})),
    )


def _default_cfg(**overrides) -> dict:
    base = {"min_tools": 2, "vote_weight": 0.6, "jaccard_weight": 0.4}
    base.update(overrides)
    return base


# --------------------------- active-set filtering ---------------------------


class TestActiveFiltering(unittest.TestCase):
    def test_zero_active_returns_none(self):
        # All tools failed → metric abstains.
        preds = [
            _pred("boltz2", success=False),
            _pred("equipnas", success=False),
        ]
        score, info = cross_tool_consensus([10, 11], preds, _default_cfg())
        self.assertIsNone(score)
        self.assertEqual(info["n_active_tools"], 0)
        self.assertEqual(info["skipped_tools"]["boltz2"], "success=False")

    def test_failed_tool_skipped_but_others_counted(self):
        preds = [
            _pred("boltz2", binding_protein=[10, 11]),
            _pred("equipnas", success=False),
            _pred("rosettafold2na", binding_protein=[10, 11]),
        ]
        score, info = cross_tool_consensus([10, 11], preds, _default_cfg())
        self.assertIsNotNone(score)
        self.assertEqual(info["n_active_tools"], 2)
        self.assertIn("equipnas", info["skipped_tools"])

    def test_empty_binding_list_skipped(self):
        # EquiPNAS-style: success=True but every prob fell below threshold,
        # so binding_protein_residues=[]. Must be skipped (not dilute Jaccard).
        # We pass per_residue with content so the schema accepts success=True.
        preds = [
            _pred("boltz2", binding_protein=[10, 11]),
            _pred("equipnas", binding_protein=[],
                  per_residue={1: 0.2, 2: 0.3}),
            _pred("rosettafold2na", binding_protein=[10, 11]),
        ]
        score, info = cross_tool_consensus([10, 11], preds, _default_cfg())
        self.assertEqual(info["n_active_tools"], 2)
        self.assertEqual(info["skipped_tools"]["equipnas"],
                         "empty binding_protein_residues")

    def test_none_binding_skipped(self):
        # Tool with binding_protein_residues=None (e.g. RNA-only Cat A
        # output). Schema accepts it (per_residue_confidence carries the
        # signal). Consensus should skip it on the protein side.
        preds = [
            _pred("boltz2", binding_protein=[10, 11]),
            _pred("rna_only_tool", binding_protein=None,
                  per_residue={5: 0.7}),
            _pred("equipnas", binding_protein=[10, 11]),
        ]
        score, info = cross_tool_consensus([10, 11], preds, _default_cfg())
        self.assertEqual(info["n_active_tools"], 2)
        self.assertIn("rna_only_tool", info["skipped_tools"])

    def test_single_active_returns_neutral(self):
        # Only one tool active → consensus is undefined; spec says 0.5.
        preds = [
            _pred("boltz2", binding_protein=[10, 11]),
            _pred("equipnas", success=False),
        ]
        score, info = cross_tool_consensus([10, 11], preds, _default_cfg())
        self.assertEqual(score, 0.5)
        self.assertEqual(info["n_active_tools"], 1)
        self.assertIn("neutral", info["reason"])


# --------------------------- agreement scenarios ----------------------------


class TestAgreementScenarios(unittest.TestCase):
    def test_three_tools_full_agreement(self):
        """All three tools predict the same set as the composite.

        composite = {10, 11, 23}; every tool predicts {10, 11, 23}.
        vote_count[i] = 3 for each i; avg_vote_ratio = 9 / (3*3) = 1.0
        Each pairwise Jaccard = 3/3 = 1.0; avg_jaccard = 1.0
        score = 0.6 * 1.0 + 0.4 * 1.0 = 1.0
        """
        preds = [
            _pred("boltz2", binding_protein=[10, 11, 23]),
            _pred("equipnas", binding_protein=[10, 11, 23]),
            _pred("rosettafold2na", binding_protein=[10, 11, 23]),
        ]
        score, info = cross_tool_consensus(
            [10, 11, 23], preds, _default_cfg(),
        )
        self.assertAlmostEqual(score, 1.0, places=6)
        self.assertEqual(info["avg_vote_ratio"], 1.0)
        self.assertEqual(info["avg_jaccard"], 1.0)
        # All three pairs present.
        self.assertEqual(len(info["pairwise_jaccard"]), 3)
        self.assertEqual(info["vote_count"], {10: 3, 11: 3, 23: 3})

    def test_three_tools_full_disagreement(self):
        """Three disjoint tools — composite is union (or whatever fusion
        emitted); vote count is 1 per residue, Jaccard is 0 per pair.

        T1={10,11}, T2={20,21}, T3={30,31}; composite = {10, 20, 30}
        vote_count = {10:1, 20:1, 30:1}
        avg_vote_ratio = 3 / (3*3) = 1/3 ≈ 0.3333
        Jaccard for every pair = 0 / 4 = 0
        avg_jaccard = 0
        score = 0.6 * 1/3 + 0.4 * 0 = 0.2
        """
        preds = [
            _pred("boltz2", binding_protein=[10, 11]),
            _pred("equipnas", binding_protein=[20, 21]),
            _pred("rosettafold2na", binding_protein=[30, 31]),
        ]
        score, info = cross_tool_consensus(
            [10, 20, 30], preds, _default_cfg(),
        )
        self.assertAlmostEqual(score, 0.2, places=6)
        self.assertAlmostEqual(info["avg_vote_ratio"], 1 / 3, places=4)
        self.assertEqual(info["avg_jaccard"], 0.0)

    def test_two_agree_one_disagree(self):
        """Two tools predict the same {10,11,12}; third predicts {50,51,52}.
        composite = {10, 11, 12} (the 2-tool consensus).

        vote_count = {10:2, 11:2, 12:2}; avg_vote_ratio = 6 / (3*3) = 0.6667
        Jaccard:
          T1-T2: 3/3 = 1.0
          T1-T3: 0/6 = 0.0
          T2-T3: 0/6 = 0.0
          avg = 1/3 ≈ 0.3333
        score = 0.6 * 0.6667 + 0.4 * 0.3333 ≈ 0.4 + 0.1333 ≈ 0.5333
        """
        preds = [
            _pred("boltz2", binding_protein=[10, 11, 12]),
            _pred("equipnas", binding_protein=[10, 11, 12]),
            _pred("rosettafold2na", binding_protein=[50, 51, 52]),
        ]
        score, info = cross_tool_consensus(
            [10, 11, 12], preds, _default_cfg(),
        )
        self.assertAlmostEqual(score, 0.6 * (2/3) + 0.4 * (1/3), places=6)
        self.assertAlmostEqual(info["avg_vote_ratio"], 2/3, places=4)
        self.assertAlmostEqual(info["avg_jaccard"], 1/3, places=4)
        self.assertEqual(
            info["pairwise_jaccard"]["boltz2|equipnas"], 1.0,
        )
        self.assertEqual(
            info["pairwise_jaccard"]["boltz2|rosettafold2na"], 0.0,
        )

    def test_partial_overlap_two_tools(self):
        """Two-tool partial overlap — hand-calc.

        T1 = {1, 2, 3, 4}; T2 = {3, 4, 5, 6}; composite = {3, 4}
        vote_count = {3:2, 4:2}; avg_vote_ratio = 4 / (2*2) = 1.0
        Jaccard T1-T2 = |{3,4}| / |{1..6}| = 2/6 ≈ 0.3333
        score = 0.6 * 1.0 + 0.4 * 0.3333 ≈ 0.7333
        """
        preds = [
            _pred("boltz2", binding_protein=[1, 2, 3, 4]),
            _pred("equipnas", binding_protein=[3, 4, 5, 6]),
        ]
        score, info = cross_tool_consensus([3, 4], preds, _default_cfg())
        self.assertAlmostEqual(score, 0.6 + 0.4 * (2/6), places=6)
        self.assertAlmostEqual(info["avg_vote_ratio"], 1.0, places=6)
        self.assertAlmostEqual(info["avg_jaccard"], 2/6, places=4)


# --------------------------- config wiring ---------------------------------


class TestConfigWiring(unittest.TestCase):
    def test_custom_vote_jaccard_weights(self):
        # Same scenario as test_two_agree_one_disagree, but flipped weights.
        # avg_vote_ratio = 2/3, avg_jaccard = 1/3
        # Custom: 0.2 * 2/3 + 0.8 * 1/3 = 0.1333 + 0.2667 = 0.4
        preds = [
            _pred("boltz2", binding_protein=[10, 11, 12]),
            _pred("equipnas", binding_protein=[10, 11, 12]),
            _pred("rosettafold2na", binding_protein=[50, 51, 52]),
        ]
        cfg = _default_cfg(vote_weight=0.2, jaccard_weight=0.8)
        score, info = cross_tool_consensus([10, 11, 12], preds, cfg)
        self.assertAlmostEqual(score, 0.2 * (2/3) + 0.8 * (1/3), places=6)
        self.assertEqual(info["weights"]["vote_weight"], 0.2)
        self.assertEqual(info["weights"]["jaccard_weight"], 0.8)

    def test_min_tools_three_falls_to_neutral_with_two(self):
        # Two active tools but config requires at least 3 → neutral 0.5.
        preds = [
            _pred("boltz2", binding_protein=[10, 11]),
            _pred("equipnas", binding_protein=[10, 11]),
        ]
        cfg = _default_cfg(min_tools=3)
        score, info = cross_tool_consensus([10, 11], preds, cfg)
        self.assertEqual(score, 0.5)
        self.assertEqual(info["n_active_tools"], 2)
        self.assertEqual(info["min_tools"], 3)

    def test_default_config_when_keys_missing(self):
        # Empty config → defaults (min_tools=2, weights 0.6 / 0.4).
        preds = [
            _pred("boltz2", binding_protein=[10, 11, 12]),
            _pred("equipnas", binding_protein=[10, 11, 12]),
        ]
        score, info = cross_tool_consensus([10, 11, 12], preds, {})
        # Full agreement of 2 tools → score = 1.0
        self.assertAlmostEqual(score, 1.0, places=6)
        self.assertEqual(info["weights"]["vote_weight"], 0.6)
        self.assertEqual(info["weights"]["jaccard_weight"], 0.4)


# --------------------------- corner cases ----------------------------------


class TestCornerCases(unittest.TestCase):
    def test_composite_empty_with_two_tools(self):
        """Composite is empty but both tools have predictions.

        avg_vote_ratio is defined as 0.0 (sum over empty set = 0).
        Jaccard is still computed over the tools' own sets.
        T1={10,11}, T2={11,12}; Jaccard = 1/3
        score = 0.6 * 0.0 + 0.4 * (1/3) ≈ 0.1333
        """
        preds = [
            _pred("boltz2", binding_protein=[10, 11]),
            _pred("equipnas", binding_protein=[11, 12]),
        ]
        score, info = cross_tool_consensus([], preds, _default_cfg())
        self.assertAlmostEqual(score, 0.4 * (1/3), places=6)
        self.assertEqual(info["avg_vote_ratio"], 0.0)
        self.assertAlmostEqual(info["avg_jaccard"], 1/3, places=4)
        self.assertEqual(info["n_composite_residues"], 0)

    def test_residue_outside_tool_predictions_zero_votes(self):
        """If composite contains a residue NO active tool predicted, its
        vote_count is 0 and pulls the average down.

        T1={10,11}, T2={11,12}; composite={10,11,12,99}
        vote_count = {10:1, 11:2, 12:1, 99:0}
        avg_vote_ratio = (1+2+1+0) / (2*4) = 4/8 = 0.5
        Jaccard T1-T2 = 1/3 ≈ 0.3333
        score = 0.6 * 0.5 + 0.4 * 1/3 = 0.3 + 0.1333 ≈ 0.4333
        """
        preds = [
            _pred("boltz2", binding_protein=[10, 11]),
            _pred("equipnas", binding_protein=[11, 12]),
        ]
        score, info = cross_tool_consensus(
            [10, 11, 12, 99], preds, _default_cfg(),
        )
        self.assertAlmostEqual(info["avg_vote_ratio"], 0.5, places=6)
        self.assertEqual(info["vote_count"][99], 0)
        self.assertAlmostEqual(score, 0.3 + 0.4 * (1/3), places=6)

    def test_score_in_unit_interval(self):
        # Defensive: every realistic input must keep score in [0, 1].
        preds = [
            _pred("boltz2", binding_protein=[1, 2, 3, 4, 5]),
            _pred("equipnas", binding_protein=[3, 4, 5, 6, 7]),
            _pred("rosettafold2na", binding_protein=[5, 6, 7, 8, 9]),
        ]
        score, _ = cross_tool_consensus([3, 4, 5, 6, 7], preds, _default_cfg())
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_active_tools_listed_in_input_order(self):
        # info["active_tools"] preserves the iteration order of the input
        # (useful for downstream debugging).
        preds = [
            _pred("zzz", binding_protein=[1]),
            _pred("aaa", binding_protein=[1]),
            _pred("mmm", binding_protein=[1]),
        ]
        _, info = cross_tool_consensus([1], preds, _default_cfg())
        self.assertEqual(info["active_tools"], ["zzz", "aaa", "mmm"])


if __name__ == "__main__":
    unittest.main()
