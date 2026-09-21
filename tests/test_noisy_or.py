"""Unit tests for step5 noisy_or fusion math.

Hand-calculated reference values are stated in the docstrings of each
test so a future reader can verify the noisy-OR formula by inspection
without re-running the asserts.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step4_tool_adapters.schemas import ToolPrediction  # noqa: E402
from step5_fusion.noisy_or import (  # noqa: E402
    _noisy_or_combine,
    _normalize_confidence,
    _protein_residue_probs,
    _rna_nuc_probs,
    equal_weights,
    noisy_or_fusion,
)


# --------------------------- helpers ----------------------------------------


def _make_pred(
    tool_id: str,
    category: str,
    *,
    success: bool = True,
    binding_protein: list[int] | None = None,
    binding_rna: list[int] | None = None,
    per_residue: dict[int, float] | None = None,
) -> ToolPrediction:
    """Construct a minimal ``ToolPrediction`` for fusion tests."""
    return ToolPrediction(
        tool_id=tool_id,
        category=category,
        sample_id="sample0",
        success=success,
        error_message=None if success else "synthetic failure",
        binding_protein_residues=binding_protein,
        binding_rna_nucleotides=binding_rna,
        per_residue_confidence=per_residue,
    )


# --------------------------- normalization ----------------------------------


class TestNormalizeConfidence(unittest.TestCase):
    def test_cat_a_divides_by_100(self):
        self.assertAlmostEqual(_normalize_confidence(85.0, "A"), 0.85)
        self.assertAlmostEqual(_normalize_confidence(0.0, "A"), 0.0)
        self.assertAlmostEqual(_normalize_confidence(100.0, "A"), 1.0)

    def test_cat_b_passthrough(self):
        self.assertAlmostEqual(_normalize_confidence(0.6, "B"), 0.6)

    def test_cat_c_passthrough(self):
        self.assertAlmostEqual(_normalize_confidence(0.92, "C"), 0.92)

    def test_clamp_above_one(self):
        # Defensive: a Cat B/C tool returning 1.5 must not feed >1 into noisy-OR.
        self.assertEqual(_normalize_confidence(1.5, "B"), 1.0)

    def test_clamp_below_zero(self):
        self.assertEqual(_normalize_confidence(-0.3, "C"), 0.0)

    def test_cat_a_overflow_clamped(self):
        # pLDDT > 100 (shouldn't happen but defend).
        self.assertEqual(_normalize_confidence(150.0, "A"), 1.0)


# ---------------------- per-tool probability extraction ---------------------


class TestPerToolExtractors(unittest.TestCase):
    def test_protein_uses_per_residue_when_present(self):
        # binding_protein_residues GATES which residues count.
        # Cat A: residue 12 has pLDDT 30 but is NOT predicted as binding,
        # so it must be omitted (pLDDT is structure quality, not interface
        # probability). Residue 11's pLDDT 90 is used for its binding
        # confidence after /100 normalisation.
        pred = _make_pred(
            "boltz2", "A",
            binding_protein=[10, 11],
            per_residue={10: 80.0, 11: 90.0, 12: 30.0},
        )
        probs = _protein_residue_probs(pred)
        self.assertEqual(set(probs), {10, 11})
        self.assertAlmostEqual(probs[10], 0.80)
        self.assertAlmostEqual(probs[11], 0.90)

    def test_protein_binding_residue_without_per_residue_score(self):
        # Cat A residue listed as binding but missing from per_residue map
        # → fall back to binary 1.0 (the tool said it binds; we just lack
        # a calibrated score for it).
        pred = _make_pred(
            "boltz2", "A",
            binding_protein=[10, 11, 13],
            per_residue={10: 80.0, 11: 90.0},
        )
        probs = _protein_residue_probs(pred)
        self.assertAlmostEqual(probs[10], 0.80)
        self.assertAlmostEqual(probs[11], 0.90)
        self.assertEqual(probs[13], 1.0)

    def test_protein_falls_back_to_binary_indicator(self):
        pred = _make_pred(
            "p2rank", "B",
            binding_protein=[5, 6, 7],
            per_residue=None,
        )
        # Cat B without per-residue probs → binary 1.0 indicators.
        probs = _protein_residue_probs(pred)
        self.assertEqual(probs, {5: 1.0, 6: 1.0, 7: 1.0})

    def test_protein_empty_binding_list_returns_empty(self):
        # Even if per_residue_confidence is populated (e.g. pLDDT for the
        # whole chain), nothing comes through if binding list is empty.
        pred = _make_pred(
            "boltz2", "A",
            binding_protein=[],
            per_residue={1: 95.0, 2: 88.0, 3: 70.0},
        )
        self.assertEqual(_protein_residue_probs(pred), {})

    def test_rna_binary_from_list(self):
        pred = _make_pred(
            "boltz2", "A",
            binding_protein=[1],
            per_residue={1: 75.0},
            binding_rna=[3, 4, 5],
        )
        self.assertEqual(_rna_nuc_probs(pred), {3: 1.0, 4: 1.0, 5: 1.0})

    def test_rna_empty_when_no_predictions(self):
        pred = _make_pred(
            "equipnas", "C",
            binding_protein=[1],
            per_residue={1: 0.7},
            binding_rna=None,
        )
        self.assertEqual(_rna_nuc_probs(pred), {})


# --------------------------- core noisy-OR ----------------------------------


class TestNoisyOrCombine(unittest.TestCase):
    def test_two_tools_three_residues_handcalc(self):
        """Hand-calculation reference.

        Tool 1: c=0.5, b(1)=0.8, b(2)=0.4, b(3)=0.0
        Tool 2: c=0.9, b(1)=0.6, b(2)=0.0, b(3)=0.7

        b_hat(1) = 1 - (1 - 0.5*0.8)*(1 - 0.9*0.6)
                 = 1 - 0.6 * 0.46 = 1 - 0.276 = 0.724
        b_hat(2) = 1 - (1 - 0.5*0.4)*(1 - 0.9*0.0)
                 = 1 - 0.8 * 1.0 = 0.2
        b_hat(3) = 1 - (1 - 0.5*0.0)*(1 - 0.9*0.7)
                 = 1 - 1.0 * 0.37 = 0.63
        """
        pairs = [
            (0.5, {1: 0.8, 2: 0.4, 3: 0.0}),
            (0.9, {1: 0.6, 2: 0.0, 3: 0.7}),
        ]
        fused = _noisy_or_combine(pairs)
        self.assertAlmostEqual(fused[1], 0.724, places=6)
        self.assertAlmostEqual(fused[2], 0.2, places=6)
        self.assertAlmostEqual(fused[3], 0.63, places=6)

    def test_unweighted_single_tool_returns_b_times_c(self):
        pairs = [(0.7, {5: 0.5})]
        fused = _noisy_or_combine(pairs)
        self.assertAlmostEqual(fused[5], 0.35, places=6)

    def test_residue_only_in_one_tool(self):
        # Tool 1 sees residue 1, tool 2 sees residue 2.
        # b_hat(1) = c1 * b1(1) (because tool 2 contributes (1 - 0) = 1)
        # b_hat(2) = c2 * b2(2)
        pairs = [
            (0.6, {1: 0.5}),
            (0.8, {2: 0.9}),
        ]
        fused = _noisy_or_combine(pairs)
        self.assertAlmostEqual(fused[1], 0.6 * 0.5, places=6)
        self.assertAlmostEqual(fused[2], 0.8 * 0.9, places=6)

    def test_zero_b_yields_zero(self):
        pairs = [(0.9, {1: 0.0}), (0.7, {1: 0.0})]
        fused = _noisy_or_combine(pairs)
        self.assertAlmostEqual(fused[1], 0.0, places=6)

    def test_empty_input_returns_empty_dict(self):
        self.assertEqual(_noisy_or_combine([]), {})


# --------------------------- top-level fusion -------------------------------


class TestNoisyOrFusion(unittest.TestCase):
    def test_two_tools_protein_only(self):
        # boltz2 (Cat A): pLDDT 80 / 70 — normalised to 0.8 / 0.7
        # equipnas (Cat C): probs 0.5 / 0.9
        # weights c_b=0.6, c_e=0.4, residues {10, 20}
        # b_hat(10) = 1 - (1 - 0.6*0.8)*(1 - 0.4*0.5)
        #          = 1 - 0.52 * 0.8 = 1 - 0.416 = 0.584
        # b_hat(20) = 1 - (1 - 0.6*0.7)*(1 - 0.4*0.9)
        #          = 1 - 0.58 * 0.64 = 1 - 0.3712 = 0.6288
        preds = [
            _make_pred(
                "boltz2", "A",
                binding_protein=[10, 20],
                per_residue={10: 80.0, 20: 70.0},
            ),
            _make_pred(
                "equipnas", "C",
                binding_protein=[10, 20],
                per_residue={10: 0.5, 20: 0.9},
            ),
        ]
        result = noisy_or_fusion(
            preds, weights={"boltz2": 0.6, "equipnas": 0.4}, threshold=0.5,
        )
        self.assertAlmostEqual(result["per_residue_probability"][10], 0.584, places=6)
        self.assertAlmostEqual(result["per_residue_probability"][20], 0.6288, places=6)
        self.assertEqual(result["binding_protein_residues"], [10, 20])
        self.assertEqual(result["binding_rna_nucleotides"], [])

    def test_all_weights_one_full_propagation(self):
        # weights all 1.0 — degenerate noisy-OR with raw b values.
        # Single residue both tools predict at 0.5 and 0.5:
        # b_hat = 1 - (1 - 0.5)(1 - 0.5) = 1 - 0.25 = 0.75
        preds = [
            _make_pred(
                "boltz2", "A",
                binding_protein=[7], per_residue={7: 50.0},
            ),
            _make_pred(
                "equipnas", "C",
                binding_protein=[7], per_residue={7: 0.5},
            ),
        ]
        result = noisy_or_fusion(
            preds, weights={"boltz2": 1.0, "equipnas": 1.0}, threshold=0.5,
        )
        self.assertAlmostEqual(result["per_residue_probability"][7], 0.75, places=6)
        self.assertEqual(result["binding_protein_residues"], [7])

    def test_single_tool_short_circuit_via_weight_one(self):
        # Single tool with c=1.0 → b_hat(i) == b(i).
        preds = [
            _make_pred(
                "equipnas", "C",
                binding_protein=[3, 4],
                per_residue={3: 0.6, 4: 0.4},
            ),
        ]
        result = noisy_or_fusion(
            preds, weights={"equipnas": 1.0}, threshold=0.5,
        )
        self.assertAlmostEqual(result["per_residue_probability"][3], 0.6, places=6)
        self.assertAlmostEqual(result["per_residue_probability"][4], 0.4, places=6)
        self.assertEqual(result["binding_protein_residues"], [3])

    def test_failed_tool_skipped(self):
        good = _make_pred(
            "equipnas", "C",
            binding_protein=[2], per_residue={2: 0.9},
        )
        bad = _make_pred(
            "boltz2", "A",
            success=False,
            binding_protein=None, per_residue=None,
        )
        result = noisy_or_fusion(
            [good, bad],
            weights={"equipnas": 0.8, "boltz2": 0.9},
            threshold=0.5,
        )
        # bad must be ignored — fusion equals 0.8 * 0.9 = 0.72
        self.assertAlmostEqual(result["per_residue_probability"][2], 0.72, places=6)
        self.assertEqual(result["binding_protein_residues"], [2])

    def test_zero_weight_tool_dropped(self):
        # A tool with weight 0 should not contribute residues that no
        # other tool sees — otherwise we'd report b_hat=0 keys.
        preds = [
            _make_pred(
                "boltz2", "A",
                binding_protein=[5], per_residue={5: 90.0},
            ),
            _make_pred(
                "p2rank", "B",
                binding_protein=[6, 7], per_residue={6: 0.7, 7: 0.6},
            ),
        ]
        result = noisy_or_fusion(
            preds,
            weights={"boltz2": 0.9, "p2rank": 0.0},
            threshold=0.5,
        )
        self.assertEqual(set(result["per_residue_probability"]), {5})

    def test_plddt_normalization_in_fusion(self):
        # If pLDDT 100 wasn't divided by 100 we'd get b=100, garbage probs.
        preds = [
            _make_pred(
                "boltz2", "A",
                binding_protein=[1], per_residue={1: 100.0},
            ),
        ]
        result = noisy_or_fusion(
            preds, weights={"boltz2": 1.0}, threshold=0.5,
        )
        self.assertAlmostEqual(result["per_residue_probability"][1], 1.0, places=6)

    def test_threshold_strict_greater(self):
        # Build a residue whose fused prob equals exactly threshold.
        # Single tool, c=1.0, b=0.5 → b_hat=0.5
        preds = [
            _make_pred(
                "equipnas", "C",
                binding_protein=[1], per_residue={1: 0.5},
            ),
        ]
        result = noisy_or_fusion(
            preds, weights={"equipnas": 1.0}, threshold=0.5,
        )
        # Spec: B_p_hat = {i : b_hat(i) > τ} (strict).
        self.assertEqual(result["binding_protein_residues"], [])
        self.assertAlmostEqual(result["per_residue_probability"][1], 0.5, places=9)

    def test_threshold_just_above_included(self):
        preds = [
            _make_pred(
                "equipnas", "C",
                binding_protein=[1], per_residue={1: 0.51},
            ),
        ]
        result = noisy_or_fusion(
            preds, weights={"equipnas": 1.0}, threshold=0.5,
        )
        self.assertEqual(result["binding_protein_residues"], [1])

    def test_rna_side_fusion(self):
        # Two tools both predict nucleotide 11 with c=0.6 / 0.7 — both binary 1.0.
        # b_hat(11) = 1 - (1 - 0.6)*(1 - 0.7) = 1 - 0.4*0.3 = 1 - 0.12 = 0.88
        preds = [
            _make_pred(
                "boltz2", "A",
                binding_protein=[1], per_residue={1: 50.0},
                binding_rna=[10, 11],
            ),
            _make_pred(
                "rosettafold2na", "A",
                binding_protein=[1], per_residue={1: 50.0},
                binding_rna=[11, 12],
            ),
        ]
        result = noisy_or_fusion(
            preds,
            weights={"boltz2": 0.6, "rosettafold2na": 0.7},
            threshold=0.5,
        )
        self.assertAlmostEqual(
            result["per_nucleotide_probability"][11], 0.88, places=6,
        )
        # Only nucleotide 11 has > threshold:
        # 10: 1 - (1 - 0.6) = 0.6 > 0.5 → also in
        # 12: 1 - (1 - 0.7) = 0.7 > 0.5 → also in
        self.assertEqual(result["binding_rna_nucleotides"], [10, 11, 12])

    def test_no_surviving_tools_returns_empty(self):
        preds = [
            _make_pred(
                "boltz2", "A", success=False,
                binding_protein=None, per_residue=None,
            ),
        ]
        result = noisy_or_fusion(preds, weights={"boltz2": 0.9}, threshold=0.5)
        self.assertEqual(result["per_residue_probability"], {})
        self.assertEqual(result["per_nucleotide_probability"], {})
        self.assertEqual(result["binding_protein_residues"], [])
        self.assertEqual(result["binding_rna_nucleotides"], [])

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            noisy_or_fusion([], weights={}, threshold=1.5)

    def test_invalid_weight_rejected(self):
        preds = [
            _make_pred(
                "boltz2", "A",
                binding_protein=[1], per_residue={1: 50.0},
            ),
        ]
        with self.assertRaises(ValueError):
            noisy_or_fusion(preds, weights={"boltz2": 1.5}, threshold=0.5)


class TestEqualWeights(unittest.TestCase):
    def test_default_value_is_one(self):
        w = equal_weights(["a", "b", "c"])
        self.assertEqual(w, {"a": 1.0, "b": 1.0, "c": 1.0})

    def test_custom_value(self):
        w = equal_weights(["a", "b"], value=0.5)
        self.assertEqual(w, {"a": 0.5, "b": 0.5})


if __name__ == "__main__":
    unittest.main()
