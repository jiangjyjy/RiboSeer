"""Unit tests for step5 schemas — ToolWeightAssignment + CompositeResult."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pydantic import ValidationError  # noqa: E402

from step5_fusion.schemas import (  # noqa: E402
    CompositeResult,
    ToolWeightAssignment,
)


def _valid_assignment(**overrides) -> dict:
    base = {
        "weights": {"p2rank": 0.4, "boltz2": 0.9, "equipnas": 0.7},
        "threshold": 0.5,
        "rationale": (
            "Boltz-2 has high pLDDT and matches EquiPNAS on the core "
            "interface; P2Rank's pocket overlaps but is downweighted "
            "because P2Rank is RNA-agnostic."
        ),
        "confidence": 0.8,
    }
    base.update(overrides)
    return base


def _valid_composite(**overrides) -> dict:
    base = {
        "sample_id": "1un6_B_F",
        "tool_weights": {"p2rank": 0.5, "boltz2": 0.9, "equipnas": 0.8},
        "binding_protein_residues": [23, 24, 25, 78, 79],
        "binding_rna_nucleotides": [10, 11, 12],
        "per_residue_probability": {23: 0.92, 24: 0.88, 25: 0.85, 78: 0.91, 79: 0.76},
        "per_nucleotide_probability": {10: 0.88, 11: 0.81, 12: 0.77},
        "threshold": 0.5,
        "fusion_rationale": "All three tools agree on the core interface.",
        "confidence": 0.85,
        "tools_fused": ["p2rank", "boltz2", "equipnas"],
        "api_usage": {"prompt_tokens": 2000, "completion_tokens": 400},
        "timestamp": "2026-04-30T12:00:00Z",
    }
    base.update(overrides)
    return base


# ----------------------- ToolWeightAssignment -------------------------------


class TestToolWeightAssignmentValid(unittest.TestCase):
    def test_happy_path(self):
        twa = ToolWeightAssignment.model_validate(_valid_assignment())
        self.assertEqual(twa.weights["boltz2"], 0.9)
        self.assertEqual(twa.threshold, 0.5)
        self.assertGreaterEqual(twa.confidence, 0)
        self.assertLessEqual(twa.confidence, 1)

    def test_threshold_default(self):
        d = _valid_assignment()
        d.pop("threshold")
        twa = ToolWeightAssignment.model_validate(d)
        self.assertEqual(twa.threshold, 0.5)

    def test_int_weight_coerced_to_float(self):
        twa = ToolWeightAssignment.model_validate(
            _valid_assignment(weights={"p2rank": 1})
        )
        self.assertEqual(twa.weights["p2rank"], 1.0)


class TestToolWeightAssignmentInvalid(unittest.TestCase):
    def test_weight_above_one_rejected(self):
        with self.assertRaises(ValidationError):
            ToolWeightAssignment.model_validate(
                _valid_assignment(weights={"p2rank": 1.2, "boltz2": 0.5})
            )

    def test_negative_weight_rejected(self):
        with self.assertRaises(ValidationError):
            ToolWeightAssignment.model_validate(
                _valid_assignment(weights={"p2rank": -0.1})
            )

    def test_empty_weights_rejected(self):
        with self.assertRaises(ValidationError):
            ToolWeightAssignment.model_validate(_valid_assignment(weights={}))

    def test_threshold_above_one_rejected(self):
        with self.assertRaises(ValidationError):
            ToolWeightAssignment.model_validate(_valid_assignment(threshold=1.5))

    def test_threshold_negative_rejected(self):
        with self.assertRaises(ValidationError):
            ToolWeightAssignment.model_validate(_valid_assignment(threshold=-0.1))

    def test_confidence_out_of_range(self):
        with self.assertRaises(ValidationError):
            ToolWeightAssignment.model_validate(_valid_assignment(confidence=1.5))

    def test_rationale_too_short(self):
        with self.assertRaises(ValidationError):
            ToolWeightAssignment.model_validate(_valid_assignment(rationale="ok"))

    def test_rationale_whitespace_only(self):
        with self.assertRaises(ValidationError):
            ToolWeightAssignment.model_validate(
                _valid_assignment(rationale="          " * 5)
            )


# --------------------------- CompositeResult --------------------------------


class TestCompositeResultValid(unittest.TestCase):
    def test_happy_path(self):
        cr = CompositeResult.model_validate(_valid_composite())
        self.assertEqual(cr.sample_id, "1un6_B_F")
        self.assertEqual(cr.binding_protein_residues, [23, 24, 25, 78, 79])
        self.assertEqual(cr.tools_fused, ["p2rank", "boltz2", "equipnas"])
        self.assertEqual(cr.threshold, 0.5)

    def test_residues_sorted_and_unique_after_validation(self):
        cr = CompositeResult.model_validate(
            _valid_composite(binding_protein_residues=[79, 24, 23, 25, 78])
        )
        self.assertEqual(cr.binding_protein_residues, [23, 24, 25, 78, 79])

    def test_per_nucleotide_optional(self):
        d = _valid_composite()
        d.pop("per_nucleotide_probability")
        cr = CompositeResult.model_validate(d)
        self.assertIsNone(cr.per_nucleotide_probability)

    def test_empty_binding_lists_allowed(self):
        cr = CompositeResult.model_validate(
            _valid_composite(
                binding_protein_residues=[],
                binding_rna_nucleotides=[],
                per_residue_probability={},
                per_nucleotide_probability={},
            )
        )
        self.assertEqual(cr.binding_protein_residues, [])

    def test_int_keys_in_per_residue_prob_kept_as_int(self):
        cr = CompositeResult.model_validate(_valid_composite())
        self.assertTrue(all(isinstance(k, int) for k in cr.per_residue_probability))


class TestCompositeResultInvalid(unittest.TestCase):
    def test_duplicate_residues_rejected(self):
        with self.assertRaises(ValidationError):
            CompositeResult.model_validate(
                _valid_composite(binding_protein_residues=[23, 23, 24])
            )

    def test_residue_below_one_rejected(self):
        with self.assertRaises(ValidationError):
            CompositeResult.model_validate(
                _valid_composite(binding_protein_residues=[0, 1, 2])
            )

    def test_per_residue_prob_out_of_range(self):
        with self.assertRaises(ValidationError):
            CompositeResult.model_validate(
                _valid_composite(per_residue_probability={23: 1.2})
            )

    def test_per_residue_prob_residue_below_one(self):
        with self.assertRaises(ValidationError):
            CompositeResult.model_validate(
                _valid_composite(per_residue_probability={0: 0.5})
            )

    def test_tool_weight_above_one_rejected(self):
        with self.assertRaises(ValidationError):
            CompositeResult.model_validate(
                _valid_composite(tool_weights={"boltz2": 1.5})
            )

    def test_threshold_out_of_range(self):
        with self.assertRaises(ValidationError):
            CompositeResult.model_validate(_valid_composite(threshold=1.1))

    def test_confidence_out_of_range(self):
        with self.assertRaises(ValidationError):
            CompositeResult.model_validate(_valid_composite(confidence=-0.1))


if __name__ == "__main__":
    unittest.main()
