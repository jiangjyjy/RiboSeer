"""Unit tests for step 4 schemas (ToolPrediction / ToolPredictionSet)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pydantic import ValidationError  # noqa: E402

from step4_tool_adapters.schemas import (  # noqa: E402
    Pocket, ToolPrediction, ToolPredictionSet, make_failure_prediction,
)


# ---------- helpers --------------------------------------------------------


def _p2rank_success() -> dict:
    return {
        "tool_id": "p2rank",
        "category": "B",
        "sample_id": "1un6_B_F",
        "success": True,
        "binding_protein_residues": [14, 15, 16, 17],
        "per_residue_confidence": {14: 0.91, 15: 0.88, 16: 0.74, 17: 0.65},
        "pockets": [{"rank": 1, "score": 12.4, "residues": [14, 15, 16, 17]}],
        "runtime_seconds": 0.42,
    }


def _boltz2_success() -> dict:
    return {
        "tool_id": "boltz2",
        "category": "A",
        "sample_id": "1un6_B_F",
        "success": True,
        "binding_protein_residues": [2, 14, 15],
        "binding_rna_nucleotides": [6, 7, 8],
        "per_residue_confidence": {2: 80.1, 14: 75.2, 15: 71.5},
        "predicted_structure_path": "/tmp/foo/predictions/x_model_0.cif",
        "plddt_mean": 75.5,
        "iptm_score": 0.62,
        "pae_mean": 8.1,
        "runtime_seconds": 234.0,
    }


def _equipnas_success() -> dict:
    return {
        "tool_id": "equipnas",
        "category": "C",
        "sample_id": "1un6_B_F",
        "success": True,
        "binding_protein_residues": [2, 14, 15],
        "per_residue_confidence": {2: 0.91, 14: 0.88, 15: 0.84},
        "runtime_seconds": 18.0,
    }


# ---------- happy-path tests ----------------------------------------------


class TestToolPredictionHappy(unittest.TestCase):
    def test_p2rank_record(self):
        rec = ToolPrediction.model_validate(_p2rank_success())
        self.assertEqual(rec.tool_id, "p2rank")
        self.assertEqual(rec.category, "B")
        self.assertEqual(len(rec.pockets), 1)
        self.assertEqual(rec.binding_rna_nucleotides, None)

    def test_boltz2_record(self):
        rec = ToolPrediction.model_validate(_boltz2_success())
        self.assertEqual(rec.binding_rna_nucleotides, [6, 7, 8])
        self.assertEqual(rec.plddt_mean, 75.5)

    def test_equipnas_record(self):
        rec = ToolPrediction.model_validate(_equipnas_success())
        self.assertEqual(rec.category, "C")
        self.assertIsNone(rec.binding_rna_nucleotides)
        self.assertIsNone(rec.predicted_structure_path)

    def test_jsonl_roundtrip(self):
        rec = ToolPrediction.model_validate(_boltz2_success())
        line = json.dumps(rec.model_dump(mode="json"))
        rec2 = ToolPrediction.model_validate_json(line)
        self.assertEqual(rec2.binding_protein_residues, rec.binding_protein_residues)
        # int dict keys survive (pydantic coerces from str when loading json)
        self.assertEqual(set(rec2.per_residue_confidence.keys()),
                         set(rec.per_residue_confidence.keys()))

    def test_residues_sorted_and_deduped(self):
        d = _p2rank_success()
        d["binding_protein_residues"] = [16, 14, 15]
        rec = ToolPrediction.model_validate(d)
        self.assertEqual(rec.binding_protein_residues, [14, 15, 16])

    def test_residues_duplicate_rejected(self):
        d = _p2rank_success()
        d["binding_protein_residues"] = [14, 14]
        with self.assertRaises(ValidationError):
            ToolPrediction.model_validate(d)


# ---------- failure-path tests --------------------------------------------


class TestToolPredictionFailure(unittest.TestCase):
    def test_failure_no_predictions_ok(self):
        rec = make_failure_prediction(
            tool_id="p2rank", category="B", sample_id="1un6_B_F",
            error_message="boom",
        )
        self.assertFalse(rec.success)
        self.assertIsNone(rec.binding_protein_residues)
        self.assertEqual(rec.error_message, "boom")

    def test_success_without_any_prediction_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            ToolPrediction.model_validate({
                "tool_id": "p2rank", "category": "B",
                "sample_id": "1un6_B_F", "success": True,
            })
        self.assertIn("no prediction fields", str(ctx.exception))

    def test_invalid_category_rejected(self):
        d = _p2rank_success()
        d["category"] = "Z"
        with self.assertRaises(ValidationError):
            ToolPrediction.model_validate(d)

    def test_plddt_out_of_range(self):
        d = _boltz2_success()
        d["plddt_mean"] = 120.0
        with self.assertRaises(ValidationError):
            ToolPrediction.model_validate(d)

    def test_iptm_out_of_range(self):
        d = _boltz2_success()
        d["iptm_score"] = 1.5
        with self.assertRaises(ValidationError):
            ToolPrediction.model_validate(d)

    def test_zero_residue_index_rejected(self):
        d = _p2rank_success()
        d["binding_protein_residues"] = [0, 14, 15]
        with self.assertRaises(ValidationError):
            ToolPrediction.model_validate(d)

    def test_extra_field_rejected(self):
        d = _p2rank_success()
        d["bogus"] = "field"
        with self.assertRaises(ValidationError):
            ToolPrediction.model_validate(d)


# ---------- Pocket tests ---------------------------------------------------


class TestPocket(unittest.TestCase):
    def test_basic(self):
        p = Pocket(rank=1, score=10.0, residues=[14, 15, 16])
        self.assertEqual(p.rank, 1)

    def test_rank_must_be_positive(self):
        with self.assertRaises(ValidationError):
            Pocket(rank=0, score=10.0, residues=[14])

    def test_dup_residues_rejected(self):
        with self.assertRaises(ValidationError):
            Pocket(rank=1, score=10.0, residues=[14, 14])


# ---------- ToolPredictionSet tests ---------------------------------------


class TestToolPredictionSet(unittest.TestCase):
    def test_aggregate(self):
        s = ToolPredictionSet(
            sample_id="1un6_B_F",
            tools_run=["p2rank", "boltz2"],
            predictions=[
                ToolPrediction.model_validate(_p2rank_success()),
                ToolPrediction.model_validate(_boltz2_success()),
            ],
            total_runtime_seconds=234.4,
            timestamp="2026-04-28T10:00:00Z",
        )
        self.assertEqual(len(s.predictions), 2)

    def test_sample_id_mismatch(self):
        with self.assertRaises(ValidationError):
            ToolPredictionSet(
                sample_id="abc",
                predictions=[ToolPrediction.model_validate(_p2rank_success())],
            )

    def test_tools_run_must_match_predictions(self):
        with self.assertRaises(ValidationError):
            ToolPredictionSet(
                sample_id="1un6_B_F",
                tools_run=["p2rank", "boltz2", "equipnas"],
                predictions=[
                    ToolPrediction.model_validate(_p2rank_success()),
                ],
            )

    def test_jsonl_serialisable(self):
        s = ToolPredictionSet(
            sample_id="1un6_B_F",
            tools_run=["p2rank"],
            predictions=[ToolPrediction.model_validate(_p2rank_success())],
        )
        line = json.dumps(s.model_dump(mode="json"))
        rt = json.loads(line)
        self.assertEqual(rt["sample_id"], "1un6_B_F")


if __name__ == "__main__":
    unittest.main(verbosity=2)
