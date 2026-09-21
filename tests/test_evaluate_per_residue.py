"""Mock tests for the per-residue correlation block of scripts/evaluate.py.

Covers the new helpers added on top of the existing evaluate test
file:
  - build_pred_gt_vectors (vector construction, Cat A pLDDT scaling,
    binary fallback when per_residue is empty)
  - compute_per_residue_correlation (degenerate-input handling)
  - collect_per_residue_correlations (reads step4 + step5 JSONLs,
    bucketed by tool / fusion)
  - aggregate_per_residue (mean / std / median per method)
  - end-to-end main() with --step4-dir / --step5-dir / --processed-dir
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.evaluate import (  # noqa: E402
    FUSION_METHOD_ID, _per_residue_to_int_dict,
    aggregate_per_residue, build_pred_gt_vectors,
    collect_per_residue_correlations,
    compute_per_residue_correlation, main, per_residue_csv_rows,
)


# ---------- build_pred_gt_vectors -----------------------------------------


class TestBuildVectors(unittest.TestCase):
    def test_basic(self):
        # protein length 5, GT residues 1 & 3, scores match the GT pattern
        pred, gt = build_pred_gt_vectors(
            per_residue={1: 0.9, 3: 0.8, 4: 0.1},
            binding_residues=[1, 3, 4],
            binding_gt=[1, 3],
            protein_length=5,
            is_cat_a=False,
        )
        self.assertEqual(gt, [1.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(pred, [0.9, 0.0, 0.8, 0.1, 0.0])

    def test_cat_a_plddt_scaled_to_unit_range(self):
        # pLDDT values 0-100 must be divided by 100 for Cat A.
        pred, _gt = build_pred_gt_vectors(
            per_residue={1: 80.0, 2: 50.0},
            binding_residues=[1, 2],
            binding_gt=[1],
            protein_length=3,
            is_cat_a=True,
        )
        self.assertEqual(pred, [0.8, 0.5, 0.0])

    def test_clamp_out_of_range(self):
        pred, _gt = build_pred_gt_vectors(
            per_residue={1: -0.5, 2: 1.5},
            binding_residues=[1, 2],
            binding_gt=[1],
            protein_length=2,
            is_cat_a=False,
        )
        self.assertEqual(pred, [0.0, 1.0])

    def test_binary_fallback_when_per_residue_empty(self):
        # No per-residue scores: fall back to 1.0 for binding residues.
        pred, gt = build_pred_gt_vectors(
            per_residue={},
            binding_residues=[2, 4],
            binding_gt=[2, 3],
            protein_length=5,
            is_cat_a=False,
        )
        self.assertEqual(pred, [0.0, 1.0, 0.0, 1.0, 0.0])
        self.assertEqual(gt, [0.0, 1.0, 1.0, 0.0, 0.0])


# ---------- compute_per_residue_correlation -------------------------------


class TestComputeCorrelation(unittest.TestCase):
    def test_perfect_alignment(self):
        # Pred matches GT exactly → Pearson = 1.0, R² = 1.0.
        out = compute_per_residue_correlation(
            per_residue={1: 1.0, 2: 0.0, 3: 1.0, 4: 0.0},
            binding_residues=[1, 3],
            binding_gt=[1, 3],
            protein_length=4,
            is_cat_a=False,
        )
        self.assertAlmostEqual(out["pearson_r"], 1.0, places=4)
        self.assertAlmostEqual(out["r_squared"], 1.0, places=4)

    def test_gt_all_zeros_returns_none(self):
        # No binding residues anywhere — correlation undefined.
        out = compute_per_residue_correlation(
            per_residue={1: 0.5},
            binding_residues=[1],
            binding_gt=[],
            protein_length=3,
            is_cat_a=False,
        )
        self.assertIsNone(out)

    def test_pred_zero_variance_returns_none(self):
        # Tool gave 0 to every residue and didn't list any binding —
        # zero variance in pred → undefined Pearson.
        out = compute_per_residue_correlation(
            per_residue={},
            binding_residues=[],
            binding_gt=[1, 2],
            protein_length=3,
            is_cat_a=False,
        )
        self.assertIsNone(out)


# ---------- _per_residue_to_int_dict --------------------------------------


class TestPerResidueCoercion(unittest.TestCase):
    def test_string_keys(self):
        out = _per_residue_to_int_dict({"1": 0.5, "2": 0.7})
        self.assertEqual(out, {1: 0.5, 2: 0.7})

    def test_int_keys(self):
        out = _per_residue_to_int_dict({1: 0.5, 2: 0.7})
        self.assertEqual(out, {1: 0.5, 2: 0.7})

    def test_mixed_and_garbage(self):
        out = _per_residue_to_int_dict({"1": 0.5, "abc": 0.6, 3: "bad"})
        self.assertEqual(out, {1: 0.5})

    def test_empty_or_none(self):
        self.assertEqual(_per_residue_to_int_dict({}), {})
        self.assertEqual(_per_residue_to_int_dict(None), {})


# ---------- end-to-end with synthetic step4/step5 JSONLs ------------------


def _write_step4(step4_dir: Path, sample_id: str, predictions: list[dict]) -> None:
    path = step4_dir / f"{sample_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "sample_id": sample_id,
        "tools_run": [p["tool_id"] for p in predictions],
        "predictions": predictions,
    }
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")


def _write_step5(step5_dir: Path, sample_id: str, per_res: dict, binding: list) -> None:
    path = step5_dir / f"{sample_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "sample_id": sample_id,
        "binding_protein_residues": binding,
        # step5 stringifies its keys when serialising.
        "per_residue_probability": {str(k): v for k, v in per_res.items()},
        "threshold": 0.5,
    }
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")


def _write_sample(processed: Path, sample_id: str, *, length: int, gt: list) -> None:
    samples = processed / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    sample = {
        "sample_id": sample_id,
        "protein": {"length": length},
        "interaction": {"binding_protein_residues": gt},
    }
    (samples / f"{sample_id}.json").write_text(
        json.dumps(sample), encoding="utf-8",
    )


def _summary_row(sid: str = "s1") -> dict:
    # Minimal results.jsonl row — only sample_id is consumed by the
    # per-residue collector, but a fuller row keeps main() happy.
    return {
        "sample_id": sid,
        "category": "X",
        "tools_attempted": 2, "tools_succeeded": 2,
        "tools_used": ["chai1", "p2rank"],
        "binding_protein_predicted": 2, "binding_protein_gt": 2,
        "precision": 1.0, "recall": 1.0, "f1": 1.0,
        "qa_total": 0.7, "qa_final": 0.7,
        "iteration_action": "accept", "total_iterations": 1,
        "termination_reason": "accepted",
        "runtime_seconds": 1.0, "total_tokens": 0,
        "timestamp": "2026-05-04T00:00:00Z",
    }


class TestCollectAndAggregate(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.processed = self.tmp / "processed"
        self.step4 = self.tmp / "step4"
        self.step5 = self.tmp / "step5"

        # Two samples; protein length 5 each; GT = {1, 3}.
        for sid in ("s1", "s2"):
            _write_sample(self.processed, sid, length=5, gt=[1, 3])
            _write_step4(self.step4, sid, [
                {  # Cat A pLDDT (0-100) — perfect alignment with GT
                    "tool_id": "chai1", "category": "A",
                    "sample_id": sid, "success": True,
                    "binding_protein_residues": [1, 3],
                    "per_residue_confidence": {"1": 90.0, "3": 80.0},
                },
                {  # Cat B already in [0, 1]; perfect too
                    "tool_id": "p2rank", "category": "B",
                    "sample_id": sid, "success": True,
                    "binding_protein_residues": [1, 3],
                    "per_residue_confidence": {"1": 0.9, "3": 0.8},
                },
            ])
            _write_step5(self.step5, sid,
                         per_res={1: 0.95, 3: 0.85}, binding=[1, 3])

    def test_collect_buckets_per_method(self):
        bucket = collect_per_residue_correlations(
            [_summary_row("s1"), _summary_row("s2")],
            step4_dir=self.step4,
            step5_dir=self.step5,
            processed_dir=self.processed,
        )
        # Three methods: chai1, p2rank, fusion
        self.assertIn("chai1", bucket)
        self.assertIn("p2rank", bucket)
        self.assertIn(FUSION_METHOD_ID, bucket)
        # Two samples each
        self.assertEqual(len(bucket["chai1"]), 2)
        self.assertEqual(len(bucket[FUSION_METHOD_ID]), 2)
        # Strong (but not perfect) alignment: pred = [0.9, 0, 0.8, 0, 0]
        # vs gt = [1, 0, 1, 0, 0] gives r ≈ 0.997.
        self.assertGreater(bucket["chai1"][0]["pearson_r"], 0.99)

    def test_aggregate_shapes(self):
        bucket = collect_per_residue_correlations(
            [_summary_row("s1"), _summary_row("s2")],
            step4_dir=self.step4,
            step5_dir=self.step5,
            processed_dir=self.processed,
        )
        agg = aggregate_per_residue(bucket)
        for method in ("chai1", "p2rank", FUSION_METHOD_ID):
            self.assertEqual(agg[method]["n_samples"], 2)
            for k in ("pearson_r", "spearman_r", "r_squared"):
                for stat in ("mean", "std", "median"):
                    self.assertIn(stat, agg[method][k])

    def test_csv_row_shape(self):
        bucket = collect_per_residue_correlations(
            [_summary_row("s1")], step4_dir=self.step4,
            step5_dir=self.step5, processed_dir=self.processed,
        )
        rows = per_residue_csv_rows(aggregate_per_residue(bucket))
        self.assertGreater(len(rows), 0)
        for r in rows:
            for col in ("method", "n_samples",
                        "pearson_r_mean", "pearson_r_std", "pearson_r_median",
                        "spearman_r_mean", "spearman_r_std", "spearman_r_median",
                        "r2_mean", "r2_std", "r2_median"):
                self.assertIn(col, r)


class TestMainEndToEnd(unittest.TestCase):
    """Drive main() to verify the new files are emitted alongside the
    existing 5."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.results = self.tmp / "summary" / "results.jsonl"
        self.results.parent.mkdir(parents=True)
        self.results.write_text(
            json.dumps(_summary_row("s1")) + "\n", encoding="utf-8",
        )
        self.processed = self.tmp / "processed"
        self.step4 = self.tmp / "step4"
        self.step5 = self.tmp / "step5"
        _write_sample(self.processed, "s1", length=5, gt=[1, 3])
        _write_step4(self.step4, "s1", [{
            "tool_id": "chai1", "category": "A", "sample_id": "s1",
            "success": True,
            "binding_protein_residues": [1, 3],
            "per_residue_confidence": {"1": 90.0, "3": 80.0},
        }])
        _write_step5(self.step5, "s1",
                     per_res={1: 0.95, 3: 0.85}, binding=[1, 3])
        self.out = self.tmp / "out"

    def test_emits_per_residue_files(self):
        rc = main([
            "--results", str(self.results),
            "--processed-dir", str(self.processed),
            "--step4-dir", str(self.step4),
            "--step5-dir", str(self.step5),
            "--output", str(self.out),
        ])
        self.assertEqual(rc, 0)
        # Existing 5 + 2 new files
        for fname in (
            "aggregate.json", "by_category.csv", "per_sample.csv",
            "tool_analysis.csv", "pocketqa_vs_f1.csv",
            "per_residue_correlation.json",
            "per_residue_correlation.csv",
        ):
            self.assertTrue((self.out / fname).is_file(),
                            f"missing {fname}")
        # Spot-check the JSON contents.
        agg = json.loads(
            (self.out / "per_residue_correlation.json")
            .read_text(encoding="utf-8")
        )
        self.assertIn("methods", agg)
        self.assertIn("chai1", agg["methods"])
        self.assertIn(FUSION_METHOD_ID, agg["methods"])

    def test_omitting_dirs_skips_per_residue_block(self):
        rc = main([
            "--results", str(self.results),
            "--output", str(self.out / "noopt"),
        ])
        self.assertEqual(rc, 0)
        # Per-residue files should NOT be written when the optional
        # flags are missing.
        self.assertFalse(
            (self.out / "noopt" / "per_residue_correlation.json").exists()
        )


if __name__ == "__main__":
    unittest.main()
