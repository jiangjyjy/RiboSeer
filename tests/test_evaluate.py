"""Mock tests for scripts/evaluate.py.

Covers:
  - load_results parses JSONL, skips blank/# lines
  - aggregate stats: mean / median / std / Pearson / Spearman
  - by_category / by_quality grouping
  - tool_analysis success rate + mean F1
  - per_sample CSV columns + tools_used joined with ;
  - pocketqa_vs_f1 drops rows where qa_total or f1 is missing
  - main() end-to-end writes 5 files
  - degenerate-input safeguards (zero variance → None correlation, etc.)
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
    _coerce_float, _mean, _pearson, _spearman, aggregate, by_category_rows,
    load_results, main, per_sample_rows, pocketqa_vs_f1_rows,
    tool_analysis_rows,
)


def _write_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row(**overrides) -> dict:
    base = {
        "sample_id": "s1",
        "category": "RRM_x_stem_loop",
        "quality_tier": "high",
        "protein_length": 80,
        "rna_length": 30,
        "tools_attempted": 4,
        "tools_succeeded": 3,
        "tools_used": ["boltz2", "p2rank", "equipnas"],
        "fusion_status": "ok",
        "binding_protein_predicted": 18,
        "binding_protein_gt": 17,
        "precision": 0.8,
        "recall": 0.9,
        "f1": 0.85,
        "qa_total": 0.7,
        "qa_final": 0.75,
        "iteration_action": "accept",
        "total_iterations": 1,
        "termination_reason": "accepted",
        "runtime_seconds": 100.0,
        "total_tokens": 5000,
        "timestamp": "2026-05-04T00:00:00Z",
    }
    base.update(overrides)
    return base


# --------------------------- pure helpers ---------------------------------


class TestPureHelpers(unittest.TestCase):
    def test_coerce_float_normal(self):
        self.assertEqual(_coerce_float(0.5), 0.5)
        self.assertEqual(_coerce_float("0.7"), 0.7)
        self.assertEqual(_coerce_float(3), 3.0)

    def test_coerce_float_none_and_bad(self):
        self.assertIsNone(_coerce_float(None))
        self.assertIsNone(_coerce_float("not a number"))
        self.assertIsNone(_coerce_float(float("nan")))
        self.assertIsNone(_coerce_float(float("inf")))

    def test_mean_empty(self):
        self.assertIsNone(_mean([]))

    def test_pearson_perfect_positive(self):
        self.assertEqual(_pearson([1, 2, 3], [2, 4, 6]), 1.0)

    def test_pearson_perfect_negative(self):
        self.assertEqual(_pearson([1, 2, 3], [3, 2, 1]), -1.0)

    def test_pearson_zero_variance_returns_none(self):
        self.assertIsNone(_pearson([1, 1, 1], [1, 2, 3]))

    def test_pearson_too_few_points(self):
        self.assertIsNone(_pearson([], []))
        self.assertIsNone(_pearson([1.0], [2.0]))

    def test_spearman_basic(self):
        # Monotonic, non-linear → Pearson < 1, Spearman = 1.
        xs = [1, 2, 3, 4]
        ys = [1, 4, 9, 16]
        self.assertEqual(_spearman(xs, ys), 1.0)

    def test_spearman_handles_ties(self):
        # Tied ranks should not crash; result must be a float in [-1, 1].
        xs = [1.0, 1.0, 2.0, 3.0]
        ys = [1.0, 2.0, 2.0, 3.0]
        v = _spearman(xs, ys)
        self.assertIsNotNone(v)
        self.assertGreaterEqual(v, -1.0)
        self.assertLessEqual(v, 1.0)


# --------------------------- aggregate -----------------------------------


class TestAggregate(unittest.TestCase):
    def test_overall_mean_f1(self):
        rows = [
            _row(f1=0.5, precision=0.4, recall=0.6),
            _row(f1=0.7, precision=0.6, recall=0.8),
            _row(f1=0.9, precision=0.8, recall=1.0),
        ]
        agg = aggregate(rows)
        self.assertAlmostEqual(agg["aggregate"]["mean_f1"], 0.7, places=4)
        self.assertAlmostEqual(agg["aggregate"]["mean_precision"], 0.6, places=4)
        self.assertAlmostEqual(agg["aggregate"]["mean_recall"], 0.8, places=4)
        self.assertAlmostEqual(agg["aggregate"]["median_f1"], 0.7, places=4)
        self.assertEqual(agg["n_samples"], 3)
        self.assertEqual(agg["n_with_f1"], 3)

    def test_missing_f1_excluded(self):
        rows = [
            _row(f1=0.5),
            _row(f1=None),
            _row(f1=0.9),
        ]
        agg = aggregate(rows)
        # Only 2 rows contribute to the mean.
        self.assertAlmostEqual(agg["aggregate"]["mean_f1"], 0.7, places=4)
        self.assertEqual(agg["n_with_f1"], 2)
        self.assertEqual(agg["n_samples"], 3)

    def test_by_category_grouping(self):
        rows = [
            _row(category="RRM_x_stem_loop", f1=0.6),
            _row(category="RRM_x_stem_loop", f1=0.8),
            _row(category="kh_x_internal", f1=0.5),
        ]
        agg = aggregate(rows)
        self.assertEqual(agg["by_category"]["RRM_x_stem_loop"]["n"], 2)
        self.assertAlmostEqual(
            agg["by_category"]["RRM_x_stem_loop"]["mean_f1"], 0.7, places=4,
        )
        self.assertEqual(agg["by_category"]["kh_x_internal"]["n"], 1)

    def test_pocketqa_correlation_perfect(self):
        # qa_total identical to f1 → Pearson = 1.
        rows = [_row(qa_total=v, f1=v) for v in (0.2, 0.5, 0.8, 0.9)]
        agg = aggregate(rows)
        self.assertEqual(agg["pocketqa_correlation"]["pearson_r"], 1.0)
        self.assertEqual(agg["pocketqa_correlation"]["spearman_rho"], 1.0)
        self.assertEqual(agg["pocketqa_correlation"]["n"], 4)

    def test_pocketqa_correlation_none_when_data_missing(self):
        # No qa_total on any row → correlation block must be safe (no crash).
        rows = [_row(qa_total=None) for _ in range(3)]
        agg = aggregate(rows)
        self.assertEqual(agg["pocketqa_correlation"]["n"], 0)
        self.assertIsNone(agg["pocketqa_correlation"]["pearson_r"])

    def test_tool_success_rate(self):
        rows = [
            _row(tools_attempted=4, tools_used=["boltz2", "p2rank"]),
            _row(tools_attempted=4, tools_used=["boltz2"]),
            _row(tools_attempted=4, tools_used=["p2rank", "equipnas"]),
        ]
        agg = aggregate(rows)
        self.assertEqual(agg["tool_success_rate"]["boltz2"]["n_succeeded"], 2)
        self.assertEqual(
            agg["tool_success_rate"]["boltz2"]["share_of_samples"],
            round(2 / 3, 4),
        )

    def test_by_quality_tier(self):
        rows = [
            _row(quality_tier="high", f1=0.8),
            _row(quality_tier="high", f1=0.9),
            _row(quality_tier="low", f1=0.4),
        ]
        agg = aggregate(rows)
        self.assertEqual(agg["by_quality_tier"]["high"]["n"], 2)
        self.assertAlmostEqual(
            agg["by_quality_tier"]["high"]["mean_f1"], 0.85, places=4,
        )


# --------------------------- per-output transformers ----------------------


class TestRowTransformers(unittest.TestCase):
    def test_by_category_row_per_category(self):
        rows = [
            _row(category="A", f1=0.5),
            _row(category="A", f1=0.7),
            _row(category="B", f1=0.9),
        ]
        out = by_category_rows(rows)
        cats = [r["category"] for r in out]
        self.assertEqual(set(cats), {"A", "B"})
        # By-category mean_f1 for A is 0.6.
        a_row = next(r for r in out if r["category"] == "A")
        self.assertAlmostEqual(a_row["mean_f1"], 0.6, places=4)
        self.assertEqual(a_row["n"], 2)

    def test_per_sample_tools_used_joined(self):
        rows = [_row(tools_used=["boltz2", "p2rank"])]
        out = per_sample_rows(rows)
        self.assertEqual(out[0]["tools_used"], "boltz2;p2rank")

    def test_per_sample_handles_empty_tools(self):
        rows = [_row(tools_used=[])]
        out = per_sample_rows(rows)
        self.assertEqual(out[0]["tools_used"], "")

    def test_tool_analysis_mean_f1(self):
        rows = [
            _row(tools_used=["boltz2"], f1=0.5),
            _row(tools_used=["boltz2"], f1=0.9),
            _row(tools_used=[],         f1=0.0),
        ]
        out = tool_analysis_rows(rows)
        b = next(r for r in out if r["tool_id"] == "boltz2")
        self.assertEqual(b["n_samples_succeeded"], 2)
        self.assertAlmostEqual(b["success_rate"], round(2 / 3, 4), places=4)
        self.assertAlmostEqual(b["mean_f1_when_succeeded"], 0.7, places=4)

    def test_pocketqa_vs_f1_drops_missing(self):
        rows = [
            _row(qa_total=0.5, f1=0.5),     # keep
            _row(qa_total=None, f1=0.7),    # drop
            _row(qa_total=0.6, f1=None),    # drop
            _row(qa_total=0.9, f1=0.8),     # keep
        ]
        out = pocketqa_vs_f1_rows(rows)
        self.assertEqual(len(out), 2)


# --------------------------- main CLI ------------------------------------


class TestMainCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.results_path = self.tmp / "summary" / "results.jsonl"
        _write_results(self.results_path, [
            _row(sample_id="s1", category="A", f1=0.6, qa_total=0.5),
            _row(sample_id="s2", category="A", f1=0.8, qa_total=0.7),
            _row(sample_id="s3", category="B", f1=0.4, qa_total=0.3),
        ])
        self.output = self.tmp / "evaluation"

    def test_produces_five_files(self):
        rc = main([
            "--results", str(self.results_path),
            "--output", str(self.output),
        ])
        self.assertEqual(rc, 0)
        for fname in (
            "aggregate.json",
            "by_category.csv",
            "per_sample.csv",
            "tool_analysis.csv",
            "pocketqa_vs_f1.csv",
        ):
            self.assertTrue(
                (self.output / fname).is_file(),
                f"missing: {fname}",
            )

    def test_aggregate_json_shape(self):
        main([
            "--results", str(self.results_path),
            "--output", str(self.output),
        ])
        agg = json.loads(
            (self.output / "aggregate.json").read_text(encoding="utf-8"),
        )
        for key in ("n_samples", "aggregate", "by_category",
                    "by_quality_tier", "tool_success_rate",
                    "pocketqa_correlation"):
            self.assertIn(key, agg)
        self.assertEqual(agg["n_samples"], 3)

    def test_per_sample_csv_columns(self):
        main([
            "--results", str(self.results_path),
            "--output", str(self.output),
        ])
        with (self.output / "per_sample.csv").open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 3)
        # Spot-check a few critical columns.
        for col in ("sample_id", "category", "f1", "qa_total",
                    "tools_used", "runtime_seconds"):
            self.assertIn(col, rows[0])

    def test_missing_results_returns_1(self):
        rc = main([
            "--results", str(self.tmp / "does_not_exist.jsonl"),
            "--output", str(self.output),
        ])
        self.assertEqual(rc, 1)

    def test_empty_results_returns_1(self):
        empty = self.tmp / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        rc = main([
            "--results", str(empty),
            "--output", str(self.output),
        ])
        self.assertEqual(rc, 1)


# --------------------------- load_results ---------------------------------


class TestLoadResults(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_skips_blank_and_comments(self):
        path = self.tmp / "results.jsonl"
        path.write_text(
            "\n"
            "# comment line\n"
            + json.dumps({"sample_id": "a"}) + "\n"
            "\n"
            + json.dumps({"sample_id": "b"}) + "\n",
            encoding="utf-8",
        )
        out = load_results(path)
        self.assertEqual([r["sample_id"] for r in out], ["a", "b"])

    def test_skips_malformed_lines(self):
        path = self.tmp / "results.jsonl"
        path.write_text(
            json.dumps({"sample_id": "a"}) + "\n"
            "this is not json\n"
            + json.dumps({"sample_id": "c"}) + "\n",
            encoding="utf-8",
        )
        out = load_results(path)
        self.assertEqual([r["sample_id"] for r in out], ["a", "c"])

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_results(self.tmp / "nope.jsonl")


if __name__ == "__main__":
    unittest.main()
