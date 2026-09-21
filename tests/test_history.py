"""Unit tests for `history.py` — PredictionHistory."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step2_target_char.history import PredictionHistory  # noqa: E402


def _rec(sid: str, cat: str, conf: float = 0.8, **extra) -> dict:
    base = {"sample_id": sid, "category": cat, "confidence": conf,
            "timestamp": "2026-04-18T00:00:00Z"}
    base.update(extra)
    return base


class TestAddRecord(unittest.TestCase):
    def test_add_and_len(self):
        h = PredictionHistory()
        self.assertEqual(len(h), 0)
        h.add_record(_rec("a", "RRM_x_stem_loop"))
        self.assertEqual(len(h), 1)

    def test_requires_sample_id_and_category(self):
        h = PredictionHistory()
        with self.assertRaises(ValueError):
            h.add_record({"sample_id": "a"})
        with self.assertRaises(ValueError):
            h.add_record({"category": "x"})

    def test_records_returns_copy(self):
        h = PredictionHistory()
        h.add_record(_rec("a", "RRM_x_stem_loop"))
        recs = h.records
        recs.clear()
        self.assertEqual(len(h), 1)


class TestCategoryStats(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(PredictionHistory().get_category_stats(), {})

    def test_counts_and_avg(self):
        h = PredictionHistory()
        h.add_record(_rec("a", "RRM_x_stem_loop", 0.9))
        h.add_record(_rec("b", "RRM_x_stem_loop", 0.7))
        h.add_record(_rec("c", "KH_x_junction", 0.6))
        stats = h.get_category_stats()
        self.assertEqual(stats["RRM_x_stem_loop"]["count"], 2)
        self.assertAlmostEqual(stats["RRM_x_stem_loop"]["avg_confidence"], 0.8, places=3)
        self.assertEqual(stats["KH_x_junction"]["count"], 1)

    def test_sorted_by_count_desc(self):
        h = PredictionHistory()
        for _ in range(3):
            h.add_record(_rec("x", "A_x_B"))
        h.add_record(_rec("y", "C_x_D"))
        cats = list(h.get_category_stats().keys())
        self.assertEqual(cats[0], "A_x_B")

    def test_missing_confidence_ignored(self):
        h = PredictionHistory()
        h.add_record({"sample_id": "a", "category": "X_x_Y"})
        stats = h.get_category_stats()
        self.assertEqual(stats["X_x_Y"]["avg_confidence"], 0.0)


class TestGetSummary(unittest.TestCase):
    def test_empty_returns_empty_string(self):
        self.assertEqual(PredictionHistory().get_summary(), "")

    def test_contains_total_and_distribution(self):
        h = PredictionHistory()
        h.add_record(_rec("a", "RRM_x_stem_loop", 0.9))
        h.add_record(_rec("b", "KH_x_junction", 0.7))
        s = h.get_summary()
        self.assertIn("Total predictions so far: 2", s)
        self.assertIn("RRM_x_stem_loop", s)
        self.assertIn("KH_x_junction", s)

    def test_category_filter(self):
        h = PredictionHistory()
        h.add_record(_rec("a", "RRM_x_stem_loop"))
        h.add_record(_rec("b", "KH_x_junction"))
        s = h.get_summary(category="KH_x_junction")
        self.assertIn("b", s)

    def test_filter_no_match_returns_empty(self):
        h = PredictionHistory()
        h.add_record(_rec("a", "RRM_x_stem_loop"))
        self.assertEqual(h.get_summary(category="nope"), "")

    def test_limit_caps_recent(self):
        h = PredictionHistory()
        for i in range(20):
            h.add_record(_rec(f"s{i}", "A_x_B"))
        s = h.get_summary(limit=3)
        self.assertIn("s17", s)
        self.assertIn("s18", s)
        self.assertIn("s19", s)
        self.assertNotIn("s0:", s)


class TestSaveLoad(unittest.TestCase):
    def test_roundtrip(self):
        h = PredictionHistory()
        h.add_record(_rec("a", "RRM_x_stem_loop", 0.9))
        h.add_record(_rec("b", "KH_x_junction", 0.6, tools_used=["tool1"]))

        tmp = Path(tempfile.mkdtemp()) / "history.jsonl"
        h.save(tmp)

        h2 = PredictionHistory.load(tmp)
        self.assertEqual(len(h2), 2)
        self.assertEqual(h2.records[0]["sample_id"], "a")
        self.assertEqual(h2.records[1]["tools_used"], ["tool1"])

    def test_load_missing_file(self):
        h = PredictionHistory.load(Path("/nonexistent/path.jsonl"))
        self.assertEqual(len(h), 0)

    def test_save_creates_parent_dirs(self):
        tmp = Path(tempfile.mkdtemp()) / "sub" / "dir" / "h.jsonl"
        h = PredictionHistory()
        h.add_record(_rec("a", "X_x_Y"))
        h.save(tmp)
        self.assertTrue(tmp.exists())

    def test_file_format_is_jsonl(self):
        tmp = Path(tempfile.mkdtemp()) / "h.jsonl"
        h = PredictionHistory()
        h.add_record(_rec("a", "X_x_Y"))
        h.add_record(_rec("b", "Z_x_W"))
        h.save(tmp)
        lines = tmp.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            json.loads(line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
