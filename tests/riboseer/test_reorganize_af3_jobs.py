"""Mock tests for scripts/riboseer/reorganize_af3_jobs.py."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import scripts.riboseer.reorganize_af3_jobs as rj  # noqa: E402


# ---- pure helpers --------------------------------------------------------


class TestDayMath(unittest.TestCase):

    def test_default_30_per_day_boundaries(self):
        # 1..30 → day 1, 31..60 → day 2, 61..90 → day 3, 91..105 → day 4
        self.assertEqual(rj.day_for_priority(1, 30), 1)
        self.assertEqual(rj.day_for_priority(30, 30), 1)
        self.assertEqual(rj.day_for_priority(31, 30), 2)
        self.assertEqual(rj.day_for_priority(60, 30), 2)
        self.assertEqual(rj.day_for_priority(61, 30), 3)
        self.assertEqual(rj.day_for_priority(90, 30), 3)
        self.assertEqual(rj.day_for_priority(91, 30), 4)
        self.assertEqual(rj.day_for_priority(105, 30), 4)

    def test_custom_per_day(self):
        self.assertEqual(rj.day_for_priority(1, 10), 1)
        self.assertEqual(rj.day_for_priority(10, 10), 1)
        self.assertEqual(rj.day_for_priority(11, 10), 2)
        self.assertEqual(rj.day_for_priority(100, 10), 10)

    def test_zero_per_day_raises(self):
        with self.assertRaises(ValueError):
            rj.day_for_priority(1, 0)
        with self.assertRaises(ValueError):
            rj.day_for_priority(5, -3)


class TestTargetPath(unittest.TestCase):

    def test_pads_priority_to_3_digits(self):
        self.assertEqual(
            rj.target_relative_path(1, "abc", 30),
            Path("day1") / "001_abc.json")
        self.assertEqual(
            rj.target_relative_path(105, "9jxs_I_A", 30),
            Path("day4") / "105_9jxs_I_A.json")
        self.assertEqual(
            rj.target_relative_path(7, "x", 30),
            Path("day1") / "007_x.json")


# ---- read_priority_order ------------------------------------------------


class TestReadPriorityOrder(unittest.TestCase):

    def test_missing_column_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.csv"
            p.write_text("priority,foo\n1,xx\n", encoding="utf-8")
            with self.assertRaises(ValueError) as cm:
                rj.read_priority_order(p)
            self.assertIn("sample_id", str(cm.exception))

    def test_returns_sorted_by_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "order.csv"
            p.write_text(
                "priority,sample_id,prot_len,rna_len,total_len\n"
                "3,c,10,10,20\n"
                "1,a,5,5,10\n"
                "2,b,7,8,15\n",
                encoding="utf-8")
            out = rj.read_priority_order(p)
        self.assertEqual(out, [(1, "a"), (2, "b"), (3, "c")])

    def test_skips_non_integer_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "order.csv"
            p.write_text(
                "priority,sample_id\n"
                "1,a\n"
                "bad,b\n"
                "2,c\n",
                encoding="utf-8")
            out = rj.read_priority_order(p)
        self.assertEqual(out, [(1, "a"), (2, "c")])


# ---- find_source --------------------------------------------------------


class TestFindSource(unittest.TestCase):
    """Cover all three locations the resolver tries: canonical target,
    flat root, and any-day fallback."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _touch(self, rel: str) -> Path:
        p = self.tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")
        return p

    def test_finds_at_canonical_target(self):
        canonical = self._touch("day1/001_a.json")
        got = rj.find_source(self.tmp, "a", 1, 30)
        self.assertEqual(got.resolve(), canonical.resolve())

    def test_finds_flat_when_no_canonical(self):
        flat = self._touch("a.json")
        got = rj.find_source(self.tmp, "a", 1, 30)
        self.assertEqual(got.resolve(), flat.resolve())

    def test_finds_in_other_day_bucket(self):
        # Imagine a prior run bucketed at per-day=25 → put priority 1
        # at day1/001_a.json, then we run again with per-day=30 →
        # canonical is still day1/001_a.json so this hits canonical.
        # The real fallback path triggers when priority changed too:
        # build it at day1/030_a.json (priority 30) then look up
        # priority 25 → fallback scans day*/ and grabs the file.
        self._touch("day2/030_a.json")
        got = rj.find_source(self.tmp, "a", 25, 25)   # canonical: day1/025_a.json
        self.assertIsNotNone(got)
        self.assertEqual(got.name, "030_a.json")

    def test_returns_none_when_absent(self):
        self.assertIsNone(rj.find_source(self.tmp, "ghost", 1, 30))


# ---- plan + execute -----------------------------------------------------


class TestPlanAndExecute(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.jobs = self.tmp / "jobs"
        self.jobs.mkdir()

    def _write_jobs(self, sample_ids):
        for sid in sample_ids:
            (self.jobs / f"{sid}.json").write_text(
                json.dumps({"name": sid}), encoding="utf-8")

    def test_plan_only_includes_moves_needed(self):
        # Pre-place one file at its canonical destination so plan
        # skips it.
        self._write_jobs(["b", "c"])  # flat
        canon = self.jobs / "day1" / "001_a.json"
        canon.parent.mkdir(parents=True, exist_ok=True)
        canon.write_text("{}", encoding="utf-8")
        order = [(1, "a"), (2, "b"), (3, "c")]
        moves, missing = rj.plan_moves(order, self.jobs, per_day=30)
        # a is already in place → not in moves; b and c flat → moves.
        srcs = {m[0].name for m in moves}
        self.assertEqual(srcs, {"b.json", "c.json"})
        self.assertEqual(missing, [])

    def test_plan_reports_missing(self):
        self._write_jobs(["b"])
        order = [(1, "a"), (2, "b")]
        moves, missing = rj.plan_moves(order, self.jobs, per_day=30)
        self.assertEqual(missing, ["a"])

    def test_execute_renames_into_day_subdirs(self):
        self._write_jobs(["a", "b"])
        moves = [(self.jobs / "a.json",
                  self.jobs / "day1" / "001_a.json"),
                 (self.jobs / "b.json",
                  self.jobs / "day1" / "002_b.json")]
        n = rj.execute_moves(moves)
        self.assertEqual(n, 2)
        self.assertTrue((self.jobs / "day1" / "001_a.json").is_file())
        self.assertTrue((self.jobs / "day1" / "002_b.json").is_file())
        # Source filenames are gone (it was a move, not a copy).
        self.assertFalse((self.jobs / "a.json").exists())
        self.assertFalse((self.jobs / "b.json").exists())

    def test_dry_run_makes_no_changes(self):
        self._write_jobs(["a"])
        moves = [(self.jobs / "a.json",
                  self.jobs / "day1" / "001_a.json")]
        n = rj.execute_moves(moves, dry_run=True)
        self.assertEqual(n, 0)
        self.assertTrue((self.jobs / "a.json").is_file())
        self.assertFalse(
            (self.jobs / "day1" / "001_a.json").exists())

    def test_idempotent_rerun_is_noop(self):
        # First pass moves files into day buckets; second pass should
        # find nothing to do (all already at canonical target).
        self._write_jobs(["a", "b"])
        order = [(1, "a"), (2, "b")]
        first_moves, _ = rj.plan_moves(order, self.jobs, per_day=30)
        rj.execute_moves(first_moves)
        second_moves, _ = rj.plan_moves(order, self.jobs, per_day=30)
        self.assertEqual(second_moves, [])


# ---- end-to-end CLI ----------------------------------------------------


class TestCli(unittest.TestCase):

    def test_main_full_smoke(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            jobs = tmp / "af3_jobs"
            jobs.mkdir()
            sids = [f"s{i:02d}" for i in range(1, 8)]  # 7 samples
            for sid in sids:
                (jobs / f"{sid}.json").write_text(
                    json.dumps({"name": sid}), encoding="utf-8")
            order = tmp / "order.csv"
            with order.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "priority", "sample_id", "prot_len",
                    "rna_len", "total_len"])
                w.writeheader()
                for prio, sid in enumerate(sids, start=1):
                    w.writerow({
                        "priority": prio, "sample_id": sid,
                        "prot_len": 10, "rna_len": 10,
                        "total_len": 20,
                    })
            rc = rj.main([
                "--order-csv", str(order),
                "--jobs-dir", str(jobs),
                "--per-day", "3",   # 7 jobs / 3-per-day = 3 days
            ])
            self.assertEqual(rc, 0)
            # Day 1 (priorities 1-3), day 2 (4-6), day 3 (7).
            self.assertTrue(
                (jobs / "day1" / "001_s01.json").is_file())
            self.assertTrue(
                (jobs / "day1" / "003_s03.json").is_file())
            self.assertTrue(
                (jobs / "day2" / "004_s04.json").is_file())
            self.assertTrue(
                (jobs / "day2" / "006_s06.json").is_file())
            self.assertTrue(
                (jobs / "day3" / "007_s07.json").is_file())
            # Originals gone.
            self.assertFalse((jobs / "s01.json").exists())

    def test_main_dry_run_keeps_originals(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            jobs = tmp / "af3_jobs"
            jobs.mkdir()
            (jobs / "a.json").write_text("{}", encoding="utf-8")
            order = tmp / "order.csv"
            order.write_text(
                "priority,sample_id\n1,a\n", encoding="utf-8")
            rc = rj.main([
                "--order-csv", str(order),
                "--jobs-dir", str(jobs),
                "--per-day", "30",
                "--dry-run",
            ])
            self.assertEqual(rc, 0)
            # Original still at root, day1 not created.
            self.assertTrue((jobs / "a.json").is_file())
            self.assertFalse((jobs / "day1").exists())

    def test_main_missing_order_csv_returns_1(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            jobs = tmp / "jobs"
            jobs.mkdir()
            rc = rj.main([
                "--order-csv", str(tmp / "nope.csv"),
                "--jobs-dir", str(jobs),
            ])
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
