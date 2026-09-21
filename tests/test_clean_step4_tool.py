"""Mock tests for scripts/clean_step4_tool.py."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.clean_step4_tool as cln  # noqa: E402


def _rec(sid, preds, *, tools_run=None, total_rt=None):
    r = {"sample_id": sid, "predictions": preds}
    if tools_run is not None:
        r["tools_run"] = tools_run
    if total_rt is not None:
        r["total_runtime_seconds"] = total_rt
    return r


def _p(tool_id, success=True, rt=None):
    d = {"tool_id": tool_id, "category": "A", "success": success}
    if rt is not None:
        d["runtime_seconds"] = rt
    return d


class TestCleanRecord(unittest.TestCase):
    def test_removes_all_matching_and_updates_runtime(self):
        rec = _rec("s1",
                   [_p("boltz2", rt=10.0), _p("haddock3", rt=4.0),
                    _p("p2rank", rt=2.0)],
                   tools_run=["boltz2", "haddock3", "p2rank"],
                   total_rt=16.0)
        out, n, rt, tr = cln._clean_record(rec, "haddock3", False)
        self.assertEqual(n, 1)
        self.assertEqual(rt, 4.0)
        self.assertTrue(tr)
        self.assertEqual([p["tool_id"] for p in out["predictions"]],
                         ["boltz2", "p2rank"])
        self.assertEqual(out["tools_run"], ["boltz2", "p2rank"])
        self.assertEqual(out["total_runtime_seconds"], 12.0)
        # input not mutated
        self.assertEqual(len(rec["predictions"]), 3)

    def test_noop_when_tool_absent(self):
        rec = _rec("s1", [_p("boltz2")], tools_run=["boltz2"])
        out, n, rt, tr = cln._clean_record(rec, "haddock3", False)
        self.assertEqual(n, 0)
        self.assertFalse(tr)
        self.assertIs(out, rec)  # unchanged → same object

    def test_only_failed_keeps_successful_and_tools_run(self):
        rec = _rec("s1",
                   [_p("rf2na", success=False, rt=1.0),
                    _p("rf2na", success=True, rt=5.0),
                    _p("boltz2", rt=3.0)],
                   tools_run=["rf2na", "boltz2"], total_rt=9.0)
        out, n, rt, tr = cln._clean_record(rec, "rf2na", True)
        self.assertEqual(n, 1)            # only the failed one
        self.assertEqual(rt, 1.0)
        # a successful rf2na survives → tools_run keeps rf2na
        self.assertFalse(tr)
        self.assertIn("rf2na", out["tools_run"])
        self.assertEqual(out["total_runtime_seconds"], 8.0)

    def test_runtime_clamped_at_zero(self):
        rec = _rec("s1", [_p("haddock3", rt=99.0)],
                   tools_run=["haddock3"], total_rt=5.0)
        out, *_ = cln._clean_record(rec, "haddock3", False)
        self.assertEqual(out["total_runtime_seconds"], 0.0)

    def test_exact_match_no_alias(self):
        # rf2na and rosettafold2na are distinct — removing rf2na must
        # NOT touch the real rosettafold2na record.
        rec = _rec("s1", [_p("rf2na", success=False),
                          _p("rosettafold2na", success=True)])
        out, n, *_ = cln._clean_record(rec, "rf2na", False)
        self.assertEqual(n, 1)
        self.assertEqual([p["tool_id"] for p in out["predictions"]],
                         ["rosettafold2na"])


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.d = self.tmp / "step4"
        self.d.mkdir()
        for sid in ("a", "b", "c"):
            (self.d / f"{sid}.jsonl").write_text(
                json.dumps(_rec(sid,
                    [_p("boltz2", rt=10.0),
                     _p("haddock3", rt=4.0)],
                    tools_run=["boltz2", "haddock3"],
                    total_rt=14.0)) + "\n", encoding="utf-8")

    def test_dry_run_writes_nothing(self):
        before = (self.d / "a.jsonl").read_text(encoding="utf-8")
        rc = cln.main(["--step4-dir", str(self.d),
                       "--remove-tool", "haddock3", "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertEqual((self.d / "a.jsonl").read_text(encoding="utf-8"),
                         before)

    def test_applies_in_place(self):
        rc = cln.main(["--step4-dir", str(self.d),
                       "--remove-tool", "haddock3"])
        self.assertEqual(rc, 0)
        for sid in ("a", "b", "c"):
            rec = json.loads(
                (self.d / f"{sid}.jsonl").read_text(encoding="utf-8"))
            ids = [p["tool_id"] for p in rec["predictions"]]
            self.assertEqual(ids, ["boltz2"])
            self.assertEqual(rec["tools_run"], ["boltz2"])
            self.assertEqual(rec["total_runtime_seconds"], 10.0)

    def test_multi_line_and_non_json_passthrough(self):
        f = self.d / "multi.jsonl"
        f.write_text(
            json.dumps(_rec("m", [_p("haddock3", rt=2.0),
                                  _p("boltz2", rt=1.0)],
                            tools_run=["haddock3", "boltz2"],
                            total_rt=3.0)) + "\n"
            + "# a comment line\n"
            + json.dumps(_rec("m", [_p("boltz2", rt=1.0)],
                              tools_run=["boltz2"], total_rt=1.0)) + "\n",
            encoding="utf-8")
        rc = cln.main(["--step4-dir", str(self.d),
                       "--remove-tool", "haddock3"])
        self.assertEqual(rc, 0)
        lines = [ln for ln in (self.d / "multi.jsonl")
                 .read_text(encoding="utf-8").splitlines() if ln]
        self.assertEqual(lines[1], "# a comment line")  # untouched
        first = json.loads(lines[0])
        self.assertEqual([p["tool_id"] for p in first["predictions"]],
                         ["boltz2"])

    def test_missing_dir_returns_1(self):
        rc = cln.main(["--step4-dir", str(self.tmp / "nope"),
                       "--remove-tool", "haddock3"])
        self.assertEqual(rc, 1)

    def test_empty_dir_returns_1(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        rc = cln.main(["--step4-dir", str(empty),
                       "--remove-tool", "haddock3"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
