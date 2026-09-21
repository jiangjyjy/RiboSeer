"""Mock tests for scripts/riboseer/reparse_haddock3.py.

The heavy bit (Haddock3Adapter.parse_output → gemmi) is patched; these
tests cover the splice/status/IO logic that decides whether a sample's
step4 JSONL gets rewritten.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import scripts.riboseer.reparse_haddock3 as rh  # noqa: E402


class _FakePred:
    """Stand-in for ToolPrediction with a model_dump()."""
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return dict(self._payload)


def _write_step4(step4: Path, sid: str, haddock_entry, others=()):
    step4.mkdir(parents=True, exist_ok=True)
    preds = list(others) + ([haddock_entry] if haddock_entry else [])
    (step4 / f"{sid}.jsonl").write_text(
        json.dumps({"sample_id": sid, "predictions": preds}) + "\n",
        encoding="utf-8")


def _write_sample(proc: Path, sid: str):
    (proc / "samples").mkdir(parents=True, exist_ok=True)
    (proc / "samples" / f"{sid}.json").write_text(json.dumps({
        "sample_id": sid,
        "source_pdb": sid.split("_")[0],
        "protein": {"chain_id": "A", "sequence": "MKRY", "length": 4},
        "rna": {"chain_id": "E", "sequence": "ACGU", "length": 4},
        "interaction": {"binding_protein_residues": [2, 3]},
    }), encoding="utf-8")


def _make_run_dir(work: Path, sid: str) -> Path:
    rd = work / sid / "haddock_run"
    (rd / "3_emref").mkdir(parents=True, exist_ok=True)
    return rd


class TestReparseSample(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4 = self.tmp / "step4"
        self.proc = self.tmp / "proc"
        self.work = self.tmp / "work"

    def _haddock_entry(self, *, success=True, per_res=None):
        e = {"tool_id": "haddock3", "category": "D",
             "success": success,
             "binding_protein_residues": [2, 3]}
        e["per_residue_pae_score"] = per_res
        return e

    def test_fixed_when_null_becomes_scored(self):
        sid = "1abc_A_E"
        _write_sample(self.proc, sid)
        _write_step4(self.step4, sid,
                     self._haddock_entry(per_res=None))
        _make_run_dir(self.work, sid)
        new_pred = _FakePred({
            "tool_id": "haddock3", "category": "D", "success": True,
            "binding_protein_residues": [2, 3],
            "per_residue_pae_score": {"2": 0.8, "3": 0.7},
        })
        with mock.patch.object(rh.Haddock3Adapter, "parse_output",
                               return_value=new_pred):
            res = rh.reparse_sample(
                sample_id=sid, step4_dir=self.step4,
                processed_dir=self.proc, work_dir=self.work,
                config={}, dry_run=False)
        self.assertEqual(res["status"], "fixed")
        self.assertEqual(res["old_per_res"], 0)
        self.assertEqual(res["new_per_res"], 2)
        # The JSONL was rewritten with the scored entry.
        rec = json.loads(
            (self.step4 / f"{sid}.jsonl").read_text(encoding="utf-8"))
        hd = next(p for p in rec["predictions"]
                  if p["tool_id"] == "haddock3")
        self.assertEqual(hd["per_residue_pae_score"],
                         {"2": 0.8, "3": 0.7})

    def test_dry_run_does_not_write(self):
        sid = "1abc_A_E"
        _write_sample(self.proc, sid)
        _write_step4(self.step4, sid,
                     self._haddock_entry(per_res=None))
        _make_run_dir(self.work, sid)
        new_pred = _FakePred({
            "tool_id": "haddock3", "category": "D", "success": True,
            "per_residue_pae_score": {"2": 0.8},
        })
        with mock.patch.object(rh.Haddock3Adapter, "parse_output",
                               return_value=new_pred):
            res = rh.reparse_sample(
                sample_id=sid, step4_dir=self.step4,
                processed_dir=self.proc, work_dir=self.work,
                config={}, dry_run=True)
        self.assertEqual(res["status"], "fixed")
        # JSONL still has the OLD null entry.
        rec = json.loads(
            (self.step4 / f"{sid}.jsonl").read_text(encoding="utf-8"))
        hd = next(p for p in rec["predictions"]
                  if p["tool_id"] == "haddock3")
        self.assertIsNone(hd["per_residue_pae_score"])

    def test_failed_haddock_skipped(self):
        sid = "1abc_A_E"
        _write_sample(self.proc, sid)
        _write_step4(self.step4, sid,
                     self._haddock_entry(success=False, per_res=None))
        _make_run_dir(self.work, sid)
        # parse_output must NOT be called for a failed entry.
        with mock.patch.object(
                rh.Haddock3Adapter, "parse_output",
                side_effect=AssertionError("should not parse failed")):
            res = rh.reparse_sample(
                sample_id=sid, step4_dir=self.step4,
                processed_dir=self.proc, work_dir=self.work,
                config={}, dry_run=False)
        self.assertEqual(res["status"], "haddock3_failed")

    def test_no_run_dir(self):
        sid = "1abc_A_E"
        _write_sample(self.proc, sid)
        _write_step4(self.step4, sid,
                     self._haddock_entry(per_res=None))
        # No run dir created.
        res = rh.reparse_sample(
            sample_id=sid, step4_dir=self.step4,
            processed_dir=self.proc, work_dir=self.work,
            config={}, dry_run=False)
        self.assertEqual(res["status"], "no_run_dir")

    def test_no_haddock_entry(self):
        sid = "1abc_A_E"
        _write_sample(self.proc, sid)
        _write_step4(self.step4, sid, None,
                     others=[{"tool_id": "boltz2", "success": True}])
        _make_run_dir(self.work, sid)
        res = rh.reparse_sample(
            sample_id=sid, step4_dir=self.step4,
            processed_dir=self.proc, work_dir=self.work,
            config={}, dry_run=False)
        self.assertEqual(res["status"], "no_haddock3_entry")

    def test_unchanged_when_already_scored(self):
        sid = "1abc_A_E"
        _write_sample(self.proc, sid)
        _write_step4(self.step4, sid,
                     self._haddock_entry(per_res={"2": 0.5}))
        _make_run_dir(self.work, sid)
        new_pred = _FakePred({
            "tool_id": "haddock3", "category": "D", "success": True,
            "per_residue_pae_score": {"2": 0.5},  # same count
        })
        with mock.patch.object(rh.Haddock3Adapter, "parse_output",
                               return_value=new_pred):
            res = rh.reparse_sample(
                sample_id=sid, step4_dir=self.step4,
                processed_dir=self.proc, work_dir=self.work,
                config={}, dry_run=False)
        self.assertEqual(res["status"], "unchanged")


class TestRunDirResolution(unittest.TestCase):

    def test_direct_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            rd = work / "s1" / "haddock_run"
            rd.mkdir(parents=True)
            self.assertEqual(
                rh._find_run_dir(work, "s1").resolve(), rd.resolve())

    def test_nested_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            rd = work / "s1" / "sub" / "haddock_run"
            rd.mkdir(parents=True)
            found = rh._find_run_dir(work, "s1")
            self.assertIsNotNone(found)
            self.assertEqual(found.name, "haddock_run")

    def test_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(rh._find_run_dir(Path(tmp), "ghost"))


class TestCli(unittest.TestCase):

    def test_main_dry_run_smoke(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            step4 = tmp / "step4"
            proc = tmp / "proc"
            work = tmp / "work"
            for sid in ("1abc_A_E", "2def_B_F"):
                _write_sample(proc, sid)
                _write_step4(step4, sid, {
                    "tool_id": "haddock3", "category": "D",
                    "success": True,
                    "per_residue_pae_score": None})
                _make_run_dir(work, sid)
            new_pred = _FakePred({
                "tool_id": "haddock3", "category": "D", "success": True,
                "per_residue_pae_score": {"2": 0.8, "3": 0.7}})
            with mock.patch.object(rh.Haddock3Adapter, "parse_output",
                                   return_value=new_pred):
                rc = rh.main([
                    "--step4-dir", str(step4),
                    "--processed-dir", str(proc),
                    "--work-dir", str(work),
                    "--dry-run",
                ])
            self.assertEqual(rc, 0)
            # Dry-run → originals still null.
            for sid in ("1abc_A_E", "2def_B_F"):
                rec = json.loads(
                    (step4 / f"{sid}.jsonl").read_text(encoding="utf-8"))
                hd = next(p for p in rec["predictions"]
                          if p["tool_id"] == "haddock3")
                self.assertIsNone(hd["per_residue_pae_score"])

    def test_main_missing_work_dir_returns_1(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            step4 = tmp / "step4"
            step4.mkdir()
            rc = rh.main([
                "--step4-dir", str(step4),
                "--processed-dir", str(tmp / "proc"),
                "--work-dir", str(tmp / "nope"),
            ])
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
