"""Mock tests for scripts/riboseer/run_adapter_to_step4.py.

A fake adapter is injected via ``get_adapter`` so no real tool (fpocket
binary / RFAA GPU) is needed — we exercise the merge-into-step4, resume,
missing-sample, and override plumbing.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step4_tool_adapters.schemas import ToolPrediction  # noqa: E402
from scripts.riboseer import run_adapter_to_step4 as rats  # noqa: E402


def _write_sample(proc: Path, sid: str):
    d = proc / "samples"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sid}.json").write_text(
        json.dumps({"sample_id": sid,
                    "protein": {"length": 5, "sequence": "ACDEF"}}),
        encoding="utf-8")


def _write_step4(step4: Path, sid: str, tools):
    step4.mkdir(parents=True, exist_ok=True)
    preds = [{"tool_id": t, "category": "A", "sample_id": sid,
              "success": True} for t in tools]
    rec = {"sample_id": sid, "tools_run": list(tools), "predictions": preds}
    (step4 / f"{sid}.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")


class _FakeAdapter:
    def __init__(self, tool_id="fpocket", success=True):
        self.tool_id = tool_id
        self.success = success

    def predict(self, sample_json, work_dir, config):
        sid = sample_json["sample_id"]
        return ToolPrediction(
            tool_id=self.tool_id, category="B", sample_id=sid,
            success=self.success,
            error_message=None if self.success else "boom",
            per_residue_confidence={1: 0.9, 2: 0.2} if self.success else None,
            binding_protein_residues=[1] if self.success else None)


class TestMergeAndCreate(unittest.TestCase):
    def test_creates_step4_when_absent(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, step4 = td / "proc", td / "step4"
            _write_sample(proc, "1abc_A_E")
            with mock.patch.object(rats, "get_adapter",
                                   lambda t: _FakeAdapter("fpocket")):
                rc = rats.main([
                    "--tool", "fpocket", "--processed-dir", str(proc),
                    "--sample-list", _list(td, ["1abc_A_E"]),
                    "--step4-config", _cfg(td), "--step4-dir", str(step4)])
            self.assertEqual(rc, 0)
            rec = json.loads((step4 / "1abc_A_E.jsonl").read_text())
            self.assertEqual([p["tool_id"] for p in rec["predictions"]],
                             ["fpocket"])

    def test_merges_keeping_other_tools(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, step4 = td / "proc", td / "step4"
            _write_sample(proc, "1abc_A_E")
            _write_step4(step4, "1abc_A_E", ["boltz2", "equipnas"])
            with mock.patch.object(rats, "get_adapter",
                                   lambda t: _FakeAdapter("fpocket")):
                rc = rats.main([
                    "--tool", "fpocket", "--processed-dir", str(proc),
                    "--sample-list", _list(td, ["1abc_A_E"]),
                    "--step4-config", _cfg(td), "--step4-dir", str(step4)])
            self.assertEqual(rc, 0)
            rec = json.loads((step4 / "1abc_A_E.jsonl").read_text())
            tools = {p["tool_id"] for p in rec["predictions"]}
            self.assertEqual(tools, {"boltz2", "equipnas", "fpocket"})
            self.assertIn("fpocket", rec["tools_run"])

    def test_replaces_same_tool(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, step4 = td / "proc", td / "step4"
            _write_sample(proc, "1abc_A_E")
            _write_step4(step4, "1abc_A_E", ["fpocket", "boltz2"])
            with mock.patch.object(rats, "get_adapter",
                                   lambda t: _FakeAdapter("fpocket")):
                rats.main([
                    "--tool", "fpocket", "--processed-dir", str(proc),
                    "--sample-list", _list(td, ["1abc_A_E"]),
                    "--step4-config", _cfg(td), "--step4-dir", str(step4)])
            rec = json.loads((step4 / "1abc_A_E.jsonl").read_text())
            fps = [p for p in rec["predictions"] if p["tool_id"] == "fpocket"]
            self.assertEqual(len(fps), 1)               # not duplicated
            self.assertEqual(fps[0]["binding_protein_residues"], [1])  # fresh


class TestResumeAndSkips(unittest.TestCase):
    def test_resume_skips_existing_success(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, step4 = td / "proc", td / "step4"
            _write_sample(proc, "1abc_A_E")
            _write_step4(step4, "1abc_A_E", ["fpocket"])
            called = {"n": 0}

            def fake(t):
                called["n"] += 1
                return _FakeAdapter("fpocket")

            with mock.patch.object(rats, "get_adapter", fake):
                rc = rats.main([
                    "--tool", "fpocket", "--processed-dir", str(proc),
                    "--sample-list", _list(td, ["1abc_A_E"]),
                    "--step4-config", _cfg(td), "--step4-dir", str(step4),
                    "--resume"])
            self.assertEqual(rc, 0)
            self.assertEqual(called["n"], 0)            # adapter never run

    def test_missing_sample_json_reported_not_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, step4 = td / "proc", td / "step4"
            _write_sample(proc, "good_A_E")
            with mock.patch.object(rats, "get_adapter",
                                   lambda t: _FakeAdapter("fpocket")):
                rc = rats.main([
                    "--tool", "fpocket", "--processed-dir", str(proc),
                    "--sample-list", _list(td, ["good_A_E", "missing_X_Y"]),
                    "--step4-config", _cfg(td), "--step4-dir", str(step4)])
            # missing sample → rc 2, but good one still written
            self.assertEqual(rc, 2)
            self.assertTrue((step4 / "good_A_E.jsonl").is_file())
            self.assertFalse((step4 / "missing_X_Y.jsonl").is_file())

    def test_tool_failure_still_written(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, step4 = td / "proc", td / "step4"
            _write_sample(proc, "1abc_A_E")
            with mock.patch.object(rats, "get_adapter",
                                   lambda t: _FakeAdapter("fpocket", success=False)):
                rc = rats.main([
                    "--tool", "fpocket", "--processed-dir", str(proc),
                    "--sample-list", _list(td, ["1abc_A_E"]),
                    "--step4-config", _cfg(td), "--step4-dir", str(step4)])
            self.assertEqual(rc, 2)                      # tool_failed
            rec = json.loads((step4 / "1abc_A_E.jsonl").read_text())
            fp = rec["predictions"][0]
            self.assertEqual(fp["tool_id"], "fpocket")
            self.assertFalse(fp["success"])


class TestOverrides(unittest.TestCase):
    def test_device_and_rawdir_injected(self):
        cfg = {"tools": {"rfaa": {"conda_env": "RFAA"}}}
        raw = Path("/data/raw")
        rats._inject_overrides(cfg, raw, "rfaa", "cuda:1")
        self.assertEqual(cfg["structure_source"]["raw_dir"], str(raw))
        self.assertEqual(cfg["tools"]["rfaa"]["device"], "cuda:1")

    def test_canonical_alias(self):
        self.assertEqual(rats._canonical("rf2na"), "rosettafold2na")
        self.assertEqual(rats._canonical("fpocket"), "fpocket")


# ---- helpers ----


def _list(td: Path, ids) -> str:
    p = td / "list.txt"
    p.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return str(p)


def _cfg(td: Path) -> str:
    p = td / "step4_config.yaml"
    p.write_text("structure_source:\n  raw_dir: data/raw\ntools:\n  fpocket: {}\n",
                 encoding="utf-8")
    return str(p)


if __name__ == "__main__":
    unittest.main()
