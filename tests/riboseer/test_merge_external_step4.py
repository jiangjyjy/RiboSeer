"""Mock tests for merge_external_step4.py."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.riboseer.merge_external_step4 as me  # noqa: E402


def _set(sample_id, preds, **extra):
    d = {"sample_id": sample_id, "tools_run": sorted({p["tool_id"] for p in preds}),
         "predictions": preds}
    d.update(extra)
    return d


class TestMergeOne(unittest.TestCase):
    def test_replaces_same_tool_keeps_others(self):
        target = _set("s", [
            {"tool_id": "boltz2", "category": "A", "sample_id": "s", "success": True,
             "binding_protein_residues": [1]},
            {"tool_id": "bindup", "category": "C", "sample_id": "s", "success": True,
             "binding_protein_residues": [9]},  # stale, should be replaced
        ], timestamp="T1")
        source = _set("s", [
            {"tool_id": "bindup", "category": "C", "sample_id": "s", "success": True,
             "binding_protein_residues": [5, 6]},
        ])
        merged, src_tools, replaced = me.merge_one(source, target, "s")
        self.assertEqual(src_tools, ["bindup"])
        self.assertEqual(replaced, ["bindup"])
        tools = {p["tool_id"]: p for p in merged["predictions"]}
        self.assertEqual(set(tools), {"boltz2", "bindup"})
        self.assertEqual(tools["bindup"]["binding_protein_residues"], [5, 6])
        self.assertEqual(tools["boltz2"]["binding_protein_residues"], [1])
        self.assertEqual(sorted(merged["tools_run"]), ["bindup", "boltz2"])
        self.assertEqual(merged["timestamp"], "T1")  # target meta preserved

    def test_create_when_no_target(self):
        source = _set("s", [{"tool_id": "bindup", "category": "C",
                             "sample_id": "s", "success": True,
                             "per_residue_confidence": {"1": 0.5}}])
        merged, src_tools, replaced = me.merge_one(source, None, "s")
        self.assertEqual(replaced, [])
        self.assertEqual([p["tool_id"] for p in merged["predictions"]], ["bindup"])
        self.assertEqual(merged["tools_run"], ["bindup"])


class TestCli(unittest.TestCase):
    def test_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "src"; src.mkdir()
            tgt = tmp / "tgt"; tgt.mkdir()
            (src / "s.jsonl").write_text(json.dumps(_set("s", [
                {"tool_id": "bindup", "category": "C", "sample_id": "s",
                 "success": True, "binding_protein_residues": [5]}])) + "\n",
                encoding="utf-8")
            (tgt / "s.jsonl").write_text(json.dumps(_set("s", [
                {"tool_id": "p2rank", "category": "B", "sample_id": "s",
                 "success": True, "binding_protein_residues": [1]}])) + "\n",
                encoding="utf-8")
            rc = me.main(["--source-dir", str(src), "--target-dir", str(tgt),
                          "--dry-run"])
            self.assertEqual(rc, 0)
            rec = json.loads((tgt / "s.jsonl").read_text(encoding="utf-8"))
            self.assertEqual([p["tool_id"] for p in rec["predictions"]], ["p2rank"])

    def test_apply_merges_and_creates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "src"; src.mkdir()
            tgt = tmp / "tgt"; tgt.mkdir()
            # s1: target exists with p2rank -> add bindup
            (tgt / "s1.jsonl").write_text(json.dumps(_set("s1", [
                {"tool_id": "p2rank", "category": "B", "sample_id": "s1",
                 "success": True, "binding_protein_residues": [1]}])) + "\n",
                encoding="utf-8")
            for sid in ("s1", "s2"):
                (src / f"{sid}.jsonl").write_text(json.dumps(_set(sid, [
                    {"tool_id": "bindup", "category": "C", "sample_id": sid,
                     "success": True, "binding_protein_residues": [7]}])) + "\n",
                    encoding="utf-8")
            rc = me.main(["--source-dir", str(src), "--target-dir", str(tgt)])
            self.assertEqual(rc, 0)
            r1 = json.loads((tgt / "s1.jsonl").read_text(encoding="utf-8"))
            self.assertEqual({p["tool_id"] for p in r1["predictions"]},
                             {"p2rank", "bindup"})
            r2 = json.loads((tgt / "s2.jsonl").read_text(encoding="utf-8"))  # created
            self.assertEqual([p["tool_id"] for p in r2["predictions"]], ["bindup"])


if __name__ == "__main__":
    unittest.main()
