"""Mock tests for the DeepPocket baseline scripts.

GPU / molgrid / fpocket are never invoked. The parser tests pin the one piece
that's easy to get wrong — pairing each CNN confidence (descending order) with
the *fpocket pocket id* from ranked.types, not a naive ``pocket{i+1}`` — plus
the PDB residue parsing and the per-residue max aggregation. The batch tests
cover the subprocess timeout (real short-lived child) and the orchestration loop
with ``run_predict`` mocked.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import step4_tool_adapters.external.deeppocket_parse as pd  # noqa: E402
import scripts.riboseer.run_deeppocket_batch as db  # noqa: E402


def atom_line(resseq, atom="CA", resn="ALA", chain="A", serial=1):
    """A PDB ATOM record with ``resseq`` placed at cols 23-26 (slice [22:26])."""
    line = list(" " * 66)
    for i, ch in enumerate("ATOM  "):
        line[i] = ch
    for i, ch in enumerate(f"{serial:>5}"):
        line[6 + i] = ch
    for i, ch in enumerate(f"{atom:<4}"):
        line[12 + i] = ch
    for i, ch in enumerate(f"{resn:>3}"):
        line[17 + i] = ch
    line[21] = chain
    for i, ch in enumerate(f"{resseq:>4}"):
        line[22 + i] = ch
    return "".join(line)


def make_pockets_dir(tmp: Path, sid: str, *, confidences, ranked_ids,
                     pocket_residues):
    """Build a synthetic ``<sid>_nowat_out/pockets`` dir."""
    pk = tmp / sid / f"{sid}_nowat_out" / "pockets"
    pk.mkdir(parents=True)
    (pk / "bary_centers_confidence.txt").write_text(
        str(list(confidences)), encoding="utf-8")
    lines = [f"{pid} 1.0 2.0 3.0 /prot.gninatypes" for pid in ranked_ids]
    (pk / "bary_centers_ranked.types").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    for pid, residues in pocket_residues.items():
        body = "\n".join(atom_line(r, serial=j + 1)
                         for j, r in enumerate(residues))
        (pk / f"pocket{pid}_atm.pdb").write_text(body + "\n", encoding="utf-8")
    return pk


# ----------------------------------------------------------------------------
# low-level parsing
# ----------------------------------------------------------------------------
class TestLowLevelParsing(unittest.TestCase):
    def test_parse_confidences_list_repr(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.txt"
            p.write_text("[0.97, 0.4, 1e-05]", encoding="utf-8")
            self.assertEqual(pd.parse_confidences(p), [0.97, 0.4, 1e-05])

    def test_parse_confidences_clamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.txt"
            p.write_text("[1.2, -0.1, 0.5]", encoding="utf-8")
            self.assertEqual(pd.parse_confidences(p), [1.0, 0.0, 0.5])

    def test_parse_ranked_pocket_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.types"
            p.write_text("2 1.0 2.0 3.0 /a\n5 4 5 6 /a\n1 7 8 9 /a\n",
                         encoding="utf-8")
            self.assertEqual(pd.parse_ranked_pocket_ids(p), [2, 5, 1])

    def test_read_pocket_residues(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "pocket1_atm.pdb"
            p.write_text("\n".join([
                atom_line(10, atom="N"), atom_line(10, atom="CA"),
                atom_line(42, atom="CA"), "TER", "END",
            ]) + "\n", encoding="utf-8")
            self.assertEqual(pd.read_pocket_residues(p), [10, 42])

    def test_find_pockets_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pk = make_pockets_dir(tmp, "1abc_A_E", confidences=[0.5],
                                  ranked_ids=[1], pocket_residues={1: [3]})
            self.assertEqual(pd.find_pockets_dir(tmp / "1abc_A_E").resolve(),
                             pk.resolve())

    def test_find_pockets_dir_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(pd.find_pockets_dir(Path(tmp) / "ghost"))


# ----------------------------------------------------------------------------
# build_prediction: the rank <-> fpocket-id <-> residue mapping
# ----------------------------------------------------------------------------
class TestBuildPrediction(unittest.TestCase):
    def _dir(self, tmp):
        # confidence-descending: 0.9 -> fpocket pocket 2, 0.4 -> 5, 0.1 -> 1
        return make_pockets_dir(
            Path(tmp), "1abc_A_E",
            confidences=[0.9, 0.4, 0.1],
            ranked_ids=[2, 5, 1],
            pocket_residues={2: [10, 11], 5: [11, 20], 1: [30]},
        )

    def test_mapping_and_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            pk = self._dir(tmp)
            pred = pd.build_prediction("1abc_A_E", pk, threshold=0.5)
            self.assertEqual(pred.tool_id, "deeppocket")
            self.assertEqual(pred.category, "B")
            # pockets in confidence order; rank1 = fpocket pocket 2's residues
            self.assertEqual([(pk_.rank, pk_.score, pk_.residues)
                              for pk_ in pred.pockets],
                             [(1, 0.9, [10, 11]),
                              (2, 0.4, [11, 20]),
                              (3, 0.1, [30])])
            # per-residue = max confidence over containing pockets
            self.assertEqual(pred.per_residue_confidence,
                             {10: 0.9, 11: 0.9, 20: 0.4, 30: 0.1})
            # binding = residues with score > 0.5
            self.assertEqual(pred.binding_protein_residues, [10, 11])

    def test_max_residue_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            pk = self._dir(tmp)
            pred = pd.build_prediction("1abc_A_E", pk, threshold=0.5,
                                       max_residue=20)
            self.assertNotIn(30, pred.per_residue_confidence)
            self.assertEqual(pred.pockets[2].residues, [])  # pocket 1 had {30}

    def test_count_mismatch_pairs_min(self):
        with tempfile.TemporaryDirectory() as tmp:
            pk = make_pockets_dir(
                Path(tmp), "x_A_E",
                confidences=[0.9, 0.4],          # 2 confidences
                ranked_ids=[2, 5, 1],            # 3 ranked ids
                pocket_residues={2: [10], 5: [20], 1: [30]})
            pred = pd.build_prediction("x_A_E", pk, threshold=0.5)
            self.assertEqual(len(pred.pockets), 2)
            self.assertNotIn(30, pred.per_residue_confidence)

    def test_empty_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            pk = make_pockets_dir(Path(tmp), "x_A_E", confidences=[],
                                  ranked_ids=[], pocket_residues={})
            with self.assertRaises(ValueError):
                pd.build_prediction("x_A_E", pk, threshold=0.5)


# ----------------------------------------------------------------------------
# merge + end-to-end parse_one_sample
# ----------------------------------------------------------------------------
class TestMergeAndParse(unittest.TestCase):
    def test_merge_preserves_other_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            step4 = Path(tmp)
            sid = "1abc_A_E"
            (step4 / f"{sid}.jsonl").write_text(json.dumps({
                "sample_id": sid, "tools_run": ["p2rank", "deeppocket"],
                "predictions": [
                    {"tool_id": "p2rank", "category": "B", "sample_id": sid,
                     "success": True, "binding_protein_residues": [7]},
                    {"tool_id": "deeppocket", "category": "B", "sample_id": sid,
                     "success": True, "per_residue_confidence": {1: 0.1}},
                ],
            }) + "\n", encoding="utf-8")
            new = pd.ToolPrediction(
                tool_id="deeppocket", category="B", sample_id=sid, success=True,
                per_residue_confidence={9: 0.8}, binding_protein_residues=[9])
            pd.merge_into_set(step4, sid, new)
            rec = json.loads((step4 / f"{sid}.jsonl").read_text(encoding="utf-8"))
            tools = {p["tool_id"] for p in rec["predictions"]}
            self.assertEqual(tools, {"p2rank", "deeppocket"})
            dp = next(p for p in rec["predictions"]
                      if p["tool_id"] == "deeppocket")
            self.assertEqual(dp["binding_protein_residues"], [9])

    def test_parse_one_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            make_pockets_dir(results, "1abc_A_E", confidences=[0.8, 0.2],
                             ranked_ids=[3, 1],
                             pocket_residues={3: [5, 6], 1: [9]})
            pred = pd.parse_one_sample(
                "1abc_A_E", results_dir=results, samples_dir=None,
                threshold=0.5)
            self.assertEqual(pred.per_residue_confidence, {5: 0.8, 6: 0.8, 9: 0.2})
            self.assertEqual(pred.binding_protein_residues, [5, 6])

    def test_parse_one_sample_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                pd.parse_one_sample("ghost", results_dir=Path(tmp),
                                    samples_dir=None, threshold=0.5)


# ----------------------------------------------------------------------------
# batch: paths, command, subprocess timeout, orchestration
# ----------------------------------------------------------------------------
class TestBatch(unittest.TestCase):
    def test_path_helpers(self):
        out = Path("/o")
        self.assertEqual(db.input_pdb_path(out, "s").as_posix(), "/o/s/s.pdb")
        self.assertEqual(db.pockets_dir(out, "s").as_posix(),
                         "/o/s/s_nowat_out/pockets")

    def test_build_predict_cmd(self):
        cmd = db.build_predict_cmd(
            protein_pdb=Path("/o/s/s.pdb"), class_ckpt="c.tar",
            seg_ckpt="g.tar", rank=3)
        self.assertEqual(cmd[1:], ["predict.py", "-p", str(Path("/o/s/s.pdb")),
                                   "-c", "c.tar", "-s", "g.tar", "-r", "3"])

    def test_is_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertFalse(db.is_done(out, "s"))
            pk = db.pockets_dir(out, "s")
            pk.mkdir(parents=True)
            (pk / "bary_centers_confidence.txt").write_text("[]", encoding="utf-8")
            self.assertTrue(db.is_done(out, "s"))

    def test_run_predict_success(self):
        rc, out, timed = db.run_predict(
            [sys.executable, "-c", "print('hi')"],
            cwd=Path("."), env=dict(__import__("os").environ), timeout=30)
        self.assertEqual(rc, 0)
        self.assertIn("hi", out)
        self.assertFalse(timed)

    def test_run_predict_timeout_kills_child(self):
        rc, _out, timed = db.run_predict(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=Path("."), env=dict(__import__("os").environ), timeout=1)
        self.assertTrue(timed)
        self.assertIsNone(rc)

    def _setup_sample(self, base: Path, sid="1abc_A_E"):
        samples = base / "samples"; samples.mkdir()
        (samples / f"{sid}.json").write_text(json.dumps({
            "sample_id": sid, "source_pdb": "1abc",
            "protein": {"chain_id": "A"}}), encoding="utf-8")
        raw = base / "raw"; raw.mkdir()
        (raw / "1abc.pdb").write_text("RAW", encoding="utf-8")
        dp = base / "dp"; dp.mkdir()
        (dp / "predict.py").write_text("# stub", encoding="utf-8")
        return samples, raw, dp

    def test_run_one_sample_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            samples, raw, dp = self._setup_sample(base)
            out = base / "out"
            sid = "1abc_A_E"

            def fake_extract(raw_path, chain_id, dst):
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                Path(dst).write_text("PDB", encoding="utf-8")
                return 50

            def fake_run_predict(cmd, cwd, env, timeout):
                # emulate DeepPocket producing the confidence file
                pk = db.pockets_dir(out, sid)
                pk.mkdir(parents=True, exist_ok=True)
                (pk / "bary_centers_confidence.txt").write_text("[0.9]",
                                                                encoding="utf-8")
                return 0, "ok\n", False

            with mock.patch.object(db, "extract_protein_chain_pdb", fake_extract), \
                 mock.patch.object(db, "run_predict", fake_run_predict):
                res = db.run_one_sample(
                    sid, samples_dir=samples, raw_dir=raw, output_dir=out,
                    deeppocket_dir=dp, class_ckpt="c", seg_ckpt="s", rank=3,
                    device="1", timeout=300, resume=False)
            self.assertEqual(res["status"], "ok", msg=res.get("error"))
            self.assertEqual(res["n_residues"], 50)
            self.assertTrue((out / sid / "predict.log").is_file())

    def test_run_one_sample_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            samples, raw, dp = self._setup_sample(base)
            out = base / "out"

            def fake_extract(raw_path, chain_id, dst):
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                Path(dst).write_text("PDB", encoding="utf-8")
                return 10

            def fake_run_predict(cmd, cwd, env, timeout):
                return None, "...", True  # timed out

            with mock.patch.object(db, "extract_protein_chain_pdb", fake_extract), \
                 mock.patch.object(db, "run_predict", fake_run_predict):
                res = db.run_one_sample(
                    "1abc_A_E", samples_dir=samples, raw_dir=raw, output_dir=out,
                    deeppocket_dir=dp, class_ckpt="c", seg_ckpt="s", rank=3,
                    device="1", timeout=300, resume=False)
            self.assertEqual(res["status"], "failed")
            self.assertTrue(res["timed_out"])

    def test_run_one_sample_resume_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            pk = db.pockets_dir(out, "1abc_A_E")
            pk.mkdir(parents=True)
            (pk / "bary_centers_confidence.txt").write_text("[0.5]",
                                                            encoding="utf-8")
            res = db.run_one_sample(
                "1abc_A_E", samples_dir=Path(tmp), raw_dir=Path(tmp),
                output_dir=out, deeppocket_dir=Path(tmp), class_ckpt="c",
                seg_ckpt="s", rank=3, device=None, timeout=300, resume=True)
            self.assertEqual(res["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
