"""Mock tests for the GraphBind baseline scripts.

GraphBind / HHblits are never invoked. Parser tests pin the CSV parsing,
Residue_ID -> probability mapping, Binary-vs-threshold binding, the alignment
warning, and the step4 merge. Batch tests cover the copy-into-querypath,
command construction, resume, and the orchestration loop with run_predict
mocked.
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

import step4_tool_adapters.external.graphbind_parse as pg  # noqa: E402
import scripts.riboseer.run_graphbind_batch as gb  # noqa: E402


_CSV = (
    ",Residue_ID,Residue,Probability,Binary\n"
    "0,9,E,0.001,0\n"
    "1,10,P,0.000,0\n"
    "2,11,K,0.873,1\n"
    "3,12,R,0.654,1\n"
)


def write_result(results_dir: Path, sid: str, text: str = _CSV) -> Path:
    d = results_dir / sid
    d.mkdir(parents=True)
    (d / "RNA-binding_result.csv").write_text(text, encoding="utf-8")
    return d


# ----------------------------------------------------------------------------
# parser: CSV parsing
# ----------------------------------------------------------------------------
class TestParseCsv(unittest.TestCase):
    def test_per_residue_and_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = write_result(Path(tmp), "1abc_A_E")
            per, binding = pg.parse_graphbind_csv(d / "RNA-binding_result.csv")
            self.assertEqual(per, {9: 0.001, 10: 0.0, 11: 0.873, 12: 0.654})
            self.assertEqual(binding, [11, 12])

    def test_threshold_overrides_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = write_result(Path(tmp), "1abc_A_E")
            per, binding = pg.parse_graphbind_csv(
                d / "RNA-binding_result.csv", prob_threshold=0.7)
            # only prob > 0.7 -> residue 11 (0.873); 12 (0.654) drops
            self.assertEqual(binding, [11])

    def test_clamp_and_bad_rows_skipped(self):
        text = (",Residue_ID,Residue,Probability,Binary\n"
                "0,5,A,1.4,1\n"      # prob clamps to 1.0
                "1,x,B,0.2,0\n"       # bad Residue_ID -> skipped
                "2,0,C,0.9,1\n")      # rid < 1 -> skipped
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.csv"
            p.write_text(text, encoding="utf-8")
            per, binding = pg.parse_graphbind_csv(p)
            self.assertEqual(per, {5: 1.0})
            self.assertEqual(binding, [5])

    def test_residue_vs_residue_id_disambiguation(self):
        # 'Residue' (one-letter) must not be mistaken for 'Residue_ID'.
        text = (",Residue,Residue_ID,Probability,Binary\n"
                "0,E,42,0.9,1\n")
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.csv"
            p.write_text(text, encoding="utf-8")
            per, binding = pg.parse_graphbind_csv(p)
            self.assertEqual(per, {42: 0.9})

    def test_missing_columns_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.csv"
            p.write_text(",foo,bar\n0,1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                pg.parse_graphbind_csv(p)


# ----------------------------------------------------------------------------
# parser: find CSV, build, merge, end-to-end
# ----------------------------------------------------------------------------
class TestBuildMergeParse(unittest.TestCase):
    def test_find_result_csv_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "s"
            d.mkdir()
            alt = d / "graphbind_result.csv"
            alt.write_text("x", encoding="utf-8")
            self.assertEqual(pg.find_result_csv(d).resolve(), alt.resolve())

    def test_build_prediction(self):
        pred = pg.build_prediction("1abc_A_E", {9: 0.1, 11: 0.87}, [11])
        self.assertEqual(pred.tool_id, "graphbind")
        self.assertEqual(pred.category, "C")
        self.assertEqual(pred.per_residue_confidence, {9: 0.1, 11: 0.87})
        self.assertEqual(pred.binding_protein_residues, [11])

    def test_merge_preserves_other_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            step4 = Path(tmp)
            sid = "1abc_A_E"
            (step4 / f"{sid}.jsonl").write_text(json.dumps({
                "sample_id": sid, "tools_run": ["equipnas", "graphbind"],
                "predictions": [
                    {"tool_id": "equipnas", "category": "C", "sample_id": sid,
                     "success": True, "binding_protein_residues": [3]},
                    {"tool_id": "graphbind", "category": "C", "sample_id": sid,
                     "success": True, "per_residue_confidence": {1: 0.1}},
                ],
            }) + "\n", encoding="utf-8")
            new = pg.build_prediction(sid, {7: 0.9}, [7])
            pg.merge_into_set(step4, sid, new)
            rec = json.loads((step4 / f"{sid}.jsonl").read_text(encoding="utf-8"))
            self.assertEqual({p["tool_id"] for p in rec["predictions"]},
                             {"equipnas", "graphbind"})
            g = next(p for p in rec["predictions"] if p["tool_id"] == "graphbind")
            self.assertEqual(g["binding_protein_residues"], [7])

    def test_parse_one_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            write_result(results, "1abc_A_E")
            pred = pg.parse_one_sample(
                "1abc_A_E", results_dir=results, samples_dir=None,
                prob_threshold=None)
            self.assertEqual(pred.binding_protein_residues, [11, 12])
            self.assertEqual(len(pred.per_residue_confidence), 4)

    def test_parse_one_sample_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                pg.parse_one_sample("ghost", results_dir=Path(tmp),
                                    samples_dir=None, prob_threshold=None)

    def test_oob_warning(self):
        # Residue_ID 9999 exceeds protein length -> warning, but kept.
        text = ",Residue_ID,Residue,Probability,Binary\n0,9999,E,0.9,1\n"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            results = tmp / "res"
            d = results / "1abc_A_E"
            d.mkdir(parents=True)
            (d / "RNA-binding_result.csv").write_text(text, encoding="utf-8")
            samples = tmp / "samples"
            samples.mkdir()
            (samples / "1abc_A_E.json").write_text(json.dumps({
                "sample_id": "1abc_A_E", "protein": {"length": 100}}),
                encoding="utf-8")
            with self.assertLogs(pg.logger, level="WARNING") as cm:
                pred = pg.parse_one_sample(
                    "1abc_A_E", results_dir=results, samples_dir=samples,
                    prob_threshold=None)
            self.assertIn(9999, pred.per_residue_confidence)  # kept
            self.assertTrue(any("outside" in m for m in cm.output))


# ----------------------------------------------------------------------------
# batch: paths, command, copy-into-querypath, resume, orchestration
# ----------------------------------------------------------------------------
class TestBatch(unittest.TestCase):
    def test_paths(self):
        out = Path("/o")
        self.assertEqual(gb.query_dir(out, "s").as_posix(), "/o/s")
        self.assertEqual(gb.result_csv(out, "s").as_posix(),
                         "/o/s/RNA-binding_result.csv")

    def test_build_predict_cmd(self):
        cmd = gb.build_predict_cmd(
            querypath=Path("/o/s"), filename="s.pdb", chainid="A",
            ligands="RNA", cpu=4)
        self.assertEqual(cmd[1:], [
            "prediction.py", "--querypath", str(Path("/o/s")),
            "--filename", "s.pdb", "--chainid", "A", "--ligands", "RNA",
            "--cpu", "4"])

    def test_is_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertFalse(gb.is_done(out, "s"))
            gb.result_csv(out, "s").parent.mkdir(parents=True)
            gb.result_csv(out, "s").write_text("x", encoding="utf-8")
            self.assertTrue(gb.is_done(out, "s"))

    def test_run_one_sample_success_copies_and_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            inputs = base / "inputs"; inputs.mkdir()
            sid = "1abc_A_E"
            (inputs / f"{sid}.pdb").write_text("PDBDATA", encoding="utf-8")
            out = base / "out"
            scripts = base / "gb" / "scripts"; scripts.mkdir(parents=True)
            captured = {}

            def fake_run_predict(cmd, cwd, env, timeout):
                captured["cmd"] = cmd
                captured["cwd"] = Path(cwd)
                # GraphBind must see the copied PDB in querypath
                qp = out / sid
                captured["staged_exists"] = (qp / f"{sid}.pdb").is_file()
                (qp / "RNA-binding_result.csv").write_text(
                    ",Residue_ID,Residue,Probability,Binary\n0,1,A,0.9,1\n",
                    encoding="utf-8")
                return 0, "ok\n", False

            with mock.patch.object(gb, "run_predict", fake_run_predict):
                res = gb.run_one_sample(
                    sid, inputs_dir=inputs, output_dir=out, scripts_dir=scripts,
                    chainid="A", ligands="RNA", cpu=4, timeout=600, resume=False)
            self.assertEqual(res["status"], "ok", msg=res.get("error"))
            self.assertTrue(captured["staged_exists"])
            self.assertEqual(captured["cwd"], scripts)
            self.assertIn("--querypath", captured["cmd"])
            self.assertTrue((out / sid / "prediction.log").is_file())

    def test_run_one_sample_missing_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            res = gb.run_one_sample(
                "ghost_A_E", inputs_dir=base / "inputs", output_dir=base / "out",
                scripts_dir=base, chainid="A", ligands="RNA", cpu=4,
                timeout=600, resume=False)
            self.assertEqual(res["status"], "failed")
            self.assertIn("input PDB missing", res["error"])

    def test_run_one_sample_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            inputs = base / "inputs"; inputs.mkdir()
            sid = "1abc_A_E"
            (inputs / f"{sid}.pdb").write_text("X", encoding="utf-8")
            out = base / "out"

            def fake_run_predict(cmd, cwd, env, timeout):
                return None, "...", True

            with mock.patch.object(gb, "run_predict", fake_run_predict):
                res = gb.run_one_sample(
                    sid, inputs_dir=inputs, output_dir=out, scripts_dir=base,
                    chainid="A", ligands="RNA", cpu=4, timeout=600, resume=False)
            self.assertEqual(res["status"], "failed")
            self.assertTrue(res["timed_out"])

    def test_run_one_sample_resume_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            gb.result_csv(out, "1abc_A_E").parent.mkdir(parents=True)
            gb.result_csv(out, "1abc_A_E").write_text("x", encoding="utf-8")
            res = gb.run_one_sample(
                "1abc_A_E", inputs_dir=Path(tmp), output_dir=out,
                scripts_dir=Path(tmp), chainid="A", ligands="RNA", cpu=4,
                timeout=600, resume=True)
            self.assertEqual(res["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
