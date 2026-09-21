"""Mock tests for the NucleicNet baseline scripts.

The heavy bits (the NucleicNet ``Server`` GPU flow, scipy) are avoided/mocked;
these cover the logic that decides correctness:

  * parse: non-site class resolution, voxel site-prob = 1 - P(Nonsite),
    voxel->nearest-residue aggregation (numpy fallback path), ToolPrediction
    assembly, and the step4 JSONL merge that must preserve other tools.
  * batch: sample-list reading (txt + csv), resume skipping, and the
    orchestration loop with a fake Server (no torch / GPU).
"""
from __future__ import annotations

import json
import pickle
import signal
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import step4_tool_adapters.external.nucleicnet_parse as pn  # noqa: E402
import scripts.riboseer.run_nucleicnet_batch as nb  # noqa: E402


# ----------------------------------------------------------------------------
# parse: class-index resolution
# ----------------------------------------------------------------------------
class TestResolveNonsite(unittest.TestCase):
    def test_plain_name(self):
        d = {"Base": 0, "P": 1, "R": 2, "Nonsite": 3}
        self.assertEqual(pn.resolve_nonsite_index(d), 3)

    def test_case_and_separator_insensitive(self):
        d = {"base": 0, "non-site": 2}
        self.assertEqual(pn.resolve_nonsite_index(d, nonsite_name="Nonsite"), 2)

    def test_override_wins(self):
        d = {"Nonsite": 3}
        self.assertEqual(pn.resolve_nonsite_index(d, override=1), 1)

    def test_missing_raises(self):
        with self.assertRaises(KeyError):
            pn.resolve_nonsite_index({"Base": 0})


# ----------------------------------------------------------------------------
# parse: voxel site probabilities
# ----------------------------------------------------------------------------
class TestVoxelSiteProbs(unittest.TestCase):
    def _df(self):
        # 2 voxels; classes 0..3, index 3 = Nonsite.
        return pd.DataFrame({
            "x": [0.0, 10.0], "y": [0.0, 10.0], "z": [0.0, 10.0],
            "Smoothened_0": [0.5, 0.0], "Smoothened_1": [0.2, 0.0],
            "Smoothened_2": [0.1, 0.1], "Smoothened_3": [0.2, 0.9],
            "Raw_0": [0.4, 0.0], "Raw_1": [0.1, 0.0],
            "Raw_2": [0.1, 0.0], "Raw_3": [0.4, 1.0],
        })

    def test_smoothened_site_is_one_minus_nonsite(self):
        coords, site = pn.compute_voxel_site_probs(
            self._df(), nonsite_index=3, prediction_type="Smoothened")
        self.assertEqual(coords.shape, (2, 3))
        np.testing.assert_allclose(site, [0.8, 0.1])

    def test_raw_family(self):
        _, site = pn.compute_voxel_site_probs(
            self._df(), nonsite_index=3, prediction_type="Raw")
        np.testing.assert_allclose(site, [0.6, 0.0])

    def test_missing_column_raises(self):
        df = self._df().drop(columns=["Smoothened_3"])
        with self.assertRaises(KeyError):
            pn.compute_voxel_site_probs(df, nonsite_index=3)

    def test_missing_coord_raises(self):
        df = self._df().drop(columns=["z"])
        with self.assertRaises(KeyError):
            pn.compute_voxel_site_probs(df, nonsite_index=3)


# ----------------------------------------------------------------------------
# parse: voxel -> residue aggregation
# ----------------------------------------------------------------------------
class TestAggregatePerResidue(unittest.TestCase):
    def setUp(self):
        # residue 1 at origin (1 atom), residue 2 far away at x=100.
        self.atoms = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        self.resids = np.array([1, 2])

    def test_max_aggregation_and_assignment(self):
        # two voxels near residue 1 (site 0.3 and 0.9), one near residue 2 (0.4)
        vox = np.array([[1.0, 0, 0], [2.0, 0, 0], [101.0, 0, 0]])
        site = np.array([0.3, 0.9, 0.4])
        out = pn.aggregate_per_residue(vox, site, self.atoms, self.resids,
                                       radius=6.0, agg="max")
        self.assertAlmostEqual(out[1], 0.9)
        self.assertAlmostEqual(out[2], 0.4)

    def test_mean_aggregation(self):
        vox = np.array([[1.0, 0, 0], [2.0, 0, 0]])
        site = np.array([0.2, 0.8])
        out = pn.aggregate_per_residue(vox, site, self.atoms, self.resids,
                                       radius=6.0, agg="mean")
        self.assertAlmostEqual(out[1], 0.5)
        self.assertAlmostEqual(out[2], 0.0)  # no voxel reached residue 2

    def test_radius_excludes_far_voxels(self):
        # voxel 7A from residue 1 -> excluded at radius 6.
        vox = np.array([[7.0, 0, 0]])
        site = np.array([0.99])
        out = pn.aggregate_per_residue(vox, site, self.atoms, self.resids,
                                       radius=6.0, agg="max")
        self.assertEqual(out[1], 0.0)
        self.assertEqual(out[2], 0.0)

    def test_all_residues_present_even_with_no_voxels(self):
        out = pn.aggregate_per_residue(
            np.empty((0, 3)), np.empty((0,)), self.atoms, self.resids)
        self.assertEqual(set(out), {1, 2})
        self.assertEqual(out[1], 0.0)
        self.assertEqual(out[2], 0.0)

    def test_numpy_fallback_matches_when_scipy_absent(self):
        # Force the brute-force path by hiding scipy, compare to a hand calc.
        vox = np.array([[1.0, 0, 0], [99.0, 0, 0]])
        site = np.array([0.7, 0.6])
        with mock.patch.dict(sys.modules, {"scipy": None, "scipy.spatial": None}):
            out = pn.aggregate_per_residue(vox, site, self.atoms, self.resids,
                                           radius=6.0, agg="max")
        self.assertAlmostEqual(out[1], 0.7)  # voxel 0 nearest atom 0 (res 1)
        self.assertAlmostEqual(out[2], 0.6)  # voxel 1 nearest atom 1 (res 2)


# ----------------------------------------------------------------------------
# parse: ToolPrediction + merge
# ----------------------------------------------------------------------------
class TestBuildAndMerge(unittest.TestCase):
    def test_build_prediction_thresholds(self):
        pred = pn.build_prediction(
            "1abc_A_E", {1: 0.9, 2: 0.1, 3: 0.55}, threshold=0.5)
        self.assertEqual(pred.tool_id, "nucleicnet")
        self.assertEqual(pred.category, "C")
        self.assertTrue(pred.success)
        self.assertEqual(pred.binding_protein_residues, [1, 3])
        self.assertEqual(pred.per_residue_confidence, {1: 0.9, 2: 0.1, 3: 0.55})

    def test_merge_preserves_other_tools_and_replaces_self(self):
        with tempfile.TemporaryDirectory() as tmp:
            step4 = Path(tmp)
            sid = "1abc_A_E"
            # pre-existing file with another tool + a stale nucleicnet entry
            (step4 / f"{sid}.jsonl").write_text(json.dumps({
                "sample_id": sid,
                "tools_run": ["boltz2", "nucleicnet"],
                "predictions": [
                    {"tool_id": "boltz2", "category": "A", "sample_id": sid,
                     "success": True, "binding_protein_residues": [5]},
                    {"tool_id": "nucleicnet", "category": "C", "sample_id": sid,
                     "success": True, "per_residue_confidence": {1: 0.1}},
                ],
            }) + "\n", encoding="utf-8")
            new_pred = pn.build_prediction(sid, {2: 0.8}, threshold=0.5)
            pn.merge_into_set(step4, sid, new_pred)

            rec = json.loads((step4 / f"{sid}.jsonl").read_text(encoding="utf-8"))
            tools = {p["tool_id"] for p in rec["predictions"]}
            self.assertEqual(tools, {"boltz2", "nucleicnet"})
            nn = next(p for p in rec["predictions"] if p["tool_id"] == "nucleicnet")
            self.assertEqual(nn["binding_protein_residues"], [2])
            self.assertEqual(sorted(rec["tools_run"]), ["boltz2", "nucleicnet"])

    def test_parse_one_sample_end_to_end(self):
        """pkl + class dict + PDB -> ToolPrediction, with gemmi mocked."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            sid = "1abc_A_E"
            sx = pn.sxpr_dir(out, sid)
            sx.mkdir(parents=True)
            df = pd.DataFrame({
                "x": [1.0, 2.0, 101.0], "y": [0.0, 0.0, 0.0],
                "z": [0.0, 0.0, 0.0],
                "Smoothened_0": [0.7, 0.8, 0.05],
                "Smoothened_1": [0.1, 0.1, 0.05],
                "Smoothened_2": [0.1, 0.05, 0.05],
                "Smoothened_3": [0.1, 0.05, 0.85],  # Nonsite
            })
            df.to_pickle(sx / "Result_EnsembleAvDf.pkl")
            with (sx / "ClassName_ClassIndex_Dict.pkl").open("wb") as f:
                pickle.dump({"Base": 0, "P": 1, "R": 2, "Nonsite": 3}, f)

            inputs = out / "nucleicnet_inputs"
            inputs.mkdir()
            (inputs / f"{sid}.pdb").write_text("dummy", encoding="utf-8")

            atoms = np.array([[0.0, 0, 0], [100.0, 0, 0]])
            resids = np.array([1, 2])
            with mock.patch.object(pn, "load_protein_atoms",
                                   return_value=(atoms, resids)):
                pred = pn.parse_one_sample(
                    sid, output_dir=out, inputs_dir=inputs,
                    prediction_type="Smoothened", nonsite_name="Nonsite",
                    nonsite_index_override=None, radius=6.0, agg="max",
                    threshold=0.5)
            # residue 1: max(1-0.1, 1-0.05)=0.95 -> binding; residue 2: 0.15
            self.assertEqual(pred.binding_protein_residues, [1])
            self.assertAlmostEqual(pred.per_residue_confidence[1], 0.95)
            self.assertAlmostEqual(pred.per_residue_confidence[2], 0.15)


# ----------------------------------------------------------------------------
# batch: sample list + resume + orchestration
# ----------------------------------------------------------------------------
class TestBatchHelpers(unittest.TestCase):
    def test_read_sample_list_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "l.txt"
            p.write_text("# comment\n1abc_A_E\n\n2def_B_F extra\n",
                         encoding="utf-8")
            self.assertEqual(nb.read_sample_list(p), ["1abc_A_E", "2def_B_F"])

    def test_read_sample_list_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "l.csv"
            p.write_text("priority,sample_id,prot_len\n1,1abc_A_E,80\n"
                         "2,2def_B_F,90\n", encoding="utf-8")
            self.assertEqual(nb.read_sample_list(p), ["1abc_A_E", "2def_B_F"])

    def test_load_sample_json_direct_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "2czj_E_F.json").write_text(
                json.dumps({"sample_id": "2czj_E_F"}), encoding="utf-8")
            got = nb.load_sample_json(d, "2czj_E_F")
            self.assertEqual(got["sample_id"], "2czj_E_F")

    def test_load_sample_json_case_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "1ABC_A_E.json").write_text(
                json.dumps({"sample_id": "1ABC_A_E"}), encoding="utf-8")
            got = nb.load_sample_json(d, "1abc_a_e")
            self.assertEqual(got["sample_id"], "1ABC_A_E")

    def test_load_sample_json_strips_invisible_chars(self):
        # A BOM / zero-width space sneaks into the id but the file is clean.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "2czj_E_F.json").write_text(
                json.dumps({"sample_id": "2czj_E_F"}), encoding="utf-8")
            for bad in ("﻿2czj_E_F", "2czj_E_F​", "  2czj_E_F\r"):
                got = nb.load_sample_json(d, bad)
                self.assertEqual(got["sample_id"], "2czj_E_F")

    def test_load_sample_json_parent_samples_child(self):
        # --samples-dir points at processed_quality/, JSONs live in samples/.
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp)
            (proc / "samples").mkdir()
            (proc / "samples" / "2czj_E_F.json").write_text(
                json.dumps({"sample_id": "2czj_E_F"}), encoding="utf-8")
            got = nb.load_sample_json(proc, "2czj_E_F")
            self.assertEqual(got["sample_id"], "2czj_E_F")

    def test_clean_sample_id(self):
        self.assertEqual(nb.clean_sample_id("﻿ 2czj_E_F \r"), "2czj_E_F")
        self.assertEqual(nb.clean_sample_id("2czj_E_F​"), "2czj_E_F")

    def test_load_sample_json_diagnostic_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "2czj_E_F.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileNotFoundError) as cm:
                nb.load_sample_json(d, "2czj_X_Y")
            msg = str(cm.exception)
            self.assertIn("dir_exists=True", msg)
            self.assertIn("json_files=1", msg)
            self.assertIn("near_matches=", msg)
            self.assertIn("2czj_E_F.json", msg)  # same-PDB near match surfaced

    def test_resume_skips_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs = Path(tmp) / "out"
            done = nb.sxpr_result_path(outputs / "1abc_A_E")
            done.parent.mkdir(parents=True)
            done.write_text("x", encoding="utf-8")
            res = nb.run_one_sample(
                "1abc_A_E", samples_dir=Path(tmp), raw_dir=Path(tmp),
                inputs_dir=Path(tmp) / "in", outputs_dir=outputs, resume=True)
            self.assertEqual(res["status"], "skipped")

    def test_run_one_sample_orchestration_with_fake_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            samples_dir = base / "samples"
            samples_dir.mkdir()
            sid = "1abc_A_E"
            (samples_dir / f"{sid}.json").write_text(json.dumps({
                "sample_id": sid, "source_pdb": "1abc",
                "protein": {"chain_id": "A"},
            }), encoding="utf-8")
            raw_dir = base / "raw"
            raw_dir.mkdir()
            (raw_dir / "1abc.pdb").write_text("RAW", encoding="utf-8")
            outputs = base / "out"
            inputs = base / "in"

            captured = {}

            def fake_extract(raw_path, chain_id, out_path):
                captured["raw"] = Path(raw_path)
                captured["chain"] = chain_id
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                Path(out_path).write_text("PDB", encoding="utf-8")
                return 42

            class FakeServer:
                def __init__(self, **kw):
                    captured["server_kwargs"] = kw
                    self.folder = Path(kw["DIR_ServerFolder"])

                def SimpleSanitise(self, DIR_InputPdbFile):
                    captured["input_pdb"] = DIR_InputPdbFile

                def MakeHalo(self): pass
                def MakeDssp(self): pass
                def MakeFeature(self): pass
                def MakeDummyTypi(self): pass

                def MakeSXPR(self):
                    # emulate the on-disk product the resume check looks for
                    res = nb.sxpr_result_path(self.folder)
                    res.parent.mkdir(parents=True, exist_ok=True)
                    res.write_text("pkl", encoding="utf-8")

            with mock.patch.object(nb, "extract_protein_chain_pdb", fake_extract), \
                 mock.patch.object(nb, "Server", FakeServer):
                res = nb.run_one_sample(
                    sid, samples_dir=samples_dir, raw_dir=raw_dir,
                    inputs_dir=inputs, outputs_dir=outputs, resume=False)

            self.assertEqual(res["status"], "ok", msg=res.get("error"))
            self.assertEqual(res["n_residues"], 42)
            self.assertEqual(captured["chain"], "A")
            self.assertEqual(captured["raw"].name, "1abc.pdb")
            self.assertTrue(captured["server_kwargs"]["SaveCleansed"])
            self.assertTrue(captured["server_kwargs"]["Select_HeavyAtoms"])
            self.assertTrue(nb.sxpr_result_path(outputs / sid).is_file())

    def test_run_one_sample_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # no sample json -> load_sample_json raises -> failure recorded
            res = nb.run_one_sample(
                "ghost_A_E", samples_dir=base / "nope", raw_dir=base,
                inputs_dir=base / "in", outputs_dir=base / "out", resume=False)
            self.assertEqual(res["status"], "failed")
            self.assertIsNotNone(res["error"])
            self.assertFalse(res["timed_out"])


# ----------------------------------------------------------------------------
# batch: timeout protection
# ----------------------------------------------------------------------------
class TestTimeout(unittest.TestCase):
    def test_time_limit_zero_is_noop(self):
        with nb.time_limit(0):
            pass  # must not raise / arm an alarm

    def test_time_limit_none_is_noop(self):
        with nb.time_limit(None):
            pass

    @unittest.skipUnless(hasattr(signal, "SIGALRM"),
                         "SIGALRM only available on Unix")
    def test_time_limit_fires_on_slow_block(self):
        with self.assertRaises(nb.SampleTimeout):
            with nb.time_limit(1):
                time.sleep(5)

    @unittest.skipUnless(hasattr(signal, "SIGALRM"),
                         "SIGALRM only available on Unix")
    def test_time_limit_restores_handler_and_cancels_alarm(self):
        before = signal.getsignal(signal.SIGALRM)
        with nb.time_limit(30):
            pass
        # handler restored and no alarm left pending (alarm(0) returns 0)
        self.assertEqual(signal.getsignal(signal.SIGALRM), before)
        self.assertEqual(signal.alarm(0), 0)

    def test_run_one_sample_marks_timeout(self):
        """A hang during the Server flow -> failed + timed_out=True, batch goes on."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            samples_dir = base / "samples"
            samples_dir.mkdir()
            sid = "1abc_A_E"
            (samples_dir / f"{sid}.json").write_text(json.dumps({
                "sample_id": sid, "source_pdb": "1abc",
                "protein": {"chain_id": "A"},
            }), encoding="utf-8")
            raw_dir = base / "raw"; raw_dir.mkdir()
            (raw_dir / "1abc.pdb").write_text("RAW", encoding="utf-8")

            def fake_extract(raw_path, chain_id, out_path):
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                Path(out_path).write_text("PDB", encoding="utf-8")
                return 10

            class HangingServer:
                def __init__(self, **kw):
                    pass

                def SimpleSanitise(self, DIR_InputPdbFile):
                    pass

                def MakeHalo(self): pass
                def MakeDssp(self): pass
                def MakeFeature(self): pass
                def MakeDummyTypi(self): pass

                def MakeSXPR(self):
                    # emulate the time_limit guard tripping on a wedged step
                    raise nb.SampleTimeout("timed out after 600s")

            with mock.patch.object(nb, "extract_protein_chain_pdb", fake_extract), \
                 mock.patch.object(nb, "Server", HangingServer):
                res = nb.run_one_sample(
                    sid, samples_dir=samples_dir, raw_dir=raw_dir,
                    inputs_dir=base / "in", outputs_dir=base / "out",
                    resume=False, timeout=600)

            self.assertEqual(res["status"], "failed")
            self.assertTrue(res["timed_out"])
            self.assertIn("SampleTimeout", res["error"])


if __name__ == "__main__":
    unittest.main()
