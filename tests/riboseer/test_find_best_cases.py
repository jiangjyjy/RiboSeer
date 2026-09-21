"""Mock tests for scripts/riboseer/find_best_cases.py.

Covers per-sample collection, diverse selection (one chain-pair per
PDB), the RiboSeer metric pack, and the export artefacts (metrics JSON +
per-residue CSV). No raw structures are provided, so DCC and the
structure copy exercise the graceful 'None' branches.
"""
import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.riboseer import find_best_cases as fbc  # noqa: E402

try:
    import sklearn  # noqa: F401
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


def _pred(tool_id, per_res, success=True):
    return {
        "tool_id": tool_id, "success": success,
        "binding_protein_residues": [],
        "per_residue_pae_score": {str(r): v for r, v in per_res.items()},
    }


def _write_sample(samples_dir, sid, length, gt, seq=None, resolved=None):
    (samples_dir / f"{sid}.json").write_text(json.dumps({
        "sample_id": sid, "source_pdb": sid[:4],
        "protein": {"chain_id": sid.split("_")[1], "length": length,
                    "sequence": seq or ("A" * length),
                    "resolved_residues": resolved
                    or list(range(1, length + 1))},
        "rna": {"chain_id": sid.split("_")[2], "length": 20,
                "sequence": "A" * 20},
        "interaction": {"binding_protein_residues": list(gt)},
    }), encoding="utf-8")


class TestResname(unittest.TestCase):
    def test_one_to_three(self):
        self.assertEqual(fbc._resname("MAK", 1), "MET")
        self.assertEqual(fbc._resname("MAK", 2), "ALA")
        self.assertEqual(fbc._resname("MAK", 3), "LYS")
        self.assertEqual(fbc._resname("MAK", 99), "UNK")    # out of range
        self.assertEqual(fbc._resname("MZK", 2), "GLX")     # Z -> GLX


class TestCollectAndSelect(unittest.TestCase):
    def _env(self, td):
        root = Path(td)
        samples = root / "data" / "samples"
        step4 = root / "step4"
        preds = root / "preds"
        for d in (samples, step4, preds):
            d.mkdir(parents=True, exist_ok=True)
        return root, samples.parent, samples, step4, preds

    def _good_case(self, samples, step4, preds, sid, length=6,
                   gt=(1, 2, 3), best_inversion=True):
        _write_sample(samples, sid, length, gt, seq="MAKLPQ"[:length])
        # a baseline with an imperfect ranking (R<1), RiboSeer near-perfect
        b = {1: 0.8, 2: 0.6, 3: 0.5, 4: 0.55, 5: 0.2, 6: 0.1}
        (step4 / f"{sid}.jsonl").write_text(json.dumps({
            "sample_id": sid, "predictions": [
                _pred("boltz2", b),
                _pred("equipnas", {1: 0.7, 2: 0.55, 3: 0.5,
                                   4: 0.45, 5: 0.3, 6: 0.1}),
                _pred("hdock", {1: 0.2, 2: 0.3, 3: 0.25,
                                4: 0.4, 5: 0.5, 6: 0.6}),
            ]}), encoding="utf-8")
        (preds / f"{sid}.json").write_text(json.dumps({
            "residue_ids": list(range(1, length + 1)),
            "probabilities": ([0.97, 0.95, 0.93][:len(gt)]
                              + [0.05, 0.04, 0.02][:length - len(gt)]),
        }), encoding="utf-8")

    def test_collect_sample(self):
        with tempfile.TemporaryDirectory() as td:
            root, data_dir, samples, step4, preds = self._env(td)
            self._good_case(samples, step4, preds, "4n0t_A_B")
            fs = fbc.load_predictions_dir(preds)
            rec = fbc.collect_sample("4n0t_A_B", data_dir=data_dir,
                                     step4_dir=step4, fs_preds=fs)
            self.assertIsNotNone(rec)
            self.assertEqual(rec["pdb_code"], "4n0t")
            self.assertEqual(rec["protein_chain"], "A")
            self.assertEqual(rec["rna_chain"], "B")
            self.assertGreater(rec["riboseer_pearson"],
                               rec["best_baseline_r"])
            self.assertGreater(rec["delta"], 0.0)
            # all three tools scored
            self.assertEqual(
                sum(1 for v in rec["tool_pearson"].values() if v is not None),
                3)

    def test_select_diverse_one_per_pdb(self):
        # two chain-pairs of the same PDB + one other PDB
        recs = [
            {"sample_id": "1abc_A_B", "pdb_code": "1abc", "delta": 0.30},
            {"sample_id": "1abc_C_D", "pdb_code": "1abc", "delta": 0.20},
            {"sample_id": "2xyz_A_B", "pdb_code": "2xyz", "delta": 0.25},
        ]
        sel = fbc.select_diverse(recs, top_k=8)
        ids = [r["sample_id"] for r in sel]
        self.assertEqual(ids, ["1abc_A_B", "2xyz_A_B"])    # higher-delta pdb1
        self.assertNotIn("1abc_C_D", ids)                  # de-duped
        # top_k truncates
        self.assertEqual(len(fbc.select_diverse(recs, top_k=1)), 1)


@unittest.skipUnless(HAVE_SKLEARN, "sklearn not installed (server-only)")
class TestMetricPackAndExport(unittest.TestCase):
    def _build_rec(self, td):
        root = Path(td)
        samples = root / "data" / "samples"
        step4 = root / "step4"
        preds = root / "preds"
        for d in (samples, step4, preds):
            d.mkdir(parents=True, exist_ok=True)
        sid = "4n0t_A_B"
        _write_sample(samples, sid, 6, [1, 2, 3], seq="MAKLPQ")
        (step4 / f"{sid}.jsonl").write_text(json.dumps({
            "sample_id": sid, "predictions": [
                _pred("boltz2", {1: 0.8, 2: 0.6, 3: 0.5,
                                 4: 0.55, 5: 0.2, 6: 0.1}),
                _pred("equipnas", {1: 0.7, 2: 0.55, 3: 0.5,
                                   4: 0.45, 5: 0.3, 6: 0.1}),
            ]}), encoding="utf-8")
        (preds / f"{sid}.json").write_text(json.dumps({
            "residue_ids": [1, 2, 3, 4, 5, 6],
            "probabilities": [0.97, 0.95, 0.93, 0.05, 0.04, 0.02],
        }), encoding="utf-8")
        fs = fbc.load_predictions_dir(preds)
        return root, samples.parent, step4, fbc.collect_sample(
            sid, data_dir=samples.parent, step4_dir=step4, fs_preds=fs)

    def test_metric_pack_no_structure(self):
        with tempfile.TemporaryDirectory() as td:
            root, data_dir, step4, rec = self._build_rec(td)
            pack = fbc.riboseer_metric_pack(
                rec, raw_dir=None, hit_threshold=4.0)
            for key in ("pearson_r", "spearman_r", "r_squared",
                        "auroc", "auprc", "topk_precision", "topk_recall",
                        "topk_f1", "mcc", "dcc"):
                self.assertIn(key, pack)
            # both classes present -> these are computed
            self.assertIsNotNone(pack["auroc"])
            self.assertIsNotNone(pack["topk_f1"])
            self.assertIsNotNone(pack["mcc"])
            self.assertIsNone(pack["dcc"])             # no raw structure

    def test_export_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root, data_dir, step4, rec = self._build_rec(td)
            pack = fbc.riboseer_metric_pack(rec, raw_dir=None,
                                            hit_threshold=4.0)
            export = root / "exp"
            w = fbc.export_sample(rec, pack, export_dir=export,
                                  raw_dir=None)
            self.assertIsNone(w["structure"])          # no raw-dir
            sdir = export / "4n0t_A_B"
            # metrics json
            doc = json.loads((sdir / w["metrics"]).read_text(
                encoding="utf-8"))
            self.assertEqual(doc["pdb_code"], "4n0t")
            self.assertEqual(doc["protein_length"], 6)
            self.assertEqual(set(doc["all_baselines"]), set(fbc.ALL_TOOLS))
            self.assertAlmostEqual(doc["all_baselines"]["boltz2"],
                                   rec["tool_pearson"]["boltz2"])
            self.assertIsNone(doc["all_baselines"]["nucleicnet"])  # absent
            # residues csv
            with (sdir / w["residues"]).open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]["resname"], "MET")
            self.assertEqual(rows[0]["gt_binding"], "1")
            self.assertEqual(rows[3]["gt_binding"], "0")
            self.assertEqual(rows[0]["nucleicnet_score"], "NaN")  # missing
            self.assertNotEqual(rows[0]["boltz2_score"], "NaN")
            # header has all 15 tool columns
            self.assertTrue(all(f"{t}_score" in rows[0]
                                for t in fbc.ALL_TOOLS))


class TestExportNoSklearn(unittest.TestCase):
    """export_sample doesn't touch sklearn — verify the CSV/JSON I/O
    locally with a hand-built metrics pack."""

    def test_export_artifacts_local(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            samples = root / "data" / "samples"
            step4 = root / "step4"
            preds = root / "preds"
            for d in (samples, step4, preds):
                d.mkdir(parents=True, exist_ok=True)
            sid = "4n0t_A_B"
            _write_sample(samples, sid, 6, [1, 2, 3], seq="MAKLPQ")
            (step4 / f"{sid}.jsonl").write_text(json.dumps({
                "predictions": [
                    _pred("boltz2", {1: 0.8, 2: 0.6, 3: 0.5,
                                     4: 0.55, 5: 0.2, 6: 0.1}),
                    _pred("chai1", {1: 0.7, 2: 0.55, 3: 0.5,
                                    4: 0.45, 5: 0.3, 6: 0.1}),
                ]}), encoding="utf-8")
            (preds / f"{sid}.json").write_text(json.dumps({
                "residue_ids": [1, 2, 3, 4, 5, 6],
                "probabilities": [0.97, 0.95, 0.93, 0.05, 0.04, 0.02],
            }), encoding="utf-8")
            fs = fbc.load_predictions_dir(preds)
            rec = fbc.collect_sample(sid, data_dir=samples.parent,
                                     step4_dir=step4, fs_preds=fs)
            self.assertIsNotNone(rec)
            fake_pack = {"pearson_r": rec["riboseer_pearson"],
                         "dcc": None, "auroc": 0.9}
            export = root / "exp"
            w = fbc.export_sample(rec, fake_pack, export_dir=export,
                                  raw_dir=None)
            sdir = export / sid
            doc = json.loads((sdir / w["metrics"]).read_text(
                encoding="utf-8"))
            self.assertEqual(doc["riboseer"]["auroc"], 0.9)
            self.assertEqual(set(doc["all_baselines"]), set(fbc.ALL_TOOLS))
            self.assertIsNone(doc["all_baselines"]["nucleicnet"])
            with (sdir / w["residues"]).open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]["resname"], "MET")
            self.assertEqual(rows[2]["gt_binding"], "1")    # residue 3
            self.assertEqual(rows[3]["gt_binding"], "0")    # residue 4
            self.assertEqual(rows[0]["nucleicnet_score"], "NaN")
            self.assertNotEqual(rows[0]["boltz2_score"], "NaN")
            self.assertTrue(all(f"{t}_score" in rows[0]
                                for t in fbc.ALL_TOOLS))


@unittest.skipUnless(HAVE_SKLEARN, "sklearn not installed (server-only)")
class TestMainEndToEnd(unittest.TestCase):
    def test_main_selects_and_exports(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            samples = root / "data" / "samples"
            step4 = root / "step4"
            preds = root / "preds"
            for d in (samples, step4, preds):
                d.mkdir(parents=True, exist_ok=True)

            tc = TestCollectAndSelect()
            # two PDBs, one with two chain-pairs -> dedup to 2 selected
            tc._good_case(samples, step4, preds, "1abc_A_B")
            tc._good_case(samples, step4, preds, "1abc_C_D")
            tc._good_case(samples, step4, preds, "2xyz_A_B")
            (root / "split.txt").write_text(
                "1abc_A_B\n1abc_C_D\n2xyz_A_B\n", encoding="utf-8")

            export = root / "exp"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = fbc.main([
                    "--data-dir", str(samples.parent),
                    "--step4-dir", str(step4),
                    "--predictions-dir", str(preds),
                    "--split-file", str(root / "split.txt"),
                    "--export-dir", str(export),
                    "--top-k", "8"])
            self.assertEqual(rc, 0)
            text = buf.getvalue()
            self.assertIn("Best Cases for Visualization", text)
            self.assertIn("Exported to:", text)
            # one PDB de-duplicated -> exactly 2 sample dirs
            dirs = sorted(p.name for p in export.iterdir() if p.is_dir())
            self.assertEqual(len(dirs), 2)
            self.assertIn("2xyz_A_B", dirs)
            # each exported dir has metrics + residues
            for d in dirs:
                self.assertTrue((export / d / f"{d}_metrics.json").is_file())
                self.assertTrue((export / d / f"{d}_residues.csv").is_file())


if __name__ == "__main__":
    unittest.main()
