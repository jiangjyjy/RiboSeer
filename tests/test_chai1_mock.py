"""Mock tests for Chai-1 adapter.

Mirrors the structure of test_boltz2_mock.py since the two adapters
share most helpers (sequence cleaning, seq_map sidecar, per-residue
pLDDT extraction). Coverage:

  - FASTA writer: chain-tagged headers, no whitespace bleed
  - sample_id sanitisation
  - runner script generation: required imports + parameters baked in
  - scores JSON parsing (happy / missing keys / corrupt / case-insensitive)
  - PAE npy mean reader
  - parse_output end-to-end with synthesised CIF + scores JSON
  - prepare_input writes FASTA + seq_map + accepts cleaning
  - run_tool wraps `python run_chai1.py` inside the chai1 conda env
  - predict() failure record on missing input
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import gemmi  # noqa: E402
import numpy as np  # noqa: E402

from step4_tool_adapters.adapters.chai1_adapter import (  # noqa: E402
    Chai1Adapter, _SEQ_MAP_FILENAME, _sanitize_job_name,
    parse_chai1_scores_json, read_pae_mean_from_npy, read_seq_map,
    write_chai1_fasta, write_runner_script, write_seq_map,
)
from step4_tool_adapters.tool_runner import ToolRunResult  # noqa: E402


# ---------- helpers --------------------------------------------------------


def _add_atom(res: gemmi.Residue, name: str, x: float, y: float, z: float,
              element: str, b_iso: float = 80.0) -> None:
    a = gemmi.Atom()
    a.name = name
    a.pos = gemmi.Position(x, y, z)
    a.element = gemmi.Element(element)
    a.b_iso = b_iso
    a.occ = 1.0
    res.add_atom(a)


def _make_synthetic_complex_cif(path: Path) -> None:
    """Tiny RNA-protein complex; protein res 1 contacts RNA nt 1."""
    s = gemmi.Structure()
    s.cell = gemmi.UnitCell()
    m = gemmi.Model("1")

    chA = gemmi.Chain("A")
    r1 = gemmi.Residue(); r1.name = "LYS"
    r1.seqid = gemmi.SeqId(1, " "); r1.label_seq = 1
    _add_atom(r1, "N",  0.0, 0.0, 0.0, "N", b_iso=85.0)
    _add_atom(r1, "CA", 1.5, 0.0, 0.0, "C", b_iso=85.0)
    chA.add_residue(r1)

    r2 = gemmi.Residue(); r2.name = "ALA"
    r2.seqid = gemmi.SeqId(2, " "); r2.label_seq = 2
    _add_atom(r2, "N", 50.0, 50.0, 50.0, "N", b_iso=70.0)
    _add_atom(r2, "CA", 51.0, 50.0, 50.0, "C", b_iso=70.0)
    chA.add_residue(r2)
    m.add_chain(chA)

    chB = gemmi.Chain("B")
    rB1 = gemmi.Residue(); rB1.name = "A"
    rB1.seqid = gemmi.SeqId(1, " "); rB1.label_seq = 1
    _add_atom(rB1, "N1", 3.0, 0.0, 0.0, "N", b_iso=55.0)
    chB.add_residue(rB1)

    rB2 = gemmi.Residue(); rB2.name = "G"
    rB2.seqid = gemmi.SeqId(2, " "); rB2.label_seq = 2
    _add_atom(rB2, "N1", 60.0, 60.0, 60.0, "N", b_iso=40.0)
    chB.add_residue(rB2)
    m.add_chain(chB)

    s.add_model(m)
    s.make_mmcif_document().write_file(str(path))


def _config(work_dir: Path) -> dict:
    return {
        "tools": {
            "chai1": {
                "conda_env": "chai1",
                "timeout": 900,
                "device": "cuda:0",
                "num_trunk_recycles": 3,
                "num_diffn_timesteps": 200,
                "seed": 42,
                "use_esm_embeddings": True,
                "use_msa_server": False,
            },
        },
        "contact_threshold": 4.5,
        "log_dir": str(work_dir / "logs"),
    }


def _ok_run() -> ToolRunResult:
    return ToolRunResult(
        command="fake", cwd=None, returncode=0,
        stdout="", stderr="", runtime_seconds=10.0, log_path=None,
    )


def _sample_json() -> dict:
    # Same minimal sample shape boltz2's tests use; both adapters share
    # the same MIN_PROTEIN_LEN / MIN_RNA_LEN gates.
    return {
        "sample_id": "1un6_B_F",
        "source_pdb": "1un6",
        "protein": {"chain_id": "B", "sequence": "MKTVLAGICK", "length": 10},
        "rna": {"chain_id": "F", "sequence": "GCCGGCCAU", "length": 9},
    }


# ---------- FASTA writer --------------------------------------------------


class TestWriteChai1Fasta(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_basic_format(self):
        path = self.tmp / "x.fasta"
        write_chai1_fasta(path, "MKTVL", "GCCAU")
        text = path.read_text(encoding="utf-8")
        # Chain-tagged headers — exactly what Chai-1 expects.
        self.assertIn(">protein|chain_A", text)
        self.assertIn(">rna|chain_B", text)
        self.assertIn("MKTVL", text)
        self.assertIn("GCCAU", text)

    def test_chain_ids_overridable(self):
        path = self.tmp / "x.fasta"
        write_chai1_fasta(path, "MKTVL", "GCCAU",
                          protein_chain="P", rna_chain="R")
        text = path.read_text(encoding="utf-8")
        self.assertIn(">protein|chain_P", text)
        self.assertIn(">rna|chain_R", text)

    def test_empty_seq_rejected(self):
        path = self.tmp / "x.fasta"
        with self.assertRaises(ValueError):
            write_chai1_fasta(path, "", "GCCAU")
        with self.assertRaises(ValueError):
            write_chai1_fasta(path, "MKTVL", "")

    def test_whitespace_seq_rejected(self):
        path = self.tmp / "x.fasta"
        with self.assertRaises(ValueError):
            write_chai1_fasta(path, "MKT VL", "GCCAU")
        with self.assertRaises(ValueError):
            write_chai1_fasta(path, "MKTVL", "GC\nCAU")


class TestSanitizeJobName(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(_sanitize_job_name("1un6_B_F"), "1un6_B_F")

    def test_dash(self):
        self.assertEqual(_sanitize_job_name("2wj8_A_-a"), "2wj8_A__a")

    def test_only_invalid_chars(self):
        self.assertEqual(_sanitize_job_name("---"), "sample")


# ---------- runner script generation --------------------------------------


class TestRunnerScript(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_runner_contains_required_pieces(self):
        runner = write_runner_script(
            self.tmp,
            sample_id="abc",
            fasta_path=self.tmp / "x.fasta",
            output_dir=self.tmp / "out",
            num_trunk_recycles=3, num_diffn_timesteps=200,
            seed=42, device="cuda:0",
            use_esm_embeddings=True, use_msa_server=False,
        )
        self.assertTrue(runner.is_file())
        text = runner.read_text(encoding="utf-8")
        # Imports the right symbol
        self.assertIn("from chai_lab.chai1 import run_inference", text)
        # All config baked in
        self.assertIn("num_trunk_recycles=3", text)
        self.assertIn("num_diffn_timesteps=200", text)
        self.assertIn("seed=42", text)
        self.assertIn("'cuda:0'", text)
        self.assertIn("use_esm_embeddings=True", text)
        self.assertIn("use_msa_server=False", text)
        # Exit-code contract: 0 success, 1 failure
        self.assertIn("return 1", text)
        self.assertIn("return 0", text)

    def test_runner_is_idempotent_overwrite(self):
        runner = write_runner_script(
            self.tmp, sample_id="a",
            fasta_path=self.tmp / "x.fasta",
            output_dir=self.tmp / "out",
        )
        first = runner.read_text(encoding="utf-8")
        # Re-run with different params — file should be rewritten.
        write_runner_script(
            self.tmp, sample_id="a",
            fasta_path=self.tmp / "x.fasta",
            output_dir=self.tmp / "out",
            num_trunk_recycles=5,
        )
        second = runner.read_text(encoding="utf-8")
        self.assertNotEqual(first, second)
        self.assertIn("num_trunk_recycles=5", second)


# ---------- scores JSON ---------------------------------------------------


class TestParseScoresJson(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_happy(self):
        path = self.tmp / "s.json"
        path.write_text(json.dumps({
            "plddt": 78.4, "iptm": 0.62, "ptm": 0.71, "pae": 4.5,
            "aggregate_score": 0.84,
        }))
        out = parse_chai1_scores_json(path)
        self.assertAlmostEqual(out["plddt"], 78.4)
        self.assertAlmostEqual(out["iptm"], 0.62)
        self.assertAlmostEqual(out["aggregate_score"], 0.84)

    def test_case_insensitive_keys(self):
        path = self.tmp / "s.json"
        # Chai-1 has historically mixed pLDDT / plddt / PLDDT.
        path.write_text(json.dumps({"PLDDT": 80.0, "iPTM": 0.55}))
        out = parse_chai1_scores_json(path)
        self.assertAlmostEqual(out["plddt"], 80.0)
        self.assertAlmostEqual(out["iptm"], 0.55)

    def test_missing_file(self):
        out = parse_chai1_scores_json(self.tmp / "nope.json")
        self.assertIsNone(out["plddt"])

    def test_corrupt_json(self):
        path = self.tmp / "bad.json"
        path.write_text("{not json")
        out = parse_chai1_scores_json(path)
        self.assertIsNone(out["plddt"])


# ---------- PAE .npy ------------------------------------------------------


class TestPaeNpy(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_basic(self):
        arr = np.array([[0.0, 4.0, 8.0], [4.0, 0.0, 6.0], [8.0, 6.0, 0.0]])
        p = self.tmp / "pae.model_idx_0.npy"
        np.save(p, arr)
        mean = read_pae_mean_from_npy(p)
        # (0+4+8+4+0+6+8+6+0)/9 = 36/9 = 4.0
        self.assertAlmostEqual(mean, 4.0)

    def test_missing_file(self):
        self.assertIsNone(read_pae_mean_from_npy(self.tmp / "no.npy"))

    def test_corrupt_file(self):
        bad = self.tmp / "bad.npy"
        bad.write_text("not a numpy file", encoding="utf-8")
        self.assertIsNone(read_pae_mean_from_npy(bad))


# ---------- seq_map sidecar -----------------------------------------------


class TestSeqMapSidecar(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_round_trip(self):
        write_seq_map(self.tmp, {1: 4, 2: 5}, {1: 1, 2: 2})
        prot, rna = read_seq_map(self.tmp)
        self.assertEqual(prot, {1: 4, 2: 5})
        self.assertEqual(rna, {1: 1, 2: 2})

    def test_distinct_filename_from_boltz2(self):
        # Sidecar lives at chai1_seq_map.json so it doesn't collide with
        # boltz2's sidecar in a hypothetical shared work_dir.
        write_seq_map(self.tmp, {1: 1}, {1: 1})
        self.assertEqual(_SEQ_MAP_FILENAME, "chai1_seq_map.json")
        self.assertTrue((self.tmp / "chai1_seq_map.json").is_file())

    def test_missing_returns_empty(self):
        prot, rna = read_seq_map(self.tmp)
        self.assertEqual(prot, {})
        self.assertEqual(rna, {})


# ---------- prepare_input -------------------------------------------------


class TestPrepareInput(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_writes_fasta_and_sidecar(self):
        adapter = Chai1Adapter()
        paths = adapter.prepare_input(_sample_json(), self.tmp, _config(self.tmp))
        self.assertTrue(paths["fasta"].is_file())
        self.assertEqual(paths["job_name"], "1un6_B_F")
        text = paths["fasta"].read_text(encoding="utf-8")
        self.assertIn("MKTVLAGICK", text)
        self.assertIn("GCCGGCCAU", text)
        # Sidecar present.
        self.assertTrue((self.tmp / _SEQ_MAP_FILENAME).is_file())

    def test_protein_with_gaps_cleaned(self):
        adapter = Chai1Adapter()
        sj = _sample_json()
        sj["protein"]["sequence"] = "---MKTVL--AGICK"
        paths = adapter.prepare_input(sj, self.tmp, _config(self.tmp))
        text = paths["fasta"].read_text(encoding="utf-8")
        self.assertIn("MKTVLAGICK", text)
        # 5 valid AAs ahead of "AGICK" at orig 11..
        self.assertEqual(paths["protein_index_map"][1], 4)

    def test_too_short_protein_raises(self):
        adapter = Chai1Adapter()
        sj = _sample_json()
        sj["protein"]["sequence"] = "AAA"  # < MIN_PROTEIN_LEN
        with self.assertRaises(ValueError) as ctx:
            adapter.prepare_input(sj, self.tmp, _config(self.tmp))
        self.assertIn("too short after cleaning", str(ctx.exception))

    def test_missing_protein_seq_raises(self):
        adapter = Chai1Adapter()
        sj = _sample_json()
        sj["protein"]["sequence"] = ""
        with self.assertRaises(ValueError):
            adapter.prepare_input(sj, self.tmp, _config(self.tmp))


# ---------- run_tool ------------------------------------------------------


class TestRunTool(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.work = self.tmp / "work"
        self.work.mkdir()
        self.fasta = self.work / "x.fasta"
        self.fasta.write_text(">protein|chain_A\nMKTVL\n>rna|chain_B\nGGG\n")

    def test_success_invokes_runner_in_conda_env(self):
        captured = {}

        def fake(env, cmd, **kwargs):
            captured["env"] = env
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return _ok_run()

        adapter = Chai1Adapter()
        with patch(
            "step4_tool_adapters.adapters.chai1_adapter.run_in_conda_env",
            side_effect=fake,
        ):
            out = adapter.run_tool(
                {"fasta": self.fasta, "job_name": "x"},
                self.work, _config(self.work),
            )
        self.assertEqual(captured["env"], "chai1")
        self.assertIn("python", captured["cmd"])
        self.assertIn("run_chai1.py", captured["cmd"])
        self.assertEqual(captured["kwargs"]["timeout"], 900)
        self.assertEqual(out, (self.work / "chai1_output").resolve())
        self.assertTrue(out.is_dir())
        # Runner script materialised next to the FASTA.
        self.assertTrue((self.work / "run_chai1.py").is_file())

    def test_failure_raises(self):
        bad = ToolRunResult(
            command="fake", cwd=None, returncode=1,
            stdout="", stderr="boom", runtime_seconds=1.0, log_path=None,
        )
        adapter = Chai1Adapter()
        with patch(
            "step4_tool_adapters.adapters.chai1_adapter.run_in_conda_env",
            return_value=bad,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.run_tool(
                    {"fasta": self.fasta, "job_name": "x"},
                    self.work, _config(self.work),
                )
        self.assertIn("rc=1", str(ctx.exception))


# ---------- parse_output --------------------------------------------------


class TestParseOutput(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.work = self.tmp / "work"
        self.work.mkdir()
        self.out = self.work / "chai1_output"
        self.out.mkdir()
        # Identity seq_map; written next to the FASTA = inside work_dir.
        write_seq_map(
            self.work,
            protein_mapping={1: 1, 2: 2}, rna_mapping={1: 1, 2: 2},
        )
        # Synthesise the structure file Chai-1 would write.
        _make_synthetic_complex_cif(self.out / "pred.model_idx_0.cif")
        # Companion scores JSON.
        (self.out / "scores.model_idx_0.json").write_text(json.dumps({
            "plddt": 78.0, "iptm": 0.55, "ptm": 0.71, "pae": 5.0,
        }))

    def test_full_parse(self):
        adapter = Chai1Adapter()
        pred = adapter.parse_output(
            self.out, _sample_json(), _config(self.work),
        )
        self.assertTrue(pred.success)
        self.assertEqual(pred.tool_id, "chai1")
        self.assertEqual(pred.category, "A")
        # Protein residue 1 contacts RNA nt 1 (within 4.5 Å).
        self.assertIn(1, pred.binding_protein_residues or [])
        self.assertIn(1, pred.binding_rna_nucleotides or [])
        # pLDDT mean comes from the scores JSON.
        self.assertAlmostEqual(pred.plddt_mean, 78.0)
        self.assertAlmostEqual(pred.iptm_score, 0.55)
        self.assertAlmostEqual(pred.pae_mean, 5.0)
        # Predicted structure path points at the CIF.
        self.assertTrue(pred.predicted_structure_path.endswith(
            "pred.model_idx_0.cif"))

    def test_missing_cif(self):
        bad_dir = self.tmp / "empty"
        bad_dir.mkdir()
        adapter = Chai1Adapter()
        pred = adapter.parse_output(
            bad_dir, _sample_json(), _config(bad_dir),
        )
        self.assertFalse(pred.success)
        self.assertIn("no Chai-1 prediction CIF", pred.error_message)


if __name__ == "__main__":
    unittest.main()
