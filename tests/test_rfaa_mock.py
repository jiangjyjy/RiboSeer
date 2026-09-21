"""Mock tests for the RoseTTAFold-All-Atom (RFAA) adapter.

No real RFAA invocation — everything is exercised against synthesised
PDB / aux fixtures and a stubbed ``run_in_conda_env``. Coverage:

  - job-name sanitisation + FASTA writer
  - seq_map sidecar round-trip (distinct filename from boltz2 / chai1)
  - read_rfaa_aux: fake-torch happy path, mean_plddt fallback, missing
    file, and the torch-unavailable graceful degradation (the parsing
    env may not have torch — pLDDT then comes from the B-factor column)
  - prepare_input: writes both FASTAs + sidecar, cleans gaps, length /
    missing-sequence guards
  - run_tool: wraps `python -m rf2aa.run_inference ...` inside the RFAA
    conda env with cwd=install_dir, the single-sequence Hydra overrides,
    and surfaces failures as RuntimeError
  - parse_output: contacts + B-factor pLDDT + distance score; aux-driven
    plddt_mean / pae_mean when the aux file is present; missing-PDB
    failure record
  - end-to-end predict() with a stub that drops a synthetic complex PDB
"""
from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import gemmi  # noqa: E402  — environment.yml dependency
import numpy as np  # noqa: E402

from step4_tool_adapters.adapters.rfaa_adapter import (  # noqa: E402
    RFAAAdapter, _SEQ_MAP_FILENAME, _sanitize_job_name,
    read_rfaa_aux, read_seq_map, write_fasta, write_seq_map,
)
from step4_tool_adapters.tool_runner import ToolRunResult  # noqa: E402


# ---------- fixtures -------------------------------------------------------


def _add_atom(res: gemmi.Residue, name: str, x: float, y: float, z: float,
              element: str, b_iso: float = 80.0) -> None:
    a = gemmi.Atom()
    a.name = name
    a.pos = gemmi.Position(x, y, z)
    a.element = gemmi.Element(element)
    a.b_iso = b_iso
    a.occ = 1.0
    res.add_atom(a)


def _make_synthetic_complex_pdb(path: Path) -> None:
    """Tiny RNA-protein complex PDB (RFAA's output shape).

    Chain A — 3 protein residues 1..3 (B-factors 85, 70, 60),
              res1 LYS:N at the origin.
    Chain B — 2 RNA nucleotides 1..2 (B-factors 55, 40),
              res1 A:N1 at (3,0,0) — ~3 Å from LYS:N.
    Expected contact: protein 1 ↔ RNA 1.
    """
    s = gemmi.Structure()
    s.cell = gemmi.UnitCell()
    m = gemmi.Model("1")

    chA = gemmi.Chain("A")
    r1 = gemmi.Residue(); r1.name = "LYS"
    r1.seqid = gemmi.SeqId(1, " "); r1.label_seq = 1
    _add_atom(r1, "N",  0.0, 0.0, 0.0, "N", b_iso=85.0)
    _add_atom(r1, "CA", 1.5, 0.0, 0.0, "C", b_iso=85.0)
    _add_atom(r1, "C",  2.5, 1.0, 0.0, "C", b_iso=85.0)
    chA.add_residue(r1)

    r2 = gemmi.Residue(); r2.name = "ALA"
    r2.seqid = gemmi.SeqId(2, " "); r2.label_seq = 2
    _add_atom(r2, "N",  50.0, 50.0, 50.0, "N", b_iso=70.0)
    _add_atom(r2, "CA", 51.5, 50.0, 50.0, "C", b_iso=70.0)
    chA.add_residue(r2)

    r3 = gemmi.Residue(); r3.name = "SER"
    r3.seqid = gemmi.SeqId(3, " "); r3.label_seq = 3
    _add_atom(r3, "N",  52.0, 50.0, 50.0, "N", b_iso=60.0)
    _add_atom(r3, "CA", 53.5, 50.0, 50.0, "C", b_iso=60.0)
    chA.add_residue(r3)
    m.add_chain(chA)

    chB = gemmi.Chain("B")
    rB1 = gemmi.Residue(); rB1.name = "A"
    rB1.seqid = gemmi.SeqId(1, " "); rB1.label_seq = 1
    _add_atom(rB1, "N1", 3.0, 0.0, 0.0, "N", b_iso=55.0)
    _add_atom(rB1, "C2", 4.0, 1.0, 0.0, "C", b_iso=55.0)
    chB.add_residue(rB1)

    rB2 = gemmi.Residue(); rB2.name = "G"
    rB2.seqid = gemmi.SeqId(2, " "); rB2.label_seq = 2
    _add_atom(rB2, "N1", 60.0, 60.0, 60.0, "N", b_iso=40.0)
    _add_atom(rB2, "C2", 61.0, 61.0, 60.0, "C", b_iso=40.0)
    chB.add_residue(rB2)
    m.add_chain(chB)

    s.add_model(m)
    s.write_pdb(str(path))


def _fake_torch(payload: dict) -> types.ModuleType:
    """A stand-in ``torch`` module whose ``load`` returns ``payload``.

    Lets us exercise read_rfaa_aux's tensor-handling without a real
    torch install (the local pocket env has none)."""
    mod = types.ModuleType("torch")

    def _load(path, map_location=None, weights_only=False):  # noqa: ARG001
        return payload

    mod.load = _load  # type: ignore[attr-defined]
    return mod


def _sample_json() -> dict:
    return {
        "sample_id": "1un6_B_F",
        "source_pdb": "1un6",
        "protein": {"chain_id": "B", "sequence": "MKTVLAGICK", "length": 10},
        "rna": {"chain_id": "F", "sequence": "GCCGGCCAU", "length": 9},
    }


def _make_install_dir(tmp: Path) -> Path:
    install_dir = tmp / "RFAA"
    (install_dir / "rf2aa" / "config" / "inference").mkdir(parents=True,
                                                           exist_ok=True)
    (install_dir / "RFAA_paper_weights.pt").write_text("stub", encoding="utf-8")
    return install_dir


def _config(tmp: Path, *, install_dir: Optional[Path] = None) -> dict:
    if install_dir is None:
        install_dir = _make_install_dir(tmp)
    return {
        "tools": {
            "rfaa": {
                "conda_env": "RFAA",
                "install_dir": str(install_dir),
                "model_weights": str(install_dir / "RFAA_paper_weights.pt"),
                "timeout": 1200,
                "device": "cuda:0",
                "config_path": "rf2aa/config/inference",
                "config_name": "base",
                "compute_distance_scores": True,
                "distance_scale": 8.0,
            },
        },
        "contact_threshold": 4.5,
        "log_dir": str(tmp / "_logs"),
    }


def _ok_run() -> ToolRunResult:
    return ToolRunResult(
        command="fake", cwd=None, returncode=0,
        stdout="ok", stderr="", runtime_seconds=0.4, log_path=None,
    )


def _bad_run(stderr: str = "boom", rc: int = 2) -> ToolRunResult:
    return ToolRunResult(
        command="fake", cwd=None, returncode=rc,
        stdout="", stderr=stderr, runtime_seconds=0.1, log_path=None,
    )


# ---------- sanitise / FASTA ----------------------------------------------


class TestSanitizeJobName(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(_sanitize_job_name("1un6_B_F"), "1un6_B_F")

    def test_dash(self):
        self.assertEqual(_sanitize_job_name("2wj8_A_-a"), "2wj8_A__a")

    def test_only_invalid(self):
        self.assertEqual(_sanitize_job_name("!!!"), "sample")


class TestWriteFasta(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_basic_wrap(self):
        out = write_fasta(self.tmp / "x.fa", "tgt", "A" * 130, line_width=60)
        body = out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(body[0], ">tgt")
        self.assertEqual(body[1], "A" * 60)
        self.assertEqual(body[3], "A" * 10)

    def test_rejects_whitespace(self):
        with self.assertRaises(ValueError):
            write_fasta(self.tmp / "x.fa", "tgt", "ACDE FGH")

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            write_fasta(self.tmp / "x.fa", "tgt", "")


# ---------- seq_map sidecar -----------------------------------------------


class TestSeqMapSidecar(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_round_trip(self):
        write_seq_map(self.tmp, {1: 4, 2: 5}, {1: 1, 2: 2})
        prot, rna = read_seq_map(self.tmp)
        self.assertEqual(prot, {1: 4, 2: 5})
        self.assertEqual(rna, {1: 1, 2: 2})

    def test_distinct_filename(self):
        self.assertEqual(_SEQ_MAP_FILENAME, "rfaa_seq_map.json")
        write_seq_map(self.tmp, {1: 1}, {1: 1})
        self.assertTrue((self.tmp / "rfaa_seq_map.json").is_file())

    def test_missing_returns_empty(self):
        self.assertEqual(read_seq_map(self.tmp), ({}, {}))


# ---------- aux.pt reader -------------------------------------------------


class TestReadRfaaAux(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.aux = self.tmp / "x_aux.pt"
        self.aux.write_text("stub", encoding="utf-8")  # only is_file matters

    def test_happy_path_with_fake_torch(self):
        payload = {
            "plddts": np.array([80.0, 75.0, 70.0]),
            "mean_plddt": 82.5,        # deliberately != mean(plddts)
            "pae": np.array([[0.0, 5.0], [5.0, 0.0]]),
            "pae_inter": 3.5,
        }
        with patch.dict(sys.modules, {"torch": _fake_torch(payload)}):
            out = read_rfaa_aux(self.aux)
        self.assertEqual(out["per_residue_plddt"], {1: 80.0, 2: 75.0, 3: 70.0})
        # plddt_mean comes from mean_plddt, NOT the per-residue average.
        self.assertAlmostEqual(out["plddt_mean"], 82.5)
        self.assertAlmostEqual(out["pae_mean"], 2.5)   # (0+5+5+0)/4
        self.assertAlmostEqual(out["pae_inter"], 3.5)

    def test_mean_plddt_fallback_to_per_residue(self):
        payload = {"plddts": np.array([80.0, 70.0])}  # no mean_plddt
        with patch.dict(sys.modules, {"torch": _fake_torch(payload)}):
            out = read_rfaa_aux(self.aux)
        self.assertAlmostEqual(out["plddt_mean"], 75.0)
        self.assertIsNone(out["pae_mean"])

    def test_missing_file_all_none(self):
        out = read_rfaa_aux(self.tmp / "nope_aux.pt")
        self.assertEqual(
            out,
            {"per_residue_plddt": None, "plddt_mean": None,
             "pae_mean": None, "pae_inter": None},
        )

    def test_torch_unavailable_degrades(self):
        # Simulate `import torch` failing (sys.modules[name] = None makes
        # the import raise ImportError) — file exists but we can't read it.
        with patch.dict(sys.modules, {"torch": None}):
            out = read_rfaa_aux(self.aux)
        self.assertIsNone(out["plddt_mean"])
        self.assertIsNone(out["pae_mean"])


# ---------- prepare_input -------------------------------------------------


class TestPrepareInput(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_writes_both_fastas_and_sidecar(self):
        adapter = RFAAAdapter()
        paths = adapter.prepare_input(_sample_json(), self.tmp, _config(self.tmp))
        self.assertTrue(paths["protein_fa"].is_file())
        self.assertTrue(paths["rna_fa"].is_file())
        self.assertEqual(paths["job_name"], "1un6_B_F")
        self.assertIn("MKTVLAGICK",
                      paths["protein_fa"].read_text(encoding="utf-8"))
        self.assertIn("GCCGGCCAU",
                      paths["rna_fa"].read_text(encoding="utf-8"))
        self.assertTrue((self.tmp / _SEQ_MAP_FILENAME).is_file())

    def test_long_sequences_written_single_line(self):
        # RFAA's parse_multichain_fasta reads each line after a header as a
        # separate sequence record, so the FASTAs must NOT wrap — even for
        # sequences well over the 60-char wrap width.
        adapter = RFAAAdapter()
        sj = _sample_json()
        long_protein = "MKTVLAGICK" * 9   # 90 aa
        long_rna = "GCCGGCCAU" * 9         # 81 nt
        sj["protein"]["sequence"] = long_protein
        sj["rna"]["sequence"] = long_rna
        paths = adapter.prepare_input(sj, self.tmp, _config(self.tmp))

        for fa, seq in ((paths["protein_fa"], long_protein),
                        (paths["rna_fa"], long_rna)):
            lines = fa.read_text(encoding="utf-8").splitlines()
            # Exactly one header + one sequence line, sequence un-wrapped.
            self.assertEqual(len(lines), 2, msg=f"{fa.name}: {lines}")
            self.assertTrue(lines[0].startswith(">"))
            self.assertEqual(lines[1], seq)

    def test_protein_gaps_cleaned_and_mapped(self):
        adapter = RFAAAdapter()
        sj = _sample_json()
        sj["protein"]["sequence"] = "---MKTVL--AGICK"
        paths = adapter.prepare_input(sj, self.tmp, _config(self.tmp))
        self.assertIn("MKTVLAGICK",
                      paths["protein_fa"].read_text(encoding="utf-8"))
        # First clean residue maps back to original index 4.
        self.assertEqual(paths["protein_index_map"][1], 4)

    def test_too_short_protein_raises(self):
        adapter = RFAAAdapter()
        sj = _sample_json()
        sj["protein"]["sequence"] = "AAA"
        with self.assertRaises(ValueError) as ctx:
            adapter.prepare_input(sj, self.tmp, _config(self.tmp))
        self.assertIn("too short after cleaning", str(ctx.exception))

    def test_missing_rna_seq_raises(self):
        adapter = RFAAAdapter()
        sj = _sample_json()
        sj["rna"]["sequence"] = ""
        with self.assertRaises(ValueError):
            adapter.prepare_input(sj, self.tmp, _config(self.tmp))


# ---------- run_tool ------------------------------------------------------


class TestRunTool(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.work = self.tmp / "work"
        self.work.mkdir()
        self.protein_fa = self.work / "p.fa"
        self.protein_fa.write_text(">p\nMKTV\n", encoding="utf-8")
        self.rna_fa = self.work / "r.fa"
        self.rna_fa.write_text(">r\nGCCG\n", encoding="utf-8")

    def _inputs(self) -> dict:
        return {
            "protein_fa": self.protein_fa,
            "rna_fa": self.rna_fa,
            "job_name": "1un6_B_F",
        }

    def test_success_returns_output_dir(self):
        adapter = RFAAAdapter()
        with patch(
            "step4_tool_adapters.adapters.rfaa_adapter.run_in_conda_env",
            return_value=_ok_run(),
        ):
            out = adapter.run_tool(self._inputs(), self.work, _config(self.tmp))
        self.assertEqual(out, (self.work / "rfaa_output").resolve())
        self.assertTrue(out.is_dir())

    def test_command_format(self):
        captured = {}

        def fake(env, cmd, **kw):
            captured.update(env=env, cmd=cmd, cwd=kw.get("cwd"),
                            timeout=kw.get("timeout"),
                            extra_env=kw.get("extra_env"))
            return _ok_run()

        adapter = RFAAAdapter()
        cfg = _config(self.tmp)
        install_dir = cfg["tools"]["rfaa"]["install_dir"]
        with patch(
            "step4_tool_adapters.adapters.rfaa_adapter.run_in_conda_env",
            side_effect=fake,
        ):
            adapter.run_tool(self._inputs(), self.work, cfg)

        cmd = captured["cmd"].replace("\\", "/")
        self.assertEqual(captured["env"], "RFAA")
        self.assertIn("python -m rf2aa.run_inference", cmd)
        self.assertIn("--config-name=base", cmd)
        self.assertIn("job_name=1un6_B_F", cmd)
        self.assertIn("output_path=", cmd)
        self.assertIn("checkpoint_path=", cmd)
        self.assertIn("RFAA_paper_weights.pt", cmd)
        self.assertIn(f"+protein_inputs.A.fasta_file="
                      f"{str(self.protein_fa.resolve()).replace(chr(92), '/')}",
                      cmd)
        self.assertIn(f"+na_inputs.B.fasta="
                      f"{str(self.rna_fa.resolve()).replace(chr(92), '/')}",
                      cmd)
        self.assertIn("+na_inputs.B.input_type=rna", cmd)
        self.assertIn('database_params.command=""', cmd)
        self.assertIn('database_params.sequencedb=""', cmd)
        self.assertIn('database_params.hhdb=""', cmd)
        self.assertIn("loader_params.n_templ=1", cmd)
        # Runs from the install dir; pins GPU 0 via CUDA_VISIBLE_DEVICES.
        self.assertEqual(captured["cwd"], str(Path(install_dir)))
        self.assertEqual(captured["timeout"], 1200)
        self.assertEqual(captured["extra_env"], {"CUDA_VISIBLE_DEVICES": "0"})

    def test_missing_install_dir(self):
        adapter = RFAAAdapter()
        cfg = _config(self.tmp)
        cfg["tools"]["rfaa"]["install_dir"] = ""
        with self.assertRaises(ValueError):
            adapter.run_tool(self._inputs(), self.work, cfg)

    def test_missing_protein_fasta(self):
        adapter = RFAAAdapter()
        self.protein_fa.unlink()
        with patch(
            "step4_tool_adapters.adapters.rfaa_adapter.run_in_conda_env",
            return_value=_ok_run(),
        ):
            with self.assertRaises(FileNotFoundError):
                adapter.run_tool(self._inputs(), self.work, _config(self.tmp))

    def test_failure_propagated(self):
        adapter = RFAAAdapter()
        with patch(
            "step4_tool_adapters.adapters.rfaa_adapter.run_in_conda_env",
            return_value=_bad_run("CUDA OOM"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.run_tool(self._inputs(), self.work, _config(self.tmp))
        msg = str(ctx.exception)
        self.assertIn("rc=2", msg)
        self.assertIn("CUDA OOM", msg)


# ---------- parse_output --------------------------------------------------


class TestParseOutput(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.work = self.tmp / "work"
        self.work.mkdir()
        self.out = self.work / "rfaa_output"
        self.out.mkdir()
        write_seq_map(self.work, {1: 1, 2: 2, 3: 3}, {1: 1, 2: 2})
        _make_synthetic_complex_pdb(self.out / "1un6_B_F.pdb")

    def test_full_parse_without_aux(self):
        adapter = RFAAAdapter()
        pred = adapter.parse_output(self.out, _sample_json(), _config(self.tmp))
        self.assertTrue(pred.success, msg=pred.error_message)
        self.assertEqual(pred.tool_id, "rfaa")
        self.assertEqual(pred.category, "A")
        self.assertEqual(pred.binding_protein_residues, [1])
        self.assertEqual(pred.binding_rna_nucleotides, [1])
        # No aux → plddt_mean = mean of B-factors (85, 70, 60).
        self.assertAlmostEqual(pred.plddt_mean, (85 + 70 + 60) / 3, places=2)
        self.assertIsNone(pred.pae_mean)
        self.assertIsNone(pred.iptm_score)
        self.assertEqual(set(pred.per_residue_confidence.keys()), {1, 2, 3})
        # Distance score populated and in [0, 1] for every protein residue.
        self.assertTrue(pred.per_residue_pae_score)
        for v in pred.per_residue_pae_score.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
        self.assertTrue(pred.predicted_structure_path.endswith("1un6_B_F.pdb"))
        self.assertIsNone(pred.pockets)

    def test_parse_uses_aux_metrics(self):
        # Drop an aux file and feed a fake torch so plddt_mean / pae_mean
        # come from the aux payload rather than the B-factor fallback.
        (self.out / "1un6_B_F_aux.pt").write_text("stub", encoding="utf-8")
        payload = {
            "plddts": np.array([90.0, 88.0, 86.0]),
            "mean_plddt": 88.0,
            "pae": np.array([[0.0, 4.0], [4.0, 0.0]]),
            "pae_inter": 2.0,
        }
        adapter = RFAAAdapter()
        with patch.dict(sys.modules, {"torch": _fake_torch(payload)}):
            pred = adapter.parse_output(
                self.out, _sample_json(), _config(self.tmp))
        self.assertTrue(pred.success, msg=pred.error_message)
        self.assertAlmostEqual(pred.plddt_mean, 88.0)
        self.assertAlmostEqual(pred.pae_mean, 2.0)  # (0+4+4+0)/4

    def test_missing_pdb_failure(self):
        bad_dir = self.tmp / "empty"
        bad_dir.mkdir()
        adapter = RFAAAdapter()
        pred = adapter.parse_output(bad_dir, _sample_json(), _config(self.tmp))
        self.assertFalse(pred.success)
        self.assertIn("no predicted .pdb", pred.error_message)


# ---------- end-to-end predict() ------------------------------------------


class TestPredictPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_full_predict_with_stub(self):
        captured = {}

        def fake(env, cmd, **kwargs):
            captured["env"] = env
            tokens = cmd.split()
            out_dir = job = None
            for t in tokens:
                if t.startswith("output_path="):
                    out_dir = Path(t.split("=", 1)[1])
                elif t.startswith("job_name="):
                    job = t.split("=", 1)[1]
            out_dir.mkdir(parents=True, exist_ok=True)
            _make_synthetic_complex_pdb(out_dir / f"{job}.pdb")
            return _ok_run()

        adapter = RFAAAdapter()
        with patch(
            "step4_tool_adapters.adapters.rfaa_adapter.run_in_conda_env",
            side_effect=fake,
        ):
            pred = adapter.predict(
                _sample_json(), self.tmp / "work", _config(self.tmp))

        self.assertEqual(captured["env"], "RFAA")
        self.assertTrue(pred.success, msg=pred.error_message)
        self.assertEqual(pred.binding_protein_residues, [1])
        self.assertEqual(pred.binding_rna_nucleotides, [1])
        self.assertTrue(pred.predicted_structure_path.endswith("1un6_B_F.pdb"))
        self.assertIsNotNone(pred.per_residue_pae_score)
        self.assertAlmostEqual(pred.plddt_mean, (85 + 70 + 60) / 3, places=2)
        self.assertIsNotNone(pred.runtime_seconds)

    def test_missing_sequence_yields_failure_record(self):
        adapter = RFAAAdapter()
        bad = _sample_json()
        bad["rna"]["sequence"] = ""
        pred = adapter.predict(bad, self.tmp / "work", _config(self.tmp))
        self.assertFalse(pred.success)
        self.assertIn("rna.sequence", pred.error_message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
