"""Mock tests for Boltz-2 adapter.

What this covers:
  - YAML writer: format matches Boltz-2 expectation, no version line
  - sample_id sanitisation for awkward IDs (dashes etc.)
  - confidence JSON parsing (happy / missing keys / corrupt)
  - PAE npz mean reader
  - per-residue pLDDT extraction from a synthetic mmCIF
  - parse_output end-to-end with synthesised CIF + JSON
  - run_tool wraps `boltz predict` inside `conda run -n boltz`
  - predict() failure record when prepare_input has missing data
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import gemmi  # noqa: E402  — environment.yml dependency
import numpy as np  # noqa: E402

from step4_tool_adapters.adapters.boltz2_adapter import (  # noqa: E402
    Boltz2Adapter, MIN_PROTEIN_LEN, MIN_RNA_LEN, _SEQ_MAP_FILENAME,
    _sanitize_job_name, clean_protein_for_boltz, clean_rna_for_boltz,
    parse_confidence_json, read_pae_mean_from_npz, read_protein_plddt_from_cif,
    read_seq_map, remap_indices, remap_per_residue, write_boltz_yaml,
    write_seq_map,
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
    """Build a tiny RNA-protein complex.

    Chain A — protein (3 residues, contiguous 1..3, varying B-factors):
       res1 LYS at origin (B=85),
       res2 ALA at (50,50,50)  (B=70, far),
       res3 SER at (52,50,50)  (B=60, far).
    Chain B — RNA (2 residues):
       res1 A   N1 ~3 Å from LYS:N (B=55),
       res2 G   far away (B=40).
    Result: protein residue 1 contacts RNA nucleotide 1.
    """
    s = gemmi.Structure()
    s.cell = gemmi.UnitCell()
    m = gemmi.Model("1")

    # Chain A protein
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

    # Chain B RNA
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
    s.make_mmcif_document().write_file(str(path))


def _mock_confidence_json() -> str:
    return json.dumps({
        "confidence_score": 0.78,
        "ptm": 0.74,
        "iptm": 0.62,
        "complex_plddt": 75.4,
        "complex_iplddt": 70.1,
        "complex_pde": 8.1,
        "complex_ipde": 9.2,
    })


def _mock_pae_npz_path(tmp_dir: Path) -> Path:
    arr = np.array([[0.0, 5.0, 12.0], [5.0, 0.0, 7.0], [12.0, 7.0, 0.0]])
    p = tmp_dir / "pae_x_model_0.npz"
    np.savez(p, pae=arr)
    return p


def _config(work_dir: Path) -> dict:
    return {
        "tools": {
            "boltz2": {
                "conda_env": "boltz",
                "flags": "--use_msa_server --no_kernels",
                "timeout": 1800,
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
    # Protein/RNA sequences must clear MIN_PROTEIN_LEN=10 / MIN_RNA_LEN=3
    # for ``prepare_input`` to accept them. 10-aa MKTVLAGICK is the
    # exact threshold; the synthetic CIF in ``_make_synthetic_complex_cif``
    # only adds 3 atoms so parse_output works regardless of input length.
    return {
        "sample_id": "1un6_B_F",
        "source_pdb": "1un6",
        "protein": {"chain_id": "B", "sequence": "MKTVLAGICK", "length": 10},
        "rna": {"chain_id": "F", "sequence": "GCCGGCCAU", "length": 9},
    }


# ---------- YAML writer ---------------------------------------------------


class TestWriteBoltzYaml(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_basic_format(self):
        path = self.tmp / "x.yaml"
        write_boltz_yaml(path, "MKTVL", "GCCAU")
        text = path.read_text(encoding="utf-8")
        # No version line — Boltz-2 rejects it.
        self.assertNotIn("version", text.lower())
        # Required structure
        self.assertTrue(text.startswith("sequences:\n"))
        self.assertIn("- protein:", text)
        self.assertIn("id: A", text)
        self.assertIn('sequence: "MKTVL"', text)
        self.assertIn("- rna:", text)
        self.assertIn("id: B", text)
        self.assertIn('sequence: "GCCAU"', text)

    def test_chain_ids_overridable(self):
        path = self.tmp / "x.yaml"
        write_boltz_yaml(path, "MKTVL", "GCCAU",
                         protein_chain="P", rna_chain="R")
        text = path.read_text(encoding="utf-8")
        self.assertIn("id: P", text)
        self.assertIn("id: R", text)

    def test_empty_seq_rejected(self):
        path = self.tmp / "x.yaml"
        with self.assertRaises(ValueError):
            write_boltz_yaml(path, "", "GCCAU")
        with self.assertRaises(ValueError):
            write_boltz_yaml(path, "MKTVL", "")

    def test_quotes_in_seq_rejected(self):
        # Defensive: a stray quote would close the YAML string and let
        # arbitrary content into the file.
        path = self.tmp / "x.yaml"
        with self.assertRaises(ValueError):
            write_boltz_yaml(path, 'MKT"VL', "GCCAU")


class TestSanitizeJobName(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(_sanitize_job_name("1un6_B_F"), "1un6_B_F")

    def test_dash(self):
        self.assertEqual(_sanitize_job_name("2wj8_A_-a"), "2wj8_A__a")

    def test_strip_leading_trailing_underscore(self):
        self.assertEqual(_sanitize_job_name("--abc--"), "abc")

    def test_only_invalid_chars(self):
        self.assertEqual(_sanitize_job_name("---"), "sample")


# ---------- confidence JSON ------------------------------------------------


class TestParseConfidenceJson(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_full(self):
        path = self.tmp / "conf.json"
        path.write_text(_mock_confidence_json())
        out = parse_confidence_json(path)
        self.assertAlmostEqual(out["plddt"], 75.4)
        self.assertAlmostEqual(out["iptm"], 0.62)
        self.assertAlmostEqual(out["pde"], 8.1)

    def test_missing_file(self):
        out = parse_confidence_json(self.tmp / "nope.json")
        self.assertIsNone(out["plddt"])
        self.assertIsNone(out["iptm"])

    def test_corrupt_json(self):
        path = self.tmp / "bad.json"
        path.write_text("{not json")
        out = parse_confidence_json(path)
        self.assertIsNone(out["plddt"])

    def test_partial_keys(self):
        path = self.tmp / "p.json"
        path.write_text(json.dumps({"iptm": 0.5}))
        out = parse_confidence_json(path)
        self.assertAlmostEqual(out["iptm"], 0.5)
        self.assertIsNone(out["plddt"])

    def test_top_level_array_returns_empty(self):
        path = self.tmp / "arr.json"
        path.write_text(json.dumps([1, 2, 3]))
        out = parse_confidence_json(path)
        self.assertIsNone(out["iptm"])


# ---------- PAE npz --------------------------------------------------------


class TestPaeNpz(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_basic(self):
        path = _mock_pae_npz_path(self.tmp)
        mean = read_pae_mean_from_npz(path)
        # arr mean = (0+5+12+5+0+7+12+7+0)/9 = 48/9 ≈ 5.333
        self.assertAlmostEqual(mean, 48 / 9, places=5)

    def test_alt_key(self):
        arr = np.array([1.0, 2.0, 3.0])
        path = self.tmp / "p.npz"
        np.savez(path, predicted_aligned_error=arr)
        self.assertAlmostEqual(read_pae_mean_from_npz(path), 2.0)

    def test_missing_file(self):
        self.assertIsNone(read_pae_mean_from_npz(self.tmp / "no.npz"))


# ---------- per-residue pLDDT from CIF ------------------------------------


class TestReadProteinPlddt(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cif = self.tmp / "model_0.cif"
        _make_synthetic_complex_cif(self.cif)

    def test_chain_A_plddt(self):
        out = read_protein_plddt_from_cif(self.cif, chain_id="A")
        self.assertEqual(set(out.keys()), {1, 2, 3})
        self.assertAlmostEqual(out[1], 85.0)
        self.assertAlmostEqual(out[2], 70.0)
        self.assertAlmostEqual(out[3], 60.0)

    def test_other_chain_empty(self):
        # chain Z doesn't exist
        out = read_protein_plddt_from_cif(self.cif, chain_id="Z")
        self.assertEqual(out, {})

    def test_missing_file(self):
        out = read_protein_plddt_from_cif(self.tmp / "no.cif", chain_id="A")
        self.assertEqual(out, {})


# ---------- adapter end-to-end --------------------------------------------


class TestRunTool(unittest.TestCase):
    """run_tool wraps the boltz CLI in a conda env."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.work = self.tmp / "work"
        self.work.mkdir()
        self.yaml = self.work / "x.yaml"
        self.yaml.write_text("sequences:\n  - protein:\n      id: A\n      sequence: \"AAA\"\n  - rna:\n      id: B\n      sequence: \"GGG\"\n")

    def test_success(self):
        captured = {}

        def fake(env, cmd, **kwargs):
            captured["env"] = env
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return _ok_run()

        adapter = Boltz2Adapter()
        with patch(
            "step4_tool_adapters.adapters.boltz2_adapter.run_in_conda_env",
            side_effect=fake,
        ):
            out = adapter.run_tool(
                {"yaml": self.yaml, "job_name": "x"},
                self.work, _config(self.work),
            )
        self.assertEqual(captured["env"], "boltz")
        self.assertIn("boltz predict", captured["cmd"])
        self.assertIn("--use_msa_server", captured["cmd"])
        self.assertIn("--no_kernels", captured["cmd"])
        self.assertIn("--out_dir", captured["cmd"])
        self.assertEqual(captured["kwargs"]["timeout"], 1800)
        self.assertEqual(out, (self.work / "boltz2_output").resolve())
        self.assertTrue(out.is_dir())

    def test_failure_raises(self):
        bad = ToolRunResult(
            command="fake", cwd=None, returncode=2,
            stdout="", stderr="oh no", runtime_seconds=1.0, log_path=None,
        )
        adapter = Boltz2Adapter()
        with patch(
            "step4_tool_adapters.adapters.boltz2_adapter.run_in_conda_env",
            return_value=bad,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.run_tool(
                    {"yaml": self.yaml, "job_name": "x"},
                    self.work, _config(self.work),
                )
        self.assertIn("rc=2", str(ctx.exception))


class TestPrepareInput(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_writes_yaml(self):
        adapter = Boltz2Adapter()
        paths = adapter.prepare_input(
            _sample_json(), self.tmp, _config(self.tmp),
        )
        self.assertTrue(paths["yaml"].is_file())
        self.assertEqual(paths["job_name"], "1un6_B_F")
        text = paths["yaml"].read_text(encoding="utf-8")
        self.assertIn('sequence: "MKTVLAGICK"', text)
        self.assertIn('sequence: "GCCGGCCAU"', text)

    def test_missing_protein_seq(self):
        adapter = Boltz2Adapter()
        bad = _sample_json()
        bad["protein"]["sequence"] = ""
        with self.assertRaises(ValueError):
            adapter.prepare_input(bad, self.tmp, _config(self.tmp))

    def test_dashed_sample_id(self):
        adapter = Boltz2Adapter()
        sj = _sample_json()
        sj["sample_id"] = "2wj8_A_-a"
        paths = adapter.prepare_input(sj, self.tmp, _config(self.tmp))
        self.assertEqual(paths["job_name"], "2wj8_A__a")
        self.assertTrue(paths["yaml"].name.endswith("2wj8_A__a.yaml"))

    def test_writes_seq_map_sidecar(self):
        adapter = Boltz2Adapter()
        adapter.prepare_input(_sample_json(), self.tmp, _config(self.tmp))
        sidecar = self.tmp / _SEQ_MAP_FILENAME
        self.assertTrue(sidecar.is_file(), "sidecar JSON not written")
        prot, rna = read_seq_map(self.tmp)
        # MKTVLAGICK → 10 standard AAs, identity mapping.
        self.assertEqual(prot, {i: i for i in range(1, 11)})
        # GCCGGCCAU → 9 standard RNA bases, identity mapping.
        self.assertEqual(rna,  {i: i for i in range(1, 10)})

    def test_protein_with_gaps_is_cleaned(self):
        # 32-gap analogue of 2bgg: protein with leading "-" + middle "-".
        # 10 standard AAs (MKTVLAGICK) interspersed with gaps so the
        # cleaned length stays >= MIN_PROTEIN_LEN.
        adapter = Boltz2Adapter()
        sj = _sample_json()
        sj["protein"]["sequence"] = "---MKTVL--AGICK"  # 10 valid + 5 gaps
        paths = adapter.prepare_input(sj, self.tmp, _config(self.tmp))
        text = paths["yaml"].read_text(encoding="utf-8")
        # YAML must NOT contain "-" inside the sequence (Boltz CCD would crash).
        self.assertIn('sequence: "MKTVLAGICK"', text)
        # MKTVL at orig 4-8, AGICK at orig 11-15 (after the 5 gaps).
        self.assertEqual(
            paths["protein_index_map"],
            {1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 11, 7: 12, 8: 13, 9: 14, 10: 15},
        )

    def test_protein_with_only_gaps_raises(self):
        adapter = Boltz2Adapter()
        sj = _sample_json()
        sj["protein"]["sequence"] = "----"
        with self.assertRaises(ValueError) as ctx:
            adapter.prepare_input(sj, self.tmp, _config(self.tmp))
        # All-gaps → cleaned len 0 → trips the MIN_PROTEIN_LEN guard.
        self.assertIn("too short after cleaning", str(ctx.exception))

    def test_rna_with_only_gaps_raises(self):
        adapter = Boltz2Adapter()
        sj = _sample_json()
        sj["rna"]["sequence"] = "---NNN"
        with self.assertRaises(ValueError) as ctx:
            adapter.prepare_input(sj, self.tmp, _config(self.tmp))
        self.assertIn("too short after cleaning", str(ctx.exception))

    def test_rna_lowercase_and_T_handled(self):
        # Lowercase + DNA-style T should still produce a valid RNA seq.
        adapter = Boltz2Adapter()
        sj = _sample_json()
        sj["rna"]["sequence"] = "guuactgc"
        paths = adapter.prepare_input(sj, self.tmp, _config(self.tmp))
        text = paths["yaml"].read_text(encoding="utf-8")
        self.assertIn('sequence: "GUUACUGC"', text)
        # Identity mapping: 8 chars in, 8 chars out (T→U, no drops).
        self.assertEqual(paths["rna_index_map"], {i: i for i in range(1, 9)})

    def test_protein_too_short_after_cleaning_raises(self):
        # 9 standard AAs + a gap → cleaned 9 aa < min 10 → reject.
        adapter = Boltz2Adapter()
        sj = _sample_json()
        sj["protein"]["sequence"] = "MKTVLAGIK-"  # 9 standard AAs
        with self.assertRaises(ValueError) as ctx:
            adapter.prepare_input(sj, self.tmp, _config(self.tmp))
        msg = str(ctx.exception)
        self.assertIn("too short after cleaning", msg)
        self.assertIn(f"min {MIN_PROTEIN_LEN}", msg)
        self.assertIn("9 aa", msg)

    def test_rna_too_short_after_cleaning_raises(self):
        # 2 standard nucleotides + 3 N's → cleaned 2 nt < min 3 → reject.
        adapter = Boltz2Adapter()
        sj = _sample_json()
        sj["rna"]["sequence"] = "AUNNN"  # 2 valid bases
        with self.assertRaises(ValueError) as ctx:
            adapter.prepare_input(sj, self.tmp, _config(self.tmp))
        msg = str(ctx.exception)
        self.assertIn("too short after cleaning", msg)
        self.assertIn(f"min {MIN_RNA_LEN}", msg)

    def test_protein_at_threshold_is_accepted(self):
        # Exactly MIN_PROTEIN_LEN standard AAs → must be accepted.
        adapter = Boltz2Adapter()
        sj = _sample_json()
        sj["protein"]["sequence"] = "MKTVLAGICK"   # exactly 10
        paths = adapter.prepare_input(sj, self.tmp, _config(self.tmp))
        self.assertTrue(paths["yaml"].is_file())

    def test_logs_drop_counts(self):
        # When cleaning removes characters, log an INFO message with
        # before/after lengths and the drop count.
        adapter = Boltz2Adapter()
        sj = _sample_json()
        sj["protein"]["sequence"] = "---MKTVLAGICK"  # 10 valid + 3 gaps
        with self.assertLogs(
            "step4_tool_adapters.adapters.boltz2_adapter", level="INFO",
        ) as captured:
            adapter.prepare_input(sj, self.tmp, _config(self.tmp))
        joined = "\n".join(captured.output)
        self.assertIn("protein cleaned", joined)
        self.assertIn("13", joined)   # original len
        self.assertIn("10", joined)   # cleaned len
        self.assertIn("dropped 3", joined)

    def test_no_log_when_nothing_dropped(self):
        # All-clean input shouldn't spam logs.
        adapter = Boltz2Adapter()
        sj = _sample_json()  # MKTVL + GCCGGCCAU, all standard
        sj["protein"]["sequence"] = "MKTVLAGICK"
        sj["rna"]["sequence"] = "GCCGGCCAU"
        # assertNoLogs is 3.10+; use assertLogs with a sentinel record
        # to detect absence portably.
        logger_name = "step4_tool_adapters.adapters.boltz2_adapter"
        try:
            with self.assertLogs(logger_name, level="INFO") as captured:
                adapter.prepare_input(sj, self.tmp, _config(self.tmp))
                # Add a synthetic record so assertLogs has at least one,
                # then check none of the captured messages match our pattern.
                logging.getLogger(logger_name).info("__sentinel__")
        except AssertionError:
            # No log records at all — that's the desired outcome.
            return
        joined = "\n".join(captured.output)
        self.assertNotIn("cleaned", joined,
                         f"unexpected cleaning log: {joined}")


class TestSequenceCleanHelpers(unittest.TestCase):
    """Pure-function tests for clean_protein_for_boltz / clean_rna_for_boltz."""

    def test_protein_drops_gaps_and_modified(self):
        clean, m = clean_protein_for_boltz("---MAGIC--K")
        self.assertEqual(clean, "MAGICK")
        self.assertEqual(m, {1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 11})

    def test_protein_drops_x_b_z_u_o(self):
        clean, m = clean_protein_for_boltz("AXBZUO")
        # Only A is standard.
        self.assertEqual(clean, "A")
        self.assertEqual(m, {1: 1})

    def test_protein_lowercase_uppercased(self):
        clean, m = clean_protein_for_boltz("amk")
        self.assertEqual(clean, "AMK")
        self.assertEqual(m, {1: 1, 2: 2, 3: 3})

    def test_protein_empty_input(self):
        self.assertEqual(clean_protein_for_boltz(""), ("", {}))

    def test_protein_all_gaps_yields_empty(self):
        self.assertEqual(clean_protein_for_boltz("---"), ("", {}))

    def test_rna_t_to_u_substitution(self):
        clean, m = clean_rna_for_boltz("AUGT")
        self.assertEqual(clean, "AUGU")  # T → U, all positions retained
        self.assertEqual(m, {1: 1, 2: 2, 3: 3, 4: 4})

    def test_rna_drops_n_and_other(self):
        clean, m = clean_rna_for_boltz("ANGNCN")
        self.assertEqual(clean, "AGC")
        self.assertEqual(m, {1: 1, 2: 3, 3: 5})


class TestRemapHelpers(unittest.TestCase):
    def test_remap_indices_basic(self):
        # Boltz reports residues 1, 3 in clean numbering. Mapping says
        # those originally lived at 4, 9 (e.g. after dropping N-term gaps).
        out = remap_indices([1, 3], {1: 4, 2: 5, 3: 9})
        self.assertEqual(out, [4, 9])

    def test_remap_indices_dedupes_and_sorts(self):
        # Pathological: two clean indices map to the same original (can't
        # happen with the clean→orig direction in practice, but the
        # helper should still de-dupe + sort).
        out = remap_indices([3, 1, 1], {1: 5, 2: 7, 3: 5})
        self.assertEqual(out, [5])

    def test_remap_indices_drops_unknown(self):
        # 99 not in mapping → silently dropped (defensive).
        out = remap_indices([1, 99], {1: 4})
        self.assertEqual(out, [4])

    def test_remap_per_residue_preserves_values(self):
        out = remap_per_residue({1: 0.7, 3: 0.9}, {1: 4, 2: 5, 3: 9})
        self.assertEqual(out, {4: 0.7, 9: 0.9})

    def test_remap_per_residue_sorts_keys(self):
        out = remap_per_residue({3: 0.9, 1: 0.7}, {1: 10, 3: 2})
        # Output is sorted by ORIGINAL index, not clean index.
        self.assertEqual(list(out.keys()), [2, 10])


class TestSeqMapSidecarIO(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_round_trip(self):
        write_seq_map(self.tmp, {1: 4, 2: 5}, {1: 1, 2: 2, 3: 3})
        prot, rna = read_seq_map(self.tmp)
        self.assertEqual(prot, {1: 4, 2: 5})
        self.assertEqual(rna, {1: 1, 2: 2, 3: 3})

    def test_read_missing_returns_empty_pair(self):
        # No sidecar in a fresh tmp dir → ({}, {}).
        prot, rna = read_seq_map(self.tmp)
        self.assertEqual(prot, {})
        self.assertEqual(rna, {})

    def test_read_corrupt_returns_empty_pair(self):
        (self.tmp / _SEQ_MAP_FILENAME).write_text("not json", encoding="utf-8")
        prot, rna = read_seq_map(self.tmp)
        self.assertEqual(prot, {})
        self.assertEqual(rna, {})


class TestParseOutput(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "boltz2_output"
        self.out.mkdir()
        # Mimic Boltz directory layout: out/predictions/<job>/...
        self.pred_dir = self.out / "predictions" / "1un6_B_F"
        self.pred_dir.mkdir(parents=True)
        _make_synthetic_complex_cif(self.pred_dir / "1un6_B_F_model_0.cif")
        (self.pred_dir / "confidence_1un6_B_F_model_0.json").write_text(
            _mock_confidence_json(), encoding="utf-8"
        )

    def test_full_parse(self):
        adapter = Boltz2Adapter()
        pred = adapter.parse_output(self.out, _sample_json(), _config(self.tmp))
        self.assertTrue(pred.success, msg=pred.error_message)
        # contacts: protein res 1 ↔ rna 1 (one near pair)
        self.assertEqual(pred.binding_protein_residues, [1])
        self.assertEqual(pred.binding_rna_nucleotides, [1])
        self.assertIsNotNone(pred.predicted_structure_path)
        self.assertTrue(pred.predicted_structure_path.endswith(
            "1un6_B_F_model_0.cif"))
        # JSON's complex_plddt 75.4 takes precedence
        self.assertAlmostEqual(pred.plddt_mean, 75.4)
        self.assertAlmostEqual(pred.iptm_score, 0.62)
        self.assertAlmostEqual(pred.pae_mean, 8.1)
        # per-residue pLDDT for chain A only
        self.assertEqual(set(pred.per_residue_confidence.keys()), {1, 2, 3})

    def test_missing_cif_failure(self):
        # Remove the CIF
        (self.pred_dir / "1un6_B_F_model_0.cif").unlink()
        adapter = Boltz2Adapter()
        pred = adapter.parse_output(self.out, _sample_json(), _config(self.tmp))
        self.assertFalse(pred.success)
        self.assertIn("model_0.cif", pred.error_message)

    def test_seq_map_remaps_residue_indices(self):
        # Simulate a sample whose original protein had 10 residues but
        # 2 leading gaps + 1 internal gap, leaving 7 standard AAs at
        # original positions [3, 4, 5, 6, 8, 9, 10]. The CIF was built
        # for the cleaned sequence, so it numbers residues 1..3 (only
        # 3 atoms in the synthetic CIF). The mapping should translate
        # those back to original positions [3, 4, 5].
        write_seq_map(
            self.tmp,
            protein_mapping={1: 3, 2: 4, 3: 5, 4: 6, 5: 8, 6: 9, 7: 10},
            rna_mapping={1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
                         6: 6, 7: 7, 8: 8, 9: 9},
        )
        adapter = Boltz2Adapter()
        pred = adapter.parse_output(self.out, _sample_json(), _config(self.tmp))
        self.assertTrue(pred.success, msg=pred.error_message)
        # Synthetic CIF puts the contacting protein residue at clean
        # idx 1 → original idx 3.
        self.assertEqual(pred.binding_protein_residues, [3])
        # per_residue_confidence keys must also be remapped (not 1/2/3).
        self.assertEqual(set(pred.per_residue_confidence.keys()), {3, 4, 5})

    def test_no_seq_map_means_no_remap_backwards_compat(self):
        # Run parse_output WITHOUT writing a sidecar — old work_dirs
        # (or callers that bypassed prepare_input) must still parse.
        adapter = Boltz2Adapter()
        pred = adapter.parse_output(self.out, _sample_json(), _config(self.tmp))
        self.assertTrue(pred.success, msg=pred.error_message)
        # No remap → original 1..3 keys preserved (matches test_full_parse).
        self.assertEqual(pred.binding_protein_residues, [1])
        self.assertEqual(set(pred.per_residue_confidence.keys()), {1, 2, 3})

    def test_no_confidence_json_uses_cif_fallback(self):
        # Delete the JSON; plddt_mean should fall back to mean of B-factors
        (self.pred_dir / "confidence_1un6_B_F_model_0.json").unlink()
        adapter = Boltz2Adapter()
        pred = adapter.parse_output(self.out, _sample_json(), _config(self.tmp))
        self.assertTrue(pred.success, msg=pred.error_message)
        # mean of (85, 70, 60) = 71.667
        self.assertAlmostEqual(pred.plddt_mean, (85 + 70 + 60) / 3, places=2)
        self.assertIsNone(pred.iptm_score)
        self.assertIsNone(pred.pae_mean)

    def test_pae_falls_back_to_npz_when_json_missing_pde(self):
        # Replace JSON with one that lacks complex_pde, and add a PAE npz
        (self.pred_dir / "confidence_1un6_B_F_model_0.json").write_text(
            json.dumps({"iptm": 0.5, "complex_plddt": 80.0}),
            encoding="utf-8",
        )
        _mock_pae_npz_path(self.pred_dir).rename(
            self.pred_dir / "pae_1un6_B_F_model_0.npz"
        )
        adapter = Boltz2Adapter()
        pred = adapter.parse_output(self.out, _sample_json(), _config(self.tmp))
        self.assertTrue(pred.success, msg=pred.error_message)
        self.assertAlmostEqual(pred.plddt_mean, 80.0)
        self.assertAlmostEqual(pred.pae_mean, 48 / 9, places=5)


class TestPredictPipeline(unittest.TestCase):
    """End-to-end predict() with stubbed run_in_conda_env that drops mock outputs."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_full_predict_with_stub(self):
        captured = {}

        def fake_run(env, cmd, **kwargs):
            # Parse --out_dir from the command, then create mock outputs.
            tokens = cmd.split()
            i = tokens.index("--out_dir")
            out_dir = Path(tokens[i + 1])
            pred_dir = out_dir / "predictions" / "1un6_B_F"
            pred_dir.mkdir(parents=True, exist_ok=True)
            _make_synthetic_complex_cif(pred_dir / "1un6_B_F_model_0.cif")
            (pred_dir / "confidence_1un6_B_F_model_0.json").write_text(
                _mock_confidence_json(), encoding="utf-8"
            )
            captured["env"] = env
            captured["cmd"] = cmd
            return _ok_run()

        adapter = Boltz2Adapter()
        with patch(
            "step4_tool_adapters.adapters.boltz2_adapter.run_in_conda_env",
            side_effect=fake_run,
        ):
            pred = adapter.predict(
                _sample_json(), self.tmp / "work", _config(self.tmp),
            )
        self.assertEqual(captured["env"], "boltz")
        self.assertTrue(pred.success, msg=pred.error_message)
        self.assertEqual(pred.binding_protein_residues, [1])
        self.assertEqual(pred.binding_rna_nucleotides, [1])
        # Windows time.monotonic() has ~15ms resolution; mock pipeline can
        # finish in <1 tick. Just confirm the field is populated and non-negative.
        self.assertIsNotNone(pred.runtime_seconds)
        self.assertGreaterEqual(pred.runtime_seconds, 0.0)

    def test_missing_sequence_yields_failure_record(self):
        adapter = Boltz2Adapter()
        bad = _sample_json()
        bad["protein"]["sequence"] = ""
        pred = adapter.predict(bad, self.tmp / "work", _config(self.tmp))
        self.assertFalse(pred.success)
        self.assertIn("protein.sequence", pred.error_message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
