"""Mock tests for P2Rank adapter (no real P2Rank invocation).

What this covers:
  - chain extraction + renumbering against real PDB files in
    rna2p_balanced/ when available; skip if not present
  - prepare_input full path with a tempdir + raw_dir override
  - parse_output against synthesised P2Rank-format CSVs
  - run_tool failure surfaces a helpful error
  - end-to-end predict() with a stubbed run_tool that just
    drops mock CSVs into the output dir
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step4_tool_adapters.adapters.p2rank_adapter import (  # noqa: E402
    P2RankAdapter, _find_raw_structure, _parse_residue_id_token,
    effective_pdb_chain_id, extract_protein_chain_pdb,
    parse_predictions_csv, parse_residues_csv,
)
from step4_tool_adapters.tool_runner import ToolRunResult  # noqa: E402


# ---------- fixtures -------------------------------------------------------


RNA2P_DIR = Path("/path/to/raw_structures")  # local raw PDB store
SAMPLE_1UN6_B_F = REPO / "data" / "processed" / "samples" / "1un6_B_F.json"


def _mock_predictions_csv() -> str:
    """P2Rank-format predictions CSV with two pockets on chain B."""
    return (
        "   name,   rank,   score, probability,  sas_points,  surf_atoms,    center_x,    center_y,    center_z, residue_ids\n"
        "pocket1,      1,   12.40,        0.89,          25,          80,        1.234,        2.345,        3.456, B_14 B_15 B_16 B_17\n"
        "pocket2,      2,    8.10,        0.55,          12,          40,        4.567,        5.678,        6.789, B_50 B_51 B_52\n"
        "pocket3,      3,    2.30,        0.12,           5,          18,        7.000,        7.000,        7.000, B_80\n"
    )


def _mock_residues_csv() -> str:
    """P2Rank-format residues CSV — 5 residues on chain B with mixed scores."""
    return (
        "chain, residue_label, residue_name,    score, probability, pocket, zscore\n"
        "    B,            14,         LYS,     0.95,        0.85,      1,    2.30\n"
        "    B,            15,         HIS,     0.88,        0.78,      1,    1.95\n"
        "    B,            16,         ARG,     0.74,        0.62,      1,    1.20\n"
        "    B,            50,         LYS,     0.55,        0.45,      2,    0.50\n"
        "    B,            80,         CYS,     0.20,        0.10,      3,   -0.80\n"
        "    B,           100,         ALA,     0.05,        0.02,     -1,   -1.50\n"
    )


def _config(tmp_raw_dir: Path) -> dict:
    return {
        "tools": {
            "p2rank": {
                "install_dir": "/fake/p2rank",
                "timeout": 60,
                "residue_score_threshold": 0.5,
                "top_pockets": 5,
            },
        },
        "structure_source": {"raw_dir": str(tmp_raw_dir)},
        "log_dir": str(tmp_raw_dir / "logs"),
    }


def _ok_run(stdout: str = "P2Rank ran fine", **_kw) -> ToolRunResult:
    return ToolRunResult(
        command="fake", cwd=None, returncode=0,
        stdout=stdout, stderr="", runtime_seconds=0.42, log_path=None,
    )


# ---------- residue token parsing -----------------------------------------


class TestResidueIdParsing(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(_parse_residue_id_token("A_14"), ("A", 14))

    def test_multi_char_chain(self):
        self.assertEqual(_parse_residue_id_token("AB_125"), ("AB", 125))

    def test_with_insertion_code(self):
        # Insertion codes are accepted but stripped; index returned is the int part.
        self.assertEqual(_parse_residue_id_token("A_125B"), ("A", 125))

    def test_negative_index(self):
        self.assertEqual(_parse_residue_id_token("A_-3"), ("A", -3))

    def test_garbage(self):
        self.assertIsNone(_parse_residue_id_token(""))
        self.assertIsNone(_parse_residue_id_token("not_an_id"))
        self.assertIsNone(_parse_residue_id_token("Aabc"))


# ---------- predictions / residues CSV parsing ----------------------------


class TestParsePredictionsCsv(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.csv = self.tmp / "x.pdb_predictions.csv"
        self.csv.write_text(_mock_predictions_csv(), encoding="utf-8")

    def test_all_pockets(self):
        pockets = parse_predictions_csv(self.csv, chain_filter="B")
        self.assertEqual(len(pockets), 3)
        self.assertEqual([p.rank for p in pockets], [1, 2, 3])
        self.assertEqual(pockets[0].residues, [14, 15, 16, 17])
        self.assertAlmostEqual(pockets[0].score, 12.40)

    def test_top_n(self):
        pockets = parse_predictions_csv(self.csv, chain_filter="B", top_n=2)
        self.assertEqual(len(pockets), 2)

    def test_chain_filter(self):
        # No residue tokens for chain "Z" → all pockets have empty residues
        pockets = parse_predictions_csv(self.csv, chain_filter="Z")
        for p in pockets:
            self.assertEqual(p.residues, [])

    def test_no_chain_filter_keeps_all(self):
        # Add a row with mixed-chain residue_ids
        mixed = self.tmp / "mixed.csv"
        mixed.write_text(
            "name,rank,score,residue_ids\n"
            "p1,1,5.0,A_10 B_20 A_15\n",
            encoding="utf-8",
        )
        pockets = parse_predictions_csv(mixed)
        self.assertEqual(pockets[0].residues, [10, 15, 20])


class TestParseResiduesCsv(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.csv = self.tmp / "x.pdb_residues.csv"
        self.csv.write_text(_mock_residues_csv(), encoding="utf-8")

    def test_chain_filter_picks_protein(self):
        scores = parse_residues_csv(self.csv, chain_filter="B")
        self.assertEqual(set(scores.keys()), {14, 15, 16, 50, 80, 100})
        self.assertAlmostEqual(scores[14], 0.95)
        self.assertAlmostEqual(scores[100], 0.05)

    def test_other_chain_empty(self):
        scores = parse_residues_csv(self.csv, chain_filter="Z")
        self.assertEqual(scores, {})

    def test_insertion_code_residue_label(self):
        custom = self.tmp / "ins.csv"
        custom.write_text(
            "chain,residue_label,score\n"
            "B,125A,0.6\n"
            "B,126,0.7\n",
            encoding="utf-8",
        )
        scores = parse_residues_csv(custom, chain_filter="B")
        self.assertEqual(set(scores.keys()), {125, 126})


# ---------- raw structure lookup ------------------------------------------


class TestFindRawStructure(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_pdb_priority(self):
        (self.tmp / "1un6.pdb").write_text("HEADER\nEND\n")
        (self.tmp / "1un6.cif").write_text("data_1un6\n")
        found = _find_raw_structure(self.tmp, "1un6")
        self.assertEqual(found.suffix.lower(), ".pdb")

    def test_cif_fallback(self):
        (self.tmp / "1un6.cif").write_text("data_1un6\n")
        found = _find_raw_structure(self.tmp, "1un6")
        self.assertEqual(found.suffix.lower(), ".cif")

    def test_gz_cif_fallback(self):
        (self.tmp / "5kpx.cif.gz").write_text("data_5kpx\n")
        found = _find_raw_structure(self.tmp, "5kpx")
        self.assertIsNotNone(found)
        self.assertTrue(found.name.endswith(".cif.gz"))

    def test_case_insensitive_directory_scan(self):
        # raw file named differently than ``source_pdb`` casing. On
        # case-sensitive FS this exercises the directory-scan fallback;
        # on Windows (case-insensitive) the verbatim lookup finds it
        # too — both paths must return a file.
        (self.tmp / "3J46.cif").write_text("data_3J46\n")
        found = _find_raw_structure(self.tmp, "3j46")
        self.assertIsNotNone(found)
        self.assertEqual(found.suffix.lower(), ".cif")

    def test_missing(self):
        self.assertIsNone(_find_raw_structure(self.tmp, "noexist"))


class TestEffectiveChainId(unittest.TestCase):
    def test_single_char_passes_through(self):
        self.assertEqual(effective_pdb_chain_id("B"), "B")
        self.assertEqual(effective_pdb_chain_id("A"), "A")

    def test_multi_char_collapses_to_A(self):
        # Multi-char chain ids (only valid in mmCIF) get rewritten to "A"
        # because the PDB format has a one-column chainID field.
        self.assertEqual(effective_pdb_chain_id("10"), "A")
        self.assertEqual(effective_pdb_chain_id("A5"), "A")
        self.assertEqual(effective_pdb_chain_id("KK"), "A")

    def test_empty_string_passes_through(self):
        # Single-char branch — an empty string is still len 0, so
        # passes through unchanged. Defensive for callers that haven't
        # validated chain_id presence yet.
        self.assertEqual(effective_pdb_chain_id(""), "")


# ---------- chain extraction (uses real rna2p_balanced PDB if present) ----


@unittest.skipUnless(
    RNA2P_DIR.is_dir() and (RNA2P_DIR / "1un6.pdb").is_file(),
    "rna2p_balanced/1un6.pdb not available locally — skip",
)
class TestExtractChainRenumber(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_chain_extracted_and_renumbered(self):
        out = self.tmp / "1un6_B_F_protein.pdb"
        extract_protein_chain_pdb(RNA2P_DIR / "1un6.pdb", "B", out)
        self.assertTrue(out.is_file())

        # The single chain in the new PDB must start at residue 1.
        atoms = [
            ln for ln in out.read_text(encoding="utf-8").splitlines()
            if ln.startswith("ATOM")
        ]
        self.assertGreater(len(atoms), 0)
        # Columns 23-26 in PDB ATOM record carry the residue sequence number.
        first_resnum = int(atoms[0][22:26].strip())
        self.assertEqual(first_resnum, 1)
        # Last residue number should equal protein length 87 (1un6 chain B).
        last_resnum = int(atoms[-1][22:26].strip())
        self.assertEqual(last_resnum, 87)
        # All ATOM lines should be on chain B.
        chains = {ln[21] for ln in atoms}
        self.assertEqual(chains, {"B"})

    def test_unknown_chain_raises(self):
        out = self.tmp / "noise.pdb"
        with self.assertRaises(ValueError):
            extract_protein_chain_pdb(RNA2P_DIR / "1un6.pdb", "Z", out)

    def test_missing_file_raises(self):
        out = self.tmp / "noise.pdb"
        with self.assertRaises(FileNotFoundError):
            extract_protein_chain_pdb(self.tmp / "no.pdb", "B", out)


# ---------- adapter-level tests -------------------------------------------


def _sample_json() -> dict:
    """Lightweight sample JSON; enough fields for the adapter."""
    return {
        "sample_id": "1un6_B_F",
        "source_pdb": "1un6",
        "protein": {"chain_id": "B"},
    }


class TestPrepareInput(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.raw = self.tmp / "raw"
        self.raw.mkdir()

    def test_no_raw_file_raises(self):
        adapter = P2RankAdapter()
        with self.assertRaises(FileNotFoundError):
            adapter.prepare_input(_sample_json(), self.tmp, _config(self.raw))

    @unittest.skipUnless(
        (RNA2P_DIR / "1un6.pdb").is_file(),
        "rna2p_balanced/1un6.pdb not available — skip",
    )
    def test_real_raw_pdb(self):
        # Copy 1un6.pdb into our fake raw_dir.
        (self.raw / "1un6.pdb").write_bytes(
            (RNA2P_DIR / "1un6.pdb").read_bytes()
        )
        adapter = P2RankAdapter()
        paths = adapter.prepare_input(_sample_json(), self.tmp, _config(self.raw))
        self.assertTrue(paths["protein_pdb"].is_file())
        self.assertEqual(paths["protein_pdb"].name, "1un6_B_F_protein.pdb")


class TestRunTool(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.work = self.tmp / "work"
        self.work.mkdir()
        self.pdb = self.work / "p.pdb"
        self.pdb.write_text("HEADER\nATOM      1  N   LYS B   1       0.000   0.000   0.000\nEND\n")

    def test_success_returns_output_dir(self):
        adapter = P2RankAdapter()
        with patch(
            "step4_tool_adapters.adapters.p2rank_adapter.run_command",
            side_effect=lambda *a, **kw: _ok_run(),
        ):
            out = adapter.run_tool(
                {"protein_pdb": self.pdb}, self.work,
                _config(self.tmp),
            )
        self.assertEqual(out, (self.work / "p2rank_output").resolve())
        self.assertTrue(out.is_dir())

    def test_missing_install_dir(self):
        adapter = P2RankAdapter()
        cfg = _config(self.tmp)
        cfg["tools"]["p2rank"]["install_dir"] = ""
        with self.assertRaises(ValueError):
            adapter.run_tool({"protein_pdb": self.pdb}, self.work, cfg)

    def test_failure_propagated(self):
        adapter = P2RankAdapter()
        bad = ToolRunResult(
            command="fake", cwd=None, returncode=2,
            stdout="", stderr="bad path",
            runtime_seconds=0.1, log_path=None,
        )
        with patch(
            "step4_tool_adapters.adapters.p2rank_adapter.run_command",
            return_value=bad,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.run_tool(
                    {"protein_pdb": self.pdb}, self.work,
                    _config(self.tmp),
                )
        self.assertIn("rc=2", str(ctx.exception))


class TestParseOutput(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "p2rank_output"
        self.out.mkdir()
        (self.out / "p.pdb_predictions.csv").write_text(
            _mock_predictions_csv(), encoding="utf-8",
        )
        (self.out / "p.pdb_residues.csv").write_text(
            _mock_residues_csv(), encoding="utf-8",
        )

    def test_full_parse(self):
        adapter = P2RankAdapter()
        pred = adapter.parse_output(self.out, _sample_json(), _config(self.tmp))
        self.assertTrue(pred.success)
        self.assertEqual(pred.tool_id, "p2rank")
        self.assertEqual(pred.category, "B")
        # threshold 0.5: 14 (0.95), 15 (0.88), 16 (0.74), 50 (0.55) qualify.
        # 80 (0.20), 100 (0.05) do not.
        self.assertEqual(pred.binding_protein_residues, [14, 15, 16, 50])
        self.assertEqual(len(pred.pockets), 3)
        self.assertEqual(pred.pockets[0].residues, [14, 15, 16, 17])
        self.assertIsNotNone(pred.per_residue_confidence)
        self.assertEqual(pred.per_residue_confidence[14], 0.95)
        self.assertEqual(pred.binding_rna_nucleotides, None)  # Cat B has no RNA
        self.assertIsNone(pred.predicted_structure_path)

    def test_missing_predictions_csv(self):
        (self.out / "p.pdb_predictions.csv").unlink()
        adapter = P2RankAdapter()
        pred = adapter.parse_output(self.out, _sample_json(), _config(self.tmp))
        self.assertFalse(pred.success)
        self.assertIn("predictions CSV", pred.error_message)

    def test_missing_residues_csv(self):
        (self.out / "p.pdb_residues.csv").unlink()
        adapter = P2RankAdapter()
        pred = adapter.parse_output(self.out, _sample_json(), _config(self.tmp))
        self.assertFalse(pred.success)
        self.assertIn("residues CSV", pred.error_message)

    def test_threshold_override(self):
        adapter = P2RankAdapter()
        cfg = _config(self.tmp)
        cfg["tools"]["p2rank"]["residue_score_threshold"] = 0.8
        pred = adapter.parse_output(self.out, _sample_json(), cfg)
        # only 14 (0.95), 15 (0.88) make it past 0.8
        self.assertEqual(pred.binding_protein_residues, [14, 15])

    def test_top_pockets_cap(self):
        adapter = P2RankAdapter()
        cfg = _config(self.tmp)
        cfg["tools"]["p2rank"]["top_pockets"] = 1
        pred = adapter.parse_output(self.out, _sample_json(), cfg)
        self.assertEqual(len(pred.pockets), 1)


class TestPredictPipeline(unittest.TestCase):
    """End-to-end predict() with stubbed run_tool."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.raw = self.tmp / "raw"
        self.raw.mkdir()

    @unittest.skipUnless(
        (RNA2P_DIR / "1un6.pdb").is_file(),
        "rna2p_balanced/1un6.pdb not available — skip",
    )
    def test_full_predict_with_real_pdb(self):
        # 1) seed raw_dir with a real PDB
        (self.raw / "1un6.pdb").write_bytes(
            (RNA2P_DIR / "1un6.pdb").read_bytes()
        )

        # 2) replace run_command with a stub that drops mock CSVs into the
        #    output dir and returns success
        def fake_run_command(cmd: str, **kwargs):
            # parse "-o <output_dir>" out of the command
            tokens = cmd.split()
            i = tokens.index("-o")
            out_dir = Path(tokens[i + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "p.pdb_predictions.csv").write_text(
                _mock_predictions_csv(), encoding="utf-8")
            (out_dir / "p.pdb_residues.csv").write_text(
                _mock_residues_csv(), encoding="utf-8")
            return _ok_run()

        with patch(
            "step4_tool_adapters.adapters.p2rank_adapter.run_command",
            side_effect=fake_run_command,
        ):
            adapter = P2RankAdapter()
            pred = adapter.predict(
                _sample_json(), self.tmp / "work", _config(self.raw),
            )

        self.assertTrue(pred.success, msg=pred.error_message)
        self.assertEqual(pred.binding_protein_residues, [14, 15, 16, 50])
        # Windows time.monotonic() has ~15ms granularity; mock can be 0.0
        self.assertIsNotNone(pred.runtime_seconds)
        self.assertGreaterEqual(pred.runtime_seconds, 0.0)

    def test_predict_with_missing_raw_returns_failure_record(self):
        adapter = P2RankAdapter()
        pred = adapter.predict(
            _sample_json(), self.tmp / "work", _config(self.raw),
        )
        self.assertFalse(pred.success)
        self.assertEqual(pred.tool_id, "p2rank")
        self.assertIn("no raw structure", pred.error_message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
