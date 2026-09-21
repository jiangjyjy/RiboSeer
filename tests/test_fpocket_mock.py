"""Mock tests for Fpocket adapter (no real fpocket invocation).

Coverage:
  - parse_info_txt: pocket header detection, druggability score parsing,
    multiple pockets, malformed lines tolerated
  - parse_pocket_atm_pdb: residue index extraction, chain filter,
    insertion-code-free indices
  - run_tool: shells out to fpocket via run_command (or run_in_conda_env
    when conda_env is set), copies the emit dir into work_dir, fails
    cleanly on missing output
  - parse_output: end-to-end, includes druggability-thresholded
    binding_protein_residues + pocket list
  - prepare_input: PDB extraction reuses p2rank_adapter helpers
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step4_tool_adapters.adapters.fpocket_adapter import (  # noqa: E402
    FpocketAdapter, parse_info_txt, parse_pocket_atm_pdb,
)
from step4_tool_adapters.tool_runner import ToolRunResult  # noqa: E402


# ---------- fixtures -------------------------------------------------------


_INFO_TXT_TWO_POCKETS = """\
Pocket 1 :
\tScore :                          0.541
\tDruggability Score :             0.892
\tNumber of Alpha Spheres :        45
\tTotal SASA :                     180.45
\tApolar SASA :                    100.00

Pocket 2 :
\tScore :                          0.310
\tDruggability Score :             0.420
\tNumber of Alpha Spheres :        20
\tTotal SASA :                     90.10
"""


def _pocket_atm_pdb(chain: str, residues: list[int]) -> str:
    """Synthesise a minimal pocket atm PDB with one ATOM line per residue."""
    lines = []
    for i, r in enumerate(residues, start=1):
        # ATOM record format — chain in column 22, resnum in 23-26.
        # Using the standard PDB record layout (right-justified resnum).
        lines.append(
            f"ATOM  {i:>5}  CA  ALA {chain}{r:>4}    "
            f"  10.000  10.000  10.000  1.00 20.00           C"
        )
    return "\n".join(lines) + "\n"


def _config(work_dir: Path, *, conda_env: str = "") -> dict:
    return {
        "tools": {
            "fpocket": {
                "timeout": 60,
                "residue_score_threshold": 0.5,
                "top_pockets": 5,
                **({"conda_env": conda_env} if conda_env else {}),
            },
        },
        "log_dir": str(work_dir / "logs"),
        "structure_source": {"raw_dir": str(work_dir / "raw")},
    }


def _ok_run() -> ToolRunResult:
    return ToolRunResult(
        command="fake", cwd=None, returncode=0,
        stdout="", stderr="", runtime_seconds=0.5, log_path=None,
    )


def _sample_json(chain_id: str = "B") -> dict:
    return {
        "sample_id": "1un6_B_F",
        "source_pdb": "1un6",
        "protein": {"chain_id": chain_id, "sequence": "M" * 50, "length": 50},
        "rna": {"chain_id": "F", "sequence": "GCG", "length": 3},
    }


# ---------- info.txt parsing ----------------------------------------------


class TestParseInfoTxt(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_two_pockets(self):
        path = self.tmp / "x_info.txt"
        path.write_text(_INFO_TXT_TWO_POCKETS, encoding="utf-8")
        out = parse_info_txt(path)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["rank"], 1)
        self.assertAlmostEqual(out[0]["druggability_score"], 0.892)
        self.assertAlmostEqual(out[0]["score"], 0.541)
        self.assertEqual(out[1]["rank"], 2)
        self.assertAlmostEqual(out[1]["druggability_score"], 0.420)

    def test_missing_file_returns_empty(self):
        self.assertEqual(parse_info_txt(self.tmp / "nope.txt"), [])

    def test_empty_file_returns_empty(self):
        path = self.tmp / "empty.txt"
        path.write_text("")
        self.assertEqual(parse_info_txt(path), [])

    def test_malformed_lines_skipped(self):
        path = self.tmp / "x_info.txt"
        path.write_text(
            "garbage line\n"
            "Pocket 1 :\n"
            "    Druggability Score :     0.7\n"
            "    not a key value pair garbage\n"
            "    Score :                  0.4\n",
            encoding="utf-8",
        )
        out = parse_info_txt(path)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["druggability_score"], 0.7)


# ---------- pocket atm PDB parsing ----------------------------------------


class TestParsePocketAtmPdb(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_basic_extraction(self):
        path = self.tmp / "pocket1_atm.pdb"
        path.write_text(_pocket_atm_pdb("B", [14, 15, 16, 14]), encoding="utf-8")
        # 14 appears twice — set semantics dedup it.
        self.assertEqual(parse_pocket_atm_pdb(path), [14, 15, 16])

    def test_chain_filter(self):
        path = self.tmp / "pocket1_atm.pdb"
        # Mix two chains; filter must keep only the requested one.
        text = (
            _pocket_atm_pdb("A", [10, 20])
            + _pocket_atm_pdb("B", [30, 40])
        )
        path.write_text(text, encoding="utf-8")
        self.assertEqual(parse_pocket_atm_pdb(path, chain_filter="A"),
                         [10, 20])
        self.assertEqual(parse_pocket_atm_pdb(path, chain_filter="B"),
                         [30, 40])

    def test_missing_file_returns_empty(self):
        self.assertEqual(parse_pocket_atm_pdb(self.tmp / "no.pdb"), [])

    def test_non_atom_lines_ignored(self):
        path = self.tmp / "pocket1_atm.pdb"
        path.write_text(
            "HEADER something\n"
            + _pocket_atm_pdb("B", [5])
            + "REMARK ignore me\n",
            encoding="utf-8",
        )
        self.assertEqual(parse_pocket_atm_pdb(path), [5])


# ---------- run_tool ------------------------------------------------------


class TestRunTool(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.work = self.tmp / "work"
        self.work.mkdir()
        self.pdb = self.work / "1un6_B_F_protein.pdb"
        self.pdb.write_text("HEADER fake\nEND\n")

    def _stage_emit_dir(self):
        """Create the dir fpocket would have produced next to the PDB."""
        emit = self.pdb.parent / f"{self.pdb.stem}_out"
        (emit / "pockets").mkdir(parents=True, exist_ok=True)
        (emit / f"{self.pdb.stem}_info.txt").write_text(
            _INFO_TXT_TWO_POCKETS, encoding="utf-8",
        )
        (emit / "pockets" / "pocket1_atm.pdb").write_text(
            _pocket_atm_pdb("B", [14, 15]), encoding="utf-8",
        )
        return emit

    def test_success_path(self):
        captured = {}

        def fake(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            self._stage_emit_dir()
            return _ok_run()

        adapter = FpocketAdapter()
        with patch(
            "step4_tool_adapters.adapters.fpocket_adapter.run_command",
            side_effect=fake,
        ):
            out = adapter.run_tool(
                {"protein_pdb": self.pdb}, self.work, _config(self.work),
            )
        self.assertIn("fpocket -f", captured["cmd"])
        self.assertEqual(captured["kwargs"]["timeout"], 60)
        self.assertEqual(out, (self.work / "fpocket_output").resolve())
        self.assertTrue(out.is_dir())
        # Emit dir was moved into work_dir, original gone.
        self.assertFalse((self.pdb.parent / f"{self.pdb.stem}_out").exists())

    def test_uses_conda_env_when_configured(self):
        captured = {}

        def fake(env, cmd, **kwargs):
            captured["env"] = env
            captured["cmd"] = cmd
            self._stage_emit_dir()
            return _ok_run()

        adapter = FpocketAdapter()
        with patch(
            "step4_tool_adapters.adapters.fpocket_adapter.run_in_conda_env",
            side_effect=fake,
        ):
            adapter.run_tool(
                {"protein_pdb": self.pdb}, self.work,
                _config(self.work, conda_env="pocket_env"),
            )
        self.assertEqual(captured["env"], "pocket_env")
        self.assertIn("fpocket -f", captured["cmd"])

    def test_failure_raises(self):
        bad = ToolRunResult(
            command="fake", cwd=None, returncode=1,
            stdout="", stderr="oh no", runtime_seconds=1.0, log_path=None,
        )
        adapter = FpocketAdapter()
        with patch(
            "step4_tool_adapters.adapters.fpocket_adapter.run_command",
            return_value=bad,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.run_tool(
                    {"protein_pdb": self.pdb}, self.work, _config(self.work),
                )
        self.assertIn("rc=1", str(ctx.exception))

    def test_missing_emit_dir_raises(self):
        # fpocket exits 0 but doesn't write the expected output dir.
        adapter = FpocketAdapter()
        with patch(
            "step4_tool_adapters.adapters.fpocket_adapter.run_command",
            return_value=_ok_run(),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.run_tool(
                    {"protein_pdb": self.pdb}, self.work, _config(self.work),
                )
        self.assertIn("expected output dir", str(ctx.exception))


# ---------- parse_output --------------------------------------------------


class TestParseOutput(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "fpocket_output"
        (self.out / "pockets").mkdir(parents=True, exist_ok=True)
        (self.out / "x_info.txt").write_text(
            _INFO_TXT_TWO_POCKETS, encoding="utf-8",
        )
        (self.out / "pockets" / "pocket1_atm.pdb").write_text(
            _pocket_atm_pdb("B", [14, 15, 16]), encoding="utf-8",
        )
        (self.out / "pockets" / "pocket2_atm.pdb").write_text(
            _pocket_atm_pdb("B", [50, 51]), encoding="utf-8",
        )

    def test_full_parse(self):
        adapter = FpocketAdapter()
        pred = adapter.parse_output(
            self.out, _sample_json(), _config(self.tmp),
        )
        self.assertTrue(pred.success)
        self.assertEqual(pred.tool_id, "fpocket")
        self.assertEqual(pred.category, "B")
        self.assertEqual(len(pred.pockets), 2)
        # Pocket 1 has druggability 0.892 > threshold 0.5 → its residues bind.
        # Pocket 2 has druggability 0.420 < threshold → residues skipped.
        self.assertEqual(pred.binding_protein_residues, [14, 15, 16])
        # Per-residue confidence keeps the highest pocket score per residue.
        self.assertAlmostEqual(pred.per_residue_confidence[14], 0.892)
        self.assertAlmostEqual(pred.per_residue_confidence[50], 0.420)

    def test_missing_info_txt(self):
        adapter = FpocketAdapter()
        empty = self.tmp / "empty"
        empty.mkdir()
        pred = adapter.parse_output(
            empty, _sample_json(), _config(self.tmp),
        )
        self.assertFalse(pred.success)
        self.assertIn("missing fpocket *_info.txt", pred.error_message)

    def test_multi_char_chain_filters_on_A(self):
        # Multi-char source chain id ("10") gets rewritten to "A" by
        # extract_protein_chain_pdb. Pocket atm PDB therefore has chain A.
        out2 = self.tmp / "fpocket_output_multi"
        (out2 / "pockets").mkdir(parents=True, exist_ok=True)
        (out2 / "x_info.txt").write_text(
            _INFO_TXT_TWO_POCKETS, encoding="utf-8",
        )
        (out2 / "pockets" / "pocket1_atm.pdb").write_text(
            _pocket_atm_pdb("A", [7, 8]), encoding="utf-8",
        )
        (out2 / "pockets" / "pocket2_atm.pdb").write_text(
            _pocket_atm_pdb("A", [40]), encoding="utf-8",
        )
        adapter = FpocketAdapter()
        pred = adapter.parse_output(
            out2, _sample_json(chain_id="10"), _config(self.tmp),
        )
        self.assertTrue(pred.success)
        self.assertEqual(pred.binding_protein_residues, [7, 8])


if __name__ == "__main__":
    unittest.main()
