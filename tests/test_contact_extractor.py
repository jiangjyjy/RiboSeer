"""Unit tests for step4_tool_adapters.contact_extractor.

Builds tiny synthetic PDB files in tempdirs so the tests don't depend
on any data file in the repo.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step4_tool_adapters.contact_extractor import (  # noqa: E402
    extract_contacts, ContactResult, _classify, DEFAULT_CONTACT_CUTOFF,
)


# ---------- synthetic PDB builders ----------------------------------------


def _atom_record(
    serial: int, name: str, res_name: str, chain: str, res_seq: int,
    x: float, y: float, z: float, element: str,
) -> str:
    """Format a single ATOM line per PDB v3.3."""
    return (
        f"ATOM  {serial:5d}  {name:<3s} {res_name:<3s} {chain:1s}"
        f"{res_seq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"  1.00 50.00          {element:>2s}\n"
    )


def _make_pdb_close_pair(path: Path) -> None:
    """Two-residue protein chain A + two-nucleotide RNA chain B with one
    close contact (LYS-N to A-N1 ~3 Å) and one far apart pair."""
    lines = ["HEADER    TEST                                                   01-JAN-26   TEST\n"]
    # protein chain A: residue 1 LYS (N at origin), residue 2 ALA (far away)
    lines.append(_atom_record(1, "N",   "LYS", "A", 1,  0.0,  0.0,  0.0, "N"))
    lines.append(_atom_record(2, "CA",  "LYS", "A", 1,  1.5,  0.0,  0.0, "C"))
    lines.append(_atom_record(3, "N",   "ALA", "A", 2, 50.0, 50.0, 50.0, "N"))
    lines.append(_atom_record(4, "CA",  "ALA", "A", 2, 51.5, 50.0, 50.0, "C"))
    lines.append("TER\n")
    # RNA chain B: residue 1 A (N1 ~3 Å from LYS:N), residue 2 G (far)
    lines.append(_atom_record(5, "N1",  "A",   "B", 1,  3.0,  0.0,  0.0, "N"))
    lines.append(_atom_record(6, "C2",  "A",   "B", 1,  4.0,  1.0,  0.0, "C"))
    lines.append(_atom_record(7, "N1",  "G",   "B", 2, 60.0, 60.0, 60.0, "N"))
    lines.append(_atom_record(8, "C2",  "G",   "B", 2, 61.0, 61.0, 60.0, "C"))
    lines.append("TER\n")
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


def _make_pdb_no_contacts(path: Path) -> None:
    lines = ["HEADER    TEST                                                   01-JAN-26   TEST\n"]
    lines.append(_atom_record(1, "N",  "LYS", "A", 1, 0.0, 0.0, 0.0, "N"))
    lines.append("TER\n")
    lines.append(_atom_record(2, "N1", "A",   "B", 1, 50.0, 50.0, 50.0, "N"))
    lines.append("TER\n")
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


def _make_pdb_no_rna(path: Path) -> None:
    lines = ["HEADER    TEST                                                   01-JAN-26   TEST\n"]
    lines.append(_atom_record(1, "N",  "LYS", "A", 1, 0.0, 0.0, 0.0, "N"))
    lines.append(_atom_record(2, "CA", "LYS", "A", 1, 1.5, 0.0, 0.0, "C"))
    lines.append("TER\n")
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


# ---------- tests ----------------------------------------------------------


class TestClassify(unittest.TestCase):
    def test_amino_acids(self):
        for aa in ("ALA", "LYS", "ARG", "MSE"):
            self.assertEqual(_classify(aa), "protein")

    def test_rna(self):
        for nt in ("A", "C", "G", "U"):
            self.assertEqual(_classify(nt), "rna")

    def test_other(self):
        self.assertEqual(_classify("HOH"), "other")


class TestExtractContacts(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_one_close_pair(self):
        pdb = self.tmp / "close.pdb"
        _make_pdb_close_pair(pdb)
        res = extract_contacts(pdb)
        self.assertIsInstance(res, ContactResult)
        self.assertEqual(res.binding_protein_residues, [1])
        self.assertEqual(res.binding_rna_nucleotides, [1])
        self.assertEqual(len(res.contact_pairs), 1)
        cp = res.contact_pairs[0]
        self.assertEqual(cp.protein_residue, 1)
        self.assertEqual(cp.rna_nucleotide, 1)
        self.assertLess(cp.distance, DEFAULT_CONTACT_CUTOFF)

    def test_no_contacts(self):
        pdb = self.tmp / "far.pdb"
        _make_pdb_no_contacts(pdb)
        res = extract_contacts(pdb)
        self.assertEqual(res.binding_protein_residues, [])
        self.assertEqual(res.binding_rna_nucleotides, [])
        self.assertEqual(len(res.contact_pairs), 0)

    def test_no_rna_chain(self):
        pdb = self.tmp / "no_rna.pdb"
        _make_pdb_no_rna(pdb)
        res = extract_contacts(pdb)
        self.assertEqual(res.binding_protein_residues, [])
        self.assertIsNotNone(res.note)

    def test_missing_file(self):
        res = extract_contacts(self.tmp / "nope.pdb")
        self.assertEqual(res.binding_protein_residues, [])
        self.assertIn("file not found", res.note)

    def test_custom_cutoff_zero_excludes_all(self):
        pdb = self.tmp / "close.pdb"
        _make_pdb_close_pair(pdb)
        res = extract_contacts(pdb, cutoff=0.5)
        self.assertEqual(res.binding_protein_residues, [])

    def test_chain_id_filter(self):
        pdb = self.tmp / "close.pdb"
        _make_pdb_close_pair(pdb)
        # explicit chain IDs
        res = extract_contacts(
            pdb, protein_chain_id="A", rna_chain_id="B",
        )
        self.assertEqual(res.protein_chain_id, "A")
        self.assertEqual(res.rna_chain_id, "B")
        self.assertEqual(res.binding_protein_residues, [1])

    def test_per_residue_min_distance(self):
        pdb = self.tmp / "close.pdb"
        _make_pdb_close_pair(pdb)
        res = extract_contacts(pdb)
        d = res.per_residue_protein_min_distance
        self.assertIn(1, d)
        self.assertLess(d[1], DEFAULT_CONTACT_CUTOFF)


if __name__ == "__main__":
    unittest.main(verbosity=2)
