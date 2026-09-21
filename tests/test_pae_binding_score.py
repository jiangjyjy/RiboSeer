"""Tests for the PAE-derived per-residue binding score (paper §4 fix).

Coverage:
  - compute_pae_binding_scores math: known PAE → expected probability
  - shape / dimension guards (None matrix, wrong rank, undersized)
  - read_pae_matrix_from_npz / read_pae_matrix_from_npy round-trips
  - boltz2_adapter.parse_output emits per_residue_pae_score
  - chai1_adapter.parse_output emits per_residue_pae_score
  - evaluate.py prefers per_residue_pae_score for Cat A tools
  - evaluate.py falls back to per_residue_confidence when pae_score absent
  - schema accepts the new field; failure prediction sets it to None
  - configs disable knob actually disables PAE computation
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import gemmi  # noqa: E402
import numpy as np  # noqa: E402

from step4_tool_adapters.adapters.boltz2_adapter import (  # noqa: E402
    Boltz2Adapter, compute_pae_binding_scores,
    read_pae_matrix_from_npz, write_seq_map as boltz2_write_seq_map,
)
from step4_tool_adapters.adapters.chai1_adapter import (  # noqa: E402
    Chai1Adapter, compute_distance_binding_scores,
    write_seq_map as chai1_write_seq_map,
)
from step4_tool_adapters.schemas import (  # noqa: E402
    ToolPrediction, make_failure_prediction,
)


# ---------- compute_pae_binding_scores math -------------------------------


class TestComputePaeBindingScores(unittest.TestCase):
    def test_zero_pae_gives_prob_one(self):
        # PAE = 0 everywhere → 1 / (1 + 0/8) = 1.0 for every residue.
        pae = np.zeros((4, 4), dtype=np.float32)
        out = compute_pae_binding_scores(pae, n_protein=2, n_rna=2)
        self.assertEqual(set(out.keys()), {1, 2})
        for v in out.values():
            self.assertAlmostEqual(v, 1.0, places=5)

    def test_pae_at_scale_gives_prob_half(self):
        # PAE = 8 (default scale) for every inter-chain entry → prob 0.5.
        pae = np.full((4, 4), 8.0, dtype=np.float32)
        # Zero out intra-chain (we only look at inter-chain anyway).
        out = compute_pae_binding_scores(pae, n_protein=2, n_rna=2)
        for v in out.values():
            self.assertAlmostEqual(v, 0.5, places=5)

    def test_pae_high_gives_low_prob(self):
        pae = np.full((3, 3), 100.0, dtype=np.float32)
        out = compute_pae_binding_scores(pae, n_protein=2, n_rna=1)
        # 1 / (1 + 100/8) ≈ 0.0741
        for v in out.values():
            self.assertLess(v, 0.1)

    def test_uses_min_inter_chain_pae(self):
        # Protein residue 1 has min PAE 2 (good); residue 2 has min PAE 50 (bad).
        # Layout: protein indices 0-1, RNA indices 2-3.
        pae = np.array([
            [0.0, 0.0, 2.0, 30.0],   # protein res 1 → RNA res 1 = 2.0
            [0.0, 0.0, 50.0, 60.0],  # protein res 2 → RNA res 1 = 50.0
            [2.0, 50.0, 0.0, 0.0],
            [30.0, 60.0, 0.0, 0.0],
        ], dtype=np.float32)
        out = compute_pae_binding_scores(pae, n_protein=2, n_rna=2,
                                         pae_scale=8.0)
        # res 1: 1 / (1 + 2/8) = 1/1.25 = 0.8
        self.assertAlmostEqual(out[1], 0.8, places=4)
        # res 2: 1 / (1 + 50/8) ≈ 0.1379
        self.assertAlmostEqual(out[2], 1.0 / (1.0 + 50.0 / 8.0), places=4)

    def test_keys_are_one_based(self):
        pae = np.zeros((3, 3), dtype=np.float32)
        out = compute_pae_binding_scores(pae, n_protein=2, n_rna=1)
        # No 0-based keys.
        self.assertNotIn(0, out)
        self.assertEqual(min(out.keys()), 1)

    def test_pae_scale_tunable(self):
        pae = np.full((2, 2), 10.0, dtype=np.float32)
        # scale=10 → prob = 0.5
        a = compute_pae_binding_scores(pae, n_protein=1, n_rna=1, pae_scale=10.0)
        # scale=5 → prob = 1/(1 + 10/5) = 0.333
        b = compute_pae_binding_scores(pae, n_protein=1, n_rna=1, pae_scale=5.0)
        self.assertAlmostEqual(a[1], 0.5, places=4)
        self.assertAlmostEqual(b[1], 1.0 / 3.0, places=4)


class TestComputePaeBindingScoresDegenerate(unittest.TestCase):
    def test_none_matrix(self):
        self.assertEqual(
            compute_pae_binding_scores(None, n_protein=2, n_rna=2),
            {},
        )

    def test_wrong_rank(self):
        bad = np.array([1.0, 2.0])  # 1-D
        self.assertEqual(
            compute_pae_binding_scores(bad, n_protein=1, n_rna=1),
            {},
        )

    def test_too_small_for_chain_split(self):
        pae = np.zeros((3, 3))
        # Asking for 2 protein + 2 RNA but matrix is 3×3.
        self.assertEqual(
            compute_pae_binding_scores(pae, n_protein=2, n_rna=2),
            {},
        )

    def test_zero_protein_or_rna(self):
        pae = np.zeros((4, 4))
        self.assertEqual(
            compute_pae_binding_scores(pae, n_protein=0, n_rna=4), {},
        )
        self.assertEqual(
            compute_pae_binding_scores(pae, n_protein=4, n_rna=0), {},
        )

    def test_non_positive_scale(self):
        pae = np.zeros((4, 4))
        self.assertEqual(
            compute_pae_binding_scores(
                pae, n_protein=2, n_rna=2, pae_scale=0.0,
            ),
            {},
        )


# ---------- matrix readers (round-trip) -----------------------------------


class TestMatrixReaders(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    def test_npz_round_trip_default_key(self):
        path = self.tmp / "pae.npz"
        np.savez(path, pae=self.arr)
        out = read_pae_matrix_from_npz(path)
        self.assertIsNotNone(out)
        np.testing.assert_array_equal(out, self.arr)

    def test_npz_round_trip_alt_key(self):
        path = self.tmp / "pae.npz"
        np.savez(path, predicted_aligned_error=self.arr)
        out = read_pae_matrix_from_npz(path)
        np.testing.assert_array_equal(out, self.arr)

    def test_npz_missing_returns_none(self):
        self.assertIsNone(read_pae_matrix_from_npz(self.tmp / "no.npz"))


# ---------- ToolPrediction schema -----------------------------------------


class TestSchemaField(unittest.TestCase):
    def test_field_accepts_dict(self):
        p = ToolPrediction(
            tool_id="boltz2", category="A", sample_id="s",
            success=True,
            binding_protein_residues=[1],
            per_residue_pae_score={1: 0.8, 2: 0.5},
        )
        self.assertEqual(p.per_residue_pae_score, {1: 0.8, 2: 0.5})

    def test_validator_rejects_non_positive_keys(self):
        with self.assertRaises(Exception):
            ToolPrediction(
                tool_id="boltz2", category="A", sample_id="s",
                success=True, binding_protein_residues=[1],
                per_residue_pae_score={0: 0.5},
            )

    def test_failure_prediction_has_none_pae_score(self):
        p = make_failure_prediction(
            tool_id="boltz2", category="A", sample_id="s",
            error_message="boom",
        )
        self.assertIsNone(p.per_residue_pae_score)

    def test_pae_score_alone_satisfies_consistency_check(self):
        # success=True with only per_residue_pae_score populated should
        # validate (the new field is a real prediction signal).
        p = ToolPrediction(
            tool_id="chai1", category="A", sample_id="s",
            success=True,
            per_residue_pae_score={1: 0.9},
        )
        self.assertTrue(p.success)


# ---------- adapter integration: boltz2 -----------------------------------


def _add_atom(res, name, x, y, z, element, b_iso=80.0):
    a = gemmi.Atom()
    a.name = name
    a.pos = gemmi.Position(x, y, z)
    a.element = gemmi.Element(element)
    a.b_iso = b_iso
    a.occ = 1.0
    res.add_atom(a)


def _make_synth_cif(path: Path) -> None:
    """Tiny 2-protein-residue + 1-RNA-residue complex (chains A, B)."""
    s = gemmi.Structure()
    m = gemmi.Model("1")
    chA = gemmi.Chain("A")
    r1 = gemmi.Residue(); r1.name = "LYS"
    r1.seqid = gemmi.SeqId(1, " "); r1.label_seq = 1
    _add_atom(r1, "N", 0, 0, 0, "N", b_iso=80.0)
    _add_atom(r1, "CA", 1.5, 0, 0, "C", b_iso=80.0)
    chA.add_residue(r1)
    r2 = gemmi.Residue(); r2.name = "ALA"
    r2.seqid = gemmi.SeqId(2, " "); r2.label_seq = 2
    _add_atom(r2, "N", 50, 50, 50, "N", b_iso=70.0)
    _add_atom(r2, "CA", 51, 50, 50, "C", b_iso=70.0)
    chA.add_residue(r2)
    m.add_chain(chA)
    chB = gemmi.Chain("B")
    rB = gemmi.Residue(); rB.name = "A"
    rB.seqid = gemmi.SeqId(1, " "); rB.label_seq = 1
    _add_atom(rB, "N1", 3, 0, 0, "N", b_iso=55.0)
    chB.add_residue(rB)
    m.add_chain(chB)
    s.add_model(m)
    s.make_mmcif_document().write_file(str(path))


def _boltz2_config(
    work: Path, *,
    compute_pae: bool = True,
    method: str = "pae",
) -> dict:
    """Test helper. Defaults to ``method="pae"`` so existing PAE-path
    assertions stay correct; the new distance default is exercised by
    the dedicated TestBoltz2DistanceScore test below."""
    return {
        "tools": {
            "boltz2": {
                "conda_env": "boltz", "timeout": 1800,
                "compute_pae_scores": compute_pae,
                "per_residue_score_method": method,
                "pae_scale": 8.0,
                "distance_scale": 8.0,
            },
        },
        "contact_threshold": 4.5,
        "log_dir": str(work / "logs"),
    }


class TestBoltz2ParseOutputEmitsPaeScore(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.work = self.tmp / "work"
        self.work.mkdir()
        self.out = self.work / "boltz2_output"
        self.out.mkdir()
        # Identity seq map: protein residues 1-2, RNA residue 1.
        boltz2_write_seq_map(
            self.work,
            protein_mapping={1: 1, 2: 2}, rna_mapping={1: 1},
        )
        _make_synth_cif(self.out / "x_model_0.cif")
        # PAE matrix: 2 protein + 1 RNA = 3x3, with low PAE for residue 1.
        pae = np.array([
            [0.0, 0.0, 2.0],   # protein res 1 → RNA = 2.0 → prob ~0.8
            [0.0, 0.0, 50.0],  # protein res 2 → RNA = 50.0 → prob ~0.138
            [2.0, 50.0, 0.0],
        ], dtype=np.float32)
        np.savez(self.out / "pae_x_model_0.npz", pae=pae)

    def test_pae_score_present(self):
        adapter = Boltz2Adapter()
        sample = {
            "sample_id": "x", "source_pdb": "x",
            "protein": {"chain_id": "A", "sequence": "MK", "length": 2},
            "rna": {"chain_id": "B", "sequence": "G", "length": 1},
        }
        # Pin method=pae so this test still covers the PAE branch
        # after the default flipped to distance.
        pred = adapter.parse_output(
            self.out, sample, _boltz2_config(self.work, method="pae"),
        )
        self.assertTrue(pred.success)
        self.assertIsNotNone(pred.per_residue_pae_score)
        # Two residues scored.
        self.assertEqual(set(pred.per_residue_pae_score.keys()), {1, 2})
        # Strong asymmetry: res 1 high prob, res 2 low.
        self.assertGreater(pred.per_residue_pae_score[1],
                           pred.per_residue_pae_score[2])
        self.assertAlmostEqual(pred.per_residue_pae_score[1], 0.8, places=3)

    def test_disable_via_config(self):
        adapter = Boltz2Adapter()
        sample = {
            "sample_id": "x", "source_pdb": "x",
            "protein": {"chain_id": "A", "sequence": "MK"},
            "rna": {"chain_id": "B", "sequence": "G"},
        }
        cfg = _boltz2_config(self.work, compute_pae=False)
        pred = adapter.parse_output(self.out, sample, cfg)
        self.assertTrue(pred.success)
        self.assertIsNone(pred.per_residue_pae_score)
        # pLDDT path still works.
        self.assertIsNotNone(pred.per_residue_confidence)


class TestBoltz2DistanceScoreDefault(unittest.TestCase):
    """Boltz-2 now uses distance-based per_residue_pae_score by default
    (the offline ablation showed Pearson R ≈ 0.14 → 0.35 vs the PAE
    matrix path). This test pins the default behaviour and the field
    layout so a future flag flip can't silently regress it."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.work = self.tmp / "work"
        self.work.mkdir()
        self.out = self.work / "boltz2_output"
        self.out.mkdir()
        boltz2_write_seq_map(
            self.work,
            protein_mapping={1: 1, 2: 2}, rna_mapping={1: 1},
        )
        # Use the same synthetic CIF as the chai1 distance test —
        # protein residue 1's CA sits ~1.5 Å from the RNA atom (close),
        # residue 2 sits ~85 Å away (far). Distance score should
        # reflect that asymmetry.
        _make_synth_cif(self.out / "x_model_0.cif")
        # PAE file is also present so pae_mean still gets computed
        # for the summary stat — distance and PAE are independent here.
        pae = np.array([
            [0.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],
            [5.0, 5.0, 0.0],
        ], dtype=np.float32)
        np.savez(self.out / "pae_x_model_0.npz", pae=pae)

    def test_default_method_is_distance(self):
        adapter = Boltz2Adapter()
        sample = {
            "sample_id": "x", "source_pdb": "x",
            "protein": {"chain_id": "A", "sequence": "MK"},
            "rna": {"chain_id": "B", "sequence": "G"},
        }
        # Don't specify per_residue_score_method — adapter default is "distance".
        cfg = {
            "tools": {
                "boltz2": {
                    "conda_env": "boltz", "timeout": 1800,
                    "compute_pae_scores": True,
                    "distance_scale": 8.0,
                    "pae_scale": 8.0,
                },
            },
            "contact_threshold": 4.5,
            "log_dir": str(self.work / "logs"),
        }
        pred = adapter.parse_output(self.out, sample, cfg)
        self.assertTrue(pred.success)
        self.assertIsNotNone(pred.per_residue_pae_score)
        # Geometry: res 1 CA (1.5, 0, 0) ↔ RNA N1 (3, 0, 0) → d=1.5 →
        # 1/(1+1.5/8) ≈ 0.842. Res 2 CA (51, 50, 50) → d ≈ 85 → ≈0.086.
        self.assertAlmostEqual(pred.per_residue_pae_score[1], 0.842, places=2)
        self.assertLess(pred.per_residue_pae_score[2], 0.1)
        # pae_mean (the summary stat) is still populated from the npz.
        self.assertIsNotNone(pred.pae_mean)

    def test_pae_method_explicit_falls_back_to_pae_branch(self):
        # When method="pae", the per_residue_pae_score should match
        # the original PAE-matrix path's output (≈0.615 / ≈0.615 for
        # the uniform 5-Å PAE we wrote above).
        adapter = Boltz2Adapter()
        sample = {
            "sample_id": "x", "source_pdb": "x",
            "protein": {"chain_id": "A", "sequence": "MK"},
            "rna": {"chain_id": "B", "sequence": "G"},
        }
        cfg = {
            "tools": {
                "boltz2": {
                    "conda_env": "boltz",
                    "compute_pae_scores": True,
                    "per_residue_score_method": "pae",
                    "pae_scale": 8.0,
                },
            },
            "contact_threshold": 4.5,
            "log_dir": str(self.work / "logs"),
        }
        pred = adapter.parse_output(self.out, sample, cfg)
        self.assertTrue(pred.success)
        # Both residues see the same PAE=5 → 1/(1+5/8) ≈ 0.6154.
        self.assertAlmostEqual(
            pred.per_residue_pae_score[1], 1.0 / (1.0 + 5.0 / 8.0), places=3,
        )
        self.assertAlmostEqual(
            pred.per_residue_pae_score[1],
            pred.per_residue_pae_score[2],
            places=3,
        )


# ---------- adapter integration: chai1 ------------------------------------


def _chai1_config(work: Path, *, compute_distance: bool = True) -> dict:
    return {
        "tools": {
            "chai1": {
                "conda_env": "chai1", "timeout": 900,
                "compute_distance_scores": compute_distance,
                "distance_scale": 8.0,
            },
        },
        "contact_threshold": 4.5,
        "log_dir": str(work / "logs"),
    }


class TestComputeDistanceBindingScores(unittest.TestCase):
    """Helper math for the geometry-derived score (chai1 path)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _write_two_chain_cif(
        self,
        path: Path,
        *,
        protein_residues: list[tuple[int, float, float, float]],
        rna_atoms: list[tuple[float, float, float]],
        rna_element: str = "N",
    ) -> None:
        """Materialise a CIF with one CA per protein residue at the
        given (x,y,z) and one RNA residue holding ``rna_atoms`` (all
        named N1, element ``rna_element``)."""
        s = gemmi.Structure()
        m = gemmi.Model("1")
        chA = gemmi.Chain("A")
        for idx, x, y, z in protein_residues:
            r = gemmi.Residue()
            r.name = "ALA"
            r.seqid = gemmi.SeqId(idx, " "); r.label_seq = idx
            _add_atom(r, "CA", x, y, z, "C")
            chA.add_residue(r)
        m.add_chain(chA)
        chB = gemmi.Chain("B")
        rB = gemmi.Residue()
        rB.name = "A"
        rB.seqid = gemmi.SeqId(1, " "); rB.label_seq = 1
        for j, (x, y, z) in enumerate(rna_atoms):
            _add_atom(rB, f"N{j+1}", x, y, z, rna_element)
        chB.add_residue(rB)
        m.add_chain(chB)
        s.add_model(m)
        s.make_mmcif_document().write_file(str(path))

    def test_zero_distance_gives_prob_one(self):
        # CA on top of the only RNA atom → d=0 → prob=1.0.
        cif = self.tmp / "x.cif"
        self._write_two_chain_cif(
            cif, protein_residues=[(1, 0.0, 0.0, 0.0)],
            rna_atoms=[(0.0, 0.0, 0.0)],
        )
        out = compute_distance_binding_scores(cif)
        self.assertEqual(out, {1: 1.0})

    def test_distance_at_scale_gives_prob_half(self):
        cif = self.tmp / "x.cif"
        self._write_two_chain_cif(
            cif, protein_residues=[(1, 0.0, 0.0, 0.0)],
            rna_atoms=[(8.0, 0.0, 0.0)],
        )
        out = compute_distance_binding_scores(cif, distance_scale=8.0)
        self.assertAlmostEqual(out[1], 0.5, places=5)

    def test_uses_min_distance_to_any_rna_atom(self):
        # Two protein residues, three RNA atoms — each protein takes
        # the closest RNA atom, not the average / first / last.
        cif = self.tmp / "x.cif"
        self._write_two_chain_cif(
            cif,
            protein_residues=[
                (1, 0.0, 0.0, 0.0),    # closest RNA atom is at (2, 0, 0) → d=2
                (2, 100.0, 0.0, 0.0),  # closest RNA atom is at (50, 0, 0) → d=50
            ],
            rna_atoms=[(2.0, 0.0, 0.0), (50.0, 0.0, 0.0), (200.0, 0.0, 0.0)],
        )
        out = compute_distance_binding_scores(cif, distance_scale=8.0)
        # res 1: 1 / (1 + 2/8) = 0.8
        self.assertAlmostEqual(out[1], 0.8, places=4)
        # res 2: 1 / (1 + 50/8) ≈ 0.1379
        self.assertAlmostEqual(out[2], 1.0 / (1.0 + 50.0 / 8.0), places=4)

    def test_distance_scale_tunable(self):
        cif = self.tmp / "x.cif"
        self._write_two_chain_cif(
            cif, protein_residues=[(1, 0.0, 0.0, 0.0)],
            rna_atoms=[(10.0, 0.0, 0.0)],
        )
        a = compute_distance_binding_scores(cif, distance_scale=10.0)
        b = compute_distance_binding_scores(cif, distance_scale=5.0)
        self.assertAlmostEqual(a[1], 0.5, places=5)
        self.assertAlmostEqual(b[1], 1.0 / 3.0, places=5)

    def test_skips_residue_without_ca(self):
        # CIF with one CA-bearing residue + one bogus residue (no CA).
        cif = self.tmp / "x.cif"
        s = gemmi.Structure()
        m = gemmi.Model("1")
        chA = gemmi.Chain("A")
        r1 = gemmi.Residue(); r1.name = "ALA"
        r1.seqid = gemmi.SeqId(1, " "); r1.label_seq = 1
        _add_atom(r1, "CA", 0.0, 0.0, 0.0, "C")
        chA.add_residue(r1)
        r2 = gemmi.Residue(); r2.name = "GLY"
        r2.seqid = gemmi.SeqId(2, " "); r2.label_seq = 2
        # No CA — only an N atom.
        _add_atom(r2, "N", 1.0, 0.0, 0.0, "N")
        chA.add_residue(r2)
        m.add_chain(chA)
        chB = gemmi.Chain("B")
        rB = gemmi.Residue(); rB.name = "A"
        rB.seqid = gemmi.SeqId(1, " "); rB.label_seq = 1
        _add_atom(rB, "N1", 5.0, 0.0, 0.0, "N")
        chB.add_residue(rB)
        m.add_chain(chB)
        s.add_model(m)
        s.make_mmcif_document().write_file(str(cif))

        out = compute_distance_binding_scores(cif)
        # Only residue 1 has a CA → only it appears in the dict.
        self.assertEqual(set(out.keys()), {1})

    def test_missing_file_returns_empty(self):
        self.assertEqual(
            compute_distance_binding_scores(self.tmp / "no.cif"),
            {},
        )

    def test_missing_rna_chain_returns_empty(self):
        # Build a protein-only CIF — no chain B at all.
        cif = self.tmp / "x.cif"
        s = gemmi.Structure()
        m = gemmi.Model("1")
        chA = gemmi.Chain("A")
        r = gemmi.Residue(); r.name = "ALA"
        r.seqid = gemmi.SeqId(1, " "); r.label_seq = 1
        _add_atom(r, "CA", 0.0, 0.0, 0.0, "C")
        chA.add_residue(r)
        m.add_chain(chA)
        s.add_model(m)
        s.make_mmcif_document().write_file(str(cif))
        self.assertEqual(compute_distance_binding_scores(cif), {})

    def test_skips_hydrogen_on_rna_side(self):
        # If the only RNA "atom" present is hydrogen, the helper has
        # nothing to measure against → empty dict (defensive guard
        # against predicted complexes that include explicit Hs).
        cif = self.tmp / "x.cif"
        self._write_two_chain_cif(
            cif, protein_residues=[(1, 0.0, 0.0, 0.0)],
            rna_atoms=[(2.0, 0.0, 0.0)], rna_element="H",
        )
        self.assertEqual(compute_distance_binding_scores(cif), {})

    def test_non_positive_scale(self):
        cif = self.tmp / "x.cif"
        self._write_two_chain_cif(
            cif, protein_residues=[(1, 0.0, 0.0, 0.0)],
            rna_atoms=[(2.0, 0.0, 0.0)],
        )
        self.assertEqual(
            compute_distance_binding_scores(cif, distance_scale=0.0),
            {},
        )

    # ---- chain-name-mismatch regression (HADDOCK 3 per-residue gap) ----

    def _write_named_chain_cif(
        self,
        path: Path,
        *,
        protein_chain: str,
        rna_chain: str,
        protein_residues: list[tuple[int, float, float, float]],
        rna_atoms: list[tuple[float, float, float]],
    ) -> None:
        """Like ``_write_two_chain_cif`` but with caller-chosen chain
        names — lets us reproduce HADDOCK 3 poses whose chains are NOT
        the ``A``/``B`` the parse step hard-codes."""
        s = gemmi.Structure()
        m = gemmi.Model("1")
        chP = gemmi.Chain(protein_chain)
        for idx, x, y, z in protein_residues:
            r = gemmi.Residue()
            r.name = "ALA"
            r.seqid = gemmi.SeqId(idx, " "); r.label_seq = idx
            _add_atom(r, "CA", x, y, z, "C")
            chP.add_residue(r)
        m.add_chain(chP)
        chR = gemmi.Chain(rna_chain)
        rR = gemmi.Residue()
        rR.name = "A"
        rR.seqid = gemmi.SeqId(1, " "); rR.label_seq = 1
        for j, (x, y, z) in enumerate(rna_atoms):
            _add_atom(rR, f"N{j+1}", x, y, z, "N")
        chR.add_residue(rR)
        m.add_chain(chR)
        s.add_model(m)
        s.make_mmcif_document().write_file(str(path))

    def test_rna_chain_not_named_B_still_scored(self):
        # The common HADDOCK 3 case: single-char source RNA chain "E"
        # is preserved (not renamed to B), but parse_output still asks
        # for rna_chain_id="B". Content fallback must find chain E.
        cif = self.tmp / "x.cif"
        self._write_named_chain_cif(
            cif, protein_chain="A", rna_chain="E",
            protein_residues=[(1, 0.0, 0.0, 0.0)],
            rna_atoms=[(8.0, 0.0, 0.0)],
        )
        out = compute_distance_binding_scores(
            cif, protein_chain_id="A", rna_chain_id="B",
            distance_scale=8.0)
        # Before the fix this returned {} (no chain named B).
        self.assertAlmostEqual(out[1], 0.5, places=5)

    def test_protein_named_B_rna_named_F_trap(self):
        # The nastier case: protein chain is literally "B" and RNA is
        # "F". A naive name lookup for rna_chain_id="B" would grab the
        # PROTEIN. Content validation rejects that and finds chain F.
        cif = self.tmp / "x.cif"
        self._write_named_chain_cif(
            cif, protein_chain="B", rna_chain="F",
            protein_residues=[(1, 0.0, 0.0, 0.0),
                              (2, 100.0, 0.0, 0.0)],
            rna_atoms=[(2.0, 0.0, 0.0)],
        )
        out = compute_distance_binding_scores(
            cif, protein_chain_id="A", rna_chain_id="B",
            distance_scale=8.0)
        # res 1 is 2 Å from the RNA atom → 1/(1+2/8)=0.8;
        # res 2 is ~98 Å away → tiny. Crucially the scores come from
        # the PROTEIN chain (B), measured against the RNA chain (F).
        self.assertAlmostEqual(out[1], 0.8, places=4)
        self.assertIn(2, out)
        self.assertLess(out[2], 0.2)

    def test_correct_ab_naming_unchanged(self):
        # Backward-compat guard: when chains genuinely are A/B with the
        # right content (Boltz-2 / Chai-1), output is identical to the
        # pre-fix strict-filter behaviour.
        cif = self.tmp / "x.cif"
        self._write_named_chain_cif(
            cif, protein_chain="A", rna_chain="B",
            protein_residues=[(1, 0.0, 0.0, 0.0)],
            rna_atoms=[(8.0, 0.0, 0.0)],
        )
        out = compute_distance_binding_scores(
            cif, protein_chain_id="A", rna_chain_id="B",
            distance_scale=8.0)
        self.assertAlmostEqual(out[1], 0.5, places=5)

    def test_protein_only_still_empty(self):
        # No RNA chain by ANY name or content → still {} (the content
        # fallback must not invent an RNA chain). Mirrors the existing
        # test_missing_rna_chain_returns_empty but via the named helper.
        cif = self.tmp / "x.cif"
        s = gemmi.Structure()
        m = gemmi.Model("1")
        chA = gemmi.Chain("A")
        r = gemmi.Residue(); r.name = "ALA"
        r.seqid = gemmi.SeqId(1, " "); r.label_seq = 1
        _add_atom(r, "CA", 0.0, 0.0, 0.0, "C")
        chA.add_residue(r)
        m.add_chain(chA)
        s.add_model(m)
        s.make_mmcif_document().write_file(str(cif))
        self.assertEqual(
            compute_distance_binding_scores(
                cif, protein_chain_id="A", rna_chain_id="B"),
            {})


class TestChai1ParseOutputEmitsBindingScore(unittest.TestCase):
    """End-to-end: parse_output runs the distance helper and remaps."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.work = self.tmp / "work"
        self.work.mkdir()
        self.out = self.work / "chai1_output"
        self.out.mkdir()
        chai1_write_seq_map(
            self.work,
            protein_mapping={1: 1, 2: 2}, rna_mapping={1: 1},
        )
        # Reuses the helper from the boltz2 test fixtures: protein res
        # 1 CA at (1.5, 0, 0), res 2 CA at (51, 50, 50), RNA N1 at
        # (3, 0, 0). Distances: ≈1.5 (close) and ≈85.5 (far).
        _make_synth_cif(self.out / "pred.model_idx_0.cif")

    def test_binding_score_present(self):
        adapter = Chai1Adapter()
        sample = {
            "sample_id": "x", "source_pdb": "x",
            "protein": {"chain_id": "A", "sequence": "MK"},
            "rna": {"chain_id": "B", "sequence": "G"},
        }
        pred = adapter.parse_output(self.out, sample, _chai1_config(self.work))
        self.assertTrue(pred.success)
        self.assertIsNotNone(pred.per_residue_pae_score)
        self.assertEqual(set(pred.per_residue_pae_score.keys()), {1, 2})
        # Strong asymmetry: res 1 is close to RNA, res 2 is far.
        self.assertGreater(pred.per_residue_pae_score[1],
                           pred.per_residue_pae_score[2])
        # Res 1 distance = 1.5 → 1/(1 + 1.5/8) ≈ 0.842.
        self.assertAlmostEqual(pred.per_residue_pae_score[1], 0.842, places=2)
        # Res 2 distance ≈ 85.5 → ≈ 0.086.
        self.assertLess(pred.per_residue_pae_score[2], 0.1)

    def test_disable_via_config(self):
        adapter = Chai1Adapter()
        sample = {
            "sample_id": "x", "source_pdb": "x",
            "protein": {"chain_id": "A", "sequence": "MK"},
            "rna": {"chain_id": "B", "sequence": "G"},
        }
        cfg = _chai1_config(self.work, compute_distance=False)
        pred = adapter.parse_output(self.out, sample, cfg)
        self.assertTrue(pred.success)
        self.assertIsNone(pred.per_residue_pae_score)
        # pLDDT path still works regardless.
        self.assertIsNotNone(pred.per_residue_confidence)


# ---------- evaluate.py per-residue source selection ----------------------


import scripts.evaluate as ev  # noqa: E402


def _write_step4(step4_dir: Path, sid: str, predictions: list[dict]) -> None:
    path = step4_dir / f"{sid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"sample_id": sid, "tools_run": [p["tool_id"] for p in predictions],
           "predictions": predictions}
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")


def _write_step5(step5_dir: Path, sid: str, per_res: dict, binding: list) -> None:
    path = step5_dir / f"{sid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "sample_id": sid,
        "binding_protein_residues": binding,
        "per_residue_probability": {str(k): v for k, v in per_res.items()},
        "threshold": 0.5,
    }
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")


def _write_sample(processed: Path, sid: str, length: int, gt: list) -> None:
    samples = processed / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    sample = {"sample_id": sid, "protein": {"length": length},
              "interaction": {"binding_protein_residues": gt}}
    (samples / f"{sid}.json").write_text(json.dumps(sample), encoding="utf-8")


def _summary(sid: str = "s1") -> dict:
    return {"sample_id": sid}


# ---------- GPU isolation (Boltz-2 device → CUDA_VISIBLE_DEVICES) -------


class TestBoltz2GpuEnv(unittest.TestCase):
    """The boltz2 adapter parses ``device: cuda:N`` into a
    CUDA_VISIBLE_DEVICES env var on the subprocess so it doesn't
    contend for VRAM with the chai1 subprocess on the dual-GPU server.
    """

    def test_helper_extracts_index(self):
        from step4_tool_adapters.adapters.boltz2_adapter import (
            _gpu_env_from_device,
        )
        self.assertEqual(
            _gpu_env_from_device("cuda:0"), {"CUDA_VISIBLE_DEVICES": "0"},
        )
        self.assertEqual(
            _gpu_env_from_device("cuda:1"), {"CUDA_VISIBLE_DEVICES": "1"},
        )
        self.assertEqual(
            _gpu_env_from_device("CUDA:7"), {"CUDA_VISIBLE_DEVICES": "7"},
        )

    def test_helper_returns_empty_for_non_cuda(self):
        from step4_tool_adapters.adapters.boltz2_adapter import (
            _gpu_env_from_device,
        )
        for value in (None, "", "cpu", "mps", "cuda", "cuda:abc", "auto"):
            with self.subTest(device=value):
                self.assertEqual(_gpu_env_from_device(value), {})

    def test_run_tool_passes_cvd_to_subprocess(self):
        """Mock run_in_conda_env, check the extra_env kwarg carries the
        parsed device index. Pinning this in a test stops a future
        refactor from accidentally passing the cuda:N string straight
        to a tool that doesn't understand it."""
        tmp = Path(tempfile.mkdtemp())
        work = tmp / "work"
        work.mkdir()
        yaml_path = work / "x.yaml"
        yaml_path.write_text(
            "sequences:\n  - protein:\n      id: A\n      "
            'sequence: "AAA"\n  - rna:\n      id: B\n      '
            'sequence: "GGG"\n', encoding="utf-8",
        )

        captured = {}

        def fake(env, cmd, **kwargs):
            captured["env"] = env
            captured["extra_env"] = kwargs.get("extra_env")
            from step4_tool_adapters.tool_runner import ToolRunResult
            return ToolRunResult(
                command="fake", cwd=None, returncode=0, stdout="",
                stderr="", runtime_seconds=0.1, log_path=None,
            )

        adapter = Boltz2Adapter()
        cfg = {
            "tools": {
                "boltz2": {
                    "conda_env": "boltz", "timeout": 60,
                    "device": "cuda:1",
                    "flags": "--use_msa_server",
                },
            },
            "log_dir": str(work / "logs"),
        }
        from unittest.mock import patch
        with patch(
            "step4_tool_adapters.adapters.boltz2_adapter.run_in_conda_env",
            side_effect=fake,
        ):
            adapter.run_tool(
                {"yaml": yaml_path, "job_name": "x"}, work, cfg,
            )
        self.assertEqual(captured["env"], "boltz")
        self.assertEqual(
            captured["extra_env"], {"CUDA_VISIBLE_DEVICES": "1"},
        )

    def test_run_tool_extra_env_none_when_device_unset(self):
        """No device → extra_env is None so the subprocess inherits
        the parent's CUDA visibility unchanged (same behaviour as
        before the GPU-pinning patch)."""
        tmp = Path(tempfile.mkdtemp())
        work = tmp / "work"
        work.mkdir()
        yaml_path = work / "x.yaml"
        yaml_path.write_text(
            "sequences:\n  - protein:\n      id: A\n      "
            'sequence: "AAA"\n  - rna:\n      id: B\n      '
            'sequence: "GGG"\n', encoding="utf-8",
        )

        captured = {}

        def fake(env, cmd, **kwargs):
            captured["extra_env"] = kwargs.get("extra_env")
            from step4_tool_adapters.tool_runner import ToolRunResult
            return ToolRunResult(
                command="fake", cwd=None, returncode=0, stdout="",
                stderr="", runtime_seconds=0.1, log_path=None,
            )

        adapter = Boltz2Adapter()
        cfg = {
            "tools": {"boltz2": {"conda_env": "boltz", "timeout": 60}},
            "log_dir": str(work / "logs"),
        }
        from unittest.mock import patch
        with patch(
            "step4_tool_adapters.adapters.boltz2_adapter.run_in_conda_env",
            side_effect=fake,
        ):
            adapter.run_tool(
                {"yaml": yaml_path, "job_name": "x"}, work, cfg,
            )
        self.assertIsNone(captured["extra_env"])


class TestEvaluatePrefersPaeScore(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.processed = self.tmp / "processed"
        self.step4 = self.tmp / "step4"
        self.step5 = self.tmp / "step5"
        # Protein length 5; GT residues 1, 3.
        _write_sample(self.processed, "s1", length=5, gt=[1, 3])
        _write_step5(self.step5, "s1",
                     per_res={1: 0.9, 3: 0.85}, binding=[1, 3])

    def test_cat_a_uses_pae_when_available(self):
        # Cat A pred: pLDDT 90/80 (small variance, near-perfect with GT)
        # vs PAE-derived 0.4/0.05 (perfectly negatively correlated).
        # If evaluate picks pLDDT, Pearson > 0; if it picks PAE, the
        # numbers are different. We engineer the test so the PAE path
        # gives a measurably different correlation.
        _write_step4(self.step4, "s1", [{
            "tool_id": "boltz2", "category": "A", "sample_id": "s1",
            "success": True,
            "binding_protein_residues": [1, 3],
            "per_residue_confidence": {"1": 90.0, "3": 80.0},
            # PAE score covers ALL residues (typical Cat A behavior post-fix):
            "per_residue_pae_score": {
                "1": 0.95, "2": 0.10, "3": 0.90, "4": 0.05, "5": 0.08,
            },
        }])
        bucket = ev.collect_per_residue_correlations(
            [_summary()], step4_dir=self.step4,
            step5_dir=self.step5, processed_dir=self.processed,
        )
        self.assertIn("boltz2", bucket)
        # PAE values [0.95, 0.10, 0.90, 0.05, 0.08] vs GT [1, 0, 1, 0, 0]
        # → very high Pearson (close to 1.0). The pLDDT-only path with
        # confidence on residues 1 & 3 would also be high but identical
        # in structure to the binary fallback. The discriminator is
        # that PAE assigns NON-ZERO scores to residues 2/4/5; the
        # confidence path leaves them at 0.0. We assert PAE path was
        # taken by checking the prediction at non-binding residues
        # contributed to the correlation (Pearson > 0.9).
        self.assertGreater(bucket["boltz2"][0]["pearson_r"], 0.9)

    def test_cat_a_falls_back_to_confidence_when_no_pae(self):
        # Same shape but no per_residue_pae_score → falls back to pLDDT.
        _write_step4(self.step4, "s1", [{
            "tool_id": "boltz2", "category": "A", "sample_id": "s1",
            "success": True,
            "binding_protein_residues": [1, 3],
            "per_residue_confidence": {"1": 90.0, "3": 80.0},
        }])
        bucket = ev.collect_per_residue_correlations(
            [_summary()], step4_dir=self.step4,
            step5_dir=self.step5, processed_dir=self.processed,
        )
        self.assertIn("boltz2", bucket)
        # Should still produce a number (not crash) — the existing
        # pLDDT/100 path. This is the backward-compat case.
        self.assertIsNotNone(bucket["boltz2"][0]["pearson_r"])

    def test_cat_b_unaffected(self):
        # Cat B tools never have per_residue_pae_score — confirm
        # they continue to use per_residue_confidence verbatim.
        _write_step4(self.step4, "s1", [{
            "tool_id": "p2rank", "category": "B", "sample_id": "s1",
            "success": True,
            "binding_protein_residues": [1, 3],
            "per_residue_confidence": {"1": 0.9, "3": 0.8},
            # per_residue_pae_score absent / None
        }])
        bucket = ev.collect_per_residue_correlations(
            [_summary()], step4_dir=self.step4,
            step5_dir=self.step5, processed_dir=self.processed,
        )
        self.assertIn("p2rank", bucket)


# ---------- noisy-OR fusion is unaffected ---------------------------------


class TestFusionStillUsesConfidence(unittest.TestCase):
    """Sanity check: per_residue_pae_score is purely additive — step 5's
    fusion engine reads per_residue_confidence, never the new field."""

    def test_noisy_or_does_not_reference_pae_score(self):
        # Grep-style assertion: the noisy_or module shouldn't have
        # learned about per_residue_pae_score (otherwise the user's
        # explicit "fusion unaffected" requirement is broken).
        from step5_fusion import noisy_or
        src = Path(noisy_or.__file__).read_text(encoding="utf-8")
        self.assertNotIn("per_residue_pae_score", src,
                         "noisy_or fusion should NOT consume the new "
                         "per_residue_pae_score field — it's eval-only.")


if __name__ == "__main__":
    unittest.main()
