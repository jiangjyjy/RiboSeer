"""Mock tests for scripts/tables/table06_complex_quality.py.

Heavy paths (gemmi structure reading, US-align subprocess) are mocked.
The geometric primitives (Kabsch, RMSD, interface, contacts, DockQ) are
exercised against synthetic coordinates with known answers; the driver
is exercised with both extract_complex_atoms and run_usalign_tmscore
patched out.
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import scripts.tables.table06_complex_quality as ecq  # noqa: E402


# ---- fixtures ------------------------------------------------------------


def _chain_atoms_from_dict(coords_per_res):
    """Build a ChainAtoms from ``{label_seq: list-of-(x,y,z)}``.

    The first atom in each residue is also used as the backbone atom
    (Cα for protein, P for RNA — the mock doesn't care which since the
    test fixtures drive each side directly)."""
    bb = {}
    heavy = {}
    for rid, atoms in coords_per_res.items():
        arr = np.asarray(atoms, dtype=np.float64)
        heavy[rid] = arr
        bb[rid] = arr[0].copy()
    return ecq.ChainAtoms(backbone=bb, heavy=heavy)


def _pred(tool_id, *, success=True, category="A",
          predicted_path="dummy.cif", binding=None):
    d = {"tool_id": tool_id, "category": category,
         "success": success,
         "predicted_structure_path": predicted_path}
    if binding is not None:
        d["binding_protein_residues"] = binding
    return d


def _write_sample(proc: Path, sid: str, *,
                  source_pdb="abcd", prot_chain="A", rna_chain="B",
                  binding=(10, 11, 12)):
    (proc / "samples").mkdir(parents=True, exist_ok=True)
    seq = "M" * 30
    (proc / "samples" / f"{sid}.json").write_text(json.dumps({
        "sample_id": sid,
        "source_pdb": source_pdb,
        "protein": {"chain_id": prot_chain, "sequence": seq,
                    "length": len(seq),
                    "resolved_residues": list(range(1, 31))},
        "rna": {"chain_id": rna_chain, "sequence": "AAAU",
                "length": 4,
                "resolved_residues": [1, 2, 3, 4]},
        "interaction": {"binding_protein_residues": list(binding),
                        "binding_rna_nucleotides": [1, 2]},
    }), encoding="utf-8")


def _write_step4(step4: Path, sid: str, predictions):
    step4.mkdir(parents=True, exist_ok=True)
    (step4 / f"{sid}.jsonl").write_text(
        json.dumps({"sample_id": sid, "predictions": predictions})
        + "\n", encoding="utf-8")


# ---- geometric primitives ------------------------------------------------


class TestKabschAndRmsd(unittest.TestCase):

    def test_identity_zero_rmsd(self):
        P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                     dtype=np.float64)
        self.assertAlmostEqual(ecq.superposed_rmsd(P, P), 0.0,
                               places=10)

    def test_rotated_set_still_zero_after_superpose(self):
        # 90° rotation about z; Kabsch should recover it.
        P = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],
                     dtype=np.float64)
        R = np.array([[0.0, -1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]])
        Q = P @ R.T + np.array([10, 5, -2])
        self.assertAlmostEqual(ecq.superposed_rmsd(P, Q), 0.0,
                               places=8)

    def test_translated_only_zero_after_superpose(self):
        P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                     dtype=np.float64)
        Q = P + np.array([100, 200, 300])
        self.assertAlmostEqual(ecq.superposed_rmsd(P, Q), 0.0,
                               places=10)

    def test_known_residual(self):
        # Two pairs match perfectly, two are offset by 2 in y → RMSD
        # = sqrt((0+0+4+4)/4) = sqrt(2). Centering doesn't zero this
        # since translation alone can't absorb the per-point offset.
        P = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]],
                     dtype=np.float64)
        Q = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 2, 0]],
                     dtype=np.float64)
        # The optimal rigid-body fit centers each set then aligns; the
        # residual is the spread that no rotation can fix.
        r = ecq.superposed_rmsd(P, Q)
        self.assertGreater(r, 0.5)
        self.assertLess(r, 2.0)

    def test_too_few_points_raises(self):
        P = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        with self.assertRaises(ValueError):
            ecq.kabsch_align(P, P)


# ---- interface / contacts -----------------------------------------------


class TestInterfaceAndContacts(unittest.TestCase):

    def test_interface_picks_close_protein_residues(self):
        # Protein residues at x=0, 5, 50; RNA at x=2 → only residue 0
        # is within 10 Å.
        prot = _chain_atoms_from_dict({
            1: [(0, 0, 0)], 2: [(5, 0, 0)], 3: [(50, 0, 0)]
        })
        rna = _chain_atoms_from_dict({10: [(2, 0, 0)]})
        # residue 1: distance 2 (in), residue 2: distance 3 (in),
        # residue 3: distance 48 (out)
        iface = ecq.interface_residues(prot, rna, cutoff=10.0)
        self.assertEqual(iface, {1, 2})

    def test_interface_empty_rna(self):
        prot = _chain_atoms_from_dict({1: [(0, 0, 0)]})
        rna = ecq.ChainAtoms(backbone={}, heavy={})
        self.assertEqual(ecq.interface_residues(prot, rna), set())

    def test_native_contacts_pair_set(self):
        prot = _chain_atoms_from_dict({
            1: [(0, 0, 0)], 2: [(100, 0, 0)]
        })
        rna = _chain_atoms_from_dict({
            10: [(1, 0, 0)],   # close to prot 1
            20: [(101, 0, 0)], # close to prot 2
        })
        c = ecq.native_contacts(prot, rna, cutoff=5.0)
        self.assertEqual(c, {(1, 10), (2, 20)})

    def test_native_contacts_just_outside_cutoff(self):
        prot = _chain_atoms_from_dict({1: [(0, 0, 0)]})
        rna = _chain_atoms_from_dict({10: [(6, 0, 0)]})
        # 6 > 5 → no contact
        self.assertEqual(ecq.native_contacts(prot, rna, cutoff=5.0),
                         set())


# ---- DockQ + composite -------------------------------------------------


class TestDockqAndComposite(unittest.TestCase):

    def test_dockq_perfect(self):
        # Fnat=1, iRMSD=0, LRMSD=0 → 1.0
        self.assertAlmostEqual(ecq.compute_dockq(1.0, 0.0, 0.0), 1.0,
                               places=10)

    def test_dockq_formula(self):
        # By hand: Fnat=0.6, iRMSD=1.5, LRMSD=8.5 → (0.6 + 0.5 + 0.5)/3
        self.assertAlmostEqual(
            ecq.compute_dockq(0.6, 1.5, 8.5),
            (0.6 + 0.5 + 0.5) / 3.0, places=10)

    def test_dockq_none_when_component_missing(self):
        self.assertIsNone(ecq.compute_dockq(None, 1.0, 1.0))
        self.assertIsNone(ecq.compute_dockq(0.5, None, 1.0))
        self.assertIsNone(ecq.compute_dockq(0.5, 1.0, None))


# ---- iRMSD / LRMSD / Fnat (with mock ChainAtoms) -----------------------


class TestComposedMetrics(unittest.TestCase):

    def _build_pair(self, *, prot_shift=(0, 0, 0),
                    rna_shift=(0, 0, 0)):
        """Build a (pred, ref) pair where pred is ``ref`` with the
        protein translated by ``prot_shift`` and the RNA translated by
        ``rna_shift`` (independent of the protein shift). Useful for
        isolating the LRMSD pathway."""
        ref_prot_coords = {r: [(float(r), 0.0, 0.0)]
                           for r in range(1, 11)}
        ref_rna_coords = {r: [(float(r), 5.0, 0.0)]
                          for r in range(1, 6)}
        pred_prot_coords = {
            r: [(float(r) + prot_shift[0],
                 prot_shift[1], prot_shift[2])]
            for r in range(1, 11)
        }
        pred_rna_coords = {
            r: [(float(r) + prot_shift[0] + rna_shift[0],
                 5.0 + prot_shift[1] + rna_shift[1],
                 prot_shift[2] + rna_shift[2])]
            for r in range(1, 6)
        }
        return (_chain_atoms_from_dict(pred_prot_coords),
                _chain_atoms_from_dict(pred_rna_coords),
                _chain_atoms_from_dict(ref_prot_coords),
                _chain_atoms_from_dict(ref_rna_coords))

    def test_perfect_alignment_zero_metrics(self):
        # protein+RNA identical → iRMSD=0, LRMSD=0, Fnat=1
        pp, pr, rp, rr = self._build_pair()
        # widen iface cutoff so the synthetic 5-Å sep counts.
        irmsd = ecq.compute_irmsd(pp, pr, rp, rr, cutoff=10.0)
        lrmsd = ecq.compute_lrmsd(pp, pr, rp, rr)
        fnat = ecq.compute_fnat(pp, pr, rp, rr, cutoff=6.0)
        self.assertAlmostEqual(irmsd, 0.0, places=8)
        self.assertAlmostEqual(lrmsd, 0.0, places=8)
        self.assertEqual(fnat, 1.0)

    def test_rigid_translation_lrmsd_still_zero(self):
        # Pure rigid-body shift of the WHOLE complex → after superposing
        # the proteins, the RNA should land on top of the reference RNA
        # (LRMSD = 0). iRMSD is also 0 (Kabsch absorbs translation).
        pp, pr, rp, rr = self._build_pair(prot_shift=(15, -7, 4))
        irmsd = ecq.compute_irmsd(pp, pr, rp, rr, cutoff=10.0)
        lrmsd = ecq.compute_lrmsd(pp, pr, rp, rr)
        self.assertAlmostEqual(irmsd, 0.0, places=6)
        self.assertAlmostEqual(lrmsd, 0.0, places=6)

    def test_rna_offset_lrmsd_equals_shift(self):
        # Shift only the RNA (extra translation applied AFTER the
        # protein shift). After superposing proteins, the RNA in pred
        # is exactly ``rna_shift`` away from ref RNA → LRMSD = ||shift||.
        pp, pr, rp, rr = self._build_pair(
            prot_shift=(0, 0, 0), rna_shift=(3, 4, 0))   # ||(3,4)|| = 5
        lrmsd = ecq.compute_lrmsd(pp, pr, rp, rr)
        self.assertAlmostEqual(lrmsd, 5.0, places=6)

    def test_fnat_partial(self):
        # Build a tiny case where pred recovers half the native
        # contacts. Reference: prot residues 1,2 each contact RNA 10
        # (within 5 Å); pred: only prot 1 contacts RNA 10.
        ref_prot = _chain_atoms_from_dict({
            1: [(0, 0, 0)], 2: [(0, 0, 0)], 3: [(100, 0, 0)],
        })
        ref_rna = _chain_atoms_from_dict({10: [(2, 0, 0)]})
        pred_prot = _chain_atoms_from_dict({
            1: [(0, 0, 0)], 2: [(100, 0, 0)], 3: [(100, 0, 0)],
        })
        pred_rna = _chain_atoms_from_dict({10: [(2, 0, 0)]})
        f = ecq.compute_fnat(pred_prot, pred_rna, ref_prot, ref_rna,
                             cutoff=5.0)
        # ref: {(1,10),(2,10)}; pred: {(1,10)} → 1/2 = 0.5
        self.assertAlmostEqual(f, 0.5)

    def test_too_few_iface_returns_none(self):
        # Only 1 interface residue → Kabsch needs ≥ 3 → None.
        ref_prot = _chain_atoms_from_dict({1: [(0, 0, 0)]})
        ref_rna = _chain_atoms_from_dict({10: [(1, 0, 0)]})
        pred_prot = _chain_atoms_from_dict({1: [(0, 0, 0)]})
        pred_rna = _chain_atoms_from_dict({10: [(1, 0, 0)]})
        self.assertIsNone(
            ecq.compute_irmsd(pred_prot, pred_rna, ref_prot, ref_rna))


# ---- US-align parsing & subprocess wrapper -----------------------------


class TestUsalignWrapper(unittest.TestCase):

    _MODERN = (
        "Aligned length= 87\n"
        "TM-score= 0.91234 (normalized by length of Structure_1: "
        "L=87, d0=3.36)\n"
        "TM-score= 0.85123 (normalized by length of Structure_2: "
        "L=92, d0=3.50)\n"
    )

    def test_returns_reference_normalised(self):
        cp = mock.Mock(returncode=0, stdout=self._MODERN, stderr="")
        with mock.patch.object(subprocess, "run", return_value=cp):
            tm = ecq.run_usalign_tmscore(
                "USalign", Path("p.cif"), Path("r.pdb"))
        # Structure_2 = reference → 0.85123
        self.assertAlmostEqual(tm, 0.85123, places=5)

    def test_returns_none_on_nonzero_rc(self):
        cp = mock.Mock(returncode=1, stdout="", stderr="boom")
        with mock.patch.object(subprocess, "run", return_value=cp):
            self.assertIsNone(ecq.run_usalign_tmscore(
                "USalign", Path("p.cif"), Path("r.pdb")))

    def test_returns_none_on_unparseable_output(self):
        cp = mock.Mock(returncode=0, stdout="hello world", stderr="")
        with mock.patch.object(subprocess, "run", return_value=cp):
            self.assertIsNone(ecq.run_usalign_tmscore(
                "USalign", Path("p.cif"), Path("r.pdb")))

    def test_returns_none_on_timeout(self):
        with mock.patch.object(
                subprocess, "run",
                side_effect=subprocess.TimeoutExpired("USalign", 10)):
            self.assertIsNone(ecq.run_usalign_tmscore(
                "USalign", Path("p.cif"), Path("r.pdb")))


# ---- consensus selection ------------------------------------------------


class TestConsensusSelection(unittest.TestCase):

    def test_picks_tool_with_highest_mean_prob(self):
        preds = [
            _pred("boltz2", binding=[1, 2, 3]),
            _pred("chai1", binding=[10, 11]),
            _pred("rosettafold2na", binding=[20, 21]),
        ]
        # Probs: boltz2 binding has mean 0.3; chai1 mean 0.9;
        # rosettafold2na mean 0.5 → chai1 wins.
        probs = {1: 0.3, 2: 0.3, 3: 0.3,
                 10: 0.95, 11: 0.85,
                 20: 0.5, 21: 0.5}
        self.assertEqual(
            ecq.select_consensus_tool(preds, probs), "chai1")

    def test_skips_failed_or_no_binding(self):
        preds = [
            _pred("boltz2", success=False, binding=[1, 2, 3]),
            _pred("chai1", binding=[]),                     # no binding
            _pred("rosettafold2na", binding=[1, 2]),
        ]
        probs = {1: 0.7, 2: 0.7}
        self.assertEqual(
            ecq.select_consensus_tool(preds, probs), "rosettafold2na")

    def test_returns_none_when_no_probs_or_no_preds(self):
        self.assertIsNone(ecq.select_consensus_tool([], {1: 0.5}))
        self.assertIsNone(ecq.select_consensus_tool(
            [_pred("boltz2", binding=[1])], {}))


# ---- driver + CLI -------------------------------------------------------


class TestDriverAndCli(unittest.TestCase):

    def setUp(self):
        # Force formula path: this fixture mocks ``extract_complex_atoms``
        # but not the DockQ package, so on a host with DockQ installed
        # ``_try_dockq_package`` would skip the fakes entirely and call
        # the real load_PDB on STUB files (fails). Patching DOCKQ_OK off
        # here keeps these tests exercising the same code path they did
        # before the DockQ-package refactor.
        self._dockq_patch = mock.patch.object(ecq, "DOCKQ_OK", False)
        self._dockq_patch.start()
        self.addCleanup(self._dockq_patch.stop)
        # Reset the one-shot warning flag so each test starts fresh.
        ecq._DOCKQ_WARNING_EMITTED = False
        self.tmp = Path(tempfile.mkdtemp())
        self.step4 = self.tmp / "step4"
        self.proc = self.tmp / "proc"
        self.raw = self.tmp / "raw"
        self.raw.mkdir()
        self.sids = []
        for i in range(3):
            sid = f"s{i:02d}"
            self.sids.append(sid)
            _write_sample(self.proc, sid, source_pdb=f"pdb{i}")
            (self.raw / f"pdb{i}.pdb").write_text("STUB", encoding="utf-8")
            # Two Cat A tools with predicted structure paths that we'll
            # claim exist on disk via a patch below.
            _write_step4(self.step4, sid, [
                _pred("boltz2",
                      predicted_path=str(self.tmp / f"b_{sid}.cif"),
                      binding=[10, 11, 12]),
                _pred("chai1",
                      predicted_path=str(self.tmp / f"c_{sid}.cif"),
                      binding=[10, 11, 12]),
                _pred("equipnas", category="C", binding=[10, 11, 12]),
            ])
            # Touch the predicted files so the driver's .is_file() guard
            # passes (their contents are irrelevant — extract is mocked).
            (self.tmp / f"b_{sid}.cif").write_text("STUB",
                                                  encoding="utf-8")
            (self.tmp / f"c_{sid}.cif").write_text("STUB",
                                                  encoding="utf-8")

    def _fake_extract(self, path):
        # Identical pred / ref structures → every metric is perfect.
        prot = _chain_atoms_from_dict({
            r: [(float(r), 0.0, 0.0)] for r in range(1, 16)})
        rna = _chain_atoms_from_dict({
            r: [(float(r), 4.0, 0.0)] for r in range(1, 6)})
        return prot, rna

    def _fake_build_ref(self, raw_dir, src_pdb, prot_chain, rna_chain,
                        dst):
        # Just touch the output so .is_file() check is satisfied.
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("REFSTUB", encoding="utf-8")
        return dst

    def test_evaluate_buckets_two_cat_a_tools(self):
        with mock.patch.object(ecq, "extract_complex_atoms",
                               side_effect=lambda p: self._fake_extract(p)), \
             mock.patch.object(ecq, "build_ref_complex_pdb",
                               side_effect=self._fake_build_ref), \
             mock.patch.object(ecq, "run_usalign_tmscore",
                               return_value=0.85):
            bucket, per_sample = ecq.evaluate(
                step4_dir=self.step4, processed_dir=self.proc,
                raw_dir=self.raw, sample_ids=self.sids,
                usalign_bin="USalign", enriched_model=None,
                work_dir=self.tmp / "work", timeout=10.0)
        self.assertEqual(set(bucket), {"boltz2", "chai1"})
        self.assertNotIn("equipnas", bucket)  # Cat C
        for tid in ("boltz2", "chai1"):
            self.assertEqual(len(bucket[tid]), len(self.sids))
            for row in bucket[tid]:
                self.assertAlmostEqual(row["tmscore"], 0.85)
                self.assertEqual(row["fnat"], 1.0)
                self.assertEqual(row["dockq"], 1.0)
                self.assertEqual(row["acceptable"], 1)
                self.assertEqual(row["medium"], 1)

    def test_docking_cat_d_tools_are_evaluated(self):
        # Regression: hdock / haddock3 report category 'D' but still emit
        # a 3-D complex. They must be bucketed (the old category=='A'
        # gate silently dropped them); equipnas (Cat C, not in
        # COMPLEX_TOOLS) must still be excluded.
        for sid in self.sids:
            _write_step4(self.step4, sid, [
                _pred("hdock", category="D",
                      predicted_path=str(self.tmp / f"h_{sid}.pdb"),
                      binding=[10, 11, 12]),
                _pred("haddock3", category="D",
                      predicted_path=str(self.tmp / f"k_{sid}.pdb"),
                      binding=[10, 11, 12]),
                _pred("equipnas", category="C", binding=[10, 11, 12]),
            ])
            (self.tmp / f"h_{sid}.pdb").write_text("STUB", encoding="utf-8")
            (self.tmp / f"k_{sid}.pdb").write_text("STUB", encoding="utf-8")
        with mock.patch.object(ecq, "extract_complex_atoms",
                               side_effect=lambda p: self._fake_extract(p)), \
             mock.patch.object(ecq, "build_ref_complex_pdb",
                               side_effect=self._fake_build_ref), \
             mock.patch.object(ecq, "run_usalign_tmscore",
                               return_value=0.7):
            bucket, _ = ecq.evaluate(
                step4_dir=self.step4, processed_dir=self.proc,
                raw_dir=self.raw, sample_ids=self.sids,
                usalign_bin="USalign", enriched_model=None,
                work_dir=self.tmp / "work", timeout=10.0)
        self.assertEqual(set(bucket), {"hdock", "haddock3"})
        self.assertNotIn("equipnas", bucket)
        for tid in ("hdock", "haddock3"):
            self.assertEqual(len(bucket[tid]), len(self.sids))

    def test_consensus_row_appended_when_model_given(self):
        # Build a tiny ridge EnrichedFusion that returns a fixed probs
        # dict — easier than training a real model.
        class FakeFusion:
            def predict_sample(self, _preds, _seq, _length):
                # Chai-1's binding [10,11,12] has high prob → consensus
                # should pick chai1.
                return {10: 0.9, 11: 0.9, 12: 0.9}
        with mock.patch.object(ecq, "extract_complex_atoms",
                               side_effect=lambda p: self._fake_extract(p)), \
             mock.patch.object(ecq, "build_ref_complex_pdb",
                               side_effect=self._fake_build_ref), \
             mock.patch.object(ecq, "run_usalign_tmscore",
                               return_value=0.85):
            bucket, per_sample = ecq.evaluate(
                step4_dir=self.step4, processed_dir=self.proc,
                raw_dir=self.raw, sample_ids=self.sids,
                usalign_bin="USalign",
                enriched_model=FakeFusion(),  # type: ignore[arg-type]
                work_dir=self.tmp / "work", timeout=10.0)
        self.assertIn("riboseer_consensus", bucket)
        self.assertEqual(len(bucket["riboseer_consensus"]),
                         len(self.sids))
        # All consensus rows should carry chosen_tool (tie → CAT_A_TOOLS
        # ordering picks boltz2 first since both binding sets are equal).
        chosen = {r["chosen_tool"] for r in bucket["riboseer_consensus"]}
        self.assertTrue(chosen <= {"boltz2", "chai1"})

    def test_aggregate_orders_tools_then_consensus(self):
        rows = ecq._aggregate({
            "boltz2": [{"tmscore": 0.9, "irmsd": 1.0, "lrmsd": 2.0,
                        "fnat": 0.8, "dockq": 0.6,
                        "acceptable": 1, "medium": 1}] * 5,
            "chai1": [{"tmscore": 0.7, "irmsd": 2.0, "lrmsd": 4.0,
                       "fnat": 0.6, "dockq": 0.4,
                       "acceptable": 1, "medium": 0}] * 9,
            "riboseer_consensus": [
                {"tmscore": 0.95, "irmsd": 0.5, "lrmsd": 1.5,
                 "fnat": 0.9, "dockq": 0.85,
                 "acceptable": 1, "medium": 1}] * 4,
        })
        # chai1 has most n; riboseer_consensus always last.
        self.assertEqual([r["method"] for r in rows],
                         ["chai1", "boltz2", "riboseer_consensus"])
        self.assertEqual(rows[0]["n_samples"], 9)

    def test_cli_writes_csvs(self):
        out = self.tmp / "eval" / "complex.csv"
        with mock.patch.object(ecq, "extract_complex_atoms",
                               side_effect=lambda p: self._fake_extract(p)), \
             mock.patch.object(ecq, "build_ref_complex_pdb",
                               side_effect=self._fake_build_ref), \
             mock.patch.object(ecq, "run_usalign_tmscore",
                               return_value=0.85):
            rc = ecq.main([
                "--step4-dir", str(self.step4),
                "--processed-dir", str(self.proc),
                "--raw-dir", str(self.raw),
                "--output", str(out),
                "--usalign-bin", "USalign",
            ])
        self.assertEqual(rc, 0)
        self.assertTrue(out.is_file())
        ps = out.with_name("complex_per_sample.csv")
        self.assertTrue(ps.is_file())
        with out.open(encoding="utf-8") as f:
            rd = list(csv.DictReader(f))
        self.assertEqual(list(rd[0].keys()), ecq._COLUMNS)
        methods = [r["method"] for r in rd]
        # Both Cat A tools represented; equipnas (Cat C) absent.
        self.assertIn("boltz2", methods)
        self.assertIn("chai1", methods)
        self.assertNotIn("equipnas", methods)

    def test_cli_returns_1_on_missing_step4_dir(self):
        rc = ecq.main([
            "--step4-dir", str(self.tmp / "nope"),
            "--processed-dir", str(self.proc),
            "--raw-dir", str(self.raw),
            "--output", str(self.tmp / "x.csv"),
        ])
        self.assertEqual(rc, 1)


# ---- DockQ-package path -------------------------------------------------


class TestDockqPackagePath(unittest.TestCase):
    """Drive ``_try_dockq_package`` + the evaluate_one wiring with the
    DockQ package mocked in. The real package is only on the server, so
    these tests force ``DOCKQ_OK=True`` and patch ``_dockq_load_PDB`` /
    ``_dockq_run`` to return synthetic chain maps."""

    def setUp(self):
        ecq._DOCKQ_WARNING_EMITTED = False
        # Pretend the package is importable for this class.
        self._on = mock.patch.object(ecq, "DOCKQ_OK", True)
        self._on.start()
        self.addCleanup(self._on.stop)

    def test_picks_best_interface_by_dockq(self):
        chain_map = {
            ("A", "B"): {"DockQ": 0.42, "iRMSD": 3.1, "LRMSD": 9.0,
                         "fnat": 0.55},
            # Decoy interface — higher DockQ wins.
            ("A", "C"): {"DockQ": 0.78, "iRMSD": 1.2, "LRMSD": 4.5,
                         "fnat": 0.80},
        }
        with mock.patch.object(ecq, "_dockq_load_PDB",
                               return_value=mock.Mock()), \
             mock.patch.object(ecq, "_dockq_run",
                               return_value=(chain_map, 0.78)):
            best = ecq._try_dockq_package(
                Path("pred.cif"), Path("ref.pdb"))
        self.assertIsNotNone(best)
        self.assertEqual(best["interface"], ("A", "C"))
        self.assertAlmostEqual(best["dockq"], 0.78)
        self.assertAlmostEqual(best["irmsd"], 1.2)
        self.assertAlmostEqual(best["lrmsd"], 4.5)
        self.assertAlmostEqual(best["fnat"], 0.80)

    def test_accepts_bare_dict_return(self):
        # Older DockQ builds return just the chain map, not a tuple.
        chain_map = {("A", "B"): {"DockQ": 0.5, "iRMSD": 2.0,
                                  "LRMSD": 6.0, "fnat": 0.5}}
        with mock.patch.object(ecq, "_dockq_load_PDB",
                               return_value=mock.Mock()), \
             mock.patch.object(ecq, "_dockq_run",
                               return_value=chain_map):
            best = ecq._try_dockq_package(
                Path("pred.cif"), Path("ref.pdb"))
        self.assertEqual(best["dockq"], 0.5)

    def test_load_pdb_failure_returns_none(self):
        with mock.patch.object(
                ecq, "_dockq_load_PDB",
                side_effect=RuntimeError("bad CIF")):
            self.assertIsNone(ecq._try_dockq_package(
                Path("pred.cif"), Path("ref.pdb")))

    def test_run_failure_returns_none(self):
        with mock.patch.object(ecq, "_dockq_load_PDB",
                               return_value=mock.Mock()), \
             mock.patch.object(
                 ecq, "_dockq_run",
                 side_effect=ValueError("interface alignment failed")):
            self.assertIsNone(ecq._try_dockq_package(
                Path("pred.cif"), Path("ref.pdb")))

    def test_empty_chain_map_returns_none(self):
        with mock.patch.object(ecq, "_dockq_load_PDB",
                               return_value=mock.Mock()), \
             mock.patch.object(ecq, "_dockq_run",
                               return_value=({}, 0.0)):
            self.assertIsNone(ecq._try_dockq_package(
                Path("pred.cif"), Path("ref.pdb")))

    def test_evaluate_one_uses_package_when_available(self):
        chain_map = {("A", "B"): {"DockQ": 0.62, "iRMSD": 1.8,
                                  "LRMSD": 5.5, "fnat": 0.7}}
        with mock.patch.object(ecq, "_dockq_load_PDB",
                               return_value=mock.Mock()), \
             mock.patch.object(ecq, "_dockq_run",
                               return_value=(chain_map, 0.62)), \
             mock.patch.object(ecq, "run_usalign_tmscore",
                               return_value=0.91), \
             mock.patch.object(
                 ecq, "extract_complex_atoms",
                 side_effect=AssertionError("formula path must NOT run "
                                            "when DockQ succeeds")):
            out = ecq.evaluate_one(
                pred_path=Path("pred.cif"), ref_path=Path("ref.pdb"),
                usalign_bin="USalign", timeout=10.0)
        self.assertEqual(out["tmscore"], 0.91)
        self.assertEqual(out["dockq"], 0.62)
        self.assertEqual(out["irmsd"], 1.8)
        self.assertEqual(out["lrmsd"], 5.5)
        self.assertEqual(out["fnat"], 0.7)
        self.assertEqual(out["acceptable"], 1)
        self.assertEqual(out["medium"], 1)

    def test_falls_back_to_formula_when_package_unavailable(self):
        # DOCKQ_OK off → formula path. The fixture's extract returns
        # identical pred/ref atoms, so iRMSD = LRMSD = 0, Fnat = 1,
        # DockQ = 1.
        prot = _chain_atoms_from_dict({
            r: [(float(r), 0.0, 0.0)] for r in range(1, 16)})
        rna = _chain_atoms_from_dict({
            r: [(float(r), 4.0, 0.0)] for r in range(1, 6)})
        with mock.patch.object(ecq, "DOCKQ_OK", False), \
             mock.patch.object(ecq, "extract_complex_atoms",
                               return_value=(prot, rna)), \
             mock.patch.object(ecq, "run_usalign_tmscore",
                               return_value=0.85):
            out = ecq.evaluate_one(
                pred_path=Path("pred.cif"), ref_path=Path("ref.pdb"),
                usalign_bin="USalign", timeout=10.0)
        self.assertEqual(out["dockq"], 1.0)
        self.assertEqual(out["acceptable"], 1)

    def test_falls_back_to_formula_when_package_call_fails(self):
        # Package present but raises → still get a row via formula.
        prot = _chain_atoms_from_dict({
            r: [(float(r), 0.0, 0.0)] for r in range(1, 16)})
        rna = _chain_atoms_from_dict({
            r: [(float(r), 4.0, 0.0)] for r in range(1, 6)})
        with mock.patch.object(
                ecq, "_dockq_load_PDB",
                side_effect=RuntimeError("malformed pred")), \
             mock.patch.object(ecq, "extract_complex_atoms",
                               return_value=(prot, rna)), \
             mock.patch.object(ecq, "run_usalign_tmscore",
                               return_value=0.85):
            out = ecq.evaluate_one(
                pred_path=Path("pred.cif"), ref_path=Path("ref.pdb"),
                usalign_bin="USalign", timeout=10.0)
        self.assertEqual(out["dockq"], 1.0)

    def test_warn_once_when_package_missing(self):
        # First call → emits stderr; second call → silent.
        import io
        captured = io.StringIO()
        with mock.patch.object(ecq, "DOCKQ_OK", False), \
             mock.patch.object(sys, "stderr", captured):
            self.assertIsNone(ecq._try_dockq_package(
                Path("a"), Path("b")))
            self.assertIsNone(ecq._try_dockq_package(
                Path("c"), Path("d")))
        msg = captured.getvalue()
        self.assertEqual(msg.count("DockQ package not importable"), 1)


# ---- skip paths ----------------------------------------------------------


class TestSkipPaths(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4 = self.tmp / "step4"
        self.proc = self.tmp / "proc"
        self.raw = self.tmp / "raw"
        self.raw.mkdir()

    def test_missing_step4(self):
        _write_sample(self.proc, "s1", source_pdb="x1")
        (self.raw / "x1.pdb").write_text("STUB", encoding="utf-8")
        bucket, _ = ecq.evaluate(
            step4_dir=self.step4, processed_dir=self.proc,
            raw_dir=self.raw, sample_ids=["s1"],
            usalign_bin="USalign", enriched_model=None,
            work_dir=self.tmp / "work", timeout=10.0)
        self.assertEqual(bucket, {})

    def test_ref_build_fail_skips_sample(self):
        _write_sample(self.proc, "s1", source_pdb="x1")
        (self.raw / "x1.pdb").write_text("STUB", encoding="utf-8")
        _write_step4(self.step4, "s1", [
            _pred("boltz2",
                  predicted_path=str(self.tmp / "b_s1.cif"),
                  binding=[1, 2])])
        (self.tmp / "b_s1.cif").write_text("STUB", encoding="utf-8")
        with mock.patch.object(
                ecq, "build_ref_complex_pdb",
                side_effect=ValueError("missing chain")):
            bucket, _ = ecq.evaluate(
                step4_dir=self.step4, processed_dir=self.proc,
                raw_dir=self.raw, sample_ids=["s1"],
                usalign_bin="USalign", enriched_model=None,
                work_dir=self.tmp / "work", timeout=10.0)
        self.assertEqual(bucket, {})


if __name__ == "__main__":
    unittest.main()
