"""Mock tests for scripts/tables/table05_pocket_geometry.py.

The structure-IO path (gemmi-based ``extract_chain_coords``) is mocked
out — the unit tests here drive the *pure* logic: pocket extraction,
DCC / DCA / IoU computation, Top-N hit decisions, aggregation, and the
CLI plumbing. Real gemmi extraction is exercised by the real-data
runs invoked manually with --step4-dir / --raw-dir.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import scripts.tables.table05_pocket_geometry as epg  # noqa: E402


# ---- helpers -------------------------------------------------------------


def _pred(tool_id, *, success=True, binding=None, pae=None, conf=None):
    d = {"tool_id": tool_id, "category": "A", "success": success}
    if binding is not None:
        d["binding_protein_residues"] = binding
    if pae is not None:
        d["per_residue_pae_score"] = pae
    if conf is not None:
        d["per_residue_confidence"] = conf
    return d


def _write_sample(proc: Path, sid: str, *, gt, prot_chain="A",
                  rna_chain="B", source_pdb="abcd", seq=None,
                  resolved=None):
    (proc / "samples").mkdir(parents=True, exist_ok=True)
    max_res = max(gt) if gt else 30
    resolved = resolved or list(range(1, max_res + 10))
    seq = seq or "".join("MKRYAVCDEF"[i % 10] for i in range(max(resolved)))
    (proc / "samples" / f"{sid}.json").write_text(json.dumps({
        "sample_id": sid,
        "source_pdb": source_pdb,
        "protein": {"chain_id": prot_chain, "sequence": seq,
                    "length": len(seq), "resolved_residues": resolved},
        "rna": {"chain_id": rna_chain, "sequence": "AAAU",
                "length": 4, "resolved_residues": [1, 2, 3, 4]},
        "interaction": {"binding_protein_residues": gt,
                        "binding_rna_nucleotides": [1, 2]},
    }), encoding="utf-8")


def _write_step4(step4: Path, sid: str, predictions):
    step4.mkdir(parents=True, exist_ok=True)
    (step4 / f"{sid}.jsonl").write_text(
        json.dumps({"sample_id": sid, "predictions": predictions}) + "\n",
        encoding="utf-8")


# ---- pocket extraction ---------------------------------------------------


class TestExtractPocketClusters(unittest.TestCase):

    def test_basic_top_k_and_clustering(self):
        # Residues 10,11,12 highest (cluster 1); 20,21 second (cluster 2);
        # rest low. k=5 should keep 10,11,12,20,21 → 2 clusters.
        scores = {r: 0.1 for r in range(1, 30)}
        scores.update({10: 0.9, 11: 0.85, 12: 0.8, 20: 0.7, 21: 0.65})
        cs = epg.extract_pocket_clusters(scores, k=5, max_gap=2)
        self.assertEqual(len(cs), 2)
        # First cluster has higher mean score → 10-12.
        self.assertEqual(cs[0], [10, 11, 12])
        self.assertEqual(cs[1], [20, 21])

    def test_empty_or_zero_k(self):
        self.assertEqual(epg.extract_pocket_clusters({}, k=3), [])
        self.assertEqual(
            epg.extract_pocket_clusters({1: 0.5}, k=0), [])

    def test_gap_threshold_merges_when_within(self):
        # 5,7 with gap=2 stays together; 5,8 with gap=2 splits.
        scores = {5: 0.9, 7: 0.85, 100: 0.5}
        cs = epg.extract_pocket_clusters(scores, k=2, max_gap=2)
        self.assertEqual(cs, [[5, 7]])
        scores2 = {5: 0.9, 8: 0.85, 100: 0.5}
        cs2 = epg.extract_pocket_clusters(scores2, k=2, max_gap=2)
        self.assertEqual(cs2, [[5], [8]])

    def test_tie_break_is_deterministic(self):
        # All scores equal → top-k picks the smallest residue ids.
        scores = {r: 0.5 for r in range(1, 11)}
        cs = epg.extract_pocket_clusters(scores, k=3, max_gap=2)
        # All-tied → adjacent residues fall in one cluster.
        self.assertEqual(cs, [[1, 2, 3]])

    def test_cluster_order_by_mean_score(self):
        # Single-residue cluster with very high score beats a fat
        # cluster with mediocre mean.
        scores = {5: 0.99,
                  20: 0.5, 21: 0.5, 22: 0.5, 23: 0.5}
        cs = epg.extract_pocket_clusters(scores, k=5, max_gap=2)
        # First cluster = the singleton {5}; second = {20..23}.
        self.assertEqual(cs[0], [5])
        self.assertEqual(cs[1], [20, 21, 22, 23])


# ---- geometry helpers ----------------------------------------------------


class TestGeometryHelpers(unittest.TestCase):

    def test_centroid_basic(self):
        ca = {1: (0, 0, 0), 2: (2, 0, 0), 3: (4, 0, 0)}
        c = epg._centroid([1, 2, 3], ca)
        self.assertEqual(c, (2.0, 0.0, 0.0))

    def test_centroid_missing_residues_dropped(self):
        ca = {1: (0, 0, 0), 2: (2, 0, 0)}
        c = epg._centroid([1, 2, 99], ca)
        self.assertEqual(c, (1.0, 0.0, 0.0))

    def test_centroid_none_when_all_missing(self):
        self.assertIsNone(epg._centroid([99], {1: (0, 0, 0)}))

    def test_dcc_perfect_overlap_is_zero(self):
        ca = {1: (0, 0, 0), 2: (3, 0, 0)}
        d = epg.compute_dcc([1, 2], [1, 2], ca)
        self.assertEqual(d, 0.0)

    def test_dcc_known_distance(self):
        ca = {1: (0, 0, 0), 2: (0, 0, 0), 3: (10, 0, 0), 4: (10, 0, 0)}
        # pred centroid (0,0,0), gt centroid (10,0,0) → 10.0
        self.assertAlmostEqual(
            epg.compute_dcc([1, 2], [3, 4], ca), 10.0)

    def test_dcc_none_on_missing_coords(self):
        ca = {1: (0, 0, 0)}
        self.assertIsNone(epg.compute_dcc([1], [99], ca))

    def test_dca_picks_nearest(self):
        ca = {1: (0, 0, 0)}
        rna = [(5, 0, 0), (3, 0, 0), (7, 0, 0)]
        self.assertAlmostEqual(epg.compute_dca([1], ca, rna), 3.0)

    def test_dca_none_when_no_rna(self):
        self.assertIsNone(
            epg.compute_dca([1], {1: (0, 0, 0)}, []))

    def test_iou_full_and_disjoint_and_partial(self):
        self.assertEqual(epg.compute_iou([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertEqual(epg.compute_iou([1, 2], [3, 4]), 0.0)
        # union={1,2,3,4} intersection={2,3} → 2/4 = 0.5
        self.assertEqual(epg.compute_iou([1, 2, 3], [2, 3, 4]), 0.5)

    def test_iou_empty_union(self):
        self.assertEqual(epg.compute_iou([], []), 0.0)


# ---- evaluate_sample_method ----------------------------------------------


class TestEvaluateSampleMethod(unittest.TestCase):
    """Drive the pack assembly logic with known coords + scores."""

    def setUp(self):
        # 30-residue protein, 5 RNA atoms, GT pocket = 10-14.
        self.ca = {r: (float(r), 0.0, 0.0) for r in range(1, 31)}
        self.rna = [(12.0, 5.0, 0.0)]   # distance 5 from residue-12 line
        self.gt = [10, 11, 12, 13, 14]

    def test_perfect_prediction(self):
        scores = {r: 0.1 for r in range(1, 31)}
        for r in self.gt:
            scores[r] = 0.9
        pack = epg.evaluate_sample_method(
            scores=scores, gt_residues=self.gt,
            ca=self.ca, rna_atoms=self.rna,
            hit_threshold=4.0)
        self.assertEqual(pack["iou"], 1.0)
        self.assertEqual(pack["dcc"], 0.0)
        self.assertEqual(pack["top1_hit"], 1)
        self.assertEqual(pack["top3_hit"], 1)

    def test_far_off_prediction(self):
        # All score weight on residues 25-29 — far from GT 10-14.
        scores = {r: 0.1 for r in range(1, 31)}
        for r in [25, 26, 27, 28, 29]:
            scores[r] = 0.9
        pack = epg.evaluate_sample_method(
            scores=scores, gt_residues=self.gt,
            ca=self.ca, rna_atoms=self.rna,
            hit_threshold=4.0)
        # IoU = 0 (sets disjoint); DCC = |27 - 12| = 15 Å.
        self.assertEqual(pack["iou"], 0.0)
        self.assertAlmostEqual(pack["dcc"], 15.0, places=2)
        self.assertEqual(pack["top1_hit"], 0)
        self.assertEqual(pack["top3_hit"], 0)

    def test_top3_rescues_when_top1_misses(self):
        # GT centroid is at residue 13 (5 residues, 10..14 → mean 12).
        # We arrange THREE clusters that all survive top-k=7 culling:
        #   1st (mean 0.94): 25-27 — far (centroid 26 → DCC≈14)
        #   2nd (mean 0.80): 3-4 — also far (centroid 3.5 → DCC≈8.5)
        #   3rd (mean 0.70): 12-13 — on the GT (centroid 12.5 → DCC≈0.5)
        # Bump GT to length 7 so all 7 high-score residues survive.
        gt = [10, 11, 12, 13, 14, 15, 16]   # k=7
        scores = {r: 0.1 for r in range(1, 31)}
        scores.update({25: 0.95, 26: 0.94, 27: 0.93})
        scores.update({3: 0.8, 4: 0.79})
        scores.update({12: 0.7, 13: 0.69})
        pack = epg.evaluate_sample_method(
            scores=scores, gt_residues=gt,
            ca=self.ca, rna_atoms=self.rna,
            hit_threshold=4.0)
        self.assertEqual(pack["top1_hit"], 0)
        self.assertEqual(pack["top3_hit"], 1)

    def test_returns_none_on_empty_inputs(self):
        self.assertIsNone(epg.evaluate_sample_method(
            scores={}, gt_residues=self.gt, ca=self.ca,
            rna_atoms=self.rna, hit_threshold=4.0))
        self.assertIsNone(epg.evaluate_sample_method(
            scores={1: 0.5}, gt_residues=[], ca=self.ca,
            rna_atoms=self.rna, hit_threshold=4.0))


# ---- aggregation + CLI ---------------------------------------------------


class TestAggregateAndCli(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4 = self.tmp / "step4"
        self.proc = self.tmp / "proc"
        self.raw = self.tmp / "raw"
        self.raw.mkdir()
        # Build 4 samples sharing the same fake structure layout.
        self.sids = []
        for i in range(4):
            sid = f"s{i:02d}"
            self.sids.append(sid)
            _write_sample(self.proc, sid,
                          gt=[10, 11, 12], source_pdb=f"pdb{i}",
                          resolved=list(range(1, 25)))
            # All "fake" PDB files exist so find_raw_pdb succeeds.
            (self.raw / f"pdb{i}.pdb").write_text("PDB stub\n",
                                                 encoding="utf-8")
            # Boltz-2 scores GT high; p2rank scores residues 20+ high
            # (so its pocket is far from GT).
            high_pae = {str(r): 0.9 for r in [10, 11, 12]}
            low_pae = {str(r): 0.1 for r in range(1, 25) if r not in
                       (10, 11, 12)}
            high_pae.update(low_pae)
            p2 = {str(r): 0.9 for r in [20, 21, 22]}
            for r in range(1, 25):
                p2.setdefault(str(r), 0.1)
            _write_step4(self.step4, sid, [
                _pred("boltz2", binding=[10, 11, 12], pae=high_pae),
                _pred("p2rank", binding=[20, 21, 22], conf=p2),
                _pred("equipnas", success=False),
            ])

    def _fake_extract(self, struct_path, prot_chain_id, rna_chain_id):
        # CA along x: residue r at (r, 0, 0). Single RNA atom near
        # residue 11.5 line (so DCA stays small even for the far pocket
        # — distinguishes DCC from DCA in the aggregate row).
        ca = {r: (float(r), 0.0, 0.0) for r in range(1, 25)}
        rna = [(11.5, 5.0, 0.0)]
        return ca, rna

    def test_evaluate_buckets_by_method(self):
        with mock.patch.object(epg, "extract_chain_coords",
                               side_effect=self._fake_extract):
            bucket, per_sample = epg.evaluate(
                step4_dir=self.step4, processed_dir=self.proc,
                raw_dir=self.raw, sample_ids=self.sids,
                hit_threshold=4.0, enriched_model=None)
        self.assertEqual(set(bucket), {"boltz2", "p2rank"})
        self.assertNotIn("equipnas", bucket)  # success=False
        self.assertEqual(len(bucket["boltz2"]), len(self.sids))
        # Boltz-2 had GT-aligned scores → DCC ≈ 0 on every sample.
        for r in bucket["boltz2"]:
            self.assertAlmostEqual(r["dcc"], 0.0, places=4)
            self.assertEqual(r["top1_hit"], 1)
        # P2Rank scored residues 20-22 → centroid 21, GT centroid 11
        # → DCC = 10, no hit.
        for r in bucket["p2rank"]:
            self.assertAlmostEqual(r["dcc"], 10.0, places=2)
            self.assertEqual(r["top1_hit"], 0)
            self.assertEqual(r["top3_hit"], 0)
        # per_sample mirrors bucket.
        self.assertEqual(
            len(per_sample), 2 * len(self.sids))  # 2 successful tools

    def test_aggregate_orders_tools_by_n_desc(self):
        rows = epg._aggregate({
            "a": [{"dcc": 1.0, "dca": 2.0, "iou": 0.5,
                   "top1_hit": 1, "top3_hit": 1}] * 3,
            "b": [{"dcc": 4.0, "dca": 6.0, "iou": 0.1,
                   "top1_hit": 0, "top3_hit": 1}] * 9,
            "enriched_fusion": [{"dcc": 0.5, "dca": 1.0, "iou": 0.9,
                                 "top1_hit": 1, "top3_hit": 1}] * 1,
        })
        self.assertEqual([r["method"] for r in rows],
                         ["b", "a", "enriched_fusion"])
        self.assertEqual(rows[0]["n_samples"], 9)
        # top1_success is a percentage.
        self.assertAlmostEqual(rows[-1]["top1_success"], 100.0)
        self.assertAlmostEqual(rows[0]["top1_success"], 0.0)

    def test_cli_writes_csvs_and_returns_zero(self):
        out = self.tmp / "eval" / "pocket_geom.csv"
        with mock.patch.object(epg, "extract_chain_coords",
                               side_effect=self._fake_extract):
            rc = epg.main([
                "--step4-dir", str(self.step4),
                "--processed-dir", str(self.proc),
                "--raw-dir", str(self.raw),
                "--output", str(out),
            ])
        self.assertEqual(rc, 0)
        self.assertTrue(out.is_file())
        ps = out.with_name("pocket_geom_per_sample.csv")
        self.assertTrue(ps.is_file())
        with out.open(encoding="utf-8") as f:
            rd = list(csv.DictReader(f))
        self.assertEqual(list(rd[0].keys()), epg._COLUMNS)
        methods = [r["method"] for r in rd]
        self.assertIn("boltz2", methods)
        self.assertNotIn("equipnas", methods)
        # Boltz row → top1_success=100.0
        boltz_row = next(r for r in rd if r["method"] == "boltz2")
        self.assertEqual(float(boltz_row["top1_success"]), 100.0)
        # P2rank row → 0.0
        p2_row = next(r for r in rd if r["method"] == "p2rank")
        self.assertEqual(float(p2_row["top1_success"]), 0.0)

    def test_cli_missing_step4_dir_returns_1(self):
        rc = epg.main([
            "--step4-dir", str(self.tmp / "nope"),
            "--processed-dir", str(self.proc),
            "--raw-dir", str(self.raw),
            "--output", str(self.tmp / "x.csv"),
        ])
        self.assertEqual(rc, 1)

    def test_cli_missing_raw_dir_returns_1(self):
        rc = epg.main([
            "--step4-dir", str(self.step4),
            "--processed-dir", str(self.proc),
            "--raw-dir", str(self.tmp / "nope"),
            "--output", str(self.tmp / "x.csv"),
        ])
        self.assertEqual(rc, 1)


# ---- skip-counting -------------------------------------------------------


class TestSkipPaths(unittest.TestCase):
    """Each skip reason returns cleanly (no traceback) and the affected
    sample contributes no rows."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4 = self.tmp / "step4"
        self.proc = self.tmp / "proc"
        self.raw = self.tmp / "raw"
        self.raw.mkdir()

    def test_missing_step4_jsonl_skips(self):
        _write_sample(self.proc, "s1", gt=[1, 2, 3], source_pdb="x1")
        (self.raw / "x1.pdb").write_text("X\n", encoding="utf-8")
        bucket, ps = epg.evaluate(
            step4_dir=self.step4, processed_dir=self.proc,
            raw_dir=self.raw, sample_ids=["s1"],
            hit_threshold=4.0, enriched_model=None)
        self.assertEqual(bucket, {})
        self.assertEqual(ps, [])

    def test_missing_gt_skips(self):
        _write_sample(self.proc, "s1", gt=[], source_pdb="x1")
        _write_step4(self.step4, "s1",
                     [_pred("boltz2", pae={"1": 0.9})])
        (self.raw / "x1.pdb").write_text("X\n", encoding="utf-8")
        bucket, _ = epg.evaluate(
            step4_dir=self.step4, processed_dir=self.proc,
            raw_dir=self.raw, sample_ids=["s1"],
            hit_threshold=4.0, enriched_model=None)
        self.assertEqual(bucket, {})

    def test_missing_structure_skips(self):
        _write_sample(self.proc, "s1", gt=[10, 11, 12], source_pdb="xx")
        _write_step4(self.step4, "s1",
                     [_pred("boltz2", pae={"10": 0.9})])
        # No xx.pdb / xx.cif under raw/.
        bucket, _ = epg.evaluate(
            step4_dir=self.step4, processed_dir=self.proc,
            raw_dir=self.raw, sample_ids=["s1"],
            hit_threshold=4.0, enriched_model=None)
        self.assertEqual(bucket, {})

    def test_structure_read_fail_skips(self):
        _write_sample(self.proc, "s1", gt=[10, 11, 12], source_pdb="x1")
        _write_step4(self.step4, "s1",
                     [_pred("boltz2", pae={"10": 0.9})])
        (self.raw / "x1.pdb").write_text("X\n", encoding="utf-8")
        with mock.patch.object(epg, "extract_chain_coords",
                               side_effect=ValueError("bad CIF")):
            bucket, _ = epg.evaluate(
                step4_dir=self.step4, processed_dir=self.proc,
                raw_dir=self.raw, sample_ids=["s1"],
                hit_threshold=4.0, enriched_model=None)
        self.assertEqual(bucket, {})


if __name__ == "__main__":
    unittest.main()
