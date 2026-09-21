"""Mock tests for scripts/cluster_stats.py."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.cluster_stats as cs  # noqa: E402


def _splits(samples_meta: dict, *, stats: dict | None = None) -> dict:
    return {
        "config": {"protein_min_seq_id": 0.3, "rna_min_seq_id": 0.8,
                   "coverage": 0.8, "split_ratio": [0.8, 0.1, 0.1],
                   "seed": 42},
        "stats": stats or {"n_samples": len(samples_meta),
                           "n_prot_clusters": 3,
                           "n_rna_clusters": 2,
                           "n_clusterable_samples": len(samples_meta)},
        "cluster_index": {"protein": {}, "rna": {}},
        "samples": samples_meta,
    }


def _meta(pc, rc, sp):
    return {"protein_cluster_id": pc, "rna_cluster_id": rc, "split": sp}


class TestSampleMeta(unittest.TestCase):
    def test_lookup_case_insensitive_and_missing(self):
        s = _splits({"AbC_X_Y": _meta("prot_1", "rna_1", "train")})
        self.assertEqual(cs.sample_meta(s, "AbC_X_Y")["split"],
                         "train")
        # case-insensitive fallback
        self.assertEqual(cs.sample_meta(s, "abc_x_y")["split"],
                         "train")
        self.assertEqual(cs.sample_meta(s, "missing"), {})


class TestSummarise(unittest.TestCase):
    def setUp(self):
        # train: 3 samples, 2 protein clusters, 2 rna clusters
        # test:  2 samples, 1 protein cluster,  1 rna cluster (disjoint
        #                                                       from train)
        # val:   1 sample
        self.s = _splits({
            "a": _meta("prot_1", "rna_1", "train"),
            "b": _meta("prot_1", "rna_1", "train"),  # same cluster as a
            "c": _meta("prot_2", "rna_2", "train"),
            "d": _meta("prot_3", "rna_3", "test"),
            "e": _meta("prot_3", "rna_3", "test"),
            "f": _meta("prot_4", "rna_4", "val"),
        })

    def test_basic_counts(self):
        r = cs.summarise_subset(
            self.s, ["a", "b", "c", "d", "e", "f"])
        self.assertEqual(r["n_samples"], 6)
        self.assertEqual(r["n_protein_clusters"], 4)
        self.assertEqual(r["n_rna_clusters"], 4)
        ps = r["per_split"]
        self.assertEqual(ps["train"]["samples"], 3)
        self.assertEqual(ps["train"]["protein_clusters"], 2)
        self.assertEqual(ps["train"]["rna_clusters"], 2)
        self.assertEqual(ps["test"]["samples"], 2)
        self.assertEqual(ps["test"]["protein_clusters"], 1)
        # group-level leakage must be empty for a leakage-safe split
        self.assertEqual(
            r["leakage"]["group_overlap_count"], 0)
        # single-axis cluster overlap is also 0 in this contrived case
        self.assertEqual(
            r["leakage"]["protein_cluster_overlap_count"], 0)
        self.assertEqual(
            r["leakage"]["rna_cluster_overlap_count"], 0)
        self.assertEqual(r["n_groups"], 4)
        self.assertEqual(ps["train"]["groups"], 2)
        self.assertEqual(ps["test"]["groups"], 1)

    def test_cluster_size_dist(self):
        r = cs.summarise_subset(self.s, ["a", "b", "c", "d", "e", "f"])
        # prot_1 has 2 members (a,b); prot_2/3/4 have 1/2/1
        ps = r["protein_cluster_size_stats"]
        self.assertEqual(ps["n"], 4)
        self.assertEqual((ps["min"], ps["max"]), (1, 2))

    def test_leakage_detection(self):
        # contrive a leak: same (prot_3, rna_3) group in train + test
        leaky = dict(self.s)
        leaky["samples"] = dict(self.s["samples"])
        leaky["samples"]["d"] = _meta("prot_3", "rna_3", "train")
        leaky["samples"]["e"] = _meta("prot_3", "rna_3", "test")
        r = cs.summarise_subset(
            leaky, ["a", "b", "c", "d", "e", "f"])
        # group-level leak detected
        self.assertEqual(r["leakage"]["group_overlap_count"], 1)
        self.assertIn("prot_3/rna_3",
                      r["leakage"]["group_overlap_examples"])
        # single-axis overlap also non-zero here
        self.assertEqual(
            r["leakage"]["protein_cluster_overlap_count"], 1)
        self.assertIn(
            "prot_3", r["leakage"]["protein_cluster_overlap_ids"])

    def test_single_axis_overlap_without_group_leak(self):
        # Same protein cluster on both sides but with DIFFERENT RNA
        # partners -> group is disjoint, single-axis overlap exists.
        s = _splits({
            "a": _meta("prot_1", "rna_1", "train"),
            "b": _meta("prot_1", "rna_2", "test"),
        })
        r = cs.summarise_subset(s, ["a", "b"])
        self.assertEqual(r["leakage"]["group_overlap_count"], 0)
        self.assertEqual(
            r["leakage"]["protein_cluster_overlap_count"], 1)

    def test_unknown_when_id_missing(self):
        r = cs.summarise_subset(self.s, ["a", "ghost"])
        self.assertEqual(r["per_split"]["unknown"]["samples"], 1)
        self.assertEqual(r["not_in_splits_json"], 1)

    def test_no_cluster_id_recorded(self):
        s = _splits({"x": _meta(None, None, "train"),
                     "y": _meta("prot_1", None, "test")})
        r = cs.summarise_subset(s, ["x", "y"])
        self.assertEqual(r["no_protein_cluster_id"], 1)
        self.assertEqual(r["no_rna_cluster_id"], 2)
        self.assertEqual(r["n_protein_clusters"], 1)
        self.assertEqual(r["n_rna_clusters"], 0)


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        splits = _splits({
            "a": _meta("prot_1", "rna_1", "train"),
            "b": _meta("prot_2", "rna_2", "test"),
        })
        self.sj = self.tmp / "splits.json"
        self.sj.write_text(json.dumps(splits), encoding="utf-8")

        # samples dir mirroring filter_quality input layout
        self.sdir = self.tmp / "samples"
        self.sdir.mkdir()
        for sid in ("a", "b", "c"):  # 'c' not in splits.json
            payload = {"sample_id": sid,
                       "protein": {"sequence": "M" * 100,
                                   "length": 100},
                       "rna": {"sequence": "G" * 50, "length": 50},
                       "interaction":
                           {"binding_protein_residues":
                            list(range(1, 9))},
                       "data_availability":
                           {"resolution": 2.0,
                            "experimental_method": "X-RAY",
                            "quality_tier": "strict"}}
            (self.sdir / f"{sid}.json").write_text(
                json.dumps(payload), encoding="utf-8")

    def test_input_dir_basic(self):
        rc = cs.main([
            "--splits-json", str(self.sj),
            "--input-dir", str(self.sdir),
            "--json", str(self.tmp / "out.json"),
        ])
        self.assertEqual(rc, 0)
        out = json.loads((self.tmp / "out.json")
                         .read_text(encoding="utf-8"))
        # 3 ids scanned; 'c' lands in unknown
        self.assertEqual(out["n_samples"], 3)
        self.assertEqual(out["per_split"]["train"]["samples"], 1)
        self.assertEqual(out["per_split"]["test"]["samples"], 1)
        self.assertEqual(out["per_split"]["unknown"]["samples"], 1)
        self.assertEqual(
            out["leakage"]["group_overlap_count"], 0)

    def test_apply_quality_filter_path(self):
        # Tighten min_gt above the mock (each has 8) -> all dropped
        rc = cs.main([
            "--splits-json", str(self.sj),
            "--input-dir", str(self.sdir),
            "--apply-quality-filter",
            "--min-gt-binding", "100",
        ])
        # all filtered out -> empty subset is an error (exit 1)
        self.assertEqual(rc, 1)

        # Loose enough to keep all 3
        rc = cs.main([
            "--splits-json", str(self.sj),
            "--input-dir", str(self.sdir),
            "--apply-quality-filter",
            "--min-gt-binding", "1",
        ])
        self.assertEqual(rc, 0)

    def test_splits_json_only(self):
        rc = cs.main([
            "--splits-json", str(self.sj),
            "--splits-json-only",
        ])
        self.assertEqual(rc, 0)

    def test_missing_splits_returns_1(self):
        rc = cs.main([
            "--splits-json", str(self.tmp / "nope.json"),
            "--input-dir", str(self.sdir),
        ])
        self.assertEqual(rc, 1)

    def test_no_subset_returns_1(self):
        rc = cs.main(["--splits-json", str(self.sj)])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
