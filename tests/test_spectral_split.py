"""Mock tests for scripts/spectral_split.py.

Contract (after the n_clusters=2 + rep-only-output rewrite):
  - n_clusters=2 -> train.txt, test.txt, NO val.txt
  - n_clusters=3 -> train.txt, val.txt, test.txt
  - Output files contain ONE rep per line (not pool samples)
  - Largest spectral group by rep count wins 'train'
  - Stale val.txt from a previous n=3 run is removed when n=2 runs
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.spectral_split as ss  # noqa: E402


class TestAffinity(unittest.TestCase):
    def test_threshold_and_nan_handling(self):
        tm = np.array([[1.0, 0.7, 0.3, np.nan],
                       [0.7, 1.0, 0.4, 0.6],
                       [0.3, 0.4, 1.0, 0.55],
                       [np.nan, 0.6, 0.55, 1.0]])
        a = ss.build_affinity(tm, 0.5)
        self.assertEqual(a[0, 3], 0.0)           # NaN -> 0
        self.assertEqual(a[0, 2], 0.0)           # below threshold
        self.assertEqual(a[1, 2], 0.0)
        self.assertAlmostEqual(a[1, 3], 0.6)     # >= threshold kept
        self.assertAlmostEqual(a[2, 3], 0.55)
        self.assertTrue(np.allclose(a, a.T))
        self.assertTrue(np.allclose(np.diag(a), 1.0))


class TestAssignSplits(unittest.TestCase):
    def test_n_clusters_2_largest_to_train(self):
        # label 1 has 5 reps, label 0 has 2 reps -> 1=train, 0=test
        labels = np.array([0, 0, 1, 1, 1, 1, 1])
        m = ss.assign_splits(labels, n_clusters=2)
        self.assertEqual(m[1], "train")
        self.assertEqual(m[0], "test")
        self.assertNotIn("val", m.values())

    def test_n_clusters_3_train_val_test(self):
        # sizes: label 1 = 4, label 2 = 3, label 0 = 1
        labels = np.array([0, 1, 1, 1, 1, 2, 2, 2])
        m = ss.assign_splits(labels, n_clusters=3)
        self.assertEqual(m[1], "train")
        self.assertEqual(m[2], "val")
        self.assertEqual(m[0], "test")

    def test_n_clusters_2_degenerate_single_label(self):
        # SpectralClustering occasionally collapses to one cluster on
        # extreme affinity matrices; we still produce a valid mapping
        # so the rest of main() doesn't crash.
        labels = np.array([0, 0, 0])
        m = ss.assign_splits(labels, n_clusters=2)
        self.assertEqual(m[0], "train")
        # the (missing) second label simply isn't in the mapping
        self.assertEqual(set(m.values()), {"train"})


class TestBucketReps(unittest.TestCase):
    def test_groups_by_split(self):
        labels = np.array([0, 1, 0, 1, 0])
        rep_sids = ["a", "b", "c", "d", "e"]
        m = {0: "train", 1: "test"}
        out = ss.bucket_reps(rep_sids, labels, m)
        self.assertEqual(sorted(out["train"]), ["a", "c", "e"])
        self.assertEqual(sorted(out["test"]), ["b", "d"])
        self.assertNotIn("val", out)


class TestGroupTmStats(unittest.TestCase):
    def test_intra_inter_means(self):
        tm = np.array([[1.0, 0.8, 0.2],
                       [0.8, 1.0, 0.3],
                       [0.2, 0.3, 1.0]])
        labels = np.array([0, 0, 1])
        m = ss.group_tm_stats(tm, labels,
                              {0: "train", 1: "test"})
        self.assertAlmostEqual(m["intra"]["train"]["mean"], 0.8)
        self.assertEqual(m["intra"]["test"]["n_pairs"], 0)
        key = "__".join(sorted(["train", "test"]))
        self.assertAlmostEqual(m["inter"][key]["mean"], 0.25)


class TestSampleSubset(unittest.TestCase):
    def test_seed_reproducible(self):
        ids = [f"s{i:03d}" for i in range(40)]
        self.assertEqual(ss.sample_subset(ids, 5, 42),
                         ss.sample_subset(ids, 5, 42))
        self.assertEqual(len(ss.sample_subset(ids, 5, 42)), 5)
        self.assertEqual(ss.sample_subset(["b", "a"], 9, 1),
                         ["a", "b"])


class TestRepToClusterParser(unittest.TestCase):
    def test_two_col_and_one_col_tolerated(self):
        tmp = Path(tempfile.mkdtemp()) / "reps.txt"
        tmp.write_text("a\tp1\nb\tp2\nc\n", encoding="utf-8")
        out = ss._rep_to_cluster_from_list(tmp)
        self.assertEqual(out, {"a": "p1", "b": "p2", "c": ""})


def _block_matrix() -> np.ndarray:
    """5x5 with two obvious clusters: {0,1,2} (>=0.85 internal) and
    {3,4} (>=0.85 internal); cross < threshold."""
    return np.array([
        [1.00, 0.92, 0.88, 0.10, 0.05],
        [0.92, 1.00, 0.90, 0.12, 0.08],
        [0.88, 0.90, 1.00, 0.07, 0.11],
        [0.10, 0.12, 0.07, 1.00, 0.85],
        [0.05, 0.08, 0.11, 0.85, 1.00],
    ], dtype=np.float32)


class TestCliEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        np.save(self.tmp / "tm.npy", _block_matrix())
        (self.tmp / "tm.ids.txt").write_text(
            "r0\nr1\nr2\nr3\nr4\n", encoding="utf-8")
        self.reps = self.tmp / "reps.txt"
        self.reps.write_text(
            "r0\tp0\nr1\tp1\nr2\tp2\nr3\tp3\nr4\tp4\n",
            encoding="utf-8")
        self.out = self.tmp / "out"

    def _run(self, *extra):
        return ss.main([
            "--tmscore-matrix", str(self.tmp / "tm.npy"),
            "--sample-list", str(self.reps),
            "--output-dir", str(self.out),
            "--similarity-threshold", "0.5",
            "--train-sample-n", "10",
            "--test-sample-n", "10",
            *extra,
        ])

    def test_n_clusters_2_writes_only_train_and_test(self):
        rc = self._run("--n-clusters", "2")
        self.assertEqual(rc, 0)
        for name in ("train.txt", "test.txt", "train_200.txt",
                     "test_200.txt", "split_stats.json"):
            self.assertTrue((self.out / name).is_file(), name)
        # val.txt MUST NOT exist
        self.assertFalse((self.out / "val.txt").exists(),
                         "val.txt should not be written when n=2")

    def test_n_clusters_2_outputs_only_reps(self):
        self._run("--n-clusters", "2")
        train = (self.out / "train.txt").read_text("utf-8").split()
        test = (self.out / "test.txt").read_text("utf-8").split()
        # Every output id must be one of the 5 rep sids - never a
        # pool sample. Total ids across train+test must equal N (=5).
        rep_set = {"r0", "r1", "r2", "r3", "r4"}
        self.assertTrue(set(train).issubset(rep_set), train)
        self.assertTrue(set(test).issubset(rep_set), test)
        self.assertEqual(len(train) + len(test), 5)
        # Largest spectral group (the {0,1,2} block) lands in train
        self.assertGreaterEqual(len(train), len(test))

    def test_n_clusters_2_stats_no_val(self):
        self._run("--n-clusters", "2")
        stats = json.loads(
            (self.out / "split_stats.json").read_text("utf-8"))
        self.assertEqual(stats["config"]["n_clusters"], 2)
        self.assertIn("train", stats["per_split"])
        self.assertIn("test", stats["per_split"])
        self.assertNotIn("val", stats["per_split"])
        # rep counts add up to N
        n = sum(s["reps"] for s in stats["per_split"].values())
        self.assertEqual(n, 5)

    def test_n_clusters_3_writes_val_txt(self):
        rc = self._run("--n-clusters", "3")
        self.assertEqual(rc, 0)
        self.assertTrue((self.out / "val.txt").is_file())
        stats = json.loads(
            (self.out / "split_stats.json").read_text("utf-8"))
        self.assertIn("val", stats["per_split"])

    def test_stale_val_txt_removed_when_n2_after_n3(self):
        # First run produces val.txt
        self.assertEqual(self._run("--n-clusters", "3"), 0)
        self.assertTrue((self.out / "val.txt").is_file())
        # Now re-run with n=2; the stale val.txt must be wiped
        self.assertEqual(self._run("--n-clusters", "2"), 0)
        self.assertFalse(
            (self.out / "val.txt").exists(),
            "stale val.txt from n=3 should have been removed")

    def test_processed_dir_and_splits_json_are_optional(self):
        """Backward-compat: old commands still parse, args are no-ops."""
        rc = self._run(
            "--n-clusters", "2",
            "--processed-dir", str(self.tmp),
            "--splits-json", str(self.tmp / "nope.json"),
        )
        self.assertEqual(rc, 0)
        self.assertTrue((self.out / "train.txt").is_file())


if __name__ == "__main__":
    unittest.main()
