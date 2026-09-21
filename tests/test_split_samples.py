"""Mock tests for scripts/split_samples.py.

Covers:
  - load_splits parses splits.json shape correctly
  - samples with split=None are skipped
  - subsample_train uses the seed (reproducibility)
  - subsample_train when n >= train returns the full set
  - write_list is atomic (.tmp removed) + one id per line
  - main() end-to-end produces 4 files with correct counts
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.split_samples import (  # noqa: E402
    build_sample_index, load_splits, main, subsample_train,
    validate_sample_ids, write_list,
)


def _make_splits_json(tmp: Path, samples: dict[str, str | None]) -> Path:
    """Write a minimal splits.json with the given sample_id → split map."""
    path = tmp / "splits.json"
    data = {
        "config": {"seed": 42},
        "stats": {"n_samples": len(samples)},
        "samples": {sid: {"split": split} for sid, split in samples.items()},
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestLoadSplits(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_basic_parse(self):
        splits_path = _make_splits_json(self.tmp, {
            "a": "train", "b": "train", "c": "val",
            "d": "test", "e": "test", "f": None,
        })
        splits, n_skipped = load_splits(splits_path)
        self.assertEqual(splits["train"], ["a", "b"])
        self.assertEqual(splits["val"], ["c"])
        self.assertEqual(splits["test"], ["d", "e"])
        self.assertEqual(n_skipped, 1)  # 'f' is unclusterable

    def test_ids_sorted_within_split(self):
        # Insert in reverse order; output must be ascending.
        splits_path = _make_splits_json(self.tmp, {
            "z": "train", "a": "train", "m": "train",
        })
        splits, _ = load_splits(splits_path)
        self.assertEqual(splits["train"], ["a", "m", "z"])

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_splits(self.tmp / "does_not_exist.json")

    def test_unknown_split_value_treated_as_skipped(self):
        splits_path = _make_splits_json(self.tmp, {
            "a": "train", "b": "weird_split", "c": "test",
        })
        splits, n_skipped = load_splits(splits_path)
        self.assertEqual(splits["train"], ["a"])
        self.assertEqual(splits["test"], ["c"])
        self.assertEqual(n_skipped, 1)


class TestSubsampleTrain(unittest.TestCase):
    def test_seed_reproducibility(self):
        # Same input + same seed → same output.
        ids = [f"s{i:04d}" for i in range(100)]
        a = subsample_train(ids, n=10, seed=42)
        b = subsample_train(ids, n=10, seed=42)
        self.assertEqual(a, b)

    def test_different_seeds_yield_different_subsets(self):
        ids = [f"s{i:04d}" for i in range(100)]
        a = subsample_train(ids, n=10, seed=42)
        b = subsample_train(ids, n=10, seed=43)
        self.assertNotEqual(a, b)
        # Both still have 10 unique ids each.
        self.assertEqual(len(a), 10)
        self.assertEqual(len(set(a)), 10)

    def test_n_zero_returns_empty(self):
        self.assertEqual(subsample_train(["a", "b"], n=0, seed=42), [])

    def test_n_larger_than_pool_returns_full(self):
        ids = ["a", "b", "c"]
        out = subsample_train(ids, n=100, seed=42)
        self.assertEqual(out, ["a", "b", "c"])

    def test_output_sorted(self):
        ids = [f"s{i:04d}" for i in range(50)]
        out = subsample_train(ids, n=10, seed=42)
        self.assertEqual(out, sorted(out))

    def test_no_duplicates(self):
        ids = [f"s{i:04d}" for i in range(50)]
        out = subsample_train(ids, n=20, seed=42)
        self.assertEqual(len(out), len(set(out)))


class TestSampleIndexAndValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.samples = self.tmp / "samples"
        self.samples.mkdir()
        # On-disk filenames use a mix of cases — mirrors the real
        # data/processed/samples/ where ``3j46_Y_1.json`` coexists with
        # ``3j46_y_1`` in splits.json.
        for name in ("3j46_Y_1", "5kpx_10_26", "1un6_B_F"):
            (self.samples / f"{name}.json").write_text("{}", encoding="utf-8")

    def test_build_index_is_lowercase_keyed(self):
        idx = build_sample_index(self.samples)
        self.assertEqual(idx["3j46_y_1"], "3j46_Y_1")
        self.assertEqual(idx["5kpx_10_26"], "5kpx_10_26")

    def test_validate_canonicalises_case(self):
        idx = build_sample_index(self.samples)
        valid, skipped = validate_sample_ids(
            ["3j46_y_1", "5kpx_10_26"], idx,
        )
        self.assertEqual(valid, ["3j46_Y_1", "5kpx_10_26"])
        self.assertEqual(skipped, 0)

    def test_validate_drops_missing(self):
        idx = build_sample_index(self.samples)
        valid, skipped = validate_sample_ids(
            ["3j46_Y_1", "ghost_id", "1un6_B_F"], idx,
        )
        self.assertEqual(valid, ["3j46_Y_1", "1un6_B_F"])
        self.assertEqual(skipped, 1)

    def test_validate_passes_through_when_index_empty(self):
        # No samples_dir → index empty → pass through unchanged.
        valid, skipped = validate_sample_ids(["a", "b"], {})
        self.assertEqual(valid, ["a", "b"])
        self.assertEqual(skipped, 0)


class TestWriteList(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_one_id_per_line(self):
        path = self.tmp / "list.txt"
        write_list(path, ["a", "b", "c"])
        self.assertEqual(
            path.read_text(encoding="utf-8").splitlines(),
            ["a", "b", "c"],
        )

    def test_atomic_write_no_tmp_left(self):
        path = self.tmp / "list.txt"
        write_list(path, ["a", "b"])
        # The .tmp file must be gone after replace().
        self.assertFalse(
            path.with_suffix(path.suffix + ".tmp").exists(),
            ".tmp file must not be left behind",
        )

    def test_empty_list_writes_empty_file(self):
        path = self.tmp / "list.txt"
        write_list(path, [])
        self.assertEqual(path.read_text(encoding="utf-8"), "")


class TestMainCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.processed = self.tmp / "processed"
        self.processed.mkdir()
        # Synthesise 1000 train / 100 val / 100 test / 5 unclusterable.
        samples = {}
        for i in range(1000):
            samples[f"s_train_{i:04d}"] = "train"
        for i in range(100):
            samples[f"s_val_{i:04d}"] = "val"
        for i in range(100):
            samples[f"s_test_{i:04d}"] = "test"
        for i in range(5):
            samples[f"s_unclu_{i:04d}"] = None
        _make_splits_json(self.processed, samples)
        self.output_dir = self.tmp / "splits_out"

    def test_full_run(self):
        rc = main([
            "--processed-dir", str(self.processed),
            "--output-dir", str(self.output_dir),
            "--train-subset", "50",
            "--seed", "42",
        ])
        self.assertEqual(rc, 0)
        # 4 files written.
        for fname in ("train.txt", "val.txt", "test.txt", "train_50.txt"):
            self.assertTrue(
                (self.output_dir / fname).is_file(),
                f"missing output file: {fname}",
            )
        # Counts.
        train = (self.output_dir / "train.txt").read_text(
            encoding="utf-8").splitlines()
        val = (self.output_dir / "val.txt").read_text(
            encoding="utf-8").splitlines()
        test = (self.output_dir / "test.txt").read_text(
            encoding="utf-8").splitlines()
        sub = (self.output_dir / "train_50.txt").read_text(
            encoding="utf-8").splitlines()
        self.assertEqual(len(train), 1000)
        self.assertEqual(len(val), 100)
        self.assertEqual(len(test), 100)
        self.assertEqual(len(sub), 50)
        # Subsample is a strict subset of train.
        self.assertTrue(set(sub).issubset(set(train)))

    def test_subset_reproducible(self):
        rc = main([
            "--processed-dir", str(self.processed),
            "--output-dir", str(self.output_dir / "run1"),
            "--train-subset", "50", "--seed", "42",
        ])
        self.assertEqual(rc, 0)
        rc = main([
            "--processed-dir", str(self.processed),
            "--output-dir", str(self.output_dir / "run2"),
            "--train-subset", "50", "--seed", "42",
        ])
        self.assertEqual(rc, 0)
        sub1 = (self.output_dir / "run1" / "train_50.txt").read_text(
            encoding="utf-8")
        sub2 = (self.output_dir / "run2" / "train_50.txt").read_text(
            encoding="utf-8")
        self.assertEqual(sub1, sub2,
                         "same seed must yield identical train subset")

    def test_missing_splits_json_returns_1(self):
        rc = main([
            "--processed-dir", str(self.tmp / "no_such_dir"),
            "--output-dir", str(self.output_dir),
        ])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
