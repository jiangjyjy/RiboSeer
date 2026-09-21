"""Mock tests for scripts/filter_dataset.py.

Coverage:
  - filter_dataset copies in-range samples and skips out-of-range
  - missing rna.length is bucketed separately (no_rna_length)
  - main() end-to-end: file counts on disk, exit code, missing
    samples dir → exit 1
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.filter_dataset import filter_dataset, main  # noqa: E402


def _write_sample(samples_dir: Path, sid: str, *, rna_length) -> None:
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample = {
        "sample_id": sid,
        "protein": {"length": 100, "sequence": "M" * 100},
        "rna": {"length": rna_length, "sequence": "G" * (rna_length or 0)},
    }
    if rna_length is None:
        # Drop the field entirely to exercise the "no_rna_length" branch.
        sample["rna"].pop("length")
    (samples_dir / f"{sid}.json").write_text(
        json.dumps(sample), encoding="utf-8",
    )


class TestFilterDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.proc = self.tmp / "processed"
        self.out = self.tmp / "filtered"
        samples = self.proc / "samples"
        # 3 in-range, 2 out-of-range, 1 missing length
        _write_sample(samples, "kept_a", rna_length=50)
        _write_sample(samples, "kept_b", rna_length=200)   # exactly at limit
        _write_sample(samples, "kept_c", rna_length=10)
        _write_sample(samples, "drop_a", rna_length=201)
        _write_sample(samples, "drop_b", rna_length=3000)
        _write_sample(samples, "no_len", rna_length=None)

    def test_basic_counts(self):
        counts = filter_dataset(self.proc, max_rna_length=200, output_dir=self.out)
        self.assertEqual(counts["total"], 6)
        self.assertEqual(counts["kept"], 3)
        self.assertEqual(counts["filtered"], 2)
        self.assertEqual(counts["no_rna_length"], 1)

    def test_files_on_disk_match_counts(self):
        filter_dataset(self.proc, max_rna_length=200, output_dir=self.out)
        out_samples = self.out / "samples"
        names = {f.stem for f in out_samples.glob("*.json")}
        self.assertEqual(names, {"kept_a", "kept_b", "kept_c"})

    def test_missing_samples_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            filter_dataset(
                self.tmp / "no_such_processed",
                max_rna_length=200,
                output_dir=self.out,
            )

    def test_idempotent_rerun(self):
        counts1 = filter_dataset(self.proc, max_rna_length=200, output_dir=self.out)
        counts2 = filter_dataset(self.proc, max_rna_length=200, output_dir=self.out)
        self.assertEqual(counts1, counts2)


class TestMainCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.proc = self.tmp / "processed"
        self.out = self.tmp / "filtered"
        samples = self.proc / "samples"
        _write_sample(samples, "k1", rna_length=50)
        _write_sample(samples, "d1", rna_length=500)

    def test_full_run(self):
        rc = main([
            "--processed-dir", str(self.proc),
            "--max-rna-length", "200",
            "--output-dir", str(self.out),
        ])
        self.assertEqual(rc, 0)
        kept = sorted(p.stem for p in (self.out / "samples").glob("*.json"))
        self.assertEqual(kept, ["k1"])

    def test_missing_processed_dir_returns_1(self):
        rc = main([
            "--processed-dir", str(self.tmp / "nope"),
            "--max-rna-length", "200",
            "--output-dir", str(self.out),
        ])
        self.assertEqual(rc, 1)

    def test_negative_max_length_returns_1(self):
        rc = main([
            "--processed-dir", str(self.proc),
            "--max-rna-length", "0",
            "--output-dir", str(self.out),
        ])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
