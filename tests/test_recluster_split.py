"""Mock test for scripts/recluster_split.py.

Only the helpers we own (samples-dir → list-of-dicts loader) are
unit-tested here; the heavy lifting (MMseqs2 + clustering helpers)
is delegated to ``src/step1_server/cluster_and_split.py`` and isn't
runnable on Windows (no mmseqs binary).

If you want a server-side smoke test, run::

    python scripts/recluster_split.py \\
        --processed-dir data/processed_filtered \\
        --max-rna-len 200
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.recluster_split import load_samples_from_dir  # noqa: E402


def _write_sample(samples_dir: Path, sid: str, **overrides) -> None:
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample = {
        "sample_id": sid,
        "source_pdb": overrides.get("source_pdb", "1xyz"),
        "protein": {
            "chain_id": overrides.get("protein_chain", "A"),
            "sequence": overrides.get("protein_sequence", "MKTVLAGICK"),
            "length": 10,
        },
        "rna": {
            "chain_id": overrides.get("rna_chain", "B"),
            "sequence": overrides.get("rna_sequence", "GCGCG"),
            "length": 5,
        },
        "data_availability": {"quality_tier": "strict"},
    }
    (samples_dir / f"{sid}.json").write_text(
        json.dumps(sample), encoding="utf-8",
    )


class TestLoadSamplesFromDir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.proc = self.tmp / "processed"

    def test_returns_expected_dict_shape(self):
        _write_sample(self.proc / "samples", "1abc_A_B")
        samples = load_samples_from_dir(self.proc)
        self.assertEqual(len(samples), 1)
        s = samples[0]
        # Same keys cluster_and_split's helpers expect.
        for key in ("sample_id", "pdb_id", "protein_chain", "rna_chain",
                    "protein_sequence", "rna_sequence", "quality_tier"):
            self.assertIn(key, s)
        self.assertEqual(s["pdb_id"], "1xyz")
        self.assertEqual(s["protein_chain"], "A")
        self.assertEqual(s["rna_chain"], "B")
        self.assertEqual(s["quality_tier"], "strict")

    def test_skips_malformed_json(self):
        _write_sample(self.proc / "samples", "good")
        # Bad file with a .json extension — should be skipped, not crash.
        (self.proc / "samples" / "bad.json").write_text(
            "{not json", encoding="utf-8",
        )
        samples = load_samples_from_dir(self.proc)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["sample_id"], "good")

    def test_missing_samples_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_samples_from_dir(self.tmp / "no_such")

    def test_quality_tier_optional(self):
        # data_availability missing → quality_tier becomes "" (not None,
        # so downstream string ops don't crash).
        samples_dir = self.proc / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        sample = {
            "sample_id": "x",
            "source_pdb": "1xyz",
            "protein": {"chain_id": "A", "sequence": "M" * 10, "length": 10},
            "rna": {"chain_id": "B", "sequence": "GCGCG", "length": 5},
            # data_availability omitted intentionally
        }
        (samples_dir / "x.json").write_text(json.dumps(sample), encoding="utf-8")
        samples = load_samples_from_dir(self.proc)
        self.assertEqual(samples[0]["quality_tier"], "")


if __name__ == "__main__":
    unittest.main()
