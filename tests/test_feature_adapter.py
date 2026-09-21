"""Unit tests for `feature_adapter.py` — uses real local samples + synthetic JSON.

Real-sample tests assume `data/processed/samples/` exists on disk (locally it's
the 45468-sample dump). Skipped with a warning if it's not there.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step2_target_char.feature_adapter import (  # noqa: E402
    extract_target_features, sid_to_filename,
    _safe_ratio, _top_n_aa, _build_structure_composition,
)
from step2_target_char.schemas import StructureComposition  # noqa: E402


SAMPLES_DIR = REPO / "data" / "processed" / "samples"


def _load(sid: str) -> dict:
    path = SAMPLES_DIR / (sid_to_filename(sid) + ".json")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class TestSidToFilename(unittest.TestCase):
    """Mirror the step1_server selftest — must stay in sync."""
    def test_lowercase_chain_marked_with_dash(self):
        self.assertEqual(sid_to_filename("3j46_y_1"), "3j46_-y_1")

    def test_uppercase_chain_unchanged(self):
        self.assertEqual(sid_to_filename("3j46_Y_1"), "3j46_Y_1")

    def test_multi_letter_chain(self):
        self.assertEqual(sid_to_filename("8ppl_Aj_A2"), "8ppl_A-j_A2")

    def test_no_underscore_tail(self):
        # No '_' in input → fallback encodes every lowercase char, even the first
        self.assertEqual(sid_to_filename("abcd"), "-a-b-c-d")


class TestSafeRatio(unittest.TestCase):
    def test_normal(self):
        self.assertAlmostEqual(_safe_ratio(3, 12), 0.25)

    def test_zero_denominator(self):
        self.assertEqual(_safe_ratio(5, 0), 0.0)

    def test_zero_numerator(self):
        self.assertEqual(_safe_ratio(0, 10), 0.0)


class TestTopN(unittest.TestCase):
    def test_basic_sort(self):
        comp = {"A": 0.1, "K": 0.3, "R": 0.2, "G": 0.05}
        top3 = _top_n_aa(comp, n=3)
        self.assertEqual([t[0] for t in top3], ["K", "R", "A"])

    def test_tiebreak_lexicographic(self):
        comp = {"B": 0.1, "A": 0.1, "C": 0.05}
        top2 = _top_n_aa(comp, n=2)
        self.assertEqual([t[0] for t in top2], ["A", "B"])

    def test_none_input(self):
        self.assertIsNone(_top_n_aa(None))
        self.assertIsNone(_top_n_aa({}))

    def test_all_zero_input(self):
        self.assertIsNone(_top_n_aa({"A": 0.0, "C": 0.0}))

    def test_roundrobin_fraction(self):
        comp = {"X": 0.12345678}
        top = _top_n_aa(comp, n=1)
        self.assertEqual(top[0][0], "X")
        self.assertAlmostEqual(top[0][1], 0.1235, places=4)


class TestBuildStructureComposition(unittest.TestCase):
    def test_none(self):
        self.assertIsNone(_build_structure_composition(None))

    def test_empty_dict(self):
        self.assertIsNone(_build_structure_composition({}))

    def test_partial_dict(self):
        # Missing one of the 5 required keys → None, not a partial object
        self.assertIsNone(_build_structure_composition({
            "paired_frac": 0.5, "hairpin_frac": 0.1, "interior_frac": 0.1,
            "multiloop_frac": 0.1,
            # external_frac missing
        }))

    def test_complete_dict(self):
        sc = _build_structure_composition({
            "paired_frac": 0.5, "hairpin_frac": 0.1, "interior_frac": 0.1,
            "multiloop_frac": 0.1, "external_frac": 0.2,
        })
        self.assertIsInstance(sc, StructureComposition)
        self.assertEqual(sc.paired_frac, 0.5)

    def test_already_wrapped(self):
        sc = StructureComposition(
            paired_frac=0.4, hairpin_frac=0.2, interior_frac=0.1,
            multiloop_frac=0.1, external_frac=0.2,
        )
        self.assertIs(_build_structure_composition(sc), sc)


class TestExtractSynthetic(unittest.TestCase):
    def _minimal_sample(self, **overrides) -> dict:
        """Build a synthetic sample with controllable nulls."""
        base = {
            "sample_id": "test_A_B",
            "protein": {
                "length": 100, "sequence": "M" * 100,
                "features": {
                    "pI": 9.2, "mean_bfactor": 40.5,
                    "aa_composition": {"K": 0.15, "R": 0.12, "E": 0.08, "M": 0.65},
                },
            },
            "rna": {
                "length": 50, "sequence": "A" * 50,
                "has_modification": False,
                "features": {
                    "gc_content": 0.4,
                    "ss_status": "done",
                    "structure_composition": {
                        "paired_frac": 0.5, "hairpin_frac": 0.2,
                        "interior_frac": 0.1, "multiloop_frac": 0.0,
                        "external_frac": 0.2,
                    },
                },
            },
            "interaction": {
                "binding_protein_residues": list(range(10)),
                "binding_rna_nucleotides": list(range(5)),
            },
            "data_availability": {
                "quality_tier": "strict", "resolution": 2.5,
                "experimental_method": "X-RAY",
            },
        }
        # shallow-merge overrides
        for k, v in overrides.items():
            base[k] = v
        return base

    def test_happy_path(self):
        f = extract_target_features(self._minimal_sample())
        self.assertEqual(f.sample_id, "test_A_B")
        self.assertEqual(f.protein_length, 100)
        self.assertEqual(f.rna_length, 50)
        self.assertEqual(f.n_binding_protein_residues, 10)
        self.assertEqual(f.n_binding_rna_nucleotides, 5)
        self.assertAlmostEqual(f.interface_ratio_protein, 0.10, places=4)
        self.assertAlmostEqual(f.interface_ratio_rna, 0.10, places=4)
        self.assertEqual(f.quality_tier, "strict")
        self.assertEqual(f.experimental_method, "X-RAY")
        self.assertEqual(f.protein_top3_aa[0], ("M", 0.65))  # top by fraction
        self.assertIsNotNone(f.rna_structure_composition)

    def test_null_pI_and_aa_all_X(self):
        """Simulate the 122 all-X protein edge case from stage 1.4."""
        sample = self._minimal_sample()
        sample["protein"]["features"]["pI"] = None
        sample["protein"]["features"]["aa_composition"] = None
        f = extract_target_features(sample)
        self.assertIsNone(f.protein_pI)
        self.assertIsNone(f.protein_top3_aa)

    def test_null_mean_bfactor(self):
        sample = self._minimal_sample()
        sample["protein"]["features"]["mean_bfactor"] = None
        f = extract_target_features(sample)
        self.assertIsNone(f.protein_mean_bfactor)

    def test_null_gc_content(self):
        """Simulate the 29 poly-inosine RNA edge case from stage 1.3a."""
        sample = self._minimal_sample()
        sample["rna"]["features"]["gc_content"] = None
        f = extract_target_features(sample)
        self.assertIsNone(f.rna_gc_content)

    def test_ss_and_structure_missing(self):
        """Local copy: stage 1.3b hasn't filled ss_status / composition yet."""
        sample = self._minimal_sample()
        sample["rna"]["features"]["ss_status"] = None
        sample["rna"]["features"]["structure_composition"] = None
        f = extract_target_features(sample)
        self.assertIsNone(f.rna_ss_status)
        self.assertIsNone(f.rna_structure_composition)

    def test_zero_length_does_not_divide(self):
        sample = self._minimal_sample()
        sample["protein"]["length"] = 0
        sample["protein"]["sequence"] = ""
        f = extract_target_features(sample)
        self.assertEqual(f.interface_ratio_protein, 0.0)

    def test_missing_features_subtree(self):
        """Sample with no `features` dict at all — should still produce an object."""
        sample = self._minimal_sample()
        sample["protein"].pop("features", None)
        sample["rna"].pop("features", None)
        f = extract_target_features(sample)
        self.assertIsNone(f.protein_pI)
        self.assertIsNone(f.rna_gc_content)

    def test_length_fallback_to_sequence(self):
        sample = self._minimal_sample()
        sample["protein"].pop("length")
        sample["protein"]["sequence"] = "MMMMM"  # len=5
        f = extract_target_features(sample)
        self.assertEqual(f.protein_length, 5)


@unittest.skipUnless(
    SAMPLES_DIR.is_dir() and any(SAMPLES_DIR.iterdir()),
    reason=f"requires local samples at {SAMPLES_DIR}",
)
class TestExtractRealSamples(unittest.TestCase):
    def test_strict_sample(self):
        s = _load("1un6_B_F")
        f = extract_target_features(s)
        self.assertEqual(f.sample_id, "1un6_B_F")
        self.assertEqual(f.quality_tier, "strict")
        self.assertEqual(f.protein_length, 87)
        self.assertEqual(f.rna_length, 61)
        self.assertEqual(f.n_binding_protein_residues, 17)
        self.assertEqual(f.n_binding_rna_nucleotides, 14)
        self.assertAlmostEqual(f.rna_gc_content, 0.6557, places=3)
        # Stage 1.3b not run locally — ss_status is None
        self.assertIsNone(f.rna_ss_status)

    def test_all_X_protein_null_pI(self):
        s = _load("3j92_x_5")
        f = extract_target_features(s)
        self.assertIsNone(f.protein_pI)
        self.assertIsNone(f.protein_top3_aa)

    def test_null_bfactor(self):
        s = _load("3j46_5_4")
        f = extract_target_features(s)
        self.assertIsNone(f.protein_mean_bfactor)


if __name__ == "__main__":
    unittest.main(verbosity=2)
