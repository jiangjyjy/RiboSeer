"""Unit tests for `schemas.py` — pure pydantic validation."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pydantic import ValidationError  # noqa: E402

from step2_target_char.schemas import (  # noqa: E402
    ProteinDomain, RNAStructure,
    StructureComposition, TargetFeatures, TargetCharOutput,
    canonical_category,
)


def _valid_output_payload(**overrides) -> dict:
    base = {
        "analysis": "This sample looks like a classic RRM bound to a stem-loop.",
        "protein_domain": "RRM",
        "rna_structure": "stem_loop",
        "category": "RRM_x_stem_loop",
        "confidence": 0.85,
        "notes": None,
    }
    base.update(overrides)
    return base


class TestTargetCharOutput(unittest.TestCase):
    def test_valid(self):
        out = TargetCharOutput.model_validate(_valid_output_payload())
        self.assertEqual(out.category, "RRM_x_stem_loop")
        self.assertEqual(out.protein_domain, ProteinDomain.RRM)
        self.assertEqual(out.rna_structure, RNAStructure.stem_loop)

    def test_category_mismatch_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            TargetCharOutput.model_validate(
                _valid_output_payload(category="KH_x_junction"),
            )
        self.assertIn("does not match", str(ctx.exception))

    def test_unknown_protein_domain_rejected(self):
        with self.assertRaises(ValidationError):
            TargetCharOutput.model_validate(
                _valid_output_payload(protein_domain="BogusDomain"),
            )

    def test_unknown_rna_structure_rejected(self):
        with self.assertRaises(ValidationError):
            TargetCharOutput.model_validate(
                _valid_output_payload(rna_structure="triple_helix"),
            )

    def test_confidence_out_of_range(self):
        with self.assertRaises(ValidationError):
            TargetCharOutput.model_validate(_valid_output_payload(confidence=1.5))
        with self.assertRaises(ValidationError):
            TargetCharOutput.model_validate(_valid_output_payload(confidence=-0.1))

    def test_analysis_too_short(self):
        with self.assertRaises(ValidationError):
            TargetCharOutput.model_validate(_valid_output_payload(analysis="short"))

    def test_analysis_whitespace_only(self):
        with self.assertRaises(ValidationError):
            TargetCharOutput.model_validate(
                _valid_output_payload(analysis=" " * 40),
            )

    def test_all_domain_x_structure_combos_accept(self):
        """64 combos (8×8 nope — 8 domain × 6 structure = 48) accept."""
        for d in ProteinDomain:
            for r in RNAStructure:
                payload = _valid_output_payload(
                    protein_domain=d.value,
                    rna_structure=r.value,
                    category=f"{d.value}_x_{r.value}",
                )
                TargetCharOutput.model_validate(payload)  # should not raise

    def test_canonical_category_helper(self):
        self.assertEqual(
            canonical_category(ProteinDomain.RRM, RNAStructure.stem_loop),
            "RRM_x_stem_loop",
        )


class TestTargetFeatures(unittest.TestCase):
    def test_minimal_valid(self):
        f = TargetFeatures(
            sample_id="x_A_B",
            rna_length=50, protein_length=100,
            n_binding_protein_residues=10, n_binding_rna_nucleotides=8,
            interface_ratio_protein=0.10, interface_ratio_rna=0.16,
            quality_tier="strict",
        )
        self.assertIsNone(f.rna_gc_content)
        self.assertIsNone(f.rna_structure_composition)
        self.assertIsNone(f.protein_top3_aa)
        self.assertEqual(f.rna_has_modification, False)

    def test_negative_length_rejected(self):
        with self.assertRaises(ValidationError):
            TargetFeatures(
                sample_id="x",
                rna_length=-1, protein_length=100,
                n_binding_protein_residues=0, n_binding_rna_nucleotides=0,
                interface_ratio_protein=0.0, interface_ratio_rna=0.0,
                quality_tier="strict",
            )

    def test_gc_content_out_of_range_rejected(self):
        with self.assertRaises(ValidationError):
            TargetFeatures(
                sample_id="x",
                rna_length=50, rna_gc_content=1.5,
                protein_length=100,
                n_binding_protein_residues=0, n_binding_rna_nucleotides=0,
                interface_ratio_protein=0.0, interface_ratio_rna=0.0,
                quality_tier="strict",
            )

    def test_structure_composition_all_frac_in_01(self):
        sc = StructureComposition(
            paired_frac=0.5, hairpin_frac=0.2, interior_frac=0.1,
            multiloop_frac=0.1, external_frac=0.1,
        )
        self.assertAlmostEqual(
            sc.paired_frac + sc.hairpin_frac + sc.interior_frac
            + sc.multiloop_frac + sc.external_frac, 1.0, places=3,
        )

    def test_structure_composition_fraction_gt_1_rejected(self):
        with self.assertRaises(ValidationError):
            StructureComposition(
                paired_frac=1.5, hairpin_frac=0, interior_frac=0,
                multiloop_frac=0, external_frac=0,
            )

    def test_top3_aa_accepts_tuples(self):
        f = TargetFeatures(
            sample_id="x",
            rna_length=10, protein_length=20,
            n_binding_protein_residues=0, n_binding_rna_nucleotides=0,
            interface_ratio_protein=0.0, interface_ratio_rna=0.0,
            quality_tier="strict",
            protein_top3_aa=[("K", 0.15), ("R", 0.10), ("E", 0.08)],
        )
        self.assertEqual(f.protein_top3_aa[0][0], "K")


if __name__ == "__main__":
    unittest.main(verbosity=2)
