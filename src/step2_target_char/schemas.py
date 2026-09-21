"""Pydantic schemas for Step 2 — target characterization.

Two sides:
  - `TargetFeatures`: compact feature summary we feed to the prompt. Keeps only
    the fields the LLM actually looks at, with NULLs preserved so the prompt
    can say "unknown" explicitly.
  - `TargetCharOutput`: structured result we parse back from the LLM. Has
    validators for enum values, category consistency, and confidence range.

Pocket category lives on two axes (Section 3.4 of the paper):
  - protein_domain ∈ {RRM, KH, zinc_finger, dsRBD, PUF, DEAD_box,
                      multi_domain, novel_fold}
  - rna_structure  ∈ {single_stranded, stem_loop, internal_loop, junction,
                      g_quadruplex, unstructured}
  - category format: "{protein_domain}_x_{rna_structure}"
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------- enums -----------------------------------------------------------


class ProteinDomain(str, Enum):
    RRM = "RRM"
    KH = "KH"
    zinc_finger = "zinc_finger"
    dsRBD = "dsRBD"
    PUF = "PUF"
    DEAD_box = "DEAD_box"
    multi_domain = "multi_domain"
    novel_fold = "novel_fold"


class RNAStructure(str, Enum):
    single_stranded = "single_stranded"
    stem_loop = "stem_loop"
    internal_loop = "internal_loop"
    junction = "junction"
    g_quadruplex = "g_quadruplex"
    unstructured = "unstructured"


# ---------- input side ------------------------------------------------------


class StructureComposition(BaseModel):
    """Secondary-structure fraction per bucket (each sums to ~1.0, rounding drift allowed)."""
    paired_frac: float = Field(ge=0, le=1)
    hairpin_frac: float = Field(ge=0, le=1)
    interior_frac: float = Field(ge=0, le=1)
    multiloop_frac: float = Field(ge=0, le=1)
    external_frac: float = Field(ge=0, le=1)


class TargetFeatures(BaseModel):
    """Feature summary fed to the LLM prompt.

    Numeric fields hold `None` when upstream couldn't compute them (e.g. pI
    for all-X proteins, structure_composition before stage 1.3b finishes on
    the server). The prompt renders None as "unknown" so the LLM sees
    missingness explicitly instead of being handed a fake zero.
    """
    sample_id: str

    # RNA
    rna_length: int = Field(ge=0)
    rna_gc_content: Optional[float] = Field(default=None, ge=0, le=1)
    rna_ss_status: Optional[str] = None
    rna_structure_composition: Optional[StructureComposition] = None
    rna_has_modification: bool = False

    # Protein
    protein_length: int = Field(ge=0)
    protein_pI: Optional[float] = None
    protein_mean_bfactor: Optional[float] = None
    # [(aa, fraction), ...] sorted desc — easier to render in prompt than dict
    protein_top3_aa: Optional[list[tuple[str, float]]] = None

    # Interaction
    n_binding_protein_residues: int = Field(ge=0)
    n_binding_rna_nucleotides: int = Field(ge=0)
    interface_ratio_protein: float = Field(ge=0)
    interface_ratio_rna: float = Field(ge=0)

    # Data provenance
    quality_tier: str
    resolution: Optional[float] = None
    experimental_method: Optional[str] = None


# ---------- output side -----------------------------------------------------


class TargetCharOutput(BaseModel):
    """Parsed LLM output. Validation guards against:

      - unknown protein_domain / rna_structure (via Enum)
      - confidence outside [0, 1]
      - `category` not matching the pair (caught by `_category_consistency`)
      - empty / suspiciously short `analysis` (min_length=20 — anything
        shorter is almost certainly a truncated / malformed response that
        should trigger an error-correction retry)
    """
    analysis: str = Field(min_length=20, max_length=4000)
    protein_domain: ProteinDomain
    rna_structure: RNAStructure
    category: str
    confidence: float = Field(ge=0, le=1)
    notes: Optional[str] = None

    @field_validator("analysis")
    @classmethod
    def _analysis_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("analysis must not be whitespace-only")
        return v

    @model_validator(mode="after")
    def _category_consistency(self) -> "TargetCharOutput":
        expected = f"{self.protein_domain.value}_x_{self.rna_structure.value}"
        if self.category != expected:
            raise ValueError(
                f"category '{self.category}' does not match "
                f"'{expected}' (protein_domain={self.protein_domain.value}, "
                f"rna_structure={self.rna_structure.value})"
            )
        return self


def canonical_category(domain: ProteinDomain, rna: RNAStructure) -> str:
    """Single source of truth for the "{domain}_x_{rna}" label format."""
    return f"{domain.value}_x_{rna.value}"
