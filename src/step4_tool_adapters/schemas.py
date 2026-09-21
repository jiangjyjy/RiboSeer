"""Unified output schema for step 4 tool adapters (formula 5 in the paper).

Every adapter must convert its tool's native output into a
``ToolPrediction`` object. Different tool categories populate different
fields — see the table in the step-4 field table below:

    field                       P2Rank(B)  Boltz-2(A)  Chai-1(A)  EquiPNAS(C)  HADDOCK3(D)
    binding_protein_residues    ✅         ✅          ✅         ✅           ✅
    binding_rna_nucleotides     ❌         ✅          ✅         ❌           ✅
    per_residue_confidence      ✅         ✅          ✅         ✅           ❌
    per_residue_pae_score       ❌         ✅          ✅         ❌           ✅
    predicted_structure_path    ❌         ✅          ✅         ❌           ✅
    plddt_mean / iptm / pae     ❌         ✅          ✅         ❌           ❌
    pockets                     ✅         ❌          ❌         ❌           ❌

Failure semantics
-----------------
``success=False`` records are valid: the core prediction fields may all
be ``None`` and the ``error_message`` field carries the reason. Callers
inspect ``success`` first; downstream consumers (step 5/6) skip failed
predictions when fusing.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Pocket(BaseModel):
    """Single pocket entry returned by Cat B tools (P2Rank, Fpocket, ...)."""
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(..., ge=1, description="1-based pocket rank from the tool")
    score: float = Field(..., description="raw score (tool-specific scale)")
    residues: list[int] = Field(
        default_factory=list,
        description="protein residue indices belonging to this pocket",
    )

    @field_validator("residues")
    @classmethod
    def _residues_unique(cls, v: list[int]) -> list[int]:
        if len(v) != len(set(v)):
            raise ValueError("pocket residues must be unique")
        return v


class ToolPrediction(BaseModel):
    """Unified per-tool prediction record.

    A ``ToolPrediction`` is JSONL-serialisable and represents the output
    of one tool on one sample. ``success=False`` records keep the same
    top-level shape but allow every prediction field to be ``None``.
    """
    model_config = ConfigDict(extra="forbid")

    # -- identity ----------------------------------------------------------
    tool_id: str = Field(..., min_length=1)
    category: str = Field(..., pattern=r"^[ABCD]$")
    sample_id: str = Field(..., min_length=1)
    success: bool
    error_message: Optional[str] = None

    # -- core predictions --------------------------------------------------
    binding_protein_residues: Optional[list[int]] = None
    binding_rna_nucleotides: Optional[list[int]] = None
    per_residue_confidence: Optional[dict[int, float]] = Field(
        default=None,
        description="{residue_index: score} — semantics depend on tool "
                    "(P2Rank: ligandability, Cat A: pLDDT, EquiPNAS: prob)",
    )
    per_residue_pae_score: Optional[dict[int, float]] = Field(
        default=None,
        description="Cat A only: {residue_index: 0-1 binding probability}. "
                    "Field name is historical (originally PAE-derived only); "
                    "two sources exist depending on which Cat A tool ran: "
                    "Boltz-2 → 1/(1 + min_inter_chain_pae / pae_scale) from "
                    "the PAE matrix; Chai-1 → 1/(1 + min_CA_RNA_dist / "
                    "distance_scale) from the predicted complex geometry "
                    "(Chai-1's scores.npz lacks a per-token PAE matrix). "
                    "Both produce comparable [0,1] interface-affinity scores; "
                    "evaluate.py treats them identically. Distinct from "
                    "per_residue_confidence (which holds pLDDT, a structure-"
                    "prediction confidence — not interface affinity). Used "
                    "for paper-table per-residue Pearson/Spearman/R²; the "
                    "noisy-OR fusion still consumes per_residue_confidence "
                    "so this field is purely additive.",
    )

    # -- structure info (Cat A only) ---------------------------------------
    predicted_structure_path: Optional[str] = None
    plddt_mean: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    iptm_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    pae_mean: Optional[float] = Field(default=None, ge=0.0)

    # -- pocket info (Cat B only) ------------------------------------------
    pockets: Optional[list[Pocket]] = None

    # -- meta --------------------------------------------------------------
    runtime_seconds: Optional[float] = Field(default=None, ge=0.0)
    raw_output_dir: Optional[str] = None

    # -- validators --------------------------------------------------------

    @field_validator("binding_protein_residues", "binding_rna_nucleotides")
    @classmethod
    def _residues_unique_sorted(cls, v: Optional[list[int]]) -> Optional[list[int]]:
        if v is None:
            return v
        if any(i < 1 for i in v):
            raise ValueError("residue indices must be >= 1 (1-based)")
        if len(v) != len(set(v)):
            raise ValueError("residue indices must be unique")
        return sorted(v)

    @field_validator("per_residue_confidence", "per_residue_pae_score")
    @classmethod
    def _per_residue_keys_positive(
        cls, v: Optional[dict[int, float]],
    ) -> Optional[dict[int, float]]:
        if v is None:
            return v
        if any(k < 1 for k in v):
            raise ValueError(
                "residue indices in per-residue dicts must be >= 1 (1-based)"
            )
        return v

    @model_validator(mode="after")
    def _consistency_checks(self) -> "ToolPrediction":
        # Failure path: error_message recommended, but core fields may be
        # None or empty — no further constraints.
        if not self.success:
            return self

        # Success path: at least one prediction field must be populated.
        # per_residue_pae_score counts as a prediction signal too — a Cat
        # A run that produced no contacts but did emit a PAE-derived
        # score field should still validate as success.
        has_any = any(
            v is not None for v in (
                self.binding_protein_residues,
                self.binding_rna_nucleotides,
                self.per_residue_confidence,
                self.per_residue_pae_score,
                self.predicted_structure_path,
                self.pockets,
            )
        )
        if not has_any:
            raise ValueError(
                "success=True but no prediction fields populated; "
                "either set success=False or provide a prediction"
            )
        return self


class ToolPredictionSet(BaseModel):
    """Aggregate of all tool predictions for one sample."""
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(..., min_length=1)
    tools_run: list[str] = Field(default_factory=list)
    predictions: list[ToolPrediction] = Field(default_factory=list)
    total_runtime_seconds: Optional[float] = Field(default=None, ge=0.0)
    timestamp: Optional[str] = None

    @model_validator(mode="after")
    def _check_sample_id_alignment(self) -> "ToolPredictionSet":
        for p in self.predictions:
            if p.sample_id != self.sample_id:
                raise ValueError(
                    f"prediction.sample_id={p.sample_id!r} does not match "
                    f"set.sample_id={self.sample_id!r}"
                )
        # tools_run should match predictions order (best-effort, not strict)
        pred_ids = [p.tool_id for p in self.predictions]
        if self.tools_run and pred_ids and set(pred_ids) != set(self.tools_run):
            raise ValueError(
                f"tools_run {self.tools_run} does not cover predictions "
                f"{pred_ids}"
            )
        return self


def make_failure_prediction(
    tool_id: str,
    category: str,
    sample_id: str,
    error_message: str,
    *,
    runtime_seconds: Optional[float] = None,
    raw_output_dir: Optional[str] = None,
) -> ToolPrediction:
    """Convenience constructor for a ``success=False`` record.

    Used by ``BaseAdapter.predict`` and unit tests so all failure
    records are shaped consistently.
    """
    return ToolPrediction(
        tool_id=tool_id,
        category=category,
        sample_id=sample_id,
        success=False,
        error_message=error_message[:2000] if error_message else None,
        binding_protein_residues=None,
        binding_rna_nucleotides=None,
        per_residue_confidence=None,
        per_residue_pae_score=None,
        predicted_structure_path=None,
        plddt_mean=None,
        iptm_score=None,
        pae_mean=None,
        pockets=None,
        runtime_seconds=runtime_seconds,
        raw_output_dir=raw_output_dir,
    )
