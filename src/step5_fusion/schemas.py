"""Pydantic schemas for Step 5 — fusion.

Two sides:
  - ``ToolWeightAssignment``: structured LLM response carrying per-tool
    weights c_k, the threshold τ, the rationale, and overall confidence.
    Validators clamp every weight + threshold into [0, 1] and reject
    whitespace-only rationales (consistent with step 2/3 patterns).
  - ``CompositeResult``: the final fused result for one sample. Contains
    the LLM-assigned weights, the noisy-OR fused residue / nucleotide
    sets, the per-residue probability dict, and book-keeping (rationale,
    confidence, tools_fused, api_usage, timestamp).

Notes on conventions
--------------------
- Residue / nucleotide indices are 1-based to stay consistent with
  ``step4_tool_adapters.schemas.ToolPrediction``.
- ``per_residue_probability`` keys are int (the residue index). When
  serialised to JSON, callers convert to str (``{str(k): v}``) so the
  output is round-trip safe — tests assert both sides.
- ``per_nucleotide_probability`` is included for symmetry with the RNA
  side; it is optional because the spec only mandates the protein-side
  per-residue probability. It is None when no RNA-side prediction was
  made by any tool.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolWeightAssignment(BaseModel):
    """Parsed LLM response — the weights it wants applied during fusion.

    The LLM returns one ``c_k ∈ [0, 1]`` per tool that was passed in, plus a
    fusion threshold τ (default 0.5, but the model is free to nudge it).
    ``rationale`` is a 2-6 sentence explanation of how it weighed the
    tools; ``confidence`` is the model's overall self-assessment of the
    fusion quality.
    """
    model_config = ConfigDict(extra="forbid")

    weights: dict[str, float] = Field(
        ...,
        description="{tool_id: c_k}, every c_k ∈ [0, 1]",
    )
    threshold: float = Field(
        0.5, ge=0.0, le=1.0,
        description="τ for B_p_hat = {i : b_hat(i) > τ}",
    )
    rationale: str = Field(..., min_length=10, max_length=4000)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("weights")
    @classmethod
    def _weights_in_unit_range(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("weights must not be empty")
        for tool_id, w in v.items():
            if not isinstance(tool_id, str) or not tool_id:
                raise ValueError(f"invalid tool_id key: {tool_id!r}")
            if not (0.0 <= float(w) <= 1.0):
                raise ValueError(
                    f"weight for tool '{tool_id}' = {w} is outside [0, 1]"
                )
        return {k: float(v) for k, v in v.items()}

    @field_validator("rationale")
    @classmethod
    def _rationale_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("rationale must not be whitespace-only")
        return v


class CompositeResult(BaseModel):
    """Final fused output for one sample (end state of Section 3.6, Phase 1).

    Persisted as one JSONL record per sample. Keys are stringified by the
    caller before json.dump to avoid the int-key serialisation gotcha.
    """
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(..., min_length=1)

    # LLM-assigned (or fallback) per-tool weights c_k
    tool_weights: dict[str, float]

    # Fused predictions
    binding_protein_residues: list[int] = Field(default_factory=list)
    binding_rna_nucleotides: list[int] = Field(default_factory=list)
    per_residue_probability: dict[int, float] = Field(default_factory=dict)
    per_nucleotide_probability: Optional[dict[int, float]] = None
    threshold: float = Field(0.5, ge=0.0, le=1.0)

    # Reasoning
    fusion_rationale: str = Field(default="", max_length=4000)
    confidence: float = Field(0.0, ge=0.0, le=1.0)

    # Bookkeeping
    tools_fused: list[str] = Field(default_factory=list)
    api_usage: dict = Field(default_factory=dict)
    timestamp: Optional[str] = None

    @field_validator("tool_weights")
    @classmethod
    def _tool_weights_in_unit_range(cls, v: dict[str, float]) -> dict[str, float]:
        for tool_id, w in v.items():
            if not (0.0 <= float(w) <= 1.0):
                raise ValueError(
                    f"tool_weight for '{tool_id}' = {w} is outside [0, 1]"
                )
        return {k: float(v) for k, v in v.items()}

    @field_validator("binding_protein_residues", "binding_rna_nucleotides")
    @classmethod
    def _residues_unique_sorted(cls, v: list[int]) -> list[int]:
        if any(i < 1 for i in v):
            raise ValueError("residue / nucleotide indices must be >= 1 (1-based)")
        if len(v) != len(set(v)):
            raise ValueError("residue / nucleotide indices must be unique")
        return sorted(v)

    @field_validator("per_residue_probability")
    @classmethod
    def _prob_keys_positive(cls, v: dict[int, float]) -> dict[int, float]:
        for i, p in v.items():
            if i < 1:
                raise ValueError(f"residue index {i} must be >= 1")
            if not (0.0 <= float(p) <= 1.0):
                raise ValueError(f"probability {p} for residue {i} outside [0, 1]")
        return {int(i): float(p) for i, p in v.items()}

    @field_validator("per_nucleotide_probability")
    @classmethod
    def _nuc_prob_keys_positive(
        cls, v: Optional[dict[int, float]],
    ) -> Optional[dict[int, float]]:
        if v is None:
            return None
        for i, p in v.items():
            if i < 1:
                raise ValueError(f"nucleotide index {i} must be >= 1")
            if not (0.0 <= float(p) <= 1.0):
                raise ValueError(f"probability {p} for nucleotide {i} outside [0, 1]")
        return {int(i): float(p) for i, p in v.items()}
