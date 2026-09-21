"""Pydantic schemas for step 7 — iterative decision loop.

Three nested models, ordered from inner to outer:

  - ``IterationAction``: the LLM's per-iteration decision (formula 24 in the paper).
    One of ``accept`` / ``refine`` / ``restart``. Action-specific fields
    (``refine_tool``, ``restart_reason``) are required only when the
    matching action is chosen — validators enforce that.

  - ``IterationRecord``: book-keeping for one loop iteration. Wraps the
    ``IterationAction`` with the score before/after the iteration, the
    score delta, the API usage stats, and a timestamp. Used both at
    runtime (the loop builds these) and on disk (one per iteration in
    the final ``IterationResult``).

  - ``IterationResult``: the full per-sample outcome (formula 23 in the paper).
    Persisted as one JSONL line per sample under ``data/step7_outputs/``.
    Carries the trajectory (``score_trajectory``, ``iterations``), the
    termination reason, and the final binding sets (which may differ
    from the initial step-5 ones if a refine/restart succeeded).

Conventions
-----------
- Scores live in [0, 1] (matches step 6's ``PocketQAResult.total_score``).
- Action enum values are lowercased strings to match what the LLM is
  asked to emit; ``ACTION_VALUES`` is the single source of truth.
- ``final_action`` is constrained to ``"accept"`` because the loop only
  exits via accept (whether LLM-chosen, converged, or capped).
- ``termination_reason`` is constrained to a small enum so step 8 can
  reason about *why* the loop stopped without parsing free-form text.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ACTION_VALUES: tuple[str, ...] = ("accept", "refine", "restart")
TERMINATION_VALUES: tuple[str, ...] = ("accepted", "converged", "max_iterations")


class IterationAction(BaseModel):
    """One LLM decision — what to do next given the current PocketQA score.

    Field semantics
    ---------------
    - ``action == "accept"``  → loop ends; ``refine_tool`` / ``restart_reason``
      must both be ``None``.
    - ``action == "refine"``  → re-run a specific tool (lightweight mode:
      drop the worst tool and re-fuse). ``refine_tool`` is required and
      names the tool to re-run / drop. ``refine_reason`` carries the
      LLM's justification.
    - ``action == "restart"`` → re-pick tools or re-weight from scratch.
      ``restart_reason`` is required.
    """
    model_config = ConfigDict(extra="forbid")

    action: str = Field(..., description="one of 'accept' / 'refine' / 'restart'")

    refine_tool: Optional[str] = Field(
        default=None, max_length=64,
        description="tool_id to re-run (refine action only)",
    )
    refine_reason: Optional[str] = Field(
        default=None, max_length=2000,
        description="why this tool needs re-running",
    )
    restart_reason: Optional[str] = Field(
        default=None, max_length=2000,
        description="why a fresh tool selection is warranted",
    )

    rationale: str = Field(..., min_length=10, max_length=4000)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("action")
    @classmethod
    def _action_known(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in ACTION_VALUES:
            raise ValueError(
                f"action {v!r} not in {ACTION_VALUES}"
            )
        return v

    @field_validator("refine_tool", "refine_reason", "restart_reason")
    @classmethod
    def _empty_string_to_none(cls, v: Optional[str]) -> Optional[str]:
        # Treat whitespace-only strings as missing — keeps validator
        # branches simple downstream.
        if v is None:
            return None
        v2 = v.strip()
        return v2 or None

    @field_validator("rationale")
    @classmethod
    def _rationale_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("rationale must not be whitespace-only")
        return v

    @model_validator(mode="after")
    def _per_action_required_fields(self) -> "IterationAction":
        if self.action == "refine":
            if not self.refine_tool:
                raise ValueError("refine action requires refine_tool")
            if self.restart_reason is not None:
                raise ValueError(
                    "refine action must not set restart_reason"
                )
        elif self.action == "restart":
            if not self.restart_reason:
                raise ValueError("restart action requires restart_reason")
            if self.refine_tool is not None or self.refine_reason is not None:
                raise ValueError(
                    "restart action must not set refine_tool / refine_reason"
                )
        else:  # accept
            if (self.refine_tool is not None
                    or self.refine_reason is not None
                    or self.restart_reason is not None):
                raise ValueError(
                    "accept action must not set refine/restart fields"
                )
        return self


class IterationRecord(BaseModel):
    """One loop iteration: the action plus the score change it produced.

    The loop fills these in order; ``score_after`` and ``delta`` are
    ``None`` only when the iteration crashed before re-scoring (and we
    fell back to accept). On a normal accept, ``score_after`` equals
    ``score_before`` and ``delta`` is 0.0.
    """
    model_config = ConfigDict(extra="forbid")

    iteration: int = Field(..., ge=0)
    action: IterationAction
    score_before: float = Field(..., ge=0.0, le=1.0)
    score_after: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    delta: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    api_usage: dict = Field(default_factory=dict)
    timestamp: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _delta_consistent_with_scores(self) -> "IterationRecord":
        # If both scores are known, delta must equal their difference
        # (within float tolerance). Caught early so a buggy callsite
        # doesn't poison the trajectory.
        if self.score_after is not None and self.delta is not None:
            expected = self.score_after - self.score_before
            if abs(expected - self.delta) > 1e-6:
                raise ValueError(
                    f"delta={self.delta!r} does not equal "
                    f"score_after - score_before = {expected!r}"
                )
        return self


class IterationResult(BaseModel):
    """Full per-sample loop outcome (one JSONL line under data/step7_outputs/)."""
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(..., min_length=1)

    # Loop summary
    final_action: str = Field(..., description="always 'accept' (only exit path)")
    total_iterations: int = Field(..., ge=1)
    final_score: float = Field(..., ge=0.0, le=1.0)
    score_trajectory: list[float] = Field(default_factory=list)
    iterations: list[IterationRecord] = Field(default_factory=list)
    termination_reason: str = Field(...)

    # Final predictions (post-refine if any). 1-based indices, sorted,
    # unique — same convention as step 5's CompositeResult.
    final_binding_protein_residues: list[int] = Field(default_factory=list)
    final_binding_rna_nucleotides: list[int] = Field(default_factory=list)

    timestamp: str = Field(..., min_length=1)

    @field_validator("final_action")
    @classmethod
    def _final_action_is_accept(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v != "accept":
            raise ValueError(
                f"final_action must be 'accept', got {v!r}"
            )
        return v

    @field_validator("termination_reason")
    @classmethod
    def _termination_known(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in TERMINATION_VALUES:
            raise ValueError(
                f"termination_reason {v!r} not in {TERMINATION_VALUES}"
            )
        return v

    @field_validator("score_trajectory")
    @classmethod
    def _trajectory_in_unit_interval(cls, v: list[float]) -> list[float]:
        for s in v:
            if not (0.0 <= float(s) <= 1.0):
                raise ValueError(
                    f"score in trajectory must be in [0, 1], got {s!r}"
                )
        return [float(s) for s in v]

    @field_validator(
        "final_binding_protein_residues", "final_binding_rna_nucleotides",
    )
    @classmethod
    def _residues_unique_sorted(cls, v: list[int]) -> list[int]:
        if any(i < 1 for i in v):
            raise ValueError(
                "residue / nucleotide indices must be >= 1 (1-based)"
            )
        if len(v) != len(set(v)):
            raise ValueError("residue / nucleotide indices must be unique")
        return sorted(int(i) for i in v)

    @model_validator(mode="after")
    def _cross_field_consistency(self) -> "IterationResult":
        # Trajectory length must equal iteration count.
        if len(self.score_trajectory) != self.total_iterations:
            raise ValueError(
                f"score_trajectory length {len(self.score_trajectory)} "
                f"!= total_iterations {self.total_iterations}"
            )
        if len(self.iterations) != self.total_iterations:
            raise ValueError(
                f"iterations list length {len(self.iterations)} "
                f"!= total_iterations {self.total_iterations}"
            )
        # iteration indices must be 0..N-1 in order.
        for expected, rec in enumerate(self.iterations):
            if rec.iteration != expected:
                raise ValueError(
                    f"iterations[{expected}].iteration = {rec.iteration}, "
                    f"expected {expected} (must be sequential from 0)"
                )
        # Last iteration's action must be accept (loop only exits there).
        if self.iterations and self.iterations[-1].action.action != "accept":
            raise ValueError(
                "last iteration's action must be 'accept' "
                f"(got {self.iterations[-1].action.action!r})"
            )
        # final_score must equal the last trajectory entry.
        if self.score_trajectory:
            last = self.score_trajectory[-1]
            if abs(last - self.final_score) > 1e-6:
                raise ValueError(
                    f"final_score={self.final_score!r} does not match "
                    f"last trajectory entry {last!r}"
                )
        return self
