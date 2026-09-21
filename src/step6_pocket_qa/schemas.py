"""Pydantic schemas for step 6 — pocket QA scoring.

Two models:

  - ``MetricDetail`` — one entry per sub-score (q_1 ... q_5). Holds the
    score value (or None when the metric couldn't be computed), a
    ``computed`` flag, an optional error message captured by the
    scorer's try/except wrapper, and a free-form ``info`` dict that
    each metric uses to expose intermediate values for downstream LLM
    analysis (step 7).

  - ``PocketQAResult`` — the final per-sample record (Section 3.7 of the paper).
    Five top-level sub-scores mirror the canonical metric names; each
    is ``Optional[float]`` so a missing-data case (e.g. no predicted
    structure → no compactness) is faithfully reported. ``total_score``
    is the *renormalised* weighted average over the metrics that
    actually computed (formula 17 in the paper, with implicit re-weighting when a
    q_m drops out — missing data should not penalise the aggregate).

Conventions
-----------
- Sub-score names appear in two places (top-level fields and
  ``details`` keys); ``_METRIC_NAMES`` in ``scorer.py`` is the single
  source of truth and validators here cross-check against it.
- ``weights_used`` lists only the metrics that actually contributed to
  the weighted sum — i.e. those with ``computed=True`` AND a positive
  config weight. This is what step 7's LLM should see, not the raw
  config (which may include weights for skipped metrics).
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


METRIC_NAMES: tuple[str, ...] = (
    "structural_plausibility",
    "physicochemical_complementarity",
    "evolutionary_conservation",
    "cross_tool_consensus",
    "known_motif_consistency",
)


class MetricDetail(BaseModel):
    """One sub-score's outcome and intermediate values.

    A metric function returns ``(score, info)``. The scorer wraps that
    pair into a ``MetricDetail`` so the result record can carry both
    the success state (``computed``) and any failure reason
    (``error``) without losing the per-metric ``info`` dict.

    Invariants
    ----------
    - ``computed=True``  ⇒ ``score`` is a float in [0, 1] and ``error``
      is None.
    - ``computed=False`` ⇒ ``score`` is None. ``error`` may be set
      (exception trapped) or None (metric chose to abstain).
    """
    model_config = ConfigDict(extra="forbid")

    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    computed: bool = False
    error: Optional[str] = Field(default=None, max_length=2000)
    info: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistency(self) -> "MetricDetail":
        if self.computed and self.score is None:
            raise ValueError("computed=True requires a non-None score")
        if not self.computed and self.score is not None:
            raise ValueError("computed=False requires score=None")
        return self


class PocketQAResult(BaseModel):
    """Step 6 quality record for one sample (Section 3.7 of the paper, formula 17).

    Five unsupervised sub-scores (each ∈ [0, 1] or None) plus the
    weighted total. Persisted as one JSONL record per sample under
    ``data/step6_outputs/`` and consumed by step 7 (decision-making
    LLM) and step 8 (weight tensor update).
    """
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(..., min_length=1)

    # 5 sub-scores (top-level for easy querying; mirrors keys in `details`).
    structural_plausibility: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    physicochemical_complementarity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    evolutionary_conservation: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cross_tool_consensus: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    known_motif_consistency: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Weighted aggregate.
    total_score: float = Field(..., ge=0.0, le=1.0)
    weights_used: dict[str, float] = Field(default_factory=dict)
    n_metrics_computed: int = Field(..., ge=0, le=len(METRIC_NAMES))

    # Per-metric details (intermediate values).
    details: dict[str, MetricDetail] = Field(default_factory=dict)

    timestamp: Optional[str] = None

    # ---------------------- validators ------------------------------------

    @field_validator("weights_used")
    @classmethod
    def _weights_keys_known(cls, v: dict[str, float]) -> dict[str, float]:
        for k, w in v.items():
            if k not in METRIC_NAMES:
                raise ValueError(
                    f"unknown metric in weights_used: {k!r}; "
                    f"must be one of {METRIC_NAMES}"
                )
            if not (0.0 <= float(w) <= 1.0):
                raise ValueError(
                    f"weight for {k!r} = {w} outside [0, 1]"
                )
        return {k: float(w) for k, w in v.items()}

    @field_validator("details")
    @classmethod
    def _details_keys_known(
        cls, v: dict[str, MetricDetail],
    ) -> dict[str, MetricDetail]:
        for k in v:
            if k not in METRIC_NAMES:
                raise ValueError(
                    f"unknown metric in details: {k!r}; "
                    f"must be one of {METRIC_NAMES}"
                )
        return v

    @model_validator(mode="after")
    def _cross_field_consistency(self) -> "PocketQAResult":
        # Top-level sub-score must match details[name].score when present.
        for name in METRIC_NAMES:
            top = getattr(self, name)
            d = self.details.get(name)
            if d is None:
                continue
            if d.score != top:
                raise ValueError(
                    f"top-level {name}={top!r} does not match "
                    f"details[{name!r}].score={d.score!r}"
                )

        # n_metrics_computed must match weights_used cardinality (only
        # computed-and-weighted metrics contribute to the aggregate).
        if self.n_metrics_computed != len(self.weights_used):
            raise ValueError(
                f"n_metrics_computed={self.n_metrics_computed} != "
                f"len(weights_used)={len(self.weights_used)}"
            )
        return self
