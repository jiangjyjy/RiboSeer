"""Pydantic schemas for step 8 — weight tensor update record.

One model per sample, persisted as one JSONL line under
``data/step8_outputs/``:

  - ``WeightUpdateResult``: what changed in W for this sample.
    Carries the EMA delta per (tool, metric), the optional LLM
    meta-correction factors γ_k, and a before/after snapshot of the
    *category slice* of W (i.e. the (K × M) plane for the sample's
    category j). Step 8 doesn't dump the full W tensor here — that's
    persisted separately by ``WeightTensor.save``.

Conventions
-----------
- Sub-score / metric names are the canonical 5 from
  ``step3_tool_selection.weight_tensor.METRICS``; the validator
  cross-checks that any metric key in ``ema_deltas`` /
  ``weights_before`` / ``weights_after`` is one of them.
- Correction factors are clamped to ``[0.5, 1.5]`` (Section 3.6, Phase 2, of the paper).
- ``meta_correction_applied=True`` ⇒ ``correction_factors`` is non-empty
  and every value sits in the clamp range. The validator enforces both.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Single source of truth — re-import so a future METRICS rename can't
# silently desync. Importing inside the module makes the schema usable
# even if step 3 changes implementation.
try:
    from step3_tool_selection.weight_tensor import METRICS as _METRICS  # noqa: E501
    METRIC_NAMES: tuple[str, ...] = tuple(_METRICS)
except Exception:  # pragma: no cover — defensive fallback
    METRIC_NAMES = (
        "structural_plausibility",
        "physicochemical_complementarity",
        "evolutionary_conservation",
        "cross_tool_consensus",
        "known_motif_consistency",
    )


# ``Phase 2 formula 16``: γ_k must lie in [0.5, 1.5].
META_CORRECTION_MIN: float = 0.5
META_CORRECTION_MAX: float = 1.5


def _check_metric_dict(d: dict[str, dict[str, float]], *, field: str,
                       allow_empty: bool = True) -> dict[str, dict[str, float]]:
    """Shared validator: keys are tool ids → {metric: float} sub-dicts."""
    if not d:
        if allow_empty:
            return {}
        raise ValueError(f"{field} must not be empty")
    for tool_id, mdict in d.items():
        if not isinstance(tool_id, str) or not tool_id:
            raise ValueError(f"{field}: tool_id must be non-empty string")
        if not isinstance(mdict, dict):
            raise ValueError(
                f"{field}[{tool_id!r}] must be a dict {{metric: float}}"
            )
        for metric, value in mdict.items():
            if metric not in METRIC_NAMES:
                raise ValueError(
                    f"{field}[{tool_id!r}]: unknown metric {metric!r}; "
                    f"must be one of {METRIC_NAMES}"
                )
            try:
                float(value)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"{field}[{tool_id!r}][{metric!r}] not a number: {value!r}"
                ) from e
    return d


class WeightUpdateResult(BaseModel):
    """Per-sample weight-update record (Section 3.6, Phase 2, of the paper).

    Field semantics
    ---------------
    - ``ema_deltas`` records the *change* in W produced by the EMA pass:
      ``delta = W_after - W_before`` for each (tool, metric) cell that
      actually moved. Cells corresponding to failed tools are absent
      (failed tools must NOT be updated — they're skipped at compute time).

    - ``correction_factors`` is None when ``enable_meta_correction`` is
      false, when there's not enough history, or when the LLM call
      failed. When set, it's ``{tool_id: γ_k}`` with all γ_k ∈ [0.5, 1.5].

    - ``weights_before`` / ``weights_after`` are snapshots of the
      ``W[:, :, j]`` slice for the sample's category. Both contain only
      the tools listed in ``tools_updated``; missing-tool reads on the
      tensor side fall back to per-category defaults so we don't explode
      the snapshot with cold-start cells that were never touched.
    """
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1,
                          description="target category j in W's third axis")

    # ----- EMA pass --------------------------------------------------------
    tools_updated: list[str] = Field(
        default_factory=list,
        description="tools that actually had at least one cell updated",
    )
    ema_deltas: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="{tool_id: {metric: delta}} — W_after - W_before per cell",
    )
    learning_rate: float = Field(..., gt=0.0, le=1.0,
                                 description="η used for this update")

    # ----- Optional LLM meta-correction (formulas 15-16) -----------------
    meta_correction_applied: bool = False
    correction_factors: Optional[dict[str, float]] = Field(
        default=None,
        description="{tool_id: γ_k}; non-None iff meta_correction_applied",
    )
    meta_rationale: Optional[str] = Field(default=None, max_length=4000)

    # ----- Snapshots (per-category slice only) -----------------------------
    weights_before: dict[str, dict[str, float]] = Field(default_factory=dict)
    weights_after: dict[str, dict[str, float]] = Field(default_factory=dict)

    # ----- Bookkeeping -----------------------------------------------------
    api_usage: dict = Field(default_factory=dict)
    timestamp: str = Field(..., min_length=1)

    # -- field-level normalisers -------------------------------------------

    @field_validator("tools_updated")
    @classmethod
    def _tools_updated_unique(cls, v: list[str]) -> list[str]:
        if any(not isinstance(t, str) or not t for t in v):
            raise ValueError("tools_updated entries must be non-empty strings")
        if len(v) != len(set(v)):
            raise ValueError("tools_updated must be unique")
        return list(v)

    @field_validator("ema_deltas", "weights_before", "weights_after")
    @classmethod
    def _check_metric_dicts(
        cls, v: dict[str, dict[str, float]], info,
    ) -> dict[str, dict[str, float]]:
        return _check_metric_dict(v, field=info.field_name, allow_empty=True)

    @field_validator("correction_factors")
    @classmethod
    def _check_correction_range(
        cls, v: Optional[dict[str, float]],
    ) -> Optional[dict[str, float]]:
        if v is None:
            return None
        for tool_id, gamma in v.items():
            if not isinstance(tool_id, str) or not tool_id:
                raise ValueError(
                    "correction_factors: tool_id must be non-empty string"
                )
            try:
                g = float(gamma)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"correction_factors[{tool_id!r}]={gamma!r} not a number"
                ) from e
            if not (META_CORRECTION_MIN <= g <= META_CORRECTION_MAX):
                raise ValueError(
                    f"correction_factors[{tool_id!r}]={g} outside "
                    f"[{META_CORRECTION_MIN}, {META_CORRECTION_MAX}]"
                )
        return {k: float(v[k]) for k in v}

    # -- cross-field consistency -------------------------------------------

    @model_validator(mode="after")
    def _consistency(self) -> "WeightUpdateResult":
        # tools_updated should match ema_deltas keys exactly.
        delta_tools = set(self.ema_deltas.keys())
        listed = set(self.tools_updated)
        if delta_tools != listed:
            raise ValueError(
                f"tools_updated {sorted(listed)} does not match "
                f"ema_deltas keys {sorted(delta_tools)}"
            )

        # before / after snapshots: tools_updated should be a subset (a
        # snapshot may include extra tools that weren't touched this turn,
        # but every updated tool must appear in both snapshots).
        for snap_name in ("weights_before", "weights_after"):
            snap = getattr(self, snap_name)
            missing = [t for t in self.tools_updated if t not in snap]
            if missing:
                raise ValueError(
                    f"{snap_name} missing snapshot rows for tools_updated: "
                    f"{missing}"
                )

        # Snapshots must align: same tool keys, same metric keys per tool.
        before_keys = set(self.weights_before.keys())
        after_keys = set(self.weights_after.keys())
        if before_keys != after_keys:
            raise ValueError(
                "weights_before and weights_after must list the same tools; "
                f"diff: only_before={sorted(before_keys - after_keys)}, "
                f"only_after={sorted(after_keys - before_keys)}"
            )
        for tool_id in before_keys:
            bm = set(self.weights_before[tool_id].keys())
            am = set(self.weights_after[tool_id].keys())
            if bm != am:
                raise ValueError(
                    f"metric keys mismatch for {tool_id!r}: "
                    f"before={sorted(bm)} after={sorted(am)}"
                )

        # delta == after - before within tolerance (catches bookkeeping bugs).
        tol = 1e-6
        for tool_id, mdict in self.ema_deltas.items():
            for metric, delta in mdict.items():
                before = self.weights_before.get(tool_id, {}).get(metric)
                after = self.weights_after.get(tool_id, {}).get(metric)
                if before is None or after is None:
                    raise ValueError(
                        f"ema_deltas[{tool_id!r}][{metric!r}] present but "
                        f"snapshot missing the same cell"
                    )
                expected = float(after) - float(before)
                if abs(expected - float(delta)) > tol:
                    raise ValueError(
                        f"delta inconsistent for {tool_id!r}/{metric!r}: "
                        f"after-before={expected:.6f} but "
                        f"ema_deltas={delta:.6f}"
                    )

        # meta_correction_applied flag must agree with correction_factors.
        if self.meta_correction_applied:
            if not self.correction_factors:
                raise ValueError(
                    "meta_correction_applied=True requires non-empty "
                    "correction_factors"
                )
        else:
            if self.correction_factors:
                raise ValueError(
                    "meta_correction_applied=False but correction_factors "
                    "is set; either flip the flag or clear the factors"
                )
        return self
