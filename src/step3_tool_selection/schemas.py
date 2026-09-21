"""Pydantic schemas for Step 3 — tool selection.

Input side:
  `ToolSelectionInput` — everything the LLM sees when deciding which tools
  to run. Assembled by `tool_selector.py` from step 2 output + weight tensor.

Output side:
  `ToolPlan` — the structured plan the LLM returns. Validated against the
  tool registry (all tool_ids must exist) and config constraints (min/max
  tools, allowed execution strategies).
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .tool_registry import ALL_TOOL_IDS, TOOL_COUNT


# ---------- enums -----------------------------------------------------------


class ExecutionStrategy(str, Enum):
    cascade = "cascade"
    parallel = "parallel"
    staged = "staged"


# ---------- output side -----------------------------------------------------


class ToolParamOverride(BaseModel):
    """One parameter tweak the LLM wants applied before running a tool."""
    tool_id: str
    param_name: str
    param_value: Any
    reason: str

    @field_validator("tool_id")
    @classmethod
    def _tool_exists(cls, v: str) -> str:
        if v not in ALL_TOOL_IDS:
            raise ValueError(f"tool_id '{v}' not in registry")
        return v


class ToolPlan(BaseModel):
    """LLM-produced tool execution plan.

    Validators enforce:
      - every selected tool_id exists in the registry
      - at least 1 tool selected
      - at most the whole library (the MANDATORY core plus the LLM's picks)
      - no duplicate tool_ids
      - confidence in [0, 1]
      - execution_strategy is a known enum
      - param_override tool_ids are a subset of selected_tools
    """
    selected_tools: list[str] = Field(min_length=1, max_length=TOOL_COUNT)
    execution_strategy: ExecutionStrategy
    param_overrides: list[ToolParamOverride] = Field(default_factory=list)
    early_stop_threshold: Optional[float] = Field(default=None, ge=0, le=1)
    rationale: str = Field(min_length=10, max_length=4000)
    confidence: float = Field(ge=0, le=1)

    @field_validator("selected_tools")
    @classmethod
    def _all_tools_exist(cls, v: list[str]) -> list[str]:
        unknown = set(v) - ALL_TOOL_IDS
        if unknown:
            raise ValueError(f"unknown tool_ids: {sorted(unknown)}")
        if len(v) != len(set(v)):
            raise ValueError("duplicate tool_ids in selected_tools")
        return v

    @model_validator(mode="after")
    def _overrides_subset_of_selected(self) -> "ToolPlan":
        selected = set(self.selected_tools)
        for ov in self.param_overrides:
            if ov.tool_id not in selected:
                raise ValueError(
                    f"param_override references '{ov.tool_id}' which is "
                    f"not in selected_tools {sorted(selected)}"
                )
        return self


# ---------- input side (assembled by tool_selector.py) ----------------------
# These are NOT sent to the LLM as JSON; they're used internally to build
# the prompt and to keep the pipeline typed. Importing step2 schemas here
# creates a cross-step dependency, which is intentional — step 3 consumes
# step 2's output.


class ToolSelectionInput(BaseModel):
    """Everything needed to build the step 3 prompt for one sample."""
    sample_id: str
    # step 2 outputs (dict form to avoid tight coupling to step2 Pydantic models
    # that may not be importable on every code path)
    target_char: dict
    target_features: dict
    weight_summary: str
    utility_scores: dict[str, float]
    available_tools: list[str]
