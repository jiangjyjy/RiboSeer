"""Unit tests for step3 schemas.py."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pydantic import ValidationError  # noqa: E402

from step3_tool_selection.schemas import (  # noqa: E402
    ExecutionStrategy, ToolParamOverride, ToolPlan, ToolSelectionInput,
)


def _valid_plan(**overrides) -> dict:
    base = {
        "selected_tools": ["equipnas", "p2rank", "boltz2"],
        "execution_strategy": "cascade",
        "param_overrides": [],
        "early_stop_threshold": 0.8,
        "rationale": "Category C tools are fast and reliable for this RRM target.",
        "confidence": 0.85,
    }
    base.update(overrides)
    return base


class TestToolPlanValid(unittest.TestCase):
    def test_happy_path(self):
        plan = ToolPlan.model_validate(_valid_plan())
        self.assertEqual(len(plan.selected_tools), 3)
        self.assertEqual(plan.execution_strategy, ExecutionStrategy.cascade)
        self.assertEqual(plan.confidence, 0.85)

    def test_all_strategies_accepted(self):
        for s in ("cascade", "parallel", "staged"):
            plan = ToolPlan.model_validate(_valid_plan(execution_strategy=s))
            self.assertEqual(plan.execution_strategy.value, s)

    def test_with_param_overrides(self):
        overrides = [{
            "tool_id": "equipnas",
            "param_name": "threshold",
            "param_value": 0.5,
            "reason": "Lower threshold for novel fold",
        }]
        plan = ToolPlan.model_validate(_valid_plan(param_overrides=overrides))
        self.assertEqual(len(plan.param_overrides), 1)
        self.assertEqual(plan.param_overrides[0].tool_id, "equipnas")

    def test_no_early_stop(self):
        plan = ToolPlan.model_validate(_valid_plan(early_stop_threshold=None))
        self.assertIsNone(plan.early_stop_threshold)

    def test_single_tool(self):
        plan = ToolPlan.model_validate(_valid_plan(selected_tools=["equipnas"]))
        self.assertEqual(len(plan.selected_tools), 1)

    def test_all_active_tools(self):
        from step3_tool_selection.tool_registry import get_all_tool_ids
        active_ids = get_all_tool_ids()
        plan = ToolPlan.model_validate(
            _valid_plan(selected_tools=active_ids)
        )
        self.assertEqual(len(plan.selected_tools), len(active_ids))


class TestToolPlanRejections(unittest.TestCase):
    def test_unknown_tool_id(self):
        with self.assertRaises(ValidationError) as ctx:
            ToolPlan.model_validate(
                _valid_plan(selected_tools=["equipnas", "nonexistent"])
            )
        self.assertIn("nonexistent", str(ctx.exception))

    def test_empty_tools_rejected(self):
        with self.assertRaises(ValidationError):
            ToolPlan.model_validate(_valid_plan(selected_tools=[]))

    def test_duplicate_tools_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            ToolPlan.model_validate(
                _valid_plan(selected_tools=["equipnas", "equipnas"])
            )
        self.assertIn("duplicate", str(ctx.exception))

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValidationError):
            ToolPlan.model_validate(_valid_plan(execution_strategy="random"))

    def test_confidence_out_of_range(self):
        with self.assertRaises(ValidationError):
            ToolPlan.model_validate(_valid_plan(confidence=1.5))
        with self.assertRaises(ValidationError):
            ToolPlan.model_validate(_valid_plan(confidence=-0.1))

    def test_rationale_too_short(self):
        with self.assertRaises(ValidationError):
            ToolPlan.model_validate(_valid_plan(rationale="short"))

    def test_early_stop_out_of_range(self):
        with self.assertRaises(ValidationError):
            ToolPlan.model_validate(_valid_plan(early_stop_threshold=1.5))

    def test_param_override_references_unselected_tool(self):
        overrides = [{
            "tool_id": "rosettafold2na",  # active tool, not in selected_tools
            "param_name": "x", "param_value": 1, "reason": "test",
        }]
        with self.assertRaises(ValidationError) as ctx:
            ToolPlan.model_validate(_valid_plan(param_overrides=overrides))
        self.assertIn("rosettafold2na", str(ctx.exception))

    def test_param_override_unknown_tool(self):
        overrides = [{
            "tool_id": "fake_tool",
            "param_name": "x", "param_value": 1, "reason": "test",
        }]
        with self.assertRaises(ValidationError) as ctx:
            ToolPlan.model_validate(
                _valid_plan(
                    selected_tools=["equipnas"],
                    param_overrides=overrides,
                )
            )
        self.assertIn("fake_tool", str(ctx.exception))


class TestToolSelectionInput(unittest.TestCase):
    def test_minimal(self):
        inp = ToolSelectionInput(
            sample_id="test_A_B",
            target_char={"category": "RRM_x_stem_loop", "confidence": 0.9},
            target_features={"protein_length": 90, "rna_length": 50},
            weight_summary="all default",
            utility_scores={"equipnas": 1.6, "p2rank": 1.5},
            available_tools=["equipnas", "p2rank"],
        )
        self.assertEqual(inp.sample_id, "test_A_B")


if __name__ == "__main__":
    unittest.main(verbosity=2)
