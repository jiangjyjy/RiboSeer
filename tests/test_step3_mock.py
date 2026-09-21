"""Mock tests for Step 3 tool selection — no real LLM calls."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step2_target_char.llm_client import LLMError  # noqa: E402
from step3_tool_selection.tool_selector import (  # noqa: E402
    MANDATORY_TOOLS, select_tools, _enforce_mandatory_tools,
    _try_extract_json, _parse_and_validate,
)
from step3_tool_selection.weight_tensor import WeightTensor  # noqa: E402


# ---------- fixtures --------------------------------------------------------


def _target_char(cat: str = "RRM_x_stem_loop", conf: float = 0.9) -> dict:
    return {
        "analysis": "Classic RRM binding a structured stem-loop RNA.",
        "protein_domain": cat.split("_x_")[0],
        "rna_structure": cat.split("_x_")[1],
        "category": cat,
        "confidence": conf,
        "notes": None,
    }


def _target_features() -> dict:
    return {
        "sample_id": "test_A_B",
        "protein_length": 90, "rna_length": 61,
        "protein_pI": 8.87, "rna_gc_content": 0.65,
        "interface_ratio_protein": 0.19, "interface_ratio_rna": 0.23,
        "quality_tier": "strict", "resolution": 3.1,
        "experimental_method": "X-RAY",
    }


def _config() -> dict:
    return {
        "api": {"temperature": 0.1},
        "selection": {"max_tools": 6},
        "fallback": {
            "tools": ["equipnas", "p2rank"],
            "strategy": "cascade",
            "confidence": 0.0,
            "notes": "fallback",
        },
    }


def _mk_response(content: str, usage: dict | None = None) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": usage or {"prompt_tokens": 2000, "completion_tokens": 400, "total_tokens": 2400},
    }


def _valid_plan_json(**overrides) -> str:
    base = {
        "selected_tools": ["equipnas", "p2rank", "boltz2"],
        "execution_strategy": "cascade",
        "param_overrides": [],
        "early_stop_threshold": 0.8,
        "rationale": "EquiPNAS gives a fast residue-level baseline for this well-characterized RRM target; "
                     "P2Rank adds pocket detection and Boltz-2 supplies a full complex prediction.",
        "confidence": 0.85,
    }
    base.update(overrides)
    return json.dumps(base)


# ---------- parser tests ---------------------------------------------------


class TestParseAndValidate(unittest.TestCase):
    def test_valid(self):
        plan, err = _parse_and_validate(_valid_plan_json())
        self.assertIsNotNone(plan)
        self.assertIsNone(err)

    def test_not_json(self):
        plan, err = _parse_and_validate("not json")
        self.assertIsNone(plan)
        self.assertIn("JSON", err)

    def test_unknown_tool(self):
        plan, err = _parse_and_validate(_valid_plan_json(
            selected_tools=["equipnas", "fake_tool"],
        ))
        self.assertIsNone(plan)
        self.assertIn("fake_tool", err)

    def test_unknown_strategy(self):
        plan, err = _parse_and_validate(_valid_plan_json(execution_strategy="yolo"))
        self.assertIsNone(plan)


# ---------- end-to-end (mocked client) ------------------------------------


class TestSelectTools(unittest.TestCase):
    def _client(self, *responses):
        c = MagicMock()
        c.call.side_effect = list(responses)
        return c

    def test_happy_path(self):
        client = self._client(_mk_response(_valid_plan_json()))
        wt = WeightTensor()
        record = select_tools(
            "test_A_B", _target_char(), _target_features(),
            client, wt, _config(),
        )
        self.assertTrue(record["success"])
        self.assertEqual(record["retries"], 0)
        self.assertEqual(record["step2_category"], "RRM_x_stem_loop")
        self.assertEqual(record["tool_plan"]["execution_strategy"], "cascade")
        self.assertIn("equipnas", record["tool_plan"]["selected_tools"])
        self.assertEqual(client.call.call_count, 1)

    def test_different_category_same_structure(self):
        """Different step2 category should still produce a valid plan."""
        client = self._client(_mk_response(_valid_plan_json(
            selected_tools=["equipnas", "chai1", "boltz2"],
            rationale="Novel fold benefits from running both Cat A tools (chai1, boltz2) as independent A-tier predictions, plus equipnas for residue-level signal.",
        )))
        wt = WeightTensor()
        record = select_tools(
            "test_B_C", _target_char("novel_fold_x_junction", 0.6),
            _target_features(), client, wt, _config(),
        )
        self.assertTrue(record["success"])
        self.assertEqual(record["step2_category"], "novel_fold_x_junction")
        self.assertIn("chai1", record["tool_plan"]["selected_tools"])

    def test_retry_rescues_bad_json(self):
        client = self._client(
            _mk_response("not json"),
            _mk_response(_valid_plan_json()),
        )
        wt = WeightTensor()
        record = select_tools(
            "test_A_B", _target_char(), _target_features(),
            client, wt, _config(),
        )
        self.assertTrue(record["success"])
        self.assertEqual(record["retries"], 1)
        self.assertEqual(client.call.call_count, 2)

    def test_retry_rescues_schema_violation(self):
        bad = json.dumps({
            "selected_tools": ["fake_tool"],
            "execution_strategy": "cascade",
            "param_overrides": [],
            "rationale": "short but enough for retry test purposes here",
            "confidence": 0.5,
        })
        client = self._client(_mk_response(bad), _mk_response(_valid_plan_json()))
        wt = WeightTensor()
        record = select_tools(
            "test_A_B", _target_char(), _target_features(),
            client, wt, _config(),
        )
        self.assertTrue(record["success"])
        self.assertEqual(record["retries"], 1)

    def test_fallback_on_persistent_bad(self):
        client = self._client(
            _mk_response("garbage"),
            _mk_response("still garbage"),
        )
        wt = WeightTensor()
        record = select_tools(
            "test_A_B", _target_char(), _target_features(),
            client, wt, _config(),
        )
        self.assertFalse(record["success"])
        self.assertEqual(record["retries"], 1)
        plan = record["tool_plan"]
        # Fallback list is preserved and PREPENDED, then the 7-tool
        # MANDATORY core is appended even on the fallback path so the
        # strongest tools always run.
        expected = ["equipnas", "p2rank"] + [
            t for t in MANDATORY_TOOLS if t not in ("equipnas", "p2rank")]
        self.assertEqual(plan["selected_tools"], expected)
        self.assertEqual(plan["execution_strategy"], "cascade")
        self.assertEqual(plan["confidence"], 0.0)
        self.assertIn("failure_reason", record)

    def test_api_error_attempt1_fallback(self):
        client = MagicMock()
        client.call.side_effect = LLMError("endpoint down")
        wt = WeightTensor()
        record = select_tools(
            "test_A_B", _target_char(), _target_features(),
            client, wt, _config(),
        )
        self.assertFalse(record["success"])
        self.assertEqual(record["retries"], 0)
        self.assertEqual(client.call.call_count, 1)

    def test_api_error_on_retry_fallback(self):
        client = MagicMock()
        client.call.side_effect = [
            _mk_response("garbage"),
            LLMError("down on retry"),
        ]
        wt = WeightTensor()
        record = select_tools(
            "test_A_B", _target_char(), _target_features(),
            client, wt, _config(),
        )
        self.assertFalse(record["success"])
        self.assertEqual(record["retries"], 1)

    def test_usage_sums_across_attempts(self):
        client = self._client(
            _mk_response("bad", usage={"prompt_tokens": 2000, "completion_tokens": 100, "total_tokens": 2100}),
            _mk_response(_valid_plan_json(), usage={"prompt_tokens": 2500, "completion_tokens": 400, "total_tokens": 2900}),
        )
        wt = WeightTensor()
        record = select_tools(
            "test_A_B", _target_char(), _target_features(),
            client, wt, _config(),
        )
        self.assertEqual(record["api_usage"]["total_tokens"], 2100 + 2900)

    def test_utility_scores_in_record(self):
        client = self._client(_mk_response(_valid_plan_json()))
        wt = WeightTensor()
        record = select_tools(
            "test_A_B", _target_char(), _target_features(),
            client, wt, _config(),
        )
        self.assertIn("utility_scores", record)
        self.assertIn("equipnas", record["utility_scores"])

    def test_record_jsonl_serializable(self):
        client = self._client(_mk_response(_valid_plan_json()))
        wt = WeightTensor()
        record = select_tools(
            "test_A_B", _target_char(), _target_features(),
            client, wt, _config(),
        )
        line = json.dumps(record)
        rt = json.loads(line)
        self.assertEqual(rt["tool_plan"]["selected_tools"][0], "equipnas")

    def test_correction_prompt_has_error(self):
        bad = json.dumps({
            "selected_tools": ["fake_tool"],
            "execution_strategy": "cascade",
            "param_overrides": [],
            "rationale": "enough text for the rationale field validation.",
            "confidence": 0.5,
        })
        client = self._client(_mk_response(bad), _mk_response(_valid_plan_json()))
        wt = WeightTensor()
        select_tools("test_A_B", _target_char(), _target_features(),
                     client, wt, _config())
        msgs2 = client.call.call_args_list[1].args[0]
        self.assertEqual(len(msgs2), 4)
        self.assertIn("Validation error", msgs2[-1]["content"])

    def test_empty_content_triggers_retry(self):
        client = self._client(
            _mk_response(""),
            _mk_response(_valid_plan_json()),
        )
        wt = WeightTensor()
        record = select_tools(
            "test_A_B", _target_char(), _target_features(),
            client, wt, _config(),
        )
        self.assertTrue(record["success"])
        self.assertEqual(record["retries"], 1)


class TestHistoryIntegration(unittest.TestCase):
    def test_broken_history_does_not_crash(self):
        class BrokenHistory:
            def get_summary(self):
                raise RuntimeError("corrupt")

        client = MagicMock()
        client.call.return_value = _mk_response(_valid_plan_json())
        wt = WeightTensor()
        record = select_tools(
            "test_A_B", _target_char(), _target_features(),
            client, wt, _config(), history=BrokenHistory(),
        )
        self.assertTrue(record["success"])


class TestMandatoryToolEnforcement(unittest.TestCase):
    """The 7-tool MANDATORY core (paper Table 1) must always end up in
    selected_tools regardless of what the LLM picked."""

    def test_mandatory_set(self):
        self.assertEqual(set(MANDATORY_TOOLS), {
            "boltz2", "chai1", "deeppocket", "equipnas",
            "nucleicnet", "rnabindrplus", "hdock"})

    def test_enforce_appends_missing(self):
        out = _enforce_mandatory_tools(["equipnas"])
        expected = ["equipnas"] + [
            t for t in MANDATORY_TOOLS if t != "equipnas"]
        self.assertEqual(out, expected)

    def test_enforce_preserves_order_and_dedupes(self):
        out = _enforce_mandatory_tools(
            ["chai1", "p2rank", "chai1", "boltz2"])
        # LLM order kept, duplicate chai1 dropped, missing core added.
        expected = ["chai1", "p2rank", "boltz2"] + [
            t for t in MANDATORY_TOOLS if t not in ("chai1", "boltz2")]
        self.assertEqual(out, expected)

    def test_enforce_noop_when_all_present(self):
        sel = ["p2rank"] + list(MANDATORY_TOOLS)
        self.assertEqual(_enforce_mandatory_tools(sel), sel)

    def test_happy_path_plan_gets_mandatory_appended(self):
        # _valid_plan_json picks equipnas/p2rank/boltz2; the rest of the
        # MANDATORY core must be appended onto the LLM's plan.
        client = MagicMock()
        client.call.return_value = _mk_response(_valid_plan_json())
        wt = WeightTensor()
        rec = select_tools(
            "test_A_B", _target_char(), _target_features(),
            client, wt, _config(),
        )
        self.assertTrue(rec["success"])
        sel = rec["tool_plan"]["selected_tools"]
        for t in MANDATORY_TOOLS:
            self.assertIn(t, sel)
        # LLM's original picks still lead the list (order preserved).
        self.assertEqual(sel[:3], ["equipnas", "p2rank", "boltz2"])
        self.assertEqual(len(sel), len(set(sel)))  # no dupes

    def test_unavailable_mandatory_tool_not_forced(self):
        # If hdock were disabled, is_available() gates it out of the
        # appended set without touching MANDATORY_TOOLS.
        import step3_tool_selection.tool_selector as ts
        real = ts.is_available
        ts.is_available = lambda t: t != "hdock" and real(t)
        try:
            out = _enforce_mandatory_tools(["equipnas"])
        finally:
            ts.is_available = real
        self.assertNotIn("hdock", out)
        self.assertEqual(
            out, ["equipnas"] + [t for t in MANDATORY_TOOLS
                                 if t not in ("equipnas", "hdock")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
