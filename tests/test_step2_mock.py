"""Mock tests for Step 2 main logic — no real LLM calls.

Covers:
  - happy path: valid JSON on first try → success record
  - markdown-wrapped JSON: regex fallback recovers it
  - malformed JSON on first, valid on retry → success record with retries=1
  - schema-invalid on first, valid on retry → success with retries=1
  - persistent bad JSON on both attempts → fallback record
  - persistent schema-invalid on both attempts → fallback record
  - API transport error on first attempt → fallback record, no retry call
  - API transport error on retry → fallback record
  - api_usage sums across attempts
  - retries count is correct (0 when first OK, 1 when retry succeeds/fails)
  - input_features always present and matches extract_target_features output
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step2_target_char.feature_adapter import extract_target_features  # noqa: E402
from step2_target_char.llm_client import LLMError  # noqa: E402
from step2_target_char.target_char import (  # noqa: E402
    characterize_target, _try_extract_json, _parse_and_validate, _sum_usage,
)


# ---------- fixtures --------------------------------------------------------


def _sample_json() -> dict:
    """Minimal sample_json satisfying extract_target_features."""
    return {
        "sample_id": "test_A_B",
        "protein": {
            "length": 90, "sequence": "M" * 90,
            "features": {
                "pI": 9.2, "mean_bfactor": 45.0,
                "aa_composition": {"K": 0.15, "R": 0.12, "A": 0.08, "M": 0.65},
            },
        },
        "rna": {
            "length": 50, "sequence": "A" * 50,
            "has_modification": False,
            "features": {"gc_content": 0.4, "ss_status": "done"},
        },
        "interaction": {
            "binding_protein_residues": list(range(15)),
            "binding_rna_nucleotides": list(range(10)),
        },
        "data_availability": {
            "quality_tier": "strict", "resolution": 2.5,
            "experimental_method": "X-RAY",
        },
    }


def _config() -> dict:
    return {
        "api": {"temperature": 0.1, "model": "llm-model"},
        "fallback": {
            "protein_domain": "novel_fold",
            "rna_structure": "unstructured",
            "confidence": 0.0,
            "notes": "LLM failed — fallback emitted",
        },
    }


def _mk_response(content: str, usage: dict | None = None) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": usage or {
            "prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200,
        },
    }


def _valid_output_json() -> str:
    return json.dumps({
        "analysis": "This sample shows a classic RRM binding profile with "
                    "basic protein (pI 9.2) and a short RNA with moderate GC.",
        "protein_domain": "RRM",
        "rna_structure": "stem_loop",
        "category": "RRM_x_stem_loop",
        "confidence": 0.75,
        "notes": None,
    })


# ---------- helper-level tests ---------------------------------------------


class TestTryExtractJson(unittest.TestCase):
    def test_clean_json(self):
        self.assertEqual(_try_extract_json('{"a": 1}'), {"a": 1})

    def test_empty_input(self):
        self.assertIsNone(_try_extract_json(""))
        self.assertIsNone(_try_extract_json("   "))

    def test_markdown_wrapped_json(self):
        wrapped = '```json\n{"category": "RRM_x_stem_loop"}\n```'
        result = _try_extract_json(wrapped)
        self.assertEqual(result, {"category": "RRM_x_stem_loop"})

    def test_prose_prefix_then_json(self):
        text = 'Here is the result:\n{"x": 1, "y": [1,2,3]}'
        self.assertEqual(_try_extract_json(text), {"x": 1, "y": [1, 2, 3]})

    def test_multiline_json(self):
        text = '{\n  "a": 1,\n  "b": "two"\n}'
        self.assertEqual(_try_extract_json(text), {"a": 1, "b": "two"})

    def test_no_json_at_all(self):
        self.assertIsNone(_try_extract_json("nothing JSON-like here"))

    def test_top_level_not_object(self):
        # Regex catches the `[1,2,3]` inside braces? No — regex needs { ... }
        self.assertIsNone(_try_extract_json("[1, 2, 3]"))


class TestParseAndValidate(unittest.TestCase):
    def test_valid(self):
        out, err = _parse_and_validate(_valid_output_json())
        self.assertIsNotNone(out)
        self.assertIsNone(err)
        self.assertEqual(out.category, "RRM_x_stem_loop")

    def test_not_json(self):
        out, err = _parse_and_validate("not json at all")
        self.assertIsNone(out)
        self.assertIn("JSON", err)

    def test_schema_invalid_category_mismatch(self):
        bad = json.dumps({
            "analysis": "A" * 50,
            "protein_domain": "RRM", "rna_structure": "stem_loop",
            "category": "KH_x_junction",  # mismatch
            "confidence": 0.5, "notes": None,
        })
        out, err = _parse_and_validate(bad)
        self.assertIsNone(out)
        self.assertIn("does not match", err)

    def test_schema_invalid_unknown_enum(self):
        bad = json.dumps({
            "analysis": "A" * 50,
            "protein_domain": "bogus_domain", "rna_structure": "stem_loop",
            "category": "bogus_domain_x_stem_loop",
            "confidence": 0.5, "notes": None,
        })
        out, err = _parse_and_validate(bad)
        self.assertIsNone(out)


class TestSumUsage(unittest.TestCase):
    def test_single(self):
        r = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        self.assertEqual(_sum_usage(r), {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
        })

    def test_multiple(self):
        r1 = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        r2 = {"usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}}
        self.assertEqual(_sum_usage(r1, r2), {
            "prompt_tokens": 30, "completion_tokens": 13, "total_tokens": 43,
        })

    def test_skips_none(self):
        r1 = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        self.assertEqual(_sum_usage(r1, None, None), {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
        })

    def test_empty(self):
        self.assertEqual(_sum_usage(), {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        })


# ---------- end-to-end (mocked client) -------------------------------------


class TestCharacterizeTarget(unittest.TestCase):
    def _client(self, *responses):
        """Build a mock client whose `call()` yields `responses` in order."""
        client = MagicMock()
        client.call.side_effect = list(responses)
        return client

    def test_happy_path(self):
        client = self._client(_mk_response(_valid_output_json()))
        record = characterize_target(_sample_json(), client, _config())

        self.assertTrue(record["success"])
        self.assertEqual(record["retries"], 0)
        self.assertEqual(record["output"]["category"], "RRM_x_stem_loop")
        self.assertEqual(record["output"]["confidence"], 0.75)
        self.assertEqual(record["api_usage"]["total_tokens"], 1200)
        self.assertEqual(client.call.call_count, 1)
        # input_features must match the extract step
        expected = extract_target_features(_sample_json()).model_dump()
        self.assertEqual(record["input_features"], expected)
        self.assertIn("timestamp", record)

    def test_markdown_wrapped_recovered(self):
        content = f"```json\n{_valid_output_json()}\n```"
        client = self._client(_mk_response(content))
        record = characterize_target(_sample_json(), client, _config())
        self.assertTrue(record["success"])
        self.assertEqual(record["retries"], 0)

    def test_retry_rescues_malformed_json(self):
        client = self._client(
            _mk_response("this is not JSON at all"),
            _mk_response(_valid_output_json(), usage={
                "prompt_tokens": 1500, "completion_tokens": 250, "total_tokens": 1750,
            }),
        )
        record = characterize_target(_sample_json(), client, _config())
        self.assertTrue(record["success"])
        self.assertEqual(record["retries"], 1)
        # usage sums across both attempts (default usage 1200 + retry usage 1750)
        self.assertEqual(record["api_usage"]["total_tokens"], 1200 + 1750)
        self.assertEqual(record["api_usage"]["completion_tokens"], 200 + 250)
        self.assertEqual(record["api_usage"]["prompt_tokens"], 1000 + 1500)
        self.assertEqual(client.call.call_count, 2)

    def test_retry_rescues_schema_violation(self):
        bad = json.dumps({
            "analysis": "A" * 40,
            "protein_domain": "RRM", "rna_structure": "stem_loop",
            "category": "KH_x_junction",  # mismatch
            "confidence": 0.5, "notes": None,
        })
        client = self._client(_mk_response(bad), _mk_response(_valid_output_json()))
        record = characterize_target(_sample_json(), client, _config())
        self.assertTrue(record["success"])
        self.assertEqual(record["retries"], 1)

    def test_correction_prompt_includes_error(self):
        """The retry call should receive a user turn quoting the validation error."""
        bad = json.dumps({
            "analysis": "A" * 40,
            "protein_domain": "RRM", "rna_structure": "stem_loop",
            "category": "WRONG_FORMAT",
            "confidence": 0.5, "notes": None,
        })
        client = self._client(_mk_response(bad), _mk_response(_valid_output_json()))
        characterize_target(_sample_json(), client, _config())
        # 2nd call's messages should have 4 items (system, user, assistant, user-correction)
        second_call_messages = client.call.call_args_list[1].args[0]
        self.assertEqual(len(second_call_messages), 4)
        self.assertEqual(second_call_messages[-2]["role"], "assistant")
        self.assertEqual(second_call_messages[-1]["role"], "user")
        self.assertIn("Validation error", second_call_messages[-1]["content"])

    def test_fallback_on_persistent_bad_json(self):
        client = self._client(
            _mk_response("garbage"),
            _mk_response("still garbage on retry"),
        )
        record = characterize_target(_sample_json(), client, _config())
        self.assertFalse(record["success"])
        self.assertEqual(record["retries"], 1)
        self.assertEqual(record["output"]["category"], "novel_fold_x_unstructured")
        self.assertEqual(record["output"]["confidence"], 0.0)
        self.assertIn("cause", record["output"]["notes"])
        self.assertIn("failure_reason", record)
        self.assertEqual(client.call.call_count, 2)

    def test_fallback_on_persistent_schema_violation(self):
        bad = json.dumps({
            "analysis": "A" * 30,
            "protein_domain": "Nope",  # unknown enum
            "rna_structure": "stem_loop",
            "category": "Nope_x_stem_loop",
            "confidence": 0.5, "notes": None,
        })
        client = self._client(_mk_response(bad), _mk_response(bad))
        record = characterize_target(_sample_json(), client, _config())
        self.assertFalse(record["success"])
        self.assertEqual(record["retries"], 1)
        self.assertIn("schema validation failed", record["failure_reason"])

    def test_api_error_attempt1_fallback_no_retry_call(self):
        client = MagicMock()
        client.call.side_effect = LLMError("endpoint down")
        record = characterize_target(_sample_json(), client, _config())
        self.assertFalse(record["success"])
        self.assertEqual(record["retries"], 0)
        self.assertIn("API error (attempt 1)", record["failure_reason"])
        # Must NOT retry on API transport error (client already retried internally)
        self.assertEqual(client.call.call_count, 1)

    def test_api_error_on_retry_fallback(self):
        client = self._client(_mk_response("garbage"))
        client.call.side_effect = [
            _mk_response("garbage"),
            LLMError("endpoint down on retry"),
        ]
        record = characterize_target(_sample_json(), client, _config())
        self.assertFalse(record["success"])
        self.assertEqual(record["retries"], 1)
        self.assertIn("API error on correction retry", record["failure_reason"])

    def test_empty_content_triggers_retry(self):
        client = self._client(
            _mk_response(""),
            _mk_response(_valid_output_json()),
        )
        record = characterize_target(_sample_json(), client, _config())
        self.assertTrue(record["success"])
        self.assertEqual(record["retries"], 1)

    def test_record_always_has_required_keys(self):
        """Both success and fallback records must have the same top-level shape."""
        required = {
            "sample_id", "input_features", "output", "api_usage",
            "timestamp", "success", "retries",
        }
        # success
        client = self._client(_mk_response(_valid_output_json()))
        rec = characterize_target(_sample_json(), client, _config())
        self.assertTrue(required.issubset(rec.keys()))
        # fallback
        client2 = self._client(_mk_response("x"), _mk_response("y"))
        rec2 = characterize_target(_sample_json(), client2, _config())
        self.assertTrue(required.issubset(rec2.keys()))

    def test_jsonl_serializable(self):
        """Records must round-trip JSON (no datetime objects, no tuples-as-keys)."""
        client = self._client(_mk_response(_valid_output_json()))
        rec = characterize_target(_sample_json(), client, _config())
        line = json.dumps(rec)
        roundtrip = json.loads(line)
        self.assertEqual(roundtrip["output"]["category"], "RRM_x_stem_loop")


class TestHistoryAdvisory(unittest.TestCase):
    def test_broken_history_does_not_crash(self):
        class BrokenHistory:
            def get_summary(self):
                raise RuntimeError("history is corrupt")

        client = MagicMock()
        client.call.return_value = _mk_response(_valid_output_json())
        record = characterize_target(
            _sample_json(), client, _config(), history=BrokenHistory(),
        )
        # Success despite history failure — history is advisory only
        self.assertTrue(record["success"])

    def test_history_summary_injected(self):
        class OkHistory:
            def get_summary(self):
                return "Past: 20 samples, avg conf 0.7"

        client = MagicMock()
        client.call.return_value = _mk_response(_valid_output_json())
        characterize_target(
            _sample_json(), client, _config(), history=OkHistory(),
        )
        # The call should have received a system prompt containing the history
        messages = client.call.call_args.args[0]
        self.assertIn("Historical context", messages[0]["content"])
        self.assertIn("avg conf 0.7", messages[0]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
