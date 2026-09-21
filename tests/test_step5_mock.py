"""Mock tests for step5 fusion main logic.

Covers:
  - fuse_predictions: 0-tool / 1-tool / multi-tool happy path
  - LLM JSON parse OK on first attempt
  - LLM parse failure → error-correction retry → success
  - LLM persistent failure → equal-weights fallback
  - LLM API error → fallback
  - LLM drops a tool_id from `weights` → caught, retry, then fallback
  - --no-llm equivalent (client=None) → fallback
  - History summary failure swallowed silently
  - WeightTensor.get_category_summary failure swallowed silently
  - evaluate_against_ground_truth math + edge cases
  - build_output_record matches the spec JSONL format

No real LLMClient is constructed: all tests substitute a tiny stub
that captures call kwargs and returns canned responses.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step2_target_char.llm_client import LLMError  # noqa: E402
from step4_tool_adapters.schemas import ToolPrediction  # noqa: E402
from step5_fusion.fusion import (  # noqa: E402
    build_output_record,
    evaluate_against_ground_truth,
    fuse_predictions,
)
from step5_fusion.schemas import CompositeResult  # noqa: E402


# ----------------------- fixtures -------------------------------------------


def _pred(
    tool_id: str,
    cat: str = "C",
    *,
    success: bool = True,
    binding_protein=None,
    binding_rna=None,
    per_residue=None,
    plddt: float | None = None,
    iptm: float | None = None,
    pae: float | None = None,
) -> ToolPrediction:
    if not success:
        return ToolPrediction(
            tool_id=tool_id, category=cat, sample_id="sample0",
            success=False, error_message=f"{tool_id} synthetic failure",
        )
    return ToolPrediction(
        tool_id=tool_id, category=cat, sample_id="sample0",
        success=True,
        binding_protein_residues=binding_protein,
        binding_rna_nucleotides=binding_rna,
        per_residue_confidence=per_residue,
        plddt_mean=plddt, iptm_score=iptm, pae_mean=pae,
    )


def _sample_json(sample_id: str = "sample0", gt: list[int] | None = None) -> dict:
    return {
        "sample_id": sample_id,
        "interaction": {
            "binding_protein_residues": gt if gt is not None else [10, 11, 23],
        },
    }


def _target_char() -> dict:
    return {
        "category": "RRM_x_stem_loop",
        "confidence": 0.9,
        "analysis": "RRM domain binding a stem-loop RNA.",
    }


def _config() -> dict:
    return {
        "fusion": {
            "default_threshold": 0.5,
            "fallback_weight": 0.5,
            "min_tools_for_llm": 2,
        },
        "api": {"temperature": 0.1},
    }


def _llm_response(content: str, *, tokens: int = 100) -> dict:
    """Build a minimal OpenAI-compatible chat completion response."""
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": tokens,
            "completion_tokens": tokens // 2,
            "total_tokens": tokens + tokens // 2,
        },
    }


def _good_assignment_json(tool_ids: list[str], threshold: float = 0.5) -> str:
    """Synthesise a valid ToolWeightAssignment JSON for the given tools."""
    weights = {tid: round(0.5 + 0.1 * i, 2) for i, tid in enumerate(tool_ids)}
    return json.dumps({
        "weights": weights,
        "threshold": threshold,
        "rationale": (
            "Assigning higher weight to Cat A tools with high pLDDT and "
            "lower weight to P2Rank since it's RNA-agnostic."
        ),
        "confidence": 0.85,
    })


# ----------------------- 0/1-tool short circuits ---------------------------


class TestEdgeCases(unittest.TestCase):
    def test_zero_surviving_tools_returns_empty_result(self):
        client = MagicMock()
        result = fuse_predictions(
            sample_json=_sample_json(),
            target_char=_target_char(),
            tool_predictions=[_pred("boltz2", "A", success=False)],
            client=client,
            weight_tensor=None,
            config=_config(),
        )
        client.call.assert_not_called()
        self.assertIsInstance(result, CompositeResult)
        self.assertEqual(result.tools_fused, [])
        self.assertEqual(result.tool_weights, {})
        self.assertEqual(result.binding_protein_residues, [])
        self.assertEqual(result.api_usage["status"], "empty")

    def test_single_surviving_tool_skips_llm(self):
        client = MagicMock()
        only = _pred(
            "equipnas", "C",
            binding_protein=[10, 11], per_residue={10: 0.8, 11: 0.4},
        )
        result = fuse_predictions(
            sample_json=_sample_json(),
            target_char=_target_char(),
            tool_predictions=[only, _pred("boltz2", "A", success=False)],
            client=client,
            weight_tensor=None,
            config=_config(),
        )
        client.call.assert_not_called()
        self.assertEqual(result.tools_fused, ["equipnas"])
        self.assertEqual(result.tool_weights, {"equipnas": 1.0})
        # b_hat == b for a single tool with c=1.0; threshold 0.5 keeps 10 only
        # (b=0.8 > 0.5) and drops 11 (b=0.4 < 0.5).
        self.assertEqual(result.binding_protein_residues, [10])
        self.assertEqual(result.api_usage["status"], "single_tool")


# ----------------------- multi-tool LLM happy path -------------------------


class TestLLMHappyPath(unittest.TestCase):
    def test_first_attempt_succeeds(self):
        preds = [
            _pred("boltz2", "A",
                  binding_protein=[10, 11], per_residue={10: 80.0, 11: 70.0}),
            _pred("equipnas", "C",
                  binding_protein=[10, 12], per_residue={10: 0.7, 12: 0.6}),
        ]
        client = MagicMock()
        client.call.return_value = _llm_response(
            _good_assignment_json(["boltz2", "equipnas"]),
            tokens=200,
        )

        result = fuse_predictions(
            sample_json=_sample_json(),
            target_char=_target_char(),
            tool_predictions=preds,
            client=client,
            weight_tensor=None,
            config=_config(),
        )

        self.assertEqual(client.call.call_count, 1)
        self.assertEqual(result.tools_fused, ["boltz2", "equipnas"])
        self.assertEqual(set(result.tool_weights), {"boltz2", "equipnas"})
        self.assertEqual(result.api_usage["status"], "ok")
        self.assertEqual(result.api_usage["retries"], 0)
        # response_format is NOT sent by default (the production
        # LLM endpoint rejects json_object). The system prompt + the
        # client's extract_json_object are enough.
        kwargs = client.call.call_args.kwargs
        self.assertNotIn("response_format", kwargs)

    def test_response_format_passed_when_use_json_mode_true(self):
        # Opt-in path: when an env DOES support json_object, the user
        # flips ``api.use_json_mode: true`` and we should send the param.
        preds = [
            _pred("boltz2", "A",
                  binding_protein=[10, 11], per_residue={10: 80.0, 11: 70.0}),
            _pred("equipnas", "C",
                  binding_protein=[10, 12], per_residue={10: 0.7, 12: 0.6}),
        ]
        client = MagicMock()
        client.call.return_value = _llm_response(
            _good_assignment_json(["boltz2", "equipnas"]), tokens=100,
        )
        cfg = _config()
        cfg["api"]["use_json_mode"] = True

        fuse_predictions(
            sample_json=_sample_json(),
            target_char=_target_char(),
            tool_predictions=preds,
            client=client,
            weight_tensor=None,
            config=cfg,
        )
        kwargs = client.call.call_args.kwargs
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})


# ----------------------- error-correction retry path ----------------------


class TestRetryPath(unittest.TestCase):
    def test_bad_json_then_good_on_retry(self):
        preds = [
            _pred("boltz2", "A",
                  binding_protein=[10], per_residue={10: 80.0}),
            _pred("equipnas", "C",
                  binding_protein=[10], per_residue={10: 0.9}),
        ]
        client = MagicMock()
        client.call.side_effect = [
            _llm_response("this is not JSON at all", tokens=50),
            _llm_response(_good_assignment_json(["boltz2", "equipnas"]),
                          tokens=80),
        ]

        result = fuse_predictions(
            sample_json=_sample_json(),
            target_char=_target_char(),
            tool_predictions=preds,
            client=client,
            weight_tensor=None,
            config=_config(),
        )

        self.assertEqual(client.call.call_count, 2)
        self.assertEqual(result.api_usage["status"], "ok")
        self.assertEqual(result.api_usage["retries"], 1)

    def test_missing_tool_id_in_weights_triggers_retry(self):
        """If LLM omits a tool from weights, parser flags it and retry runs."""
        preds = [
            _pred("boltz2", "A",
                  binding_protein=[10], per_residue={10: 80.0}),
            _pred("equipnas", "C",
                  binding_protein=[10], per_residue={10: 0.9}),
        ]
        bad_assignment = json.dumps({
            "weights": {"boltz2": 0.9},  # missing equipnas!
            "threshold": 0.5,
            "rationale": "I'm only listing one tool which is wrong",
            "confidence": 0.8,
        })
        client = MagicMock()
        client.call.side_effect = [
            _llm_response(bad_assignment, tokens=60),
            _llm_response(_good_assignment_json(["boltz2", "equipnas"]),
                          tokens=80),
        ]

        result = fuse_predictions(
            sample_json=_sample_json(),
            target_char=_target_char(),
            tool_predictions=preds,
            client=client,
            weight_tensor=None,
            config=_config(),
        )
        self.assertEqual(client.call.call_count, 2)
        self.assertEqual(result.api_usage["status"], "ok")
        # Correction prompt must reference the missing tool_id.
        correction_msgs = client.call.call_args_list[1].args[0]
        last_user = correction_msgs[-1]["content"]
        self.assertIn("equipnas", last_user)


# ----------------------- fallback paths ------------------------------------


class TestFallbackPaths(unittest.TestCase):
    def test_persistent_validation_failure_fallback(self):
        preds = [
            _pred("boltz2", "A",
                  binding_protein=[10], per_residue={10: 80.0}),
            _pred("equipnas", "C",
                  binding_protein=[10], per_residue={10: 0.9}),
        ]
        client = MagicMock()
        client.call.side_effect = [
            _llm_response("not json", tokens=30),
            _llm_response("still not json", tokens=30),
        ]

        result = fuse_predictions(
            sample_json=_sample_json(),
            target_char=_target_char(),
            tool_predictions=preds,
            client=client,
            weight_tensor=None,
            config=_config(),
        )
        self.assertEqual(client.call.call_count, 2)
        self.assertEqual(result.api_usage["status"], "fallback")
        self.assertEqual(result.api_usage["retries"], 1)
        # Equal weights at fallback_weight (0.5) for both.
        self.assertEqual(set(result.tool_weights.values()), {0.5})
        # Tokens still summed.
        self.assertEqual(result.api_usage["total_tokens"], 30 * 1.5 * 2)

    def test_api_error_on_first_call_fallback(self):
        preds = [
            _pred("boltz2", "A",
                  binding_protein=[10], per_residue={10: 80.0}),
            _pred("equipnas", "C",
                  binding_protein=[10], per_residue={10: 0.9}),
        ]
        client = MagicMock()
        client.call.side_effect = LLMError("upstream 503")

        result = fuse_predictions(
            sample_json=_sample_json(),
            target_char=_target_char(),
            tool_predictions=preds,
            client=client,
            weight_tensor=None,
            config=_config(),
        )
        # LLMError on first attempt → no retry, fallback immediately.
        self.assertEqual(client.call.call_count, 1)
        self.assertEqual(result.api_usage["status"], "fallback")
        self.assertEqual(result.api_usage["retries"], 0)
        self.assertIn("API error", result.api_usage["failure_reason"])

    def test_api_error_on_retry_fallback(self):
        preds = [
            _pred("boltz2", "A",
                  binding_protein=[10], per_residue={10: 80.0}),
            _pred("equipnas", "C",
                  binding_protein=[10], per_residue={10: 0.9}),
        ]
        client = MagicMock()
        client.call.side_effect = [
            _llm_response("not json", tokens=30),
            LLMError("upstream 500 on retry"),
        ]

        result = fuse_predictions(
            sample_json=_sample_json(),
            target_char=_target_char(),
            tool_predictions=preds,
            client=client,
            weight_tensor=None,
            config=_config(),
        )
        self.assertEqual(client.call.call_count, 2)
        self.assertEqual(result.api_usage["status"], "fallback")
        self.assertEqual(result.api_usage["retries"], 1)

    def test_no_llm_offline_mode(self):
        """client=None → equal-weights fallback without any API call."""
        preds = [
            _pred("boltz2", "A",
                  binding_protein=[10, 11], per_residue={10: 80.0, 11: 70.0}),
            _pred("equipnas", "C",
                  binding_protein=[10], per_residue={10: 0.9}),
        ]
        result = fuse_predictions(
            sample_json=_sample_json(),
            target_char=_target_char(),
            tool_predictions=preds,
            client=None,
            weight_tensor=None,
            config=_config(),
        )
        self.assertEqual(result.api_usage["status"], "fallback")
        self.assertEqual(set(result.tool_weights.values()), {0.5})
        # All surviving tools are listed.
        self.assertEqual(set(result.tools_fused), {"boltz2", "equipnas"})


# ----------------------- best-effort soft inputs ---------------------------


class TestSoftInputs(unittest.TestCase):
    def test_history_get_summary_failure_swallowed(self):
        """A history that raises in get_summary should NOT abort fusion."""
        preds = [
            _pred("boltz2", "A",
                  binding_protein=[10], per_residue={10: 80.0}),
            _pred("equipnas", "C",
                  binding_protein=[10], per_residue={10: 0.9}),
        ]
        client = MagicMock()
        client.call.return_value = _llm_response(
            _good_assignment_json(["boltz2", "equipnas"]),
        )

        bad_history = MagicMock()
        bad_history.get_summary.side_effect = RuntimeError("disk full")

        result = fuse_predictions(
            sample_json=_sample_json(),
            target_char=_target_char(),
            tool_predictions=preds,
            client=client,
            weight_tensor=None,
            config=_config(),
            history=bad_history,
        )
        self.assertEqual(result.api_usage["status"], "ok")
        bad_history.get_summary.assert_called_once()

    def test_weight_tensor_summary_failure_swallowed(self):
        preds = [
            _pred("boltz2", "A",
                  binding_protein=[10], per_residue={10: 80.0}),
            _pred("equipnas", "C",
                  binding_protein=[10], per_residue={10: 0.9}),
        ]
        client = MagicMock()
        client.call.return_value = _llm_response(
            _good_assignment_json(["boltz2", "equipnas"]),
        )
        bad_wt = MagicMock()
        bad_wt.get_category_summary.side_effect = RuntimeError("corrupt tensor")

        result = fuse_predictions(
            sample_json=_sample_json(),
            target_char=_target_char(),
            tool_predictions=preds,
            client=client,
            weight_tensor=bad_wt,
            config=_config(),
        )
        self.assertEqual(result.api_usage["status"], "ok")


# ----------------------- ground-truth comparison ---------------------------


class TestGroundTruthEval(unittest.TestCase):
    def test_perfect_match(self):
        composite = CompositeResult(
            sample_id="s",
            tool_weights={"a": 0.9},
            binding_protein_residues=[10, 11, 23],
            tools_fused=["a"],
        )
        sample_json = _sample_json(gt=[10, 11, 23])
        m = evaluate_against_ground_truth(composite, sample_json)
        self.assertEqual(m["precision"], 1.0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["f1"], 1.0)

    def test_partial_match(self):
        composite = CompositeResult(
            sample_id="s",
            tool_weights={"a": 0.9},
            binding_protein_residues=[10, 11, 99],  # 99 is FP
            tools_fused=["a"],
        )
        sample_json = _sample_json(gt=[10, 11, 23])  # 23 is FN
        m = evaluate_against_ground_truth(composite, sample_json)
        # TP=2, FP=1, FN=1
        self.assertAlmostEqual(m["precision"], 2 / 3, places=4)
        self.assertAlmostEqual(m["recall"], 2 / 3, places=4)
        self.assertAlmostEqual(m["f1"], 2 / 3, places=4)

    def test_empty_predictions_zero_metrics(self):
        composite = CompositeResult(
            sample_id="s",
            tool_weights={},
            binding_protein_residues=[],
            tools_fused=[],
        )
        m = evaluate_against_ground_truth(composite, _sample_json(gt=[10, 11]))
        self.assertEqual(m["precision"], 0.0)
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["f1"], 0.0)

    def test_missing_ground_truth_returns_zeros(self):
        composite = CompositeResult(
            sample_id="s",
            tool_weights={"a": 1.0},
            binding_protein_residues=[10],
            tools_fused=["a"],
        )
        sample_json = {"sample_id": "s"}  # no interaction key
        m = evaluate_against_ground_truth(composite, sample_json)
        self.assertEqual(m["ground_truth_protein"], [])


# ----------------------- output record format ------------------------------


class TestOutputRecord(unittest.TestCase):
    def test_record_matches_spec_keys(self):
        composite = CompositeResult(
            sample_id="1un6_B_F",
            tool_weights={"boltz2": 0.9, "equipnas": 0.7},
            binding_protein_residues=[10, 11, 23],
            binding_rna_nucleotides=[3, 4],
            per_residue_probability={10: 0.92, 11: 0.85, 23: 0.71},
            per_nucleotide_probability={3: 0.8, 4: 0.6},
            threshold=0.5,
            fusion_rationale="ok",
            confidence=0.85,
            tools_fused=["boltz2", "equipnas"],
            api_usage={"prompt_tokens": 1000, "completion_tokens": 200,
                       "total_tokens": 1200, "status": "ok", "retries": 0},
            timestamp="2026-04-30T00:00:00Z",
        )
        sample_json = _sample_json("1un6_B_F", gt=[10, 11, 23])
        record = build_output_record(composite, sample_json, _target_char())

        # Required spec keys.
        for key in (
            "sample_id", "step2_category", "tools_fused", "tool_weights",
            "threshold", "binding_protein_residues",
            "binding_rna_nucleotides", "per_residue_probability",
            "fusion_rationale", "confidence", "ground_truth_protein",
            "precision", "recall", "f1", "api_usage", "timestamp",
        ):
            self.assertIn(key, record, msg=f"missing key {key!r}")

        # JSON-serialisable: integer keys → strings.
        self.assertTrue(all(isinstance(k, str)
                            for k in record["per_residue_probability"]))
        self.assertEqual(record["step2_category"], "RRM_x_stem_loop")
        # Round-trips through json.dumps without error.
        json.dumps(record)


if __name__ == "__main__":
    unittest.main(verbosity=2)
