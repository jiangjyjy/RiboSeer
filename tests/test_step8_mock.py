"""Mock end-to-end tests for step 8 — meta_correction + run.py CLI.

Covers the scenarios called out in the spec:
  1. EMA update actually moves W (sanity reuse of ema_updater)
  2. meta_correction disabled by config → no API call, status='disabled_by_config'
  3. meta_correction enabled but history < min_history → no call,
     status='insufficient_history'
  4. meta_correction enabled, mock LLM returns valid factors → applied
     (and clamped to [0.5, 1.5] by schema)
  5. meta_correction enabled, mock LLM returns out-of-range factors →
     correction-retry once → fallback if still bad
  6. meta_correction enabled, LLMError → status='api_error', factors=None
  7. weight_tensor persistence: save → reload → values preserved
  8. CLI round-trip: write fake step 2-7 JSONL set, run main, validate
     the WeightUpdateResult that comes out

No real LLM constructed; LLMClient is a MagicMock returning canned
chat-completion responses. The EMA pass is real (pure code).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step2_target_char.llm_client import LLMError  # noqa: E402
from step2_target_char.history import PredictionHistory  # noqa: E402
from step3_tool_selection.weight_tensor import METRICS, WeightTensor  # noqa: E402
from step4_tool_adapters.schemas import ToolPrediction  # noqa: E402
from step5_fusion.fusion import build_output_record, fuse_predictions  # noqa: E402
from step5_fusion.schemas import CompositeResult  # noqa: E402
from step6_pocket_qa.scorer import score_prediction  # noqa: E402
from step7_iteration.iterator import run_iteration_loop  # noqa: E402

import step8_weight_update.run as step8_run  # noqa: E402
from step8_weight_update.ema_updater import (  # noqa: E402
    compute_per_tool_scores,
    ema_update,
    slice_for_snapshot,
)
from step8_weight_update.meta_correction import (  # noqa: E402
    MetaCorrectionResult,
    apply_correction_factors,
    meta_correct,
)
from step8_weight_update.schemas import (  # noqa: E402
    META_CORRECTION_MAX,
    META_CORRECTION_MIN,
    WeightUpdateResult,
)


# -------------------- shared fixtures ---------------------------------------


PROTEIN_SEQ = ("A" * 20) + "KGFGFVKF" + ("A" * 12) + "CWMHKKK" + ("A" * 53)
RNA_SEQ = "GCAUGCAUGC" * 3


def _sample_json(sid: str = "mock_sample",
                 category: str = "RRM_x_stem_loop") -> dict:
    return {
        "sample_id": sid,
        "protein": {"chain_id": "A", "sequence": PROTEIN_SEQ,
                    "length": len(PROTEIN_SEQ),
                    "features": {"pI": 9.5}},
        "rna": {"chain_id": "B", "sequence": RNA_SEQ, "length": len(RNA_SEQ)},
        "target_char": {"category": category},
    }


def _pred(tool_id: str, *, cat: str = "C", success: bool = True,
          binding=None, per_residue=None,
          sample_id: str = "mock_sample") -> ToolPrediction:
    if not success:
        return ToolPrediction(
            tool_id=tool_id, category=cat, sample_id=sample_id,
            success=False, error_message=f"{tool_id} mock failure",
        )
    return ToolPrediction(
        tool_id=tool_id, category=cat, sample_id=sample_id, success=True,
        binding_protein_residues=binding,
        per_residue_confidence=per_residue,
    )


def _initial_state(sid: str = "mock_sample"):
    sample = _sample_json(sid)
    preds = [
        _pred("equipnas", cat="C",
              binding=list(range(21, 29)) + [41, 42, 43],
              per_residue={i: 0.85 for i in list(range(21, 29)) + [41, 42, 43]}),
        _pred("p2rank", cat="B",
              binding=list(range(21, 29)) + [41, 42],
              per_residue={i: 0.7 for i in list(range(21, 29)) + [41, 42]}),
    ]
    composite = fuse_predictions(
        sample_json=sample,
        target_char={"category": "RRM_x_stem_loop"},
        tool_predictions=preds,
        client=None, weight_tensor=None,
        config={"fusion": {"default_threshold": 0.5,
                           "fallback_weight": 0.5,
                           "min_tools_for_llm": 2},
                "api": {"temperature": 0.1}},
    )
    qa = score_prediction(
        sample_json=sample, composite_result=composite,
        tool_predictions=preds,
        config={"pocket_qa": {"weights": {n: 0.2 for n in METRICS}}},
    )
    return sample, preds, composite, qa


def _llm_response(content: str, *, tokens: int = 80) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": tokens,
            "completion_tokens": tokens // 2,
            "total_tokens": tokens + tokens // 2,
        },
    }


def _factors_json(factors: dict[str, float],
                  rationale: str = "Recent runs show stable performance; "
                                   "minor calibration only.") -> str:
    return json.dumps({"correction_factors": factors, "rationale": rationale})


def _history_records(n: int, category: str = "RRM_x_stem_loop") -> list[dict]:
    """Synthesise N history records on the same category."""
    return [
        {
            "sample_id": f"sample_{i}",
            "category": category,
            "timestamp": "2026-05-04T00:00:00Z",
            "scores": {n: 0.7 for n in METRICS} | {"total": 0.7},
            "tools_used": ["boltz2", "equipnas"],
            "final_action": "accept",
            "final_score": 0.7,
        }
        for i in range(n)
    ]


# -------------------- 1. EMA still moves W ---------------------------------


class TestEmaIntegration(unittest.TestCase):
    def test_ema_changes_weights_per_sample(self):
        sample, preds, comp, qa = _initial_state()
        per_tool = compute_per_tool_scores(qa, preds, comp,
                                           config={"decompose": {}})
        wt = WeightTensor()
        snap_before = slice_for_snapshot(wt, "RRM_x_stem_loop",
                                         per_tool.keys())
        ema_update(wt, "RRM_x_stem_loop", per_tool, learning_rate=0.5)
        snap_after = slice_for_snapshot(wt, "RRM_x_stem_loop",
                                        per_tool.keys())
        # At least one cell should have moved.
        moved = [
            (t, m)
            for t in snap_before
            for m in snap_before[t]
            if abs(snap_after[t][m] - snap_before[t][m]) > 1e-9
        ]
        self.assertGreater(len(moved), 0,
                           "EMA pass should move at least one cell")


# -------------------- 2/3/4/5/6. meta_correct short circuits + happy path ---


class TestMetaCorrectShortCircuits(unittest.TestCase):
    def test_disabled_by_config_skips_api(self):
        wt = WeightTensor()
        client = MagicMock()
        result = meta_correct(
            weight_tensor=wt, category="RRM_x_stem_loop",
            history=_history_records(20),
            client=client,
            config={"weight_update": {"enable_meta_correction": False}},
        )
        client.call.assert_not_called()
        self.assertIsInstance(result, MetaCorrectionResult)
        self.assertIsNone(result.factors)
        self.assertEqual(result.status, "disabled_by_config")

    def test_insufficient_history_skips_api(self):
        wt = WeightTensor()
        client = MagicMock()
        # 3 records, min_history=5 → skip.
        result = meta_correct(
            weight_tensor=wt, category="RRM_x_stem_loop",
            history=_history_records(3),
            client=client,
            config={"weight_update": {
                "enable_meta_correction": True,
                "meta_correction": {"min_history": 5},
            }},
        )
        client.call.assert_not_called()
        self.assertIsNone(result.factors)
        self.assertEqual(result.status, "insufficient_history")

    def test_history_other_category_does_not_count(self):
        # 10 records but on a different category → still insufficient.
        wt = WeightTensor()
        client = MagicMock()
        result = meta_correct(
            weight_tensor=wt, category="RRM_x_stem_loop",
            history=_history_records(10, category="kh_x_stem_loop"),
            client=client,
            config={"weight_update": {
                "enable_meta_correction": True,
                "meta_correction": {"min_history": 5},
            }},
        )
        client.call.assert_not_called()
        self.assertEqual(result.status, "insufficient_history")

    def test_no_client_returns_no_correction(self):
        wt = WeightTensor()
        result = meta_correct(
            weight_tensor=wt, category="RRM_x_stem_loop",
            history=_history_records(20),
            client=None,
            config={"weight_update": {
                "enable_meta_correction": True,
                "meta_correction": {"min_history": 5},
            }},
        )
        self.assertEqual(result.status, "no_client")
        self.assertIsNone(result.factors)


class TestMetaCorrectHappyPath(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "weight_update": {
                "enable_meta_correction": True,
                "meta_correction": {"min_history": 5,
                                    "min_factor": 0.5, "max_factor": 1.5},
            },
            "api": {"temperature": 0.1},
        }

    def test_valid_factors_returned(self):
        wt = WeightTensor()
        client = MagicMock()
        client.call.return_value = _llm_response(
            _factors_json({"boltz2": 1.2, "equipnas": 0.8,
                           "p2rank": 1.0, "rosettafold2na": 1.05}),
        )
        result = meta_correct(
            weight_tensor=wt, category="RRM_x_stem_loop",
            history=_history_records(20),
            client=client, config=self._config(),
        )
        self.assertEqual(result.status, "ok")
        self.assertIsNotNone(result.factors)
        self.assertAlmostEqual(result.factors["boltz2"], 1.2)
        # All factors in clamp range.
        for g in result.factors.values():
            self.assertGreaterEqual(g, META_CORRECTION_MIN)
            self.assertLessEqual(g, META_CORRECTION_MAX)
        # LLM called once (no retry needed).
        self.assertEqual(client.call.call_count, 1)

    def test_out_of_range_first_then_valid_on_retry(self):
        wt = WeightTensor()
        client = MagicMock()
        bad = _factors_json({"boltz2": 5.0, "equipnas": 0.8})  # 5.0 > max
        good = _factors_json({"boltz2": 1.4, "equipnas": 0.8})
        client.call.side_effect = [
            _llm_response(bad), _llm_response(good),
        ]
        result = meta_correct(
            weight_tensor=wt, category="RRM_x_stem_loop",
            history=_history_records(20),
            client=client, config=self._config(),
        )
        self.assertEqual(client.call.call_count, 2)
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.factors["boltz2"], 1.4)
        self.assertEqual(result.api_usage.get("retries"), 1)

    def test_persistent_bad_json_returns_validation_failed(self):
        wt = WeightTensor()
        client = MagicMock()
        client.call.side_effect = [
            _llm_response("not json"),
            _llm_response("still not json"),
        ]
        result = meta_correct(
            weight_tensor=wt, category="RRM_x_stem_loop",
            history=_history_records(20),
            client=client, config=self._config(),
        )
        self.assertEqual(client.call.call_count, 2)
        self.assertEqual(result.status, "schema_validation_failed")
        self.assertIsNone(result.factors)

    def test_llm_error_returns_api_error(self):
        wt = WeightTensor()
        client = MagicMock()
        client.call.side_effect = LLMError("upstream 503")
        result = meta_correct(
            weight_tensor=wt, category="RRM_x_stem_loop",
            history=_history_records(20),
            client=client, config=self._config(),
        )
        self.assertEqual(result.status, "api_error")
        self.assertIsNone(result.factors)
        self.assertIn("API error", result.api_usage.get("failure_reason", ""))

    def test_all_neutral_factors_treated_as_no_correction(self):
        wt = WeightTensor()
        client = MagicMock()
        client.call.return_value = _llm_response(
            _factors_json({"boltz2": 1.0, "equipnas": 1.0}),
        )
        result = meta_correct(
            weight_tensor=wt, category="RRM_x_stem_loop",
            history=_history_records(20),
            client=client, config=self._config(),
        )
        self.assertEqual(result.status, "all_neutral")
        # No factors applied (caller treats as "no correction").
        self.assertIsNone(result.factors)

    def test_hallucinated_tool_ids_filtered_out(self):
        wt = WeightTensor()
        client = MagicMock()
        # 'magic_tool_2000' is not in the registry.
        client.call.return_value = _llm_response(
            _factors_json({
                "boltz2": 1.2,
                "magic_tool_2000": 0.5,  # hallucinated → filtered
            }),
        )
        result = meta_correct(
            weight_tensor=wt, category="RRM_x_stem_loop",
            history=_history_records(20),
            client=client, config=self._config(),
        )
        self.assertEqual(result.status, "ok")
        self.assertIn("boltz2", result.factors)
        self.assertNotIn("magic_tool_2000", result.factors)


# -------------------- apply_correction_factors -----------------------------


class TestApplyCorrectionFactors(unittest.TestCase):
    def test_multiplies_existing_cells_only(self):
        wt = WeightTensor()
        # Write one cell explicitly so it exists.
        wt.update("equipnas", "RRM_x_stem_loop",
                  "structural_plausibility", 0.6)
        # Boltz2 has no explicit value → cold-start default; γ should NOT
        # apply (would mass-move untouched cells).
        snap_after = apply_correction_factors(
            weight_tensor=wt, category="RRM_x_stem_loop",
            factors={"equipnas": 1.5, "boltz2": 0.5},
        )
        # equipnas q1 should have been multiplied (0.6 × 1.5 = 0.9, clipped).
        self.assertAlmostEqual(
            wt.get_weight("equipnas", "structural_plausibility",
                          "RRM_x_stem_loop"),
            0.9, places=6,
        )
        # boltz2 q1 should still be the cold-start default (0.7 for Cat A).
        self.assertAlmostEqual(
            wt.get_weight("boltz2", "structural_plausibility",
                          "RRM_x_stem_loop"),
            0.7, places=6,
        )
        # Snapshot includes both even though boltz2 wasn't mutated.
        self.assertIn("boltz2", snap_after)

    def test_clipping_keeps_weights_in_unit_interval(self):
        wt = WeightTensor()
        wt.update("equipnas", "j", "structural_plausibility", 0.9)
        apply_correction_factors(
            weight_tensor=wt, category="j",
            factors={"equipnas": META_CORRECTION_MAX},  # 0.9 × 1.5 = 1.35 → clip 1.0
        )
        self.assertAlmostEqual(
            wt.get_weight("equipnas", "structural_plausibility", "j"),
            1.0, places=6,
        )

    def test_out_of_range_factor_silently_skipped(self):
        # apply_correction_factors does its own range check (defensive,
        # in case the caller forgot schema validation).
        wt = WeightTensor()
        wt.update("equipnas", "j", "structural_plausibility", 0.5)
        apply_correction_factors(
            weight_tensor=wt, category="j",
            factors={"equipnas": 5.0},  # outside clamp → skipped
        )
        # Value unchanged.
        self.assertAlmostEqual(
            wt.get_weight("equipnas", "structural_plausibility", "j"),
            0.5, places=6,
        )


# -------------------- 7. weight_tensor persistence --------------------------


class TestPersistence(unittest.TestCase):
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tensor.json"
            wt = WeightTensor()
            ema_update(wt, "j",
                       {"equipnas": {"structural_plausibility": 1.0}},
                       learning_rate=0.5)
            wt.save(path)

            wt2 = WeightTensor.load(path)
            self.assertAlmostEqual(
                wt2.get_weight("equipnas", "structural_plausibility", "j"),
                wt.get_weight("equipnas", "structural_plausibility", "j"),
                places=6,
            )
            self.assertEqual(wt2.get_count("equipnas", "j"), 1)

    def test_load_missing_file_returns_cold_start(self):
        wt = WeightTensor.load(Path("/nonexistent/tensor.json"))
        # Cat A default = 0.7.
        self.assertAlmostEqual(
            wt.get_weight("boltz2", "structural_plausibility", "j"),
            0.7, places=6,
        )


# -------------------- 8. CLI round-trip ------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestRunCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

        self.processed = self.root / "processed"
        (self.processed / "samples").mkdir(parents=True)
        (self.processed / "samples" / "mock_sample.json").write_text(
            json.dumps(_sample_json()), encoding="utf-8",
        )

        # Build initial step 4/5/6/7 records.
        sample, preds, comp, qa = _initial_state()
        target_char = {"category": "RRM_x_stem_loop"}

        s5_record = build_output_record(comp, sample, target_char)
        self.s5_path = self.root / "step5.jsonl"
        _write_jsonl(self.s5_path, [s5_record])

        s4_record = {
            "sample_id": "mock_sample",
            "tools_run": [p.tool_id for p in preds],
            "predictions": [p.model_dump(mode="json") for p in preds],
            "total_runtime_seconds": 1.0,
            "timestamp": "2026-05-04T00:00:00Z",
        }
        self.s4_path = self.root / "step4.jsonl"
        _write_jsonl(self.s4_path, [s4_record])

        self.s6_path = self.root / "step6.jsonl"
        _write_jsonl(self.s6_path, [qa.model_dump(mode="json")])

        # step 7: produce a real IterationResult via run_iteration_loop
        # in --no-llm mode → iter-0 synth accept.
        iter_result = run_iteration_loop(
            sample_json=sample, target_char=target_char,
            tool_predictions=preds, composite_result=comp,
            qa_result=qa, client=None,
            config={"iteration": {"max_iterations": 1,
                                  "convergence_threshold": 0.02,
                                  "mode": "lightweight"},
                    "api": {"temperature": 0.1}},
        )
        self.s7_path = self.root / "step7.jsonl"
        _write_jsonl(self.s7_path, [iter_result.model_dump(mode="json")])

        self.s2_path = self.root / "step2.jsonl"
        _write_jsonl(self.s2_path, [{
            "sample_id": "mock_sample",
            "output": {"category": "RRM_x_stem_loop", "confidence": 0.9},
        }])

        # Configs.
        self.cfg_path = self.root / "step8_config.yaml"
        self.cfg_path.write_text(
            "weight_update:\n"
            "  learning_rate: 0.1\n"
            "  enable_meta_correction: false\n"
            "  meta_correction:\n"
            "    min_factor: 0.5\n"
            "    max_factor: 1.5\n"
            "    min_history: 5\n"
            "  decompose:\n"
            "    cat_a_q1_full: 1.0\n"
            "    other_q1_share: 0.5\n"
            "api:\n"
            "  temperature: 0.1\n"
            "  use_json_mode: false\n",
            encoding="utf-8",
        )

        self.tensor_path = self.root / "weights" / "tensor.json"
        self.history_path = self.root / "history" / "history.jsonl"
        self.out_path = self.root / "step8.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *, extra_argv=None) -> int:
        argv = [
            "--processed-dir", str(self.processed),
            "--step4-output", str(self.s4_path),
            "--step5-output", str(self.s5_path),
            "--step6-output", str(self.s6_path),
            "--step7-output", str(self.s7_path),
            "--step2-output", str(self.s2_path),
            "--weight-tensor", str(self.tensor_path),
            "--history", str(self.history_path),
            "--config", str(self.cfg_path),
            "--output", str(self.out_path),
        ]
        if extra_argv:
            argv += extra_argv
        return step8_run.main(argv)

    def test_emit_valid_record_and_persist_tensor(self):
        rc = self._run()
        self.assertEqual(rc, 0)
        # WeightUpdateResult JSONL is valid.
        rows = [json.loads(l) for l in self.out_path.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        self.assertEqual(len(rows), 1)
        rec = rows[0]
        result = WeightUpdateResult.model_validate(rec)
        self.assertEqual(result.sample_id, "mock_sample")
        self.assertEqual(result.category, "RRM_x_stem_loop")
        self.assertFalse(result.meta_correction_applied)
        self.assertGreater(len(result.tools_updated), 0)
        # Tensor file written.
        self.assertTrue(self.tensor_path.is_file())
        wt = WeightTensor.load(self.tensor_path)
        # At least one cell should have been incremented.
        self.assertGreaterEqual(
            wt.get_count(result.tools_updated[0], "RRM_x_stem_loop"), 1,
        )

    def test_history_appended(self):
        self._run()
        history = PredictionHistory.load(self.history_path)
        self.assertEqual(len(history), 1)
        rec = history.records[0]
        self.assertEqual(rec["sample_id"], "mock_sample")
        self.assertEqual(rec["category"], "RRM_x_stem_loop")
        self.assertIn("scores", rec)
        self.assertIn("tools_used", rec)

    def test_idempotent_second_run_continues_ema(self):
        self._run()
        # First run already wrote the tensor; second run should LOAD it
        # and continue the EMA from those values.
        wt_before = WeightTensor.load(self.tensor_path)
        self._run()
        wt_after = WeightTensor.load(self.tensor_path)
        # Counter should be 2 after two runs.
        for tid in ("equipnas", "p2rank"):
            self.assertEqual(
                wt_after.get_count(tid, "RRM_x_stem_loop"),
                wt_before.get_count(tid, "RRM_x_stem_loop") + 1,
                f"counter for {tid} should bump by 1 on second run",
            )

    def test_skip_when_step7_missing(self):
        # Drop the step 7 record → no sample ids resolved → exit 1.
        self.s7_path.write_text("", encoding="utf-8")
        rc = self._run()
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
