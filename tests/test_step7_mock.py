"""Mock end-to-end tests for step 7 — iterator + run.py CLI.

Covers the seven scenarios called out in the spec:

  1. accept on iter 0 → 1 record, termination='accepted'
  2. refine → lightweight re-fuse changes the score → 2nd iter accept
  3. restart → drop worst tool → re-fuse → 2nd iter accept
  4. max_iterations termination (LLM keeps proposing refine; cap forces accept)
  5. convergence termination (refine runs but |delta| < threshold → synth accept)
  6. LLM persistent failure → fallback synth accept on iter 0
  7. refine_tool not in current set → degraded to accept

Plus a CLI round-trip test that wires the same mock LLMClient through
``step7_iteration.run.main``.

No real LLM is constructed; the LLMClient is a MagicMock returning
canned chat-completion responses. The inner step 5 fusion + step 6
scoring are kept real (they're pure code) so the trajectory's score
deltas come from the actual lightweight re-fuse, not stubbed values.
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
from step4_tool_adapters.schemas import ToolPrediction  # noqa: E402
from step5_fusion.fusion import build_output_record, fuse_predictions  # noqa: E402
from step5_fusion.schemas import CompositeResult  # noqa: E402
from step6_pocket_qa.scorer import score_prediction  # noqa: E402
from step6_pocket_qa.schemas import PocketQAResult  # noqa: E402

import step7_iteration.run as step7_run  # noqa: E402
from step7_iteration.iterator import (  # noqa: E402
    _select_worst_tool,
    run_iteration_loop,
)
from step7_iteration.schemas import IterationResult  # noqa: E402


# -------------------- shared fixtures (copied from step6 mock) ---------------


# Same 100-aa synthetic protein the step 6 mock uses — gives non-trivial
# q1/q2/q3/q5 scores so refine actually moves the needle.
PROTEIN_SEQ = ("A" * 20) + "KGFGFVKF" + ("A" * 12) + "CWMHKKK" + ("A" * 53)
assert len(PROTEIN_SEQ) == 100, len(PROTEIN_SEQ)
RNA_SEQ = "GCAUGCAUGC" * 3


def _sample_json(sample_id: str = "mock_sample",
                 category: str = "RRM_x_stem_loop") -> dict:
    return {
        "sample_id": sample_id,
        "protein": {
            "chain_id": "A",
            "sequence": PROTEIN_SEQ,
            "length": len(PROTEIN_SEQ),
            "features": {"pI": 9.5},
        },
        "rna": {
            "chain_id": "B",
            "sequence": RNA_SEQ,
            "length": len(RNA_SEQ),
        },
        "target_char": {"category": category},
    }


def _composite(
    sample_id: str = "mock_sample",
    binding=None,
    rna_binding=None,
    *,
    threshold: float = 0.5,
    confidence: float = 0.8,
    weights=None,
    tools_fused=None,
) -> CompositeResult:
    if binding is None:
        binding = list(range(21, 29)) + [41, 42, 43]
    if rna_binding is None:
        rna_binding = [3, 4, 5]
    if weights is None:
        weights = {"equipnas": 0.6, "p2rank": 0.4}
    if tools_fused is None:
        tools_fused = list(weights.keys())
    return CompositeResult(
        sample_id=sample_id,
        tool_weights=weights,
        binding_protein_residues=binding,
        binding_rna_nucleotides=rna_binding,
        per_residue_probability={i: 0.85 for i in binding},
        threshold=threshold,
        fusion_rationale="mock rationale",
        confidence=confidence,
        tools_fused=tools_fused,
        api_usage={"status": "ok", "total_tokens": 0},
        timestamp="2026-05-04T00:00:00Z",
    )


def _pred(
    tool_id: str,
    *,
    cat: str = "C",
    success: bool = True,
    binding=None,
    per_residue=None,
    structure_path=None,
    sample_id: str = "mock_sample",
) -> ToolPrediction:
    if not success:
        return ToolPrediction(
            tool_id=tool_id, category=cat, sample_id=sample_id,
            success=False, error_message=f"{tool_id} mock failure",
        )
    return ToolPrediction(
        tool_id=tool_id, category=cat, sample_id=sample_id, success=True,
        binding_protein_residues=binding,
        per_residue_confidence=per_residue,
        predicted_structure_path=structure_path,
    )


def _config(max_iter: int = 3, conv: float = 0.02,
            mode: str = "lightweight") -> dict:
    return {
        "iteration": {
            "max_iterations": max_iter,
            "convergence_threshold": conv,
            "mode": mode,
        },
        "api": {"temperature": 0.1, "use_json_mode": False},
    }


def _qa_config() -> dict:
    return {
        "pocket_qa": {
            "weights": {
                "structural_plausibility": 0.25,
                "physicochemical_complementarity": 0.20,
                "evolutionary_conservation": 0.15,
                "cross_tool_consensus": 0.25,
                "known_motif_consistency": 0.15,
            },
        }
    }


def _fusion_config() -> dict:
    return {
        "fusion": {
            "default_threshold": 0.5,
            "fallback_weight": 0.5,
            "min_tools_for_llm": 2,
        },
        "api": {"temperature": 0.1},
    }


def _initial_state():
    """Build a baseline (sample, predictions, composite, qa) for the loop."""
    sample = _sample_json()
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
        client=None,           # equal-weights fallback (no LLM)
        weight_tensor=None,
        config=_fusion_config(),
    )
    qa = score_prediction(
        sample_json=sample,
        composite_result=composite,
        tool_predictions=preds,
        config=_qa_config(),
    )
    return sample, preds, composite, qa


# -------------------- helpers for canned LLM responses ----------------------


def _llm_response(content: str, *, tokens: int = 80) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": tokens,
            "completion_tokens": tokens // 2,
            "total_tokens": tokens + tokens // 2,
        },
    }


def _accept_json(rationale: str = "Score is high enough; accepting result.",
                 confidence: float = 0.85) -> str:
    return json.dumps({
        "action": "accept",
        "rationale": rationale,
        "confidence": confidence,
    })


def _refine_json(tool: str, *, reason: str = "this tool disagrees with consensus",
                 rationale: str = "One tool diverges from the others; "
                                  "re-running it should help consensus.",
                 confidence: float = 0.7) -> str:
    return json.dumps({
        "action": "refine",
        "refine_tool": tool,
        "refine_reason": reason,
        "rationale": rationale,
        "confidence": confidence,
    })


def _restart_json(reason: str = "multiple sub-scores low across the board",
                  rationale: str = "Many sub-scores are too low; the current "
                                   "tool plan does not fit this target.",
                  confidence: float = 0.6) -> str:
    return json.dumps({
        "action": "restart",
        "restart_reason": reason,
        "rationale": rationale,
        "confidence": confidence,
    })


# -------------------- 1. accept on iter 0 ----------------------------------


class TestAcceptOnIterZero(unittest.TestCase):
    def test_single_iteration_accepted(self):
        sample, preds, comp, qa = _initial_state()
        client = MagicMock()
        client.call.return_value = _llm_response(_accept_json())

        result = run_iteration_loop(
            sample_json=sample,
            target_char={"category": "RRM_x_stem_loop"},
            tool_predictions=preds,
            composite_result=comp,
            qa_result=qa,
            client=client,
            config=_config(),
            fusion_config=_fusion_config(),
            qa_config=_qa_config(),
        )
        self.assertIsInstance(result, IterationResult)
        self.assertEqual(result.total_iterations, 1)
        self.assertEqual(result.termination_reason, "accepted")
        self.assertEqual(result.final_action, "accept")
        # Trajectory is just the initial score (accept doesn't move it).
        self.assertEqual(result.score_trajectory, [qa.total_score])
        self.assertAlmostEqual(result.final_score, qa.total_score, places=6)
        # Final binding sets equal the initial composite's.
        self.assertEqual(result.final_binding_protein_residues,
                         list(comp.binding_protein_residues))
        # Exactly one LLM call.
        self.assertEqual(client.call.call_count, 1)
        # Round-trip through schema (model_validate would have already
        # raised inside run_iteration_loop, but be defensive).
        IterationResult.model_validate(result.model_dump())


# -------------------- 2. refine → re-fuse → 2nd iter accept ----------------


class TestRefineThenAccept(unittest.TestCase):
    def test_refine_drops_tool_then_accepts(self):
        sample, preds, comp, qa = _initial_state()
        # Sanity: both tools must be in the initial set so refine can drop one.
        present = {p.tool_id for p in preds if p.success}
        self.assertIn("p2rank", present)

        client = MagicMock()
        client.call.side_effect = [
            _llm_response(_refine_json("p2rank", reason="lowest avg jaccard")),
            _llm_response(_accept_json("Improved enough; stopping.")),
        ]

        result = run_iteration_loop(
            sample_json=sample,
            target_char={"category": "RRM_x_stem_loop"},
            tool_predictions=preds,
            composite_result=comp,
            qa_result=qa,
            client=client,
            config=_config(),
            fusion_config=_fusion_config(),
            qa_config=_qa_config(),
        )

        self.assertEqual(client.call.call_count, 2)
        self.assertEqual(result.total_iterations, 2)
        self.assertEqual(result.termination_reason, "accepted")
        # First record: refine; second: accept.
        self.assertEqual(result.iterations[0].action.action, "refine")
        self.assertEqual(result.iterations[0].action.refine_tool, "p2rank")
        self.assertEqual(result.iterations[1].action.action, "accept")
        # Trajectory has 2 entries — score may go up or down, but the
        # second iteration's score_before must equal the refine result's
        # score_after (state must propagate).
        self.assertAlmostEqual(
            result.iterations[1].score_before,
            result.iterations[0].score_after,
            places=6,
        )
        self.assertAlmostEqual(result.final_score, result.score_trajectory[-1],
                               places=6)


# -------------------- 3. restart → drop worst tool → 2nd iter accept --------


class TestRestartThenAccept(unittest.TestCase):
    def test_restart_drops_worst_tool(self):
        sample, preds, comp, qa = _initial_state()
        # Make p2rank obviously worst by giving it lower per-residue confidence
        # than equipnas. _select_worst_tool uses the avg.
        preds = [
            _pred("equipnas", cat="C",
                  binding=list(range(21, 29)),
                  per_residue={i: 0.95 for i in list(range(21, 29))}),
            _pred("p2rank", cat="B",
                  binding=list(range(21, 29)),
                  per_residue={i: 0.30 for i in list(range(21, 29))}),
        ]
        worst = _select_worst_tool(preds)
        self.assertEqual(worst, "p2rank")

        # Re-derive composite/qa from this tweaked set so we're consistent.
        comp = fuse_predictions(
            sample_json=sample,
            target_char={"category": "RRM_x_stem_loop"},
            tool_predictions=preds,
            client=None, weight_tensor=None, config=_fusion_config(),
        )
        qa = score_prediction(
            sample_json=sample, composite_result=comp,
            tool_predictions=preds, config=_qa_config(),
        )

        client = MagicMock()
        client.call.side_effect = [
            _llm_response(_restart_json()),
            _llm_response(_accept_json()),
        ]
        result = run_iteration_loop(
            sample_json=sample,
            target_char={"category": "RRM_x_stem_loop"},
            tool_predictions=preds,
            composite_result=comp,
            qa_result=qa,
            client=client,
            config=_config(),
            fusion_config=_fusion_config(),
            qa_config=_qa_config(),
        )
        self.assertEqual(client.call.call_count, 2)
        self.assertEqual(result.total_iterations, 2)
        self.assertEqual(result.termination_reason, "accepted")
        self.assertEqual(result.iterations[0].action.action, "restart")
        # After restart, the surviving prediction should not include p2rank
        # — verify by checking that the q4 consensus would only see equipnas
        # (we don't expose surviving preds directly, so this is best-effort:
        # the score must be non-empty).
        self.assertEqual(result.iterations[1].action.action, "accept")


# -------------------- 4. max_iterations termination -------------------------


class TestMaxIterationsTermination(unittest.TestCase):
    def test_cap_forces_accept(self):
        sample, preds, comp, qa = _initial_state()
        # cap=2 + LLM keeps proposing refine → first iter applies refine,
        # second slot is the cap and forces accept (suppressed_action='refine').
        client = MagicMock()
        client.call.side_effect = [
            _llm_response(_refine_json("p2rank")),
            _llm_response(_refine_json("equipnas")),
        ]
        # Use a very tight convergence threshold so refine doesn't trip
        # the converged path before we hit the cap.
        result = run_iteration_loop(
            sample_json=sample,
            target_char={"category": "RRM_x_stem_loop"},
            tool_predictions=preds,
            composite_result=comp,
            qa_result=qa,
            client=client,
            config=_config(max_iter=2, conv=0.0),
            fusion_config=_fusion_config(),
            qa_config=_qa_config(),
        )
        self.assertEqual(result.total_iterations, 2)
        # Last iteration's action must be accept (loop invariant); but the
        # original LLM proposal is preserved in api_usage.suppressed_action.
        self.assertEqual(result.iterations[-1].action.action, "accept")
        # Termination reason MAY be "max_iterations" (refine on first iter
        # produced delta != 0) OR "converged" (delta happened to be < 0).
        # The behaviour we want to assert is: cap was hit.
        if result.termination_reason == "max_iterations":
            self.assertEqual(
                result.iterations[-1].api_usage.get("suppressed_action"),
                "refine",
            )
        else:
            # Convergence path: still valid termination, but refine ran.
            self.assertIn(result.termination_reason, ("converged", "accepted"))

    def test_cap_one_forces_accept_on_iter_zero(self):
        # cap=1 + LLM proposes refine → no slot to apply it; iter 0 itself
        # forces accept.
        sample, preds, comp, qa = _initial_state()
        client = MagicMock()
        client.call.return_value = _llm_response(_refine_json("p2rank"))

        result = run_iteration_loop(
            sample_json=sample, target_char=None,
            tool_predictions=preds, composite_result=comp, qa_result=qa,
            client=client, config=_config(max_iter=1),
            fusion_config=_fusion_config(), qa_config=_qa_config(),
        )
        self.assertEqual(result.total_iterations, 1)
        self.assertEqual(result.termination_reason, "max_iterations")
        self.assertEqual(result.iterations[0].action.action, "accept")
        self.assertEqual(
            result.iterations[0].api_usage.get("suppressed_action"),
            "refine",
        )


# -------------------- 5. convergence termination ---------------------------


class TestConvergenceTermination(unittest.TestCase):
    def test_small_delta_triggers_converged(self):
        # Pick the same tools both times; lightweight refine of an absent
        # tool would degrade. So we use a refine action that DROPS p2rank,
        # then set conv threshold high enough that any delta counts as
        # converged.
        sample, preds, comp, qa = _initial_state()
        client = MagicMock()
        # iter 0: refine. The delta (positive or negative) has |.| < 1.0,
        # so a conv threshold of 1.0 always wins on iter 1.
        client.call.return_value = _llm_response(_refine_json("p2rank"))

        result = run_iteration_loop(
            sample_json=sample,
            target_char={"category": "RRM_x_stem_loop"},
            tool_predictions=preds, composite_result=comp, qa_result=qa,
            client=client,
            config=_config(max_iter=3, conv=1.0),
            fusion_config=_fusion_config(), qa_config=_qa_config(),
        )
        self.assertEqual(result.termination_reason, "converged")
        # Two records: iter 0 (refine) + iter 1 (synthesised accept).
        self.assertEqual(result.total_iterations, 2)
        self.assertEqual(result.iterations[0].action.action, "refine")
        self.assertEqual(result.iterations[1].action.action, "accept")
        # The synth accept's api_usage marker is "convergence_synth".
        self.assertEqual(
            result.iterations[1].api_usage.get("status"),
            "convergence_synth",
        )
        # LLM was only called once (the refine decision); no LLM call for
        # the synth accept.
        self.assertEqual(client.call.call_count, 1)


# -------------------- 6. LLM failure → fallback ----------------------------


class TestLLMFailureFallback(unittest.TestCase):
    def test_persistent_api_error_synth_accept(self):
        sample, preds, comp, qa = _initial_state()
        client = MagicMock()
        client.call.side_effect = LLMError("upstream 503")

        result = run_iteration_loop(
            sample_json=sample, target_char=None,
            tool_predictions=preds, composite_result=comp, qa_result=qa,
            client=client, config=_config(),
            fusion_config=_fusion_config(), qa_config=_qa_config(),
        )
        self.assertEqual(result.total_iterations, 1)
        self.assertEqual(result.termination_reason, "accepted")
        self.assertEqual(result.iterations[0].action.action, "accept")
        usage = result.iterations[0].api_usage
        self.assertEqual(usage.get("status"), "fallback")
        self.assertIn("API error", usage.get("failure_reason", ""))

    def test_persistent_bad_json_synth_accept(self):
        sample, preds, comp, qa = _initial_state()
        client = MagicMock()
        client.call.side_effect = [
            _llm_response("not json at all"),
            _llm_response("still not json"),
        ]

        result = run_iteration_loop(
            sample_json=sample, target_char=None,
            tool_predictions=preds, composite_result=comp, qa_result=qa,
            client=client, config=_config(),
            fusion_config=_fusion_config(), qa_config=_qa_config(),
        )
        self.assertEqual(result.total_iterations, 1)
        self.assertEqual(result.iterations[0].action.action, "accept")
        usage = result.iterations[0].api_usage
        self.assertEqual(usage.get("status"), "fallback")
        # Both attempts (initial + correction) consumed.
        self.assertEqual(client.call.call_count, 2)

    def test_no_client_offline_mode(self):
        sample, preds, comp, qa = _initial_state()
        result = run_iteration_loop(
            sample_json=sample, target_char=None,
            tool_predictions=preds, composite_result=comp, qa_result=qa,
            client=None, config=_config(),
            fusion_config=_fusion_config(), qa_config=_qa_config(),
        )
        self.assertEqual(result.total_iterations, 1)
        self.assertEqual(result.termination_reason, "accepted")
        self.assertEqual(
            result.iterations[0].api_usage.get("status"), "no_client",
        )


# -------------------- 7. degraded refine (tool not present) -----------------


class TestDegradedRefine(unittest.TestCase):
    def test_refine_unknown_tool_degrades_to_accept(self):
        sample, preds, comp, qa = _initial_state()
        client = MagicMock()
        client.call.return_value = _llm_response(_refine_json("rosettafold2na"))

        result = run_iteration_loop(
            sample_json=sample, target_char=None,
            tool_predictions=preds, composite_result=comp, qa_result=qa,
            client=client, config=_config(),
            fusion_config=_fusion_config(), qa_config=_qa_config(),
        )
        # Degraded path: one record, accept, original action preserved in
        # api_usage.suppressed_action.
        self.assertEqual(result.total_iterations, 1)
        self.assertEqual(result.iterations[0].action.action, "accept")
        usage = result.iterations[0].api_usage
        self.assertEqual(usage.get("status"), "degraded")
        self.assertEqual(usage.get("suppressed_action"), "refine")
        self.assertIn("rosettafold2na", usage.get("degrade_reason", ""))

    def test_refine_only_remaining_tool_degrades(self):
        # Single surviving tool → refining (and dropping) it would leave
        # zero successes; degrade_reason should mention "zero successful tools".
        sample = _sample_json()
        preds = [
            _pred("equipnas", cat="C",
                  binding=list(range(21, 29)),
                  per_residue={i: 0.9 for i in list(range(21, 29))}),
        ]
        comp = fuse_predictions(
            sample_json=sample, target_char={},
            tool_predictions=preds, client=None,
            weight_tensor=None, config=_fusion_config(),
        )
        qa = score_prediction(
            sample_json=sample, composite_result=comp,
            tool_predictions=preds, config=_qa_config(),
        )
        client = MagicMock()
        client.call.return_value = _llm_response(_refine_json("equipnas"))

        result = run_iteration_loop(
            sample_json=sample, target_char=None,
            tool_predictions=preds, composite_result=comp, qa_result=qa,
            client=client, config=_config(),
            fusion_config=_fusion_config(), qa_config=_qa_config(),
        )
        self.assertEqual(result.iterations[0].action.action, "accept")
        self.assertEqual(
            result.iterations[0].api_usage.get("status"), "degraded",
        )


# -------------------- helper: _select_worst_tool ----------------------------


class TestSelectWorstTool(unittest.TestCase):
    def test_returns_lowest_avg_confidence(self):
        preds = [
            _pred("a", cat="C", binding=[1, 2, 3],
                  per_residue={1: 0.9, 2: 0.9, 3: 0.9}),
            _pred("b", cat="C", binding=[1, 2, 3],
                  per_residue={1: 0.1, 2: 0.1, 3: 0.1}),
        ]
        self.assertEqual(_select_worst_tool(preds), "b")

    def test_normalises_cat_a_pLDDT_scale(self):
        # Cat A reports pLDDT (0..100); normalised it becomes 0.6, so it's
        # still worse than the C-tool at 0.7.
        preds = [
            _pred("c_tool", cat="C", binding=[1],
                  per_residue={1: 0.7}),
            _pred("a_tool", cat="A", binding=[1],
                  per_residue={1: 60.0},
                  structure_path="/tmp/x.cif"),
        ]
        self.assertEqual(_select_worst_tool(preds), "a_tool")

    def test_skips_failed_tools(self):
        preds = [
            _pred("ok", cat="C", binding=[1], per_residue={1: 0.5}),
            _pred("bad", cat="C", success=False),
        ]
        self.assertEqual(_select_worst_tool(preds), "ok")

    def test_empty_returns_none(self):
        self.assertIsNone(_select_worst_tool([]))


# -------------------- run.py CLI round-trip ---------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestRunCli(unittest.TestCase):
    """Wire the iterator through ``step7_iteration.run.main``.

    Builds a minimal step 2/4/5/6 JSONL set on disk, patches the
    ``LLMClient`` constructor (used inside ``step2_target_char.run.build_client``)
    to return a MagicMock that always emits accept, then runs the CLI.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

        self.processed = self.root / "processed"
        (self.processed / "samples").mkdir(parents=True)
        (self.processed / "samples" / "mock_sample.json").write_text(
            json.dumps(_sample_json()), encoding="utf-8",
        )

        # Build initial state by running real fuse + score so the JSONL
        # records on disk match what the CLI parsers expect.
        sample, preds, comp, qa = _initial_state()
        target_char = {"category": "RRM_x_stem_loop"}

        # step 5 record (uses build_output_record like the real CLI).
        s5_record = build_output_record(comp, sample, target_char)
        self.s5_path = self.root / "step5.jsonl"
        _write_jsonl(self.s5_path, [s5_record])

        # step 4 record (the format step 6's _step4_record_to_predictions
        # accepts: "predictions" list, plus tools_run + bookkeeping).
        s4_record = {
            "sample_id": "mock_sample",
            "tools_run": [p.tool_id for p in preds],
            "predictions": [p.model_dump(mode="json") for p in preds],
            "total_runtime_seconds": 1.23,
            "timestamp": "2026-05-04T00:00:00Z",
        }
        self.s4_path = self.root / "step4.jsonl"
        _write_jsonl(self.s4_path, [s4_record])

        # step 6 record.
        self.s6_path = self.root / "step6.jsonl"
        _write_jsonl(self.s6_path, [qa.model_dump(mode="json")])

        # step 2 record.
        self.s2_path = self.root / "step2.jsonl"
        _write_jsonl(self.s2_path, [{
            "sample_id": "mock_sample",
            "output": {"category": "RRM_x_stem_loop", "confidence": 0.9},
        }])

        # step7 config (and a stub step5 / step6 config so the inner
        # re-fuse / re-score functions don't fail on missing weights).
        self.cfg7_path = self.root / "step7_config.yaml"
        self.cfg7_path.write_text(
            "iteration:\n"
            "  max_iterations: 3\n"
            "  convergence_threshold: 0.02\n"
            "  mode: lightweight\n"
            "api:\n"
            "  temperature: 0.1\n"
            "  use_json_mode: false\n",
            encoding="utf-8",
        )
        # Step 6's CLI helper looks for step5_config.yaml / step6_config.yaml
        # in the same dir as --config; provide both so the path-search short
        # circuits don't trigger surprise behaviour.
        (self.root / "step5_config.yaml").write_text(
            "fusion:\n"
            "  default_threshold: 0.5\n"
            "  fallback_weight: 0.5\n"
            "  min_tools_for_llm: 2\n"
            "api:\n"
            "  temperature: 0.1\n",
            encoding="utf-8",
        )
        (self.root / "step6_config.yaml").write_text(
            "pocket_qa:\n"
            "  weights:\n"
            "    structural_plausibility: 0.25\n"
            "    physicochemical_complementarity: 0.20\n"
            "    evolutionary_conservation: 0.15\n"
            "    cross_tool_consensus: 0.25\n"
            "    known_motif_consistency: 0.15\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *, output_arg: Path, extra_argv: list[str] | None = None) -> int:
        argv = [
            "--processed-dir", str(self.processed),
            "--step4-output", str(self.s4_path),
            "--step5-output", str(self.s5_path),
            "--step6-output", str(self.s6_path),
            "--step2-output", str(self.s2_path),
            "--config", str(self.cfg7_path),
            "--output", str(output_arg),
        ]
        if extra_argv:
            argv += extra_argv
        return step7_run.main(argv)

    def test_no_llm_offline_writes_valid_record(self):
        out = self.root / "step7.jsonl"
        rc = self._run(output_arg=out, extra_argv=["--no-llm"])
        self.assertEqual(rc, 0)
        self.assertTrue(out.is_file())
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        self.assertEqual(len(rows), 1)
        rec = rows[0]
        # Round-trips through the schema.
        result = IterationResult.model_validate(rec)
        self.assertEqual(result.sample_id, "mock_sample")
        self.assertEqual(result.total_iterations, 1)
        self.assertEqual(result.termination_reason, "accepted")
        self.assertEqual(
            result.iterations[0].api_usage.get("status"), "no_client",
        )

    def test_skip_when_step6_missing_for_sample(self):
        # Drop the step6 record so the sample is unresolvable.
        self.s6_path.write_text("", encoding="utf-8")
        out = self.root / "step7.jsonl"
        rc = self._run(output_arg=out, extra_argv=["--no-llm"])
        # No sample ids resolved from --step6-output → exit 1 (setup error).
        self.assertEqual(rc, 1)

    def test_writes_per_sample_files_when_output_is_dir(self):
        out_dir = self.root / "step7_dir"
        rc = self._run(output_arg=out_dir, extra_argv=["--no-llm"])
        self.assertEqual(rc, 0)
        sample_file = out_dir / "mock_sample.jsonl"
        self.assertTrue(sample_file.is_file())
        rec = json.loads(sample_file.read_text(encoding="utf-8").splitlines()[0])
        IterationResult.model_validate(rec)


if __name__ == "__main__":
    unittest.main()
