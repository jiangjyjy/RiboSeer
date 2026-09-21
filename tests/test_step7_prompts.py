"""Unit tests for step 7 prompt builders.

Covers:
  - build_messages structure (system + user, correct roles, both non-empty)
  - prompt budget (< 2000 tokens for typical input, < 3000 with long history)
  - SYSTEM_PROMPT contains the action menu, per-metric guidance, schema rules
  - tool_ids are injected from the live registry
  - summarize_qa renders all 5 sub-scores; abstained metrics show reason
  - summarize_qa accepts both Pydantic model and dict inputs
  - summarize_history renders empty / single / multi-iteration cases
  - summarize_prediction handles missing weights / threshold gracefully
  - build_error_correction_messages preserves prior turns + appends 2 messages
  - estimate_prompt_tokens is monotonic in content size

No LLM client is constructed; all data is synthetic.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step5_fusion.schemas import CompositeResult  # noqa: E402
from step6_pocket_qa.schemas import (  # noqa: E402
    METRIC_NAMES,
    MetricDetail,
    PocketQAResult,
)
from step7_iteration.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    build_error_correction_messages,
    build_messages,
    estimate_prompt_tokens,
    format_user_prompt,
    summarize_history,
    summarize_prediction,
    summarize_qa,
)
from step7_iteration.schemas import IterationAction, IterationRecord  # noqa: E402


# ---------------------------- fixtures --------------------------------------


def _qa_dict(**overrides) -> dict:
    base = {
        "sample_id": "s1",
        "structural_plausibility": 0.82,
        "physicochemical_complementarity": 0.65,
        "evolutionary_conservation": 0.71,
        "cross_tool_consensus": 0.42,
        "known_motif_consistency": 0.55,
        "total_score": 0.62,
        "n_metrics_computed": 5,
        "weights_used": {n: 0.2 for n in METRIC_NAMES},
        "details": {
            "structural_plausibility": {
                "computed": True, "score": 0.82,
                "info": {"n_clusters": 2, "interface_ratio": 0.21,
                         "rg": 8.4, "expected_rg": 9.3},
            },
            "physicochemical_complementarity": {
                "computed": True, "score": 0.65,
                "info": {"positive_ratio": 0.33, "aromatic_ratio": 0.08,
                         "polar_ratio": 0.25, "gp_ratio": 0.10},
            },
            "evolutionary_conservation": {
                "computed": True, "score": 0.71,
                "info": {"rare_ratio": 0.22, "terminal_share": 0.0,
                         "pI": 8.87},
            },
            "cross_tool_consensus": {
                "computed": True, "score": 0.42,
                "info": {"n_active_tools": 3, "avg_vote_ratio": 0.55,
                         "avg_jaccard": 0.21,
                         "pairwise_jaccard": {
                             "boltz2|p2rank": 0.18,
                             "boltz2|equipnas": 0.32,
                             "p2rank|equipnas": 0.13,
                         }},
            },
            "known_motif_consistency": {
                "computed": True, "score": 0.55,
                "info": {"domain": "RRM", "n_motifs_found": 1,
                         "coverage_ratio": 0.55},
            },
        },
        "timestamp": "2026-05-04T00:00:00Z",
    }
    base.update(overrides)
    return base


def _qa_model(**overrides) -> PocketQAResult:
    return PocketQAResult.model_validate(_qa_dict(**overrides))


def _composite_dict(**overrides) -> dict:
    base = {
        "sample_id": "s1",
        "binding_protein_residues": [10, 11, 12, 23, 24, 25],
        "binding_rna_nucleotides": [3, 4, 5, 6],
        "tool_weights": {"boltz2": 0.7, "p2rank": 0.4, "equipnas": 0.6},
        "threshold": 0.5,
        "confidence": 0.72,
    }
    base.update(overrides)
    return base


def _composite_model(**overrides) -> CompositeResult:
    return CompositeResult.model_validate(_composite_dict(**overrides))


def _history_one_refine(delta: float = 0.01) -> list[IterationRecord]:
    return [
        IterationRecord(
            iteration=0,
            action=IterationAction(
                action="refine",
                refine_tool="equipnas",
                refine_reason="low motif coverage",
                rationale="q5 was the bottleneck on the first pass; "
                          "tuning equipnas threshold should help.",
                confidence=0.7,
            ),
            score_before=0.61,
            score_after=0.61 + delta,
            delta=delta,
            api_usage={"total_tokens": 850},
            timestamp="2026-05-04T00:00:00Z",
        ),
    ]


# ---------------------------- system prompt content -------------------------


class TestSystemPromptContent(unittest.TestCase):
    def test_mentions_three_actions(self):
        for word in ("accept", "refine", "restart"):
            self.assertIn(word, SYSTEM_PROMPT)

    def test_mentions_per_metric_guidance(self):
        # Each q_m's name AND its action-rule should appear at least once.
        for q in ("q1", "q2", "q3", "q4", "q5"):
            self.assertIn(q, SYSTEM_PROMPT, f"{q} guidance missing")
        # Spec says q2/q3 should not be refined — keyword check.
        self.assertIn("DO NOT refine", SYSTEM_PROMPT)

    def test_mentions_history_rules(self):
        self.assertIn("History", SYSTEM_PROMPT)
        # The "don't refine the same tool that already failed" rule is the
        # most important to encode — keyword check.
        self.assertIn("did not improve", SYSTEM_PROMPT)

    def test_mentions_json_output_keys(self):
        for key in ("action", "refine_tool", "refine_reason",
                    "restart_reason", "rationale", "confidence"):
            self.assertIn(key, SYSTEM_PROMPT)

    def test_tool_ids_injected_into_built_prompt(self):
        # The placeholder %(TOOL_IDS_LINE)s should be substituted.
        msgs = build_messages(
            sample_id="s1",
            target_char={"category": "RRM_x_stem_loop"},
            qa_result=_qa_dict(),
            composite_result=_composite_dict(),
        )
        sys_msg = msgs[0]["content"]
        self.assertNotIn("%(TOOL_IDS_LINE)s", sys_msg)
        # Every tool of the library (paper Table 1) is listed, so the
        # refine decision can name any of them.
        for tid in ("boltz2", "chai1", "rosettafold2na", "rfaa", "alphafold3",
                    "p2rank", "fpocket", "deeppocket", "equipnas",
                    "nucleicnet", "graphbind", "rnabindrplus", "bindup",
                    "hdock", "haddock3"):
            self.assertIn(tid, sys_msg)


# ---------------------------- summarize_qa ----------------------------------


class TestSummarizeQa(unittest.TestCase):
    def test_dict_input(self):
        out = summarize_qa(_qa_dict())
        for name in METRIC_NAMES:
            self.assertIn(name, out, f"{name} missing")
        self.assertIn("total_score = 0.620", out)
        self.assertIn("(5/5 metrics computed)", out)

    def test_pydantic_input(self):
        out = summarize_qa(_qa_model())
        for name in METRIC_NAMES:
            self.assertIn(name, out)
        self.assertIn("total_score = 0.620", out)

    def test_abstained_metric_shows_reason(self):
        # consensus abstains: top-level None + computed=False + reason
        qa = _qa_dict(
            cross_tool_consensus=None,
            n_metrics_computed=4,
            weights_used={n: 0.2 for n in METRIC_NAMES if n != "cross_tool_consensus"},
        )
        qa["details"]["cross_tool_consensus"] = {
            "computed": False, "score": None,
            "info": {"reason": "no tools survived the success+non-empty filter"},
        }
        out = summarize_qa(qa)
        self.assertIn("cross_tool_consensus: --", out)
        self.assertIn("abstained", out)
        self.assertIn("no tools survived", out)

    def test_pairwise_jaccard_rendered_compactly(self):
        out = summarize_qa(_qa_dict())
        # The pairwise dict should be there (LLM uses it to pick the
        # disagreeing tool for refine).
        self.assertIn("pairwise_jaccard", out)
        self.assertIn("boltz2|p2rank=0.18", out)

    def test_unknown_info_keys_dropped(self):
        # Whitelist filtering: keys outside _INFO_KEYS_BY_METRIC should
        # not appear (debug noise stays out of the prompt).
        qa = _qa_dict()
        qa["details"]["structural_plausibility"]["info"]["n_binding_input"] = 99
        qa["details"]["structural_plausibility"]["info"]["debug_noise"] = "x"
        out = summarize_qa(qa)
        self.assertNotIn("debug_noise", out)
        self.assertNotIn("n_binding_input", out)
        # But whitelisted keys still appear.
        self.assertIn("n_clusters=2", out)


# ---------------------------- summarize_prediction --------------------------


class TestSummarizePrediction(unittest.TestCase):
    def test_dict_input(self):
        out = summarize_prediction(_composite_dict())
        self.assertIn("binding(protein): 6 residues", out)
        self.assertIn("binding(rna)    : 4 nucleotides", out)
        self.assertIn("threshold (tau) : 0.500", out)
        self.assertIn("boltz2=0.70", out)

    def test_pydantic_input(self):
        out = summarize_prediction(_composite_model())
        self.assertIn("binding(protein): 6 residues", out)

    def test_truncates_long_residue_list(self):
        long_binding = list(range(1, 51))  # 50 residues
        out = summarize_prediction(_composite_dict(
            binding_protein_residues=long_binding,
        ))
        self.assertIn("50 residues", out)
        self.assertIn("...", out)
        # First 20 should be shown.
        self.assertIn("[1, 2, 3", out)

    def test_handles_empty_weights(self):
        out = summarize_prediction(_composite_dict(tool_weights={}))
        self.assertIn("binding(protein):", out)
        self.assertNotIn("tool_weights", out)


# ---------------------------- summarize_history -----------------------------


class TestSummarizeHistory(unittest.TestCase):
    def test_empty_history(self):
        out = summarize_history([])
        self.assertIn("first iteration", out)

    def test_none_history(self):
        out = summarize_history(None)
        self.assertIn("first iteration", out)

    def test_single_refine(self):
        out = summarize_history(_history_one_refine())
        self.assertIn("iter 0: action=refine", out)
        self.assertIn("0.610 -> 0.620", out)
        self.assertIn("delta=+0.010", out)
        self.assertIn("tool=equipnas", out)

    def test_negative_delta_formatted_with_sign(self):
        out = summarize_history(_history_one_refine(delta=-0.05))
        self.assertIn("delta=-0.050", out)

    def test_dict_form_input(self):
        # Iterator might pass dicts (e.g. after JSONL round-trip).
        history = [
            {
                "iteration": 0,
                "action": {
                    "action": "restart",
                    "restart_reason": "multiple low scores",
                    "rationale": "Many sub-scores were too low; switching plan.",
                    "confidence": 0.6,
                },
                "score_before": 0.30,
                "score_after": 0.55,
                "delta": 0.25,
                "api_usage": {},
                "timestamp": "2026-05-04T00:00:00Z",
            },
        ]
        out = summarize_history(history)
        self.assertIn("action=restart", out)
        self.assertIn("multiple low scores", out)

    def test_score_after_none_renders_n_a(self):
        # Iteration that crashed before re-scoring.
        history = [
            IterationRecord(
                iteration=0,
                action=IterationAction(
                    action="accept",
                    rationale="Score was already acceptable; keeping it.",
                    confidence=0.8,
                ),
                score_before=0.7,
                score_after=None,
                delta=None,
                api_usage={},
                timestamp="2026-05-04T00:00:00Z",
            ),
        ]
        out = summarize_history(history)
        self.assertIn("--", out)
        self.assertIn("delta=n/a", out)


# ---------------------------- build_messages --------------------------------


class TestBuildMessages(unittest.TestCase):
    def test_returns_two_role_blocks(self):
        msgs = build_messages(
            sample_id="s1",
            target_char={"category": "RRM_x_stem_loop"},
            qa_result=_qa_dict(),
            composite_result=_composite_dict(),
        )
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertGreater(len(msgs[0]["content"]), 100)
        self.assertGreater(len(msgs[1]["content"]), 50)

    def test_user_block_has_all_sections(self):
        msgs = build_messages(
            sample_id="1un6_B_F",
            target_char={"category": "RRM_x_stem_loop", "confidence": 0.9},
            qa_result=_qa_dict(),
            composite_result=_composite_dict(),
            history=_history_one_refine(),
            iteration_index=1,
            max_iterations=3,
        )
        user = msgs[1]["content"]
        for header in (
            "Sample ID: 1un6_B_F",
            "Iteration: 1 of max 3",
            "# Target",
            "RRM_x_stem_loop",
            "# Current PocketQA scores",
            "# Current prediction",
            "# Iteration history",
            "iter 0: action=refine",
        ):
            self.assertIn(header, user, f"missing: {header}")

    def test_first_iteration_history_placeholder(self):
        msgs = build_messages(
            sample_id="s1",
            target_char={"category": "RRM_x_stem_loop"},
            qa_result=_qa_dict(),
            composite_result=_composite_dict(),
            history=None,
            iteration_index=0,
        )
        self.assertIn("first iteration", msgs[1]["content"])

    def test_missing_target_char_renders_placeholder(self):
        msgs = build_messages(
            sample_id="s1",
            target_char=None,
            qa_result=_qa_dict(),
            composite_result=_composite_dict(),
        )
        self.assertIn("not available", msgs[1]["content"])

    def test_format_user_prompt_matches_build_messages_user(self):
        # Sanity: build_messages just delegates to format_user_prompt.
        kwargs = dict(
            sample_id="s1",
            target_char={"category": "RRM_x_stem_loop"},
            qa_result=_qa_dict(),
            composite_result=_composite_dict(),
            history=_history_one_refine(),
            iteration_index=1,
            max_iterations=3,
        )
        msgs = build_messages(**kwargs)
        body = format_user_prompt(**kwargs)
        self.assertEqual(msgs[1]["content"], body)


# ---------------------------- token budget ----------------------------------


class TestTokenBudget(unittest.TestCase):
    def test_first_iteration_under_2000_tokens(self):
        msgs = build_messages(
            sample_id="1un6_B_F",
            target_char={"category": "RRM_x_stem_loop", "confidence": 0.9},
            qa_result=_qa_dict(),
            composite_result=_composite_dict(),
            history=None,
            iteration_index=0,
        )
        est = estimate_prompt_tokens(msgs)
        self.assertLess(est, 2000, f"first iteration ~{est} tokens "
                        f"exceeds 2000-token budget")

    def test_with_history_under_2000_tokens(self):
        # Three prior refines + history should still fit.
        history = _history_one_refine() * 2 + [
            IterationRecord(
                iteration=2,
                action=IterationAction(
                    action="restart",
                    restart_reason="prior refines did not improve scores",
                    rationale="Two refine attempts in a row failed; "
                              "switching to a fresh tool plan.",
                    confidence=0.6,
                ),
                score_before=0.62,
                score_after=0.63,
                delta=0.01,
                api_usage={},
                timestamp="2026-05-04T00:00:00Z",
            ),
        ]
        # Fix iteration indices so the history block parses cleanly.
        for i, rec in enumerate(history):
            history[i] = rec.model_copy(update={"iteration": i})
        msgs = build_messages(
            sample_id="s1",
            target_char={"category": "RRM_x_stem_loop"},
            qa_result=_qa_dict(),
            composite_result=_composite_dict(),
            history=history,
            iteration_index=3,
        )
        est = estimate_prompt_tokens(msgs)
        self.assertLess(est, 2000, f"long-history prompt ~{est} tokens "
                        f"exceeds 2000-token budget")

    def test_estimate_monotonic_in_content(self):
        small = [{"role": "user", "content": "hi"}]
        big = [{"role": "user", "content": "x" * 4000}]
        self.assertLess(
            estimate_prompt_tokens(small), estimate_prompt_tokens(big),
        )


# ---------------------------- error correction ------------------------------


class TestBuildErrorCorrectionMessages(unittest.TestCase):
    def test_appends_two_turns(self):
        prev = build_messages(
            sample_id="s1",
            target_char={"category": "RRM_x_stem_loop"},
            qa_result=_qa_dict(),
            composite_result=_composite_dict(),
        )
        corrected = build_error_correction_messages(
            prev_messages=prev,
            bad_output='{"action": "frob"}',
            error_message="action 'frob' not in ('accept', 'refine', 'restart')",
        )
        self.assertEqual(len(corrected), len(prev) + 2)
        self.assertEqual(corrected[-2]["role"], "assistant")
        self.assertEqual(corrected[-2]["content"], '{"action": "frob"}')
        self.assertEqual(corrected[-1]["role"], "user")

    def test_correction_contains_validation_error(self):
        msgs = build_error_correction_messages(
            prev_messages=[],
            bad_output="not even json",
            error_message="missing field 'rationale'",
        )
        self.assertIn("missing field 'rationale'", msgs[-1]["content"])
        self.assertIn("action", msgs[-1]["content"])
        self.assertIn("rationale", msgs[-1]["content"])

    def test_does_not_mutate_prev_messages(self):
        prev = [{"role": "system", "content": "x"}]
        before_len = len(prev)
        build_error_correction_messages(
            prev_messages=prev,
            bad_output="bad",
            error_message="whatever",
        )
        self.assertEqual(len(prev), before_len)


# ---------------------------- abstained-input edge cases --------------------


class TestEdgeCases(unittest.TestCase):
    def test_qa_with_zero_metrics_computed(self):
        # Step 6 may emit n=0 when binding is empty.
        qa = _qa_dict(
            structural_plausibility=None,
            physicochemical_complementarity=None,
            evolutionary_conservation=None,
            cross_tool_consensus=None,
            known_motif_consistency=None,
            total_score=0.0,
            n_metrics_computed=0,
            weights_used={},
            details={
                name: {"computed": False, "score": None,
                       "info": {"reason": "nothing to score"}}
                for name in METRIC_NAMES
            },
        )
        out = summarize_qa(qa)
        self.assertIn("(0/5 metrics computed)", out)
        for name in METRIC_NAMES:
            self.assertIn(f"{name}: --", out)

    def test_metric_detail_pydantic_pass_through(self):
        # Build a real PocketQAResult with one abstained metric and pass
        # the model directly — validators on the way out should not
        # mangle the rendering.
        qa_dict = _qa_dict(
            cross_tool_consensus=None,
            n_metrics_computed=4,
            weights_used={n: 0.2 for n in METRIC_NAMES
                          if n != "cross_tool_consensus"},
        )
        qa_dict["details"]["cross_tool_consensus"] = MetricDetail(
            score=None, computed=False,
            info={"reason": "no tools active"},
        ).model_dump()
        qa = PocketQAResult.model_validate(qa_dict)
        out = summarize_qa(qa)
        self.assertIn("cross_tool_consensus: --", out)
        self.assertIn("no tools active", out)


if __name__ == "__main__":
    unittest.main()
