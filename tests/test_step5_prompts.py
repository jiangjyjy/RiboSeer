"""Unit tests for step5 prompts.py — pure-string rendering, no API calls."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step4_tool_adapters.schemas import Pocket, ToolPrediction  # noqa: E402
from step5_fusion.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    build_error_correction_messages,
    build_messages,
    compute_consensus,
    estimate_prompt_tokens,
    format_user_prompt,
    summarize_predictions,
)


# --------------------------- fixtures ---------------------------------------


def _boltz2(success: bool = True, **overrides) -> ToolPrediction:
    base = dict(
        tool_id="boltz2", category="A", sample_id="sample0",
        success=success,
        binding_protein_residues=[10, 11, 12, 23, 24],
        binding_rna_nucleotides=[3, 4, 5],
        per_residue_confidence={10: 88.0, 11: 92.0, 12: 87.0, 23: 75.0, 24: 70.0},
        plddt_mean=82.5, iptm_score=0.71, pae_mean=8.4,
        runtime_seconds=312.0,
    )
    if not success:
        base["binding_protein_residues"] = None
        base["binding_rna_nucleotides"] = None
        base["per_residue_confidence"] = None
        base["plddt_mean"] = None
        base["iptm_score"] = None
        base["pae_mean"] = None
        base["error_message"] = overrides.pop("error_message", "boltz failed")
    base.update(overrides)
    return ToolPrediction(**base)


def _p2rank(**overrides) -> ToolPrediction:
    base = dict(
        tool_id="p2rank", category="B", sample_id="sample0",
        success=True,
        binding_protein_residues=[11, 12, 23, 50, 51, 52],
        per_residue_confidence={11: 0.65, 12: 0.62, 23: 0.55, 50: 0.40, 51: 0.41, 52: 0.39},
        pockets=[
            Pocket(rank=1, score=12.4, residues=[11, 12, 23, 50, 51, 52]),
            Pocket(rank=2, score=6.8, residues=[80, 81, 82]),
        ],
        runtime_seconds=14.0,
    )
    base.update(overrides)
    return ToolPrediction(**base)


def _equipnas(**overrides) -> ToolPrediction:
    base = dict(
        tool_id="equipnas", category="C", sample_id="sample0",
        success=True,
        binding_protein_residues=[10, 11, 23, 24, 25],
        per_residue_confidence={10: 0.85, 11: 0.91, 23: 0.78, 24: 0.66, 25: 0.55},
        runtime_seconds=58.0,
    )
    base.update(overrides)
    return ToolPrediction(**base)


def _target_char() -> dict:
    return {
        "category": "RRM_x_stem_loop",
        "confidence": 0.9,
        "analysis": (
            "RRM domain (~87 aa) with basic pI 8.87 and high Lys/His content "
            "binding a stem-loop RNA (61 nt, paired_frac 0.75)."
        ),
        "notes": None,
    }


# --------------------------- compute_consensus ------------------------------


class TestComputeConsensus(unittest.TestCase):
    def test_protein_side_three_tools(self):
        consensus = compute_consensus([_boltz2(), _p2rank(), _equipnas()], "protein")
        # Residue 11: boltz2 + p2rank + equipnas → 3
        # Residue 23: boltz2 + p2rank + equipnas → 3
        # Residue 10: boltz2 + equipnas        → 2
        # Residue 12: boltz2 + p2rank          → 2
        # Residue 24: boltz2 + equipnas        → 2
        # Residues 25, 50, 51, 52: 1 each
        self.assertEqual(consensus[11], 3)
        self.assertEqual(consensus[23], 3)
        self.assertEqual(consensus[10], 2)
        self.assertEqual(consensus[12], 2)
        self.assertEqual(consensus[24], 2)
        self.assertEqual(consensus[25], 1)
        self.assertEqual(consensus[50], 1)

    def test_rna_side_only_boltz2_predicts(self):
        consensus = compute_consensus([_boltz2(), _p2rank(), _equipnas()], "rna")
        self.assertEqual(consensus, {3: 1, 4: 1, 5: 1})

    def test_failed_tool_skipped(self):
        consensus = compute_consensus(
            [_boltz2(success=False), _equipnas()], "protein",
        )
        # Failed boltz2 contributes nothing.
        self.assertEqual(consensus[10], 1)
        self.assertEqual(consensus[11], 1)
        # Boltz2-only residue 12 should not appear.
        self.assertNotIn(12, consensus)

    def test_invalid_side_rejected(self):
        with self.assertRaises(ValueError):
            compute_consensus([_boltz2()], side="garbage")

    def test_empty_predictions_returns_empty(self):
        self.assertEqual(compute_consensus([], "protein"), {})
        self.assertEqual(compute_consensus([], "rna"), {})


# --------------------------- summarize_predictions --------------------------


class TestSummarizePredictions(unittest.TestCase):
    def test_contains_each_tool_block(self):
        out = summarize_predictions([_boltz2(), _p2rank(), _equipnas()])
        for tid in ("boltz2", "p2rank", "equipnas"):
            self.assertIn(tid, out)

    def test_renders_structure_metrics_for_cat_a(self):
        out = summarize_predictions([_boltz2()])
        self.assertIn("pLDDT=82.5", out)
        self.assertIn("ipTM=0.710", out)
        self.assertIn("pAE=8.40", out)

    def test_avg_unit_label_differs_by_category(self):
        out = summarize_predictions([_boltz2(), _equipnas()])
        # Cat A → "avg pLDDT"
        self.assertIn("avg pLDDT", out)
        # Cat C → "avg prob"
        self.assertIn("avg prob", out)

    def test_pocket_summary_for_cat_b(self):
        out = summarize_predictions([_p2rank()])
        self.assertIn("pockets:", out)
        self.assertIn("#1 score=12.40", out)
        self.assertIn("#2 score=6.80", out)

    def test_pocket_summary_truncates_to_three(self):
        many_pockets = [
            Pocket(rank=i, score=10.0 - i, residues=[i * 10])
            for i in range(1, 7)
        ]
        out = summarize_predictions([_p2rank(pockets=many_pockets)])
        self.assertIn("#1 score=", out)
        self.assertIn("#3 score=", out)
        self.assertIn("(+3 more)", out)

    def test_failed_tool_renders_failed_line(self):
        out = summarize_predictions([_boltz2(success=False, error_message="OOM")])
        self.assertIn("FAILED", out)
        self.assertIn("OOM", out)

    def test_consensus_block_present(self):
        out = summarize_predictions([_boltz2(), _equipnas()])
        self.assertIn("Cross-tool consensus", out)
        self.assertIn("predicted by 2 tool(s)", out)

    def test_empty_predictions(self):
        out = summarize_predictions([])
        self.assertIn("(no predictions provided)", out)
        self.assertIn("Cross-tool consensus", out)
        # No protein / RNA predictions means both sides report nothing.
        self.assertIn("no successful protein-side predictions", out)
        self.assertIn("no successful RNA-side predictions", out)

    def test_long_consensus_list_truncated(self):
        # Build a single tool predicting 50 residues — only 25 should be rendered.
        pred = ToolPrediction(
            tool_id="equipnas", category="C", sample_id="sample0",
            success=True,
            binding_protein_residues=list(range(1, 51)),
            per_residue_confidence={i: 0.5 for i in range(1, 51)},
        )
        out = summarize_predictions([pred])
        self.assertIn("predicted by 1 tool(s): 50 residues", out)
        self.assertIn("(+25 more)", out)


# --------------------------- format_user_prompt -----------------------------


class TestFormatUserPrompt(unittest.TestCase):
    def test_contains_sample_id(self):
        out = format_user_prompt(
            "1un6_B_F", _target_char(), [_boltz2(), _equipnas()],
        )
        self.assertIn("Sample ID: 1un6_B_F", out)

    def test_contains_target_char(self):
        out = format_user_prompt(
            "s", _target_char(), [_boltz2()],
        )
        self.assertIn("RRM_x_stem_loop", out)
        self.assertIn("RRM domain", out)

    def test_required_tool_ids_listed(self):
        out = format_user_prompt(
            "s", _target_char(), [_boltz2(), _p2rank(), _equipnas()],
        )
        self.assertIn("# Required tool_ids in `weights`", out)
        self.assertIn("boltz2", out)
        self.assertIn("p2rank", out)
        self.assertIn("equipnas", out)

    def test_weight_summary_block_optional(self):
        out_with = format_user_prompt(
            "s", _target_char(), [_boltz2()],
            weight_summary="boltz2: avg_weight=0.7",
        )
        self.assertIn("Tool reliability", out_with)
        self.assertIn("avg_weight=0.7", out_with)

        out_without = format_user_prompt(
            "s", _target_char(), [_boltz2()],
        )
        self.assertNotIn("Tool reliability", out_without)

    def test_empty_target_char_handled(self):
        out = format_user_prompt("s", {}, [_boltz2()])
        self.assertIn("(step 2 characterization not available)", out)


# --------------------------- build_messages ---------------------------------


class TestBuildMessages(unittest.TestCase):
    def test_shape(self):
        msgs = build_messages("s", _target_char(), [_boltz2()])
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")

    def test_system_contains_schema_keys(self):
        msgs = build_messages("s", _target_char(), [_boltz2()])
        system = msgs[0]["content"]
        for key in ("weights", "threshold", "rationale", "confidence"):
            self.assertIn(f'"{key}"', system)

    def test_system_contains_noisy_or_formula(self):
        msgs = build_messages("s", _target_char(), [_boltz2()])
        system = msgs[0]["content"]
        self.assertIn("b_hat(i)", system)
        self.assertIn("prod_k", system)

    def test_history_summary_injected(self):
        msgs = build_messages(
            "s", _target_char(), [_boltz2()],
            history_summary="Past 30 fusions average c_boltz=0.78",
        )
        self.assertIn("Historical context", msgs[0]["content"])
        self.assertIn("c_boltz=0.78", msgs[0]["content"])

    def test_no_history_no_leak(self):
        msgs = build_messages("s", _target_char(), [_boltz2()])
        self.assertNotIn("Historical context", msgs[0]["content"])

    def test_token_budget_under_3000(self):
        """system + user under stated 3000-token budget for typical 3-tool input."""
        msgs = build_messages(
            "1un6_B_F",
            _target_char(),
            [_boltz2(), _p2rank(), _equipnas()],
            weight_summary=(
                "Tool reliability weights for category 'RRM_x_stem_loop':\n"
                "  Boltz-2 (boltz2): UCB=1.700, avg_weight=0.700, evaluations=0, category=A\n"
                "  EquiPNAS (equipnas): UCB=1.600, avg_weight=0.600, evaluations=0, category=C\n"
                "  P2Rank (p2rank): UCB=1.500, avg_weight=0.500, evaluations=0, category=B"
            ),
        )
        est = estimate_prompt_tokens(msgs)
        self.assertLess(est, 3000, f"prompt estimated at {est} tokens")

    def test_user_lists_all_tool_ids(self):
        msgs = build_messages(
            "s", _target_char(), [_boltz2(), _p2rank(), _equipnas()],
        )
        user = msgs[1]["content"]
        for tid in ("boltz2", "p2rank", "equipnas"):
            self.assertIn(tid, user)


# --------------------------- error correction -------------------------------


class TestErrorCorrection(unittest.TestCase):
    def test_appends_two_turns(self):
        prev = build_messages("s", _target_char(), [_boltz2()])
        corrected = build_error_correction_messages(
            prev, bad_output='{"weights": "garbage"}',
            error_message="weights must be a dict[str, float]",
        )
        self.assertEqual(len(corrected), len(prev) + 2)
        self.assertEqual(corrected[-2]["role"], "assistant")
        self.assertEqual(corrected[-1]["role"], "user")
        self.assertIn("Validation error", corrected[-1]["content"])
        self.assertIn("must be a dict", corrected[-1]["content"])

    def test_correction_mentions_required_keys(self):
        prev = build_messages("s", _target_char(), [_boltz2()])
        corrected = build_error_correction_messages(
            prev, bad_output="bad", error_message="any err",
        )
        last = corrected[-1]["content"]
        for key in ("weights", "threshold", "rationale", "confidence"):
            self.assertIn(f"`{key}`", last)

    def test_prev_messages_unmutated(self):
        prev = build_messages("s", _target_char(), [_boltz2()])
        prev_len = len(prev)
        _ = build_error_correction_messages(prev, "bad", "err")
        self.assertEqual(len(prev), prev_len)


# --------------------------- system prompt guardrails -----------------------


class TestSystemPromptContent(unittest.TestCase):
    def test_json_only_directive(self):
        self.assertIn("ONLY the JSON object", SYSTEM_PROMPT)

    def test_english_required(self):
        self.assertIn("English", SYSTEM_PROMPT)

    def test_mentions_each_tool_category(self):
        for cat in ("Cat A", "Cat B", "Cat C"):
            self.assertIn(cat, SYSTEM_PROMPT)

    def test_mentions_threshold_default(self):
        # Default 0.5 should be stated.
        self.assertIn("0.5", SYSTEM_PROMPT)

    def test_mentions_novel_fold_caveat(self):
        self.assertIn("novel_fold", SYSTEM_PROMPT)

    def test_failed_tool_handling_documented(self):
        # The system prompt explains what to do for FAILED tools.
        self.assertIn("FAILED", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
