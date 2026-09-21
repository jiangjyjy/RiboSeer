"""Unit tests for step 7 schemas — IterationAction / Record / Result.

Pure schema validation; no LLM client, no iterator. Asserts every
validator the schemas declare:
  - action enum
  - per-action required fields (refine_tool, restart_reason)
  - delta consistency with score_before / score_after
  - trajectory length matches total_iterations
  - last iteration's action must be accept
  - residue index uniqueness + 1-based positivity
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pydantic import ValidationError  # noqa: E402

from step7_iteration.schemas import (  # noqa: E402
    ACTION_VALUES,
    TERMINATION_VALUES,
    IterationAction,
    IterationRecord,
    IterationResult,
)


# ---------------------------- helpers ---------------------------------------


def _accept(rationale="Score is high enough; no further refinement needed.",
            confidence=0.85) -> IterationAction:
    return IterationAction(
        action="accept", rationale=rationale, confidence=confidence,
    )


def _refine(tool="equipnas", reason="vote ratio is low for this tool",
            rationale="One tool disagrees with the consensus; re-running it "
                      "should improve consensus.",
            confidence=0.7) -> IterationAction:
    return IterationAction(
        action="refine", refine_tool=tool, refine_reason=reason,
        rationale=rationale, confidence=confidence,
    )


def _restart(reason="Multiple sub-scores low; current tool set is mismatched.",
             rationale="The structural and consensus scores are both poor — "
                       "switching to a different tool family should help.",
             confidence=0.6) -> IterationAction:
    return IterationAction(
        action="restart", restart_reason=reason,
        rationale=rationale, confidence=confidence,
    )


def _record(iter_idx=0, action=None, score_before=0.6, score_after=0.7,
            delta=None) -> IterationRecord:
    if action is None:
        action = _accept()
    if delta is None and score_after is not None:
        delta = score_after - score_before
    return IterationRecord(
        iteration=iter_idx,
        action=action,
        score_before=score_before,
        score_after=score_after,
        delta=delta,
        api_usage={"total_tokens": 100},
        timestamp="2026-05-03T00:00:00Z",
    )


def _valid_result(**overrides) -> dict:
    base = {
        "sample_id": "1un6_B_F",
        "final_action": "accept",
        "total_iterations": 1,
        "final_score": 0.7,
        "score_trajectory": [0.7],
        "iterations": [_record(0, _accept(), score_before=0.7,
                               score_after=0.7, delta=0.0).model_dump()],
        "termination_reason": "accepted",
        "final_binding_protein_residues": [10, 11, 23],
        "final_binding_rna_nucleotides": [5, 6],
        "timestamp": "2026-05-03T00:00:00Z",
    }
    base.update(overrides)
    return base


# ---------------------------- IterationAction -------------------------------


class TestIterationActionValid(unittest.TestCase):
    def test_accept(self):
        a = _accept()
        self.assertEqual(a.action, "accept")
        self.assertIsNone(a.refine_tool)
        self.assertIsNone(a.restart_reason)

    def test_refine_with_tool(self):
        a = _refine()
        self.assertEqual(a.action, "refine")
        self.assertEqual(a.refine_tool, "equipnas")

    def test_restart_with_reason(self):
        a = _restart()
        self.assertEqual(a.action, "restart")
        self.assertIn("Multiple", a.restart_reason)

    def test_action_lowercased(self):
        # LLM may return "ACCEPT" / " accept " — validator normalises.
        a = IterationAction(
            action="  ACCEPT ",
            rationale="Decided to accept the result.",
            confidence=0.9,
        )
        self.assertEqual(a.action, "accept")

    def test_whitespace_optional_fields_become_none(self):
        # The accept-mode validator forbids non-None refine fields.
        # An empty string from the LLM should normalise to None and pass.
        a = IterationAction(
            action="accept",
            refine_tool="   ",
            refine_reason="",
            restart_reason="  ",
            rationale="Decided to accept the result.",
            confidence=0.9,
        )
        self.assertIsNone(a.refine_tool)
        self.assertIsNone(a.refine_reason)
        self.assertIsNone(a.restart_reason)


class TestIterationActionInvalid(unittest.TestCase):
    def test_unknown_action_rejected(self):
        with self.assertRaises(ValidationError):
            IterationAction(
                action="abandon",
                rationale="Doesn't matter — action is bad.",
                confidence=0.5,
            )

    def test_refine_without_tool_rejected(self):
        with self.assertRaises(ValidationError):
            IterationAction(
                action="refine",
                rationale="Want to refine but no tool specified.",
                confidence=0.5,
            )

    def test_restart_without_reason_rejected(self):
        with self.assertRaises(ValidationError):
            IterationAction(
                action="restart",
                rationale="Want to restart but no reason given.",
                confidence=0.5,
            )

    def test_accept_with_refine_tool_rejected(self):
        with self.assertRaises(ValidationError):
            IterationAction(
                action="accept", refine_tool="equipnas",
                rationale="Inconsistent — accept shouldn't carry refine_tool.",
                confidence=0.5,
            )

    def test_refine_with_restart_reason_rejected(self):
        with self.assertRaises(ValidationError):
            IterationAction(
                action="refine", refine_tool="equipnas",
                restart_reason="should not be here",
                rationale="refine action mistakenly carries restart_reason.",
                confidence=0.5,
            )

    def test_restart_with_refine_tool_rejected(self):
        with self.assertRaises(ValidationError):
            IterationAction(
                action="restart", restart_reason="multiple scores low",
                refine_tool="equipnas",
                rationale="restart action mistakenly carries refine_tool.",
                confidence=0.5,
            )

    def test_confidence_above_one_rejected(self):
        with self.assertRaises(ValidationError):
            IterationAction(
                action="accept",
                rationale="Some rationale here for the test.",
                confidence=1.5,
            )

    def test_short_rationale_rejected(self):
        with self.assertRaises(ValidationError):
            IterationAction(
                action="accept", rationale="ok", confidence=0.9,
            )

    def test_whitespace_rationale_rejected(self):
        with self.assertRaises(ValidationError):
            IterationAction(
                action="accept", rationale="          ", confidence=0.9,
            )

    def test_extra_field_forbidden(self):
        with self.assertRaises(ValidationError):
            IterationAction(
                action="accept",
                rationale="Some rationale here for the test.",
                confidence=0.9,
                future_field="x",
            )


# ---------------------------- IterationRecord -------------------------------


class TestIterationRecordValid(unittest.TestCase):
    def test_happy_record(self):
        r = _record(score_before=0.5, score_after=0.62, delta=0.12)
        self.assertEqual(r.iteration, 0)
        self.assertAlmostEqual(r.delta, 0.12)

    def test_negative_delta_allowed(self):
        # A refine can lower the score (and that's exactly what convergence
        # detection cares about); the schema must accept it.
        r = _record(action=_refine(), score_before=0.7, score_after=0.6,
                    delta=-0.1)
        self.assertEqual(r.delta, -0.1)

    def test_score_after_none_skips_consistency(self):
        # If the iteration crashed before re-scoring, score_after / delta
        # are both None and the cross-check is skipped.
        r = IterationRecord(
            iteration=2, action=_accept(),
            score_before=0.5, score_after=None, delta=None,
            api_usage={}, timestamp="2026-05-03T00:00:00Z",
        )
        self.assertIsNone(r.score_after)
        self.assertIsNone(r.delta)


class TestIterationRecordInvalid(unittest.TestCase):
    def test_delta_inconsistent_with_scores(self):
        with self.assertRaises(ValidationError):
            IterationRecord(
                iteration=0, action=_accept(),
                score_before=0.5, score_after=0.6, delta=0.5,
                api_usage={}, timestamp="2026-05-03T00:00:00Z",
            )

    def test_negative_iteration_rejected(self):
        with self.assertRaises(ValidationError):
            IterationRecord(
                iteration=-1, action=_accept(),
                score_before=0.5, score_after=0.5, delta=0.0,
                api_usage={}, timestamp="2026-05-03T00:00:00Z",
            )

    def test_score_above_one_rejected(self):
        with self.assertRaises(ValidationError):
            IterationRecord(
                iteration=0, action=_accept(),
                score_before=1.5, score_after=0.5, delta=-1.0,
                api_usage={}, timestamp="2026-05-03T00:00:00Z",
            )

    def test_extra_field_forbidden(self):
        with self.assertRaises(ValidationError):
            IterationRecord(
                iteration=0, action=_accept(),
                score_before=0.5, score_after=0.5, delta=0.0,
                api_usage={}, timestamp="2026-05-03T00:00:00Z",
                stuff="bad",
            )


# ---------------------------- IterationResult -------------------------------


class TestIterationResultValid(unittest.TestCase):
    def test_single_iteration_accept(self):
        r = IterationResult.model_validate(_valid_result())
        self.assertEqual(r.total_iterations, 1)
        self.assertEqual(r.final_action, "accept")
        self.assertEqual(r.termination_reason, "accepted")

    def test_three_iterations_with_refine(self):
        recs = [
            _record(0, _refine(), score_before=0.5,
                    score_after=0.62, delta=0.12),
            _record(1, _refine(tool="p2rank",
                               reason="re-run for second-pass refinement",
                               rationale="Try the other tool to push score higher."),
                    score_before=0.62, score_after=0.66, delta=0.04),
            _record(2, _accept(rationale="Score is high enough."),
                    score_before=0.66, score_after=0.66, delta=0.0),
        ]
        r = IterationResult.model_validate(_valid_result(
            total_iterations=3,
            final_score=0.66,
            score_trajectory=[0.62, 0.66, 0.66],
            iterations=[rec.model_dump() for rec in recs],
            termination_reason="accepted",
        ))
        self.assertEqual(r.total_iterations, 3)
        self.assertEqual(r.final_score, 0.66)
        self.assertEqual(len(r.iterations), 3)

    def test_residues_sorted_and_uniqued(self):
        r = IterationResult.model_validate(_valid_result(
            final_binding_protein_residues=[23, 10, 11],
            final_binding_rna_nucleotides=[6, 5],
        ))
        self.assertEqual(r.final_binding_protein_residues, [10, 11, 23])
        self.assertEqual(r.final_binding_rna_nucleotides, [5, 6])


class TestIterationResultInvalid(unittest.TestCase):
    def test_final_action_must_be_accept(self):
        with self.assertRaises(ValidationError):
            IterationResult.model_validate(_valid_result(final_action="refine"))

    def test_unknown_termination_reason_rejected(self):
        with self.assertRaises(ValidationError):
            IterationResult.model_validate(_valid_result(
                termination_reason="cancelled",
            ))

    def test_trajectory_length_mismatch_rejected(self):
        # 2 trajectory entries but total_iterations=1 — inconsistent.
        with self.assertRaises(ValidationError):
            IterationResult.model_validate(_valid_result(
                score_trajectory=[0.5, 0.7],
            ))

    def test_iterations_length_mismatch_rejected(self):
        recs = [
            _record(0, _accept(), score_before=0.7, score_after=0.7, delta=0.0).model_dump(),
            _record(1, _accept(), score_before=0.7, score_after=0.7, delta=0.0).model_dump(),
        ]
        with self.assertRaises(ValidationError):
            IterationResult.model_validate(_valid_result(
                total_iterations=1,
                iterations=recs,
                score_trajectory=[0.7],
            ))

    def test_iteration_indices_must_be_sequential(self):
        recs = [
            _record(0, _refine(), score_before=0.5, score_after=0.6,
                    delta=0.1).model_dump(),
            # second record has iteration=2 (skipped 1)
            _record(2, _accept(), score_before=0.6, score_after=0.6,
                    delta=0.0).model_dump(),
        ]
        with self.assertRaises(ValidationError):
            IterationResult.model_validate(_valid_result(
                total_iterations=2,
                score_trajectory=[0.6, 0.6],
                iterations=recs,
                final_score=0.6,
            ))

    def test_last_iteration_must_be_accept(self):
        recs = [
            _record(0, _refine(), score_before=0.5, score_after=0.55,
                    delta=0.05).model_dump(),
        ]
        with self.assertRaises(ValidationError):
            IterationResult.model_validate(_valid_result(
                total_iterations=1,
                score_trajectory=[0.55],
                iterations=recs,
                final_score=0.55,
            ))

    def test_final_score_must_match_last_trajectory(self):
        # final_score=0.7 but trajectory ends at 0.5 → reject.
        recs = [
            _record(0, _accept(), score_before=0.5, score_after=0.5,
                    delta=0.0).model_dump(),
        ]
        with self.assertRaises(ValidationError):
            IterationResult.model_validate(_valid_result(
                final_score=0.7,
                score_trajectory=[0.5],
                iterations=recs,
            ))

    def test_residue_index_zero_rejected(self):
        with self.assertRaises(ValidationError):
            IterationResult.model_validate(_valid_result(
                final_binding_protein_residues=[0, 1, 2],
            ))

    def test_duplicate_residue_rejected(self):
        with self.assertRaises(ValidationError):
            IterationResult.model_validate(_valid_result(
                final_binding_protein_residues=[10, 10, 11],
            ))

    def test_total_iterations_zero_rejected(self):
        with self.assertRaises(ValidationError):
            IterationResult.model_validate(_valid_result(
                total_iterations=0,
                score_trajectory=[],
                iterations=[],
            ))

    def test_final_score_above_one_rejected(self):
        with self.assertRaises(ValidationError):
            IterationResult.model_validate(_valid_result(final_score=1.2))

    def test_extra_field_forbidden(self):
        with self.assertRaises(ValidationError):
            IterationResult.model_validate(_valid_result(extra="bad"))

    def test_empty_sample_id_rejected(self):
        with self.assertRaises(ValidationError):
            IterationResult.model_validate(_valid_result(sample_id=""))


# ---------------------------- enum invariants -------------------------------


class TestEnumInvariants(unittest.TestCase):
    def test_action_values_canonical(self):
        self.assertEqual(ACTION_VALUES, ("accept", "refine", "restart"))

    def test_termination_values_canonical(self):
        self.assertEqual(
            TERMINATION_VALUES,
            ("accepted", "converged", "max_iterations"),
        )


if __name__ == "__main__":
    unittest.main()
