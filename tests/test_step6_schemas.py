"""Unit tests for step 6 schemas — MetricDetail + PocketQAResult.

Pure schema validation; no scorer / metric calls. Asserts every
validator the schema declares (range, key whitelist, cross-field
consistency between top-level sub-scores and ``details``).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pydantic import ValidationError  # noqa: E402

from step6_pocket_qa.schemas import (  # noqa: E402
    METRIC_NAMES,
    MetricDetail,
    PocketQAResult,
)


# --------------------------- helpers ----------------------------------------


def _md(score=0.7, computed=True, error=None, info=None) -> MetricDetail:
    return MetricDetail(
        score=score, computed=computed, error=error, info=info or {},
    )


def _valid_result(**overrides) -> dict:
    base = {
        "sample_id": "1un6_B_F",
        "structural_plausibility": 0.72,
        "physicochemical_complementarity": 0.65,
        "evolutionary_conservation": 0.58,
        "cross_tool_consensus": 0.85,
        "known_motif_consistency": 0.70,
        "total_score": 0.71,
        "weights_used": {
            "structural_plausibility": 0.25,
            "physicochemical_complementarity": 0.20,
            "evolutionary_conservation": 0.15,
            "cross_tool_consensus": 0.25,
            "known_motif_consistency": 0.15,
        },
        "n_metrics_computed": 5,
        "details": {
            "structural_plausibility": _md(score=0.72),
            "physicochemical_complementarity": _md(score=0.65),
            "evolutionary_conservation": _md(score=0.58),
            "cross_tool_consensus": _md(score=0.85),
            "known_motif_consistency": _md(score=0.70),
        },
        "timestamp": "2026-05-03T00:00:00Z",
    }
    base.update(overrides)
    return base


# --------------------------- MetricDetail -----------------------------------


class TestMetricDetailValid(unittest.TestCase):
    def test_computed_with_score(self):
        d = MetricDetail(score=0.5, computed=True)
        self.assertEqual(d.score, 0.5)
        self.assertTrue(d.computed)
        self.assertIsNone(d.error)
        self.assertEqual(d.info, {})

    def test_default_uncomputed(self):
        d = MetricDetail()
        self.assertIsNone(d.score)
        self.assertFalse(d.computed)

    def test_uncomputed_with_error(self):
        d = MetricDetail(
            score=None, computed=False, error="missing structure",
        )
        self.assertEqual(d.error, "missing structure")

    def test_info_dict_passthrough(self):
        d = MetricDetail(
            score=0.8, computed=True,
            info={"n_clusters": 2, "rg": 9.4},
        )
        self.assertEqual(d.info["n_clusters"], 2)


class TestMetricDetailInvalid(unittest.TestCase):
    def test_score_above_one_rejected(self):
        with self.assertRaises(ValidationError):
            MetricDetail(score=1.5, computed=True)

    def test_score_negative_rejected(self):
        with self.assertRaises(ValidationError):
            MetricDetail(score=-0.1, computed=True)

    def test_computed_true_requires_score(self):
        # computed=True with score=None must fail.
        with self.assertRaises(ValidationError):
            MetricDetail(score=None, computed=True)

    def test_computed_false_with_score_rejected(self):
        # computed=False but score is set — inconsistent.
        with self.assertRaises(ValidationError):
            MetricDetail(score=0.5, computed=False)

    def test_extra_field_forbidden(self):
        with self.assertRaises(ValidationError):
            MetricDetail(score=0.5, computed=True, foo="bar")


# --------------------------- PocketQAResult ---------------------------------


class TestPocketQAResultValid(unittest.TestCase):
    def test_happy_path(self):
        r = PocketQAResult.model_validate(_valid_result())
        self.assertEqual(r.sample_id, "1un6_B_F")
        self.assertEqual(r.n_metrics_computed, 5)
        self.assertEqual(r.total_score, 0.71)
        self.assertEqual(set(r.details), set(METRIC_NAMES))

    def test_all_metrics_none_zero_aggregate(self):
        r = PocketQAResult.model_validate(_valid_result(
            structural_plausibility=None,
            physicochemical_complementarity=None,
            evolutionary_conservation=None,
            cross_tool_consensus=None,
            known_motif_consistency=None,
            total_score=0.0,
            weights_used={},
            n_metrics_computed=0,
            details={
                name: MetricDetail(score=None, computed=False)
                for name in METRIC_NAMES
            },
        ))
        self.assertEqual(r.total_score, 0.0)
        self.assertEqual(r.weights_used, {})
        self.assertEqual(r.n_metrics_computed, 0)

    def test_partial_metrics_subset_weights_used(self):
        # 3 of 5 metrics computed → weights_used has 3 keys; n_metrics_computed=3.
        r = PocketQAResult.model_validate(_valid_result(
            evolutionary_conservation=None,
            known_motif_consistency=None,
            weights_used={
                "structural_plausibility": 0.25,
                "physicochemical_complementarity": 0.20,
                "cross_tool_consensus": 0.25,
            },
            n_metrics_computed=3,
            details={
                "structural_plausibility": _md(score=0.72),
                "physicochemical_complementarity": _md(score=0.65),
                "evolutionary_conservation": MetricDetail(
                    score=None, computed=False, error="no structure"),
                "cross_tool_consensus": _md(score=0.85),
                "known_motif_consistency": MetricDetail(
                    score=None, computed=False),
            },
        ))
        self.assertEqual(r.n_metrics_computed, 3)
        self.assertEqual(set(r.weights_used), {
            "structural_plausibility",
            "physicochemical_complementarity",
            "cross_tool_consensus",
        })

    def test_empty_details_allowed_when_n_zero(self):
        r = PocketQAResult.model_validate(_valid_result(
            structural_plausibility=None,
            physicochemical_complementarity=None,
            evolutionary_conservation=None,
            cross_tool_consensus=None,
            known_motif_consistency=None,
            total_score=0.0,
            weights_used={},
            n_metrics_computed=0,
            details={},
        ))
        self.assertEqual(r.details, {})


class TestPocketQAResultInvalid(unittest.TestCase):
    def test_total_score_above_one_rejected(self):
        with self.assertRaises(ValidationError):
            PocketQAResult.model_validate(_valid_result(total_score=1.2))

    def test_total_score_negative_rejected(self):
        with self.assertRaises(ValidationError):
            PocketQAResult.model_validate(_valid_result(total_score=-0.1))

    def test_sub_score_above_one_rejected(self):
        with self.assertRaises(ValidationError):
            PocketQAResult.model_validate(_valid_result(
                structural_plausibility=1.5,
            ))

    def test_n_metrics_above_five_rejected(self):
        with self.assertRaises(ValidationError):
            PocketQAResult.model_validate(_valid_result(
                n_metrics_computed=6,
            ))

    def test_unknown_weight_key_rejected(self):
        with self.assertRaises(ValidationError):
            PocketQAResult.model_validate(_valid_result(
                weights_used={"nonexistent_metric": 0.5},
                n_metrics_computed=1,
            ))

    def test_weight_above_one_rejected(self):
        with self.assertRaises(ValidationError):
            PocketQAResult.model_validate(_valid_result(
                weights_used={**_valid_result()["weights_used"],
                              "structural_plausibility": 1.2},
            ))

    def test_unknown_details_key_rejected(self):
        # details key not in METRIC_NAMES → reject.
        bad = _valid_result()
        bad["details"]["mystery_metric"] = _md(score=0.5)
        with self.assertRaises(ValidationError):
            PocketQAResult.model_validate(bad)

    def test_top_level_mismatch_with_details(self):
        # top-level says 0.72 but details says 0.91 → cross-field inconsistency.
        bad = _valid_result()
        bad["details"]["structural_plausibility"] = _md(score=0.91)
        with self.assertRaises(ValidationError):
            PocketQAResult.model_validate(bad)

    def test_n_metrics_disagrees_with_weights_used(self):
        # Says 5 computed but weights_used has only 3 keys → reject.
        with self.assertRaises(ValidationError):
            PocketQAResult.model_validate(_valid_result(
                weights_used={
                    "structural_plausibility": 0.25,
                    "physicochemical_complementarity": 0.20,
                    "cross_tool_consensus": 0.25,
                },
                n_metrics_computed=5,
            ))

    def test_extra_field_forbidden(self):
        with self.assertRaises(ValidationError):
            PocketQAResult.model_validate(_valid_result(extra_field="x"))

    def test_empty_sample_id_rejected(self):
        with self.assertRaises(ValidationError):
            PocketQAResult.model_validate(_valid_result(sample_id=""))


# --------------------------- METRIC_NAMES invariant ------------------------


class TestMetricNamesInvariant(unittest.TestCase):
    def test_exactly_five(self):
        self.assertEqual(len(METRIC_NAMES), 5)

    def test_canonical_order(self):
        self.assertEqual(METRIC_NAMES, (
            "structural_plausibility",
            "physicochemical_complementarity",
            "evolutionary_conservation",
            "cross_tool_consensus",
            "known_motif_consistency",
        ))


if __name__ == "__main__":
    unittest.main()
