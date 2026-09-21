"""Unit tests for step 8 schemas — WeightUpdateResult.

Covers every validator the schema declares:
  - tools_updated uniqueness + non-empty strings
  - ema_deltas / weights_before / weights_after metric-name validation
  - tools_updated keys must equal ema_deltas keys exactly
  - snapshot must contain every updated tool
  - delta == after - before (within tolerance)
  - meta_correction_applied flag agrees with correction_factors
  - correction factors clamped to [0.5, 1.5]
  - learning_rate ∈ (0, 1]
  - extra fields forbidden
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pydantic import ValidationError  # noqa: E402

from step8_weight_update.schemas import (  # noqa: E402
    META_CORRECTION_MAX,
    META_CORRECTION_MIN,
    METRIC_NAMES,
    WeightUpdateResult,
)


# ---------------------------- helpers ---------------------------------------


def _valid_payload(**overrides) -> dict:
    """A minimal-but-valid WeightUpdateResult dict. Override any field."""
    base = {
        "sample_id": "1un6_B_F",
        "category": "RRM_x_stem_loop",
        "tools_updated": ["boltz2", "equipnas"],
        "ema_deltas": {
            "boltz2": {"structural_plausibility": 0.05},
            "equipnas": {"cross_tool_consensus": -0.02},
        },
        "learning_rate": 0.1,
        "meta_correction_applied": False,
        "correction_factors": None,
        "weights_before": {
            "boltz2":   {"structural_plausibility": 0.70},
            "equipnas": {"cross_tool_consensus":     0.50},
        },
        "weights_after": {
            "boltz2":   {"structural_plausibility": 0.75},
            "equipnas": {"cross_tool_consensus":     0.48},
        },
        "api_usage": {"status": "ok"},
        "timestamp": "2026-05-04T00:00:00Z",
    }
    base.update(overrides)
    return base


# ---------------------------- happy paths -----------------------------------


class TestValidPayloads(unittest.TestCase):
    def test_minimal_valid(self):
        r = WeightUpdateResult.model_validate(_valid_payload())
        self.assertEqual(r.sample_id, "1un6_B_F")
        self.assertEqual(set(r.tools_updated), {"boltz2", "equipnas"})
        self.assertFalse(r.meta_correction_applied)
        self.assertIsNone(r.correction_factors)

    def test_no_tools_updated_is_fine(self):
        # A run where every tool failed → no EMA, no snapshot rows for
        # updated tools. Schema must accept the empty case.
        r = WeightUpdateResult.model_validate(_valid_payload(
            tools_updated=[],
            ema_deltas={},
            weights_before={},
            weights_after={},
        ))
        self.assertEqual(r.tools_updated, [])
        self.assertEqual(r.ema_deltas, {})

    def test_meta_correction_applied(self):
        r = WeightUpdateResult.model_validate(_valid_payload(
            meta_correction_applied=True,
            correction_factors={"boltz2": 1.2, "equipnas": 0.8},
            meta_rationale="Boltz2 has been outperforming on RRM samples; "
                           "boost slightly.",
        ))
        self.assertTrue(r.meta_correction_applied)
        self.assertAlmostEqual(r.correction_factors["boltz2"], 1.2)


# ---------------------------- invalid paths ---------------------------------


class TestInvalidPayloads(unittest.TestCase):
    def test_unknown_metric_in_deltas_rejected(self):
        with self.assertRaises(ValidationError):
            WeightUpdateResult.model_validate(_valid_payload(
                ema_deltas={"boltz2": {"made_up_metric": 0.1}},
            ))

    def test_unknown_metric_in_snapshot_rejected(self):
        with self.assertRaises(ValidationError):
            WeightUpdateResult.model_validate(_valid_payload(
                weights_before={"boltz2": {"made_up_metric": 0.5}},
            ))

    def test_tools_updated_must_match_delta_keys(self):
        # tools_updated lists 3 names but ema_deltas only has 2 → reject.
        with self.assertRaises(ValidationError):
            WeightUpdateResult.model_validate(_valid_payload(
                tools_updated=["boltz2", "equipnas", "p2rank"],
            ))

    def test_snapshot_missing_updated_tool_rejected(self):
        # weights_before has only boltz2; equipnas listed in tools_updated.
        with self.assertRaises(ValidationError):
            WeightUpdateResult.model_validate(_valid_payload(
                weights_before={
                    "boltz2": {"structural_plausibility": 0.7},
                },
            ))

    def test_snapshot_tool_keys_must_match(self):
        with self.assertRaises(ValidationError):
            WeightUpdateResult.model_validate(_valid_payload(
                weights_after={
                    "boltz2":   {"structural_plausibility": 0.75},
                    # missing equipnas
                },
            ))

    def test_snapshot_metric_keys_must_match_per_tool(self):
        with self.assertRaises(ValidationError):
            WeightUpdateResult.model_validate(_valid_payload(
                weights_before={
                    "boltz2": {
                        "structural_plausibility": 0.70,
                        "cross_tool_consensus":    0.50,  # extra metric
                    },
                    "equipnas": {"cross_tool_consensus": 0.50},
                },
            ))

    def test_delta_inconsistent_with_snapshots_rejected(self):
        # snapshot before/after says 0.05 delta, but ema_deltas claims 0.50.
        with self.assertRaises(ValidationError):
            WeightUpdateResult.model_validate(_valid_payload(
                ema_deltas={
                    "boltz2": {"structural_plausibility": 0.50},
                    "equipnas": {"cross_tool_consensus": -0.02},
                },
            ))

    def test_delta_present_but_snapshot_missing_cell(self):
        with self.assertRaises(ValidationError):
            WeightUpdateResult.model_validate(_valid_payload(
                ema_deltas={
                    "boltz2": {
                        "structural_plausibility": 0.05,
                        # No matching cell in before / after.
                        "cross_tool_consensus": -0.01,
                    },
                    "equipnas": {"cross_tool_consensus": -0.02},
                },
            ))

    def test_meta_applied_true_requires_factors(self):
        with self.assertRaises(ValidationError):
            WeightUpdateResult.model_validate(_valid_payload(
                meta_correction_applied=True,
                correction_factors=None,
            ))

    def test_meta_applied_false_forbids_factors(self):
        with self.assertRaises(ValidationError):
            WeightUpdateResult.model_validate(_valid_payload(
                meta_correction_applied=False,
                correction_factors={"boltz2": 1.0},
            ))

    def test_correction_factor_above_max_rejected(self):
        with self.assertRaises(ValidationError):
            WeightUpdateResult.model_validate(_valid_payload(
                meta_correction_applied=True,
                correction_factors={"boltz2": META_CORRECTION_MAX + 0.01},
            ))

    def test_correction_factor_below_min_rejected(self):
        with self.assertRaises(ValidationError):
            WeightUpdateResult.model_validate(_valid_payload(
                meta_correction_applied=True,
                correction_factors={"boltz2": META_CORRECTION_MIN - 0.01},
            ))

    def test_correction_at_clamp_boundary_accepted(self):
        # boundary-inclusive: 0.5 and 1.5 should both pass.
        r = WeightUpdateResult.model_validate(_valid_payload(
            meta_correction_applied=True,
            correction_factors={"boltz2": META_CORRECTION_MIN,
                                "equipnas": META_CORRECTION_MAX},
        ))
        self.assertEqual(r.correction_factors["boltz2"], META_CORRECTION_MIN)

    def test_learning_rate_zero_rejected(self):
        with self.assertRaises(ValidationError):
            WeightUpdateResult.model_validate(_valid_payload(learning_rate=0.0))

    def test_learning_rate_above_one_rejected(self):
        with self.assertRaises(ValidationError):
            WeightUpdateResult.model_validate(_valid_payload(learning_rate=1.5))

    def test_duplicate_tools_updated_rejected(self):
        with self.assertRaises(ValidationError):
            WeightUpdateResult.model_validate(_valid_payload(
                tools_updated=["boltz2", "boltz2"],
                ema_deltas={"boltz2": {"structural_plausibility": 0.05}},
                weights_before={"boltz2": {"structural_plausibility": 0.70}},
                weights_after={"boltz2": {"structural_plausibility": 0.75}},
            ))

    def test_extra_field_forbidden(self):
        with self.assertRaises(ValidationError):
            WeightUpdateResult.model_validate(_valid_payload(extra_stuff="x"))

    def test_empty_sample_id_rejected(self):
        with self.assertRaises(ValidationError):
            WeightUpdateResult.model_validate(_valid_payload(sample_id=""))

    def test_empty_category_rejected(self):
        with self.assertRaises(ValidationError):
            WeightUpdateResult.model_validate(_valid_payload(category=""))


# ---------------------------- enum invariants -------------------------------


class TestEnumInvariants(unittest.TestCase):
    def test_metric_names_match_step3(self):
        # Schema imports from step3 — make sure the canonical 5 are there.
        for n in (
            "structural_plausibility",
            "physicochemical_complementarity",
            "evolutionary_conservation",
            "cross_tool_consensus",
            "known_motif_consistency",
        ):
            self.assertIn(n, METRIC_NAMES)
        self.assertEqual(len(METRIC_NAMES), 5)

    def test_clamp_range_canonical(self):
        self.assertEqual(META_CORRECTION_MIN, 0.5)
        self.assertEqual(META_CORRECTION_MAX, 1.5)


if __name__ == "__main__":
    unittest.main()
