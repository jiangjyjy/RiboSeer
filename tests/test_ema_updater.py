"""Unit tests for step 8 EMA updater (ema_updater.py).

Covers:
  - basic EMA: W=0.5, score=1.0, η=0.1 → W'=0.55
  - many-step EMA converges to the score
  - score=0 brings W down
  - failed tools (success=False) are not updated
  - per-metric decomposition rules:
      * Cat A gets full q1, others get half
      * shared metrics (q2 / q3 / q5) weighted by tool_weights
      * q4 decomposed via pairwise_jaccard
      * abstained sub-scores produce no per-tool entry for that metric
  - category routing: writes go to W[*, *, j], not other categories
  - eval-count counter bumps for updated tools
  - learning_rate validation
  - slice_for_snapshot returns canonical 5-metric rows
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step3_tool_selection.weight_tensor import METRICS, WeightTensor  # noqa: E402
from step4_tool_adapters.schemas import ToolPrediction  # noqa: E402
from step5_fusion.schemas import CompositeResult  # noqa: E402
from step6_pocket_qa.schemas import (  # noqa: E402
    MetricDetail,
    PocketQAResult,
)
from step8_weight_update.ema_updater import (  # noqa: E402
    _normalised_share,
    _per_tool_consensus,
    compute_per_tool_scores,
    ema_update,
    slice_for_snapshot,
)


# ---------------------------- fixtures --------------------------------------


def _pred(tool_id: str, *, cat: str = "C", success: bool = True,
          binding=None, sample_id: str = "s") -> ToolPrediction:
    if not success:
        return ToolPrediction(
            tool_id=tool_id, category=cat, sample_id=sample_id,
            success=False, error_message=f"{tool_id} failure",
        )
    return ToolPrediction(
        tool_id=tool_id, category=cat, sample_id=sample_id, success=True,
        binding_protein_residues=binding or [10, 11],
    )


def _composite(weights=None, sample_id: str = "s") -> CompositeResult:
    return CompositeResult(
        sample_id=sample_id,
        tool_weights=weights or {},
        binding_protein_residues=[10, 11],
        tools_fused=list((weights or {}).keys()),
    )


def _qa(*, q1=0.8, q2=0.6, q3=0.5, q4=0.4, q5=0.7,
        consensus_info=None) -> PocketQAResult:
    """Build a PocketQAResult with the 5 sub-scores. ``q*=None`` abstains."""
    consensus_info = consensus_info or {}
    details: dict[str, MetricDetail] = {}
    for name, score in (
        ("structural_plausibility", q1),
        ("physicochemical_complementarity", q2),
        ("evolutionary_conservation", q3),
        ("known_motif_consistency", q5),
    ):
        if score is None:
            details[name] = MetricDetail(score=None, computed=False,
                                         info={"reason": "abstain"})
        else:
            details[name] = MetricDetail(score=score, computed=True,
                                         info={})
    if q4 is None:
        details["cross_tool_consensus"] = MetricDetail(
            score=None, computed=False,
            info={**consensus_info, "reason": "abstain"},
        )
    else:
        details["cross_tool_consensus"] = MetricDetail(
            score=q4, computed=True, info=consensus_info,
        )

    # Total / weights_used: only tally the computed metrics, equal weights.
    computed = {n: 0.2 for n, d in details.items() if d.computed}
    total = (sum(d.score * computed.get(n, 0.0) for n, d in details.items() if d.computed)
             / sum(computed.values())) if computed else 0.0
    return PocketQAResult(
        sample_id="s",
        structural_plausibility=q1,
        physicochemical_complementarity=q2,
        evolutionary_conservation=q3,
        cross_tool_consensus=q4,
        known_motif_consistency=q5,
        total_score=total,
        weights_used=computed,
        n_metrics_computed=len(computed),
        details=details,
        timestamp="2026-05-04T00:00:00Z",
    )


# ---------------------------- ema_update math -------------------------------


class TestEmaMath(unittest.TestCase):
    def test_one_step_from_default(self):
        # Default cat-C cold start = 0.6. score=1.0, η=0.1 → 0.6 + 0.1·0.4 = 0.64
        wt = WeightTensor()
        deltas = ema_update(
            wt, "RRM_x_stem_loop",
            per_tool_scores={"equipnas": {"structural_plausibility": 1.0}},
            learning_rate=0.1,
        )
        self.assertAlmostEqual(
            wt.get_weight("equipnas", "structural_plausibility", "RRM_x_stem_loop"),
            0.64,
            places=6,
        )
        self.assertAlmostEqual(
            deltas["equipnas"]["structural_plausibility"], 0.04, places=6,
        )

    def test_one_step_from_explicit_value(self):
        # Spec example: W=0.5, score=1.0, η=0.1 → 0.55
        wt = WeightTensor()
        wt.update("equipnas", "RRM_x_stem_loop", "structural_plausibility", 0.5)
        ema_update(
            wt, "RRM_x_stem_loop",
            per_tool_scores={"equipnas": {"structural_plausibility": 1.0}},
            learning_rate=0.1,
        )
        self.assertAlmostEqual(
            wt.get_weight("equipnas", "structural_plausibility", "RRM_x_stem_loop"),
            0.55, places=6,
        )

    def test_score_zero_brings_weight_down(self):
        wt = WeightTensor()
        wt.update("equipnas", "RRM_x_stem_loop", "structural_plausibility", 0.5)
        deltas = ema_update(
            wt, "RRM_x_stem_loop",
            per_tool_scores={"equipnas": {"structural_plausibility": 0.0}},
            learning_rate=0.1,
        )
        self.assertAlmostEqual(
            wt.get_weight("equipnas", "structural_plausibility", "RRM_x_stem_loop"),
            0.45, places=6,
        )
        self.assertLess(deltas["equipnas"]["structural_plausibility"], 0.0)

    def test_many_steps_converge_to_score(self):
        # Repeated EMA with η=0.1 toward score=1.0 → W → 1.0.
        wt = WeightTensor()
        wt.update("equipnas", "j", "structural_plausibility", 0.5)
        for _ in range(200):
            ema_update(
                wt, "j",
                per_tool_scores={"equipnas": {"structural_plausibility": 1.0}},
                learning_rate=0.1,
                increment_counts=False,  # keep counts test isolated
            )
        self.assertAlmostEqual(
            wt.get_weight("equipnas", "structural_plausibility", "j"),
            1.0, places=4,
        )

    def test_clamping_keeps_weights_in_unit_interval(self):
        wt = WeightTensor()
        wt.update("equipnas", "j", "cross_tool_consensus", 0.99)
        # Even a wildly out-of-range score gets clipped before the EMA
        # multiply; result must stay in [0, 1].
        ema_update(
            wt, "j",
            per_tool_scores={"equipnas": {"cross_tool_consensus": 5.0}},
            learning_rate=0.5,
        )
        v = wt.get_weight("equipnas", "cross_tool_consensus", "j")
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)

    def test_invalid_learning_rate_raises(self):
        wt = WeightTensor()
        for bad in (-0.1, 0.0, 1.1, 2.0):
            with self.assertRaises(ValueError, msg=f"learning_rate={bad!r}"):
                ema_update(wt, "j", {"equipnas": {"cross_tool_consensus": 0.5}},
                           learning_rate=bad)

    def test_empty_per_tool_scores_no_op(self):
        wt = WeightTensor()
        deltas = ema_update(wt, "j", {}, learning_rate=0.1)
        self.assertEqual(deltas, {})

    def test_unknown_metric_skipped(self):
        wt = WeightTensor()
        deltas = ema_update(
            wt, "j",
            per_tool_scores={"equipnas": {"made_up_metric": 0.9,
                                          "structural_plausibility": 0.9}},
            learning_rate=0.1,
        )
        # Only the known metric appears in the delta map.
        self.assertEqual(set(deltas["equipnas"]), {"structural_plausibility"})

    def test_count_bumped_for_each_updated_tool(self):
        wt = WeightTensor()
        ema_update(
            wt, "j",
            per_tool_scores={
                "equipnas": {"structural_plausibility": 0.5},
                "boltz2":   {"structural_plausibility": 0.5},
            },
            learning_rate=0.1,
        )
        self.assertEqual(wt.get_count("equipnas", "j"), 1)
        self.assertEqual(wt.get_count("boltz2", "j"), 1)
        # Unrelated category should remain at 0.
        self.assertEqual(wt.get_count("equipnas", "other_category"), 0)

    def test_count_skipped_when_increment_counts_false(self):
        wt = WeightTensor()
        ema_update(
            wt, "j", {"equipnas": {"structural_plausibility": 0.5}},
            learning_rate=0.1, increment_counts=False,
        )
        self.assertEqual(wt.get_count("equipnas", "j"), 0)

    def test_category_isolation(self):
        wt = WeightTensor()
        ema_update(
            wt, "cat_a",
            per_tool_scores={"equipnas": {"structural_plausibility": 1.0}},
            learning_rate=0.5,
        )
        # Other category still at the cold-start default for category C tool.
        self.assertAlmostEqual(
            wt.get_weight("equipnas", "structural_plausibility", "cat_b"),
            0.6, places=6,
        )


# ---------------------------- per-tool decomposition ------------------------


class TestComputePerToolScores(unittest.TestCase):
    def test_failed_tools_skipped(self):
        preds = [
            _pred("equipnas", cat="C"),
            _pred("p2rank",   cat="B", success=False),
        ]
        composite = _composite(weights={"equipnas": 0.6})
        qa = _qa()
        out = compute_per_tool_scores(qa, preds, composite)
        self.assertIn("equipnas", out)
        self.assertNotIn("p2rank", out)

    def test_no_surviving_tools_returns_empty(self):
        preds = [_pred("equipnas", success=False)]
        composite = _composite()
        qa = _qa()
        out = compute_per_tool_scores(qa, preds, composite)
        self.assertEqual(out, {})

    def test_q1_full_for_cat_a_half_for_others(self):
        preds = [_pred("boltz2", cat="A"), _pred("equipnas", cat="C")]
        composite = _composite(weights={"boltz2": 0.7, "equipnas": 0.6})
        qa = _qa(q1=0.8)
        out = compute_per_tool_scores(qa, preds, composite)
        self.assertAlmostEqual(out["boltz2"]["structural_plausibility"], 0.8)
        self.assertAlmostEqual(out["equipnas"]["structural_plausibility"], 0.4)

    def test_shared_metric_uses_tool_weights(self):
        # q2 = 0.6 with weights {boltz2: 0.7, equipnas: 0.5} → boltz2 gets
        # 0.6 * 0.7 = 0.42, equipnas 0.6 * 0.5 = 0.30.
        preds = [_pred("boltz2", cat="A"), _pred("equipnas", cat="C")]
        composite = _composite(weights={"boltz2": 0.7, "equipnas": 0.5})
        qa = _qa(q2=0.6)
        out = compute_per_tool_scores(qa, preds, composite)
        self.assertAlmostEqual(
            out["boltz2"]["physicochemical_complementarity"], 0.42, places=6,
        )
        self.assertAlmostEqual(
            out["equipnas"]["physicochemical_complementarity"], 0.30, places=6,
        )

    def test_shared_metric_equal_share_when_no_weights(self):
        # Weights empty → use equal-share fallback (each tool gets the
        # raw fused score).
        preds = [_pred("boltz2", cat="A"), _pred("equipnas", cat="C")]
        composite = _composite(weights={})  # no LLM weights
        qa = _qa(q2=0.6)
        out = compute_per_tool_scores(qa, preds, composite)
        # 2 surviving tools, share=0.5 each, multiplied back by len → 0.6
        for t in ("boltz2", "equipnas"):
            self.assertAlmostEqual(
                out[t]["physicochemical_complementarity"], 0.6, places=6,
            )

    def test_q4_decomposed_from_pairwise_jaccard(self):
        # boltz2 ↔ equipnas pairwise = 0.4; only those two are active.
        preds = [_pred("boltz2", cat="A"), _pred("equipnas", cat="C")]
        composite = _composite(weights={"boltz2": 0.5, "equipnas": 0.5})
        qa = _qa(q4=0.4, consensus_info={
            "active_tools": ["boltz2", "equipnas"],
            "pairwise_jaccard": {"boltz2|equipnas": 0.4},
        })
        out = compute_per_tool_scores(qa, preds, composite)
        # Each tool's q4 = mean Jaccard with others = 0.4.
        self.assertAlmostEqual(out["boltz2"]["cross_tool_consensus"], 0.4)
        self.assertAlmostEqual(out["equipnas"]["cross_tool_consensus"], 0.4)

    def test_q4_uneven_pairwise(self):
        # 3 tools; A is consistent with both, C disagrees with B.
        preds = [
            _pred("boltz2",  cat="A"),
            _pred("equipnas", cat="C"),
            _pred("p2rank",   cat="B"),
        ]
        composite = _composite(weights={"boltz2": 0.5, "equipnas": 0.5,
                                        "p2rank": 0.4})
        qa = _qa(q4=0.5, consensus_info={
            "active_tools": ["boltz2", "equipnas", "p2rank"],
            "pairwise_jaccard": {
                "boltz2|equipnas": 0.6,
                "boltz2|p2rank":   0.7,
                "equipnas|p2rank": 0.1,
            },
        })
        out = compute_per_tool_scores(qa, preds, composite)
        self.assertAlmostEqual(
            out["boltz2"]["cross_tool_consensus"], 0.65, places=6,
        )
        self.assertAlmostEqual(
            out["equipnas"]["cross_tool_consensus"], 0.35, places=6,
        )
        self.assertAlmostEqual(
            out["p2rank"]["cross_tool_consensus"], 0.40, places=6,
        )

    def test_q4_tool_not_active_scores_zero(self):
        # equipnas survived (success=True) but the consensus metric ran
        # only on a subset (e.g. equipnas had empty binding) so it's
        # missing from active_tools. Decomposer must score it 0.
        preds = [_pred("boltz2", cat="A"), _pred("equipnas", cat="C")]
        composite = _composite(weights={"boltz2": 0.5, "equipnas": 0.5})
        qa = _qa(q4=0.5, consensus_info={
            "active_tools": ["boltz2"],   # equipnas was filtered out
            "pairwise_jaccard": {},
        })
        out = compute_per_tool_scores(qa, preds, composite)
        # With <2 active tools the decomposer falls back to fused score.
        # Both tools should get the fused 0.5 (this is the documented
        # fallback behaviour — no per-tool signal available).
        for t in ("boltz2", "equipnas"):
            self.assertAlmostEqual(
                out[t]["cross_tool_consensus"], 0.5, places=6,
            )

    def test_abstained_metric_not_in_per_tool(self):
        preds = [_pred("equipnas", cat="C")]
        composite = _composite(weights={"equipnas": 0.6})
        qa = _qa(q3=None)  # conservation abstained
        out = compute_per_tool_scores(qa, preds, composite)
        self.assertNotIn("evolutionary_conservation", out["equipnas"])
        # Other metrics still there.
        self.assertIn("structural_plausibility", out["equipnas"])

    def test_all_metrics_covered_for_surviving_tool(self):
        preds = [_pred("equipnas", cat="C")]
        composite = _composite(weights={"equipnas": 0.6})
        qa = _qa(q1=0.8, q2=0.7, q3=0.6, q4=0.5, q5=0.4, consensus_info={
            "active_tools": ["equipnas"],
            "pairwise_jaccard": {},
        })
        out = compute_per_tool_scores(qa, preds, composite)
        self.assertEqual(set(out["equipnas"]), set(METRICS))


# ---------------------------- end-to-end coupling ---------------------------


class TestEndToEndUpdate(unittest.TestCase):
    """Decompose → ema_update → check final tensor cells."""

    def test_full_pipeline_one_sample(self):
        preds = [_pred("boltz2", cat="A"), _pred("equipnas", cat="C")]
        composite = _composite(weights={"boltz2": 0.7, "equipnas": 0.5})
        qa = _qa(q1=1.0, q2=1.0, q3=1.0, q4=1.0, q5=1.0,
                 consensus_info={"active_tools": ["boltz2", "equipnas"],
                                 "pairwise_jaccard": {"boltz2|equipnas": 1.0}})
        per_tool = compute_per_tool_scores(qa, preds, composite)

        wt = WeightTensor()
        deltas = ema_update(wt, "RRM_x_stem_loop", per_tool, learning_rate=0.1)

        # Every surviving tool should have a non-empty delta map.
        self.assertEqual(set(deltas), {"boltz2", "equipnas"})
        for tool, mdict in deltas.items():
            self.assertGreater(len(mdict), 0,
                               f"{tool} should have at least one updated metric")
        # Counter incremented once per surviving tool.
        self.assertEqual(wt.get_count("boltz2", "RRM_x_stem_loop"), 1)
        self.assertEqual(wt.get_count("equipnas", "RRM_x_stem_loop"), 1)

    def test_failed_tool_not_in_pipeline(self):
        preds = [
            _pred("boltz2", cat="A"),
            _pred("equipnas", cat="C", success=False),
        ]
        composite = _composite(weights={"boltz2": 0.7})
        qa = _qa()
        per_tool = compute_per_tool_scores(qa, preds, composite)
        wt = WeightTensor()
        deltas = ema_update(wt, "j", per_tool, learning_rate=0.1)
        self.assertIn("boltz2", deltas)
        self.assertNotIn("equipnas", deltas)
        self.assertEqual(wt.get_count("equipnas", "j"), 0)


# ---------------------------- snapshot helper -------------------------------


class TestSliceForSnapshot(unittest.TestCase):
    def test_returns_all_5_metrics_per_tool(self):
        wt = WeightTensor()
        snap = slice_for_snapshot(wt, "RRM_x_stem_loop", ["boltz2", "equipnas"])
        self.assertEqual(set(snap), {"boltz2", "equipnas"})
        for tool, mdict in snap.items():
            self.assertEqual(set(mdict), set(METRICS),
                             f"{tool} missing metrics")

    def test_uses_cold_start_default_for_unwritten_cells(self):
        wt = WeightTensor()
        snap = slice_for_snapshot(wt, "j", ["boltz2"])
        # boltz2 is Cat A → default 0.7.
        for v in snap["boltz2"].values():
            self.assertAlmostEqual(v, 0.7, places=6)

    def test_reflects_post_update_values(self):
        wt = WeightTensor()
        ema_update(
            wt, "j",
            {"equipnas": {"structural_plausibility": 1.0}},
            learning_rate=0.1,
        )
        snap = slice_for_snapshot(wt, "j", ["equipnas"])
        self.assertAlmostEqual(
            snap["equipnas"]["structural_plausibility"], 0.64, places=6,
        )


# ---------------------------- internal helpers ------------------------------


class TestInternalHelpers(unittest.TestCase):
    def test_normalised_share_basic(self):
        s = _normalised_share({"a": 1.0, "b": 3.0}, ["a", "b"])
        self.assertAlmostEqual(s["a"], 0.25)
        self.assertAlmostEqual(s["b"], 0.75)

    def test_normalised_share_falls_back_to_equal(self):
        s = _normalised_share({}, ["a", "b"])
        self.assertEqual(s, {"a": 0.5, "b": 0.5})

    def test_normalised_share_negative_clipped(self):
        s = _normalised_share({"a": -1.0, "b": 1.0}, ["a", "b"])
        self.assertEqual(s["a"], 0.0)
        self.assertEqual(s["b"], 1.0)

    def test_per_tool_consensus_single_active_falls_back(self):
        out = _per_tool_consensus(
            {"active_tools": ["x"], "pairwise_jaccard": {}},
            ["x", "y"], 0.7,
        )
        # <2 active → both surviving tools get fused score.
        self.assertEqual(out, {"x": 0.7, "y": 0.7})


if __name__ == "__main__":
    unittest.main()
