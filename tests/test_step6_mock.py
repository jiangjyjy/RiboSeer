"""Mock end-to-end tests for step 6 — scorer + run.py.

Covers:
  - happy path: 5 metrics computed, weighted total matches a hand-calc
  - metric abstains (returns ``None`` score) → dropped from aggregate;
    weights renormalise; n_metrics_computed reflects the survivors
  - all metrics abstain → total_score=0.0, weights_used={}, n=0
  - one metric raises → wrapped into MetricDetail.error; others succeed
  - bad metric return shape (non-tuple) trapped as error
  - ``_pick_structure_path`` honours category=A + success
  - ``_aggregate_weighted`` renorm + clamping math
  - run.py CLI: round-trip a mock step4/step5/step2 jsonl set →
    PocketQAResult JSONL output is a valid PocketQAResult and matches
    the in-process score_prediction call

No real LLM, no real structure files. The structural metric is given
``structure_path=None`` everywhere so we don't depend on gemmi being
installed; q1 still scores with the 2-axis (continuity + ratio) path.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step4_tool_adapters.schemas import ToolPrediction  # noqa: E402
from step5_fusion.fusion import build_output_record  # noqa: E402
from step5_fusion.schemas import CompositeResult  # noqa: E402

from step6_pocket_qa.scorer import (  # noqa: E402
    _aggregate_weighted,
    _pick_structure_path,
    _safe_metric_call,
    score_prediction,
)
from step6_pocket_qa.schemas import (  # noqa: E402
    METRIC_NAMES,
    MetricDetail,
    PocketQAResult,
)
import step6_pocket_qa.run as step6_run  # noqa: E402


# ----------------------- fixtures -------------------------------------------


# 100-aa synthetic protein matching the integration-smoke from phase 6.4:
#   A*20 + KGFGFVKF (20..27 RNP1, 1-based 21..28)
#   + A*12 + CWMHKKK (40..46, 1-based 41..47)
#   + A*53
PROTEIN_SEQ = ("A" * 20) + "KGFGFVKF" + ("A" * 12) + "CWMHKKK" + ("A" * 53)
assert len(PROTEIN_SEQ) == 100, len(PROTEIN_SEQ)
RNA_SEQ = "GCAUGCAUGC" * 3  # 30 nt — value not load-bearing


def _sample_json(*, with_pi: bool = True, category: str = "RRM_x_stem_loop") -> dict:
    feats: dict = {}
    if with_pi:
        feats["pI"] = 9.5
    return {
        "sample_id": "mock_sample",
        "protein": {
            "chain_id": "A",
            "sequence": PROTEIN_SEQ,
            "length": len(PROTEIN_SEQ),
            "features": feats,
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
) -> CompositeResult:
    if binding is None:
        binding = list(range(21, 29)) + [41, 42, 43]  # RNP1 + CWM
    if rna_binding is None:
        rna_binding = [3, 4, 5]
    return CompositeResult(
        sample_id=sample_id,
        tool_weights={"equipnas": 0.6, "p2rank": 0.4},
        binding_protein_residues=binding,
        binding_rna_nucleotides=rna_binding,
        per_residue_probability={i: 0.85 for i in binding},
        threshold=threshold,
        fusion_rationale="mock rationale",
        confidence=confidence,
        tools_fused=["equipnas", "p2rank"],
        api_usage={"status": "ok", "total_tokens": 0},
        timestamp="2026-05-03T00:00:00Z",
    )


def _pred(
    tool_id: str,
    *,
    cat: str = "C",
    success: bool = True,
    binding=None,
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
        predicted_structure_path=structure_path,
    )


# ----------------------- score_prediction happy paths -----------------------


class TestScorePredictionHappyPath(unittest.TestCase):
    """5-metric run with two consistent active tools and known motif coverage."""

    def setUp(self):
        self.sample_json = _sample_json()
        self.composite = _composite()
        self.predictions = [
            _pred("equipnas", cat="C", binding=list(range(21, 29)) + [41, 42, 43]),
            _pred("p2rank", cat="B", binding=list(range(21, 29)) + [41, 42]),
        ]
        self.config = {
            "pocket_qa": {
                "weights": {
                    "structural_plausibility": 0.25,
                    "physicochemical_complementarity": 0.20,
                    "evolutionary_conservation": 0.15,
                    "cross_tool_consensus": 0.25,
                    "known_motif_consistency": 0.15,
                },
                "structural": {},
                "physicochemical": {},
                "conservation": {},
                "consensus": {},
                "motif": {},
            }
        }

    def test_returns_validated_PocketQAResult(self):
        result = score_prediction(
            sample_json=self.sample_json,
            composite_result=self.composite,
            tool_predictions=self.predictions,
            config=self.config,
        )
        self.assertIsInstance(result, PocketQAResult)
        self.assertEqual(result.sample_id, "mock_sample")
        # Schema validators ran (model_validate would raise otherwise).
        PocketQAResult.model_validate(result.model_dump())

    def test_all_five_metrics_computed(self):
        result = score_prediction(
            sample_json=self.sample_json,
            composite_result=self.composite,
            tool_predictions=self.predictions,
            config=self.config,
        )
        self.assertEqual(result.n_metrics_computed, 5)
        self.assertEqual(set(result.weights_used), set(METRIC_NAMES))
        for name in METRIC_NAMES:
            d = result.details[name]
            self.assertTrue(d.computed, f"{name} should be computed")
            self.assertIsNotNone(d.score)
            self.assertGreaterEqual(d.score, 0.0)
            self.assertLessEqual(d.score, 1.0)

    def test_total_score_matches_weighted_sum(self):
        result = score_prediction(
            sample_json=self.sample_json,
            composite_result=self.composite,
            tool_predictions=self.predictions,
            config=self.config,
        )
        weights = self.config["pocket_qa"]["weights"]
        expected = sum(
            weights[name] * result.details[name].score
            for name in METRIC_NAMES
        ) / sum(weights[name] for name in METRIC_NAMES)
        self.assertAlmostEqual(result.total_score, expected, places=5)
        self.assertGreaterEqual(result.total_score, 0.0)
        self.assertLessEqual(result.total_score, 1.0)

    def test_top_level_matches_details(self):
        result = score_prediction(
            sample_json=self.sample_json,
            composite_result=self.composite,
            tool_predictions=self.predictions,
            config=self.config,
        )
        for name in METRIC_NAMES:
            self.assertEqual(
                getattr(result, name), result.details[name].score,
                f"{name}: top-level vs details mismatch",
            )


# ----------------------- abstain paths --------------------------------------


class TestScorePredictionAbstainPaths(unittest.TestCase):
    def _config(self) -> dict:
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

    def test_no_active_tools_consensus_abstains(self):
        # All tools failed → consensus returns (None, info) → metric drops out
        # but other metrics still computed.
        result = score_prediction(
            sample_json=_sample_json(),
            composite_result=_composite(),
            tool_predictions=[
                _pred("equipnas", success=False),
                _pred("p2rank", success=False),
            ],
            config=self._config(),
        )
        self.assertFalse(result.details["cross_tool_consensus"].computed)
        self.assertIsNone(result.cross_tool_consensus)
        # Other 4 should still compute.
        self.assertEqual(result.n_metrics_computed, 4)
        self.assertNotIn("cross_tool_consensus", result.weights_used)

    def test_empty_binding_aborts_most_metrics(self):
        # Empty binding → q1/q2/q3/q5 all abstain (each requires
        # binding); only consensus runs (it operates on tool sets,
        # composite is allowed empty), and yields a number.
        result = score_prediction(
            sample_json=_sample_json(),
            composite_result=_composite(binding=[]),
            tool_predictions=[
                _pred("equipnas", binding=[10, 20, 30]),
                _pred("p2rank", binding=[15, 25, 35]),
            ],
            config=self._config(),
        )
        self.assertTrue(result.details["cross_tool_consensus"].computed)
        for n in (
            "structural_plausibility",
            "physicochemical_complementarity",
            "evolutionary_conservation",
            "known_motif_consistency",
        ):
            self.assertFalse(result.details[n].computed, f"{n} should abstain")
        self.assertEqual(result.n_metrics_computed, 1)
        # Only consensus's weight should appear in weights_used.
        self.assertEqual(set(result.weights_used), {"cross_tool_consensus"})

    def test_zero_metrics_yields_zero_total(self):
        # No binding, no tools → every metric abstains.
        result = score_prediction(
            sample_json=_sample_json(),
            composite_result=_composite(binding=[]),
            tool_predictions=[],
            config=self._config(),
        )
        self.assertEqual(result.n_metrics_computed, 0)
        self.assertEqual(result.total_score, 0.0)
        self.assertEqual(result.weights_used, {})
        for name in METRIC_NAMES:
            self.assertIsNone(getattr(result, name))

    def test_missing_pi_does_not_abstain_q3(self):
        # q3 design: pI missing → drops one of the 3 axes but still
        # returns a score. Verify q3 still computes.
        result = score_prediction(
            sample_json=_sample_json(with_pi=False),
            composite_result=_composite(),
            tool_predictions=[
                _pred("equipnas", binding=list(range(21, 29))),
                _pred("p2rank", binding=list(range(21, 29))),
            ],
            config=self._config(),
        )
        self.assertTrue(result.details["evolutionary_conservation"].computed)


# ----------------------- failure isolation ----------------------------------


class TestMetricFailureIsolation(unittest.TestCase):
    def test_one_metric_raises_others_succeed(self):
        # Patch consensus to raise; verify others still compute.
        from step6_pocket_qa import scorer as scorer_mod

        def _boom(**kwargs):
            raise RuntimeError("synthetic explosion")

        with patch.object(scorer_mod, "cross_tool_consensus", _boom):
            result = score_prediction(
                sample_json=_sample_json(),
                composite_result=_composite(),
                tool_predictions=[
                    _pred("equipnas", binding=list(range(21, 29))),
                    _pred("p2rank", binding=list(range(21, 29))),
                ],
                config={"pocket_qa": {"weights": {n: 0.2 for n in METRIC_NAMES}}},
            )
        cons = result.details["cross_tool_consensus"]
        self.assertFalse(cons.computed)
        self.assertIsNotNone(cons.error)
        self.assertIn("synthetic explosion", cons.error)
        # 4 surviving metrics.
        self.assertEqual(result.n_metrics_computed, 4)
        self.assertNotIn("cross_tool_consensus", result.weights_used)
        # total_score still valid.
        self.assertGreaterEqual(result.total_score, 0.0)
        self.assertLessEqual(result.total_score, 1.0)

    def test_metric_returning_bad_shape_is_trapped(self):
        # Patch a metric to return a bare float instead of (score, info).
        from step6_pocket_qa import scorer as scorer_mod

        def _bad(**kwargs):
            return 0.5  # missing the info dict

        with patch.object(scorer_mod, "physicochemical_complementarity", _bad):
            result = score_prediction(
                sample_json=_sample_json(),
                composite_result=_composite(),
                tool_predictions=[
                    _pred("equipnas", binding=list(range(21, 29))),
                    _pred("p2rank", binding=list(range(21, 29))),
                ],
                config={"pocket_qa": {"weights": {n: 0.2 for n in METRIC_NAMES}}},
            )
        d = result.details["physicochemical_complementarity"]
        self.assertFalse(d.computed)
        self.assertIsNotNone(d.error)
        self.assertIn("expected (score, info)", d.error)

    def test_metric_returning_string_score_trapped_as_error(self):
        from step6_pocket_qa import scorer as scorer_mod

        def _str(**kwargs):
            return "not-a-number", {"x": 1}

        with patch.object(scorer_mod, "evolutionary_conservation", _str):
            result = score_prediction(
                sample_json=_sample_json(),
                composite_result=_composite(),
                tool_predictions=[_pred("equipnas", binding=[21, 22, 23])],
                config={"pocket_qa": {"weights": {n: 0.2 for n in METRIC_NAMES}}},
            )
        d = result.details["evolutionary_conservation"]
        self.assertFalse(d.computed)
        self.assertIsNotNone(d.error)
        # ``info`` should still be carried through.
        self.assertEqual(d.info.get("x"), 1)


# ----------------------- helper-level unit tests ----------------------------


class TestSafeMetricCall(unittest.TestCase):
    def test_computed_path(self):
        d = _safe_metric_call(lambda: (0.42, {"k": "v"}))
        self.assertTrue(d.computed)
        self.assertEqual(d.score, 0.42)
        self.assertIsNone(d.error)
        self.assertEqual(d.info["k"], "v")

    def test_abstain_path_none_score(self):
        d = _safe_metric_call(lambda: (None, {"reason": "nope"}))
        self.assertFalse(d.computed)
        self.assertIsNone(d.score)
        self.assertIsNone(d.error)
        self.assertEqual(d.info["reason"], "nope")

    def test_exception_path(self):
        def _boom():
            raise ValueError("bad")
        d = _safe_metric_call(_boom)
        self.assertFalse(d.computed)
        self.assertIsNone(d.score)
        self.assertIn("bad", d.error)
        self.assertEqual(d.info, {})

    def test_score_clamped_above_one(self):
        d = _safe_metric_call(lambda: (1.5, {}))
        self.assertTrue(d.computed)
        self.assertEqual(d.score, 1.0)

    def test_score_clamped_below_zero(self):
        d = _safe_metric_call(lambda: (-0.3, {}))
        self.assertTrue(d.computed)
        self.assertEqual(d.score, 0.0)

    def test_non_dict_info_replaced(self):
        d = _safe_metric_call(lambda: (0.5, ["not", "a", "dict"]))
        self.assertTrue(d.computed)
        # info preserved as a sentinel describing the original shape.
        self.assertEqual(d.info.get("_raw_info_type"), "list")


class TestPickStructurePath(unittest.TestCase):
    def test_first_cat_a_success_wins(self):
        preds = [
            _pred("equipnas", cat="C", binding=[1]),
            _pred("boltz2", cat="A", binding=[2], structure_path="/tmp/a.cif"),
            _pred("rf2na", cat="A", binding=[3], structure_path="/tmp/b.cif"),
        ]
        self.assertEqual(_pick_structure_path(preds), "/tmp/a.cif")

    def test_skips_failed_cat_a(self):
        preds = [
            ToolPrediction(
                tool_id="boltz2", category="A", sample_id="x",
                success=False, error_message="oops",
            ),
            _pred("rf2na", cat="A", binding=[3], structure_path="/tmp/b.cif"),
        ]
        self.assertEqual(_pick_structure_path(preds), "/tmp/b.cif")

    def test_no_cat_a_returns_none(self):
        preds = [_pred("equipnas", cat="C", binding=[1])]
        self.assertIsNone(_pick_structure_path(preds))

    def test_cat_a_without_structure_path_skipped(self):
        preds = [_pred("boltz2", cat="A", binding=[1])]
        self.assertIsNone(_pick_structure_path(preds))


class TestAggregateWeighted(unittest.TestCase):
    def _details(self, scores: dict) -> dict[str, MetricDetail]:
        out: dict[str, MetricDetail] = {}
        for name in METRIC_NAMES:
            s = scores.get(name)
            if s is None:
                out[name] = MetricDetail(score=None, computed=False)
            else:
                out[name] = MetricDetail(score=s, computed=True)
        return out

    def test_renormalises_when_metric_missing(self):
        # Two metrics present, weights 0.25 / 0.25; renorm denominator 0.5.
        details = self._details({
            "structural_plausibility": 0.6,
            "cross_tool_consensus": 0.8,
        })
        weights = {n: 0.2 for n in METRIC_NAMES}
        weights["structural_plausibility"] = 0.25
        weights["cross_tool_consensus"] = 0.25
        total, used, n = _aggregate_weighted(details, weights)
        self.assertAlmostEqual(total, (0.25 * 0.6 + 0.25 * 0.8) / 0.5, places=5)
        self.assertEqual(n, 2)
        self.assertEqual(set(used), {"structural_plausibility", "cross_tool_consensus"})

    def test_zero_metrics_total_zero(self):
        details = self._details({})
        total, used, n = _aggregate_weighted(details, {n: 0.2 for n in METRIC_NAMES})
        self.assertEqual(total, 0.0)
        self.assertEqual(used, {})
        self.assertEqual(n, 0)

    def test_zero_weight_excludes_metric(self):
        # Weight=0 means the metric is configured-off; even computed it
        # should not enter the sum and not count toward n_metrics_computed.
        details = self._details({
            "structural_plausibility": 0.6,
            "cross_tool_consensus": 0.8,
        })
        weights = {n: 0.0 for n in METRIC_NAMES}
        weights["cross_tool_consensus"] = 0.5
        total, used, n = _aggregate_weighted(details, weights)
        self.assertEqual(total, 0.8)
        self.assertEqual(used, {"cross_tool_consensus": 0.5})
        self.assertEqual(n, 1)

    def test_total_clamped_to_unit_interval(self):
        # Pathological: feed a 1.0 score with weight, divide by tiny
        # denom → still in [0, 1] because each score is already ≤ 1.0.
        details = self._details({"cross_tool_consensus": 1.0})
        total, _, _ = _aggregate_weighted(
            details, {"cross_tool_consensus": 0.0001},
        )
        self.assertEqual(total, 1.0)


# ----------------------- run.py CLI round-trip ------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestRunCli(unittest.TestCase):
    """End-to-end CLI: write fake step2/4/5 jsonl + sample json, run main()."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.processed = self.root / "processed"
        (self.processed / "samples").mkdir(parents=True)
        (self.processed / "samples" / "mock_sample.json").write_text(
            json.dumps(_sample_json()), encoding="utf-8",
        )

        composite = _composite()
        target_char = {"category": "RRM_x_stem_loop"}
        s5_record = build_output_record(composite, _sample_json(), target_char)
        self.s5_path = self.root / "step5.jsonl"
        _write_jsonl(self.s5_path, [s5_record])

        s4_record = {
            "sample_id": "mock_sample",
            "tools_run": ["equipnas", "p2rank"],
            "predictions": [
                _pred("equipnas", binding=list(range(21, 29))).model_dump(),
                _pred("p2rank", cat="B",
                      binding=list(range(21, 29))).model_dump(),
            ],
            "total_runtime_seconds": 1.23,
            "timestamp": "2026-05-03T00:00:00Z",
        }
        self.s4_path = self.root / "step4.jsonl"
        _write_jsonl(self.s4_path, [s4_record])

        self.s2_path = self.root / "step2.jsonl"
        _write_jsonl(self.s2_path, [{
            "sample_id": "mock_sample",
            "output": {"category": "RRM_x_stem_loop", "confidence": 0.9},
        }])

        # Minimal config matching the production yaml shape.
        self.cfg_path = self.root / "step6_config.yaml"
        self.cfg_path.write_text(
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

    def _run(self, *, output_arg: Path) -> int:
        argv = [
            "--processed-dir", str(self.processed),
            "--step4-output", str(self.s4_path),
            "--step5-output", str(self.s5_path),
            "--step2-output", str(self.s2_path),
            "--config", str(self.cfg_path),
            "--output", str(output_arg),
        ]
        return step6_run.main(argv)

    def test_writes_single_jsonl_and_validates(self):
        out = self.root / "step6.jsonl"
        rc = self._run(output_arg=out)
        self.assertEqual(rc, 0)
        self.assertTrue(out.is_file())
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        self.assertEqual(len(rows), 1)
        rec = rows[0]
        # Round-trip through the schema.
        result = PocketQAResult.model_validate(rec)
        self.assertEqual(result.sample_id, "mock_sample")
        self.assertEqual(result.n_metrics_computed, 5)

    def test_cli_matches_in_process_score(self):
        out = self.root / "step6.jsonl"
        self._run(output_arg=out)
        cli_rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])

        # Recompute in-process and compare on numeric fields only
        # (timestamp differs).
        in_proc = score_prediction(
            sample_json=_sample_json(),
            composite_result=_composite(),
            tool_predictions=[
                _pred("equipnas", binding=list(range(21, 29))),
                _pred("p2rank", cat="B", binding=list(range(21, 29))),
            ],
            config={
                "pocket_qa": {
                    "weights": {
                        "structural_plausibility": 0.25,
                        "physicochemical_complementarity": 0.20,
                        "evolutionary_conservation": 0.15,
                        "cross_tool_consensus": 0.25,
                        "known_motif_consistency": 0.15,
                    }
                }
            },
        ).model_dump(mode="json")
        for name in METRIC_NAMES:
            self.assertAlmostEqual(
                cli_rec[name], in_proc[name], places=5,
                msg=f"{name} differs CLI vs in-process",
            )
        self.assertAlmostEqual(
            cli_rec["total_score"], in_proc["total_score"], places=5,
        )
        self.assertEqual(cli_rec["n_metrics_computed"], in_proc["n_metrics_computed"])
        self.assertEqual(cli_rec["weights_used"], in_proc["weights_used"])

    def test_writes_per_sample_files_when_output_is_dir(self):
        out_dir = self.root / "step6_dir"
        rc = self._run(output_arg=out_dir)
        self.assertEqual(rc, 0)
        sample_file = out_dir / "mock_sample.jsonl"
        self.assertTrue(sample_file.is_file())
        rec = json.loads(sample_file.read_text(encoding="utf-8").splitlines()[0])
        PocketQAResult.model_validate(rec)

    def test_skips_sample_missing_step4(self):
        # Drop the step4 file → CLI should skip the sample → no records,
        # n_total=0 → exit code 2.
        empty_s4 = self.root / "empty_step4.jsonl"
        empty_s4.write_text("", encoding="utf-8")
        out = self.root / "step6.jsonl"
        argv = [
            "--processed-dir", str(self.processed),
            "--step4-output", str(empty_s4),
            "--step5-output", str(self.s5_path),
            "--config", str(self.cfg_path),
            "--output", str(out),
        ]
        rc = step6_run.main(argv)
        self.assertEqual(rc, 2)

    def test_no_step2_falls_back_to_step5_category(self):
        out = self.root / "step6_nos2.jsonl"
        argv = [
            "--processed-dir", str(self.processed),
            "--step4-output", str(self.s4_path),
            "--step5-output", str(self.s5_path),
            "--config", str(self.cfg_path),
            "--output", str(out),
        ]
        rc = step6_run.main(argv)
        self.assertEqual(rc, 0)
        rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        # Motif metric should still find RRM motif via step5_category fallback.
        info = rec["details"]["known_motif_consistency"]["info"]
        # Either domain found via fallback (good) or neutral (acceptable);
        # we just want the metric to run, not abstain on missing input.
        self.assertTrue(rec["details"]["known_motif_consistency"]["computed"])


if __name__ == "__main__":
    unittest.main()
