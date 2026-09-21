"""Mock tests for scripts/rerun_single_tool.py.

Verifies:
  - missing step 4 record → recorded as failure, downstream not run
  - failed haddock3 entry is replaced (not duplicated) and other tool
    predictions are preserved verbatim
  - --resume skips samples where the tool already succeeded
  - downstream step 5/6/7/8 JSONLs are rewritten with the patched set
  - --no-weight-update suppresses step 8 + leaves W on disk untouched
  - CNS_SOLVE resolves from CLI > config > env var
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from step3_tool_selection.weight_tensor import WeightTensor  # noqa: E402
from step4_tool_adapters.schemas import ToolPrediction, ToolPredictionSet  # noqa: E402
from step5_fusion.schemas import CompositeResult  # noqa: E402
from step6_pocket_qa.schemas import MetricDetail, PocketQAResult  # noqa: E402
from step7_iteration.schemas import (  # noqa: E402
    IterationAction, IterationRecord, IterationResult,
)

import scripts.rerun_single_tool as rs  # noqa: E402


# ---------- canned step records --------------------------------------------


def _sample_json(sid: str) -> dict:
    return {
        "sample_id": sid,
        "source_pdb": "1abc",
        "protein": {"chain_id": "A", "sequence": "M" * 30, "length": 30},
        "rna": {"chain_id": "B", "sequence": "GCGCGCGCGC", "length": 10},
        "interaction": {"binding_protein_residues": [10, 11]},
        "data_availability": {"quality_tier": "high"},
    }


def _step2_record(sid: str) -> dict:
    return {
        "sample_id": sid,
        "input_features": {"sample_id": sid},
        "output": {
            "category": "RRM_x_stem_loop", "confidence": 0.9,
            "analysis": "RRM binding stem-loop",
        },
        "api_usage": {"total_tokens": 200, "status": "ok"},
        "success": True, "retries": 0,
        "timestamp": "2026-05-10T00:00:00Z",
    }


def _make_pred(tool_id: str, sid: str, success: bool, *,
               error: str = "", category: str = "C") -> dict:
    if success:
        return ToolPrediction(
            tool_id=tool_id, category=category, sample_id=sid, success=True,
            binding_protein_residues=[10, 11],
            per_residue_confidence={10: 0.9, 11: 0.8},
        ).model_dump(mode="json")
    return ToolPrediction(
        tool_id=tool_id, category=category, sample_id=sid, success=False,
        error_message=error or "synthetic failure",
    ).model_dump(mode="json")


def _step4_record(sid: str, *, haddock3_success: bool) -> dict:
    preds = [
        _make_pred("p2rank", sid, True, category="B"),
        _make_pred("equipnas", sid, True, category="C"),
        _make_pred("boltz2", sid, True, category="A"),
        _make_pred("chai1", sid, True, category="A"),
        _make_pred("haddock3", sid, haddock3_success,
                   category="D",
                   error=("HADDOCK 3 failed (rc=1): /opt/conda/envs/"
                          "haddock3 not found")),
    ]
    return ToolPredictionSet(
        sample_id=sid,
        tools_run=["boltz2", "chai1", "p2rank", "equipnas", "haddock3"],
        predictions=[ToolPrediction.model_validate(p) for p in preds],
        total_runtime_seconds=42.0,
        timestamp="2026-05-10T00:00:00Z",
    ).model_dump(mode="json")


def _step5_composite(sid: str) -> CompositeResult:
    return CompositeResult(
        sample_id=sid,
        tool_weights={"p2rank": 0.5, "equipnas": 0.6, "boltz2": 0.7,
                      "chai1": 0.7, "haddock3": 0.4},
        binding_protein_residues=[10, 11, 12],
        binding_rna_nucleotides=[3, 4],
        per_residue_probability={10: 0.9, 11: 0.7, 12: 0.6},
        threshold=0.5,
        fusion_rationale="ok",
        confidence=0.8,
        tools_fused=["p2rank", "equipnas", "boltz2", "chai1", "haddock3"],
        api_usage={"status": "ok", "total_tokens": 400},
        timestamp="2026-05-10T00:00:00Z",
    )


def _step6_qa(sid: str) -> PocketQAResult:
    return PocketQAResult(
        sample_id=sid,
        structural_plausibility=0.7,
        physicochemical_complementarity=0.6,
        evolutionary_conservation=0.5,
        cross_tool_consensus=0.65,
        known_motif_consistency=0.55,
        total_score=0.6,
        weights_used={n: 0.2 for n in (
            "structural_plausibility",
            "physicochemical_complementarity",
            "evolutionary_conservation",
            "cross_tool_consensus",
            "known_motif_consistency",
        )},
        n_metrics_computed=5,
        details={
            "structural_plausibility": MetricDetail(score=0.7, computed=True, info={}),
            "physicochemical_complementarity": MetricDetail(score=0.6, computed=True, info={}),
            "evolutionary_conservation": MetricDetail(score=0.5, computed=True, info={}),
            "cross_tool_consensus": MetricDetail(
                score=0.65, computed=True,
                info={"active_tools": ["p2rank", "equipnas", "boltz2",
                                       "chai1", "haddock3"],
                      "pairwise_jaccard": {}}),
            "known_motif_consistency": MetricDetail(score=0.55, computed=True, info={}),
        },
        timestamp="2026-05-10T00:00:00Z",
    )


def _step7_iter(sid: str) -> IterationResult:
    return IterationResult(
        sample_id=sid, final_action="accept", total_iterations=1,
        final_score=0.6, score_trajectory=[0.6],
        iterations=[IterationRecord(
            iteration=0,
            action=IterationAction(action="accept", rationale="ok"*5,
                                   confidence=0.8),
            score_before=0.6, score_after=0.6, delta=0.0,
            api_usage={"total_tokens": 500, "status": "ok"},
            timestamp="2026-05-10T00:00:00Z",
        )],
        termination_reason="accepted",
        final_binding_protein_residues=[10, 11, 12],
        final_binding_rna_nucleotides=[3, 4],
        timestamp="2026-05-10T00:00:00Z",
    )


# ---------- harness --------------------------------------------------------


class _RerunHarness(unittest.TestCase):
    """Build a synthetic ``--output-dir`` with step 2 + step 4 already on
    disk for 3 samples (s1, s2, s3). haddock3 is a failure in step 4 for
    every sample; the rerun script's job is to turn it into a success
    record and refresh the downstream files."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

        # processed_dir / samples / <id>.json
        self.processed = self.tmp / "processed"
        (self.processed / "samples").mkdir(parents=True)
        for sid in ("s1", "s2", "s3"):
            (self.processed / "samples" / f"{sid}.json").write_text(
                json.dumps(_sample_json(sid)), encoding="utf-8",
            )

        # output_dir/step2 + step4 pre-populated
        self.output = self.tmp / "out"
        (self.output / "step2").mkdir(parents=True)
        (self.output / "step4").mkdir(parents=True)
        for sid in ("s1", "s2", "s3"):
            (self.output / "step2" / f"{sid}.jsonl").write_text(
                json.dumps(_step2_record(sid)) + "\n", encoding="utf-8",
            )
            (self.output / "step4" / f"{sid}.jsonl").write_text(
                json.dumps(_step4_record(sid, haddock3_success=False)) + "\n",
                encoding="utf-8",
            )

        # configs (empty stubs — inner code tolerates)
        self.config_dir = self.tmp / "configs"
        self.config_dir.mkdir()
        for n in ("step2", "step3", "step4", "step5",
                  "step6", "step7", "step8"):
            (self.config_dir / f"{n}_config.yaml").write_text(
                "", encoding="utf-8",
            )

        # W tensor + sample list paths
        self.tensor_path = self.tmp / "W.json"
        self.list_path = self.tmp / "samples.txt"
        self.list_path.write_text("s1\ns2\ns3\n", encoding="utf-8")

        # Patch the four downstream entry points + the adapter dispatch.
        # We give haddock3 a *successful* synthetic prediction by default
        # so the rerun looks "fixed".
        self.fake_adapter = _FakeHaddockAdapter()
        self.patches = [
            patch("scripts.rerun_single_tool.get_adapter",
                  return_value=self.fake_adapter),
            patch("scripts.rerun_single_tool.fuse_predictions",
                  side_effect=lambda **kw:
                      _step5_composite(kw["sample_json"]["sample_id"])),
            patch("scripts.rerun_single_tool.score_prediction",
                  side_effect=lambda **kw:
                      _step6_qa(kw["sample_json"]["sample_id"])),
            patch("scripts.rerun_single_tool.run_iteration_loop",
                  side_effect=lambda **kw:
                      _step7_iter(kw["sample_json"]["sample_id"])),
        ]
        for p in self.patches:
            p.start()

        os.environ["LLM_API_KEY"] = "test-key-not-real"

    def tearDown(self):
        for p in self.patches:
            p.stop()


class _FakeHaddockAdapter:
    """Synthetic Haddock3Adapter — returns a success record without
    actually running anything. Switchable failure-mode for tests that
    want to verify how the script records errors."""

    def __init__(self, succeed: bool = True):
        self.succeed = succeed
        self.calls: list[str] = []

    def predict(self, sample_json, work_dir, config):  # noqa: D401
        sid = sample_json["sample_id"]
        self.calls.append(sid)
        if not self.succeed:
            return ToolPrediction(
                tool_id="haddock3", category="D", sample_id=sid,
                success=False, error_message="still broken",
            )
        return ToolPrediction(
            tool_id="haddock3", category="D", sample_id=sid, success=True,
            binding_protein_residues=[10, 11],
            per_residue_pae_score={10: 0.7, 11: 0.6},
            predicted_structure_path="/tmp/fake.pdb",
        )


def _common_argv(h) -> list[str]:
    return [
        "--tool", "haddock3",
        "--processed-dir", str(h.processed),
        "--sample-list", str(h.list_path),
        "--config-dir", str(h.config_dir),
        "--output-dir", str(h.output),
        "--weight-tensor", str(h.tensor_path),
        "--no-llm",
    ]


# ---------- success path ---------------------------------------------------


class TestBasicRerun(_RerunHarness):
    def test_replaces_failed_haddock3_entry(self):
        rc = rs.main(_common_argv(self))
        self.assertEqual(rc, 0)
        for sid in ("s1", "s2", "s3"):
            rec = json.loads(
                (self.output / "step4" / f"{sid}.jsonl")
                .read_text(encoding="utf-8").strip()
            )
            haddock = [p for p in rec["predictions"]
                       if p["tool_id"] == "haddock3"]
            # Exactly one haddock3 entry (no duplicate) and it succeeded.
            self.assertEqual(len(haddock), 1)
            self.assertTrue(haddock[0]["success"])
            self.assertIn("haddock3", rec["tools_run"])

    def test_preserves_other_tool_predictions(self):
        # Capture the original p2rank entry, run, and verify it survived
        # the patch round-trip.
        original = json.loads(
            (self.output / "step4" / "s1.jsonl")
            .read_text(encoding="utf-8").strip()
        )
        original_p2rank = next(p for p in original["predictions"]
                               if p["tool_id"] == "p2rank")

        rs.main(_common_argv(self))

        after = json.loads(
            (self.output / "step4" / "s1.jsonl")
            .read_text(encoding="utf-8").strip()
        )
        after_p2rank = next(p for p in after["predictions"]
                            if p["tool_id"] == "p2rank")
        self.assertEqual(original_p2rank, after_p2rank)
        # All five tools are still present.
        tool_ids = {p["tool_id"] for p in after["predictions"]}
        self.assertEqual(
            tool_ids,
            {"p2rank", "equipnas", "boltz2", "chai1", "haddock3"},
        )

    def test_downstream_files_written(self):
        rs.main(_common_argv(self))
        for sid in ("s1", "s2", "s3"):
            for step in ("step5", "step6", "step7", "step8"):
                self.assertTrue(
                    (self.output / step / f"{sid}.jsonl").is_file(),
                    f"missing {step}/{sid}.jsonl",
                )

    def test_summary_jsonl_one_row_per_sample(self):
        rs.main(_common_argv(self))
        rows = [
            json.loads(line) for line in
            (self.output / "summary" / "rerun_results.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertEqual(r["tool"], "haddock3")
            self.assertEqual(r["action"], "replace")
            self.assertTrue(r["tool_success"])
            self.assertTrue(r["step4_patched"])
            self.assertTrue(r["downstream_rerun"])


# ---------- --resume -------------------------------------------------------


class TestResume(_RerunHarness):
    def test_resume_skips_already_successful(self):
        # Mark s2 as already-success-on-disk by rewriting its step 4 file.
        (self.output / "step4" / "s2.jsonl").write_text(
            json.dumps(_step4_record("s2", haddock3_success=True)) + "\n",
            encoding="utf-8",
        )
        rc = rs.main(_common_argv(self) + ["--resume"])
        self.assertEqual(rc, 0)
        # Adapter was only invoked on s1 + s3 (s2 was skipped).
        self.assertEqual(sorted(self.fake_adapter.calls), ["s1", "s3"])
        rows = [
            json.loads(line) for line in
            (self.output / "summary" / "rerun_results.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        actions = {r["sample_id"]: r["action"] for r in rows}
        self.assertEqual(actions["s2"], "skip_existing_success")
        self.assertEqual(actions["s1"], "replace")
        self.assertEqual(actions["s3"], "replace")


# ---------- append (no entry on disk) --------------------------------------


class TestAppendWhenMissing(_RerunHarness):
    def test_append_when_tool_absent(self):
        # Strip haddock3 from s1's step 4 record so the action becomes
        # "append" rather than "replace".
        rec = json.loads(
            (self.output / "step4" / "s1.jsonl")
            .read_text(encoding="utf-8").strip()
        )
        rec["predictions"] = [p for p in rec["predictions"]
                              if p["tool_id"] != "haddock3"]
        rec["tools_run"] = [t for t in rec["tools_run"] if t != "haddock3"]
        (self.output / "step4" / "s1.jsonl").write_text(
            json.dumps(rec) + "\n", encoding="utf-8",
        )

        rs.main(_common_argv(self))

        after = json.loads(
            (self.output / "step4" / "s1.jsonl")
            .read_text(encoding="utf-8").strip()
        )
        haddock = [p for p in after["predictions"]
                   if p["tool_id"] == "haddock3"]
        self.assertEqual(len(haddock), 1)
        self.assertTrue(haddock[0]["success"])


# ---------- --no-weight-update --------------------------------------------


class TestNoWeightUpdate(_RerunHarness):
    def test_step8_skipped_and_tensor_untouched(self):
        wt = WeightTensor()
        wt.update("p2rank", "RRM_x_stem_loop",
                  "structural_plausibility", 0.42)
        wt.save(self.tensor_path)
        snapshot = self.tensor_path.read_text(encoding="utf-8")

        rc = rs.main(_common_argv(self) + ["--no-weight-update"])
        self.assertEqual(rc, 0)
        for sid in ("s1", "s2", "s3"):
            self.assertFalse(
                (self.output / "step8" / f"{sid}.jsonl").is_file()
            )
        self.assertEqual(
            self.tensor_path.read_text(encoding="utf-8"), snapshot,
            "eval mode must not mutate the weight tensor",
        )


# ---------- missing step 4 file -------------------------------------------


class TestMissingStep4(_RerunHarness):
    def test_missing_step4_recorded_as_failure(self):
        # Remove s2's step 4 file → the script can't patch it.
        (self.output / "step4" / "s2.jsonl").unlink()
        rc = rs.main(_common_argv(self))
        self.assertEqual(rc, 2)  # at least one sample failed
        failures = [
            json.loads(line) for line in
            (self.output / "summary" / "rerun_failures.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        sids = {r["sample_id"] for r in failures}
        self.assertEqual(sids, {"s2"})
        # The other two still succeeded.
        ok_sids = {
            json.loads(line)["sample_id"] for line in
            (self.output / "summary" / "rerun_results.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()
        }
        self.assertEqual(ok_sids, {"s1", "s3"})


# ---------- CNS_SOLVE resolution -------------------------------------------


class TestCnsResolution(unittest.TestCase):
    def test_cli_beats_config_and_env(self):
        cfgs = {"step4": {"tools": {"haddock3": {"cns_solve": "/from/config"}}}}
        os.environ["CNS_SOLVE"] = "/from/env"
        try:
            resolved = rs.inject_cns_solve(cfgs, "/from/cli")
        finally:
            os.environ.pop("CNS_SOLVE", None)
        self.assertEqual(resolved, "/from/cli")
        self.assertEqual(
            cfgs["step4"]["tools"]["haddock3"]["cns_solve"], "/from/cli",
        )

    def test_config_beats_env(self):
        cfgs = {"step4": {"tools": {"haddock3": {"cns_solve": "/from/config"}}}}
        os.environ["CNS_SOLVE"] = "/from/env"
        try:
            resolved = rs.inject_cns_solve(cfgs, None)
        finally:
            os.environ.pop("CNS_SOLVE", None)
        self.assertEqual(resolved, "/from/config")

    def test_env_when_only_env(self):
        cfgs: dict = {}
        os.environ["CNS_SOLVE"] = "/from/env"
        try:
            resolved = rs.inject_cns_solve(cfgs, None)
        finally:
            os.environ.pop("CNS_SOLVE", None)
        self.assertEqual(resolved, "/from/env")
        self.assertEqual(
            cfgs["step4"]["tools"]["haddock3"]["cns_solve"], "/from/env",
        )

    def test_none_when_no_source(self):
        cfgs: dict = {}
        os.environ.pop("CNS_SOLVE", None)
        resolved = rs.inject_cns_solve(cfgs, None)
        self.assertIsNone(resolved)
        # The config block is still seeded with empty tooling section so
        # downstream reads don't KeyError.
        self.assertIn("tools", cfgs["step4"])


class TestToolCategoryResolution(unittest.TestCase):
    """Bug fix: the adapter-lookup-failure fallback must take its
    category from the registry, never a hard-coded 'D'. RF2NA
    (rosettafold2na) is Cat A; mislabelling it 'D' poisons fusion."""

    def test_canonical_alias(self):
        self.assertEqual(rs._canonical_tool_id("rf2na"),
                         "rosettafold2na")
        self.assertEqual(rs._canonical_tool_id("boltz2"), "boltz2")

    def test_registry_category(self):
        self.assertEqual(rs._registry_category("rosettafold2na"), "A")
        self.assertEqual(rs._registry_category("rf2na"), "A")  # alias
        self.assertEqual(rs._registry_category("boltz2"), "A")
        self.assertEqual(rs._registry_category("p2rank"), "B")
        # genuinely unknown id → explicit default only
        self.assertEqual(
            rs._registry_category("not_a_tool", default="D"), "D")

    def test_run_single_tool_failure_uses_registry_category(self):
        # Force the adapter lookup to fail so the fallback path runs.
        with patch("scripts.rerun_single_tool.get_adapter",
                   side_effect=ValueError("boom")):
            for tid in ("rosettafold2na", "rf2na"):
                pred = rs.run_single_tool(
                    tid, {"sample_id": "s1"}, Path("."), {})
                self.assertFalse(pred.success)
                self.assertEqual(pred.category, "A", tid)
            # unknown tool still degrades to D (not a crash).
            pred = rs.run_single_tool(
                "totally_unknown", {"sample_id": "s1"},
                Path("."), {})
            self.assertEqual(pred.category, "D")


if __name__ == "__main__":
    unittest.main()
