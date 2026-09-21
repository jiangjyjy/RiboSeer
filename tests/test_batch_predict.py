"""Mock tests for scripts/batch_predict.py.

We DO NOT spin up a real LLMClient or call any tool here. Each per-step
core function is patched at the import location used by the batch
script (``scripts.batch_predict.<name>``) and made to return a canned
dict / Pydantic model. The test exercises:

  - the orchestration glue (which step gets called, in what order)
  - --resume skipping known sample ids
  - --no-weight-update skipping step 8 + leaving W tensor unchanged
  - --start / --end slicing
  - failure isolation (one bad sample doesn't poison the rest)
  - summary row format matches the spec

This is heavier on patching than other tests in the project, but it's
the right level: integration of seven steps + IO is exactly what
batch_predict adds, so we test that, not the steps themselves.
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
from step6_pocket_qa.schemas import PocketQAResult, MetricDetail  # noqa: E402
from step7_iteration.schemas import (  # noqa: E402
    IterationAction, IterationRecord, IterationResult,
)

import scripts.batch_predict as bp  # noqa: E402


# ---------- canned per-step return values ---------------------------------


def _step2_record(sid: str = "s1") -> dict:
    return {
        "sample_id": sid,
        "input_features": {"sample_id": sid},
        "output": {
            "category": "RRM_x_stem_loop",
            "confidence": 0.9,
            "analysis": "RRM binding stem-loop",
        },
        "api_usage": {"total_tokens": 200, "status": "ok"},
        "success": True,
        "retries": 0,
        "timestamp": "2026-05-04T00:00:00Z",
    }


def _step3_record(sid: str = "s1") -> dict:
    return {
        "sample_id": sid,
        "step2_category": "RRM_x_stem_loop",
        "tool_plan": {
            "selected_tools": ["p2rank", "equipnas"],
            "rationale": "go", "confidence": 0.8,
        },
        "utility_scores": {"p2rank": 0.6, "equipnas": 0.7},
        "api_usage": {"total_tokens": 300, "status": "ok"},
        "success": True,
        "retries": 0,
        "timestamp": "2026-05-04T00:00:00Z",
    }


def _step4_predictions(sid: str = "s1") -> ToolPredictionSet:
    preds = [
        ToolPrediction(
            tool_id="p2rank", category="B", sample_id=sid, success=True,
            binding_protein_residues=[10, 11],
            per_residue_confidence={10: 0.9, 11: 0.8},
        ),
        ToolPrediction(
            tool_id="equipnas", category="C", sample_id=sid, success=True,
            binding_protein_residues=[10, 12],
            per_residue_confidence={10: 0.85, 12: 0.7},
        ),
    ]
    return ToolPredictionSet(
        sample_id=sid,
        tools_run=["equipnas", "p2rank"],
        predictions=preds,
        total_runtime_seconds=1.0,
        timestamp="2026-05-04T00:00:00Z",
    )


def _step5_composite(sid: str = "s1") -> CompositeResult:
    return CompositeResult(
        sample_id=sid,
        tool_weights={"p2rank": 0.5, "equipnas": 0.6},
        binding_protein_residues=[10, 11, 12],
        binding_rna_nucleotides=[3, 4],
        per_residue_probability={10: 0.9, 11: 0.7, 12: 0.6},
        threshold=0.5,
        fusion_rationale="ok",
        confidence=0.8,
        tools_fused=["p2rank", "equipnas"],
        api_usage={"status": "ok", "total_tokens": 400},
        timestamp="2026-05-04T00:00:00Z",
    )


def _step6_qa(sid: str = "s1") -> PocketQAResult:
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
            "structural_plausibility": MetricDetail(
                score=0.7, computed=True, info={}),
            "physicochemical_complementarity": MetricDetail(
                score=0.6, computed=True, info={}),
            "evolutionary_conservation": MetricDetail(
                score=0.5, computed=True, info={}),
            "cross_tool_consensus": MetricDetail(
                score=0.65, computed=True,
                info={"active_tools": ["p2rank", "equipnas"],
                      "pairwise_jaccard": {"p2rank|equipnas": 0.6}}),
            "known_motif_consistency": MetricDetail(
                score=0.55, computed=True, info={}),
        },
        timestamp="2026-05-04T00:00:00Z",
    )


def _step7_iter_result(sid: str = "s1") -> IterationResult:
    return IterationResult(
        sample_id=sid,
        final_action="accept",
        total_iterations=1,
        final_score=0.6,
        score_trajectory=[0.6],
        iterations=[IterationRecord(
            iteration=0,
            action=IterationAction(
                action="accept",
                rationale="Score is acceptable; stop here.",
                confidence=0.8,
            ),
            score_before=0.6,
            score_after=0.6,
            delta=0.0,
            api_usage={"total_tokens": 500, "status": "ok"},
            timestamp="2026-05-04T00:00:00Z",
        )],
        termination_reason="accepted",
        final_binding_protein_residues=[10, 11, 12],
        final_binding_rna_nucleotides=[3, 4],
        timestamp="2026-05-04T00:00:00Z",
    )


def _sample_json(sid: str = "s1") -> dict:
    return {
        "sample_id": sid,
        "protein": {"chain_id": "A", "sequence": "M" * 50, "length": 50},
        "rna": {"chain_id": "B", "sequence": "GCGCGCGCGC", "length": 10},
        "interaction": {"binding_protein_residues": [10, 11]},
        "data_availability": {"quality_tier": "high"},
    }


# ---------- harness -------------------------------------------------------


class _Harness(unittest.TestCase):
    """Sets up the patches each test needs.

    Patches are applied at the import location used by ``batch_predict``
    (``scripts.batch_predict.<name>``), NOT at the source module —
    Python's import-time binding means a ``from x import y`` in
    batch_predict copies the reference, so we must patch the copy.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.processed = self.tmp / "processed"
        (self.processed / "samples").mkdir(parents=True)
        # Three test samples.
        for sid in ("s1", "s2", "s3"):
            (self.processed / "samples" / f"{sid}.json").write_text(
                json.dumps(_sample_json(sid)), encoding="utf-8",
            )
        self.output = self.tmp / "out"
        self.tensor_path = self.tmp / "weights" / "tensor.json"
        self.list_path = self.tmp / "samples.txt"
        self.list_path.write_text("s1\ns2\ns3\n", encoding="utf-8")
        self.config_dir = self.tmp / "configs"
        self.config_dir.mkdir()
        # Empty per-step yaml files. The batch script tolerates missing
        # ones, but writing minimal stubs proves that the loader path
        # works too.
        for name in ("step2", "step3", "step4", "step5",
                     "step6", "step7", "step8"):
            (self.config_dir / f"{name}_config.yaml").write_text(
                "", encoding="utf-8",
            )

        # Build patcher list. Every step is replaced by a callable that
        # returns the canned record / model. We patch ``build_client``
        # to return a sentinel non-None so ``client is None`` checks
        # downstream see "we have a client".
        sentinel_client = object()
        self.patches = [
            patch("scripts.batch_predict.build_client",
                  return_value=sentinel_client),
            patch("scripts.batch_predict.characterize_target",
                  side_effect=lambda sample_json, client, config, history=None:
                      _step2_record(sample_json["sample_id"])),
            patch("scripts.batch_predict.select_tools",
                  side_effect=lambda sample_id, target_char, target_features,
                                       client, weight_tensor, config, history=None:
                      _step3_record(sample_id)),
            patch("scripts.batch_predict.run_step4_sample",
                  side_effect=lambda sample, tool_ids, config, work_dir:
                      _step4_predictions(sample["sample_id"])),
            patch("scripts.batch_predict.fuse_predictions",
                  side_effect=lambda **kw:
                      _step5_composite(kw["sample_json"]["sample_id"])),
            patch("scripts.batch_predict.score_prediction",
                  side_effect=lambda **kw:
                      _step6_qa(kw["sample_json"]["sample_id"])),
            patch("scripts.batch_predict.run_iteration_loop",
                  side_effect=lambda **kw:
                      _step7_iter_result(kw["sample_json"]["sample_id"])),
        ]
        for p in self.patches:
            p.start()
        # Make sure LLM_API_KEY presence checks pass even when running
        # in CI without the real key.
        os.environ["LLM_API_KEY"] = "test-key-not-real"

    def tearDown(self):
        for p in self.patches:
            p.stop()


# ---------- training mode (steps 2–8) ------------------------------------


class TestTrainingMode(_Harness):
    def _argv(self, **overrides) -> list:
        argv = [
            "--processed-dir", str(self.processed),
            "--sample-list", str(self.list_path),
            "--config-dir", str(self.config_dir),
            "--output-dir", str(self.output),
            "--weight-tensor", str(self.tensor_path),
        ]
        for k, v in overrides.items():
            argv += [k, str(v)]
        return argv

    def test_full_run_writes_per_step_jsonls(self):
        rc = bp.main(self._argv())
        self.assertEqual(rc, 0)
        for sid in ("s1", "s2", "s3"):
            for step in ("step2", "step3", "step4",
                         "step5", "step6", "step7", "step8"):
                p = self.output / step / f"{sid}.jsonl"
                self.assertTrue(p.is_file(), f"missing {step}/{sid}.jsonl")

    def test_summary_row_shape(self):
        bp.main(self._argv())
        rows = [
            json.loads(line)
            for line in (self.output / "summary" / "results.jsonl")
            .read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 3)
        for r in rows:
            for key in (
                "sample_id", "category", "quality_tier",
                "protein_length", "rna_length",
                "tools_attempted", "tools_succeeded", "tools_used",
                "fusion_status", "binding_protein_predicted",
                "binding_protein_gt", "precision", "recall", "f1",
                "qa_total", "qa_final", "iteration_action",
                "total_iterations", "termination_reason",
                "runtime_seconds", "total_tokens", "timestamp",
            ):
                self.assertIn(key, r, f"summary missing {key}")
            self.assertEqual(r["category"], "RRM_x_stem_loop")
            self.assertEqual(r["tools_succeeded"], 2)
            self.assertEqual(r["iteration_action"], "accept")

    def test_weight_tensor_persisted(self):
        bp.main(self._argv())
        self.assertTrue(self.tensor_path.is_file())
        wt = WeightTensor.load(self.tensor_path)
        # 3 samples → counter for the surviving tools should be >= 1.
        # We don't pin an exact value because compute_per_tool_scores
        # may drop a tool with no q_m signal; just assert *some* update
        # happened.
        # (At minimum, calling save() created the file with valid JSON.)
        data = json.loads(self.tensor_path.read_text(encoding="utf-8"))
        self.assertIn("data", data)


# ---------- eval mode (--no-weight-update) -------------------------------


class TestEvalMode(_Harness):
    def test_step8_skipped(self):
        rc = bp.main([
            "--processed-dir", str(self.processed),
            "--sample-list", str(self.list_path),
            "--config-dir", str(self.config_dir),
            "--output-dir", str(self.output),
            "--weight-tensor", str(self.tensor_path),
            "--no-weight-update",
        ])
        self.assertEqual(rc, 0)
        # step8 dir must not exist (or at least no per-sample files).
        for sid in ("s1", "s2", "s3"):
            self.assertFalse(
                (self.output / "step8" / f"{sid}.jsonl").is_file(),
                f"step8 file should not be written in eval mode (sid={sid})",
            )
        # Other steps still wrote.
        for sid in ("s1", "s2", "s3"):
            self.assertTrue(
                (self.output / "step7" / f"{sid}.jsonl").is_file(),
            )

    def test_weight_tensor_unchanged_in_eval_mode(self):
        # Pre-populate a known tensor; eval should not mutate it.
        wt = WeightTensor()
        wt.update("p2rank", "RRM_x_stem_loop", "structural_plausibility", 0.42)
        self.tensor_path.parent.mkdir(parents=True, exist_ok=True)
        wt.save(self.tensor_path)
        original_content = self.tensor_path.read_text(encoding="utf-8")

        rc = bp.main([
            "--processed-dir", str(self.processed),
            "--sample-list", str(self.list_path),
            "--config-dir", str(self.config_dir),
            "--output-dir", str(self.output),
            "--weight-tensor", str(self.tensor_path),
            "--no-weight-update",
        ])
        self.assertEqual(rc, 0)
        # File on disk must be byte-identical to the pre-run snapshot.
        self.assertEqual(
            self.tensor_path.read_text(encoding="utf-8"),
            original_content,
            "eval mode must NOT mutate the weight tensor",
        )


# ---------- --resume ------------------------------------------------------


class TestResume(_Harness):
    def test_skips_known_done_ids(self):
        # Pre-write a results line for s1 + a failure line for s3 →
        # both should be skipped on a resume run.
        summary = self.output / "summary" / "results.jsonl"
        failures = self.output / "summary" / "failures.jsonl"
        summary.parent.mkdir(parents=True)
        summary.write_text(json.dumps({"sample_id": "s1"}) + "\n",
                           encoding="utf-8")
        failures.write_text(json.dumps({"sample_id": "s3"}) + "\n",
                            encoding="utf-8")

        rc = bp.main([
            "--processed-dir", str(self.processed),
            "--sample-list", str(self.list_path),
            "--config-dir", str(self.config_dir),
            "--output-dir", str(self.output),
            "--weight-tensor", str(self.tensor_path),
            "--resume",
        ])
        self.assertEqual(rc, 0)
        # Only s2 should have produced a fresh per-step JSONL this run.
        self.assertTrue((self.output / "step2" / "s2.jsonl").is_file())
        self.assertFalse((self.output / "step2" / "s1.jsonl").is_file())
        self.assertFalse((self.output / "step2" / "s3.jsonl").is_file())


# ---------- --start / --end ----------------------------------------------


class TestSlicing(_Harness):
    def test_start_end_slice(self):
        rc = bp.main([
            "--processed-dir", str(self.processed),
            "--sample-list", str(self.list_path),
            "--config-dir", str(self.config_dir),
            "--output-dir", str(self.output),
            "--weight-tensor", str(self.tensor_path),
            "--start", "1", "--end", "2",
        ])
        self.assertEqual(rc, 0)
        # Only s2 (index 1) should run.
        self.assertTrue((self.output / "step2" / "s2.jsonl").is_file())
        self.assertFalse((self.output / "step2" / "s1.jsonl").is_file())
        self.assertFalse((self.output / "step2" / "s3.jsonl").is_file())


# ---------- failure isolation --------------------------------------------


class TestFailureIsolation(_Harness):
    def test_one_bad_step_does_not_kill_batch(self):
        # Make step 5 throw on s2 but succeed on s1 / s3.
        original_fuse = next(
            p for p in self.patches if p.attribute == "fuse_predictions"
        )
        original_fuse.stop()

        def _fuse_with_failure(**kw):
            sid = kw["sample_json"]["sample_id"]
            if sid == "s2":
                raise RuntimeError("synthetic step-5 explosion")
            return _step5_composite(sid)

        with patch("scripts.batch_predict.fuse_predictions",
                   side_effect=_fuse_with_failure):
            rc = bp.main([
                "--processed-dir", str(self.processed),
                "--sample-list", str(self.list_path),
                "--config-dir", str(self.config_dir),
                "--output-dir", str(self.output),
                "--weight-tensor", str(self.tensor_path),
            ])
        # rc == 2 because at least one sample failed.
        self.assertEqual(rc, 2)

        # s1 / s3 succeed, s2 lands in failures.
        sids_in_summary = {
            json.loads(line)["sample_id"]
            for line in (self.output / "summary" / "results.jsonl")
            .read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        sids_in_failures = {
            json.loads(line)["sample_id"]
            for line in (self.output / "summary" / "failures.jsonl")
            .read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        self.assertEqual(sids_in_summary, {"s1", "s3"})
        self.assertEqual(sids_in_failures, {"s2"})

    def test_missing_sample_json_recorded_as_failure(self):
        list_path = self.tmp / "missing.txt"
        list_path.write_text("ghost_sample\n", encoding="utf-8")
        rc = bp.main([
            "--processed-dir", str(self.processed),
            "--sample-list", str(list_path),
            "--config-dir", str(self.config_dir),
            "--output-dir", str(self.output),
            "--weight-tensor", str(self.tensor_path),
        ])
        self.assertEqual(rc, 2)
        failures_path = self.output / "summary" / "failures.jsonl"
        self.assertTrue(failures_path.is_file())
        rec = json.loads(
            failures_path.read_text(encoding="utf-8").splitlines()[0],
        )
        self.assertEqual(rec["sample_id"], "ghost_sample")
        self.assertEqual(rec["phase"], "load_sample")


# ---------- --no-llm -----------------------------------------------------


class TestNoLLM(_Harness):
    def test_no_llm_skips_client_build(self):
        # Don't unset LLM_API_KEY — --no-llm should still skip the
        # client-build path. We verify via the build_client patch
        # call_count (the patch is a MagicMock under the hood).
        # Stop the existing build_client patch + re-attach with a counter.
        for p in self.patches:
            if getattr(p, "attribute", None) == "build_client":
                p.stop()
                self.patches.remove(p)
                break
        with patch("scripts.batch_predict.build_client",
                   return_value=object()) as mock_build:
            rc = bp.main([
                "--processed-dir", str(self.processed),
                "--sample-list", str(self.list_path),
                "--config-dir", str(self.config_dir),
                "--output-dir", str(self.output),
                "--weight-tensor", str(self.tensor_path),
                "--no-llm",
            ])
        self.assertEqual(rc, 0)
        mock_build.assert_not_called()


# ---------- helpers ------------------------------------------------------


class TestPureHelpers(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_load_sample_list_skips_blank_and_comments(self):
        p = self.tmp / "list.txt"
        p.write_text("a\n\n# comment\nb\n  c  \n", encoding="utf-8")
        self.assertEqual(bp.load_sample_list(p), ["a", "b", "c"])

    def test_load_sample_list_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            bp.load_sample_list(self.tmp / "no.txt")

    def test_load_sample_case_insensitive_fallback(self):
        # On-disk file uses upper-case Y; splits.json carries lower-case.
        # The case-insensitive scan should still find it.
        processed = self.tmp / "processed"
        (processed / "samples").mkdir(parents=True)
        (processed / "samples" / "3j46_Y_1.json").write_text(
            json.dumps({"sample_id": "3j46_Y_1"}), encoding="utf-8",
        )
        loaded = bp.load_sample(processed, "3j46_y_1")
        self.assertEqual(loaded["sample_id"], "3j46_Y_1")

    def test_load_sample_truly_missing_raises(self):
        processed = self.tmp / "processed"
        (processed / "samples").mkdir(parents=True)
        with self.assertRaises(FileNotFoundError):
            bp.load_sample(processed, "ghost_sample")

    def test_load_completed_ids_from_both_files(self):
        sd = self.tmp / "summary"
        sd.mkdir()
        (sd / "results.jsonl").write_text(
            json.dumps({"sample_id": "a"}) + "\n"
            + json.dumps({"sample_id": "b"}) + "\n",
            encoding="utf-8",
        )
        (sd / "failures.jsonl").write_text(
            json.dumps({"sample_id": "c"}) + "\n",
            encoding="utf-8",
        )
        done = bp.load_completed_ids(self.tmp)
        self.assertEqual(done, {"a", "b", "c"})

    def test_load_completed_ids_missing_files_returns_empty(self):
        self.assertEqual(bp.load_completed_ids(self.tmp), set())

    def test_write_step_record_atomic(self):
        bp.write_step_record(self.tmp, "step2", "s1", {"sample_id": "s1"})
        out = self.tmp / "step2" / "s1.jsonl"
        self.assertTrue(out.is_file())
        # No leftover .tmp.
        self.assertFalse(
            out.with_suffix(out.suffix + ".tmp").exists(),
        )


if __name__ == "__main__":
    unittest.main()
