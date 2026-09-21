"""Mock tests for XGBFusion (skipped when xgboost can't import).

Reuses the same synthetic step4 dataset shape the Ridge tests use,
so the two model classes are exercised on byte-identical input.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from step5_fusion.xgb_fusion import XGB_OK, XGBFusion  # noqa: E402


# Skip the whole module on a host that doesn't have xgboost installed.
# The CI / dev env normally does (we install it explicitly per the
# task spec); the skip guard keeps the rest of the suite green when
# someone clones to a fresh env.
@unittest.skipUnless(XGB_OK, "xgboost not installed")
class _XGBTestBase(unittest.TestCase):
    pass


# ---- synthetic dataset (mirrors test_learned_fusion's helper) -----------


def _write_step4(step4_dir: Path, sid: str, predictions: list[dict]) -> None:
    step4_dir.mkdir(parents=True, exist_ok=True)
    rec = {"sample_id": sid,
           "tools_run": [p["tool_id"] for p in predictions],
           "predictions": predictions}
    (step4_dir / f"{sid}.jsonl").write_text(
        json.dumps(rec) + "\n", encoding="utf-8",
    )


def _write_sample(processed: Path, sid: str, *, length: int,
                  sequence: str, gt: list[int]) -> None:
    samples = processed / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    sample = {
        "sample_id": sid,
        "protein": {"length": length, "sequence": sequence},
        "rna": {"length": 5, "sequence": "GCGCG"},
        "interaction": {"binding_protein_residues": gt},
    }
    (samples / f"{sid}.json").write_text(
        json.dumps(sample), encoding="utf-8",
    )


def _make_synthetic_dataset(
    tmp: Path, *, n_samples: int = 6, length: int = 8,
) -> tuple[Path, Path, list[str]]:
    step4 = tmp / "step4"
    processed = tmp / "processed"
    sids: list[str] = []
    for i in range(n_samples):
        sid = f"s{i:03d}"
        sids.append(sid)
        gt = [1, 4, 7]
        seq = "MKRYAVCDEFGHIKLM"[:length]
        _write_sample(processed, sid, length=length, sequence=seq, gt=gt)
        pae = {}
        for r in range(1, length + 1):
            base = 0.85 if r in gt else 0.10
            pae[str(r)] = round(base + (i * 0.01) % 0.05, 4)
        conf = {}
        for r in range(1, length + 1):
            base = 0.7 if r in gt else 0.05
            conf[str(r)] = round(base + (i * 0.005) % 0.03, 4)
        p2 = {}
        for r in range(1, length + 1):
            base = 0.6 if (r in gt and r % 2 == 1) else 0.15
            p2[str(r)] = round(base + (i * 0.01) % 0.04, 4)
        _write_step4(step4, sid, [
            {"tool_id": "boltz2", "category": "A", "sample_id": sid,
             "success": True, "binding_protein_residues": list(gt),
             "per_residue_pae_score": pae},
            {"tool_id": "equipnas", "category": "C", "sample_id": sid,
             "success": True, "binding_protein_residues": list(gt),
             "per_residue_confidence": conf},
            {"tool_id": "p2rank", "category": "B", "sample_id": sid,
             "success": True,
             "binding_protein_residues": [r for r in gt if r % 2 == 1],
             "per_residue_confidence": p2},
        ])
    return step4, processed, sids


def _small_xgb_cfg(extra: dict = None) -> dict:
    """Tiny XGBoost so the test trains in <1 s."""
    cfg = {
        "use_residue_type": False,
        "use_gating": True,
        "n_estimators": 20,
        "max_depth": 3,
        "learning_rate": 0.3,
        "min_child_weight": 1,
        "seed": 0,
    }
    if extra:
        cfg.update(extra)
    return cfg


# ---- test cases ---------------------------------------------------------


class TestXGBFusionTrain(_XGBTestBase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4, self.processed, self.sids = \
            _make_synthetic_dataset(self.tmp)

    def test_train_returns_report_and_separates_classes(self):
        fusion = XGBFusion(_small_xgb_cfg())
        report = fusion.train(
            step4_dir=self.step4, processed_dir=self.processed,
            verbose=False,
        )
        # Report shape.
        self.assertGreater(report["n_rows"], 0)
        self.assertEqual(report["n_features"], 7)  # 3 tool + 3 gate + 1 vote
        self.assertGreater(report["scale_pos_weight"], 1.0)
        # Pearson R on the train set should be substantially > 0
        # given the strong signal in the synthetic data.
        self.assertIsNotNone(report["pearson_r"])
        self.assertGreater(report["pearson_r"], 0.3)
        # Top features list is non-empty.
        self.assertGreater(len(report["feature_importances_top"]), 0)
        # tool_order frozen alphabetically.
        self.assertEqual(fusion.tool_order, ["boltz2", "equipnas", "p2rank"])

    def test_predict_sample_separates_binding_residues(self):
        fusion = XGBFusion(_small_xgb_cfg())
        fusion.train(
            step4_dir=self.step4, processed_dir=self.processed,
            verbose=False,
        )
        sample = json.loads(
            (self.processed / "samples" / f"{self.sids[0]}.json")
            .read_text(encoding="utf-8")
        )
        s4 = json.loads(
            (self.step4 / f"{self.sids[0]}.jsonl")
            .read_text(encoding="utf-8")
        )
        per_res = fusion.predict_sample(
            s4["predictions"], sample["protein"]["sequence"],
            sample["protein"]["length"],
        )
        gt = set(sample["interaction"]["binding_protein_residues"])
        binding_mean = sum(per_res[r] for r in gt) / len(gt)
        nonbinding = [r for r in per_res if r not in gt]
        non_mean = sum(per_res[r] for r in nonbinding) / len(nonbinding)
        self.assertGreater(binding_mean, non_mean)
        # All predictions in [0, 1].
        for v in per_res.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_train_with_no_data_raises(self):
        fusion = XGBFusion(_small_xgb_cfg())
        empty = self.tmp / "empty"
        empty.mkdir()
        # No step4 JSONLs under empty/, so the collector returns 0 rows.
        with self.assertRaises(RuntimeError):
            fusion.train(step4_dir=empty, processed_dir=self.processed,
                         verbose=False)


class TestXGBFusionSaveLoad(_XGBTestBase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4, self.processed, self.sids = \
            _make_synthetic_dataset(self.tmp)

    def test_save_writes_expected_files(self):
        fusion = XGBFusion(_small_xgb_cfg())
        fusion.train(
            step4_dir=self.step4, processed_dir=self.processed,
            verbose=False,
        )
        bundle = self.tmp / "bundle"
        fusion.save(bundle)
        for name in ("standardizer.json", "xgb_model.json",
                     "xgb_meta.json", "training_report.json"):
            self.assertTrue((bundle / name).is_file(), f"missing {name}")
        # Meta records the model_type marker for cross-version compat.
        meta = json.loads(
            (bundle / "xgb_meta.json").read_text(encoding="utf-8")
        )
        self.assertEqual(meta["model_type"], "xgboost")
        # Training report carries the headline metric.
        report = json.loads(
            (bundle / "training_report.json").read_text(encoding="utf-8")
        )
        self.assertIn("pearson_r", report)
        self.assertIn("scale_pos_weight", report)

    def test_load_round_trip_preserves_predictions(self):
        fusion = XGBFusion(_small_xgb_cfg())
        fusion.train(
            step4_dir=self.step4, processed_dir=self.processed,
            verbose=False,
        )
        bundle = self.tmp / "bundle_rt"
        fusion.save(bundle)

        sample = json.loads(
            (self.processed / "samples" / f"{self.sids[0]}.json")
            .read_text(encoding="utf-8")
        )
        s4 = json.loads(
            (self.step4 / f"{self.sids[0]}.jsonl")
            .read_text(encoding="utf-8")
        )
        before = fusion.predict_sample(
            s4["predictions"], sample["protein"]["sequence"],
            sample["protein"]["length"],
        )
        loaded = XGBFusion.load(bundle)
        # Loaded bundle preserves layout knobs.
        self.assertTrue(loaded.feature_builder.use_gating)
        self.assertEqual(loaded.tool_order, fusion.tool_order)
        after = loaded.predict_sample(
            s4["predictions"], sample["protein"]["sequence"],
            sample["protein"]["length"],
        )
        self.assertEqual(set(before.keys()), set(after.keys()))
        for r in before:
            self.assertAlmostEqual(before[r], after[r], places=5)

    def test_save_before_train_raises(self):
        fusion = XGBFusion(_small_xgb_cfg())
        with self.assertRaises(RuntimeError):
            fusion.save(self.tmp / "no_model")


class TestXGBFeatureLayoutMatchesRidge(_XGBTestBase):
    """Defensive check: XGBFusion's feature row must be byte-identical
    to the WeightOptimizer Ridge path with the same knobs. If this
    breaks, comparison numbers in the paper are silently wrong."""

    def test_feature_row_matches_weight_optimizer(self):
        from step5_fusion.weight_optimizer import WeightOptimizer

        cfg = {
            "use_residue_type": True,
            "use_gating": True,
            "use_cross_terms": False,
        }
        f = XGBFusion(cfg)
        f.tool_order = ["a", "b"]
        f.feature_builder.tool_order = list(f.tool_order)

        ridge = WeightOptimizer(
            use_residue_type=True, use_gating=True, use_cross_terms=False,
        )
        ridge.tool_order = ["a", "b"]

        scores = {"a": 0.5, "b": 0.7}
        flags = {"a": True, "b": False}
        a = f.build_features(scores, "K", flags)
        b = ridge.build_features(scores, "K", flags)
        np.testing.assert_array_equal(a, b)


# ---- evaluate-script auto-detect ----------------------------------------


import scripts.evaluate_learned_fusion as eval_cli  # noqa: E402


class TestEvaluateAutoDetect(_XGBTestBase):
    """The evaluate CLI should pick up an XGBoost bundle without
    --model-type and pick the Ridge path on a Ridge bundle."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4, self.processed, self.sids = \
            _make_synthetic_dataset(self.tmp)
        self.xgb_bundle = self.tmp / "xgb_bundle"
        # Train an XGB bundle.
        f = XGBFusion(_small_xgb_cfg())
        f.train(step4_dir=self.step4, processed_dir=self.processed,
                verbose=False)
        f.save(self.xgb_bundle)

    def test_auto_detects_xgboost_bundle(self):
        out = self.tmp / "eval_out"
        rc = eval_cli.main([
            "--test-step4-dir", str(self.step4),
            "--processed-dir", str(self.processed),
            "--fusion-model", str(self.xgb_bundle),
            "--output", str(out),
        ])
        self.assertEqual(rc, 0)
        # metrics.json records the resolved model type.
        metrics = json.loads(
            (out / "learned_fusion_metrics.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(metrics["model_type"], "xgboost")

    def test_unknown_bundle_returns_1(self):
        empty = self.tmp / "empty_bundle"
        empty.mkdir()
        rc = eval_cli.main([
            "--test-step4-dir", str(self.step4),
            "--processed-dir", str(self.processed),
            "--fusion-model", str(empty),
            "--output", str(self.tmp / "eval_no_bundle"),
        ])
        self.assertEqual(rc, 1)


# ---- train_xgb_fusion CLI -----------------------------------------------


import scripts.train_xgb_fusion as train_cli  # noqa: E402


class TestTrainXgbFusionCli(_XGBTestBase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4, self.processed, self.sids = \
            _make_synthetic_dataset(self.tmp)
        # Write a minimal YAML config so the CLI exercises the loader.
        self.cfg_path = self.tmp / "cfg.yaml"
        self.cfg_path.write_text(
            "xgb_fusion:\n"
            "  use_residue_type: false\n"
            "  use_gating: true\n"
            "  n_estimators: 20\n"
            "  max_depth: 3\n"
            "  learning_rate: 0.3\n"
            "  min_child_weight: 1\n"
            "  seed: 0\n",
            encoding="utf-8",
        )

    def test_writes_bundle_files(self):
        out_dir = self.tmp / "model"
        rc = train_cli.main([
            "--train-step4-dir", str(self.step4),
            "--processed-dir", str(self.processed),
            "--config", str(self.cfg_path),
            "--output", str(out_dir),
            "--quiet",
        ])
        self.assertEqual(rc, 0)
        for name in ("standardizer.json", "xgb_model.json",
                     "xgb_meta.json", "training_report.json"):
            self.assertTrue((out_dir / name).is_file(), f"missing {name}")

    def test_missing_step4_dir_returns_1(self):
        rc = train_cli.main([
            "--train-step4-dir", str(self.tmp / "no_such"),
            "--processed-dir", str(self.processed),
            "--output", str(self.tmp / "out_bad"),
            "--quiet",
        ])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
