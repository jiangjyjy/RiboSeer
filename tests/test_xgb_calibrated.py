"""Mock tests for the AgentCalibratedXGBFusion path.

This is the "agent + small model" architecture: agent's noisy-OR
fusion (step5) stays primary, XGBoost only re-ranks. Coverage:

  - Feature row prepends [agent_fusion_prob, agent_in_binding]
    before the WeightOptimizer base columns.
  - End-to-end training requires step4 + step5 + GT; rows with no
    step5 enrichment are dropped.
  - predict_sample requires agent_probs + agent_binding_set; passing
    None raises (silent zero-out would break the calibration claim).
  - save/load round-trip preserves predictions and the calibrated marker.
  - Train CLI ``--mode calibrated`` requires --train-step5-dir.
  - Evaluate CLI auto-detects the calibrated bundle from
    xgb_meta.json and forwards step5 agent output at inference.
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

from step5_fusion.xgb_fusion import (  # noqa: E402
    AGENT_FEATURE_NAMES, AgentCalibratedXGBFusion, XGB_OK, XGBFusion,
)


@unittest.skipUnless(XGB_OK, "xgboost not installed")
class _XGBCalibratedBase(unittest.TestCase):
    pass


# ---- synthetic fixtures -------------------------------------------------


def _write_step4(step4_dir: Path, sid: str, predictions: list[dict]) -> None:
    step4_dir.mkdir(parents=True, exist_ok=True)
    rec = {"sample_id": sid,
           "tools_run": [p["tool_id"] for p in predictions],
           "predictions": predictions}
    (step4_dir / f"{sid}.jsonl").write_text(
        json.dumps(rec) + "\n", encoding="utf-8",
    )


def _write_step5(step5_dir: Path, sid: str, *,
                 per_residue_probability: dict,
                 binding_residues: list[int]) -> None:
    step5_dir.mkdir(parents=True, exist_ok=True)
    (step5_dir / f"{sid}.jsonl").write_text(json.dumps({
        "sample_id": sid,
        "per_residue_probability": {
            str(k): float(v) for k, v in per_residue_probability.items()
        },
        "binding_protein_residues": list(binding_residues),
        "threshold": 0.5,
    }) + "\n", encoding="utf-8")


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


def _make_calibrated_dataset(
    tmp: Path, *, n_samples: int = 6, length: int = 8,
) -> tuple[Path, Path, Path, list[str]]:
    """6 samples × 8 residues, GT = {1, 4, 7}. step5 records carry
    agent fusion probs that almost-but-not-quite match GT — leaves
    a residual for XGBoost to calibrate."""
    step4 = tmp / "step4"
    step5 = tmp / "step5"
    processed = tmp / "processed"
    sids: list[str] = []
    gt = [1, 4, 7]
    for i in range(n_samples):
        sid = f"s{i:03d}"
        sids.append(sid)
        seq = "MKRYAVCDEFGHIKLM"[:length]
        _write_sample(processed, sid, length=length, sequence=seq, gt=gt)

        # step4 — three tools, mostly aligned with GT.
        pae = {str(r): 0.85 if r in gt else 0.10 for r in range(1, length + 1)}
        conf = {str(r): 0.7 if r in gt else 0.05 for r in range(1, length + 1)}
        p2 = {str(r): (0.6 if (r in gt and r % 2 == 1) else 0.15)
              for r in range(1, length + 1)}
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

        # step5 — agent fusion. Strong on residue 1/7 (everyone agrees),
        # weaker on residue 4 (p2rank disagreed). Random-ish noise on
        # non-binding residues so the calibrator has variance to fit.
        agent_probs = {
            r: (0.92 if r in (1, 7)
                else 0.55 if r == 4
                else 0.05 + 0.02 * (i % 3))
            for r in range(1, length + 1)
        }
        _write_step5(step5, sid,
                     per_residue_probability=agent_probs,
                     binding_residues=list(gt))
    return step4, step5, processed, sids


def _small_cfg(extra: dict = None) -> dict:
    cfg = {
        "use_residue_type": False,
        "use_gating": True,
        "n_estimators": 20,
        "max_depth": 3,
        "learning_rate": 0.3,
        "min_child_weight": 1,  # lower than prod default so synthetic
                                # data with 6 samples still fits leaves
        "seed": 0,
    }
    if extra:
        cfg.update(extra)
    return cfg


# ---- feature layout -----------------------------------------------------


class TestCalibratedFeatureLayout(_XGBCalibratedBase):
    def test_agent_features_prepended(self):
        f = AgentCalibratedXGBFusion(_small_cfg())
        f.tool_order = ["a", "b"]
        f.feature_builder.tool_order = ["a", "b"]
        row = f.build_features(
            {"a": 0.3, "b": 0.7}, "K",
            {"a": True, "b": False},
            agent_prob=0.85, agent_in_binding=True,
        )
        # Layout: [agent_prob, agent_in_binding,
        #          tool:a, tool:b, gate:a, gate:b, vote_count]
        self.assertEqual(row.shape, (7,))
        self.assertAlmostEqual(row[0], 0.85)
        self.assertAlmostEqual(row[1], 1.0)
        self.assertAlmostEqual(row[2], 0.3)   # tool:a
        self.assertAlmostEqual(row[3], 0.7)   # tool:b
        self.assertAlmostEqual(row[4], 1.0)   # gate:a
        self.assertAlmostEqual(row[5], 0.0)   # gate:b
        self.assertAlmostEqual(row[6], 0.5)   # vote_count

    def test_full_feature_names(self):
        f = AgentCalibratedXGBFusion(_small_cfg())
        f.tool_order = ["a", "b"]
        f.feature_builder.tool_order = ["a", "b"]
        names = f._full_feature_names()
        self.assertEqual(names[:2], list(AGENT_FEATURE_NAMES))
        # Base names follow.
        self.assertIn("tool:a", names)
        self.assertIn("vote_count", names)

    def test_agent_in_binding_false_encodes_as_zero(self):
        f = AgentCalibratedXGBFusion(_small_cfg())
        f.tool_order = ["a"]
        f.feature_builder.tool_order = ["a"]
        row = f.build_features(
            {"a": 0.5}, "K", {"a": False},
            agent_prob=0.3, agent_in_binding=False,
        )
        self.assertAlmostEqual(row[0], 0.3)
        self.assertAlmostEqual(row[1], 0.0)


# ---- training -----------------------------------------------------------


class TestCalibratedTrain(_XGBCalibratedBase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4, self.step5, self.processed, self.sids = \
            _make_calibrated_dataset(self.tmp)

    def test_train_returns_calibrated_report(self):
        f = AgentCalibratedXGBFusion(_small_cfg())
        report = f.train(
            step4_dir=self.step4, step5_dir=self.step5,
            processed_dir=self.processed, verbose=False,
        )
        self.assertEqual(report["mode"], "calibrated")
        # 2 agent + 3 tool + 3 gate + 1 vote = 9 features.
        self.assertEqual(report["n_features"], 9)
        self.assertGreater(report["n_rows"], 0)
        # Pearson R should be substantial — agent already gave a
        # near-perfect ranking and the calibrator only refines it.
        self.assertIsNotNone(report["pearson_r"])
        self.assertGreater(report["pearson_r"], 0.5)
        # Top features list is non-empty.
        self.assertGreater(len(report["feature_importances_top"]), 0)

    def test_agent_features_appear_in_top_importances(self):
        # The agent fusion prob is engineered to be the strongest
        # signal, so it should show up among the top-3 features.
        f = AgentCalibratedXGBFusion(_small_cfg())
        f.train(
            step4_dir=self.step4, step5_dir=self.step5,
            processed_dir=self.processed, verbose=False,
        )
        top_names = [
            entry["feature"]
            for entry in f.training_report["feature_importances_top"][:5]
        ]
        # At least one of the two agent features should be in top 5.
        self.assertTrue(
            any(n in top_names for n in AGENT_FEATURE_NAMES),
            f"expected an agent feature in top 5, got: {top_names}",
        )

    def test_drops_rows_when_step5_missing_for_sample(self):
        # Remove step5 for one sample; collector should drop only
        # those rows, not the whole sample bundle.
        (self.step5 / f"{self.sids[0]}.jsonl").unlink()
        f = AgentCalibratedXGBFusion(_small_cfg())
        report = f.train(
            step4_dir=self.step4, step5_dir=self.step5,
            processed_dir=self.processed, verbose=False,
        )
        self.assertGreater(report["n_dropped_no_agent"], 0)
        # But still trained on the other samples.
        self.assertGreater(report["n_rows"], 0)

    def test_train_with_no_step5_at_all_raises(self):
        empty5 = self.tmp / "empty_step5"
        empty5.mkdir()
        f = AgentCalibratedXGBFusion(_small_cfg())
        with self.assertRaises(RuntimeError):
            f.train(
                step4_dir=self.step4, step5_dir=empty5,
                processed_dir=self.processed, verbose=False,
            )


# ---- inference ----------------------------------------------------------


class TestCalibratedPredict(_XGBCalibratedBase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4, self.step5, self.processed, self.sids = \
            _make_calibrated_dataset(self.tmp)
        self.fusion = AgentCalibratedXGBFusion(_small_cfg())
        self.fusion.train(
            step4_dir=self.step4, step5_dir=self.step5,
            processed_dir=self.processed, verbose=False,
        )

    def _load_sample(self, sid: str) -> dict:
        return json.loads(
            (self.processed / "samples" / f"{sid}.json")
            .read_text(encoding="utf-8")
        )

    def _load_step4(self, sid: str) -> dict:
        return json.loads(
            (self.step4 / f"{sid}.jsonl").read_text(encoding="utf-8")
        )

    def _load_step5(self, sid: str) -> dict:
        return json.loads(
            (self.step5 / f"{sid}.jsonl").read_text(encoding="utf-8")
        )

    def test_predict_requires_agent_inputs(self):
        sample = self._load_sample(self.sids[0])
        s4 = self._load_step4(self.sids[0])
        # Missing both kwargs → ValueError.
        with self.assertRaises(ValueError):
            self.fusion.predict_sample(
                s4["predictions"], sample["protein"]["sequence"],
                sample["protein"]["length"],
            )
        # Only one kwarg → also raises.
        with self.assertRaises(ValueError):
            self.fusion.predict_sample(
                s4["predictions"], sample["protein"]["sequence"],
                sample["protein"]["length"],
                agent_probs={"1": 0.9}, agent_binding_set=None,
            )

    def test_predict_separates_binding_residues(self):
        sample = self._load_sample(self.sids[0])
        s4 = self._load_step4(self.sids[0])
        s5 = self._load_step5(self.sids[0])
        per_res = self.fusion.predict_sample(
            s4["predictions"], sample["protein"]["sequence"],
            sample["protein"]["length"],
            agent_probs=s5["per_residue_probability"],
            agent_binding_set=set(s5["binding_protein_residues"]),
        )
        gt = set(sample["interaction"]["binding_protein_residues"])
        binding_mean = sum(per_res[r] for r in gt) / len(gt)
        non_mean = sum(per_res[r] for r in per_res if r not in gt) / max(
            sum(1 for r in per_res if r not in gt), 1,
        )
        self.assertGreater(binding_mean, non_mean)
        for v in per_res.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_agent_input_actually_changes_prediction(self):
        # Same sample, two different fake agent_probs distributions.
        # The model's prediction must shift — otherwise the
        # calibration channel isn't being used.
        sample = self._load_sample(self.sids[0])
        s4 = self._load_step4(self.sids[0])
        length = sample["protein"]["length"]
        # All-low agent → predictions should be lower than all-high.
        low = {str(r): 0.05 for r in range(1, length + 1)}
        high = {str(r): 0.95 for r in range(1, length + 1)}
        binding = set(range(1, length + 1))  # match keys regardless

        per_low = self.fusion.predict_sample(
            s4["predictions"], sample["protein"]["sequence"], length,
            agent_probs=low, agent_binding_set=set(),
        )
        per_high = self.fusion.predict_sample(
            s4["predictions"], sample["protein"]["sequence"], length,
            agent_probs=high, agent_binding_set=binding,
        )
        mean_low = sum(per_low.values()) / len(per_low)
        mean_high = sum(per_high.values()) / len(per_high)
        self.assertGreater(mean_high, mean_low)


# ---- save / load --------------------------------------------------------


class TestCalibratedSaveLoad(_XGBCalibratedBase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4, self.step5, self.processed, self.sids = \
            _make_calibrated_dataset(self.tmp)
        self.fusion = AgentCalibratedXGBFusion(_small_cfg())
        self.fusion.train(
            step4_dir=self.step4, step5_dir=self.step5,
            processed_dir=self.processed, verbose=False,
        )

    def test_save_writes_calibrated_marker(self):
        bundle = self.tmp / "bundle"
        self.fusion.save(bundle)
        meta = json.loads(
            (bundle / "xgb_meta.json").read_text(encoding="utf-8")
        )
        self.assertEqual(meta["model_type"], "xgboost_calibrated")
        self.assertTrue(meta.get("calibrated"))
        self.assertEqual(
            meta["agent_feature_names"], list(AGENT_FEATURE_NAMES),
        )

    def test_load_round_trip_preserves_predictions(self):
        bundle = self.tmp / "bundle_rt"
        self.fusion.save(bundle)
        sample = json.loads(
            (self.processed / "samples" / f"{self.sids[0]}.json")
            .read_text(encoding="utf-8")
        )
        s4 = json.loads(
            (self.step4 / f"{self.sids[0]}.jsonl")
            .read_text(encoding="utf-8")
        )
        s5 = json.loads(
            (self.step5 / f"{self.sids[0]}.jsonl")
            .read_text(encoding="utf-8")
        )
        before = self.fusion.predict_sample(
            s4["predictions"], sample["protein"]["sequence"],
            sample["protein"]["length"],
            agent_probs=s5["per_residue_probability"],
            agent_binding_set=set(s5["binding_protein_residues"]),
        )
        loaded = AgentCalibratedXGBFusion.load(bundle)
        self.assertEqual(loaded.tool_order, self.fusion.tool_order)
        after = loaded.predict_sample(
            s4["predictions"], sample["protein"]["sequence"],
            sample["protein"]["length"],
            agent_probs=s5["per_residue_probability"],
            agent_binding_set=set(s5["binding_protein_residues"]),
        )
        for r in before:
            self.assertAlmostEqual(before[r], after[r], places=5)


# ---- train_xgb_fusion CLI with --mode calibrated ------------------------


import scripts.train_xgb_fusion as train_cli  # noqa: E402


class TestTrainCalibratedCli(_XGBCalibratedBase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4, self.step5, self.processed, self.sids = \
            _make_calibrated_dataset(self.tmp)
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

    def test_calibrated_mode_writes_bundle(self):
        out_dir = self.tmp / "model"
        rc = train_cli.main([
            "--train-step4-dir", str(self.step4),
            "--train-step5-dir", str(self.step5),
            "--processed-dir", str(self.processed),
            "--config", str(self.cfg_path),
            "--output", str(out_dir),
            "--mode", "calibrated",
            "--quiet",
        ])
        self.assertEqual(rc, 0)
        for name in ("standardizer.json", "xgb_model.json",
                     "xgb_meta.json", "training_report.json"):
            self.assertTrue((out_dir / name).is_file(), f"missing {name}")
        # Bundle marker is set.
        meta = json.loads(
            (out_dir / "xgb_meta.json").read_text(encoding="utf-8")
        )
        self.assertEqual(meta["model_type"], "xgboost_calibrated")

    def test_calibrated_without_step5_dir_returns_1(self):
        out_dir = self.tmp / "model_bad"
        rc = train_cli.main([
            "--train-step4-dir", str(self.step4),
            "--processed-dir", str(self.processed),
            "--output", str(out_dir),
            "--mode", "calibrated",
            "--quiet",
        ])
        self.assertEqual(rc, 1)
        self.assertFalse(out_dir.exists())

    def test_mode_from_yaml_when_cli_omitted(self):
        cfg_path = self.tmp / "cfg_calibrated.yaml"
        cfg_path.write_text(
            "xgb_fusion:\n"
            "  mode: calibrated\n"
            "  use_residue_type: false\n"
            "  use_gating: true\n"
            "  n_estimators: 20\n"
            "  max_depth: 3\n"
            "  learning_rate: 0.3\n"
            "  min_child_weight: 1\n"
            "  seed: 0\n",
            encoding="utf-8",
        )
        out_dir = self.tmp / "model_yaml"
        rc = train_cli.main([
            "--train-step4-dir", str(self.step4),
            "--train-step5-dir", str(self.step5),
            "--processed-dir", str(self.processed),
            "--config", str(cfg_path),
            "--output", str(out_dir),
            "--quiet",
            # no --mode flag — should pick up calibrated from YAML
        ])
        self.assertEqual(rc, 0)
        meta = json.loads(
            (out_dir / "xgb_meta.json").read_text(encoding="utf-8")
        )
        self.assertEqual(meta["model_type"], "xgboost_calibrated")

    def test_standalone_mode_unchanged_by_step5_flag(self):
        # Passing --train-step5-dir without --mode calibrated should
        # be ignored (standalone mode doesn't use it).
        out_dir = self.tmp / "model_standalone"
        rc = train_cli.main([
            "--train-step4-dir", str(self.step4),
            "--train-step5-dir", str(self.step5),
            "--processed-dir", str(self.processed),
            "--config", str(self.cfg_path),
            "--output", str(out_dir),
            "--quiet",
        ])
        self.assertEqual(rc, 0)
        meta = json.loads(
            (out_dir / "xgb_meta.json").read_text(encoding="utf-8")
        )
        self.assertEqual(meta["model_type"], "xgboost")


# ---- evaluate auto-detect routes to calibrated --------------------------


import scripts.evaluate_learned_fusion as eval_cli  # noqa: E402


class TestEvaluateCalibratedAutoDetect(_XGBCalibratedBase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4, self.step5, self.processed, self.sids = \
            _make_calibrated_dataset(self.tmp)
        self.bundle = self.tmp / "calibrated_bundle"
        f = AgentCalibratedXGBFusion(_small_cfg())
        f.train(
            step4_dir=self.step4, step5_dir=self.step5,
            processed_dir=self.processed, verbose=False,
        )
        f.save(self.bundle)

    def test_auto_detects_calibrated_bundle(self):
        out = self.tmp / "eval_calibrated"
        rc = eval_cli.main([
            "--test-step4-dir", str(self.step4),
            "--test-step5-dir", str(self.step5),
            "--processed-dir", str(self.processed),
            "--fusion-model", str(self.bundle),
            "--output", str(out),
        ])
        self.assertEqual(rc, 0)
        metrics = json.loads(
            (out / "learned_fusion_metrics.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(metrics["model_type"], "xgboost_calibrated")
        # Both methods reported (learned + noisy_or_baseline).
        self.assertIn("learned_fusion", metrics["methods"])
        self.assertIn("noisy_or_baseline", metrics["methods"])

    def test_calibrated_bundle_without_step5_dir_returns_1(self):
        # Calibrated bundle needs the agent input; missing
        # --test-step5-dir is a hard error, not a silent fallback.
        out = self.tmp / "eval_no_step5"
        rc = eval_cli.main([
            "--test-step4-dir", str(self.step4),
            "--processed-dir", str(self.processed),
            "--fusion-model", str(self.bundle),
            "--output", str(out),
        ])
        self.assertEqual(rc, 1)


# ---- defensive: existing standalone bundles still detect as xgboost -----


class TestStandaloneStillStandalone(_XGBCalibratedBase):
    """The new auto-detect must NOT misroute a plain XGBFusion bundle
    into the calibrated loader — the calibrated marker is the only
    thing that should distinguish them."""

    def test_standalone_bundle_routes_to_xgboost(self):
        from step5_fusion.xgb_fusion import XGBFusion as Standalone
        tmp = Path(tempfile.mkdtemp())
        step4, _step5, processed, _sids = _make_calibrated_dataset(tmp)
        f = Standalone(_small_cfg())
        f.train(step4_dir=step4, processed_dir=processed, verbose=False)
        bundle = tmp / "standalone_bundle"
        f.save(bundle)

        # _detect_model_type should pick "xgboost", not "xgboost_calibrated".
        from scripts.evaluate_learned_fusion import _detect_model_type
        self.assertEqual(_detect_model_type(bundle), "xgboost")


if __name__ == "__main__":
    unittest.main()
