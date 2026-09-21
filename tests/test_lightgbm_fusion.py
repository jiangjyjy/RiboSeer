"""Tests for the HARMONY fusion API used by the pipeline.

LightGBM itself is not required: the model is injected as a deterministic
stand-in, so these tests pin the *contract* the pipeline relies on —
building a per-sample bundle, predicting per-residue probabilities, and
turning them into the ``CompositeResult`` that VERDICT/MEMORY consume.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from argparse import Namespace  # noqa: E402

from scripts.batch_predict import resolve_fusion_model  # noqa: E402
from step5_fusion.lightgbm_fusion import (  # noqa: E402
    fuse_sample, sample_for_prediction,
)


class _FakeModel:
    """Per-row score = mean of the non-NaN columns — enough to exercise the
    feature matrix and the record assembly without any ML dependency."""

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            return np.nanmean(np.where(np.isnan(X), np.nan, X), axis=1).filled(0.0) \
                if hasattr(np.nanmean(np.where(np.isnan(X), np.nan, X), axis=1), "filled") \
                else np.nan_to_num(np.nanmean(np.where(np.isnan(X), np.nan, X), axis=1))


def _pred(tool_id, *, pae=None, conf=None, binding=None):
    return {
        "tool_id": tool_id, "success": True,
        "per_residue_pae_score": pae or {},
        "per_residue_confidence": conf or {},
        "binding_protein_residues": binding or [],
    }


def _sample(sid="s1", length=6, binding=(1, 2, 4)):
    return {
        "sample_id": sid,
        "protein": {"length": length, "sequence": "A" * length,
                    "resolved_residues": list(range(1, length + 1))},
        "rna": {"length": 20, "sequence": "G" * 20},
        "interaction": {"binding_protein_residues": list(binding)},
    }


def _step4_record(sid="s1"):
    return {
        "sample_id": sid,
        "predictions": [
            _pred("boltz2", pae={i: 0.9 - 0.1 * i for i in range(1, 7)},
                  binding=[1, 2]),
            _pred("equipnas", conf={i: 0.5 + 0.05 * i for i in range(1, 7)},
                  binding=[4]),
        ],
    }


class TestSampleForPrediction(unittest.TestCase):
    def test_builds_a_bundle_without_ground_truth(self):
        sample = sample_for_prediction("s1", _sample(), _step4_record())
        self.assertIsNotNone(sample)
        self.assertEqual(sample.sid, "s1")
        self.assertEqual(sample.residue_ids, [1, 2, 3, 4, 5, 6])
        self.assertEqual(sample.protein_len, 6)
        self.assertEqual(sample.available_tools, ["boltz2", "equipnas"])
        # the cauto profile is filled in, so SCOPE never sees an empty block
        self.assertTrue(sample.cauto_profile)

    def test_requires_ground_truth_only_when_asked(self):
        no_gt = _sample(binding=())
        self.assertIsNotNone(sample_for_prediction("s1", no_gt, _step4_record()))
        self.assertIsNone(
            sample_for_prediction("s1", no_gt, _step4_record(),
                                  require_ground_truth=True))

    def test_none_without_step4_or_length(self):
        self.assertIsNone(sample_for_prediction("s1", _sample(), None))
        empty = _sample()
        empty["protein"] = {"length": 0, "sequence": ""}
        self.assertIsNone(sample_for_prediction("s1", empty, _step4_record()))


class TestFuseSample(unittest.TestCase):
    def test_returns_a_composite_result_in_unit_range(self):
        composite = fuse_sample(
            _FakeModel(), sample_id="s1", sample_json=_sample(),
            step4_record=_step4_record(),
        )
        self.assertEqual(composite.sample_id, "s1")
        self.assertEqual(len(composite.per_residue_probability), 6)
        for p in composite.per_residue_probability.values():
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_reports_the_active_tools_and_the_model(self):
        composite = fuse_sample(
            _FakeModel(), sample_id="s1", sample_json=_sample(),
            step4_record=_step4_record(),
        )
        self.assertEqual(composite.tools_fused, ["boltz2", "equipnas"])
        self.assertEqual(set(composite.tool_weights), {"boltz2", "equipnas"})
        self.assertIn("LightGBM", composite.fusion_rationale)

    def test_threshold_selects_the_binding_set(self):
        class _AllHigh(_FakeModel):
            def predict(self, X):
                return np.ones(len(X))

        composite = fuse_sample(
            _AllHigh(), sample_id="s1", sample_json=_sample(),
            step4_record=_step4_record(), threshold=0.5,
        )
        self.assertEqual(composite.binding_protein_residues, [1, 2, 3, 4, 5, 6])

    def test_polish_is_mentioned_when_actions_are_passed(self):
        composite = fuse_sample(
            _FakeModel(), sample_id="s1", sample_json=_sample(),
            step4_record=_step4_record(), polish_actions={"s1": {"rounds": []}},
        )
        self.assertIn("POLISH", composite.fusion_rationale)

    def test_raises_on_a_sample_without_step4(self):
        with self.assertRaises(ValueError):
            fuse_sample(_FakeModel(), sample_id="s1", sample_json=_sample(),
                        step4_record=None)


class TestResolveFusionModel(unittest.TestCase):
    def _args(self, **over):
        base = dict(fusion="auto", fusion_model_dir=None,
                    fusion_train_step4_dir=None, fusion_train_list=None,
                    fusion_scope_dir=None, fusion_maestro_dir=None,
                    processed_dir=Path(tempfile.mkdtemp()))
        base.update(over)
        return Namespace(**base)

    def test_noisy_or_skips_the_model_entirely(self):
        with mock.patch("scripts.batch_predict.load_lightgbm_model") as load:
            self.assertIsNone(
                resolve_fusion_model(self._args(fusion="noisy-or"),
                                     Path(tempfile.mkdtemp())))
            load.assert_not_called()

    def test_lightgbm_without_a_bundle_is_an_error(self):
        with self.assertRaises(RuntimeError):
            resolve_fusion_model(self._args(fusion="lightgbm"),
                                 Path(tempfile.mkdtemp()))

    def test_auto_falls_back_to_noisy_or(self):
        self.assertIsNone(
            resolve_fusion_model(self._args(), Path(tempfile.mkdtemp())))

    def test_loads_an_existing_bundle(self):
        tmp = Path(tempfile.mkdtemp())
        model_dir = tmp / "harmony_model"
        model_dir.mkdir()
        (model_dir / "model.txt").write_text("stub", encoding="utf-8")
        sentinel = object()
        with mock.patch("scripts.batch_predict.load_lightgbm_model",
                        return_value=sentinel) as load:
            got = resolve_fusion_model(
                self._args(fusion="lightgbm", fusion_model_dir=model_dir), tmp)
            self.assertIs(got, sentinel)
            load.assert_called_once_with(model_dir)

    def test_trains_and_caches_when_given_training_inputs(self):
        tmp = Path(tempfile.mkdtemp())
        list_path = tmp / "train.txt"
        list_path.write_text("s1\n", encoding="utf-8")
        model = _FakeModel()
        with mock.patch("scripts.batch_predict.collect_samples",
                        return_value=["a-sample"]) as collect, \
                mock.patch("scripts.batch_predict.train_fullsystem_model",
                           return_value=model) as train, \
                mock.patch("scripts.batch_predict.save_lightgbm_model") as save:
            got = resolve_fusion_model(
                self._args(fusion="lightgbm",
                           fusion_train_step4_dir=tmp / "step4",
                           fusion_train_list=list_path), tmp)
        self.assertIs(got, model)
        collect.assert_called_once()
        train.assert_called_once()
        save.assert_called_once()

    def test_training_inputs_without_samples_is_an_error(self):
        tmp = Path(tempfile.mkdtemp())
        list_path = tmp / "train.txt"
        list_path.write_text("s1\n", encoding="utf-8")
        with mock.patch("scripts.batch_predict.collect_samples", return_value=[]):
            with self.assertRaises(RuntimeError):
                resolve_fusion_model(
                    self._args(fusion="lightgbm",
                               fusion_train_step4_dir=tmp / "step4",
                               fusion_train_list=list_path), tmp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
