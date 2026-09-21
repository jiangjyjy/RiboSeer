"""Mock tests for scripts/tables/table07_fusion_method.py.

LightGBM/XGBoost aren't required: the trainable estimators are patched with
deterministic fakes, so we exercise the headline 154-D tool resolution
(MANDATORY ∪ MAESTRO), the no-train raw-score methods over the selected
tools, the NaN→0 imputation split (sklearn vs tree), the 8-row aggregation,
and CSV I/O — without any ML dep.
"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.tables import table07_fusion_method as fm  # noqa: E402
from scripts.riboseer.config import MANDATORY_TOOLS  # noqa: E402


# ---- fixtures -----------------------------------------------------------


def _pred(tool_id, *, pae=None, conf=None, binding=None):
    return {"tool_id": tool_id, "success": True,
            "per_residue_pae_score": pae or {},
            "per_residue_confidence": conf or {},
            "binding_protein_residues": binding or []}


def _write_sample(proc, sid, length, binding, rna_len=40):
    samples = proc / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    doc = {"sample_id": sid,
           "protein": {"length": length, "sequence": "A" * length,
                       "resolved_residues": list(range(1, length + 1))},
           "rna": {"length": rna_len, "sequence": "G" * rna_len},
           "interaction": {"binding_protein_residues": list(binding)}}
    (samples / f"{sid}.json").write_text(json.dumps(doc), encoding="utf-8")


def _write_step4(s4, sid, preds):
    s4.mkdir(parents=True, exist_ok=True)
    (s4 / f"{sid}.jsonl").write_text(
        json.dumps({"sample_id": sid, "predictions": preds}) + "\n",
        encoding="utf-8")


def _varied_preds():
    # boltz2 (pae), equipnas + deeppocket (conf) → all in MANDATORY_7.
    return [
        _pred("boltz2", pae={i: round(0.95 - 0.13 * i, 3) for i in range(1, 7)},
              conf={i: 80 - i for i in range(1, 7)}, binding=[1, 2, 4]),
        _pred("equipnas", conf={i: 0.6 + 0.03 * i for i in range(1, 7)},
              binding=[1, 4]),
        _pred("deeppocket", conf={i: 0.4 + 0.05 * i for i in range(1, 7)},
              binding=[2, 3]),
    ]


def _make_samples():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        proc, s4 = td / "proc", td / "s4"
        ids = ("tr1", "tr2", "te1", "te2")
        for sid in ids:
            _write_sample(proc, sid, 6, binding=[1, 2, 4])
            _write_step4(s4, sid, _varied_preds())
        train = fm.collect_samples(s4, proc, ["tr1", "tr2"])
        test = fm.collect_samples(s4, proc, ["te1", "te2"])
    selections = {sid: {"selected_tools": ["boltz2"]} for sid in ids}
    return train, test, selections


class _SumModel:
    """sklearn-style fake: prediction = row sum of NaN→0 features."""

    def fit(self, X, y):
        self._d = X.shape[1]
        return self

    def predict(self, X):
        return np.nan_to_num(np.asarray(X, dtype=np.float64),
                             nan=0.0).sum(axis=1)


# ---- tests --------------------------------------------------------------


class TestSelectedRawScores(unittest.TestCase):
    def test_restricted_to_selected_and_present(self):
        train, _, sel = _make_samples()
        s = train[0]
        # MAESTRO picked only boltz2; MANDATORY_7 adds equipnas+deeppocket
        # (both present in step4). boltz2 present too.
        raw = fm.selected_raw_scores(s, sel)
        self.assertEqual(set(raw), {"boltz2", "equipnas", "deeppocket"})

    def test_unselected_present_tool_excluded(self):
        train, _, sel = _make_samples()
        s = train[0]
        # Force a selection of ONLY boltz2 with NO mandatory by monkeypatching
        # MANDATORY to empty → equipnas/deeppocket drop out.
        with mock.patch.object(fm, "MANDATORY_TOOLS", ()):
            raw = fm.selected_raw_scores(s, sel)
        self.assertEqual(set(raw), {"boltz2"})


class TestNoTrain(unittest.TestCase):
    def setUp(self):
        self.rid = [1, 2, 3]
        self.raw = {"a": {1: 0.2, 2: 0.8}, "b": {1: 0.4, 3: 0.6}}

    def test_mean(self):
        out = fm.predict_mean(self.raw, self.rid)
        # residue 1: (0.2+0.4)/2 = 0.3
        self.assertAlmostEqual(out[0], 0.3)

    def test_max(self):
        out = fm.predict_max(self.raw, self.rid)
        self.assertAlmostEqual(out[0], 0.4)
        self.assertAlmostEqual(out[1], 0.8)

    def test_noisy_or_range_and_formula(self):
        out = fm.predict_noisy_or(self.raw, self.rid)
        # c = 1/2; residue 1: 1-(1-.5*.2)(1-.5*.4) = 1-.9*.8 = 0.28
        self.assertAlmostEqual(out[0], 0.28)
        self.assertTrue(np.all((out >= 0) & (out <= 1)))

    def test_empty_is_zeros(self):
        self.assertTrue(np.all(fm.predict_mean({}, self.rid) == 0))


class TestImputeFlag(unittest.TestCase):
    def test_sklearn_methods_impute_tree_methods_not(self):
        flags = {k: imp for k, _l, _f, imp in fm.TRAINABLE}
        self.assertTrue(flags["logistic_regression"])
        self.assertTrue(flags["random_forest"])
        self.assertTrue(flags["mlp_3x256"])
        self.assertFalse(flags["xgboost"])
        self.assertFalse(flags["lightgbm"])


class TestMatrix(unittest.TestCase):
    def test_154_dim(self):
        train, _, sel = _make_samples()
        X = fm.build_matrix(train[0], {}, sel)
        self.assertEqual(X.shape, (6, 154))


class TestRunAblation(unittest.TestCase):
    def test_eight_rows_and_metrics(self):
        train, test, sel = _make_samples()

        def fake_factory(name):
            return lambda: fm.TrainablePredictor(name, _SumModel())

        # Patch each trainable factory to a deterministic sum model.
        patched = [(k, lbl, fake_factory(k), imp)
                   for k, lbl, _f, imp in fm.TRAINABLE]
        with mock.patch.object(fm, "TRAINABLE", patched):
            bucket = fm.run_ablation(
                train=train, test=test,
                profiles_train={}, profiles_test={},
                selections_train=sel, selections_test=sel)
        rows = fm.aggregate(bucket)
        self.assertEqual(len(rows), 8)
        self.assertEqual([r["method"] for r in rows], fm.METHOD_ORDER)
        self.assertEqual(rows[0]["label"], "Mean of raw scores")
        self.assertEqual(rows[-1]["label"], "LightGBM (ours)")
        # Every method produced usable per-sample rows.
        for r in rows:
            self.assertGreater(r["n_samples"], 0)
            self.assertIsNotNone(r["pearson_r"])


class TestIO(unittest.TestCase):
    def test_csv_round_trip(self):
        rows = [{"method": "lightgbm", "label": "LightGBM (ours)",
                 "n_samples": 107, "pearson_r": 0.593, "spearman_r": 0.47,
                 "r_squared": 0.40}]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "t7_v2.csv"
            fm.write_csv(out, rows)
            with out.open(encoding="utf-8") as f:
                back = list(csv.DictReader(f))
        self.assertEqual(back[0]["label"], "LightGBM (ours)")
        self.assertEqual(back[0]["pearson_r"], "0.593")


if __name__ == "__main__":
    unittest.main()
