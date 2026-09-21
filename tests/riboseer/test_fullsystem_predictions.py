"""Mock tests for the full-system (on/on/on) prediction export + the
``--predictions-dir`` read path on the downstream table scripts.

No LightGBM needed: ``table09_llm_modules._train_model`` is patched with a
deterministic fake (the same trick the ablation's own test uses), so we
exercise the prediction generation and the file round-trip without an ML
dependency.
"""
from __future__ import annotations

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

from scripts.tables import table09_llm_modules as alm  # noqa: E402
from step5_fusion import lightgbm_fusion as lgf  # noqa: E402
from step5_fusion import prediction_io as fio  # noqa: E402
import scripts.tables.table20_topk as t21  # noqa: E402
import scripts.tables.table24_distribution as t25  # noqa: E402
import scripts.tables.table04_naive_ensembles as cne  # noqa: E402


# ---- fixtures (mirror test_table09_llm_modules) ------------------------


def _pred(tool_id, *, success=True, pae=None, conf=None, binding=None):
    return {"tool_id": tool_id, "success": success,
            "per_residue_pae_score": pae or {},
            "per_residue_confidence": conf or {},
            "binding_protein_residues": binding or []}


def _varied_pred():
    pae = {i: round(0.95 - 0.13 * i, 3) for i in range(1, 7)}
    return [
        _pred("boltz2", pae=pae, conf={i: 80 - i for i in range(1, 7)},
              binding=[1, 2, 4]),
        _pred("equipnas", conf={i: 0.6 + 0.01 * i for i in range(1, 7)},
              binding=[1, 4]),
    ]


def _write_sample(processed_dir: Path, sid: str, length: int, binding):
    samples = processed_dir / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    doc = {"sample_id": sid,
           "protein": {"length": length, "sequence": "A" * length,
                       "resolved_residues": list(range(1, length + 1))},
           "rna": {"length": 40, "sequence": "G" * 40},
           "interaction": {"binding_protein_residues": list(binding)}}
    (samples / f"{sid}.json").write_text(json.dumps(doc), encoding="utf-8")


def _write_step4(step4_dir: Path, sid: str, predictions):
    step4_dir.mkdir(parents=True, exist_ok=True)
    rec = {"sample_id": sid, "predictions": predictions}
    (step4_dir / f"{sid}.jsonl").write_text(json.dumps(rec) + "\n",
                                            encoding="utf-8")


class _FakeModel:
    def predict(self, X):
        return np.nan_to_num(np.asarray(X, dtype=np.float64),
                             nan=0.0).sum(axis=1)


class TestFullsystemIO(unittest.TestCase):
    def test_round_trip_and_fill(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            fio.save_prob_dict(d, "s1", {1: 0.1, 3: 0.9}, residue_ids=[1, 2, 3])
            loaded = fio.load_predictions_dir(d)
            self.assertEqual(loaded["s1"], {1: 0.1, 2: 0.0, 3: 0.9})
            vec = fio.predictions_to_vector(loaded["s1"], [1, 2, 3, 4])
            self.assertEqual(list(vec), [0.1, 0.0, 0.9, 0.0])

    def test_missing_dir_and_sample(self):
        self.assertEqual(fio.load_predictions_dir(None), {})
        self.assertEqual(fio.load_predictions_dir(Path("/no/such")), {})
        self.assertEqual(list(fio.predictions_to_vector(None, [1, 2])),
                         [0.0, 0.0])


class TestGenerateAndSave(unittest.TestCase):
    def test_compute_and_save(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, s4 = td / "proc", td / "s4"
            for sid in ("tr1", "te1", "te2"):
                _write_sample(proc, sid, 6, binding=[1, 2, 4])
                _write_step4(s4, sid, _varied_pred())
            train = alm.collect_samples(s4, proc, ["tr1"])
            test = alm.collect_samples(s4, proc, ["te1", "te2"])
            with mock.patch.object(lgf, "_train_model",
                                   lambda X, y: _FakeModel()):
                preds = alm.compute_fullsystem_predictions(
                    train=train, test=test,
                    profiles_train={}, profiles_test={},
                    selections_train={}, selections_test={},
                    polish_actions={})
            self.assertEqual(set(preds), {"te1", "te2"})
            # every test residue id has a probability
            for s in test:
                self.assertEqual(set(preds[s.sid]), set(s.residue_ids))

            out = td / "fs"
            n = alm.save_fullsystem_predictions(out, test, preds)
            self.assertEqual(n, 2)
            reloaded = fio.load_predictions_dir(out)
            self.assertEqual(reloaded["te1"], preds["te1"])


class TestPredictionsDirReadPath(unittest.TestCase):
    """The downstream scripts read RiboSeer from --predictions-dir without
    loading any model bundle."""

    def _setup(self, td: Path):
        proc, s4, fs = td / "proc", td / "s4", td / "fs"
        for sid in ("te1", "te2"):
            _write_sample(proc, sid, 6, binding=[1, 2, 4])
            _write_step4(s4, sid, _varied_pred())
        # full-system predictions favouring the GT residues
        for sid in ("te1", "te2"):
            fio.save_prob_dict(
                fs, sid,
                {r: (0.9 if r in (1, 2, 4) else 0.1) for r in range(1, 7)},
                residue_ids=list(range(1, 7)))
        (td / "test.txt").write_text("te1\nte2\n", encoding="utf-8")
        return proc, s4, fs

    def test_table21_reads_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, s4, fs = self._setup(td)
            rc = t21.main([
                "--data-dir", str(proc), "--step4-dir", str(s4),
                "--predictions-dir", str(fs),
                "--split-file", str(td / "test.txt")])
            self.assertEqual(rc, 0)

    def test_table21_requires_a_source(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, s4, _ = self._setup(td)
            rc = t21.main([
                "--data-dir", str(proc), "--step4-dir", str(s4),
                "--split-file", str(td / "test.txt")])
            self.assertEqual(rc, 1)  # neither --predictions-dir nor --model-dir

    def test_table25_reads_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, s4, fs = self._setup(td)
            rc = t25.main([
                "--data-dir", str(proc), "--step4-dir", str(s4),
                "--predictions-dir", str(fs),
                "--split-file", str(td / "test.txt")])
            self.assertEqual(rc, 0)


class TestNaiveEnsemblesMetrics(unittest.TestCase):
    """Cat-E ensembles report Pearson + Spearman + R² (per-sample mean)."""

    def test_three_metrics_present(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, tr4, te4 = td / "proc", td / "tr4", td / "te4"
            for sid in ("tr1", "tr2"):
                _write_sample(proc, sid, 6, binding=[1, 2, 4])
                _write_step4(tr4, sid, _varied_pred())
            for sid in ("te1", "te2"):
                _write_sample(proc, sid, 6, binding=[1, 2, 4])
                _write_step4(te4, sid, _varied_pred())
            tools = ["boltz2", "equipnas"]
            train = cne.collect(tr4, proc, ["tr1", "tr2"], tools)
            test = cne.collect(te4, proc, ["te1", "te2"], tools)
            dist, weights = cne.evaluate(train, test, tools)
        self.assertEqual(set(dist), set(cne.METHOD_ORDER))
        for m in cne.METHOD_ORDER:
            self.assertTrue(dist[m], f"{m} produced no rows")
            row = dist[m][0]
            # every per-sample corr carries all three metric keys
            for key in ("pearson_r", "spearman_r", "r_squared"):
                self.assertIn(key, row)
            # aggregate helper returns a number for each metric
            for key in ("pearson_r", "spearman_r", "r_squared"):
                self.assertIsInstance(cne._mean_of(dist[m], key), float)


class TestModelSaveLoad(unittest.TestCase):
    """on/on/on LightGBM export → Table 27 read path (needs lightgbm)."""

    def _lgbm_or_skip(self):
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            self.skipTest("lightgbm not installed")

    def test_save_load_gain_round_trip(self):
        self._lgbm_or_skip()
        import lightgbm as lgb
        import scripts.tables.table26_feature_importance as t27
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 5))
        y = (X[:, 0] + 0.5 * X[:, 3] > 0).astype(float)
        reg = lgb.LGBMRegressor(n_estimators=20, verbose=-1).fit(X, y)
        names = [f"feat_{i}" for i in range(5)]
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            fio.save_lightgbm_model(d, reg, names)
            self.assertTrue(fio.has_lightgbm_model(d))
            got_names, gains = fio.load_lightgbm_gain(d)
            self.assertEqual(got_names, names)
            self.assertEqual(len(gains), 5)
            self.assertTrue(any(g > 0 for g in gains))
            # Table 27 main routes to the on/on/on loader on model.txt
            rc = t27.main(["--model-dir", str(d), "--top-k", "3"])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
