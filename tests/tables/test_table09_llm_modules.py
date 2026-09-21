"""Mock tests for scripts/tables/table09_llm_modules.py.

LightGBM isn't required: ``_train_model`` is patched with a deterministic
fake (sum of non-NaN features), so we exercise the orchestration —
8-combo ordering, (scope, maestro) training cache, POLISH post-processing,
collection skip rules, resolvers, and CSV I/O — without any ML dep.
"""
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
from step5_fusion.features_15tool import encode_scope_profile  # noqa: E402


# ---- fixtures -----------------------------------------------------------


def _pred(tool_id, *, success=True, pae=None, conf=None, binding=None):
    return {
        "tool_id": tool_id, "success": success,
        "per_residue_pae_score": pae or {},
        "per_residue_confidence": conf or {},
        "binding_protein_residues": binding or [],
    }


def _write_sample(processed_dir: Path, sid: str, length: int,
                  binding, resolved=None, rna_len=40):
    samples = processed_dir / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    doc = {
        "sample_id": sid,
        "protein": {"length": length, "sequence": "A" * length,
                    "resolved_residues": resolved or list(range(1, length + 1))},
        "rna": {"length": rna_len, "sequence": "G" * rna_len},
        "interaction": {"binding_protein_residues": list(binding)},
    }
    (samples / f"{sid}.json").write_text(json.dumps(doc), encoding="utf-8")


def _write_step4(step4_dir: Path, sid: str, predictions):
    step4_dir.mkdir(parents=True, exist_ok=True)
    rec = {"sample_id": sid, "predictions": predictions}
    (step4_dir / f"{sid}.jsonl").write_text(
        json.dumps(rec) + "\n", encoding="utf-8")


def _varied_pred(seed_scale=1.0):
    pae = {i: round((0.95 - 0.13 * i) * seed_scale % 1.0, 3)
           for i in range(1, 7)}
    return [
        _pred("boltz2", pae=pae, conf={i: 80 - i for i in range(1, 7)},
              binding=[1, 2, 4]),
        _pred("equipnas", conf={i: 0.6 + 0.01 * i for i in range(1, 7)},
              binding=[1, 4]),
    ]


class _FakeModel:
    """Deterministic stand-in for LGBMRegressor: prediction = row sum of
    NaN→0 features. Responds to MAESTRO (NaN→0 toggles tool columns) and
    is fed by per-residue tool scores so corr is well defined."""

    def predict(self, X):
        return np.nan_to_num(np.asarray(X, dtype=np.float64),
                             nan=0.0).sum(axis=1)


def _make_samples():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        proc = td / "proc"
        s4 = td / "s4"
        for sid in ("tr1", "tr2", "te1", "te2"):
            _write_sample(proc, sid, 6, binding=[1, 2, 4])
            _write_step4(s4, sid, _varied_pred())
        train = alm.collect_samples(s4, proc, ["tr1", "tr2"])
        test = alm.collect_samples(s4, proc, ["te1", "te2"])
        return train, test


class TestCollect(unittest.TestCase):
    def test_collect_and_skips(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, s4 = td / "proc", td / "s4"
            _write_sample(proc, "good", 6, binding=[1, 2])
            _write_step4(s4, "good", _varied_pred())
            # zero-GT sample → skipped
            _write_sample(proc, "nogt", 6, binding=[])
            _write_step4(s4, "nogt", _varied_pred())
            # no step4 → skipped
            _write_sample(proc, "no4", 6, binding=[1])
            # no successful tool → skipped
            _write_sample(proc, "fail", 6, binding=[1])
            _write_step4(s4, "fail", [_pred("boltz2", success=False)])

            got = alm.collect_samples(
                s4, proc, ["good", "nogt", "no4", "fail"])
            self.assertEqual([s.sid for s in got], ["good"])
            s = got[0]
            self.assertEqual(s.protein_len, 6)
            self.assertEqual(s.rna_len, 40)
            self.assertIn("boltz2", s.available_tools)
            self.assertIn("equipnas", s.available_tools)


class TestResolvers(unittest.TestCase):
    def setUp(self):
        train, _ = _make_samples()
        self.s = train[0]

    def test_maestro_off_is_fixed5(self):
        self.assertEqual(alm.resolve_selected_tools(self.s, False, {}),
                         list(alm.FIXED5))

    def test_maestro_on_no_json_is_all_available(self):
        got = alm.resolve_selected_tools(self.s, True, {})
        self.assertEqual(set(got), set(self.s.available_tools))

    def test_maestro_on_uses_llm_selection(self):
        sel = {self.s.sid: {"selected_tools": ["boltz2", "chai1"]}}
        got = alm.resolve_selected_tools(self.s, True, sel)
        self.assertEqual(got, ["boltz2", "chai1"])

    def test_scope_off_is_zero_block(self):
        v = alm.resolve_scope_vector(self.s, False, {})
        self.assertTrue(np.all(v == 0.0))

    def test_scope_on_no_json_is_cauto(self):
        v = alm.resolve_scope_vector(self.s, True, {})
        self.assertFalse(np.all(v == 0.0))  # cauto fills len bins etc.

    def test_scope_on_uses_llm_profile(self):
        prof = {self.s.sid: {"protein_family": "RRM", "confidence": 0.95}}
        v = alm.resolve_scope_vector(self.s, True, prof)
        expect = encode_scope_profile(prof[self.s.sid], self.s.protein_len,
                                      self.s.rna_len)
        self.assertTrue(np.allclose(v, expect))


class TestPolishApply(unittest.TestCase):
    def setUp(self):
        train, _ = _make_samples()
        self.s = train[0]

    def test_polish_off_unchanged(self):
        prob = {1: 0.9, 2: 0.8, 3: 0.1}
        self.assertEqual(alm.apply_polish(prob, self.s, False, {}), prob)

    def test_polish_on_auto_changes(self):
        prob = {i: 0.9 for i in range(1, 9)}
        prob[8] = 0.51
        out = alm.apply_polish(prob, self.s, True, {})
        # weakest binder soft-masked (default 0.2×, no confidence on auto)
        self.assertAlmostEqual(out[8], 0.51 * 0.2)

    def test_polish_on_uses_saved_actions(self):
        prob = {1: 0.9, 2: 0.8, 3: 0.7}
        saved = {self.s.sid: {"actions": [{"action": "mask",
                                           "residues": [1]}]}}
        out = alm.apply_polish(prob, self.s, True, saved)
        self.assertAlmostEqual(out[1], 0.9 * 0.2)   # soft-masked, default 0.2×

    def test_polish_on_uses_iterative_rounds(self):
        # Table 12/14 iterative format: {"rounds": [...]} applied in order.
        prob = {1: 0.9, 2: 0.8, 3: 0.7}
        saved = {self.s.sid: {"rounds": [
            {"round": 1, "action": "mask", "residues": [1]},
            {"round": 2, "action": "mask", "residues": [2]},
            {"round": 3, "action": "accept"}]}}
        out = alm.apply_polish(prob, self.s, True, saved)
        self.assertAlmostEqual(out[1], 0.9 * 0.2)   # both masks applied
        self.assertAlmostEqual(out[2], 0.8 * 0.2)
        self.assertEqual(out[3], 0.7)               # accept = no-op


class TestRunAblation(unittest.TestCase):
    def test_eight_rows_order_and_cache(self):
        train, test = _make_samples()
        calls = {"n": 0}

        def fake_train(X, y):
            calls["n"] += 1
            return _FakeModel()

        with mock.patch.object(lgf, "_train_model", fake_train):
            rows, per_sample = alm.run_ablation(
                train=train, test=test,
                profiles_train={}, profiles_test={},
                selections_train={}, selections_test={},
                polish_actions={})

        self.assertEqual(len(rows), 8)
        # 4 distinct (scope, maestro) settings → 4 trainings (cache).
        self.assertEqual(calls["n"], 4)
        # Order matches COMBOS.
        labels = [(r["scope"], r["maestro"], r["polish"]) for r in rows]
        expect = [("off", "off", "off"), ("on", "off", "off"),
                  ("off", "on", "off"), ("off", "off", "on"),
                  ("on", "on", "off"), ("on", "off", "on"),
                  ("off", "on", "on"), ("on", "on", "on")]
        self.assertEqual(labels, expect)
        # Per-sample rows carry the toggles.
        self.assertTrue(per_sample)
        self.assertEqual(set(per_sample[0]),
                         {"scope", "maestro", "polish", "sample_id",
                          "pearson_r", "spearman_r", "r_squared"})

    def test_maestro_or_polish_changes_result(self):
        train, test = _make_samples()
        with mock.patch.object(lgf, "_train_model",
                               lambda X, y: _FakeModel()):
            rows, _ = alm.run_ablation(
                train=train, test=test,
                profiles_train={}, profiles_test={},
                selections_train={}, selections_test={},
                polish_actions={})
        means = [r["pearson_r_mean"] for r in rows]
        # At least one combo differs (MAESTRO toggles NaN→0; POLISH masks).
        self.assertGreater(len(set(means)), 1)


class TestIO(unittest.TestCase):
    def test_csv_round_trip(self):
        rows = [{"scope": "on", "maestro": "off", "polish": "on",
                 "n_samples": 5, "pearson_r_mean": 0.5, "pearson_r_std": 0.1,
                 "pearson_r_median": 0.5, "spearman_r_mean": 0.4,
                 "r2_mean": 0.25}]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "t9.csv"
            alm.write_csv(out, rows)
            text = out.read_text(encoding="utf-8")
            self.assertIn("scope,maestro,polish", text)
            self.assertIn("0.5", text)

    def test_load_json_dir_missing(self):
        self.assertEqual(alm._load_json_dir(None), {})
        self.assertEqual(alm._load_json_dir(Path("/no/such/dir")), {})

    def test_load_json_dir(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "a.json").write_text('{"x": 1}', encoding="utf-8")
            (d / "bad.json").write_text("{not json", encoding="utf-8")
            got = alm._load_json_dir(d)
            self.assertEqual(got["a"], {"x": 1})
            self.assertNotIn("bad", got)


class TestMainSmoke(unittest.TestCase):
    def test_main_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, s4 = td / "proc", td / "s4"
            for sid in ("tr1", "te1", "te2"):
                _write_sample(proc, sid, 6, binding=[1, 2, 4])
                _write_step4(s4, sid, _varied_pred())
            (td / "train.txt").write_text("tr1\n", encoding="utf-8")
            (td / "test.txt").write_text("te1\nte2\n", encoding="utf-8")
            out = td / "out.csv"
            argv = [
                "--train-step4-dir", str(s4), "--test-step4-dir", str(s4),
                "--processed-dir", str(proc),
                "--train-list", str(td / "train.txt"),
                "--test-list", str(td / "test.txt"),
                "--output", str(out)]
            with mock.patch.object(lgf, "LGBM_OK", True), \
                    mock.patch.object(lgf, "_train_model",
                                      lambda X, y: _FakeModel()):
                rc = alm.main(argv)
            self.assertEqual(rc, 0)
            self.assertTrue(out.is_file())
            self.assertTrue(
                out.with_name("out_per_sample.csv").is_file())
            lines = out.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1 + 8)  # header + 8 combos


if __name__ == "__main__":
    unittest.main()
