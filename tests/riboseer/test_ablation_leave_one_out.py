"""Mock tests for scripts/riboseer/ablation_leave_one_out.py.

LightGBM isn't required: ``_train_model`` is patched with a deterministic
fake (sum of NaN→0 features), so we exercise config building, the
leave-one-out orchestration, the ΔPearson column, the fixed best-single
reference row, and CSV I/O — without any ML dep.
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

from scripts.riboseer import ablation_leave_one_out as loo  # noqa: E402
from step5_fusion.features_15tool import ALL_KNOWN_TOOLS  # noqa: E402


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
    return [
        _pred("boltz2", pae={i: round(0.95 - 0.13 * i, 3) for i in range(1, 7)},
              conf={i: 80 - i for i in range(1, 7)}, binding=[1, 2, 4]),
        _pred("equipnas", conf={i: 0.6 + 0.03 * i for i in range(1, 7)},
              binding=[1, 4]),
    ]


def _make_samples():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        proc, s4 = td / "proc", td / "s4"
        for sid in ("tr1", "tr2", "te1", "te2"):
            _write_sample(proc, sid, 6, binding=[1, 2, 4])
            _write_step4(s4, sid, _varied_preds())
        return (loo.collect_samples(s4, proc, ["tr1", "tr2"]),
                loo.collect_samples(s4, proc, ["te1", "te2"]))


class _FakeModel:
    """prediction = row sum of NaN→0 features (responds to tool drops)."""

    def predict(self, X):
        return np.nan_to_num(np.asarray(X, dtype=np.float64),
                             nan=0.0).sum(axis=1)


# ---- tests --------------------------------------------------------------


class TestConfigs(unittest.TestCase):
    def test_eighteen_rows_in_order(self):
        cfgs = loo.build_configs()
        self.assertEqual(len(cfgs), 18)
        self.assertEqual(cfgs[0].label, "Full system (15 tools)")
        self.assertEqual(cfgs[0].tools, ALL_KNOWN_TOOLS)
        # rows 2-16 are the 15 single-tool drops, in library order.
        self.assertEqual(cfgs[1].label, "w/o Boltz-2")
        self.assertNotIn("boltz2", cfgs[1].tools)
        self.assertEqual(len(cfgs[1].tools), 14)
        self.assertEqual(cfgs[-2].label, "Cat. A only (5 tools)")
        self.assertEqual(cfgs[-2].tools, loo.CAT_A)
        self.assertEqual(cfgs[-1].label, "Best single (Boltz-2)")
        self.assertIsNone(cfgs[-1].tools)

    def test_every_tool_has_a_drop_row(self):
        labels = {c.label for c in loo.build_configs()}
        for t in ALL_KNOWN_TOOLS:
            self.assertIn(f"w/o {loo.DISPLAY[t]}", labels)


class TestMatrix(unittest.TestCase):
    def test_column_layout_constant_across_subsets(self):
        train, _ = _make_samples()
        s = train[0]
        full = loo.build_matrix(s, ALL_KNOWN_TOOLS)
        dropped = loo.build_matrix(s, tuple(t for t in ALL_KNOWN_TOOLS
                                            if t != "boltz2"))
        self.assertEqual(full.shape[0], s.protein_len)
        self.assertEqual(full.shape[1], dropped.shape[1])  # layout fixed

    def test_dropping_present_tool_changes_features(self):
        train, _ = _make_samples()
        s = train[0]
        full = loo.build_matrix(s, ALL_KNOWN_TOOLS)
        dropped = loo.build_matrix(s, tuple(t for t in ALL_KNOWN_TOOLS
                                            if t != "boltz2"))
        # boltz2 columns present in full become NaN when dropped.
        self.assertFalse(np.allclose(np.nan_to_num(full),
                                     np.nan_to_num(dropped)))


class TestAggregate(unittest.TestCase):
    def test_mean_and_n(self):
        corrs = [{"pearson_r": 0.4, "spearman_r": 0.3, "r_squared": 0.16},
                 {"pearson_r": 0.6, "spearman_r": 0.5, "r_squared": 0.36}]
        row = loo._aggregate("x", corrs)
        self.assertEqual(row["n_samples"], 2)
        self.assertAlmostEqual(row["pearson_r"], 0.5)
        self.assertAlmostEqual(row["spearman_r"], 0.4)

    def test_empty_is_none(self):
        row = loo._aggregate("x", [])
        self.assertEqual(row["n_samples"], 0)
        self.assertIsNone(row["pearson_r"])


class TestRunTable10(unittest.TestCase):
    def test_rows_delta_and_reference(self):
        train, test = _make_samples()
        with mock.patch.object(loo, "_train_model",
                               lambda X, y: _FakeModel()):
            rows = loo.run_table10(train, test)
        self.assertEqual(len(rows), 18)
        # Full system: no delta.
        self.assertEqual(rows[0]["configuration"], "Full system (15 tools)")
        self.assertIsNone(rows[0]["delta_pearson_r"])
        # Best single: fixed Table-4 values, delta filled vs full.
        best = rows[-1]
        self.assertEqual(best["configuration"], "Best single (Boltz-2)")
        self.assertEqual(best["pearson_r"], 0.439)
        self.assertEqual(best["n_samples"], 104)
        if rows[0]["pearson_r"] is not None:
            self.assertAlmostEqual(
                best["delta_pearson_r"], round(0.439 - rows[0]["pearson_r"], 4))

    def test_no_scope_block_used(self):
        # run_table10 must build features with use_scope=False.
        train, test = _make_samples()
        seen = {}
        real = loo.build_15tool_features

        def spy(*a, **k):
            seen["use_scope"] = k.get("use_scope")
            return real(*a, **k)

        with mock.patch.object(loo, "_train_model",
                               lambda X, y: _FakeModel()), \
             mock.patch.object(loo, "build_15tool_features", spy):
            loo.run_table10(train, test)
        self.assertFalse(seen["use_scope"])


class TestIO(unittest.TestCase):
    def test_csv_round_trip(self):
        rows = [{"configuration": "Full system (15 tools)", "n_samples": 107,
                 "pearson_r": 0.55, "spearman_r": 0.5, "r_squared": 0.30,
                 "delta_pearson_r": None},
                {"configuration": "w/o Boltz-2", "n_samples": 107,
                 "pearson_r": 0.50, "spearman_r": 0.45, "r_squared": 0.25,
                 "delta_pearson_r": -0.05}]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "t10.csv"
            loo.write_csv(out, rows)
            with out.open(encoding="utf-8") as f:
                back = list(csv.DictReader(f))
        self.assertEqual(back[0]["configuration"], "Full system (15 tools)")
        self.assertEqual(back[1]["delta_pearson_r"], "-0.05")


if __name__ == "__main__":
    unittest.main()
