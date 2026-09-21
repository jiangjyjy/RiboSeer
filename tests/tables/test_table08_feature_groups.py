"""Mock tests for scripts/tables/table08_feature_groups.py.

LightGBM isn't required: ``_train_model`` is patched with a deterministic
fake. Covers the G1/G2/G3/G4 column routing on the real 154-col contract,
the per-config column slicing, the 6-row ablation, and CSV I/O.
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

from scripts.tables import table08_feature_groups as fg  # noqa: E402
from step5_fusion.features_15tool import table9_feature_names  # noqa: E402


# ---- column routing -----------------------------------------------------


class TestGrouping(unittest.TestCase):
    def setUp(self):
        self.names = table9_feature_names(use_context=True, use_scope=True)
        self.idx = fg.feature_group_indices(self.names)

    def test_total_is_154(self):
        self.assertEqual(len(self.names), 154)

    def test_group_sizes(self):
        sizes = {g: len(v) for g, v in self.idx.items()}
        self.assertEqual(sizes["G1"], 90)   # 15 tools × 6 per-tool cols
        self.assertEqual(sizes["G2"], 33)   # 15 cross + 15 gate + 3 summary
        self.assertEqual(sizes["G3"], 15)   # 6 win5 + 6 density5 + 3
        self.assertEqual(sizes["G4"], 16)   # scope block

    def test_partition_is_disjoint_and_complete(self):
        allidx = sorted(i for v in self.idx.values() for i in v)
        self.assertEqual(allidx, list(range(154)))

    def test_summary_counters_in_xt(self):
        g2_names = {self.names[i] for i in self.idx["G2"]}
        for c in ("vote_count", "catA_agree", "n_tools_active"):
            self.assertIn(c, g2_names)

    def test_per_tool_gate_in_pr_not_xt(self):
        g1_names = {self.names[i] for i in self.idx["G1"]}
        g2_names = {self.names[i] for i in self.idx["G2"]}
        self.assertIn("boltz2_gate", g1_names)        # per-tool gate → PR
        self.assertNotIn("boltz2_gate", g2_names)

    def test_win5_summary_in_nb(self):
        g3_names = {self.names[i] for i in self.idx["G3"]}
        self.assertIn("vote_count_win5", g3_names)    # not in XT
        self.assertIn("binding_streak", g3_names)

    def test_scope_in_g4(self):
        g4_names = {self.names[i] for i in self.idx["G4"]}
        self.assertTrue(all(n.startswith("scope_") for n in g4_names))


class TestConfigColumns(unittest.TestCase):
    def test_union_sorted(self):
        idx = {"G1": [0, 2], "G2": [1, 5], "G3": [3], "G4": [4]}
        self.assertEqual(fg.config_columns(("G1", "G2"), idx), [0, 1, 2, 5])

    def test_six_configs_full_last(self):
        self.assertEqual(len(fg.CONFIGS), 6)
        self.assertEqual(fg.CONFIGS[0], ("G1",))
        self.assertEqual(fg.CONFIGS[-1], ("G1", "G2", "G3", "G4"))


# ---- ablation -----------------------------------------------------------


def _pred(tool_id, *, conf=None, binding=None):
    return {"tool_id": tool_id, "success": True,
            "per_residue_pae_score": {},
            "per_residue_confidence": conf or {},
            "binding_protein_residues": binding or []}


def _write_sample(proc, sid, length, binding):
    (proc / "samples").mkdir(parents=True, exist_ok=True)
    doc = {"sample_id": sid,
           "protein": {"length": length, "sequence": "A" * length,
                       "resolved_residues": list(range(1, length + 1))},
           "rna": {"length": 40, "sequence": "G" * 40},
           "interaction": {"binding_protein_residues": list(binding)}}
    (proc / "samples" / f"{sid}.json").write_text(json.dumps(doc),
                                                  encoding="utf-8")


def _write_step4(s4, sid, preds):
    s4.mkdir(parents=True, exist_ok=True)
    (s4 / f"{sid}.jsonl").write_text(
        json.dumps({"sample_id": sid, "predictions": preds}) + "\n",
        encoding="utf-8")


def _make_samples():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        proc, s4 = td / "proc", td / "s4"
        preds = [_pred("boltz2", conf={i: 0.9 - 0.1 * i for i in range(1, 7)},
                       binding=[1, 2, 4]),
                 _pred("equipnas", conf={i: 0.5 for i in range(1, 7)},
                       binding=[1, 4])]
        for sid in ("tr1", "tr2", "te1", "te2"):
            _write_sample(proc, sid, 6, binding=[1, 2, 4])
            _write_step4(s4, sid, preds)
        return (fg.collect_samples(s4, proc, ["tr1", "tr2"]),
                fg.collect_samples(s4, proc, ["te1", "te2"]))


class _FakeModel:
    def __init__(self, ncol):
        self.ncol = ncol

    def predict(self, X):
        assert X.shape[1] == self.ncol
        return np.nan_to_num(np.asarray(X, dtype=np.float64),
                             nan=0.0).sum(axis=1)


class TestAblation(unittest.TestCase):
    def test_six_rows_with_dims(self):
        train, test = _make_samples()

        def fake_train(X, y):
            return _FakeModel(X.shape[1])

        with mock.patch.object(fg, "_train_model", fake_train):
            rows = fg.run_ablation(train, test, {}, {}, {}, {})
        self.assertEqual(len(rows), 6)
        # G1 row → 90 dims; Full → 154 dims.
        self.assertEqual(rows[0]["n_dims"], 90)
        self.assertEqual(rows[-1]["n_dims"], 154)
        self.assertTrue(rows[-1]["G4"])           # Full has SCOPE
        self.assertFalse(rows[0]["G2"])
        # XT+NB row dims = 33 + 15 = 48.
        xtnb = next(r for r in rows if r["G2"] and r["G3"] and not r["G1"])
        self.assertEqual(xtnb["n_dims"], 48)


class TestIO(unittest.TestCase):
    def test_csv_checkmarks_as_ints(self):
        rows = [{"G1": True, "G2": False, "G3": False, "G4": False,
                 "n_dims": 90, "n_samples": 107, "pearson_r": 0.55,
                 "spearman_r": 0.44, "r_squared": 0.30},
                {"G1": True, "G2": True, "G3": True, "G4": True,
                 "n_dims": 154, "n_samples": 107, "pearson_r": 0.593,
                 "spearman_r": 0.454, "r_squared": 0.436}]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "t8.csv"
            fg.write_csv(out, rows)
            with out.open(encoding="utf-8") as f:
                back = list(csv.DictReader(f))
        self.assertEqual(back[0]["pr"], "1")
        self.assertEqual(back[0]["sc"], "0")
        self.assertEqual(back[1]["sc"], "1")
        self.assertEqual(back[1]["pearson_r"], "0.593")


if __name__ == "__main__":
    unittest.main()
