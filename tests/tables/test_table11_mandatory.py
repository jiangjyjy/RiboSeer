"""Mock tests for scripts/tables/table11_mandatory.py.

LightGBM isn't required: ``_train_model`` is patched with a deterministic
fake (sum of NaN→0 features), so we exercise the three selection policies
(free / mandatory-optimal-7 / all-15), the headline on/on/on pipeline
(SCOPE + MAESTRO + POLISH), the #tools/sample + runtime columns, and CSV
I/O — without any ML dep.
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

from scripts.tables import table11_mandatory as man  # noqa: E402
from step5_fusion.features_15tool import ALL_KNOWN_TOOLS  # noqa: E402


# ---- fixtures -----------------------------------------------------------


def _pred(tool_id, *, pae=None, conf=None, binding=None, runtime=None):
    rec = {"tool_id": tool_id, "success": True,
           "per_residue_pae_score": pae or {},
           "per_residue_confidence": conf or {},
           "binding_protein_residues": binding or []}
    if runtime is not None:
        rec["runtime_seconds"] = runtime
    return rec


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
    # boltz2 carries a real runtime; equipnas/deeppocket do not (→ timeout).
    return [
        _pred("boltz2", pae={i: round(0.95 - 0.13 * i, 3) for i in range(1, 7)},
              conf={i: 80 - i for i in range(1, 7)}, binding=[1, 2, 4],
              runtime=120.0),
        _pred("chai1", pae={i: round(0.9 - 0.1 * i, 3) for i in range(1, 7)},
              conf={i: 75 - i for i in range(1, 7)}, binding=[1, 2]),
        _pred("equipnas", conf={i: 0.6 + 0.03 * i for i in range(1, 7)},
              binding=[1, 4]),
        _pred("deeppocket", conf={i: 0.4 + 0.05 * i for i in range(1, 7)},
              binding=[2, 3]),
    ]


def _make_samples():
    """Returns (train, test, selections) all in mem."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        proc, s4 = td / "proc", td / "s4"
        ids = ("tr1", "tr2", "te1", "te2")
        for sid in ids:
            _write_sample(proc, sid, 6, binding=[1, 2, 4])
            _write_step4(s4, sid, _varied_preds())
        train = man.collect_samples(s4, proc, ["tr1", "tr2"])
        test = man.collect_samples(s4, proc, ["te1", "te2"])
    # MAESTRO picked just boltz2 + chai1 for every sample.
    selections = {sid: {"selected_tools": ["boltz2", "chai1"]} for sid in ids}
    return train, test, selections


class _FakeModel:
    def predict(self, X):
        return np.nan_to_num(np.asarray(X, dtype=np.float64),
                             nan=0.0).sum(axis=1)


# ---- tests --------------------------------------------------------------


class TestPolicyResolution(unittest.TestCase):
    def test_free_uses_only_llm_picks(self):
        train, _, sel = _make_samples()
        kept = man.resolve_policy_tools(train[0], sel, "free")
        self.assertEqual(sorted(kept), ["boltz2", "chai1"])

    def test_mandatory7_is_union(self):
        train, _, sel = _make_samples()
        kept = man.resolve_policy_tools(train[0], sel, "mandatory7")
        # union of {boltz2, chai1} and the 7-core = the 7-core (chai1/boltz2
        # already in it).
        self.assertEqual(set(kept), set(man.MANDATORY_7))
        self.assertEqual(len(kept), 7)
        # library order preserved.
        self.assertEqual(kept, sorted(kept, key=man._LIB_INDEX.get))

    def test_mandatory7_adds_extra_llm_pick(self):
        train, _, sel = _make_samples()
        sid = train[0].sid
        sel2 = dict(sel)
        sel2[sid] = {"selected_tools": ["boltz2", "p2rank"]}  # p2rank not core
        kept = man.resolve_policy_tools(train[0], sel2, "mandatory7")
        self.assertEqual(set(kept), set(man.MANDATORY_7) | {"p2rank"})

    def test_all_is_fifteen(self):
        train, _, sel = _make_samples()
        kept = man.resolve_policy_tools(train[0], sel, "all")
        self.assertEqual(list(kept), list(ALL_KNOWN_TOOLS))


class TestMatrix(unittest.TestCase):
    def test_layout_constant_across_policies(self):
        train, _, sel = _make_samples()
        s = train[0]
        a = man.build_matrix(s, {}, sel, "free")
        b = man.build_matrix(s, {}, sel, "mandatory7")
        c = man.build_matrix(s, {}, sel, "all")
        self.assertEqual(a.shape[1], b.shape[1])
        self.assertEqual(b.shape[1], c.shape[1])
        self.assertEqual(a.shape[0], s.protein_len)

    def test_mandatory_activates_more_columns_than_free(self):
        train, _, sel = _make_samples()
        s = train[0]
        free = man.build_matrix(s, {}, sel, "free")
        mand = man.build_matrix(s, {}, sel, "mandatory7")
        # mandatory7 turns on deeppocket/equipnas (present in step4) that free
        # left NaN → fewer NaNs under mandatory7.
        self.assertLess(np.isnan(mand).sum(), np.isnan(free).sum())


class TestRuntime(unittest.TestCase):
    def test_real_prefers_step4_else_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            s4 = Path(td) / "s4"
            _write_step4(s4, "x", _varied_preds())
            rt, src = man.resolve_runtimes(s4, "real")
        self.assertEqual(src["boltz2"], "real")     # had runtime_seconds
        self.assertEqual(rt["boltz2"], 120.0)
        self.assertEqual(src["equipnas"], "timeout")  # none logged
        self.assertEqual(rt["equipnas"], man.TOOL_TIMEOUT["equipnas"])

    def test_timeout_mode_ignores_real(self):
        with tempfile.TemporaryDirectory() as td:
            s4 = Path(td) / "s4"
            _write_step4(s4, "x", _varied_preds())
            rt, src = man.resolve_runtimes(s4, "timeout")
        self.assertEqual(src["boltz2"], "timeout")
        self.assertEqual(rt["boltz2"], man.TOOL_TIMEOUT["boltz2"])

    def test_all_policy_runtime_is_sum_of_all_timeouts(self):
        train, test, sel = _make_samples()
        rt, _ = man.resolve_runtimes(None, "timeout")
        mins = man.policy_runtime_minutes(test, sel, "all", rt)
        self.assertAlmostEqual(mins, sum(man.TOOL_TIMEOUT.values()) / 60.0, 2)


class TestRunTable11V2(unittest.TestCase):
    def test_three_rows_with_metrics(self):
        train, test, sel = _make_samples()
        rt, _ = man.resolve_runtimes(None, "timeout")
        with mock.patch.object(man, "_train_model",
                               lambda X, y: _FakeModel()):
            rows = man.run_table11_v2(train, test, {}, {}, sel, sel, {}, rt)
        self.assertEqual([r.policy for r in rows],
                         ["free", "mandatory7", "all"])
        self.assertEqual(rows[0].label, "Free LLM selection (3-15 tools)")
        self.assertEqual(rows[2].avg_tools, 15.0)
        self.assertEqual(rows[0].avg_tools, 2.0)          # only boltz2+chai1
        self.assertEqual(rows[1].avg_tools, 7.0)          # the 7-core
        for r in rows:
            self.assertIsNotNone(r.runtime_min)

    def test_uses_scope_and_polish(self):
        train, test, sel = _make_samples()
        rt, _ = man.resolve_runtimes(None, "timeout")
        seen = {}
        real_feat = man.build_15tool_features
        real_polish = man.apply_polish

        def feat_spy(*a, **k):
            seen["use_scope"] = k.get("use_scope")
            return real_feat(*a, **k)

        def polish_spy(prob, sample, polish_on, actions):
            seen["polish_on"] = polish_on
            return real_polish(prob, sample, polish_on, actions)

        with mock.patch.object(man, "_train_model",
                               lambda X, y: _FakeModel()), \
             mock.patch.object(man, "build_15tool_features", feat_spy), \
             mock.patch.object(man, "apply_polish", polish_spy):
            man.run_table11_v2(train, test, {}, {}, sel, sel, {}, rt)
        self.assertTrue(seen["use_scope"])
        self.assertTrue(seen["polish_on"])

    def test_polish_off_propagates(self):
        # polish_on=False must reach apply_polish (POLISH removed).
        train, test, sel = _make_samples()
        rt, _ = man.resolve_runtimes(None, "timeout")
        seen = {}
        real_polish = man.apply_polish

        def polish_spy(prob, sample, polish_on, actions):
            seen["polish_on"] = polish_on
            return real_polish(prob, sample, polish_on, actions)

        with mock.patch.object(man, "_train_model",
                               lambda X, y: _FakeModel()), \
             mock.patch.object(man, "apply_polish", polish_spy):
            man.run_table11_v2(train, test, {}, {}, sel, sel, {}, rt,
                               polish_on=False)
        self.assertFalse(seen["polish_on"])


class TestIO(unittest.TestCase):
    def test_csv_round_trip(self):
        rows = [man.PolicyResult(
            policy="free", label="Free LLM selection (3-15 tools)",
            n_samples=107, pearson_r=0.58, spearman_r=0.46, r_squared=0.40,
            avg_tools=5.2, runtime_min=33.1)]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "t11_v2.csv"
            man.write_csv(out, rows)
            with out.open(encoding="utf-8") as f:
                back = list(csv.DictReader(f))
        self.assertEqual(back[0]["policy"], "free")
        self.assertEqual(back[0]["avg_tools"], "5.2")


if __name__ == "__main__":
    unittest.main()
