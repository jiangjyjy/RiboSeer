"""Tests for the shared MANDATORY guardrail (scripts/riboseer/config.py) and
its wiring into the headline tool resolution.

Covers the single source of truth (``config.MANDATORY_TOOLS`` = optimal-7),
the union helper, and that ``table09_llm_modules.resolve_selected_tools`` /
``build_sample_matrix`` inject it only when asked (headline) — never for the
free / UCB arms — and never for the MAESTRO-off baseline.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step5_fusion import lightgbm_fusion as alm  # noqa: E402
from scripts.riboseer import config as cfg  # noqa: E402
from step5_fusion.features_15tool import ALL_KNOWN_TOOLS  # noqa: E402


OPTIMAL_7 = {"boltz2", "chai1", "equipnas", "nucleicnet", "rnabindrplus",
             "hdock", "deeppocket"}


def _pred(tool_id, *, conf=None, binding=None):
    return {"tool_id": tool_id, "success": True,
            "per_residue_pae_score": {},
            "per_residue_confidence": conf or {},
            "binding_protein_residues": binding or []}


def _sample():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        proc, s4 = td / "proc", td / "s4"
        (proc / "samples").mkdir(parents=True)
        (proc / "samples" / "x.json").write_text(json.dumps({
            "sample_id": "x",
            "protein": {"length": 6, "sequence": "A" * 6,
                        "resolved_residues": list(range(1, 7))},
            "rna": {"length": 40, "sequence": "G" * 40},
            "interaction": {"binding_protein_residues": [1, 2, 4]}}),
            encoding="utf-8")
        s4.mkdir()
        # step4 carries boltz2 + chai1 + deeppocket + equipnas.
        preds = [_pred("boltz2", conf={i: 80 - i for i in range(1, 7)},
                       binding=[1, 2]),
                 _pred("chai1", conf={i: 70 - i for i in range(1, 7)},
                       binding=[1]),
                 _pred("deeppocket", conf={i: 0.5 for i in range(1, 7)},
                       binding=[2]),
                 _pred("equipnas", conf={i: 0.6 for i in range(1, 7)},
                       binding=[4])]
        (s4 / "x.jsonl").write_text(
            json.dumps({"sample_id": "x", "predictions": preds}) + "\n",
            encoding="utf-8")
        return alm.collect_samples(s4, proc, ["x"])[0]


class TestConfig(unittest.TestCase):
    def test_mandatory_is_optimal_7(self):
        self.assertEqual(set(cfg.MANDATORY_TOOLS), OPTIMAL_7)
        self.assertEqual(len(cfg.MANDATORY_TOOLS), 7)

    def test_inject_unions_in_library_order(self):
        got = cfg.inject_mandatory(["p2rank", "boltz2"])
        self.assertEqual(set(got), OPTIMAL_7 | {"p2rank"})
        order = [ALL_KNOWN_TOOLS.index(t) for t in got]
        self.assertEqual(order, sorted(order))

    def test_inject_drops_unknown_and_dedupes(self):
        got = cfg.inject_mandatory(["boltz2", "boltz2", "bogus"])
        self.assertEqual(set(got), OPTIMAL_7)


class TestResolveWiring(unittest.TestCase):
    def setUp(self):
        self.s = _sample()
        self.sel = {"x": {"selected_tools": ["boltz2", "chai1"]}}

    def test_no_mandatory_is_raw_selection(self):
        self.assertEqual(
            alm.resolve_selected_tools(self.s, True, self.sel),
            ["boltz2", "chai1"])

    def test_mandatory_unions_optimal_7(self):
        got = alm.resolve_selected_tools(self.s, True, self.sel,
                                         mandatory=cfg.MANDATORY_TOOLS)
        self.assertEqual(set(got), OPTIMAL_7)

    def test_maestro_off_ignores_mandatory(self):
        got = alm.resolve_selected_tools(self.s, False, self.sel,
                                         mandatory=cfg.MANDATORY_TOOLS)
        self.assertEqual(got, list(alm.FIXED5))

    def test_mandatory_activates_more_step4_columns(self):
        # deeppocket/equipnas are in step4 but not in the LLM pick; injecting
        # the guardrail turns them on → fewer NaNs than the raw selection.
        raw = alm.build_sample_matrix(self.s, True, True, {}, self.sel)
        head = alm.build_sample_matrix(self.s, True, True, {}, self.sel,
                                       mandatory=cfg.MANDATORY_TOOLS)
        self.assertEqual(raw.shape, head.shape)
        self.assertLess(np.isnan(head).sum(), np.isnan(raw).sum())


if __name__ == "__main__":
    unittest.main()
