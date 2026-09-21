"""Mock tests for scripts/riboseer/generate_maestro_selections.py."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step3_tool_selection.weight_tensor import WeightTensor  # noqa: E402
from scripts.riboseer import compute_weight_tensor as cwt  # noqa: E402
from scripts.riboseer import generate_maestro_selections as gm  # noqa: E402


def _pred(tool_id, success=True):
    return {"tool_id": tool_id, "category": "A", "sample_id": "x",
            "success": success}


def _write_sample(proc, sid, plen=80, rlen=40):
    d = proc / "samples"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sid}.json").write_text(json.dumps({
        "sample_id": sid,
        "protein": {"length": plen, "sequence": "A" * plen},
        "rna": {"length": rlen, "sequence": "G" * rlen}}), encoding="utf-8")


def _write_step4(step4, sid, tools):
    step4.mkdir(parents=True, exist_ok=True)
    rec = {"sample_id": sid, "predictions": [_pred(t) for t in tools]}
    (step4 / f"{sid}.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")


class TestHelpers(unittest.TestCase):
    def test_available_tools_order(self):
        s4 = {"predictions": [_pred("equipnas"), _pred("boltz2"),
                              _pred("fpocket", success=False)]}
        # library order: boltz2 before equipnas; failed fpocket excluded
        self.assertEqual(gm.available_tools(s4), ["boltz2", "equipnas"])

    def test_greedy_topk(self):
        avail = ["boltz2", "equipnas", "p2rank", "fpocket"]
        util = {"boltz2": 0.7, "equipnas": 0.6, "p2rank": 0.5, "fpocket": 0.4}
        self.assertEqual(gm.greedy_ucb_select(avail, util, 2),
                         ["boltz2", "equipnas"])

    def test_greedy_guarantees_cat_a(self):
        # Only p2rank/equipnas score high, but a Cat-A tool must appear.
        avail = ["p2rank", "equipnas", "boltz2"]
        util = {"p2rank": 0.9, "equipnas": 0.8, "boltz2": 0.1}
        out = gm.greedy_ucb_select(avail, util, 2)
        self.assertIn("boltz2", out)               # Cat A forced in
        self.assertEqual(len(out), 2)

    def test_category_from_profile(self):
        cat = gm.category_for("x", None, {"protein_family": "RRM",
                                          "rna_context": "stem-loop"})
        self.assertEqual(cat, "rrm_x_stem-loop")

    def test_category_default(self):
        self.assertEqual(gm.category_for("x", None, None),
                         gm.DEFAULT_CATEGORY)


class TestUtilityScores(unittest.TestCase):
    """Every tool of the library gets a real UCB, and it resolves for ANY
    pocket category via the global cell (the UCB=0.000 fix)."""

    def _wt(self):
        rows = cwt.assemble_tool_stats(
            {"nucleicnet": 0.45, "bindup": 0.44, "boltz2": 0.30},
            {"nucleicnet": 200, "bindup": 200, "boltz2": 220})
        return cwt.build_weight_tensor(rows)

    def test_every_library_tool_gets_nonzero_scores(self):
        wt = self._wt()
        avail = ["boltz2", "nucleicnet", "bindup"]
        # A pocket category that is NOT the storage key — used to be 0.000.
        util = gm.utility_scores(wt, avail, "novel_x_stem-loop")
        for t in avail:
            self.assertIn(t, util)
            self.assertGreater(util[t], 0.0)
        # higher μ̂ → higher UCB
        self.assertGreater(util["nucleicnet"], util["boltz2"])

    def test_registry_utility_covers_the_whole_library(self):
        # compute_utility_scores ranks every tool in the registry, so a
        # headline row can always score the tools the paper reports.
        wt = self._wt()
        filtered = wt.compute_utility_scores("novel_x_stem-loop")
        self.assertIn("nucleicnet", filtered)
        self.assertIn("bindup", filtered)


class TestPromptFairness(unittest.TestCase):
    """The LLM prompt must rank tools strictly by UCB and carry no
    category/mandatory bias toward the original structure tools."""

    def _prompt(self):
        # library order would put boltz2 first, but its UCB is lowest;
        # nucleicnet/bindup (high UCB) must lead.
        available = ["boltz2", "p2rank", "nucleicnet", "bindup",
                     "rnabindrplus"]
        utility = {"boltz2": 0.20, "p2rank": 0.24, "nucleicnet": 0.45,
                   "bindup": 0.44, "rnabindrplus": 0.42}
        return gm.build_maestro_prompt(
            {"protein_family": "RRM", "rna_context": "stem-loop",
             "difficulty": "medium"},
            available, utility, max_tools=6, protein_len=80, rna_len=40)

    def test_tools_ranked_by_ucb_descending(self):
        p = self._prompt()
        order = [p.index(t) for t in
                 ("nucleicnet", "bindup", "rnabindrplus", "p2rank", "boltz2")]
        self.assertEqual(order, sorted(order))      # high-UCB tools appear first

    def test_no_category_labels(self):
        p = self._prompt()
        self.assertNotIn("Category", p)
        self.assertNotIn("(Category A", p)

    def test_no_mandatory_or_avoid_language(self):
        p = self._prompt()
        for banned in ("at least 1", "mandatory", "Avoid tools",
                       "must include", "recommended"):
            self.assertNotIn(banned, p)

    def test_states_ucb_is_real_performance_and_strict(self):
        p = self._prompt()
        self.assertIn("strictly by UCB", p)
        self.assertIn("training data", p)
        self.assertIn("equally eligible", p)

    def test_uniform_per_tool_format(self):
        p = self._prompt()
        for t in ("boltz2", "nucleicnet", "bindup"):
            self.assertIn(f"- {t}: UCB ", p)        # identical shape per tool


class TestGenerate(unittest.TestCase):
    def _fixture(self, td):
        proc, step4 = td / "proc", td / "s4"
        _write_sample(proc, "1abc_A_E")
        _write_step4(step4, "1abc_A_E", ["boltz2", "equipnas", "p2rank"])
        return proc, step4

    def test_auto_mode(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, step4 = self._fixture(td)
            out = td / "sel"
            stats = gm.generate(
                processed_dir=proc, sample_ids=["1abc_A_E"], step4_dir=step4,
                out_dir=out, mode="auto", weight_tensor=WeightTensor(),
                scope_profiles={}, step2_dir=None, config_path=None,
                max_tools=6, top_k=2)
            self.assertEqual(stats["written"], 1)
            rec = json.loads((out / "1abc_A_E.json").read_text())
            self.assertEqual(rec["source"], "ucb")
            self.assertTrue(set(rec["selected_tools"]) <=
                            {"boltz2", "equipnas", "p2rank"})
            self.assertTrue(rec["selected_tools"])

    def test_llm_mode_uses_llm_select(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, step4 = self._fixture(td)
            out = td / "sel"
            with mock.patch.object(gm, "_build_llm_client",
                                   lambda c, b=None: (object(), {})), \
                 mock.patch.object(gm, "llm_select",
                                   lambda *a, **k: (["boltz2"], "llm picked")):
                stats = gm.generate(
                    processed_dir=proc, sample_ids=["1abc_A_E"],
                    step4_dir=step4, out_dir=out, mode="llm",
                    weight_tensor=WeightTensor(), scope_profiles={},
                    step2_dir=None, config_path=None, max_tools=6, top_k=2)
            rec = json.loads((out / "1abc_A_E.json").read_text())
            self.assertEqual(rec["source"], "llm")
            self.assertEqual(rec["selected_tools"], ["boltz2"])
            self.assertEqual(stats["llm"], 1)

    def test_llm_fallback_to_ucb(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, step4 = self._fixture(td)
            out = td / "sel"
            with mock.patch.object(gm, "_build_llm_client",
                                   lambda c, b=None: (object(), {})), \
                 mock.patch.object(gm, "llm_select", lambda *a, **k: None):
                gm.generate(
                    processed_dir=proc, sample_ids=["1abc_A_E"],
                    step4_dir=step4, out_dir=out, mode="llm",
                    weight_tensor=WeightTensor(), scope_profiles={},
                    step2_dir=None, config_path=None, max_tools=6, top_k=2)
            rec = json.loads((out / "1abc_A_E.json").read_text())
            self.assertEqual(rec["source"], "llm_fallback")
            self.assertTrue(rec["selected_tools"])   # greedy filled it


if __name__ == "__main__":
    unittest.main()
