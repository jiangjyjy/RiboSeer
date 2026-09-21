"""Mock tests for scripts/riboseer/search_best_tool_subset.py.

The greedy / exhaustive search logic is driven by an injectable
``score_fn``, so these tests use a synthetic scoring function (no
LightGBM, no data) to pin down the search behaviour deterministically.
"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.riboseer import search_best_tool_subset as sb  # noqa: E402

NOOP = lambda *a, **k: None  # noqa: E731

TOOLS = ("a", "b", "c", "d")


def _score(value):
    """A score dict with a given Pearson R (other metrics derived)."""
    return {"pearson_r": value, "spearman_r": round(value * 0.9, 4),
            "r_squared": round(value * value, 4), "n_samples": 10}


class TestBackward(unittest.TestCase):
    def test_removes_until_no_improvement(self):
        # 'd' is pure noise: any subset's score = 0.5 + 0.1*(#useful present),
        # useful = {a,b,c}; 'd' contributes nothing, so removing it never
        # hurts but never strictly improves either → backward keeps it.
        # Make 'd' actively harmful instead so removal improves.
        useful = {"a", "b", "c"}

        def score_fn(tools):
            s = 0.5 + 0.1 * len(useful & set(tools)) - 0.2 * ("d" in tools)
            return _score(round(s, 4))

        hist = sb.backward_elimination(TOOLS, score_fn, log=NOOP)
        chosen = hist[-1]["tools"]
        self.assertNotIn("d", chosen)           # harmful tool removed
        self.assertEqual(set(chosen), useful)   # useful tools kept
        self.assertEqual(hist[0]["action"], "baseline")
        self.assertEqual(hist[1]["action"], "remove")

    def test_stops_when_full_is_best(self):
        # Every tool helps → no removal improves → only the baseline step.
        def score_fn(tools):
            return _score(round(0.1 * len(tools), 4))

        hist = sb.backward_elimination(TOOLS, score_fn, log=NOOP)
        self.assertEqual(len(hist), 1)
        self.assertEqual(set(hist[-1]["tools"]), set(TOOLS))

    def test_respects_min_tools(self):
        # Fewer is always better → would shrink to 1, but min_tools=2 caps it.
        def score_fn(tools):
            return _score(round(1.0 - 0.1 * len(tools), 4))

        hist = sb.backward_elimination(TOOLS, score_fn, min_tools=2, log=NOOP)
        self.assertEqual(len(hist[-1]["tools"]), 2)


class TestForward(unittest.TestCase):
    def test_adds_best_then_stops(self):
        # score = 0.1*(#useful) ; useful={a,b}. 'c','d' add nothing.
        useful = {"a", "b"}

        def score_fn(tools):
            return _score(round(0.1 * len(useful & set(tools)), 4))

        hist = sb.forward_selection(TOOLS, score_fn, log=NOOP)
        chosen = hist[-1]["tools"]
        self.assertEqual(set(chosen), useful)
        self.assertEqual(hist[0]["action"], "baseline")
        self.assertTrue(all(h["action"] == "add" for h in hist[1:]))

    def test_first_pick_when_all_equal_single(self):
        # All single tools score the same; forward still takes one (first
        # improvement over the None baseline) then stops.
        def score_fn(tools):
            return _score(0.3) if tools else None

        hist = sb.forward_selection(TOOLS, score_fn, log=NOOP)
        self.assertEqual(len(hist[-1]["tools"]), 1)

    def test_respects_max_tools(self):
        def score_fn(tools):
            return _score(round(0.1 * len(tools), 4))  # more is better

        hist = sb.forward_selection(TOOLS, score_fn, max_tools=2, log=NOOP)
        self.assertEqual(len(hist[-1]["tools"]), 2)


class TestExhaustive(unittest.TestCase):
    def test_top_k_sorted_and_bounded(self):
        # score favours exactly {a,c}; sizes restricted to 2.
        def score_fn(tools):
            return _score(0.9 if set(tools) == {"a", "c"} else 0.4)

        top = sb.exhaustive_search(TOOLS, score_fn, min_tools=2, max_tools=2,
                                   top_k=3, log=NOOP)
        self.assertEqual(len(top), 3)
        self.assertEqual(set(top[0]["tools"]), {"a", "c"})   # best first
        # descending order
        prs = [sb._pr(h["score"]) for h in top]
        self.assertEqual(prs, sorted(prs, reverse=True))

    def test_skips_undefined_scores(self):
        def score_fn(tools):
            return None if "a" in tools else _score(0.4)

        top = sb.exhaustive_search(TOOLS, score_fn, min_tools=1, max_tools=1,
                                   top_k=10, log=NOOP)
        self.assertTrue(all("a" not in h["tools"] for h in top))


class TestCacheAndIO(unittest.TestCase):
    def test_cache_dedups_trainings(self):
        calls = {"n": 0}

        def fake_score_subset(train, test, tools):
            calls["n"] += 1
            return _score(0.5)

        # Patch the underlying trainer; the cache should collapse repeats.
        orig = sb.score_subset
        sb.score_subset = fake_score_subset
        try:
            scorer, cache = sb.make_cached_scorer([], [])
            scorer(("a", "b"))
            scorer(("b", "a"))      # same set, different order
            scorer(("a", "b"))
        finally:
            sb.score_subset = orig
        self.assertEqual(calls["n"], 1)         # trained once
        self.assertEqual(len(cache), 1)

    def test_build_rows_and_csv(self):
        backward = [
            {"action": "baseline", "tool": None, "tools": ["a", "b", "c"],
             "score": _score(0.58)},
            {"action": "remove", "tool": "c", "tools": ["a", "b"],
             "score": _score(0.60)}]
        forward = [
            {"action": "baseline", "tool": None, "tools": [], "score": None},
            {"action": "add", "tool": "a", "tools": ["a"], "score": _score(0.4)}]
        rows = sb.build_output_rows(backward, forward, None)
        self.assertEqual([r["method"] for r in rows],
                         ["backward", "backward", "forward", "forward"])
        self.assertEqual(rows[1]["tools"], "a+b")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "s.csv"
            sb.write_csv(out, rows)
            with out.open(encoding="utf-8") as f:
                back = list(csv.DictReader(f))
        self.assertEqual(back[1]["pearson_r"], "0.6")
        self.assertEqual(back[3]["action"], "add")


if __name__ == "__main__":
    unittest.main()
