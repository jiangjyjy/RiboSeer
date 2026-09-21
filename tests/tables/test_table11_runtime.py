"""Mock tests for scripts/tables/table11_runtime.py."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.tables import table11_runtime as rt  # noqa: E402
from step5_fusion.features_15tool import ALL_KNOWN_TOOLS  # noqa: E402


def _write_sel(directory, sid, tools):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{sid}.json").write_text(
        json.dumps({"sample_id": sid, "selected_tools": tools}),
        encoding="utf-8")


def _write_step4(directory, sid, preds):
    """preds: list of (tool_id, success, runtime_seconds)."""
    directory.mkdir(parents=True, exist_ok=True)
    rec = {"sample_id": sid, "predictions": [
        {"tool_id": t, "success": s, "runtime_seconds": r}
        for t, s, r in preds]}
    (directory / f"{sid}.jsonl").write_text(json.dumps(rec) + "\n",
                                            encoding="utf-8")


class TestDefaults(unittest.TestCase):
    def test_every_library_tool_has_an_estimate(self):
        for t in ALL_KNOWN_TOOLS:
            self.assertIn(t, rt.DEFAULT_RUNTIME_SECONDS)


class TestExtractRealTimings(unittest.TestCase):
    def test_means_over_successful_numeric(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "s4"
            _write_step4(d, "a", [("boltz2", True, 280),
                                  ("chai1", False, 999)])      # failed → skip
            _write_step4(d, "b", [("boltz2", True, 320),
                                  ("boltz2", True, None)])      # None → skip
            real = rt.extract_tool_runtimes(d)
            self.assertEqual(real["boltz2"], 300.0)             # (280+320)/2
            self.assertNotIn("chai1", real)                     # only failed

    def test_alias_canonicalised(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "s4"
            _write_step4(d, "a", [("rf2na", True, 600)])
            real = rt.extract_tool_runtimes(d)
            self.assertEqual(real.get("rosettafold2na"), 600.0)

    def test_missing_dir_returns_empty(self):
        self.assertEqual(rt.extract_tool_runtimes(None), {})
        self.assertEqual(rt.extract_tool_runtimes(Path("/no/such")), {})


class TestResolveRuntimes(unittest.TestCase):
    def test_real_overrides_estimate(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "s4"
            _write_step4(d, "a", [("boltz2", True, 250)])
            runtimes, source = rt.resolve_runtimes(d)
            self.assertEqual(runtimes["boltz2"], 250.0)
            self.assertEqual(source["boltz2"], "real")
            # fpocket had no step4 timing → estimate
            self.assertEqual(runtimes["fpocket"],
                             float(rt.DEFAULT_RUNTIME_SECONDS["fpocket"]))
            self.assertEqual(source["fpocket"], "estimate")

    def test_all_estimate_when_no_step4(self):
        runtimes, source = rt.resolve_runtimes(None)
        self.assertTrue(all(v == "estimate" for v in source.values()))
        self.assertEqual(len(runtimes), len(ALL_KNOWN_TOOLS))


class TestSampleRuntime(unittest.TestCase):
    def _rt(self):
        return {t: float(rt.DEFAULT_RUNTIME_SECONDS[t]) for t in ALL_KNOWN_TOOLS}

    def test_sequential_sums(self):
        # boltz2(300) + p2rank(10) + fpocket(5) = 315
        self.assertEqual(
            rt.sample_runtime(["boltz2", "p2rank", "fpocket"],
                              self._rt(), "sequential"), 315.0)

    def test_parallel_takes_max(self):
        self.assertEqual(
            rt.sample_runtime(["boltz2", "p2rank", "fpocket"],
                              self._rt(), "parallel"), 300.0)

    def test_unknown_tool_ignored(self):
        self.assertEqual(
            rt.sample_runtime(["bogus", "fpocket"], self._rt(),
                              "sequential"), 5.0)

    def test_empty(self):
        self.assertEqual(rt.sample_runtime([], self._rt(), "sequential"), 0.0)


class TestBuildRowAndCli(unittest.TestCase):
    def test_build_row_minutes(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            sel = td / "sel"
            # sample a: boltz2(300)+fpocket(5)=305s; sample b: p2rank(10)=10s
            _write_sel(sel, "a", ["boltz2", "fpocket"])
            _write_sel(sel, "b", ["p2rank"])
            runtimes, _ = rt.resolve_runtimes(None)
            row = rt.build_row("free", sel, runtimes)
            self.assertEqual(row["n_samples"], 2)
            self.assertEqual(row["avg_tools"], 1.5)
            # mean(305,10)/60 = 157.5/60 = 2.625 → 2.62
            self.assertEqual(row["runtime_sequential_min"], 2.62)
            # parallel: mean(300,10)/60 = 155/60 = 2.583 → 2.58
            self.assertEqual(row["runtime_parallel_min"], 2.58)

    def test_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            free, mand, alls = td / "f", td / "m", td / "a"
            _write_sel(free, "x", ["boltz2", "equipnas"])
            _write_sel(mand, "x", ["boltz2", "chai1", "rosettafold2na",
                                   "rfaa", "alphafold3", "equipnas"])
            _write_sel(alls, "x", list(ALL_KNOWN_TOOLS))
            out = td / "rt.csv"
            rc = rt.main(["--free-sel", str(free),
                          "--mandatory-sel", str(mand),
                          "--all-sel", str(alls), "--output", str(out)])
            self.assertEqual(rc, 0)
            with out.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([r["policy"] for r in rows],
                             ["free", "mandatory", "all"])
            # all-policy sequential = sum of all 15 estimates / 60
            total = sum(rt.DEFAULT_RUNTIME_SECONDS.values()) / 60.0
            self.assertAlmostEqual(
                float(rows[2]["runtime_sequential_min"]), round(total, 2),
                places=2)
            self.assertEqual(float(rows[2]["avg_tools"]), 15.0)


if __name__ == "__main__":
    unittest.main()
