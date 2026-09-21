"""Mock tests for scripts/tables/table13_prompt.py."""
import csv
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

from scripts.tables import table13_prompt as rt  # noqa: E402

_ABL_COLS = ["scope", "maestro", "polish", "n_samples",
             "pearson_r_mean", "pearson_r_std", "pearson_r_median",
             "spearman_r_mean", "r2_mean"]


def _write_ablation_csv(path, pear, spear, r2, n=107):
    combos = [(s, m, p) for s in ("off", "on") for m in ("off", "on")
              for p in ("off", "on")]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_ABL_COLS)
        w.writeheader()
        for s, m, p in combos:
            hit = (s == "on" and m == "on" and p == "on")
            w.writerow({"scope": s, "maestro": m, "polish": p, "n_samples": n,
                        "pearson_r_mean": pear if hit else 0.1,
                        "pearson_r_std": 0.05, "pearson_r_median": 0.5,
                        "spearman_r_mean": spear if hit else 0.2,
                        "r2_mean": r2 if hit else 0.05})


def _write_json(directory, name, obj):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(obj), encoding="utf-8")


class TestConfigMatrix(unittest.TestCase):
    def test_five_cells_and_default(self):
        self.assertEqual(len(rt.CONFIGS), 5)
        defaults = [c for c in rt.CONFIGS if rt.is_default(c)]
        self.assertEqual(len(defaults), 1)
        self.assertTrue(defaults[0]["cot"])
        self.assertEqual(defaults[0]["temperature"], 0.7)

    def test_config_tag(self):
        self.assertEqual(rt.config_tag({"cot": True, "temperature": 0.7}),
                         "cot1_t07")
        self.assertEqual(rt.config_tag({"cot": False, "temperature": 0.0}),
                         "cot0_t00")
        self.assertEqual(rt.config_tag({"cot": True, "temperature": 1.0}),
                         "cot1_t10")


class TestBuildArgvs(unittest.TestCase):
    def _paths(self, td):
        return {k: td / k for k in (
            "processed_dir", "train_list", "test_list", "train_step4_dir",
            "test_step4_dir", "weight_tensor", "enriched_model_dir",
            "step6_dir", "scope_config", "maestro_config", "polish_config")}

    def test_style_flags_threaded(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg = {"cot": False, "temperature": 0.0}
            dirs = rt.cell_dirs(td / "work", cfg)
            argvs = rt.build_stage_argvs(cfg, self._paths(td), dirs)
            for stage in ("scope_train", "maestro_test", "polish_test"):
                self.assertIn("--no-cot", argvs[stage])
                self.assertIn("--temperature", argvs[stage])
                self.assertIn("0.0", argvs[stage])
                self.assertIn("--mode", argvs[stage])
                self.assertIn("llm", argvs[stage])

    def test_cot_on_uses_cot_flag(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg = {"cot": True, "temperature": 1.0}
            dirs = rt.cell_dirs(td / "work", cfg)
            argvs = rt.build_stage_argvs(cfg, self._paths(td), dirs)
            self.assertIn("--cot", argvs["scope_test"])
            self.assertNotIn("--no-cot", argvs["scope_test"])

    def test_ablation_argv_wires_all_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            cfg = {"cot": True, "temperature": 0.7}
            dirs = rt.cell_dirs(td / "work", cfg)
            ab = rt.build_stage_argvs(cfg, self._paths(td), dirs)["ablation"]
            for flag in ("--scope-profiles-train", "--scope-profiles-test",
                         "--maestro-selections-train",
                         "--maestro-selections-test", "--polish-actions",
                         "--output"):
                self.assertIn(flag, ab)
            self.assertIn(str(dirs["polish_test"]), ab)


class TestFailureRate(unittest.TestCase):
    def test_combined_rate(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            dirs = {k: td / k for k in (
                "scope_train", "scope_test", "maestro_train",
                "maestro_test", "polish_test")}
            # scope_test: 2 llm + 1 fallback
            _write_json(dirs["scope_test"], "a", {"source": "llm"})
            _write_json(dirs["scope_test"], "b", {"source": "llm"})
            _write_json(dirs["scope_test"], "c", {"source": "cauto_fallback"})
            # maestro_test: 1 llm + 1 fallback
            _write_json(dirs["maestro_test"], "a", {"source": "llm"})
            _write_json(dirs["maestro_test"], "b", {"source": "llm_fallback"})
            # polish_test: rounds 1 llm + 1 fallback
            _write_json(dirs["polish_test"], "a", {"rounds": [
                {"source": "llm"}, {"source": "llm_fallback"}]})
            # make empty dirs exist
            for k in ("scope_train", "maestro_train"):
                dirs[k].mkdir(parents=True, exist_ok=True)
            fr = rt.failure_rate(dirs)
            self.assertEqual(fr["attempts"], 7)
            self.assertEqual(fr["fails"], 3)
            self.assertEqual(fr["pct_fail"], round(100 * 3 / 7, 1))

    def test_empty_dirs_none(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            dirs = {k: td / k for k in (
                "scope_train", "scope_test", "maestro_train",
                "maestro_test", "polish_test")}
            self.assertIsNone(rt.failure_rate(dirs)["pct_fail"])


class TestReadOnonAndAssemble(unittest.TestCase):
    def test_read_onon_row(self):
        with tempfile.TemporaryDirectory() as td:
            csvp = Path(td) / "abl.csv"
            _write_ablation_csv(csvp, 0.565, 0.446, 0.395)
            rec = rt.read_onon_row(csvp)
            self.assertEqual(float(rec["pearson_r_mean"]), 0.565)
            self.assertEqual(float(rec["spearman_r_mean"]), 0.446)

    def test_assemble_row(self):
        with tempfile.TemporaryDirectory() as td:
            csvp = Path(td) / "abl.csv"
            _write_ablation_csv(csvp, 0.565, 0.446, 0.395)
            row = rt.assemble_row({"cot": True, "temperature": 0.7},
                                  csvp, {"pct_fail": 2.0}, reused=True)
            self.assertEqual(row["cot"], 1)
            self.assertEqual(row["pearson_r_mean"], 0.565)
            self.assertEqual(row["pct_fail"], 2.0)
            self.assertEqual(row["status"], "reused")
            self.assertEqual(row["n_samples"], 107)


class TestDriver(unittest.TestCase):
    def _common_args(self, td, output, work, extra):
        a = []
        for flag in ("--processed-dir", "--train-list", "--test-list",
                     "--train-step4-dir", "--test-step4-dir",
                     "--weight-tensor", "--enriched-model-dir", "--step6-dir",
                     "--scope-config", "--maestro-config", "--polish-config"):
            a += [flag, str(td / flag.strip("-"))]
        a += ["--work-dir", str(work), "--output", str(output)]
        return a + extra

    def test_reused_default_skips_generation(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            done = td / "done.csv"
            _write_ablation_csv(done, 0.565, 0.446, 0.395)
            out = td / "table14.csv"
            args = self._common_args(
                td, out, td / "work",
                ["--done-ablation-csv", str(done), "--only", "cot1_t07"])
            with mock.patch.object(rt, "run_cell_generation",
                                   side_effect=AssertionError("should reuse")):
                rc = rt.main(args)
            self.assertEqual(rc, 0)
            with out.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "reused")
            self.assertEqual(float(rows[0]["pearson_r_mean"]), 0.565)

    def test_runs_a_fresh_cell(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            out = td / "table14.csv"
            work = td / "work"

            def fake_gen(cfg, paths, dirs):
                _write_json(dirs["maestro_test"], "a", {"source": "llm"})
                _write_json(dirs["maestro_test"], "b",
                            {"source": "llm_fallback"})
                _write_ablation_csv(dirs["ablation_csv"], 0.55, 0.44, 0.39)

            args = self._common_args(td, out, work, ["--only", "cot0_t00"])
            with mock.patch.object(rt, "run_cell_generation", fake_gen):
                rc = rt.main(args)
            self.assertEqual(rc, 0)
            with out.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["cot"], "0")
            self.assertEqual(rows[0]["status"], "run")
            self.assertEqual(float(rows[0]["pearson_r_mean"]), 0.55)
            self.assertEqual(float(rows[0]["pct_fail"]), 50.0)  # 1/2 fallback


if __name__ == "__main__":
    unittest.main()
