"""Mock tests for scripts/riboseer/summarize_table11.py."""
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

from scripts.riboseer import summarize_table11 as st  # noqa: E402

# columns table09_llm_modules.py writes
_ABL_COLS = ["scope", "maestro", "polish", "n_samples",
             "pearson_r_mean", "pearson_r_std", "pearson_r_median",
             "spearman_r_mean", "r2_mean"]


def _write_ablation_csv(path, target_pearson, target_r2, n=107):
    """8-combo CSV; the scope=on/maestro=on/polish=off row carries the
    target metrics, every other row a distractor value."""
    combos = [(s, m, p) for s in ("off", "on") for m in ("off", "on")
              for p in ("off", "on")]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_ABL_COLS)
        w.writeheader()
        for s, m, p in combos:
            hit = (s == "on" and m == "on" and p == "off")
            w.writerow({
                "scope": s, "maestro": m, "polish": p, "n_samples": n,
                "pearson_r_mean": target_pearson if hit else 0.111,
                "pearson_r_std": 0.05, "pearson_r_median": 0.5,
                "spearman_r_mean": 0.4,
                "r2_mean": target_r2 if hit else 0.222})


def _write_sel(directory, sid, tools):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{sid}.json").write_text(
        json.dumps({"sample_id": sid, "selected_tools": tools}),
        encoding="utf-8")


class TestReadAblationRow(unittest.TestCase):
    def test_picks_the_right_row(self):
        with tempfile.TemporaryDirectory() as td:
            csvp = Path(td) / "abl.csv"
            _write_ablation_csv(csvp, 0.569, 0.401)
            rec = st.read_ablation_row(csvp)
            self.assertEqual(float(rec["pearson_r_mean"]), 0.569)
            self.assertEqual(float(rec["r2_mean"]), 0.401)

    def test_missing_file(self):
        self.assertIsNone(st.read_ablation_row(Path("/no/such.csv")))


class TestBuildRow(unittest.TestCase):
    def test_combines_metrics_and_tool_count(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            csvp = td / "abl.csv"
            _write_ablation_csv(csvp, 0.59, 0.42)
            sel = td / "sel"
            _write_sel(sel, "a", ["boltz2", "chai1"])
            _write_sel(sel, "b", ["boltz2", "chai1", "equipnas", "alphafold3"])
            row = st.build_row("mandatory", csvp, sel)
            self.assertEqual(row["pearson_r"], 0.59)
            self.assertEqual(row["r_squared"], 0.42)
            self.assertEqual(row["tools_per_sample"], 3.0)
            self.assertEqual(row["n_samples"], 107)

    def test_missing_csv_yields_none_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            sel = td / "sel"
            _write_sel(sel, "a", ["boltz2"])
            row = st.build_row("all", td / "nope.csv", sel)
            self.assertIsNone(row["pearson_r"])
            self.assertEqual(row["tools_per_sample"], 1.0)


class TestRuntimeMerge(unittest.TestCase):
    def test_runtime_csv_read_and_merged(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            rtcsv = td / "rt.csv"
            with rtcsv.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "policy", "n_samples", "avg_tools",
                    "runtime_sequential_min", "runtime_parallel_min"])
                w.writeheader()
                w.writerow({"policy": "free", "n_samples": 107,
                            "avg_tools": 5.7, "runtime_sequential_min": 28.5,
                            "runtime_parallel_min": 10.0})
            rtm = st.read_runtime_csv(rtcsv)
            self.assertEqual(rtm["free"], 28.5)
            # build_row threads it onto the row
            csvp = td / "abl.csv"
            _write_ablation_csv(csvp, 0.569, 0.401)
            sel = td / "sel"
            _write_sel(sel, "a", ["boltz2"])
            row = st.build_row("free", csvp, sel, rtm.get("free"))
            self.assertEqual(row["runtime_min"], 28.5)

    def test_missing_runtime_csv_is_blank(self):
        self.assertEqual(st.read_runtime_csv(None), {})
        self.assertEqual(st.read_runtime_csv(Path("/no/such.csv")), {})


class TestCli(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            free_csv, mand_csv, all_csv = (td / "f.csv", td / "m.csv",
                                           td / "a.csv")
            _write_ablation_csv(free_csv, 0.569, 0.401)
            _write_ablation_csv(mand_csv, 0.560, 0.395)
            _write_ablation_csv(all_csv, 0.540, 0.380)
            free_sel, mand_sel, all_sel = (td / "fs", td / "ms", td / "as")
            _write_sel(free_sel, "a", ["boltz2", "chai1", "equipnas"])
            _write_sel(mand_sel, "a", ["boltz2", "chai1", "rosettafold2na",
                                       "rfaa", "alphafold3", "equipnas"])
            _write_sel(all_sel, "a",
                       ["boltz2", "chai1", "rosettafold2na", "rfaa",
                        "alphafold3", "p2rank", "fpocket", "deeppocket",
                        "equipnas", "nucleicnet", "graphbind",
                        "rnabindrplus", "bindup", "hdock", "haddock3"])
            out = td / "table11.csv"
            rc = st.main([
                "--free-csv", str(free_csv), "--free-sel", str(free_sel),
                "--mandatory-csv", str(mand_csv), "--mandatory-sel",
                str(mand_sel),
                "--all-csv", str(all_csv), "--all-sel", str(all_sel),
                "--output", str(out)])
            self.assertEqual(rc, 0)
            with out.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([r["policy"] for r in rows],
                             ["free", "mandatory", "all"])
            free = rows[0]
            self.assertEqual(float(free["pearson_r"]), 0.569)
            self.assertEqual(float(free["tools_per_sample"]), 3.0)
            self.assertEqual(float(rows[2]["tools_per_sample"]), 15.0)


if __name__ == "__main__":
    unittest.main()
