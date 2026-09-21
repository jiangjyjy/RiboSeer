"""Mock tests for scripts/riboseer/make_table11_selections.py."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.riboseer import make_table11_selections as mt  # noqa: E402
from step5_fusion.features_15tool import (  # noqa: E402
    ALL_KNOWN_TOOLS, TOOL_CATEGORY,
)


def _write_sel(directory, sid, tools):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{sid}.json").write_text(json.dumps({
        "sample_id": sid, "selected_tools": tools,
        "reasoning": "llm picked", "category": "novel_x_stem-loop",
        "available_tools": list(ALL_KNOWN_TOOLS), "source": "llm"}),
        encoding="utf-8")


class TestMandatoryConstant(unittest.TestCase):
    def test_cat_a_set_is_the_five_structure_tools(self):
        self.assertEqual(
            set(mt.MANDATORY_CAT_A),
            {"boltz2", "chai1", "rosettafold2na", "rfaa", "alphafold3"})
        # every mandatory tool is Category A
        for t in mt.MANDATORY_CAT_A:
            self.assertEqual(TOOL_CATEGORY[t], "A")


class TestCleanTools(unittest.TestCase):
    def test_aliases_canonicalised(self):
        self.assertEqual(mt.clean_tools(["rf2na", "af3"]),
                         ["rosettafold2na", "alphafold3"])

    def test_unknown_dropped_and_deduped_in_library_order(self):
        out = mt.clean_tools(["equipnas", "bogus", "boltz2", "boltz2"])
        self.assertEqual(out, ["boltz2", "equipnas"])  # library order, deduped

    def test_empty(self):
        self.assertEqual(mt.clean_tools([]), [])


class TestPolicies(unittest.TestCase):
    def test_mandatory_unions_cat_a(self):
        # LLM picked only two Cat-C tools → all 5 Cat-A get forced in.
        out = mt.mandatory_tools(["equipnas", "nucleicnet"])
        self.assertTrue(set(mt.MANDATORY_CAT_A) <= set(out))
        self.assertIn("equipnas", out)
        self.assertIn("nucleicnet", out)
        self.assertEqual(len(out), 7)

    def test_mandatory_no_double_count_when_llm_already_picked_cat_a(self):
        out = mt.mandatory_tools(["boltz2", "equipnas"])
        self.assertEqual(out.count("boltz2"), 1)
        self.assertEqual(len(out), 6)  # 5 Cat-A + equipnas

    def test_mandatory_library_order(self):
        out = mt.mandatory_tools(["nucleicnet", "boltz2"])
        idx = [ALL_KNOWN_TOOLS.index(t) for t in out]
        self.assertEqual(idx, sorted(idx))

    def test_all_policy_is_full_library(self):
        self.assertEqual(mt.all_tools(), list(ALL_KNOWN_TOOLS))
        self.assertEqual(len(mt.all_tools()), 15)

    def test_all_ignores_original(self):
        self.assertEqual(mt.selected_for_policy(["boltz2"], "all"),
                         list(ALL_KNOWN_TOOLS))

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            mt.selected_for_policy(["boltz2"], "nope")


class TestBuildRecord(unittest.TestCase):
    def test_keeps_metadata_and_records_provenance(self):
        rec = {"sample_id": "1abc_A_E", "selected_tools": ["equipnas", "af3"],
               "category": "novel_x_stem-loop", "reasoning": "r"}
        out = mt.build_record(rec, "mandatory")
        self.assertEqual(out["policy"], "mandatory")
        self.assertEqual(out["source"], "table11_mandatory")
        self.assertEqual(out["category"], "novel_x_stem-loop")
        # llm picks preserved (canonicalised) under a separate key
        self.assertEqual(out["llm_selected_tools"], ["alphafold3", "equipnas"])
        self.assertIn("alphafold3", out["selected_tools"])  # af3 canonical + Cat-A


class TestAvgTools(unittest.TestCase):
    def test_mean_len(self):
        recs = {"a": {"selected_tools": ["boltz2", "chai1"]},
                "b": {"selected_tools": ["boltz2", "chai1", "equipnas", "af3"]}}
        self.assertEqual(mt.avg_tools_per_sample(recs), 3.0)

    def test_empty_dir(self):
        self.assertIsNone(mt.avg_tools_per_sample({}))


class TestGenerateAndCli(unittest.TestCase):
    def test_generate_mandatory(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src, out = td / "in", td / "out"
            _write_sel(src, "1abc_A_E", ["equipnas", "nucleicnet"])
            _write_sel(src, "2def_B_F", ["boltz2"])
            stats = mt.generate(src, out, "mandatory")
            self.assertEqual(stats["written"], 2)
            rec = json.loads((out / "1abc_A_E.json").read_text())
            self.assertTrue(set(mt.MANDATORY_CAT_A) <= set(rec["selected_tools"]))
            # avg = (7 + 5) / 2 = 6.0
            self.assertEqual(stats["avg_tools"], 6.0)

    def test_generate_all(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src, out = td / "in", td / "out"
            _write_sel(src, "1abc_A_E", ["equipnas"])
            stats = mt.generate(src, out, "all")
            self.assertEqual(stats["avg_tools"], 15.0)
            rec = json.loads((out / "1abc_A_E.json").read_text())
            self.assertEqual(rec["selected_tools"], list(ALL_KNOWN_TOOLS))

    def test_cli_writes_and_reports(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src, out = td / "in", td / "out"
            _write_sel(src, "1abc_A_E", ["equipnas", "nucleicnet"])
            rc = mt.main(["--policy", "mandatory",
                          "--input-dir", str(src), "--out-dir", str(out)])
            self.assertEqual(rc, 0)
            self.assertTrue((out / "1abc_A_E.json").is_file())

    def test_cli_missing_input_dir(self):
        with tempfile.TemporaryDirectory() as td:
            rc = mt.main(["--policy", "all",
                          "--input-dir", str(Path(td) / "nope"),
                          "--out-dir", str(Path(td) / "out")])
            self.assertEqual(rc, 1)

    def test_cli_empty_input_dir(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src, out = td / "in", td / "out"
            src.mkdir()
            rc = mt.main(["--policy", "all",
                          "--input-dir", str(src), "--out-dir", str(out)])
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
