"""Mock tests for src/step4_tool_adapters/external/af3_inputs.py.

Pure-function coverage (sequence cleaning, row aggregation, JSON
shape, submission ordering) plus an end-to-end CLI smoke that writes
real files into a tempdir.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import step4_tool_adapters.external.af3_inputs as caf  # noqa: E402


# ---- cleaning primitives ------------------------------------------------


class TestCleanSequence(unittest.TestCase):

    def test_strips_gap_dash(self):
        out, n = caf.clean_sequence("-APV-LE-NR-", caf.STANDARD_AA)
        self.assertEqual(out, "APVLENR")
        self.assertEqual(n, 4)

    def test_strips_dot_and_spaces(self):
        out, n = caf.clean_sequence("M K  . R\tY", caf.STANDARD_AA)
        self.assertEqual(out, "MKRY")
        # 1 space + 2 spaces + 1 dot + 1 space + 1 tab = 6 stripped.
        self.assertEqual(n, 6)

    def test_strips_nonstandard_aa(self):
        # B (asx), J, O (pyrrolysine), U (selenocysteine), X, Z (glx)
        # all dropped; standard letters survive.
        out, n = caf.clean_sequence("BJOUXZACDEFGHIKLMNPQRSTVWY",
                                    caf.STANDARD_AA)
        self.assertEqual(out, "ACDEFGHIKLMNPQRSTVWY")
        self.assertEqual(n, 6)

    def test_strips_n_and_t_from_rna(self):
        out, n = caf.clean_sequence("GGGGGUGNAAACGGUCUCGACT",
                                    caf.STANDARD_RNA)
        # N and T are non-standard for RNA.
        self.assertEqual(out, "GGGGGUGAAACGGUCUCGAC")
        self.assertEqual(n, 2)

    def test_uppercases_lowercase_input(self):
        out, n = caf.clean_sequence("acGu", caf.STANDARD_RNA)
        self.assertEqual(out, "ACGU")
        # Lowercase letters in the standard set are kept after upcase.
        self.assertEqual(n, 0)

    def test_empty(self):
        self.assertEqual(caf.clean_sequence("", caf.STANDARD_AA),
                         ("", 0))


class TestCleanRow(unittest.TestCase):

    def test_no_change_marks_unmodified(self):
        r = caf.clean_row("s1", "MKRY", "ACGU")
        self.assertFalse(r.modified)
        self.assertEqual(r.n_prot_removed, 0)
        self.assertEqual(r.n_rna_removed, 0)
        self.assertEqual(r.prot_len, 4)
        self.assertEqual(r.rna_len, 4)

    def test_strip_marks_modified(self):
        r = caf.clean_row("s1", "-MKRY", "AC.GU")
        self.assertTrue(r.modified)
        self.assertEqual(r.n_prot_removed, 1)
        self.assertEqual(r.n_rna_removed, 1)
        self.assertEqual(r.prot_len, 4)
        self.assertEqual(r.rna_len, 4)

    def test_uppercase_counts_as_modified(self):
        # Lowercase input → cleaned output is uppercase even though
        # no chars are dropped — flag as modified so the user can see
        # the change in stats.
        r = caf.clean_row("s1", "mkry", "ACGU")
        self.assertTrue(r.modified)
        self.assertEqual(r.n_prot_removed, 0)
        self.assertEqual(r.protein_seq, "MKRY")


# ---- JSON job shape -----------------------------------------------------


class TestBuildJobJson(unittest.TestCase):
    """Pin the AF Server JSON shape. ``rnaSequence`` (not
    ``rnaChain``), top-level array, dialect + version present —
    these were the things AF Server signalled were wrong before
    the fix."""

    def test_top_level_is_a_list(self):
        r = caf.clean_row("2czj_E_F", "APVLENR", "GGGGGUG")
        payload = caf.build_job_json(r)
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 1)

    def test_inner_job_schema_matches_af_server(self):
        r = caf.clean_row("2czj_E_F", "APVLENR", "GGGGGUG")
        job = caf.build_job_object(r)
        self.assertEqual(job["name"], "2czj_E_F")
        self.assertEqual(job["modelSeeds"], [42])
        self.assertEqual(job["dialect"], "alphafoldserver")
        self.assertEqual(job["version"], 1)
        self.assertEqual(len(job["sequences"]), 2)
        # Protein chain — key is ``proteinChain``.
        self.assertEqual(job["sequences"][0],
                         {"proteinChain":
                          {"sequence": "APVLENR", "count": 1}})
        # RNA chain — key is ``rnaSequence`` (NOT ``rnaChain``).
        # If this assertion ever fails because someone "fixed" the
        # naming back to rnaChain, AF Server will say "No jobs found
        # in file" and the test must catch it first.
        self.assertEqual(job["sequences"][1],
                         {"rnaSequence":
                          {"sequence": "GGGGGUG", "count": 1}})
        self.assertNotIn("rnaChain",
                         {k for d in job["sequences"] for k in d})

    def test_custom_seeds_override(self):
        r = caf.clean_row("s1", "MKRY", "ACGU")
        payload = caf.build_job_json(r, model_seeds=[1, 2, 3])
        self.assertEqual(payload[0]["modelSeeds"], [1, 2, 3])


# ---- submission order ---------------------------------------------------


class TestSubmissionOrder(unittest.TestCase):

    def test_sorted_by_total_len_asc(self):
        rows = [
            caf.clean_row("big", "M" * 200, "A" * 100),     # 300
            caf.clean_row("tiny", "MKRY", "ACGU"),          # 8
            caf.clean_row("mid", "M" * 50, "A" * 50),       # 100
        ]
        order = caf.build_submission_order(rows)
        self.assertEqual([o["sample_id"] for o in order],
                         ["tiny", "mid", "big"])
        # priority is 1-based.
        self.assertEqual([o["priority"] for o in order], [1, 2, 3])
        self.assertEqual(order[0]["total_len"], 8)
        self.assertEqual(order[-1]["total_len"], 300)

    def test_deterministic_tie_break(self):
        # Same total length, different prot_len → shorter prot first.
        a = caf.clean_row("a", "M" * 50, "A" * 50)   # prot 50
        b = caf.clean_row("b", "M" * 30, "A" * 70)   # prot 30
        order = caf.build_submission_order([a, b])
        self.assertEqual([o["sample_id"] for o in order], ["b", "a"])

    def test_sample_id_tiebreak_after_lens_match(self):
        # Same prot_len + rna_len → alphabetical sample_id wins.
        rows = [
            caf.clean_row("zzz", "M" * 50, "A" * 50),
            caf.clean_row("aaa", "M" * 50, "A" * 50),
        ]
        order = caf.build_submission_order(rows)
        self.assertEqual([o["sample_id"] for o in order],
                         ["aaa", "zzz"])


# ---- TSV IO -------------------------------------------------------------


class TestReadInputTsv(unittest.TestCase):

    def test_missing_required_columns_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.tsv"
            p.write_text("sample_id\tprotein_seq\nx\tM\n",
                         encoding="utf-8")  # no rna_seq
            with self.assertRaises(ValueError) as cm:
                caf.read_input_tsv(p)
            self.assertIn("rna_seq", str(cm.exception))

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            src = tmp / "in.tsv"
            src.write_text(
                "sample_id\tprotein_seq\trna_seq\tprot_len\trna_len\n"
                "x1\t-MKRY-\tACGU\t6\t4\n"
                "x2\tMKRY\tGGGGGUGN\t4\t8\n",
                encoding="utf-8")
            rows = caf.read_input_tsv(src)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["sample_id"], "x1")
            self.assertEqual(rows[0]["protein_seq"], "-MKRY-")


# ---- end-to-end CLI ----------------------------------------------------


class TestCli(unittest.TestCase):

    def test_main_writes_all_three_artefacts(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            src = tmp / "af3_inputs.tsv"
            src.write_text(
                "sample_id\tprotein_seq\trna_seq\tprot_len\trna_len\n"
                "big\t-MKRYACDEFGHIKLMNPQRSTVWY-\tGGGGGGGGGG\t26\t10\n"
                "tiny\tMKRY\tACGU\t4\t4\n"
                "empty\t-..-\t.NN.\t0\t0\n",
                encoding="utf-8")
            out_dir = tmp / "out"
            rc = caf.main([
                "--input", str(src),
                "--output-dir", str(out_dir),
            ])
            self.assertEqual(rc, 0)
            # 1. Clean TSV present with re-computed lengths.
            clean_tsv = out_dir / "af3_inputs_clean.tsv"
            self.assertTrue(clean_tsv.is_file())
            with clean_tsv.open(encoding="utf-8") as f:
                rd = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual({r["sample_id"] for r in rd},
                             {"big", "tiny", "empty"})
            big = next(r for r in rd if r["sample_id"] == "big")
            self.assertEqual(big["protein_seq"],
                             "MKRYACDEFGHIKLMNPQRSTVWY")
            self.assertEqual(big["prot_len"], "24")
            # 2. JSON jobs subdir — "empty" skipped (no protein after
            # clean), big + tiny present.
            jobs = out_dir / "af3_jobs"
            self.assertTrue(jobs.is_dir())
            names = sorted(p.stem for p in jobs.glob("*.json"))
            self.assertEqual(names, ["big", "tiny"])
            with (jobs / "tiny.json").open(encoding="utf-8") as f:
                payload = json.load(f)
            # AF Server expects a top-level list.
            self.assertIsInstance(payload, list)
            self.assertEqual(len(payload), 1)
            job = payload[0]
            self.assertEqual(job["sequences"][0]["proteinChain"][
                "sequence"], "MKRY")
            # RNA key is ``rnaSequence``.
            self.assertEqual(job["sequences"][1]["rnaSequence"][
                "sequence"], "ACGU")
            self.assertEqual(job["modelSeeds"], [42])
            self.assertEqual(job["dialect"], "alphafoldserver")
            self.assertEqual(job["version"], 1)
            # 3. Submission order CSV — sorted by total_len asc,
            #    "empty" lands at priority 1 (total=0), then tiny, big.
            order_csv = out_dir / "af3_submission_order.csv"
            self.assertTrue(order_csv.is_file())
            with order_csv.open(encoding="utf-8") as f:
                ord_rows = list(csv.DictReader(f))
            self.assertEqual([r["sample_id"] for r in ord_rows],
                             ["empty", "tiny", "big"])
            self.assertEqual([int(r["priority"]) for r in ord_rows],
                             [1, 2, 3])

    def test_main_missing_input_returns_1(self):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            rc = caf.main([
                "--input", str(tmp / "nope.tsv"),
                "--output-dir", str(tmp / "out"),
            ])
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
