"""Tests for scripts/riboseer/run_cmsearch_classification.py — no external
tools. Covers FASTA extraction, Infernal/HMMER tblout parsing (real column
layouts), best-hit-by-E-value, the classified/unclassified threshold, and
the summary CSV.
"""
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

from scripts.riboseer import run_cmsearch_classification as cc  # noqa: E402


# Real cmsearch --tblout layout: target(seq) acc query(family) query-acc
# mdl mfrom mto sfrom sto strand trunc pass gc bias score E-value inc desc
_CM_TBL = """\
#target name         accession query name           accession ...
4n0t_A_B             -         tRNA                 RF00005   cm 1 71 1 71 + no 1 0.55 0.0 65.3 2.1e-30 ! desc
4n0t_A_B             -         5S_rRNA              RF00001   cm 1 119 1 60 + no 1 0.5 0.0 20.1 3.4e-02 ? desc
6xh2_A_D             -         SSU_rRNA_eukarya     RF01960   cm 1 1800 1 200 + no 1 0.5 0.0 120.0 1.2e-45 ! desc
"""

# Real hmmscan --tblout layout: target(domain) acc query(seq) qacc
# Efull score bias Edom ...
_HMM_TBL = """\
#target name        accession  query name           accession    E-value ...
RRM_1                PF00076.20 4n0t_A_B             -            3.4e-12  40.1 0.1 5e-12 39 0.1 1 1 1 1 1 0 0 0 desc
KH_1                 PF00013.29 4n0t_A_B             -            2.0e-03  12.0 0.1 3e-03 11 0.1 1 1 1 1 1 0 0 0 desc
zf-CCCH              PF00642.25 6xh2_A_D             -            1.5e-15  55.0 0.1 2e-15 54 0.1 1 1 1 1 1 0 0 0 desc
"""


def _write(p, txt):
    p.write_text(txt, encoding="utf-8")


# ---- extraction ---------------------------------------------------------


class TestExtract(unittest.TestCase):
    def _make(self, td):
        proc = td / "proc"
        (proc / "samples").mkdir(parents=True)
        (proc / "samples" / "4n0t_A_B.json").write_text(json.dumps({
            "sample_id": "4n0t_A_B",
            "protein": {"sequence": "MKVLAA" * 5},
            "rna": {"sequence": "AUGCGUACG" * 4}}), encoding="utf-8")
        (proc / "samples" / "no_rna.json").write_text(json.dumps({
            "sample_id": "no_rna",
            "protein": {"sequence": "MKV"}, "rna": {"sequence": ""}}),
            encoding="utf-8")
        return proc

    def test_writes_both_fastas(self):
        with tempfile.TemporaryDirectory() as t:
            td = Path(t)
            proc = self._make(td)
            out = td / "out"
            n_rna, n_prot = cc.extract_fastas(
                proc, ["4n0t_A_B", "no_rna"], out)
            self.assertEqual(n_rna, 1)        # no_rna skipped
            self.assertEqual(n_prot, 2)
            rna = (out / cc.RNA_FASTA).read_text(encoding="utf-8")
            self.assertIn(">4n0t_A_B", rna)
            self.assertIn("AUGCGUACG", rna.replace("\n", ""))
            self.assertNotIn(">no_rna", rna)
            prot = (out / cc.PROTEIN_FASTA).read_text(encoding="utf-8")
            self.assertIn(">no_rna", prot)


# ---- parsing ------------------------------------------------------------


class TestParse(unittest.TestCase):
    def test_infernal_best_hit(self):
        with tempfile.TemporaryDirectory() as t:
            tbl = Path(t) / "rna.tbl"
            _write(tbl, _CM_TBL)
            hits = cc.parse_infernal_tbl(tbl)
        # 4n0t: tRNA (2.1e-30) beats 5S_rRNA (3.4e-02).
        self.assertEqual(hits["4n0t_A_B"]["family"], "tRNA")
        self.assertEqual(hits["4n0t_A_B"]["accession"], "RF00005")
        self.assertAlmostEqual(hits["4n0t_A_B"]["evalue"], 2.1e-30)
        self.assertEqual(hits["6xh2_A_D"]["family"], "SSU_rRNA_eukarya")

    def test_hmmer_best_hit_and_version_strip(self):
        with tempfile.TemporaryDirectory() as t:
            tbl = Path(t) / "prot.tbl"
            _write(tbl, _HMM_TBL)
            hits = cc.parse_hmmer_tbl(tbl)
        # 4n0t: RRM_1 (3.4e-12) beats KH_1 (2.0e-3).
        self.assertEqual(hits["4n0t_A_B"]["domain"], "RRM_1")
        self.assertEqual(hits["4n0t_A_B"]["accession"], "PF00076")  # .20 stripped
        self.assertEqual(hits["6xh2_A_D"]["domain"], "zf-CCCH")

    def test_missing_file_is_empty(self):
        self.assertEqual(cc.parse_infernal_tbl(Path("/no/such.tbl")), {})
        self.assertEqual(cc.parse_hmmer_tbl(Path("/no/such.tbl")), {})


# ---- summary ------------------------------------------------------------


class TestSummary(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as t:
            td = Path(t)
            _write(td / "r.tbl", _CM_TBL)
            _write(td / "p.tbl", _HMM_TBL)
            self.rna = cc.parse_infernal_tbl(td / "r.tbl")
            self.prot = cc.parse_hmmer_tbl(td / "p.tbl")

    def test_classified_rows(self):
        rows = cc.build_summary(["4n0t_A_B", "6xh2_A_D"], self.rna, self.prot,
                                rna_evalue=0.01, protein_evalue=0.01)
        by = {r["sample_id"]: r for r in rows}
        self.assertEqual(by["4n0t_A_B"]["rna_rfam_family"], "tRNA")
        self.assertEqual(by["4n0t_A_B"]["protein_pfam_domain"], "RRM_1")
        self.assertEqual(by["4n0t_A_B"]["rna_rfam_accession"], "RF00005")

    def test_unclassified_when_missing_or_weak(self):
        rows = cc.build_summary(["nohit_X_Y"], self.rna, self.prot,
                                rna_evalue=0.01, protein_evalue=0.01)
        self.assertEqual(rows[0]["rna_rfam_family"], cc.UNCLASSIFIED)
        self.assertEqual(rows[0]["protein_pfam_domain"], cc.UNCLASSIFIED)
        self.assertEqual(rows[0]["rna_evalue"], "")

    def test_threshold_filters_weak_hit(self):
        # tighten RNA threshold below the 6xh2 hit's 1.2e-45? no — loosen so a
        # weak hit would pass only if threshold high. Use a sample whose best
        # hit is weak: craft via a stricter threshold on protein KH (2e-3).
        rna = {"s": {"family": "X", "accession": "RFX", "evalue": 0.5}}
        rows = cc.build_summary(["s"], rna, {}, rna_evalue=0.01,
                                protein_evalue=0.01)
        self.assertEqual(rows[0]["rna_rfam_family"], cc.UNCLASSIFIED)  # 0.5>0.01

    def test_csv_round_trip(self):
        rows = cc.build_summary(["4n0t_A_B"], self.rna, self.prot,
                                rna_evalue=0.01, protein_evalue=0.01)
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "summary.csv"
            cc.write_summary(out, rows)
            with out.open(encoding="utf-8") as f:
                back = list(csv.DictReader(f))
        self.assertEqual(back[0]["rna_rfam_family"], "tRNA")
        self.assertEqual(back[0]["protein_pfam_accession"], "PF00076")


# ---- driver modes (no tools) --------------------------------------------


class TestDriver(unittest.TestCase):
    def test_extract_only_mode(self):
        with tempfile.TemporaryDirectory() as t:
            td = Path(t)
            proc = td / "proc"
            (proc / "samples").mkdir(parents=True)
            (proc / "samples" / "a_A_B.json").write_text(json.dumps({
                "sample_id": "a_A_B", "protein": {"sequence": "MKV"},
                "rna": {"sequence": "AUGC"}}), encoding="utf-8")
            split = td / "split.txt"
            split.write_text("a_A_B\n", encoding="utf-8")
            out = td / "out"
            rc = cc.main(["--processed-dir", str(proc), "--split-file",
                          str(split), "--output-dir", str(out),
                          "--extract-only"])
            self.assertEqual(rc, 0)
            self.assertTrue((out / cc.RNA_FASTA).is_file())
            self.assertTrue((out / cc.PROTEIN_FASTA).is_file())

    def test_parse_only_mode(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            _write(out / cc.RNA_TBL, _CM_TBL)
            _write(out / cc.PROTEIN_TBL, _HMM_TBL)
            rc = cc.main(["--output-dir", str(out), "--parse-only"])
            self.assertEqual(rc, 0)
            with (out / cc.SUMMARY_CSV).open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            ids = {r["sample_id"] for r in rows}
            self.assertEqual(ids, {"4n0t_A_B", "6xh2_A_D"})


if __name__ == "__main__":
    unittest.main()
