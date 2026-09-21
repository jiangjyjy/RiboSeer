"""Mock tests for scripts/riboseer/make_af3_train_inputs.py."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.riboseer import make_af3_train_inputs as mk  # noqa: E402


def _write_sample(proc: Path, sid: str, prot, rna, *, source_pdb=None,
                  chain="A"):
    d = proc / "samples"
    d.mkdir(parents=True, exist_ok=True)
    doc = {
        "sample_id": sid,
        "source_pdb": source_pdb or sid.split("_")[0],
        "protein": {"chain_id": chain, "sequence": prot, "length": len(prot)},
        "rna": {"chain_id": "B", "sequence": rna, "length": len(rna)},
    }
    (d / f"{sid}.json").write_text(json.dumps(doc), encoding="utf-8")


class TestExtract(unittest.TestCase):
    def test_cleans_sequences(self):
        with tempfile.TemporaryDirectory() as td:
            proc = Path(td)
            # X (non-standard AA) dropped; N + gap dropped from RNA
            _write_sample(proc, "1abc_C_E", "ACDX-EF", "AUGN-C",
                          source_pdb="1abc", chain="C")
            s = mk.extract_sample(proc, "1abc_C_E")
            self.assertEqual(s.protein_seq, "ACDEF")
            self.assertEqual(s.rna_seq, "AUGC")
            self.assertEqual(s.prot_len, 5)
            self.assertEqual(s.rna_len, 4)
            self.assertEqual(s.n_prot_removed, 2)   # X and -
            self.assertEqual(s.n_rna_removed, 2)    # N and -
            self.assertTrue(s.ok)
            self.assertEqual(s.protein_name, "1abc_C")

    def test_missing_json(self):
        with tempfile.TemporaryDirectory() as td:
            s = mk.extract_sample(Path(td), "zzzz_X_Y")
            self.assertTrue(s.missing_json)
            self.assertFalse(s.ok)

    def test_empty_after_clean_not_ok(self):
        with tempfile.TemporaryDirectory() as td:
            proc = Path(td)
            _write_sample(proc, "9inf_T_X", "ACDEF", "NNNNN")  # RNA all N
            s = mk.extract_sample(proc, "9inf_T_X")
            self.assertEqual(s.rna_seq, "")
            self.assertFalse(s.ok)


class TestMainEndToEnd(unittest.TestCase):
    def test_day_buckets_and_manifests(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc = td / "proc"
            ids = [f"{i}abc_A_E" for i in range(1, 6)]   # 5 samples
            for sid in ids:
                _write_sample(proc, sid, "ACDEF", "AUGC")
            slist = td / "train.txt"
            slist.write_text("\n".join(ids) + "\n", encoding="utf-8")
            out = td / "af3"
            rc = mk.main(["--sample-list", str(slist),
                          "--processed-dir", str(proc),
                          "--out-dir", str(out), "--per-day", "2"])
            self.assertEqual(rc, 0)
            # 5 samples, 2/day → day1(2), day2(2), day3(1)
            self.assertTrue((out / "day1").is_dir())
            self.assertTrue((out / "day3").is_dir())
            self.assertFalse((out / "day4").exists())
            # JSON only — no txt anywhere
            self.assertEqual(list(out.rglob("*.txt")), [])
            j1 = sorted(p.name for p in (out / "day1").glob("*.json"))
            self.assertEqual(len(j1), 2)
            self.assertTrue(j1[0].startswith("01_"))
            payload = json.loads(
                (out / "day1" / j1[0]).read_text(encoding="utf-8"))
            self.assertIsInstance(payload, list)            # top-level array
            seqs = payload[0]["sequences"]
            self.assertIn("proteinChain", seqs[0])
            self.assertIn("rnaSequence", seqs[1])           # NOT rnaChain
            self.assertEqual(payload[0]["dialect"], "alphafoldserver")
            man = list(csv.DictReader(
                (out / "day1" / "manifest.csv").open(encoding="utf-8")))
            self.assertEqual(len(man), 2)
            self.assertEqual(set(man[0]),
                             {"sample_id", "protein_name",
                              "protein_length", "rna_length"})
            self.assertEqual(man[0]["protein_length"], "5")

    def test_missing_and_empty_skipped_from_text_but_in_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc = td / "proc"
            _write_sample(proc, "1abc_A_E", "ACDEF", "AUGC")
            _write_sample(proc, "2inf_T_X", "ACDEF", "NNNN")  # empty RNA
            # 3abc has no JSON
            slist = td / "train.txt"
            slist.write_text("1abc_A_E\n2inf_T_X\n3abc_A_E\n", encoding="utf-8")
            out = td / "af3"
            rc = mk.main(["--sample-list", str(slist),
                          "--processed-dir", str(proc),
                          "--out-dir", str(out), "--per-day", "30"])
            self.assertEqual(rc, 0)
            jsons = sorted(p.name for p in (out / "day1").glob("*.json"))
            self.assertEqual(jsons, ["01_1abc_A_E.json"])   # only the ok one
            man = list(csv.DictReader(
                (out / "day1" / "manifest.csv").open(encoding="utf-8")))
            # manifest still rosters all 3 (empty ones at length 0)
            self.assertEqual(len(man), 3)
            empty = [m for m in man if m["sample_id"] == "2inf_T_X"][0]
            self.assertEqual(empty["rna_length"], "0")

    def test_raw_fallback_recovers_missing_json(self):
        # No step-1 JSON → recovered from raw via --raw-dir (mocked).
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, raw = td / "proc", td / "raw"
            proc.mkdir()
            raw.mkdir()
            slist = td / "train.txt"
            slist.write_text("9zzz_a_N\n", encoding="utf-8")
            out = td / "af3"
            with mock.patch.object(
                    mk, "extract_seqs_from_raw",
                    lambda rd, sid: ("ACDEFGHIK", "AUGCAUGC")):
                rc = mk.main(["--sample-list", str(slist),
                              "--processed-dir", str(proc),
                              "--raw-dir", str(raw),
                              "--out-dir", str(out), "--per-day", "30"])
            self.assertEqual(rc, 0)
            j = list((out / "day1").glob("*.json"))
            self.assertEqual(len(j), 1)
            payload = json.loads(j[0].read_text(encoding="utf-8"))
            self.assertEqual(
                payload[0]["sequences"][0]["proteinChain"]["sequence"],
                "ACDEFGHIK")


if __name__ == "__main__":
    unittest.main()
