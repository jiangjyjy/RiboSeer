"""Mock tests for the RNABindRPlus FASTA generator and result-parser framework.

No web/email I/O. The FASTA tests pin the gap-stripping + index map (the part
that protects per-residue alignment). The parser tests pin the format-
independent core — the submitted->original residue remap, the tolerant table
parser on a couple of plausible formats, and the step4 merge.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import step4_tool_adapters.external.rnabindrplus_inputs as mk  # noqa: E402
import step4_tool_adapters.external.rnabindrplus_parse as pr  # noqa: E402


def write_sample(samples_dir: Path, sid: str, seq: str, length=None):
    samples_dir.mkdir(parents=True, exist_ok=True)
    (samples_dir / f"{sid}.json").write_text(json.dumps({
        "sample_id": sid, "source_pdb": sid.split("_")[0],
        "protein": {"chain_id": sid.split("_")[1], "sequence": seq,
                    "length": length if length is not None else len(seq)},
    }), encoding="utf-8")


# ----------------------------------------------------------------------------
# FASTA generation
# ----------------------------------------------------------------------------
class TestFasta(unittest.TestCase):
    def test_clean_sequence_drops_gaps_and_records_positions(self):
        # positions 1-2 gaps, 3 X (non-standard), 4-6 real
        clean, kept = mk.clean_sequence("--XACD")
        self.assertEqual(clean, "ACD")
        self.assertEqual(kept, [4, 5, 6])

    def test_generation_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            samples = tmp / "samples"
            write_sample(samples, "1abc_A_E", "--MKR")   # kept MKR at 3,4,5
            write_sample(samples, "2def_B_F", "GGGG")     # all standard
            slist = tmp / "list.txt"
            slist.write_text("1abc_A_E\n2def_B_F\n", encoding="utf-8")
            out = tmp / "out"
            rc = mk.main([
                "--sample-list", str(slist),
                "--samples-dir", str(samples),
                "--out-dir", str(out),
                "--batch-size", "1",
            ])
            self.assertEqual(rc, 0)
            fasta = (out / "rnabindrplus_input.fasta").read_text(encoding="utf-8")
            self.assertIn(">1abc_A_E\nMKR", fasta)
            self.assertIn(">2def_B_F\nGGGG", fasta)
            # batch-size 1 -> two batch files
            self.assertTrue((out / "rnabindrplus_input_batch1.fasta").is_file())
            self.assertTrue((out / "rnabindrplus_input_batch2.fasta").is_file())
            m = json.loads((out / "rnabindrplus_index_map.json").read_text(encoding="utf-8"))
            self.assertEqual(m["1abc_A_E"]["kept_positions"], [3, 4, 5])
            self.assertEqual(m["1abc_A_E"]["n_removed"], 2)
            self.assertEqual(m["2def_B_F"]["kept_positions"], [1, 2, 3, 4])

    def test_missing_sample_json_is_skipped_not_fatal(self):
        # Local data is a subset: a sample id without a JSON on disk must
        # be skipped (warn) instead of crashing the whole run.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            samples = tmp / "samples"
            write_sample(samples, "2def_B_F", "GGGG")
            slist = tmp / "list.txt"
            slist.write_text("9zzz_x_9\n2def_B_F\n", encoding="utf-8")
            out = tmp / "out"
            rc = mk.main([
                "--sample-list", str(slist),
                "--samples-dir", str(samples),
                "--out-dir", str(out),
                "--batch-size", "30",
            ])
            self.assertEqual(rc, 0)
            fasta = (out / "rnabindrplus_input.fasta").read_text(encoding="utf-8")
            self.assertIn(">2def_B_F\nGGGG", fasta)
            self.assertNotIn("9zzz_x_9", fasta)
            m = json.loads(
                (out / "rnabindrplus_index_map.json").read_text(encoding="utf-8"))
            self.assertIn("2def_B_F", m)
            self.assertNotIn("9zzz_x_9", m)


# ----------------------------------------------------------------------------
# parser: real RNABindRPlus "finalpredictions" format
# ----------------------------------------------------------------------------
# Real block: label\t...\tcomma-separated values; we take the RNABindRPlus rows.
_BLOCK = (
    "    #Input sequence length: 3\n"
    "#Number of binding residues predicted by RNABindRPlus: 1\n"
    ">3wbm_A_X\n"
    "sequence:\t\t\tM,K,R\n"
    "Prediction from HomPRIP:\t\t?,?,?\n"
    "Predicted score from HomPRIP:\t?,?,?\n"
    "Prediction from SVM:\t\t\t0,1,0\n"
    "Predicted score from SVM:\t\t0.10,0.80,0.20\n"
    "Prediction from RNABindRPlus:\t\t0,1,0\n"
    "Predicted score from RNABindRPlus:\t0.05,0.70,0.10\n"
)


class TestParseRealFormat(unittest.TestCase):
    def test_values_after_colon(self):
        self.assertEqual(
            pr._values_after_colon("Predicted score from RNABindRPlus:\t0.05,0.70,0.10"),
            ["0.05", "0.70", "0.10"])

    def test_parse_block_uses_rnabindrplus_rows(self):
        rows = pr.parse_block(_BLOCK.splitlines())
        self.assertEqual(rows, [(1, 0.05, 0), (2, 0.70, 1), (3, 0.10, 0)])

    def test_parse_block_empty_when_no_rnabindrplus(self):
        # only HomPRIP placeholders (the 8jft_C_B case) -> no rows
        block = (">x_A_B\nsequence:\tM,K\n"
                 "Prediction from HomPRIP:\t?,?\n"
                 "Predicted score from HomPRIP:\t?,?\n")
        self.assertEqual(pr.parse_block(block.splitlines()[1:]), [])

    def test_parse_combined_file_splits_blocks(self):
        text = _BLOCK + (">7p0v_A_B\n"
                         "Prediction from RNABindRPlus:\t1,0\n"
                         "Predicted score from RNABindRPlus:\t0.9,0.1\n")
        combined = pr.parse_combined_file(text)
        self.assertEqual(set(combined), {"3wbm_A_X", "7p0v_A_B"})
        self.assertEqual(combined["3wbm_A_X"], [(1, 0.05, 0), (2, 0.70, 1), (3, 0.10, 0)])
        self.assertEqual(combined["7p0v_A_B"], [(1, 0.9, 1), (2, 0.1, 0)])

    def test_score_only_block(self):
        # binary row absent -> binary None, prob still parsed
        block = ("Predicted score from RNABindRPlus:\t0.9,0.2\n")
        self.assertEqual(pr.parse_block(block.splitlines()),
                         [(1, 0.9, None), (2, 0.2, None)])


# ----------------------------------------------------------------------------
# parser: remap + build + merge
# ----------------------------------------------------------------------------
class TestBuildAndRemap(unittest.TestCase):
    def test_remap_submitted_to_original(self):
        # submitted residues 1,2,3 -> original positions 3,4,5
        kept = [3, 4, 5]
        rows = [(1, 0.1, 0), (2, 0.8, 1), (3, 0.6, 1)]
        pred = pr.build_prediction("1abc_A_E", rows, kept, threshold=0.5)
        self.assertEqual(pred.tool_id, "rnabindrplus")
        self.assertEqual(pred.category, "C")
        self.assertEqual(pred.per_residue_confidence, {3: 0.1, 4: 0.8, 5: 0.6})
        self.assertEqual(pred.binding_protein_residues, [4, 5])  # binary==1

    def test_remap_threshold_when_no_binary(self):
        pred = pr.build_prediction(
            "x_A_E", [(1, 0.9, None), (2, 0.2, None)], [10, 11], threshold=0.5)
        self.assertEqual(pred.binding_protein_residues, [10])  # 0.9 > 0.5

    def test_identity_when_no_map(self):
        pred = pr.build_prediction(
            "x_A_E", [(1, 0.9, 1), (2, 0.1, 0)], None, threshold=0.5)
        self.assertEqual(pred.per_residue_confidence, {1: 0.9, 2: 0.1})
        self.assertEqual(pred.binding_protein_residues, [1])

    def test_rows_beyond_submitted_len_ignored(self):
        pred = pr.build_prediction(
            "x_A_E", [(1, 0.9, 1), (5, 0.9, 1)], [10, 11], threshold=0.5)
        self.assertEqual(pred.per_residue_confidence, {10: 0.9})  # pos 5 dropped

    def test_kept_positions_for_falls_back_to_sample_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples = Path(tmp) / "samples"
            write_sample(samples, "1abc_A_E", "--MKR")
            kept = pr.kept_positions_for("1abc_A_E", {}, samples)
            self.assertEqual(kept, [3, 4, 5])

    def test_merge_preserves_other_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            step4 = Path(tmp)
            sid = "1abc_A_E"
            (step4 / f"{sid}.jsonl").write_text(json.dumps({
                "sample_id": sid, "tools_run": ["graphbind", "rnabindrplus"],
                "predictions": [
                    {"tool_id": "graphbind", "category": "C", "sample_id": sid,
                     "success": True, "binding_protein_residues": [3]},
                    {"tool_id": "rnabindrplus", "category": "C", "sample_id": sid,
                     "success": True, "per_residue_confidence": {1: 0.1}},
                ]}) + "\n", encoding="utf-8")
            new = pr.build_prediction(sid, [(1, 0.9, 1)], [7], threshold=0.5)
            pr.merge_into_set(step4, sid, new)
            rec = json.loads((step4 / f"{sid}.jsonl").read_text(encoding="utf-8"))
            self.assertEqual({p["tool_id"] for p in rec["predictions"]},
                             {"graphbind", "rnabindrplus"})
            rb = next(p for p in rec["predictions"] if p["tool_id"] == "rnabindrplus")
            self.assertEqual(rb["binding_protein_residues"], [7])

    def test_combined_file_to_prediction_end_to_end(self):
        # parse a combined batch file -> rows -> remap -> prediction
        text = (">1abc_A_E\n"
                "Prediction from RNABindRPlus:\t0,1,1\n"
                "Predicted score from RNABindRPlus:\t0.1,0.8,0.6\n")
        combined = pr.parse_combined_file(text)
        rows = combined["1abc_A_E"]
        pred = pr.build_prediction("1abc_A_E", rows, [3, 4, 5], threshold=0.5)
        self.assertEqual(pred.per_residue_confidence, {3: 0.1, 4: 0.8, 5: 0.6})
        self.assertEqual(pred.binding_protein_residues, [4, 5])


if __name__ == "__main__":
    unittest.main()
