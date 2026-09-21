"""Mock tests for bindup_parse.py.

Patch-list parsing, residue-token parsing, patch scoring (1.0/0.67/0.33 with
max-on-overlap), binding=Patch1, the (pdb,chain)->sample_id mapping (incl. one
result shared by two samples), and the step4 merge are covered with synthetic
fixtures. The author->label_seq gemmi map is tested both via a mocked map (logic)
and, where gemmi is available, against a tiny hand-built PDB.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import step4_tool_adapters.external.bindup_parse as pb  # noqa: E402

try:
    import gemmi  # noqa: F401
    _HAVE_GEMMI = True
except ImportError:
    _HAVE_GEMMI = False


_PATCH_TXT = """\
============
PDB ID: 2czj
Chain E
This chain is classified as NA-binding (73% confidence)
Largest Positive Patches:
Patch 1: ALA2 PRO3 VAL4
Patch 2: TYR14 GLU18
Patch 3: GLU30 ILE60 VAL4
"""


# ----------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------
class TestParsing(unittest.TestCase):
    def test_residue_tokens(self):
        self.assertEqual(pb.parse_residue_tokens("ALA2 PRO3 VAL4"),
                         [(2, ""), (3, ""), (4, "")])

    def test_residue_tokens_icode_and_dedup(self):
        self.assertEqual(pb.parse_residue_tokens("HIS12A HIS12A GLU30"),
                         [(12, "A"), (30, "")])

    def test_parse_blocks(self):
        blocks = pb.parse_patch_list_text(_PATCH_TXT)
        self.assertEqual(len(blocks), 1)
        b = blocks[0]
        self.assertEqual(b["pdb"], "2czj")
        self.assertEqual(b["chain"], "E")
        self.assertEqual(len(b["patches"]), 3)
        self.assertEqual(b["patches"][0], [(2, ""), (3, ""), (4, "")])

    def test_parse_multi_chain_file(self):
        txt = ("PDB ID: 1abc\nChain A\nPatch 1: ALA1 GLY2\n"
               "PDB ID: 1abc\nChain B\nPatch 1: SER5\n")
        blocks = pb.parse_patch_list_text(txt)
        self.assertEqual({(b["pdb"], b["chain"]) for b in blocks},
                         {("1abc", "A"), ("1abc", "B")})

    def test_build_index_and_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp)
            (rd / "2czjE_patch_list.txt").write_text(_PATCH_TXT, encoding="utf-8")
            idx = pb.build_bindup_index(rd)
            self.assertIn(("2czj", "E"), idx)
            # chain-case tolerance
            self.assertIsNotNone(pb.lookup_patches(idx, "2CZJ", "e"))
            self.assertIsNone(pb.lookup_patches(idx, "9zzz", "A"))


# ----------------------------------------------------------------------------
# author -> label mapping
# ----------------------------------------------------------------------------
class TestMapToken(unittest.TestCase):
    def test_map_token_icode_and_bare(self):
        amap = {(2, ""): 1, 2: 1, (5, "A"): 4, 5: 3}
        self.assertEqual(pb.map_token(amap, 2, ""), 1)
        self.assertEqual(pb.map_token(amap, 5, "A"), 4)  # icode-specific
        self.assertEqual(pb.map_token(amap, 5, ""), 3)   # bare fallback
        self.assertIsNone(pb.map_token(amap, 99, ""))

    @unittest.skipUnless(_HAVE_GEMMI, "gemmi not installed")
    def test_build_auth_to_label_real_pdb(self):
        # chain B with author numbering starting at 104 -> label_seq 1,2,3
        lines = []
        for i, auth in enumerate((104, 105, 106), start=1):
            lines.append(
                "ATOM  {:>5} CA   ALA B{:>4}      0.000   0.000   0.000  "
                "1.00  0.00           C".format(i, auth))
        with tempfile.TemporaryDirectory() as tmp:
            pdb = Path(tmp) / "x.pdb"
            pdb.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
            amap = pb.build_auth_to_label(pdb, "B")
            self.assertEqual(amap.get(104), 1)
            self.assertEqual(amap.get(105), 2)
            self.assertEqual(amap.get(106), 3)


# ----------------------------------------------------------------------------
# build_prediction: scoring, overlap-max, binding=Patch1
# ----------------------------------------------------------------------------
class TestBuildPrediction(unittest.TestCase):
    def test_scoring_and_binding(self):
        # patch1 {1,2,3}, patch2 {14,18}, patch3 {30,60,3}; residue 3 in p1&p3
        pred = pb.build_prediction(
            "1abc_A_E", [[1, 2, 3], [14, 18], [30, 60, 3]],
            patch_scores=[1.0, 0.67, 0.33])
        self.assertEqual(pred.tool_id, "bindup")
        self.assertEqual(pred.category, "C")
        pr = pred.per_residue_confidence
        self.assertEqual(pr[1], 1.0)
        self.assertEqual(pr[3], 1.0)        # max(1.0 from p1, 0.33 from p3)
        self.assertEqual(pr[14], 0.67)
        self.assertEqual(pr[30], 0.33)
        self.assertEqual(pred.binding_protein_residues, [1, 2, 3])  # Patch 1

    def test_patches_beyond_scores_ignored(self):
        pred = pb.build_prediction(
            "x_A_E", [[1], [2], [3], [4]], patch_scores=[1.0, 0.67, 0.33])
        self.assertNotIn(4, pred.per_residue_confidence)  # 4th patch -> 0

    def test_no_patches_valid_empty(self):
        pred = pb.build_prediction("x_A_E", [], patch_scores=[1.0])
        self.assertEqual(pred.binding_protein_residues, [])
        self.assertIsNone(pred.per_residue_confidence)
        self.assertTrue(pred.success)


# ----------------------------------------------------------------------------
# merge + end-to-end parse_one_sample (mapping mocked)
# ----------------------------------------------------------------------------
class TestMergeAndParse(unittest.TestCase):
    def _write_sample(self, samples_dir: Path, sid: str, pdb: str, chain: str):
        samples_dir.mkdir(parents=True, exist_ok=True)
        (samples_dir / f"{sid}.json").write_text(json.dumps({
            "sample_id": sid, "source_pdb": pdb,
            "protein": {"chain_id": chain, "length": 200}}), encoding="utf-8")

    def test_merge_preserves_other_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            step4 = Path(tmp)
            sid = "2czj_E_F"
            (step4 / f"{sid}.jsonl").write_text(json.dumps({
                "sample_id": sid, "tools_run": ["graphbind", "bindup"],
                "predictions": [
                    {"tool_id": "graphbind", "category": "C", "sample_id": sid,
                     "success": True, "binding_protein_residues": [3]},
                    {"tool_id": "bindup", "category": "C", "sample_id": sid,
                     "success": True, "per_residue_confidence": {1: 0.1}},
                ]}) + "\n", encoding="utf-8")
            new = pb.build_prediction(sid, [[5, 6]], patch_scores=[1.0])
            pb.merge_into_set(step4, sid, new)
            rec = json.loads((step4 / f"{sid}.jsonl").read_text(encoding="utf-8"))
            self.assertEqual({p["tool_id"] for p in rec["predictions"]},
                             {"graphbind", "bindup"})
            bu = next(p for p in rec["predictions"] if p["tool_id"] == "bindup")
            self.assertEqual(bu["binding_protein_residues"], [5, 6])

    def test_parse_one_sample_maps_author_to_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            samples = tmp / "samples"
            self._write_sample(samples, "2czj_E_F", "2czj", "E")
            idx = pb.build_bindup_index(_results_with(tmp, _PATCH_TXT, "2czjE"))
            # author->label: shift by -1 (author 2->1, etc.)
            amap = {n: n - 1 for n in range(1, 100)}
            amap.update({(n, ""): n - 1 for n in range(1, 100)})
            with mock.patch.object(pb, "find_raw_structure",
                                   return_value=tmp / "fake.pdb"), \
                 mock.patch.object(pb, "build_auth_to_label", return_value=amap):
                pred = pb.parse_one_sample(
                    "2czj_E_F", bindup_index=idx, samples_dir=samples,
                    raw_dir=tmp, patch_scores=[1.0, 0.67, 0.33], amap_cache={})
            # Patch1 ALA2 PRO3 VAL4 -> labels 1,2,3
            self.assertEqual(pred.binding_protein_residues, [1, 2, 3])
            # VAL4 (label 3) in p1(1.0) and p3(0.33) -> 1.0
            self.assertEqual(pred.per_residue_confidence[3], 1.0)
            self.assertEqual(pred.per_residue_confidence[13], 0.67)  # TYR14->13

    def test_shared_result_two_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            samples = tmp / "samples"
            # same protein chain E, different RNA partners -> two samples
            self._write_sample(samples, "2czj_E_F", "2czj", "E")
            self._write_sample(samples, "2czj_E_G", "2czj", "E")
            idx = pb.build_bindup_index(_results_with(tmp, _PATCH_TXT, "2czjE"))
            amap = {n: n for n in range(1, 100)}
            amap.update({(n, ""): n for n in range(1, 100)})
            cache: dict = {}
            with mock.patch.object(pb, "find_raw_structure",
                                   return_value=tmp / "fake.pdb"), \
                 mock.patch.object(pb, "build_auth_to_label", return_value=amap):
                p1 = pb.parse_one_sample(
                    "2czj_E_F", bindup_index=idx, samples_dir=samples,
                    raw_dir=tmp, patch_scores=[1.0, 0.67, 0.33], amap_cache=cache)
                p2 = pb.parse_one_sample(
                    "2czj_E_G", bindup_index=idx, samples_dir=samples,
                    raw_dir=tmp, patch_scores=[1.0, 0.67, 0.33], amap_cache=cache)
            self.assertEqual(p1.binding_protein_residues, p2.binding_protein_residues)
            self.assertEqual(p1.per_residue_confidence, p2.per_residue_confidence)
            self.assertEqual(len(cache), 1)  # author->label built once, cached

    def test_parse_one_sample_missing_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            samples = tmp / "samples"
            self._write_sample(samples, "9zzz_A_B", "9zzz", "A")
            with self.assertRaises(FileNotFoundError):
                pb.parse_one_sample(
                    "9zzz_A_B", bindup_index={}, samples_dir=samples,
                    raw_dir=tmp, patch_scores=[1.0], amap_cache={})

    def test_all_unmapped_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            samples = tmp / "samples"
            self._write_sample(samples, "2czj_E_F", "2czj", "E")
            idx = pb.build_bindup_index(_results_with(tmp, _PATCH_TXT, "2czjE"))
            with mock.patch.object(pb, "find_raw_structure",
                                   return_value=tmp / "fake.pdb"), \
                 mock.patch.object(pb, "build_auth_to_label", return_value={}):
                with self.assertRaises(ValueError):
                    pb.parse_one_sample(
                        "2czj_E_F", bindup_index=idx, samples_dir=samples,
                        raw_dir=tmp, patch_scores=[1.0], amap_cache={})


def _results_with(tmp: Path, text: str, stem: str) -> Path:
    rd = tmp / "results"
    rd.mkdir(exist_ok=True)
    (rd / f"{stem}_patch_list.txt").write_text(text, encoding="utf-8")
    return rd


if __name__ == "__main__":
    unittest.main()
