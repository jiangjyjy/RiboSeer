"""Mock tests for submit_bindup_single.py pure helpers (no network)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.riboseer.submit_bindup_single as sb  # noqa: E402


class TestHelpers(unittest.TestCase):
    def test_build_fields_pos_patches_no_email(self):
        f = sb.build_fields("A", 3, None)
        self.assertEqual(f["model_type"], "experimental")
        self.assertEqual(f["input_type"], "file")
        self.assertEqual(f["chain_type"], "selected_chain")
        self.assertEqual(f["chain_id"], "A")
        self.assertEqual(f["is_pos_patch"], "yes")
        self.assertEqual(f["patch_num"], "3")
        self.assertNotIn("is_neg_patch", f)   # negative patches off
        self.assertNotIn("email", f)

    def test_build_fields_with_email(self):
        f = sb.build_fields("A", 2, "me@x.com")
        self.assertEqual(f["is_email"], "yes")
        self.assertEqual(f["email"], "me@x.com")

    def test_job_urls_from_redirect(self):
        u = sb.job_urls("https://bindup.technion.ac.il/1780401572/results.html",
                        "8k22", "8k22_C_P")
        self.assertEqual(u["base"], "https://bindup.technion.ac.il/1780401572")
        self.assertEqual(u["patch_list"],
                         "https://bindup.technion.ac.il/1780401572/8k22_patch_list.txt")
        self.assertEqual(u["zip"],
                         "https://bindup.technion.ac.il/1780401572/"
                         "8k22_C_P_BindUP-Alpha_Results.zip")

    def test_rewrite_header_pdb_file_and_chain(self):
        # single-mode header: "PDB file: 8k22_C_P" + "Chain A"
        raw = ("==================\n"
               "PDB file: 8k22_C_P\n"
               "==================\n\n"
               "Chain A\n"
               "=======\n\n"
               "Patch 1: ASN16 LYS18\n")
        out = sb.rewrite_header(raw, "8k22", "C")
        self.assertIn("PDB ID: 8k22", out)
        self.assertNotIn("PDB file:", out)
        self.assertIn("\nChain C\n", out)
        self.assertNotIn("Chain A\n", out)
        # patch lines untouched
        self.assertIn("Patch 1: ASN16 LYS18", out)

    def test_rewrite_header_idempotent_on_batch_style(self):
        raw = "PDB ID: 2czj\nChain E\nPatch 1: ALA2\n"
        out = sb.rewrite_header(raw, "2czj", "E")
        self.assertIn("PDB ID: 2czj", out)
        self.assertIn("Chain E", out)

    def test_looks_ready(self):
        self.assertFalse(sb._looks_ready("BindUP is calculating ..."))
        self.assertTrue(sb._looks_ready("Largest Positive Patches:\nPatch 1: ALA2"))


if __name__ == "__main__":
    unittest.main()
