"""Mock tests for scripts/riboseer/submit_bindup_batch.py (no network)."""
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.riboseer import submit_bindup_batch as sb  # noqa: E402


class TestReadIds(unittest.TestCase):
    def test_reads_and_skips_blanks_comments(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ids.txt"
            p.write_text("2czj\n# comment\n\n3wbm extra\n", encoding="utf-8")
            self.assertEqual(sb.read_ids(p), ["2czj", "3wbm"])

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            sb.read_ids(Path("/no/such/ids.txt"))


class TestChunk(unittest.TestCase):
    def test_single_job_when_size_zero(self):
        self.assertEqual(sb.chunk(["a", "b", "c"], 0), [["a", "b", "c"]])

    def test_splits(self):
        self.assertEqual(sb.chunk(["a", "b", "c", "d", "e"], 2),
                         [["a", "b"], ["c", "d"], ["e"]])


class TestBuildFields(unittest.TestCase):
    def test_text_mode_and_list_joined(self):
        f = sb.build_fields(["2czj", "3wbm"], 3, True, False,
                            "x@y.com", "job1")
        self.assertEqual(f["input_method"], "text")
        self.assertEqual(f["list"], "2czj\n3wbm")
        self.assertEqual(f["patch_num"], "3")
        self.assertEqual(f["is_pos_patch"], "yes")
        self.assertNotIn("is_neg_patch", f)
        self.assertEqual(f["is_email"], "yes")
        self.assertEqual(f["email"], "x@y.com")
        self.assertEqual(f["job_name"], "job1")

    def test_neg_and_no_email(self):
        f = sb.build_fields(["2czj"], 1, False, True, None, None)
        self.assertNotIn("is_pos_patch", f)
        self.assertEqual(f["is_neg_patch"], "yes")
        self.assertNotIn("is_email", f)
        self.assertNotIn("email", f)
        self.assertNotIn("job_name", f)


class TestDryRun(unittest.TestCase):
    def test_dry_run_no_network(self):
        with tempfile.TemporaryDirectory() as td:
            ids = Path(td) / "ids.txt"
            ids.write_text("2czj\n3wbm\n4abc\n", encoding="utf-8")
            rc = sb.main(["--ids", str(ids), "--email", "x@y.com",
                          "--job-name", "t", "--dry-run", "--insecure"])
            self.assertEqual(rc, 0)
            # dry-run writes no response files
            self.assertEqual(
                list(Path(td).glob("*response*")), [])

    def test_submit_requires_ids(self):
        rc = sb.main(["--email", "x@y.com"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
