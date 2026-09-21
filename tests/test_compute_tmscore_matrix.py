"""Mock tests for scripts/compute_tmscore_matrix.py (subprocess mocked)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.compute_tmscore_matrix as cm  # noqa: E402


# Legacy wording, older US-align builds (~2020). Kept so we never
# regress that path.
USALIGN_STDOUT = """ ********************************************************
 * US-align (Version 20231221)                          *
 ********************************************************
Name of Structure_1: a.pdb (size= 200)
Name of Structure_2: b.pdb (size= 180)

Aligned length=170, RMSD=2.10, Seq_ID=0.31
TM-score= 0.71234 (if normalized by length of Chain_1, L=200, d0=5.20)
TM-score= 0.81234 (if normalized by length of Chain_2, L=180, d0=4.90)
"""

# Current wording, what the user's server actually prints. The old
# regex hard-coded "if normalized by length of Chain_1" and silently
# refused to match this - cause of the all-failed smoke run.
USALIGN_STDOUT_NEW = """ ********************************************************
 * US-align                                             *
 ********************************************************
Name of Structure_1: a.pdb (size= 87)
Name of Structure_2: b.pdb (size= 58)

Aligned length=42, RMSD=3.10, Seq_ID=0.21
TM-score= 0.36176 (normalized by length of Structure_1: L=87, d0=3.36)
TM-score= 0.51056 (normalized by length of Structure_2: L=58, d0=2.54)
"""


class TestParser(unittest.TestCase):
    def test_parse_tm12_legacy_chain_wording(self):
        a, b = cm.parse_usalign_output(USALIGN_STDOUT)
        self.assertAlmostEqual(a, 0.71234, places=4)
        self.assertAlmostEqual(b, 0.81234, places=4)

    def test_parse_tm12_current_structure_wording(self):
        """Regression: the regex must accept ``Structure_1`` (no
        ``if`` prefix, colon separator) - that's what the user's
        US-align prints."""
        a, b = cm.parse_usalign_output(USALIGN_STDOUT_NEW)
        self.assertAlmostEqual(a, 0.36176, places=4)
        self.assertAlmostEqual(b, 0.51056, places=4)

    def test_parse_mixed_whitespace_and_case(self):
        text = ("TM-score   =   0.5  (  IF  Normalized  by  length  "
                "of Chain_1 , L=10 )\n"
                "tm-score=0.4 (normalized by length of structure_2: "
                "L=20)\n")
        a, b = cm.parse_usalign_output(text)
        self.assertAlmostEqual(a, 0.5)
        self.assertAlmostEqual(b, 0.4)

    def test_parse_raises_on_garbage(self):
        with self.assertRaises(ValueError) as cm_:
            cm.parse_usalign_output("nope nothing here")
        # error includes a peek at the actual text so the next drift
        # is easy to diagnose
        self.assertIn("nope", str(cm_.exception))


def _write_pdb(path: Path, chains: dict[str, int]) -> None:
    """Toy PDB: ``chains[chain_id] = n_atoms``. Atom lines are valid
    PDB column 22 = chain."""
    lines = []
    serial = 1
    for ch, n in chains.items():
        for k in range(n):
            # Cols: ATOM  serial name   resn chain resi    x   y   z
            lines.append(
                f"ATOM  {serial:5d}  CA  ALA {ch}{k + 1:4d}    "
                f"{k * 1.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00\n")
            serial += 1
    lines.append("END\n")
    path.write_text("".join(lines), encoding="utf-8")


class TestExtractAndStage(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.raw = self.tmp / "raw" / "rna2p_balanced"
        self.raw.mkdir(parents=True)
        _write_pdb(self.raw / "abcd.pdb",
                   {"A": 30, "B": 20, "C": 0})
        self.processed = self.tmp / "processed"
        (self.processed / "samples").mkdir(parents=True)
        for sid, ch in (("abcd_A_X", "A"), ("abcd_B_X", "B")):
            (self.processed / "samples" / f"{sid}.json").write_text(
                json.dumps({"sample_id": sid, "source_pdb": "abcd",
                            "protein": {"chain_id": ch}}),
                encoding="utf-8")

    def test_find_raw_pdb_searches_subdir(self):
        # raw/ root has no file; rna2p_balanced subdir does
        p = cm.find_raw_pdb(self.tmp / "raw", "abcd")
        self.assertEqual(p.name, "abcd.pdb")

    def test_extract_chain_filters_correctly(self):
        dst = self.tmp / "chainA.pdb"
        n = cm.extract_chain_to_pdb(self.raw / "abcd.pdb", "A", dst)
        self.assertEqual(n, 30)
        # all written ATOM lines must have chain id A in col 22
        for line in dst.read_text(encoding="utf-8").splitlines():
            if line.startswith("ATOM"):
                self.assertEqual(line[21], "A")

    def test_extract_chain_missing_raises(self):
        with self.assertRaises(ValueError):
            cm.extract_chain_to_pdb(self.raw / "abcd.pdb", "Z",
                                    self.tmp / "z.pdb")

    def test_stage_chains_caches(self):
        cache = self.tmp / "cache"
        sids = ["abcd_A_X", "abcd_B_X"]
        out = cm._stage_chains(sids, self.processed,
                               self.tmp / "raw", cache)
        self.assertEqual(set(out), set(sids))
        self.assertTrue((cache / "abcd_A_X.pdb").is_file())
        # re-run should be a no-op (mtime preserved)
        t0 = (cache / "abcd_A_X.pdb").stat().st_mtime
        out2 = cm._stage_chains(sids, self.processed,
                                self.tmp / "raw", cache)
        self.assertEqual(set(out2), set(sids))
        self.assertEqual(
            (cache / "abcd_A_X.pdb").stat().st_mtime, t0)


def _write_cif(path: Path, chains: dict[str, int]) -> None:
    """Toy mmCIF: one row per atom, ``label_asym_id`` and
    ``auth_asym_id`` both set to the chain key (gemmi keys ``chain.name``
    off of ``auth_asym_id``)."""
    rows = []
    serial = 1
    for ch, n in chains.items():
        for k in range(n):
            rows.append(
                f"ATOM {serial} C CA . ALA A {k + 1} ? "
                f"{k * 1.0:.3f} 0.000 0.000 1.00 20.00 {ch} {ch} 1")
            serial += 1
    body = "\n".join(rows)
    cif = ("data_test\n"
           "loop_\n"
           "_atom_site.group_PDB\n"
           "_atom_site.id\n"
           "_atom_site.type_symbol\n"
           "_atom_site.label_atom_id\n"
           "_atom_site.label_alt_id\n"
           "_atom_site.label_comp_id\n"
           "_atom_site.label_asym_id\n"
           "_atom_site.label_seq_id\n"
           "_atom_site.pdbx_PDB_ins_code\n"
           "_atom_site.Cartn_x\n"
           "_atom_site.Cartn_y\n"
           "_atom_site.Cartn_z\n"
           "_atom_site.occupancy\n"
           "_atom_site.B_iso_or_equiv\n"
           "_atom_site.auth_asym_id\n"
           "_atom_site.label_asym_id_real\n"
           "_atom_site.pdbx_PDB_model_num\n"
           f"{body}\n")
    path.write_text(cif, encoding="utf-8")


class TestCifFallback(unittest.TestCase):
    """The fix's heart: multi-char chain ids (e.g. 'Lm', '4C', 'U2')
    can't be expressed in PDB column 22, so we must read them from the
    mmCIF via gemmi and rename to single-char 'A' before writing."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.raw = self.tmp / "raw"
        self.raw.mkdir()
        # CIF carries a multi-char auth_asym_id 'Lm'; the PDB file
        # exists but has only the single-char chain 'A' (the multi-char
        # one was lost in PDB conversion - reality for these samples).
        _write_pdb(self.raw / "abcd.pdb", {"A": 10})
        _write_cif(self.raw / "abcd.cif", {"Lm": 8, "A": 10})
        self.processed = self.tmp / "processed"
        (self.processed / "samples").mkdir(parents=True)
        for sid, ch in (("abcd_Lm_X", "Lm"),    # multi-char -> CIF
                        ("abcd_A_X", "A"),       # single-char -> PDB
                        ("abcd_Zz_X", "Zz")):    # missing entirely
            (self.processed / "samples" / f"{sid}.json").write_text(
                json.dumps({"sample_id": sid, "source_pdb": "abcd",
                            "protein": {"chain_id": ch}}),
                encoding="utf-8")

    def test_find_raw_cif(self):
        self.assertEqual(
            cm.find_raw_cif(self.raw, "abcd").name, "abcd.cif")
        self.assertIsNone(cm.find_raw_cif(self.raw, "missing"))

    def test_extract_chain_from_cif_renames_to_A(self):
        dst = self.tmp / "Lm.pdb"
        n = cm.extract_chain_from_cif(
            self.raw / "abcd.cif", "Lm", dst)
        self.assertGreater(n, 0)
        # Every ATOM line in the output must carry single-char 'A' at
        # column 22 - that's the whole point of the rename.
        lines = [ln for ln in dst.read_text("utf-8").splitlines()
                 if ln.startswith("ATOM")]
        self.assertTrue(lines, "no ATOM lines in CIF-derived PDB")
        for ln in lines:
            self.assertEqual(ln[21], "A", ln)

    def test_extract_chain_from_cif_missing_raises(self):
        with self.assertRaises(ValueError):
            cm.extract_chain_from_cif(
                self.raw / "abcd.cif", "Zz",
                self.tmp / "z.pdb")

    def test_stage_one_chain_uses_pdb_for_single_char(self):
        dst = self.tmp / "a.pdb"
        kind = cm.stage_one_chain(
            "abcd_A_X", self.processed, self.raw, dst)
        self.assertEqual(kind, "pdb")
        # Original chain id 'A' preserved (no rename on PDB path)
        for ln in dst.read_text("utf-8").splitlines():
            if ln.startswith("ATOM"):
                self.assertEqual(ln[21], "A")

    def test_stage_one_chain_falls_back_to_cif_for_multichar(self):
        dst = self.tmp / "lm.pdb"
        kind = cm.stage_one_chain(
            "abcd_Lm_X", self.processed, self.raw, dst)
        self.assertEqual(kind, "cif")
        # Chain renamed to 'A' so US-align is happy
        atom_lines = [ln for ln in dst.read_text("utf-8").splitlines()
                      if ln.startswith("ATOM")]
        self.assertTrue(atom_lines)
        for ln in atom_lines:
            self.assertEqual(ln[21], "A")

    def test_stage_one_chain_skips_if_neither_has_chain(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            cm.stage_one_chain(
                "abcd_Zz_X", self.processed, self.raw,
                self.tmp / "zz.pdb")

    def test_stage_one_chain_cif_only_structure(self):
        # No PDB at all -> must go via CIF
        raw2 = self.tmp / "raw_cif_only"
        raw2.mkdir()
        _write_cif(raw2 / "wxyz.cif", {"Lm": 6})
        (self.processed / "samples" / "wxyz_Lm_X.json").write_text(
            json.dumps({"sample_id": "wxyz_Lm_X",
                        "source_pdb": "wxyz",
                        "protein": {"chain_id": "Lm"}}),
            encoding="utf-8")
        dst = self.tmp / "wxyz.pdb"
        kind = cm.stage_one_chain(
            "wxyz_Lm_X", self.processed, raw2, dst)
        self.assertEqual(kind, "cif")

    def test_stage_chains_reports_pdb_and_cif_counts(self):
        # 1 single-char -> PDB; 1 multi-char -> CIF; 1 missing -> err
        out = cm._stage_chains(
            ["abcd_A_X", "abcd_Lm_X", "abcd_Zz_X"],
            self.processed, self.raw, self.tmp / "cache")
        self.assertEqual(set(out), {"abcd_A_X", "abcd_Lm_X"})
        self.assertNotIn("abcd_Zz_X", out)


class TestAtomicAndCache(unittest.TestCase):
    """The smoking-gun bug: extract_chain_to_pdb used to leave an
    ``END\\n``-only stub when the chain wasn't found, the old cache
    check accepted it, and US-align then silently failed on every pair
    that touched the affected sid. These tests pin down the fix."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.raw = self.tmp / "raw"
        self.raw.mkdir()
        _write_pdb(self.raw / "abcd.pdb", {"A": 12, "B": 8})

    def test_extract_chain_to_pdb_leaves_no_stub_on_missing_chain(self):
        dst = self.tmp / "missing.pdb"
        with self.assertRaises(ValueError):
            cm.extract_chain_to_pdb(
                self.raw / "abcd.pdb", "Z", dst)
        # Critical: nothing left on disk for the cache to mistakenly
        # reuse later.
        self.assertFalse(dst.exists(),
                         f"stub leaked at {dst}")
        # And no .tmp sibling either
        self.assertFalse(dst.with_name(dst.name + ".tmp").exists())

    def test_extract_chain_to_pdb_success_replaces_existing(self):
        dst = self.tmp / "out.pdb"
        # Pre-populate with junk to confirm atomic replace clobbers it
        dst.write_text("PREVIOUS CONTENT\n", encoding="utf-8")
        n = cm.extract_chain_to_pdb(
            self.raw / "abcd.pdb", "A", dst)
        self.assertEqual(n, 12)
        self.assertNotIn("PREVIOUS",
                         dst.read_text(encoding="utf-8"))

    def test_cache_valid_rejects_end_only_stub(self):
        stub = self.tmp / "stub.pdb"
        stub.write_text("END\n", encoding="utf-8")
        self.assertFalse(cm._cache_valid(stub))

    def test_cache_valid_accepts_real_chain(self):
        good = self.tmp / "good.pdb"
        cm.extract_chain_to_pdb(self.raw / "abcd.pdb", "A", good)
        self.assertTrue(cm._cache_valid(good))

    def test_cache_valid_false_when_missing(self):
        self.assertFalse(cm._cache_valid(self.tmp / "nope.pdb"))

    def test_stage_chains_evicts_legacy_stub(self):
        # Pre-seed the cache with a stub (simulates the buggy run that
        # the user just hit on the server).
        processed = self.tmp / "processed"
        (processed / "samples").mkdir(parents=True)
        (processed / "samples" / "abcd_A_X.json").write_text(
            json.dumps({"sample_id": "abcd_A_X",
                        "source_pdb": "abcd",
                        "protein": {"chain_id": "A"}}),
            encoding="utf-8")
        cache = self.tmp / "chains"
        cache.mkdir()
        stub = cache / "abcd_A_X.pdb"
        stub.write_text("END\n", encoding="utf-8")  # 5-byte garbage

        out = cm._stage_chains(["abcd_A_X"], processed,
                               self.raw, cache)
        self.assertIn("abcd_A_X", out)
        # The stub must be replaced with a real PDB carrying ATOM lines
        # so US-align can read it.
        self.assertTrue(cm._cache_valid(stub),
                        "stub should have been re-extracted")
        with stub.open(encoding="utf-8") as fh:
            self.assertTrue(any(ln.startswith("ATOM") for ln in fh))

    def test_force_restage_wipes_valid_cache_too(self):
        processed = self.tmp / "processed"
        (processed / "samples").mkdir(parents=True)
        (processed / "samples" / "abcd_A_X.json").write_text(
            json.dumps({"sample_id": "abcd_A_X",
                        "source_pdb": "abcd",
                        "protein": {"chain_id": "A"}}),
            encoding="utf-8")
        cache = self.tmp / "chains"
        cache.mkdir()
        # Drop a real (valid) PDB into cache, then mark its mtime
        cm.extract_chain_to_pdb(
            self.raw / "abcd.pdb", "A", cache / "abcd_A_X.pdb")
        t0 = (cache / "abcd_A_X.pdb").stat().st_mtime_ns
        # Sleep-less: force_restage must rewrite regardless of mtime
        out = cm._stage_chains(["abcd_A_X"], processed,
                               self.raw, cache, force_restage=True)
        self.assertIn("abcd_A_X", out)
        t1 = (cache / "abcd_A_X.pdb").stat().st_mtime_ns
        # mtime changed => the file was indeed re-extracted (Windows
        # FS resolution can collapse equal mtimes, so on a tie we just
        # confirm the file is still valid)
        self.assertTrue(cm._cache_valid(cache / "abcd_A_X.pdb"))
        self.assertGreaterEqual(t1, t0)


class TestExtractCifAtomic(unittest.TestCase):
    """The CIF extractor must also clean up after itself."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        _write_cif(self.tmp / "abcd.cif", {"Lm": 5})

    def test_no_stub_when_chain_missing(self):
        dst = self.tmp / "missing.pdb"
        with self.assertRaises(ValueError):
            cm.extract_chain_from_cif(
                self.tmp / "abcd.cif", "Zz", dst)
        self.assertFalse(dst.exists())
        self.assertFalse(dst.with_name(dst.name + ".tmp").exists())


class TestRunUsalignMocked(unittest.TestCase):
    def test_success(self):
        cp = subprocess.CompletedProcess(
            args=["x"], returncode=0,
            stdout=USALIGN_STDOUT, stderr="")
        with mock.patch.object(cm.subprocess, "run",
                               return_value=cp) as m:
            score = cm.run_usalign("USalign", Path("a"), Path("b"))
        self.assertAlmostEqual(score, 0.81234, places=4)
        # Confirm we called with -ter 0 as the spec says
        args = m.call_args[0][0]
        self.assertIn("-ter", args)
        self.assertEqual(args[args.index("-ter") + 1], "0")

    def test_success_new_format(self):
        """The server's actual stdout format must score correctly."""
        cp = subprocess.CompletedProcess(
            args=["x"], returncode=0,
            stdout=USALIGN_STDOUT_NEW, stderr="")
        with mock.patch.object(cm.subprocess, "run", return_value=cp):
            score = cm.run_usalign("USalign", Path("a"), Path("b"))
        self.assertAlmostEqual(score, 0.51056, places=4)

    def test_verbose_prints_cmd_and_stdout(self):
        cp = subprocess.CompletedProcess(
            args=["x"], returncode=0,
            stdout=USALIGN_STDOUT_NEW, stderr="hi")
        import io
        buf = io.StringIO()
        with mock.patch.object(cm.subprocess, "run", return_value=cp), \
                mock.patch.object(cm.sys, "stderr", buf):
            cm.run_usalign("USalign", Path("a"), Path("b"),
                           verbose=True)
        s = buf.getvalue()
        self.assertIn("[verbose] cmd:", s)
        self.assertIn("-ter", s)
        self.assertIn("rc=0", s)
        # stdout snippet present so format drift is visible
        self.assertIn("Structure_1", s)

    def test_nonzero_exit_raises(self):
        cp = subprocess.CompletedProcess(
            args=["x"], returncode=2, stdout="", stderr="boom")
        with mock.patch.object(cm.subprocess, "run", return_value=cp):
            with self.assertRaises(RuntimeError):
                cm.run_usalign("USalign", Path("a"), Path("b"))


class TestWorker(unittest.TestCase):
    """The worker tuple grew a trailing ``verbose`` flag; make sure
    main() builds it correctly and the flag actually flows through."""

    def test_worker_propagates_verbose(self):
        seen = {}

        def fake_run_usalign(bin_path, a, b, timeout, verbose=False):
            seen["verbose"] = verbose
            return 0.5

        with mock.patch.object(cm, "run_usalign", fake_run_usalign):
            r = cm._worker((0, 1, Path("a"), Path("b"),
                            "USalign", 30.0, True))
        self.assertEqual(r[:3], (0, 1, 0.5))
        self.assertIsNone(r[3])
        self.assertTrue(seen["verbose"])

    def test_worker_returns_error_on_exception(self):
        with mock.patch.object(cm, "run_usalign",
                               side_effect=RuntimeError("boom")):
            i, j, score, err = cm._worker(
                (3, 7, Path("a"), Path("b"), "USalign", 30.0, False))
        self.assertEqual((i, j), (3, 7))
        # NaN signals failure; err string carries the original message
        import math
        self.assertTrue(math.isnan(score))
        self.assertIn("boom", err)


class TestReadSampleList(unittest.TestCase):
    def test_strips_comments_and_tabs(self):
        tmp = Path(tempfile.mkdtemp()) / "ids.txt"
        tmp.write_text("# header\nabc_A_X\tprot_1\n\n"
                       "def_B_Y\tprot_2\n", encoding="utf-8")
        self.assertEqual(cm.read_sample_list(tmp),
                         ["abc_A_X", "def_B_Y"])


class TestUndonePairsAndCheckpoint(unittest.TestCase):
    def test_undone_skips_done_entries(self):
        m = np.full((3, 3), -1.0, dtype=np.float32)
        np.fill_diagonal(m, 1.0)
        m[0, 1] = m[1, 0] = 0.7  # done
        undone = cm._undone_pairs(m)
        self.assertEqual(set(undone), {(0, 2), (1, 2)})

    def test_save_matrix_writes_sidecars(self):
        tmp = Path(tempfile.mkdtemp())
        m = np.eye(2, dtype=np.float32)
        cm._save_matrix(m, tmp / "tm.npy",
                        ["a", "b"], {"n": 2})
        self.assertTrue((tmp / "tm.npy").is_file())
        self.assertEqual(
            (tmp / "tm.ids.txt").read_text("utf-8").split(),
            ["a", "b"])
        meta = json.loads(
            (tmp / "tm.meta.json").read_text("utf-8"))
        self.assertEqual(meta["n"], 2)


class TestCliEndToEnd(unittest.TestCase):
    """Drive main() end-to-end with USalign mocked."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.raw = self.tmp / "raw" / "rna2p_balanced"
        self.raw.mkdir(parents=True)
        _write_pdb(self.raw / "abcd.pdb",
                   {"A": 10, "B": 10, "C": 10})
        self.processed = self.tmp / "processed"
        (self.processed / "samples").mkdir(parents=True)
        for sid, ch in (("abcd_A_X", "A"),
                        ("abcd_B_X", "B"),
                        ("abcd_C_X", "C")):
            (self.processed / "samples"
             / f"{sid}.json").write_text(
                json.dumps({"sample_id": sid,
                            "source_pdb": "abcd",
                            "protein": {"chain_id": ch}}),
                encoding="utf-8")
        self.list_ = self.tmp / "reps.txt"
        self.list_.write_text(
            "abcd_A_X\tp1\nabcd_B_X\tp2\nabcd_C_X\tp3\n",
            encoding="utf-8")
        self.out = self.tmp / "out" / "tm.npy"

    def test_resume_recovers_nan_rows_for_rescued_sids(self):
        """A sid that failed staging last time leaves a NaN row in the
        on-disk matrix. If the cache fix now rescues it (e.g. the CIF
        fallback succeeds where the previous PDB-only path didn't),
        the row must be reset to ``-1`` so pairs get recomputed.
        Without recovery, ``_undone_pairs`` would still treat the row
        as missing and skip every pair touching it."""
        cp = subprocess.CompletedProcess(
            args=["x"], returncode=0,
            stdout=USALIGN_STDOUT, stderr="")
        # Seed a stale matrix: sid index 1 was "missing" last run
        # (all-NaN row). The other two sids have a real off-diagonal
        # entry so we can confirm the recovery only touches row 1.
        n = 3
        mat = np.full((n, n), -1.0, dtype=np.float32)
        np.fill_diagonal(mat, 1.0)
        mat[0, 2] = mat[2, 0] = 0.42  # an already-computed pair
        mat[1, :] = np.nan
        mat[:, 1] = np.nan
        out_dir = self.tmp / "out"
        out_dir.mkdir()
        np.save(out_dir / "tm.npy", mat)

        with mock.patch.object(cm.subprocess, "run", return_value=cp):
            rc = cm.main([
                "--sample-list", str(self.list_),
                "--raw-dir", str(self.tmp / "raw"),
                "--processed-dir", str(self.processed),
                "--output", str(out_dir / "tm.npy"),
                "--n-workers", "1",
                "--usalign-bin", "USalign",
                "--resume",
            ])
        self.assertEqual(rc, 0)
        m = np.load(out_dir / "tm.npy")
        # Row 1 now has real values for both off-diagonals; the
        # already-done (0,2) pair stays at 0.42 (no re-do).
        self.assertTrue(np.isfinite(m[0, 1]))
        self.assertTrue(np.isfinite(m[1, 2]))
        self.assertAlmostEqual(float(m[0, 2]), 0.42, places=3)

    def test_main_uses_n_workers_1_and_writes_matrix(self):
        cp = subprocess.CompletedProcess(
            args=["x"], returncode=0,
            stdout=USALIGN_STDOUT, stderr="")
        with mock.patch.object(cm.subprocess, "run",
                               return_value=cp):
            rc = cm.main([
                "--sample-list", str(self.list_),
                "--raw-dir", str(self.tmp / "raw"),
                "--processed-dir", str(self.processed),
                "--output", str(self.out),
                "--n-workers", "1",
                "--usalign-bin", "USalign",
            ])
        self.assertEqual(rc, 0)
        m = np.load(self.out)
        self.assertEqual(m.shape, (3, 3))
        # diagonal = 1, off-diagonals = max(0.71234, 0.81234) ~ 0.812
        self.assertTrue(np.allclose(np.diag(m), 1.0))
        self.assertAlmostEqual(float(m[0, 1]), 0.81234, places=3)
        self.assertAlmostEqual(float(m[1, 0]), 0.81234, places=3)
        self.assertEqual(
            (self.out.with_suffix(".ids.txt")
             .read_text("utf-8").split()),
            ["abcd_A_X", "abcd_B_X", "abcd_C_X"])


if __name__ == "__main__":
    unittest.main()
