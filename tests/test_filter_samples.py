"""Mock tests for scripts/filter_samples.py."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.filter_samples as fs  # noqa: E402


def _sample(sid, *, rna_len=None, rna_seq=None, split=None):
    rna: dict = {}
    if rna_len is not None:
        rna["length"] = rna_len
    if rna_seq is not None:
        rna["sequence"] = rna_seq
    return {"sample_id": sid, "protein": {"sequence": "MKR"},
            "rna": rna,
            "interaction": {"binding_protein_residues": [1]}}


def _build_input(tmp: Path):
    """6 valid + edge cases. splits.json gives train/val/test/None."""
    sdir = tmp / "in"
    sdir.mkdir(parents=True)
    specs = [
        ("a", 50, "train"), ("b", 80, "train"), ("c", 100, "train"),
        ("d", 90, "val"), ("e", 70, "test"), ("f", 95, "test"),
        ("g", 150, "train"),         # too long → filtered
        ("h", 250, "test"),          # too long → filtered
        ("i", None, "train"),        # no rna length at all
    ]
    samples_meta = {}
    for sid, rlen, sp in specs:
        if sid == "i":
            s = _sample(sid)              # rna {} → no length
        elif sid == "f":
            s = _sample(sid, rna_seq="G" * rlen)  # length via sequence
        else:
            s = _sample(sid, rna_len=rlen)
        (sdir / f"{sid}.json").write_text(json.dumps(s),
                                          encoding="utf-8")
        samples_meta[sid] = {"protein_cluster_id": "p", "split": sp}
    # one sample present on disk but absent from splits.json → unknown
    (sdir / "z.json").write_text(
        json.dumps(_sample("z", rna_len=60)), encoding="utf-8")
    # a malformed json
    (sdir / "bad.json").write_text("{not json", encoding="utf-8")
    splits = tmp / "splits.json"
    splits.write_text(json.dumps({"samples": samples_meta}),
                      encoding="utf-8")
    return sdir, splits


class TestCore(unittest.TestCase):
    def test_rna_length_of(self):
        self.assertEqual(fs.rna_length_of(_sample("x", rna_len=42)), 42)
        self.assertEqual(
            fs.rna_length_of(_sample("x", rna_seq="ACGUACGU")), 8)
        # length wins over sequence when both present + valid
        self.assertEqual(fs.rna_length_of(
            {"rna": {"length": 5, "sequence": "ACGUACGU"}}), 5)
        # invalid length falls back to sequence
        self.assertEqual(fs.rna_length_of(
            {"rna": {"length": 0, "sequence": "ACG"}}), 3)
        self.assertIsNone(fs.rna_length_of(_sample("x")))
        self.assertIsNone(fs.rna_length_of({}))

    def test_sample_subset_deterministic_and_capped(self):
        ids = [f"s{i:03d}" for i in range(50)]
        a = fs.sample_subset(ids, 10, 42)
        b = fs.sample_subset(ids, 10, 42)
        self.assertEqual(a, b)                 # reproducible
        self.assertEqual(len(a), 10)
        self.assertEqual(a, sorted(a))         # output sorted
        self.assertNotEqual(a, fs.sample_subset(ids, 10, 7))  # seed matters
        # fewer than n → all (sorted)
        self.assertEqual(fs.sample_subset(["b", "a"], 5, 42),
                         ["a", "b"])

    def test_length_stats(self):
        s = fs._length_stats([10, 20, 30, 40])
        self.assertEqual((s["n"], s["min"], s["max"]), (4, 10, 40))
        self.assertEqual(s["mean"], 25.0)
        self.assertEqual(s["median"], 25.0)
        self.assertEqual(fs._length_stats([])["n"], 0)


class TestFilter(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.sdir, self.splits = _build_input(self.tmp)
        self.smap = fs.load_splits_map(self.splits)

    def test_splits_map(self):
        self.assertEqual(self.smap["a"], "train")
        self.assertEqual(self.smap["e"], "test")
        self.assertIn("a", self.smap)

    def test_filter_counts_and_split_buckets(self):
        r = fs.filter_samples(self.sdir, self.smap, 100)
        c = r["counts"]
        # a,b,c,d,e,f,z kept (≤100); g,h too long; i no rna; bad json
        self.assertEqual(c["kept"], 7)
        self.assertEqual(c["filtered_too_long"], 2)
        self.assertEqual(c["no_rna_length"], 1)
        self.assertEqual(c["bad_json"], 1)
        bs = r["by_split"]
        self.assertEqual(sorted(bs["train"]), ["a", "b", "c"])
        self.assertEqual(sorted(bs["val"]), ["d"])
        self.assertEqual(sorted(bs["test"]), ["e", "f"])
        self.assertEqual(bs["unknown"], ["z"])  # not in splits.json
        self.assertEqual(sorted(r["rna_lens"]),
                         [50, 60, 70, 80, 90, 95, 100])
        # default min=0 → no lower-bound filtering
        self.assertEqual(r["counts"]["filtered_too_short"], 0)

    def test_min_rna_length_filters_short(self):
        # min=80 → drop a(50), e(70), z(60); keep b,c,d,f
        r = fs.filter_samples(self.sdir, self.smap, 100,
                              min_rna_length=80)
        c = r["counts"]
        self.assertEqual(c["filtered_too_short"], 3)
        self.assertEqual(c["filtered_too_long"], 2)
        self.assertEqual(c["kept"], 4)
        self.assertEqual(sorted(r["rna_lens"]), [80, 90, 95, 100])
        self.assertEqual(sorted(r["by_split"]["train"]), ["b", "c"])

    def test_band_min_and_max(self):
        # 70 <= L <= 90 → e(70), b(80), d(90); drop a(50),z(60) short,
        # c(100),f(95) long
        r = fs.filter_samples(self.sdir, self.smap, 90,
                              min_rna_length=70)
        self.assertEqual(sorted(r["rna_lens"]), [70, 80, 90])


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.sdir, self.splits = _build_input(self.tmp)
        self.out = self.tmp / "out"

    def _run(self, *extra):
        return fs.main([
            "--input-dir", str(self.sdir),
            "--output-dir", str(self.out),
            "--splits-json", str(self.splits),
            "--max-rna-length", "100",
            "--train-sample-n", "2",
            "--test-sample-n", "2",
            *extra])

    def test_writes_layout(self):
        self.assertEqual(self._run(), 0)
        # samples copied
        for sid in ("a", "b", "c", "d", "e", "f", "z"):
            self.assertTrue(
                (self.out / "samples" / f"{sid}.json").is_file(), sid)
        for sid in ("g", "h", "i"):
            self.assertFalse(
                (self.out / "samples" / f"{sid}.json").exists(), sid)
        sp = self.out / "splits"
        self.assertEqual(
            (sp / "train.txt").read_text(encoding="utf-8").split(),
            ["a", "b", "c"])
        self.assertEqual(
            (sp / "val.txt").read_text(encoding="utf-8").split(), ["d"])
        self.assertEqual(
            sorted((sp / "test.txt").read_text(
                encoding="utf-8").split()), ["e", "f"])
        # train_200 capped at 2, test_200 has only 2 available
        t2 = (sp / "train_200.txt").read_text(encoding="utf-8").split()
        self.assertEqual(len(t2), 2)
        self.assertTrue(set(t2).issubset({"a", "b", "c"}))
        self.assertEqual(
            sorted((sp / "test_200.txt").read_text(
                encoding="utf-8").split()), ["e", "f"])
        stats = json.loads(
            (self.out / "filter_stats.json").read_text(encoding="utf-8"))
        self.assertEqual(stats["counts"]["kept"], 7)
        self.assertEqual(stats["rna_length_stats"]["max"], 100)
        self.assertEqual(stats["max_rna_length"], 100)

    def test_train_200_reproducible(self):
        self._run()
        first = (self.out / "splits" / "train_200.txt").read_text(
            encoding="utf-8")
        # wipe + rerun → identical sampling (fixed seed)
        import shutil
        shutil.rmtree(self.out)
        self._run()
        self.assertEqual(
            (self.out / "splits" / "train_200.txt").read_text(
                encoding="utf-8"), first)

    def test_dry_run_writes_nothing(self):
        self.assertEqual(self._run("--dry-run"), 0)
        self.assertFalse(self.out.exists())

    def test_missing_input_dir_returns_1(self):
        rc = fs.main(["--input-dir", str(self.tmp / "nope"),
                      "--output-dir", str(self.out),
                      "--splits-json", str(self.splits)])
        self.assertEqual(rc, 1)

    def test_bad_max_len_returns_1(self):
        rc = fs.main(["--input-dir", str(self.sdir),
                      "--output-dir", str(self.out),
                      "--max-rna-length", "0"])
        self.assertEqual(rc, 1)

    def test_min_band_applied_and_recorded(self):
        # 80 <= L <= 100 keeps b,c,d,f (drops a/e/z short, g/h long).
        self.assertEqual(self._run("--min-rna-length", "80"), 0)
        present = sorted(
            p.stem for p in (self.out / "samples").glob("*.json"))
        self.assertEqual(present, ["b", "c", "d", "f"])
        for sid in ("a", "e", "z"):
            self.assertFalse(
                (self.out / "samples" / f"{sid}.json").exists(), sid)
        stats = json.loads(
            (self.out / "filter_stats.json").read_text(encoding="utf-8"))
        self.assertEqual(stats["min_rna_length"], 80)
        self.assertEqual(stats["max_rna_length"], 100)
        self.assertEqual(stats["counts"]["filtered_too_short"], 3)
        self.assertEqual(stats["counts"]["kept"], 4)
        self.assertEqual(stats["rna_length_stats"]["min"], 80)

    def test_default_min_is_20(self):
        # no --min-rna-length flag → default 20 recorded in stats.
        self.assertEqual(self._run(), 0)
        stats = json.loads(
            (self.out / "filter_stats.json").read_text(encoding="utf-8"))
        self.assertEqual(stats["min_rna_length"], 20)

    def test_min_gt_max_returns_1(self):
        rc = fs.main(["--input-dir", str(self.sdir),
                      "--output-dir", str(self.out),
                      "--min-rna-length", "150",
                      "--max-rna-length", "100"])
        self.assertEqual(rc, 1)

    def test_negative_min_returns_1(self):
        rc = fs.main(["--input-dir", str(self.sdir),
                      "--output-dir", str(self.out),
                      "--min-rna-length", "-5"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
