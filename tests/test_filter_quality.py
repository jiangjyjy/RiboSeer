"""Mock tests for scripts/filter_quality.py (+ analyze_samples import)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.filter_quality as fq  # noqa: E402


def _sample(sid, *, plen=100, rlen=60, gt=8, res=2.5, domain=None,
            method="X-RAY", tier="strict"):
    return {
        "sample_id": sid,
        "protein": {"sequence": "M" * (plen or 0),
                    "length": plen, "domain": domain},
        "rna": {"sequence": "G" * (rlen or 0), "length": rlen},
        "interaction": {"binding_protein_residues":
                        list(range(1, gt + 1))},
        "data_availability": {"resolution": res,
                              "experimental_method": method,
                              "quality_tier": tier},
    }


class TestExtractors(unittest.TestCase):
    def test_protein_length(self):
        self.assertEqual(fq.protein_length_of(_sample("a", plen=42)), 42)
        # falls back to sequence length when length missing/invalid
        self.assertEqual(fq.protein_length_of(
            {"protein": {"sequence": "MKRMK", "length": 0}}), 5)
        self.assertIsNone(fq.protein_length_of({"protein": {}}))
        self.assertIsNone(fq.protein_length_of({}))

    def test_gt_and_resolution_and_ratio(self):
        s = _sample("a", plen=100, gt=10, res=3.0)
        self.assertEqual(fq.gt_binding_count(s), 10)
        self.assertEqual(fq.resolution_of(s), 3.0)
        self.assertAlmostEqual(fq.binding_ratio_of(s), 0.1)
        # null resolution -> None (NMR/cryo-EM/predicted)
        s2 = _sample("b", res=None)
        self.assertIsNone(fq.resolution_of(s2))
        # absent interaction -> 0, ratio 0
        self.assertEqual(fq.gt_binding_count({}), 0)
        self.assertEqual(fq.binding_ratio_of(_sample("c", gt=0)), 0.0)

    def test_domain_method_tier_buckets(self):
        self.assertEqual(fq.domain_of(_sample("a")), "<none>")
        self.assertEqual(
            fq.domain_of(_sample("a", domain="RRM")), "RRM")
        self.assertEqual(fq.method_of(_sample("a")), "X-RAY")
        self.assertEqual(fq.tier_of(_sample("a")), "strict")


class TestPredicate(unittest.TestCase):
    def test_each_gate_fires_in_order(self):
        T = fq.QualityThresholds
        # rna band
        self.assertEqual(T(min_rna=20).passes(
            _sample("a", rlen=10))[1], "rna_too_short")
        self.assertEqual(T(max_rna=100).passes(
            _sample("a", rlen=150))[1], "rna_too_long")
        # protein band
        self.assertEqual(T(max_prot=400).passes(
            _sample("a", plen=999))[1], "protein_too_long")
        self.assertEqual(T(min_prot=50).passes(
            _sample("a", plen=10))[1], "protein_too_short")
        # gt
        self.assertEqual(T(min_gt=5).passes(
            _sample("a", gt=2))[1], "gt_too_few")
        # ratio
        self.assertEqual(T(min_ratio=0.5).passes(
            _sample("a", plen=100, gt=10))[1], "binding_ratio_low")
        # resolution numeric cap
        self.assertEqual(T(max_res=3.0).passes(
            _sample("a", res=4.0))[1], "resolution_poor")
        # kept
        self.assertEqual(T(max_prot=400, min_gt=5, max_res=3.5)
                         .passes(_sample("a"))[0], True)

    def test_quality_tier_gate(self):
        T = fq.QualityThresholds
        s = _sample("a", tier="low")
        self.assertTrue(T().passes(s)[0])  # None = accept any
        ok, why = T(allowed_tiers=["strict", "standard"]).passes(s)
        self.assertFalse(ok)
        self.assertEqual(why, "tier_excluded")
        self.assertTrue(
            T(allowed_tiers=["strict"]).passes(
                _sample("b", tier="strict"))[0])

    def test_missing_resolution_kept_by_default_dropped_on_flag(self):
        T = fq.QualityThresholds
        s = _sample("a", res=None)
        self.assertTrue(T(max_res=3.0).passes(s)[0])
        ok, why = T(max_res=3.0, drop_missing_res=True).passes(s)
        self.assertFalse(ok)
        self.assertEqual(why, "resolution_missing")

    def test_no_rna_or_protein_length(self):
        T = fq.QualityThresholds
        self.assertEqual(
            T().passes({"rna": {}})[1], "no_rna_length")
        self.assertEqual(
            T().passes({"rna": {"length": 60},
                        "protein": {}})[1], "no_protein_length")


def _build_input(tmp: Path):
    sdir = tmp / "in"
    sdir.mkdir(parents=True)
    # id, plen, gt, res, split
    specs = [
        ("a", 100, 10, 2.0, "train"),   # easy, keep
        ("b", 300, 6, 3.0, "train"),    # keep at prot<=400,gt>=5,res<=3.5
        ("c", 800, 12, 2.5, "train"),   # protein too long
        ("d", 150, 2, 2.0, "val"),      # too few GT
        ("e", 200, 7, 4.5, "test"),     # resolution poor
        ("f", 250, 9, None, "test"),    # missing res -> kept by default
        ("g", 120, 8, 2.8, "test"),     # easy, keep
    ]
    meta = {}
    for sid, pl, gt, res, sp in specs:
        (sdir / f"{sid}.json").write_text(
            json.dumps(_sample(sid, plen=pl, gt=gt, res=res)),
            encoding="utf-8")
        meta[sid] = {"split": sp}
    (sdir / "bad.json").write_text("{nope", encoding="utf-8")
    splits = tmp / "splits.json"
    splits.write_text(json.dumps({"samples": meta}), encoding="utf-8")
    return sdir, splits


class TestApplyFilter(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.sdir, self.splits = _build_input(self.tmp)
        self.smap = fq.load_splits_map(self.splits)
        self.scanned = fq.scan_dir(self.sdir)

    def test_scan_skips_bad_json(self):
        ids = sorted(sid for sid, _, _ in self.scanned)
        self.assertEqual(ids, ["a", "b", "c", "d", "e", "f", "g"])

    def test_combo_prot400_gt5_res35(self):
        thr = fq.QualityThresholds(max_prot=400, min_gt=5,
                                   max_res=3.5)
        r = fq.apply_filter(self.scanned, self.smap, thr)
        kept = sorted(s for s, _, _, _ in r["kept"])
        # a,b kept (train); g,f kept (test, f via missing-res keep);
        # c long, d few GT, e poor res
        self.assertEqual(kept, ["a", "b", "f", "g"])
        self.assertEqual(sorted(r["by_split"]["train"]), ["a", "b"])
        self.assertEqual(sorted(r["by_split"]["test"]), ["f", "g"])
        self.assertEqual(r["reasons"]["protein_too_long"], 1)
        self.assertEqual(r["reasons"]["gt_too_few"], 1)
        self.assertEqual(r["reasons"]["resolution_poor"], 1)

    def test_drop_missing_resolution_removes_f(self):
        thr = fq.QualityThresholds(max_prot=400, min_gt=5,
                                   max_res=3.5,
                                   drop_missing_res=True)
        r = fq.apply_filter(self.scanned, self.smap, thr)
        kept = sorted(s for s, _, _, _ in r["kept"])
        self.assertEqual(kept, ["a", "b", "g"])
        self.assertEqual(r["reasons"]["resolution_missing"], 1)

    def test_combo_table_shape(self):
        rows = fq.combo_table(self.scanned, self.smap,
                              fq.DEFAULT_COMBOS,
                              dict(min_rna=20, max_rna=100))
        self.assertEqual(len(rows), len(fq.DEFAULT_COMBOS))
        for row in rows:
            self.assertEqual(
                row["kept"],
                row["train"] + row["val"] + row["test"]
                + row["unknown"])
        # tighter combos never keep more than looser ones
        kept = [r["kept"] for r in rows]
        self.assertGreaterEqual(kept[0], kept[-1])

    def test_sample_subset_reproducible(self):
        ids = [f"s{i:03d}" for i in range(40)]
        a = fq.sample_subset(ids, 10, 42)
        self.assertEqual(a, fq.sample_subset(ids, 10, 42))
        self.assertEqual(len(a), 10)
        self.assertEqual(fq.sample_subset(["b", "a"], 9, 1),
                         ["a", "b"])


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.sdir, self.splits = _build_input(self.tmp)
        self.out = self.tmp / "out"

    def _run(self, *extra):
        return fq.main([
            "--input-dir", str(self.sdir),
            "--output-dir", str(self.out),
            "--splits-json", str(self.splits),
            "--max-protein-length", "400",
            "--min-gt-binding", "5",
            "--max-resolution", "3.5",
            "--train-sample-n", "2",
            "--test-sample-n", "2",
            *extra])

    def test_writes_layout_and_stats(self):
        self.assertEqual(self._run(), 0)
        for sid in ("a", "b", "f", "g"):
            self.assertTrue(
                (self.out / "samples" / f"{sid}.json").is_file(), sid)
        for sid in ("c", "d", "e"):
            self.assertFalse(
                (self.out / "samples" / f"{sid}.json").exists(), sid)
        sp = self.out / "splits"
        self.assertEqual(
            sorted((sp / "train.txt").read_text("utf-8").split()),
            ["a", "b"])
        self.assertEqual(
            sorted((sp / "test.txt").read_text("utf-8").split()),
            ["f", "g"])
        st = json.loads((self.out / "filter_quality_stats.json")
                        .read_text("utf-8"))
        self.assertEqual(st["counts"]["kept"], 4)
        self.assertEqual(st["thresholds"]["max_protein_length"], 400)
        self.assertEqual(st["drop_reasons"]["protein_too_long"], 1)

    def test_dry_run_writes_nothing(self):
        self.assertEqual(self._run("--dry-run"), 0)
        self.assertFalse(self.out.exists())

    def test_bad_args_return_1(self):
        self.assertEqual(fq.main([
            "--input-dir", str(self.sdir), "--output-dir",
            str(self.out), "--max-protein-length", "0"]), 1)
        self.assertEqual(fq.main([
            "--input-dir", str(self.tmp / "nope"),
            "--output-dir", str(self.out)]), 1)
        self.assertEqual(fq.main([
            "--input-dir", str(self.sdir), "--output-dir",
            str(self.out), "--min-rna-length", "150",
            "--max-rna-length", "100"]), 1)


class TestAnalyzeImport(unittest.TestCase):
    """analyze_samples must import cleanly and reuse fq extractors."""

    def test_import_and_combo_reuse(self):
        import scripts.analyze_samples as an
        self.assertIs(an.gt_binding_count, fq.gt_binding_count)
        self.assertIs(an.protein_length_of, fq.protein_length_of)


if __name__ == "__main__":
    unittest.main()
