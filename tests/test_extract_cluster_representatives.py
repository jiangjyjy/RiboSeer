"""Mock tests for scripts/extract_cluster_representatives.py."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import scripts.extract_cluster_representatives as ecr  # noqa: E402


def _sample(sid, *, res=2.0, gt=8, pdb="abcd", chain="A"):
    return {
        "sample_id": sid,
        "source_pdb": pdb,
        "protein": {"sequence": "M" * 50, "length": 50,
                    "chain_id": chain},
        "rna": {"sequence": "G" * 30, "length": 30},
        "interaction": {"binding_protein_residues":
                        list(range(1, gt + 1))},
        "data_availability": {"resolution": res,
                              "experimental_method": "X-RAY",
                              "quality_tier": "strict"},
    }


def _splits(meta_per_sid):
    return {"samples": {sid: m for sid, m in meta_per_sid.items()}}


def _scanned(samples):
    return [(s["sample_id"], Path(s["sample_id"] + ".json"), s)
            for s in samples]


class TestRankKey(unittest.TestCase):
    def test_lower_resolution_wins(self):
        a = ecr._rank_key(_sample("a", res=2.0))
        b = ecr._rank_key(_sample("b", res=3.0))
        self.assertLess(a, b)

    def test_missing_resolution_loses(self):
        # A real number (even high) beats null
        nmr = _sample("a", res=None)  # type: ignore[arg-type]
        nmr["data_availability"]["resolution"] = None
        xray = _sample("b", res=4.5)
        self.assertLess(ecr._rank_key(xray), ecr._rank_key(nmr))

    def test_tie_breaks_on_gt_then_id(self):
        a = ecr._rank_key(_sample("a", res=2.0, gt=8))
        b = ecr._rank_key(_sample("b", res=2.0, gt=12))
        # larger gt wins -> b's key smaller
        self.assertLess(b, a)
        c = ecr._rank_key(_sample("c", res=2.0, gt=8))
        # same gt, same res -> sample_id ascending: a < c
        self.assertLess(a, c)


class TestPickReps(unittest.TestCase):
    def test_one_rep_per_cluster_lowest_res(self):
        samples = [_sample("a", res=3.0), _sample("b", res=2.0),
                   _sample("c", res=1.5), _sample("d", res=2.5)]
        splits = _splits({
            "a": {"protein_cluster_id": "p1"},
            "b": {"protein_cluster_id": "p1"},
            "c": {"protein_cluster_id": "p2"},
            "d": {"protein_cluster_id": "p2"},
        })
        smap = ecr._splits_map_full(splits)
        reps = ecr.pick_representatives(_scanned(samples), smap)
        self.assertEqual(reps["p1"]["sample_id"], "b")  # 2.0 < 3.0
        self.assertEqual(reps["p2"]["sample_id"], "c")  # 1.5 < 2.5
        self.assertEqual(reps["p1"]["cluster_size_in_pool"], 2)

    def test_unclustered_bucket(self):
        samples = [_sample("x", res=2.0)]
        # sample not in splits.json -> __unclustered__
        reps = ecr.pick_representatives(_scanned(samples),
                                        ecr._splits_map_full({}))
        self.assertIn("__unclustered__", reps)
        self.assertEqual(
            reps["__unclustered__"]["sample_id"], "x")


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        sdir = self.tmp / "samples"
        sdir.mkdir()
        for sid, res, pc in (("a", 3.0, "p1"), ("b", 2.0, "p1"),
                             ("c", 1.5, "p2")):
            (sdir / f"{sid}.json").write_text(
                json.dumps(_sample(sid, res=res)), encoding="utf-8")
        self.sdir = sdir
        self.sj = self.tmp / "splits.json"
        self.sj.write_text(json.dumps(_splits({
            "a": {"protein_cluster_id": "p1"},
            "b": {"protein_cluster_id": "p1"},
            "c": {"protein_cluster_id": "p2"},
        })), encoding="utf-8")
        self.out = self.tmp / "reps.txt"

    def test_writes_txt_and_sidecar_json(self):
        rc = ecr.main(["--input-dir", str(self.sdir),
                       "--splits-json", str(self.sj),
                       "--output", str(self.out)])
        self.assertEqual(rc, 0)
        lines = [ln for ln in self.out.read_text(
            encoding="utf-8").splitlines() if ln.strip()]
        # 2 clusters -> 2 lines, ordered by cluster id
        self.assertEqual(lines, ["b\tp1", "c\tp2"])
        meta = json.loads((self.tmp / "reps.json")
                          .read_text(encoding="utf-8"))
        self.assertEqual(meta["n_clusters"], 2)
        self.assertEqual({r["sample_id"]
                          for r in meta["representatives"]},
                         {"b", "c"})


if __name__ == "__main__":
    unittest.main()
