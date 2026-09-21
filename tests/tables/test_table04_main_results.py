"""Mock tests for scripts/tables/table04_main_results.py."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import scripts.tables.table04_main_results as ept  # noqa: E402
from step5_fusion.enriched_fusion import XGB_OK, EnrichedFusion  # noqa: E402


# ---- fixtures ------------------------------------------------------------


def _pred(tool_id, *, success=True, binding=None, pae=None, conf=None):
    d = {"tool_id": tool_id, "category": "A", "success": success}
    if binding is not None:
        d["binding_protein_residues"] = binding
    if pae is not None:
        d["per_residue_pae_score"] = pae
    if conf is not None:
        d["per_residue_confidence"] = conf
    return d


def _write_sample(proc: Path, sid: str, *, resolved, gt, seq=None):
    (proc / "samples").mkdir(parents=True, exist_ok=True)
    seq = seq or "".join("MKRYAVCDEF"[i % 10]
                         for i in range(max(resolved)))
    (proc / "samples" / f"{sid}.json").write_text(json.dumps({
        "sample_id": sid,
        "protein": {"sequence": seq, "length": len(seq),
                    "resolved_residues": resolved},
        "interaction": {"binding_protein_residues": gt},
    }), encoding="utf-8")


def _write_step4(step4: Path, sid: str, predictions, *, extra_lines=0):
    step4.mkdir(parents=True, exist_ok=True)
    rec = {"sample_id": sid, "predictions": predictions}
    lines = []
    # extra_lines: stale earlier records the reader must IGNORE (only
    # the last line counts).
    for _ in range(extra_lines):
        lines.append(json.dumps({"sample_id": sid, "predictions": []}))
    lines.append(json.dumps(rec))
    (step4 / f"{sid}.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def _dataset(tmp: Path, n=6, *, extra_lines=1):
    step4, proc = tmp / "step4", tmp / "proc"
    res = [1, 2, 3, 4, 5, 6]
    gt = [2, 4, 6]
    sids = []
    for i in range(n):
        sid = f"s{i:02d}"
        sids.append(sid)
        _write_sample(proc, sid, resolved=res, gt=gt)
        pae = {str(r): (0.85 if r in gt else 0.1) + 0.01 * i
               for r in res}
        conf = {str(r): (0.7 if r in gt else 0.2) for r in res}
        rf = {str(r): (0.8 if r in gt else 0.15) for r in res}
        _write_step4(step4, sid, [
            _pred("boltz2", binding=gt, pae=pae, conf=conf),
            _pred("rosettafold2na", binding=gt, pae=rf, conf=conf),
            _pred("p2rank", binding=gt, conf=conf),  # conf-only
            _pred("equipnas", success=False, binding=gt, conf=conf),
        ], extra_lines=extra_lines)
    return step4, proc, sids


# ---- helper-level tests --------------------------------------------------


class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_last_line_wins(self):
        f = self.tmp / "x.jsonl"
        f.write_text(
            json.dumps({"predictions": [{"tool_id": "old"}]}) + "\n"
            + json.dumps({"predictions": [{"tool_id": "new"}]}) + "\n",
            encoding="utf-8")
        rec = ept._read_last_jsonl_record(f)
        self.assertEqual(rec["predictions"][0]["tool_id"], "new")

    def test_last_line_missing_file(self):
        self.assertIsNone(
            ept._read_last_jsonl_record(self.tmp / "nope.jsonl"))

    def test_tool_scores_prefers_pae(self):
        p = _pred("t", pae={"1": 0.9}, conf={"1": 0.1, "2": 0.2})
        self.assertEqual(ept._tool_scores(p), {1: 0.9})
        # falls back to confidence when pae absent/empty
        p2 = _pred("t", conf={"3": 0.5})
        self.assertEqual(ept._tool_scores(p2), {3: 0.5})

    def test_corr_row_skips_degenerate(self):
        self.assertIsNone(ept._corr_row([1.0], [1.0]))           # n<2
        self.assertIsNone(ept._corr_row([0.5, 0.5], [0.0, 1.0]))  # const pred
        self.assertIsNone(ept._corr_row([0.1, 0.9], [1.0, 1.0]))  # GT no var
        c = ept._corr_row([0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1])
        self.assertIsNotNone(c)
        # r² ≈ pearson² (both independently rounded to 4dp, so only
        # approx — exact equality doesn't survive the rounding).
        self.assertAlmostEqual(c["r_squared"], c["pearson_r"] ** 2,
                               places=2)
        self.assertGreater(c["pearson_r"], 0.9)


# ---- evaluate() ----------------------------------------------------------


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4, self.proc, self.sids = _dataset(self.tmp)

    def test_discovers_tools_skips_failed(self):
        bucket, per_sample = ept.evaluate(
            step4_dir=self.step4, processed_dir=self.proc,
            sample_ids=self.sids, min_gt=0, enriched_model=None)
        # auto-discovered, includes rosettafold2na; equipnas failed →
        # never appears.
        self.assertEqual(set(bucket),
                         {"boltz2", "rosettafold2na", "p2rank"})
        self.assertNotIn("equipnas", bucket)
        self.assertEqual(len(bucket["boltz2"]), len(self.sids))
        # strong synthetic signal → high Pearson.
        prs = [r["pearson_r"] for r in bucket["boltz2"]]
        self.assertGreater(sum(prs) / len(prs), 0.5)

    def test_min_gt_filters_samples(self):
        # GT has 3 residues; min_gt=3 drops every sample (count<=3).
        bucket, _ = ept.evaluate(
            step4_dir=self.step4, processed_dir=self.proc,
            sample_ids=self.sids, min_gt=3, enriched_model=None)
        self.assertEqual(bucket, {})

    def test_resolved_residue_order_and_missing_zero(self):
        # Tool scores only residue 2; vector must be length-6 (resolved)
        # with 0s elsewhere — correlation still defined vs GT [.,1,...].
        step4, proc = self.tmp / "s4b", self.tmp / "pb"
        _write_sample(proc, "z", resolved=[1, 2, 3, 4],
                      gt=[2, 4])
        # binding residues share one score, non-binding share another →
        # perfectly separable, Pearson == 1.0. residue 4 is left
        # UNSCORED so the "missing → 0" path is exercised too (0 is
        # still the low value, so separation holds).
        _write_step4(step4, "z", [
            _pred("boltz2", binding=[2, 4],
                  pae={"2": 0.9, "1": 0.0, "3": 0.0})])
        bucket, _ = ept.evaluate(
            step4_dir=step4, processed_dir=proc,
            sample_ids=["z"], min_gt=0, enriched_model=None)
        self.assertIn("boltz2", bucket)
        # vec over resolved [1,2,3,4] = [0, 0.9, 0, 0] (res4 missing→0);
        # gt = [0,1,0,1]. r = 0.9/sqrt(.6075*1)... not 1.0 — assert the
        # missing residue became 0 and corr is defined & positive.
        self.assertGreater(bucket["boltz2"][0]["pearson_r"], 0.0)
        self.assertLess(bucket["boltz2"][0]["pearson_r"], 1.0)


# ---- aggregate ordering + CSV --------------------------------------------


class TestAggregateAndCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4, self.proc, self.sids = _dataset(self.tmp)

    def test_sorted_by_n_desc(self):
        # Give p2rank fewer usable samples by making its scores constant
        # (skipped) on half the samples.
        rows = ept._aggregate({
            "a": [{"pearson_r": 0.5, "spearman_r": 0.5,
                   "r_squared": 0.25}] * 3,
            "b": [{"pearson_r": 0.4, "spearman_r": 0.4,
                   "r_squared": 0.16}] * 9,
            "enriched_fusion": [{"pearson_r": 0.7, "spearman_r": 0.7,
                                 "r_squared": 0.49}] * 1,
        })
        # tools by n desc, enriched_fusion always last regardless of n.
        self.assertEqual([r["method"] for r in rows],
                         ["b", "a", "enriched_fusion"])
        self.assertEqual(rows[0]["n_samples"], 9)

    def test_cli_writes_csv_and_per_sample(self):
        out = self.tmp / "eval" / "per_tool_correlation.csv"
        rc = ept.main([
            "--step4-dir", str(self.step4),
            "--processed-dir", str(self.proc),
            "--output", str(out)])
        self.assertEqual(rc, 0)
        self.assertTrue(out.is_file())
        ps = out.with_name("per_tool_correlation_per_sample.csv")
        self.assertTrue(ps.is_file())
        with out.open(encoding="utf-8") as f:
            rd = list(csv.DictReader(f))
        # exact header order matches evaluate.py's per_residue CSV.
        self.assertEqual(
            list(rd[0].keys()),
            ["method", "n_samples",
             "pearson_r_mean", "pearson_r_std", "pearson_r_median",
             "spearman_r_mean", "spearman_r_std", "spearman_r_median",
             "r2_mean", "r2_std", "r2_median"])
        methods = [r["method"] for r in rd]
        self.assertIn("rosettafold2na", methods)
        self.assertNotIn("equipnas", methods)  # failed everywhere
        # descending n_samples
        ns = [int(r["n_samples"]) for r in rd]
        self.assertEqual(ns, sorted(ns, reverse=True))

    def test_cli_missing_step4_dir_returns_1(self):
        rc = ept.main([
            "--step4-dir", str(self.tmp / "nope"),
            "--processed-dir", str(self.proc),
            "--output", str(self.tmp / "x.csv")])
        self.assertEqual(rc, 1)


@unittest.skipUnless(XGB_OK, "xgboost not installed")
class TestEnrichedAppended(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # EnrichedFusion._collect reads the FIRST JSONL record, so this
        # dataset has no stale leading lines (the per-tool script's
        # last-line behaviour is covered separately).
        self.step4, self.proc, self.sids = _dataset(
            self.tmp, n=8, extra_lines=0)

    def test_enriched_row_appended_last(self):
        model_dir = self.tmp / "model"
        f = EnrichedFusion({"model": "xgboost", "n_estimators": 20,
                            "max_depth": 3, "learning_rate": 0.3,
                            "min_child_weight": 1, "seed": 0})
        f.train(step4_dir=self.step4, processed_dir=self.proc,
                sample_ids=self.sids, verbose=False)
        f.save(model_dir)

        out = self.tmp / "eval.csv"
        rc = ept.main([
            "--step4-dir", str(self.step4),
            "--processed-dir", str(self.proc),
            "--output", str(out),
            "--enriched-model-dir", str(model_dir)])
        self.assertEqual(rc, 0)
        with out.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(rows[-1]["method"], "enriched_fusion")
        self.assertGreater(int(rows[-1]["n_samples"]), 0)


if __name__ == "__main__":
    unittest.main()
