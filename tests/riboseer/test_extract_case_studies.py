"""Mock tests for scripts/riboseer/extract_case_studies.py.

Pure helpers (jaccard, segments, Pearson, case scorers) plus an
end-to-end collect_sample + main() run on synthetic step4 / sample /
SCOPE / POLISH / prediction JSON. No raw structures are provided, so the
DCC path exercises the graceful ``None`` branch.
"""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.riboseer import extract_case_studies as ecs  # noqa: E402


# ---- pure helpers --------------------------------------------------------


class TestHelpers(unittest.TestCase):
    def test_jaccard(self):
        self.assertAlmostEqual(ecs.jaccard({1, 2, 3}, {2, 3, 4}), 2 / 4)
        self.assertEqual(ecs.jaccard({1}, {1}), 1.0)
        self.assertIsNone(ecs.jaccard(set(), set()))     # undefined

    def test_mean_pairwise_jaccard(self):
        gates = {"boltz2": {1, 2, 3}, "chai1": {1, 2, 3},
                 "rosettafold2na": {1, 2, 3}}
        self.assertEqual(ecs.mean_pairwise_jaccard(gates), 1.0)
        # < 2 tools with a gate -> None
        self.assertIsNone(ecs.mean_pairwise_jaccard({"boltz2": {1}}))
        self.assertIsNone(ecs.mean_pairwise_jaccard(
            {"boltz2": {1}, "chai1": set()}))

    def test_count_segments(self):
        self.assertEqual(ecs.count_segments([]), 0)
        self.assertEqual(ecs.count_segments([1, 2, 3, 4]), 1)
        # gap > 3 starts a new run
        self.assertEqual(ecs.count_segments([1, 2, 3, 50, 51, 100]), 3)
        self.assertEqual(ecs.count_segments([5, 7, 9]), 1)   # gaps <= 3

    def test_pearson_over(self):
        # perfectly aligned -> R = 1.0
        r = ecs._pearson_over({1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0},
                              [1, 2, 3, 4], {1, 2})
        self.assertAlmostEqual(r, 1.0, places=4)
        # constant prediction -> undefined
        self.assertIsNone(ecs._pearson_over(
            {1: 0.5, 2: 0.5, 3: 0.5}, [1, 2, 3], {1}))
        # all-GT (no variance in gt) -> undefined
        self.assertIsNone(ecs._pearson_over(
            {1: 0.9, 2: 0.1}, [1, 2], {1, 2}))


# ---- case scorers --------------------------------------------------------


def _rec(**kw):
    base = dict(
        sample_id="s", scope_family="Novel", rna_context="unstructured",
        catA_jaccard=0.4, polish_action="accept", gt_segments=1,
        n_gt=10, Lr=30, Lp=120, beats_best=True, beats_all=True,
        delta_pearson=0.1, riboseer_dcc=2.0, best_tool_dcc=5.0,
        delta_dcc=3.0,
    )
    base.update(kw)
    return base


class TestScorers(unittest.TestCase):
    def test_case1_perfect(self):
        rec = _rec(scope_family="RRM", rna_context="stem-loop",
                   catA_jaccard=0.8, polish_action="extend")
        score, crit = ecs.score_case1(rec)
        self.assertEqual(sum(p for _, p in crit), 5)        # all pass
        self.assertGreater(score, 5.0)                      # +tiebreak

    def test_case1_dropped_when_no_improvement(self):
        self.assertIsNone(ecs.score_case1(_rec(beats_best=False)))

    def test_case2_disagreement(self):
        rec = _rec(scope_family="Multi", catA_jaccard=0.15,
                   polish_action="relocate", beats_all=True)
        score, crit = ecs.score_case2(rec)
        self.assertEqual(sum(p for _, p in crit), 4)
        # segments alternative also counts as "multi"
        rec2 = _rec(scope_family="Novel", gt_segments=3,
                    catA_jaccard=0.15, polish_action="relocate")
        _, crit2 = ecs.score_case2(rec2)
        self.assertTrue(crit2[0][1])                        # multi via segs

    def test_case3_long_rna(self):
        rec = _rec(Lr=80, rna_context="junction", n_gt=30)
        score, crit = ecs.score_case3(rec, gt_mean=10.0)
        labels = {l: p for l, p in crit}
        self.assertTrue(any("junction" in l and p for l, p in crit))
        self.assertTrue(any("70" in l and p for l, p in crit))
        self.assertEqual(sum(p for _, p in crit), 5)

    def test_partial_match_still_scores(self):
        # only family matches for case 1 -> score ~1 (+tiebreak), not None
        rec = _rec(scope_family="RRM")
        res = ecs.score_case1(rec)
        self.assertIsNotNone(res)
        score, crit = res
        self.assertEqual(sum(p for _, p in crit), 2)  # family + beats_best


# ---- end-to-end ----------------------------------------------------------


def _pred(tool_id, binding, per_res, success=True):
    """A step4 tool prediction: binding gate + explicit per-residue pae
    score map (residue id -> score)."""
    return {
        "tool_id": tool_id, "success": success,
        "binding_protein_residues": list(binding),
        "per_residue_pae_score": {str(r): v for r, v in per_res.items()},
        "predicted_structure_path": None,
    }


class TestEndToEnd(unittest.TestCase):
    def _setup(self, td):
        root = Path(td)
        samples = root / "data" / "samples"
        step4 = root / "step4"
        preds = root / "preds"
        scope = root / "scope"
        polish = root / "polish"
        raw = root / "raw"
        for d in (samples, step4, preds, scope, polish, raw):
            d.mkdir(parents=True, exist_ok=True)

        sid = "rrm1_a_b"
        # protein length 6, GT binding = {1,2,3}
        (samples / f"{sid}.json").write_text(json.dumps({
            "sample_id": sid, "source_pdb": "rrm1",
            "protein": {"chain_id": "A", "length": 6,
                        "sequence": "MKLPQR",
                        "resolved_residues": [1, 2, 3, 4, 5, 6]},
            "rna": {"chain_id": "B", "length": 30,
                    "sequence": "A" * 30},
            "interaction": {"binding_protein_residues": [1, 2, 3]},
        }), encoding="utf-8")

        # 3 Cat-A tools that strongly agree on their gate (~{1,2,3}) ->
        # high pairwise Jaccard, but each has IMPERFECT per-residue
        # ranking (a non-binding residue scores above a binding one), so
        # their Pearson R < 1 and RiboSeer can improve on them. chai1 is
        # the strongest single tool.
        (step4 / f"{sid}.jsonl").write_text(json.dumps({
            "sample_id": sid,
            "predictions": [
                _pred("boltz2", [1, 2, 3],
                      {1: 0.8, 2: 0.6, 3: 0.5, 4: 0.55, 5: 0.2, 6: 0.1}),
                _pred("chai1", [1, 2, 3],
                      {1: 0.85, 2: 0.55, 3: 0.5, 4: 0.6, 5: 0.3, 6: 0.1}),
                _pred("rosettafold2na", [1, 2],
                      {1: 0.7, 2: 0.6, 3: 0.45, 4: 0.5, 5: 0.4, 6: 0.2}),
            ],
        }), encoding="utf-8")

        # RiboSeer full-system prediction: better separation than any tool
        (preds / f"{sid}.json").write_text(json.dumps({
            "residue_ids": [1, 2, 3, 4, 5, 6],
            "probabilities": [0.97, 0.95, 0.93, 0.05, 0.04, 0.02],
        }), encoding="utf-8")

        (scope / f"{sid}.json").write_text(json.dumps({
            "sample_id": sid, "protein_family": "RRM",
            "rna_context": "stem-loop", "difficulty": "Medium",
            "confidence": 0.8, "source": "llm",
        }), encoding="utf-8")
        (polish / f"{sid}.json").write_text(json.dumps({
            "sample_id": sid, "action": "extend", "residues": [3],
            "target_residues": [4], "source": "llm",
        }), encoding="utf-8")

        (root / "split.txt").write_text(sid + "\n", encoding="utf-8")
        return root, samples.parent, step4, preds, scope, polish, raw

    def test_collect_sample(self):
        with tempfile.TemporaryDirectory() as td:
            (root, data_dir, step4, preds, scope,
             polish, raw) = self._setup(td)
            fs = ecs.load_predictions_dir(preds)
            rec = ecs.collect_sample(
                "rrm1_a_b", data_dir=data_dir, step4_dir=step4,
                raw_dir=raw, scope_dir=scope, polish_dir=polish,
                fs_preds=fs)
            self.assertIsNotNone(rec)
            self.assertEqual(rec["scope_family"], "RRM")
            self.assertEqual(rec["rna_context"], "stem-loop")
            self.assertEqual(rec["Lr"], 30)
            self.assertEqual(rec["n_gt"], 3)
            self.assertIn(rec["best_tool"], ecs.CAT_A_TOOLS)
            self.assertEqual(rec["polish_action"], "extend")
            # high Cat-A agreement
            self.assertGreater(rec["catA_jaccard"], 0.5)
            # RiboSeer improves Pearson
            self.assertTrue(rec["beats_best"])
            self.assertGreater(rec["delta_pearson"], 0.0)
            # no structure -> DCC None (graceful)
            self.assertIsNone(rec["riboseer_dcc"])

    def test_main_runs_and_reports_case1(self):
        with tempfile.TemporaryDirectory() as td:
            (root, data_dir, step4, preds, scope,
             polish, raw) = self._setup(td)
            out_json = root / "out.json"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = ecs.main([
                    "--data-dir", str(data_dir),
                    "--step4-dir", str(step4),
                    "--predictions-dir", str(preds),
                    "--scope-dir", str(scope),
                    "--polish-dir", str(polish),
                    "--raw-dir", str(raw),
                    "--split-file", str(root / "split.txt"),
                    "--output", str(out_json),
                ])
            self.assertEqual(rc, 0)
            text = buf.getvalue()
            self.assertIn("Case 1", text)
            self.assertIn("rrm1_a_b", text)
            dump = json.loads(out_json.read_text(encoding="utf-8"))
            case1_key = next(k for k in dump if k.startswith("Case 1"))
            top = dump[case1_key]
            self.assertTrue(top)
            self.assertEqual(top[0]["sample_id"], "rrm1_a_b")
            # perfect case-1 match: all 5 criteria pass
            self.assertTrue(all(c["passed"] for c in top[0]["criteria"]))


if __name__ == "__main__":
    unittest.main()
