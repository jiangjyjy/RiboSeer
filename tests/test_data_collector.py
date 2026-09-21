"""Mock tests for the shared per-residue data collector.

The collector is the single source of truth for the training set
fed to both ``LearnedFusion`` (Ridge) and ``XGBFusion`` — a quirk in
its skip rules would silently corrupt every downstream comparison.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from step5_fusion.data_collector import (  # noqa: E402
    DEFAULT_SCORE_FIELDS,
    aa_for, collect_per_residue_data, extract_score,
    load_sample_json, per_residue_to_int_dict, read_jsonl_record,
    resolve_score_fields,
)


# ---- helpers (mirror the synthetic dataset shape used elsewhere) -------


def _write_step4(step4_dir: Path, sid: str, predictions: list[dict]) -> None:
    step4_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "sample_id": sid,
        "tools_run": [p["tool_id"] for p in predictions],
        "predictions": predictions,
    }
    (step4_dir / f"{sid}.jsonl").write_text(
        json.dumps(rec) + "\n", encoding="utf-8",
    )


def _write_sample(processed: Path, sid: str, *, length: int,
                  sequence: str, gt: list[int]) -> None:
    samples = processed / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    sample = {
        "sample_id": sid,
        "protein": {"length": length, "sequence": sequence},
        "rna": {"length": 5, "sequence": "GCGCG"},
        "interaction": {"binding_protein_residues": gt},
    }
    (samples / f"{sid}.json").write_text(
        json.dumps(sample), encoding="utf-8",
    )


# ---- helpers --------------------------------------------------------------


class TestSmallHelpers(unittest.TestCase):
    def test_per_residue_to_int_dict(self):
        self.assertEqual(
            per_residue_to_int_dict({"1": 0.5, "2": 0.7}),
            {1: 0.5, 2: 0.7},
        )
        self.assertEqual(per_residue_to_int_dict({}), {})
        self.assertEqual(per_residue_to_int_dict(None), {})
        # Mixed valid / invalid: only the valid pair survives.
        self.assertEqual(
            per_residue_to_int_dict({"1": 0.5, "abc": "x", 3: "bad"}),
            {1: 0.5},
        )

    def test_aa_for_bounds(self):
        self.assertEqual(aa_for("MKR", 1), "M")
        self.assertEqual(aa_for("MKR", 3), "R")
        self.assertEqual(aa_for("MKR", 0), "")
        self.assertEqual(aa_for("MKR", 4), "")
        self.assertEqual(aa_for("", 1), "")

    def test_resolve_score_fields_defaults(self):
        out = resolve_score_fields({})
        for tool, field in DEFAULT_SCORE_FIELDS.items():
            self.assertEqual(out[tool], field)

    def test_resolve_score_fields_flat_override(self):
        out = resolve_score_fields(
            {"score_fields": {"boltz2": "per_residue_confidence"}}
        )
        self.assertEqual(out["boltz2"], "per_residue_confidence")

    def test_resolve_score_fields_nested_override(self):
        out = resolve_score_fields(
            {"tools": {"chai1": {"score_field": "per_residue_confidence"}}}
        )
        self.assertEqual(out["chai1"], "per_residue_confidence")

    def test_extract_score_falls_back_to_alternate_field(self):
        # Configured field empty → fall back to the other.
        pred = {
            "tool_id": "boltz2",
            "per_residue_pae_score": None,
            "per_residue_confidence": {"3": 0.7},
        }
        self.assertAlmostEqual(
            extract_score(
                pred, 3,
                {"boltz2": "per_residue_pae_score"},
            ),
            0.7,
        )

    def test_extract_score_returns_none_if_neither_field(self):
        pred = {"tool_id": "boltz2"}
        self.assertIsNone(extract_score(
            pred, 1, {"boltz2": "per_residue_pae_score"},
        ))


# ---- read_jsonl_record / load_sample_json --------------------------------


class TestRecordReaders(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_read_jsonl_record_skips_blank_and_comments(self):
        path = self.tmp / "x.jsonl"
        path.write_text(
            "\n# header\n"
            + json.dumps({"sample_id": "s1", "x": 1}) + "\n",
            encoding="utf-8",
        )
        self.assertEqual(read_jsonl_record(path)["x"], 1)

    def test_read_jsonl_record_missing_returns_none(self):
        self.assertIsNone(read_jsonl_record(self.tmp / "nope.jsonl"))

    def test_read_jsonl_record_malformed_returns_none(self):
        path = self.tmp / "bad.jsonl"
        path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(read_jsonl_record(path))

    def test_load_sample_json_case_insensitive_fallback(self):
        proc = self.tmp / "proc"
        samples = proc / "samples"
        samples.mkdir(parents=True)
        # On-disk has uppercase Y; splits.json carries lowercase.
        (samples / "3J46_y_1.json").write_text(
            json.dumps({"sample_id": "3J46_y_1"}), encoding="utf-8",
        )
        loaded = load_sample_json(proc, "3j46_y_1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["sample_id"], "3J46_y_1")


# ---- the main collector ---------------------------------------------------


class TestCollectPerResidueData(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.step4 = self.tmp / "step4"
        self.processed = self.tmp / "processed"

    def test_basic_collection(self):
        # 1 sample, length 4, GT residues 1 & 3, two tools.
        _write_sample(self.processed, "s1", length=4, sequence="MKRY",
                      gt=[1, 3])
        _write_step4(self.step4, "s1", [
            {"tool_id": "boltz2", "category": "A", "sample_id": "s1",
             "success": True, "binding_protein_residues": [1, 3],
             "per_residue_pae_score": {"1": 0.9, "3": 0.85}},
            {"tool_id": "p2rank", "category": "B", "sample_id": "s1",
             "success": True, "binding_protein_residues": [1],
             "per_residue_confidence": {"1": 0.7, "2": 0.1}},
        ])
        out = collect_per_residue_data(
            step4_dir=self.step4, processed_dir=self.processed,
        )
        rows = out["rows"]
        # Residue 1 has score from both, 2 has p2rank only, 3 has boltz2,
        # 4 has nothing AND is in no binding list → skipped.
        self.assertEqual(out["n_samples_used"], 1)
        rids = sorted(r["residue_id"] for r in rows)
        self.assertEqual(rids, [1, 2, 3])
        # Residue 1 row carries both tool scores + correct flags.
        r1 = next(r for r in rows if r["residue_id"] == 1)
        self.assertEqual(r1["sample_id"], "s1")
        self.assertEqual(r1["aa"], "M")
        self.assertEqual(r1["label"], 1)
        self.assertEqual(r1["tool_binding_flags"], {"boltz2": True, "p2rank": True})
        self.assertAlmostEqual(r1["tool_scores"]["boltz2"], 0.9)
        self.assertAlmostEqual(r1["tool_scores"]["p2rank"], 0.7)
        # Residue 3 has boltz2 score AND boltz2 binding flag, no p2rank.
        r3 = next(r for r in rows if r["residue_id"] == 3)
        self.assertEqual(r3["label"], 1)
        self.assertEqual(r3["tool_binding_flags"], {"boltz2": True, "p2rank": False})
        self.assertNotIn("p2rank", r3["tool_scores"])
        # per_tool_scores collects every per-residue value contributed.
        self.assertEqual(len(out["per_tool_scores"]["boltz2"]), 2)  # 1, 3
        self.assertEqual(len(out["per_tool_scores"]["p2rank"]), 2)  # 1, 2

    def test_skips_sample_without_gt(self):
        _write_sample(self.processed, "no_gt",
                      length=3, sequence="MKR", gt=[])
        _write_step4(self.step4, "no_gt", [{
            "tool_id": "boltz2", "category": "A", "sample_id": "no_gt",
            "success": True, "binding_protein_residues": [],
            "per_residue_pae_score": {"1": 0.5},
        }])
        out = collect_per_residue_data(
            step4_dir=self.step4, processed_dir=self.processed,
        )
        self.assertEqual(out["rows"], [])
        self.assertEqual(out["n_samples_used"], 0)
        self.assertEqual(out["n_samples_skipped"], 1)

    def test_keeps_residue_when_only_gating_flag_present(self):
        # Tool emits a binding-list flag but no per-residue score.
        # The collector should keep the row so the gating column has
        # a chance to contribute (regression test for the new
        # "skip pure-zero rows" rule's edge case).
        _write_sample(self.processed, "s1", length=3, sequence="MKR", gt=[2])
        _write_step4(self.step4, "s1", [{
            "tool_id": "fpocket", "category": "B", "sample_id": "s1",
            "success": True, "binding_protein_residues": [2],
            # No per_residue_confidence — only the binding-list flag.
        }])
        out = collect_per_residue_data(
            step4_dir=self.step4, processed_dir=self.processed,
        )
        rows = out["rows"]
        r2 = next(r for r in rows if r["residue_id"] == 2)
        self.assertEqual(r2["tool_binding_flags"]["fpocket"], True)
        self.assertEqual(r2["tool_scores"], {})

    def test_sample_ids_filter(self):
        for sid in ("s1", "s2", "s3"):
            _write_sample(self.processed, sid, length=2,
                          sequence="MK", gt=[1])
            _write_step4(self.step4, sid, [{
                "tool_id": "boltz2", "category": "A", "sample_id": sid,
                "success": True, "binding_protein_residues": [1],
                "per_residue_pae_score": {"1": 0.9},
            }])
        out = collect_per_residue_data(
            step4_dir=self.step4, processed_dir=self.processed,
            sample_ids=["s1", "s3"],
        )
        sids = {r["sample_id"] for r in out["rows"]}
        self.assertEqual(sids, {"s1", "s3"})
        self.assertEqual(out["n_samples_used"], 2)


if __name__ == "__main__":
    unittest.main()
