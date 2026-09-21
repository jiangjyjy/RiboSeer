"""Mock tests for scripts/riboseer/generate_polish_iterative.py."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.riboseer import generate_polish_iterative as gpi  # noqa: E402


def _accept(*a, **k):
    return {"action": "accept", "residues": [], "target_residues": [],
            "confidence": None, "reasoning": "ok"}


class TestIterateAuto(unittest.TestCase):
    def test_auto_converges_to_accept(self):
        # 10 binders → auto masks the weakest 25% each round; the binding set
        # shrinks until <4 → accept. Verifies the loop terminates.
        base = {r: 0.6 for r in range(1, 11)}
        rounds = gpi.iterate_sample(
            plen=10, base_prob=base, step4_data={}, verdict=None,
            mode="auto", client=None, config={}, max_rounds=20)
        self.assertGreaterEqual(len(rounds), 2)
        self.assertEqual(rounds[-1]["action"], "accept")
        self.assertTrue(all(r["source"] == "auto" for r in rounds))
        # round numbers are 1-based and contiguous
        self.assertEqual([r["round"] for r in rounds],
                         list(range(1, len(rounds) + 1)))

    def test_auto_accept_when_too_few_binders(self):
        base = {1: 0.9, 2: 0.1, 3: 0.1}      # only 1 binder → immediate accept
        rounds = gpi.iterate_sample(
            plen=3, base_prob=base, step4_data={}, verdict=None,
            mode="auto", client=None, config={}, max_rounds=5)
        self.assertEqual(len(rounds), 1)
        self.assertEqual(rounds[0]["action"], "accept")


class TestIterateLLM(unittest.TestCase):
    def test_llm_accept_first_round(self):
        with mock.patch.object(gpi, "llm_action", _accept):
            rounds = gpi.iterate_sample(
                plen=10, base_prob={1: 0.9, 2: 0.8}, step4_data={},
                verdict=None, mode="llm", client=object(), config={},
                max_rounds=5)
        self.assertEqual(len(rounds), 1)
        self.assertEqual(rounds[0]["action"], "accept")
        self.assertEqual(rounds[0]["source"], "llm")

    def test_llm_mask_then_accept(self):
        calls = {"n": 0}

        def fake(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"action": "mask", "residues": [1],
                        "target_residues": [], "confidence": 0.7,
                        "reasoning": "fp"}
            return _accept()

        with mock.patch.object(gpi, "llm_action", fake):
            rounds = gpi.iterate_sample(
                plen=10, base_prob={1: 0.9, 2: 0.8, 3: 0.2}, step4_data={},
                verdict=None, mode="llm", client=object(), config={},
                max_rounds=5)
        self.assertEqual([r["action"] for r in rounds], ["mask", "accept"])
        self.assertEqual(rounds[0]["residues"], [1])
        self.assertEqual(rounds[0]["confidence"], 0.7)

    def test_llm_failure_falls_back_to_auto(self):
        with mock.patch.object(gpi, "llm_action", lambda *a, **k: None):
            rounds = gpi.iterate_sample(
                plen=10, base_prob={r: 0.6 for r in range(1, 11)},
                step4_data={}, verdict=None, mode="llm", client=object(),
                config={}, max_rounds=3)
        self.assertEqual(rounds[0]["source"], "llm_fallback")
        self.assertLessEqual(len(rounds), 3)             # capped by max_rounds


class TestGenerateAndCli(unittest.TestCase):
    def _fixture(self, td):
        proc, step4 = td / "proc", td / "s4"
        (proc / "samples").mkdir(parents=True)
        (proc / "samples" / "1abc_A_E.json").write_text(json.dumps({
            "sample_id": "1abc_A_E",
            "protein": {"length": 10, "sequence": "A" * 10},
            "rna": {"length": 5, "sequence": "G" * 5}}), encoding="utf-8")
        step4.mkdir(parents=True)
        rec = {"sample_id": "1abc_A_E", "predictions": [{
            "tool_id": "boltz2", "category": "A", "success": True,
            "per_residue_pae_score": {str(r): 0.6 for r in range(1, 11)},
            "per_residue_confidence": {}}]}
        (step4 / "1abc_A_E.jsonl").write_text(json.dumps(rec) + "\n",
                                              encoding="utf-8")
        return proc, step4

    def test_generate_auto_writes_schema(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, step4 = self._fixture(td)
            out = td / "iter"
            stats = gpi.generate(
                processed_dir=proc, sample_ids=["1abc_A_E"], step4_dir=step4,
                out_dir=out, mode="auto", enriched_model=None,
                step6_dir=None, config_path=None, max_rounds=5)
            self.assertEqual(stats["written"], 1)
            rec = json.loads((out / "1abc_A_E.json").read_text())
            self.assertIn("rounds", rec)
            self.assertEqual(rec["total_rounds"], len(rec["rounds"]))
            self.assertEqual(rec["max_rounds"], 5)
            self.assertTrue(1 <= len(rec["rounds"]) <= 5)
            for i, rd in enumerate(rec["rounds"], start=1):
                self.assertEqual(rd["round"], i)
                self.assertIn(rd["action"],
                              {"mask", "extend", "relocate", "accept"})
                self.assertEqual(rd["source"], "auto")

    def test_cli_rejects_bad_max_rounds(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, step4 = self._fixture(td)
            rc = gpi.main(["--processed-dir", str(proc),
                           "--sample-list", str(td / "missing.txt"),
                           "--step4-dir", str(step4),
                           "--out-dir", str(td / "o"),
                           "--max-rounds", "0"])
            self.assertEqual(rc, 1)

    def test_cli_missing_dirs(self):
        rc = gpi.main(["--processed-dir", "/no/such",
                       "--sample-list", "/no/list.txt",
                       "--step4-dir", "/no/s4", "--out-dir", "/tmp/o"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
