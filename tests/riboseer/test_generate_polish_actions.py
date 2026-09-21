"""Mock tests for scripts/riboseer/generate_polish_actions.py."""
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

from scripts.riboseer import generate_polish_actions as pa  # noqa: E402


def _pred(tool_id, *, pae=None, conf=None, binding=None, success=True):
    return {"tool_id": tool_id, "category": "A", "sample_id": "x",
            "success": success, "per_residue_pae_score": pae or {},
            "per_residue_confidence": conf or {},
            "binding_protein_residues": binding or []}


def _write_sample(proc, sid, plen=8):
    d = proc / "samples"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sid}.json").write_text(json.dumps({
        "sample_id": sid,
        "protein": {"length": plen, "sequence": "A" * plen}}),
        encoding="utf-8")


def _write_step4(step4, sid, preds):
    step4.mkdir(parents=True, exist_ok=True)
    rec = {"sample_id": sid, "predictions": preds}
    (step4 / f"{sid}.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")


class TestFusedProbability(unittest.TestCase):
    def test_mean_of_tools(self):
        s4 = {"predictions": [
            _pred("boltz2", pae={1: 0.8, 2: 0.4}),
            _pred("equipnas", conf={1: 0.6, 2: 0.2})]}
        prob = pa.mean_fused_probability(s4)
        self.assertAlmostEqual(prob[1], (0.8 + 0.6) / 2)
        self.assertAlmostEqual(prob[2], (0.4 + 0.2) / 2)

    def test_fused_falls_back_to_mean_without_model(self):
        s4 = {"predictions": [_pred("boltz2", pae={1: 0.9})]}
        prob = pa.fused_probability(s4, {"protein": {"length": 1}}, None)
        self.assertAlmostEqual(prob[1], 0.9)

    def test_tool_agreement_summary(self):
        s4 = {"predictions": [
            _pred("boltz2", binding=[1, 2]),
            _pred("chai1", binding=[1])]}
        summary = pa.tool_agreement_summary(s4, [1, 2])
        self.assertIn("r1:2/2", summary)
        self.assertIn("r2:1/2", summary)


class TestPrompt(unittest.TestCase):
    def test_grouped_probability_bands(self):
        prob = {1: 0.92, 2: 0.45, 3: 0.1, 4: 0.7, 5: 0.55}
        high, boundary, high_txt, bnd_txt = pa.grouped_probability_lines(prob)
        self.assertEqual(high, [1])                 # only p>0.7
        self.assertEqual(boundary, [2, 4, 5])       # 0.3<=p<=0.7
        self.assertIn("1(0.92)", high_txt)
        self.assertIn("2(0.45)", bnd_txt)

    def test_per_tool_boundary_lines_support_count(self):
        s4 = {"predictions": [
            _pred("boltz2", pae={22: 0.6, 28: 0.1}),
            _pred("chai1", conf={22: 0.3, 28: 0.7}),
            _pred("equipnas", conf={22: 0.8, 28: 0.2})]}
        txt = pa.per_tool_boundary_lines(s4, [22, 28])
        self.assertIn("Residue 22", txt)
        self.assertIn("boltz2=0.6", txt)
        self.assertIn("→ 2/3 support", txt)         # 0.6,0.8 > 0.5
        self.assertIn("→ 1/3 support", txt)         # only chai1 0.7 > 0.5

    def test_prompt_has_bands_tools_and_confidence(self):
        prob = {1: 0.92, 2: 0.45}
        s4 = {"predictions": [_pred("boltz2", pae={2: 0.45})]}
        prompt = pa.build_polish_prompt(8, prob, s4, 0.7)
        self.assertIn("High-confidence", prompt)
        self.assertIn("Boundary residues", prompt)
        self.assertIn("Per-tool predictions for the boundary residues", prompt)
        self.assertIn("confidence", prompt)

    def test_llm_action_parses_confidence(self):
        from unittest import mock
        fake_resp = object()
        client = mock.Mock()
        client.call.return_value = fake_resp
        with mock.patch(
            "step2_target_char.llm_client.extract_content",
            lambda r: '{"action":"mask","residues":[2],"confidence":0.7,'
                      '"reasoning":"weak"}'), \
             mock.patch(
            "step2_target_char.llm_client.extract_json_object",
            lambda s: __import__("json").loads(s)):
            act = pa.llm_action(8, {1: 0.9, 2: 0.45},
                                {"predictions": []}, 0.5, client, {})
        self.assertEqual(act["action"], "mask")
        self.assertEqual(act["residues"], [2])
        self.assertAlmostEqual(act["confidence"], 0.7)


class TestGenerate(unittest.TestCase):
    def _fixture(self, td):
        proc, step4 = td / "proc", td / "s4"
        _write_sample(proc, "1abc_A_E", plen=8)
        # 5 strong binders + 1 weak → auto rule masks the weakest.
        preds = [_pred("boltz2",
                       pae={1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9, 5: 0.9, 6: 0.55},
                       binding=[1, 2, 3, 4, 5, 6])]
        _write_step4(step4, "1abc_A_E", preds)
        return proc, step4

    def test_auto_mode(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, step4 = self._fixture(td)
            out = td / "pol"
            stats = pa.generate(
                processed_dir=proc, sample_ids=["1abc_A_E"], step4_dir=step4,
                out_dir=out, mode="auto", enriched_model=None,
                step6_dir=None, config_path=None)
            self.assertEqual(stats["written"], 1)
            rec = json.loads((out / "1abc_A_E.json").read_text())
            self.assertEqual(rec["source"], "auto")
            self.assertEqual(rec["action"], "mask")
            self.assertIn(6, rec["residues"])       # weakest binder masked
            self.assertIn("target_residues", rec)

    def test_llm_mode(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, step4 = self._fixture(td)
            out = td / "pol"
            fake = {"action": "relocate", "residues": [1],
                    "target_residues": [3], "confidence": 0.6,
                    "reasoning": "llm"}
            with mock.patch.object(pa, "_build_llm_client",
                                   lambda c: (object(), {})), \
                 mock.patch.object(pa, "llm_action", lambda *a, **k: fake):
                pa.generate(
                    processed_dir=proc, sample_ids=["1abc_A_E"],
                    step4_dir=step4, out_dir=out, mode="llm",
                    enriched_model=None, step6_dir=None, config_path=None)
            rec = json.loads((out / "1abc_A_E.json").read_text())
            self.assertEqual(rec["source"], "llm")
            self.assertEqual(rec["action"], "relocate")
            self.assertEqual(rec["target_residues"], [3])
            self.assertAlmostEqual(rec["confidence"], 0.6)   # persisted

    def test_llm_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proc, step4 = self._fixture(td)
            out = td / "pol"
            with mock.patch.object(pa, "_build_llm_client",
                                   lambda c: (object(), {})), \
                 mock.patch.object(pa, "llm_action", lambda *a, **k: None):
                pa.generate(
                    processed_dir=proc, sample_ids=["1abc_A_E"],
                    step4_dir=step4, out_dir=out, mode="llm",
                    enriched_model=None, step6_dir=None, config_path=None)
            rec = json.loads((out / "1abc_A_E.json").read_text())
            self.assertEqual(rec["source"], "llm_fallback")
            self.assertEqual(rec["action"], "mask")   # auto rule

    def test_output_consumable_by_ablation_apply_polish(self):
        # The flat record must drive apply_polish (acts=[record]).
        from step7_iteration.polish_ops import apply_polish_to_probability
        rec = {"sample_id": "x", "action": "mask", "residues": [6],
               "target_residues": [], "confidence": None, "reasoning": ""}
        prob = {1: 0.9, 6: 0.55}
        out = apply_polish_to_probability(prob, rec)
        self.assertAlmostEqual(out[6], 0.55 * 0.2)   # soft suppress, default


if __name__ == "__main__":
    unittest.main()
