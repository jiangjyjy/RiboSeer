"""Tests for prompt_style.py and CoT/temperature threading in the three
LLM generators (Table 14 prompt ablation)."""
import argparse
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.riboseer import prompt_style as ps  # noqa: E402
from scripts.riboseer import generate_scope_profiles as gsp  # noqa: E402
from scripts.riboseer import generate_maestro_selections as gm  # noqa: E402
from scripts.riboseer import generate_polish_actions as gpa  # noqa: E402


class FakeClient:
    """Records the temperature passed to .call and returns a fixed
    OpenAI-style response body."""

    def __init__(self, content):
        self.content = content
        self.temps = []

    def call(self, messages, temperature=None):
        self.temps.append(temperature)
        return {"choices": [{"message": {"content": self.content}}]}


class TestPromptStyle(unittest.TestCase):
    def test_cot_clause_on_off(self):
        on = ps.cot_clause(True)
        off = ps.cot_clause(False)
        self.assertIn("step by step", on)
        self.assertNotIn("step by step", off)
        self.assertIn("ONLY", off)

    def test_with_cot_appends(self):
        self.assertTrue(ps.with_cot("BODY", True).startswith("BODY"))
        self.assertIn("step by step", ps.with_cot("BODY", True))
        self.assertNotIn("step by step", ps.with_cot("BODY", False))

    def test_resolve_temperature_priority(self):
        self.assertEqual(ps.resolve_temperature({}, 0.0), 0.0)       # override
        self.assertEqual(ps.resolve_temperature(
            {"api": {"temperature": 0.9}}), 0.9)                     # config
        self.assertEqual(ps.resolve_temperature({}, default=0.3), 0.3)
        self.assertEqual(ps.resolve_temperature(None, default=0.3), 0.3)

    def test_resolve_temperature_override_zero_is_used(self):
        # 0.0 must NOT be treated as "unset" — explicit greedy-deterministic.
        self.assertEqual(
            ps.resolve_temperature({"api": {"temperature": 0.9}}, 0.0), 0.0)

    def test_resolve_temperature_tolerates_garbage(self):
        self.assertEqual(ps.resolve_temperature({"api": "nope"},
                                                default=0.5), 0.5)
        self.assertEqual(ps.resolve_temperature({}, "x", default=0.5), 0.5)

    def test_add_args_defaults(self):
        p = argparse.ArgumentParser()
        ps.add_prompt_style_args(p)
        a = p.parse_args([])
        self.assertTrue(a.cot)
        self.assertEqual(a.temperature, 0.7)
        b = p.parse_args(["--no-cot", "--temperature", "0.0"])
        self.assertFalse(b.cot)
        self.assertEqual(b.temperature, 0.0)


class TestCotInPrompts(unittest.TestCase):
    def _scope(self, cot):
        return gsp.build_scope_prompt(
            {"protein": {"length": 10, "sequence": "A" * 10},
             "rna": {"length": 5, "sequence": "G" * 5}}, cot=cot)

    def _maestro(self, cot):
        return gm.build_maestro_prompt(
            {"protein_family": "RRM"}, ["boltz2"], {"boltz2": 0.5},
            6, 80, 40, cot=cot)

    def _polish(self, cot):
        return gpa.build_polish_prompt(
            80, {1: 0.9, 2: 0.4}, {"predictions": []}, None, cot=cot)

    def test_scope_cot_toggle(self):
        self.assertIn("step by step", self._scope(True))
        self.assertNotIn("step by step", self._scope(False))

    def test_maestro_cot_toggle(self):
        self.assertIn("step by step", self._maestro(True))
        self.assertNotIn("step by step", self._maestro(False))
        # the anti-bias guarantees still hold under CoT
        self.assertIn("strictly by UCB", self._maestro(True))

    def test_polish_cot_toggle(self):
        self.assertIn("step by step", self._polish(True))
        self.assertNotIn("step by step", self._polish(False))


class TestTemperatureThreading(unittest.TestCase):
    def test_scope_passes_temperature(self):
        c = FakeClient('{"protein_family": "Novel", "rna_context": '
                       '"unstructured", "difficulty": "Easy", '
                       '"confidence": 0.5}')
        prof = gsp.llm_profile_for_sample(
            "x", {"protein": {"length": 5, "sequence": "AAAAA"}}, c, {},
            temperature=0.0, cot=False)
        self.assertEqual(prof["source"], "llm")
        self.assertEqual(c.temps, [0.0])

    def test_maestro_passes_temperature(self):
        c = FakeClient('{"selected_tools": ["boltz2"], "reasoning": "r"}')
        res = gm.llm_select("x", {"protein_family": "RRM"}, ["boltz2"],
                            {"boltz2": 0.5}, 80, 40, c, {}, 6,
                            temperature=1.0, cot=True)
        self.assertEqual(res[0], ["boltz2"])
        self.assertEqual(c.temps, [1.0])

    def test_polish_passes_temperature(self):
        c = FakeClient('{"action": "accept", "residues": [], '
                       '"confidence": 0.5}')
        act = gpa.llm_action(80, {1: 0.9}, {"predictions": []}, None, c, {},
                             temperature=0.7, cot=False)
        self.assertEqual(act["action"], "accept")
        self.assertEqual(c.temps, [0.7])


if __name__ == "__main__":
    unittest.main()
