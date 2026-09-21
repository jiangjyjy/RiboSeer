"""Unit tests for `prompts.py` — pure-string rendering, no API calls."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step2_target_char.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    build_messages, build_error_correction_messages,
    format_features_for_prompt, estimate_prompt_tokens,
    _fmt_opt, _fmt_top3,
)
from step2_target_char.schemas import (  # noqa: E402
    StructureComposition, TargetFeatures,
)


def _full_features(**overrides) -> TargetFeatures:
    base = dict(
        sample_id="7z20_w_a",
        rna_length=119, rna_gc_content=0.6525,
        rna_ss_status="done",
        rna_structure_composition=StructureComposition(
            paired_frac=0.55, hairpin_frac=0.12, interior_frac=0.10,
            multiloop_frac=0.08, external_frac=0.15,
        ),
        rna_has_modification=False,
        protein_length=94, protein_pI=9.599, protein_mean_bfactor=81.902,
        protein_top3_aa=[("A", 0.117), ("K", 0.117), ("V", 0.085)],
        n_binding_protein_residues=21, n_binding_rna_nucleotides=16,
        interface_ratio_protein=0.2234, interface_ratio_rna=0.1345,
        quality_tier="strict", resolution=2.29,
        experimental_method="ELECTRON MICROSCOPY",
    )
    base.update(overrides)
    return TargetFeatures(**base)


def _null_features() -> TargetFeatures:
    """Mimic an all-X protein + poly-inosine RNA + stage-1.3b-missing sample."""
    return TargetFeatures(
        sample_id="edge_A_B",
        rna_length=50, rna_gc_content=None,
        rna_ss_status=None, rna_structure_composition=None,
        rna_has_modification=True,
        protein_length=60, protein_pI=None, protein_mean_bfactor=None,
        protein_top3_aa=None,
        n_binding_protein_residues=5, n_binding_rna_nucleotides=3,
        interface_ratio_protein=0.0833, interface_ratio_rna=0.06,
        quality_tier="low", resolution=None, experimental_method=None,
    )


class TestFmtHelpers(unittest.TestCase):
    def test_fmt_opt_none(self):
        self.assertIn("unknown", _fmt_opt(None))

    def test_fmt_opt_value(self):
        self.assertEqual(_fmt_opt(3.14, "{:.2f}"), "3.14")

    def test_fmt_opt_custom_unknown(self):
        s = _fmt_opt(None, unknown="no data")
        self.assertIn("no data", s)

    def test_fmt_top3_empty(self):
        self.assertIn("unknown", _fmt_top3(None))
        self.assertIn("unknown", _fmt_top3([]))

    def test_fmt_top3_values(self):
        s = _fmt_top3([("K", 0.15), ("R", 0.10)])
        self.assertIn("K=0.150", s)
        self.assertIn("R=0.100", s)


class TestFormatFeaturesForPrompt(unittest.TestCase):
    def test_contains_all_section_headers(self):
        out = format_features_for_prompt(_full_features())
        for header in ("# RNA", "# Protein", "# Interaction", "# Data provenance"):
            self.assertIn(header, out)

    def test_nulls_rendered_explicitly(self):
        out = format_features_for_prompt(_null_features())
        self.assertIn("(unknown", out)
        # pI null uses the specialized "all-X" marker we embed in the formatter
        self.assertIn("all-X", out)
        # structure composition null mentions stage 1.3b explicitly
        self.assertIn("1.3b", out)

    def test_numeric_formatting(self):
        out = format_features_for_prompt(_full_features())
        # GC content 0.6525 may round to 0.652 or 0.653 per banker's rounding
        self.assertTrue("0.652" in out or "0.653" in out)
        self.assertIn("9.60", out)     # pI at .2f
        self.assertIn("paired=0.55", out)
        self.assertIn("hairpin=0.12", out)
        self.assertIn("2.29 Å", out)


class TestBuildMessages(unittest.TestCase):
    def test_shape(self):
        msgs = build_messages(_full_features())
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")

    def test_system_contains_all_enum_values(self):
        msgs = build_messages(_full_features())
        system = msgs[0]["content"]
        for v in ["RRM", "KH", "zinc_finger", "dsRBD", "PUF", "DEAD_box",
                  "multi_domain", "novel_fold"]:
            self.assertIn(f'"{v}"', system)
        for v in ["single_stranded", "stem_loop", "internal_loop",
                  "junction", "g_quadruplex", "unstructured"]:
            self.assertIn(f'"{v}"', system)

    def test_system_declares_json_schema(self):
        system = build_messages(_full_features())[0]["content"]
        for key in ("analysis", "protein_domain", "rna_structure",
                    "category", "confidence", "notes"):
            self.assertIn(f'"{key}"', system)

    def test_history_summary_injected(self):
        msgs = build_messages(
            _full_features(),
            history_summary="Past 50 predictions: 40% RRM_x_stem_loop, avg conf 0.72",
        )
        self.assertIn("Historical context", msgs[0]["content"])
        self.assertIn("40% RRM_x_stem_loop", msgs[0]["content"])

    def test_no_history_no_leak(self):
        msgs = build_messages(_full_features())
        self.assertNotIn("Historical context", msgs[0]["content"])

    def test_token_budget_under_2000(self):
        """System + user should stay under the stated 2000-token budget."""
        msgs = build_messages(_full_features())
        est = estimate_prompt_tokens(msgs)
        self.assertLess(est, 2000, f"prompt estimated at {est} tokens")


class TestErrorCorrection(unittest.TestCase):
    def test_appends_two_turns(self):
        prev = build_messages(_full_features())
        corrected = build_error_correction_messages(
            prev, bad_output='{"garbage": true}',
            error_message="category 'X_x_Y' does not match 'RRM_x_stem_loop'",
        )
        self.assertEqual(len(corrected), len(prev) + 2)
        self.assertEqual(corrected[-2]["role"], "assistant")
        self.assertEqual(corrected[-1]["role"], "user")
        self.assertIn("Validation error", corrected[-1]["content"])
        self.assertIn("does not match", corrected[-1]["content"])

    def test_prev_messages_unmutated(self):
        """build_error_correction_messages must not mutate the original list."""
        prev = build_messages(_full_features())
        prev_len_before = len(prev)
        _ = build_error_correction_messages(prev, "bad", "err")
        self.assertEqual(len(prev), prev_len_before)


class TestSystemPromptContent(unittest.TestCase):
    """Guardrail: these key strings must be present for the prompt to work."""
    def test_json_mode_reminder(self):
        self.assertIn("JSON", SYSTEM_PROMPT)

    def test_english_requirement(self):
        self.assertIn("English", SYSTEM_PROMPT)

    def test_category_format(self):
        self.assertIn("_x_", SYSTEM_PROMPT)

    def test_missing_info_instructions(self):
        self.assertIn("unknown", SYSTEM_PROMPT.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
