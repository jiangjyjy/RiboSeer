"""Unit tests for tool_registry.py."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step3_tool_selection.tool_registry import (  # noqa: E402
    ToolInfo, get_tool, get_tools_by_category, get_all_tools,
    get_all_tool_ids, format_tool_descriptions, get_timeout, is_web_submission,
    ALL_TOOL_IDS, TOOL_COUNT, MANDATORY_TOOLS, WEB_SUBMISSION_TOOLS,
)


# The 15-tool library, in paper Table 1 order.
LIBRARY = ("boltz2", "chai1", "rosettafold2na", "rfaa", "alphafold3",
           "p2rank", "fpocket", "deeppocket",
           "equipnas", "nucleicnet", "graphbind", "rnabindrplus", "bindup",
           "hdock", "haddock3")

# Paper Table 1 "Mandatory" column (✓M) — the 7-tool core.
MANDATORY = {"boltz2", "chai1", "deeppocket", "equipnas",
             "nucleicnet", "rnabindrplus", "hdock"}

# Tools with no local install: the user submits through the authors' web
# server and drops the downloaded results under data/external/<tool_id>/.
WEB_ONLY = {"alphafold3", "rnabindrplus", "bindup"}

# Per-tool timeout column of paper Table 1.
TIMEOUTS = {
    "boltz2": 900, "chai1": 900, "rosettafold2na": 600, "rfaa": 1200,
    "alphafold3": 900, "p2rank": 60, "fpocket": 60, "deeppocket": 300,
    "equipnas": 300, "nucleicnet": 600, "graphbind": 600,
    "rnabindrplus": 300, "bindup": 300, "hdock": 600, "haddock3": 600,
}


class TestToolCount(unittest.TestCase):
    def test_registry_holds_the_15_tools_of_table_1(self):
        self.assertEqual(TOOL_COUNT, 15)
        self.assertEqual(len(ALL_TOOL_IDS), 15)
        self.assertEqual(len(get_all_tools()), 15)
        self.assertEqual(tuple(get_all_tool_ids()), LIBRARY)

    def test_no_extra_tool_beyond_the_paper(self):
        all_ids = set(get_all_tool_ids(include_unavailable=True))
        self.assertEqual(all_ids, set(LIBRARY))

    def test_category_counts(self):
        self.assertEqual(len(get_tools_by_category("A")), 5)
        self.assertEqual(len(get_tools_by_category("B")), 3)
        self.assertEqual(len(get_tools_by_category("C")), 5)
        self.assertEqual(len(get_tools_by_category("D")), 2)

    def test_empty_category(self):
        self.assertEqual(get_tools_by_category("Z"), [])
        self.assertEqual(get_tools_by_category("Z", include_unavailable=True), [])


class TestMandatoryCore(unittest.TestCase):
    def test_seven_tool_core(self):
        self.assertEqual(set(MANDATORY_TOOLS), MANDATORY)
        self.assertEqual(len(MANDATORY_TOOLS), 7)

    def test_mandatory_matches_registry_flags(self):
        flagged = {t.tool_id for t in get_all_tools() if t.mandatory}
        self.assertEqual(flagged, MANDATORY)

    def test_mandatory_is_in_library_order(self):
        # MAESTRO appends the core in registry order, so MANDATORY_TOOLS
        # must follow LIBRARY.
        expected = tuple(t for t in LIBRARY if t in MANDATORY)
        self.assertEqual(MANDATORY_TOOLS, expected)


class TestWebSubmissionTools(unittest.TestCase):
    def test_web_only_set(self):
        self.assertEqual(set(WEB_SUBMISSION_TOOLS), WEB_ONLY)

    def test_is_web_submission(self):
        self.assertTrue(is_web_submission("alphafold3"))
        self.assertFalse(is_web_submission("boltz2"))
        self.assertFalse(is_web_submission("nonexistent_tool"))


class TestGetTool(unittest.TestCase):
    def test_tool_metadata(self):
        t = get_tool("boltz2")
        self.assertEqual(t.name, "Boltz-2")
        self.assertEqual(t.category, "A")
        self.assertTrue(t.mandatory)
        self.assertEqual(t.execution, "local")

    def test_web_tool_metadata(self):
        t = get_tool("alphafold3")
        self.assertEqual(t.name, "AlphaFold 3")
        self.assertEqual(t.category, "A")
        self.assertFalse(t.mandatory)
        self.assertEqual(t.execution, "web")

    def test_missing(self):
        with self.assertRaises(KeyError):
            get_tool("nonexistent_tool")

    def test_structure_requirement(self):
        self.assertTrue(get_tool("p2rank").requires_structure)
        self.assertTrue(get_tool("equipnas").requires_structure)
        self.assertFalse(get_tool("rnabindrplus").requires_structure)
        self.assertFalse(get_tool("bindup").requires_structure)

    def test_timeouts_match_table_1(self):
        for tool_id, expected in TIMEOUTS.items():
            self.assertEqual(get_tool(tool_id).timeout_seconds, expected,
                             f"{tool_id} timeout")
            self.assertEqual(get_timeout(tool_id), expected)

    def test_all_ids_unique(self):
        ids = get_all_tool_ids(include_unavailable=True)
        self.assertEqual(len(ids), len(set(ids)))


class TestFormatDescriptions(unittest.TestCase):
    def test_lists_every_tool(self):
        text = format_tool_descriptions()
        for tool_id in LIBRARY:
            self.assertIn(tool_id, text)

    def test_all_category_headers(self):
        text = format_tool_descriptions()
        for cat in ("A", "B", "C", "D"):
            self.assertIn(f"Category {cat}", text)

    def test_cascade_order_C_first(self):
        text = format_tool_descriptions()
        pos_c = text.index("Category C")
        pos_b = text.index("Category B")
        pos_d = text.index("Category D")
        pos_a = text.index("Category A")
        self.assertLess(pos_c, pos_b)
        self.assertLess(pos_b, pos_d)
        self.assertLess(pos_d, pos_a)

    def test_structure_annotation(self):
        text = format_tool_descriptions()
        self.assertIn("[requires 3D structure]", text)
        self.assertIn("[sequence only]", text)

    def test_web_tools_are_flagged(self):
        text = format_tool_descriptions()
        self.assertIn("[web submission]", text)


class TestToolInfoFrozen(unittest.TestCase):
    def test_immutable(self):
        t = get_tool("hdock")
        with self.assertRaises(AttributeError):
            t.name = "other"


class TestAdapterCoverage(unittest.TestCase):
    """Every registered tool must be dispatchable by RELAY."""

    def test_every_tool_has_an_adapter(self):
        from step4_tool_adapters.run import ADAPTER_REGISTRY
        self.assertEqual(set(ADAPTER_REGISTRY), set(LIBRARY))

    def test_adapter_tool_id_and_category_match_registry(self):
        from step4_tool_adapters.run import ADAPTER_REGISTRY
        for tool_id, cls in ADAPTER_REGISTRY.items():
            adapter = cls()
            self.assertEqual(adapter.tool_id, tool_id)
            self.assertEqual(adapter.category, get_tool(tool_id).category)
            self.assertIn(adapter.category, {"A", "B", "C", "D"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
