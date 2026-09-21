"""Tool registry — the 15 tools of the MAESTRO library (paper Table 1).

This module is the single source of truth for the tool library: tool IDs,
category, per-tool timeout, and the MANDATORY core. Every other module
(step 3 schemas, step 4 dispatch, the feature builder, the experiment
scripts) derives its tool list from here.

Four categories, ordered by typical runtime:

  C (seconds)  — RNA-binding residue prediction (sequence / structure → per-residue prob)
  B (minutes)  — protein surface pocket detection (structure → pocket location)
  D (min-hrs)  — molecular docking (RNA + protein structure → docked poses)
  A (hours)    — end-to-end complex structure prediction (sequences → 3D complex)

Default cascade: C → B → D → A (fast first; early-stop if intermediate
results are already sufficient).

Execution modes
---------------
``execution`` records how a tool is obtained, which matters for
reproducibility:

  ``"local"`` — RELAY runs it as a subprocess through its step-4 adapter.
  ``"web"``   — the tool is only reachable through its authors' web server
                (AlphaFold 3 Server, BindWeb, RNABindRPlus, BindUP). Its
                adapter executes nothing: it writes the submission payload
                and reads back results the user has downloaded, from
                ``data/external/<tool_id>/``. See the "Tools that require
                manual web submission" section of the README.

The MANDATORY core is the 7-tool set chosen by forward greedy search on the
training set; MAESTRO force-includes it in every sample's plan
(``K_sel = K_LLM ∪ MANDATORY_TOOLS``), which is the guardrail evaluated in
paper Table 11.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ToolInfo:
    name: str
    tool_id: str
    category: str                        # "A" / "B" / "C" / "D"
    description: str
    input_requirements: list[str]        # e.g. ["protein_sequence", "rna_sequence"]
    output_type: str                     # e.g. "complex_structure", "binding_residues"
    typical_runtime: str                 # "seconds" / "minutes" / "hours"
    requires_structure: bool             # needs a pre-existing 3D structure as input
    timeout_seconds: int                 # per-sample budget (paper Table 1)
    execution: str = "local"             # "local" (RELAY runs it) / "web" (manual submission)
    mandatory: bool = False              # force-included by MAESTRO (paper Table 1, ✓M)
    available: bool = True               # kept for API compatibility; all 15 ship enabled


# ---------- registry --------------------------------------------------------

_TOOLS: list[ToolInfo] = [
    # ---- Category A — end-to-end complex structure prediction (5) ----------
    ToolInfo(
        name="Boltz-2", tool_id="boltz2", category="A",
        description="Open-source biomolecular structure prediction",
        input_requirements=["protein_sequence", "rna_sequence"],
        output_type="complex_structure",
        typical_runtime="hours",
        requires_structure=False,
        timeout_seconds=900,
        mandatory=True,
    ),
    ToolInfo(
        name="Chai-1", tool_id="chai1", category="A",
        description="Multi-modal complex structure prediction; "
                    "single-sequence inference (no MSA required)",
        input_requirements=["protein_sequence", "rna_sequence"],
        output_type="complex_structure",
        typical_runtime="minutes",
        requires_structure=False,
        timeout_seconds=900,
        mandatory=True,
    ),
    ToolInfo(
        name="RoseTTAFold2NA", tool_id="rosettafold2na", category="A",
        description="RNA-protein specialised structure prediction; "
                    "single-sequence inference (MSA search skipped)",
        input_requirements=["protein_sequence", "rna_sequence"],
        output_type="complex_structure",
        typical_runtime="minutes",
        requires_structure=False,
        timeout_seconds=600,
    ),
    ToolInfo(
        name="RoseTTAFold-All-Atom", tool_id="rfaa", category="A",
        description="All-atom general-purpose structure prediction; "
                    "single-sequence inference (no MSA / template search)",
        input_requirements=["protein_sequence", "rna_sequence"],
        output_type="complex_structure",
        typical_runtime="minutes",
        requires_structure=False,
        timeout_seconds=1200,
    ),
    ToolInfo(
        name="AlphaFold 3", tool_id="alphafold3", category="A",
        description="Full-atom complex structure prediction using deep learning "
                    "with MSA and templates",
        input_requirements=["protein_sequence", "rna_sequence", "msa"],
        output_type="complex_structure",
        typical_runtime="hours",
        requires_structure=False,
        timeout_seconds=900,
        execution="web",
    ),
    # ---- Category B — protein surface pocket detection (3) ----------------
    ToolInfo(
        name="P2Rank", tool_id="p2rank", category="B",
        description="ML-based ligand-binding pocket prediction on the protein surface",
        input_requirements=["protein_structure"],
        output_type="pocket_locations",
        typical_runtime="minutes",
        requires_structure=True,
        timeout_seconds=60,
    ),
    ToolInfo(
        name="Fpocket", tool_id="fpocket", category="B",
        description="Geometry-based pocket detection using Voronoi tessellation; "
                    "complementary signal to P2Rank's ML-based ranking",
        input_requirements=["protein_structure"],
        output_type="pocket_locations",
        typical_runtime="seconds",
        requires_structure=True,
        timeout_seconds=60,
    ),
    ToolInfo(
        name="DeepPocket", tool_id="deeppocket", category="B",
        description="3D CNN pocket segmentation on the protein surface; ranks the "
                    "fpocket candidates with a classification head",
        input_requirements=["protein_structure"],
        output_type="pocket_locations",
        typical_runtime="minutes",
        requires_structure=True,
        timeout_seconds=300,
        mandatory=True,
    ),
    # ---- Category C — RNA-binding residue prediction (5) ------------------
    ToolInfo(
        name="EquiPNAS", tool_id="equipnas", category="C",
        description="E(3) equivariant GNN + protein language model for "
                    "RNA-binding residue prediction",
        input_requirements=["protein_structure"],
        output_type="binding_residues",
        typical_runtime="minutes",
        requires_structure=True,
        timeout_seconds=300,
        mandatory=True,
    ),
    ToolInfo(
        name="NucleicNet", tool_id="nucleicnet", category="C",
        description="Surface voxelisation and classification of "
                    "nucleic-acid-binding sites (SXPR head)",
        input_requirements=["protein_structure"],
        output_type="binding_residues",
        typical_runtime="minutes",
        requires_structure=True,
        timeout_seconds=600,
        mandatory=True,
    ),
    ToolInfo(
        name="GraphBind", tool_id="graphbind", category="C",
        description="Hierarchical graph neural network for per-residue "
                    "RNA-binding probability prediction",
        input_requirements=["protein_structure"],
        output_type="binding_residues",
        typical_runtime="seconds",
        requires_structure=True,
        timeout_seconds=600,
    ),
    ToolInfo(
        name="RNABindRPlus", tool_id="rnabindrplus", category="C",
        description="Sequence + homology-based per-residue RNA-binding prediction",
        input_requirements=["protein_sequence"],
        output_type="binding_residues",
        typical_runtime="seconds",
        requires_structure=False,
        timeout_seconds=300,
        execution="web",
        mandatory=True,
    ),
    ToolInfo(
        name="BindUP", tool_id="bindup", category="C",
        description="Non-homology sequence-based per-residue RNA-binding prediction",
        input_requirements=["protein_sequence"],
        output_type="binding_residues",
        typical_runtime="seconds",
        requires_structure=False,
        timeout_seconds=300,
        execution="web",
    ),
    # ---- Category D — molecular docking (2) -------------------------------
    ToolInfo(
        name="HDOCK", tool_id="hdock", category="D",
        description="Hybrid rigid-body docking for RNA-protein complex pose "
                    "generation and scoring (HDOCKlite)",
        input_requirements=["protein_structure", "rna_structure"],
        output_type="docked_poses",
        typical_runtime="minutes",
        requires_structure=True,
        timeout_seconds=600,
        mandatory=True,
    ),
    ToolInfo(
        name="HADDOCK 3", tool_id="haddock3", category="D",
        description="Data-driven protein-RNA docking — runs rigidbody → flexref "
                    "→ emref on a protein + RNA PDB pair and outputs a ranked "
                    "complex with interface residues",
        input_requirements=["protein_structure", "rna_structure"],
        output_type="docked_complex",
        typical_runtime="minutes",
        requires_structure=True,
        timeout_seconds=600,
    ),
]

_BY_ID: dict[str, ToolInfo] = {t.tool_id: t for t in _TOOLS}
_BY_CAT: dict[str, list[ToolInfo]] = {}
for _t in _TOOLS:
    _BY_CAT.setdefault(_t.category, []).append(_t)

# ALL_TOOL_IDS / TOOL_COUNT reflect the *available* tools (what schemas and
# prompts validate against).
ALL_TOOL_IDS: frozenset[str] = frozenset(t.tool_id for t in _TOOLS if t.available)
TOOL_COUNT = sum(1 for t in _TOOLS if t.available)

# The 7-tool MANDATORY core (paper Table 1, "Mandatory" column): force-included
# in every MAESTRO plan, in registry order.
MANDATORY_TOOLS: tuple[str, ...] = tuple(t.tool_id for t in _TOOLS if t.mandatory)

# Tools reachable only through a manual web submission; their adapters ingest
# results from data/external/<tool_id>/ instead of executing anything.
WEB_SUBMISSION_TOOLS: tuple[str, ...] = tuple(
    t.tool_id for t in _TOOLS if t.execution == "web")

# ---------- public API ------------------------------------------------------


def get_tool(tool_id: str) -> ToolInfo:
    """Raise KeyError if `tool_id` not in registry."""
    return _BY_ID[tool_id]


def is_available(tool_id: str) -> bool:
    """True iff ``tool_id`` exists in the registry AND is enabled.

    Unknown ids return ``False`` (not a KeyError) so callers can use this as
    a guard — e.g. tool_selector only force-adds a mandatory tool when
    ``is_available`` is True.
    """
    info = _BY_ID.get(tool_id)
    return bool(info and info.available)


def is_web_submission(tool_id: str) -> bool:
    """True iff the tool needs a manual web submission (no local execution)."""
    return tool_id in WEB_SUBMISSION_TOOLS


def get_timeout(tool_id: str, default: Optional[int] = None) -> Optional[int]:
    """Per-sample timeout in seconds (paper Table 1)."""
    info = _BY_ID.get(tool_id)
    if info is None:
        return default
    return info.timeout_seconds


def get_tools_by_category(
    category: str, *, include_unavailable: bool = False,
) -> list[ToolInfo]:
    tools = _BY_CAT.get(category, [])
    if include_unavailable:
        return list(tools)
    return [t for t in tools if t.available]


def get_all_tools(*, include_unavailable: bool = False) -> list[ToolInfo]:
    if include_unavailable:
        return list(_TOOLS)
    return [t for t in _TOOLS if t.available]


def get_all_tool_ids(*, include_unavailable: bool = False) -> list[str]:
    return [t.tool_id for t in get_all_tools(include_unavailable=include_unavailable)]


def format_tool_descriptions(*, include_unavailable: bool = False) -> str:
    """Render tools as a prompt-embeddable text block.

    Tools are listed in cascade order (C → B → D → A) so the cheapest
    categories come first; an entry notes when the tool needs a manual web
    submission.
    """
    cascade_order = ["C", "B", "D", "A"]
    cat_labels = {
        "A": "Category A — End-to-end complex structure prediction (hours)",
        "B": "Category B — Protein surface pocket detection (minutes)",
        "C": "Category C — RNA-binding residue prediction (seconds)",
        "D": "Category D — Molecular docking (minutes–hours)",
    }
    lines: list[str] = []
    for cat in cascade_order:
        tools = get_tools_by_category(cat, include_unavailable=include_unavailable)
        if not tools:
            continue
        lines.append(f"## {cat_labels[cat]}")
        for t in tools:
            reqs = ", ".join(t.input_requirements)
            struct_note = " [requires 3D structure]" if t.requires_structure else " [sequence only]"
            web_note = " [web submission]" if t.execution == "web" else ""
            lines.append(
                f"- **{t.name}** (`{t.tool_id}`): {t.description}. "
                f"Input: {reqs}{struct_note}{web_note}. Output: {t.output_type}."
            )
        lines.append("")
    return "\n".join(lines)
