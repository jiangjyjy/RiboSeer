"""Shared RiboSeer configuration — MANDATORY-tool guardrail helpers.

The **MANDATORY tool guardrail** is the set of tools the headline pipeline
force-includes in every sample's MAESTRO selection (``K_sel = K_LLM ∪
MANDATORY_TOOLS``). Both the tool library and the mandatory core live in
``step3_tool_selection.tool_registry``; this module re-exports them so the
headline scripts keep importing ``MANDATORY_TOOLS`` / ``inject_mandatory``
from one place.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step5_fusion.features_15tool import (  # noqa: E402
    ALL_KNOWN_TOOLS, _canonical,
)
from step3_tool_selection.tool_registry import (  # noqa: E402
    MANDATORY_TOOLS as MANDATORY_TOOLS,
)

_LIB_INDEX = {t: i for i, t in enumerate(ALL_KNOWN_TOOLS)}


def inject_mandatory(tools: Iterable[str],
                     mandatory: Iterable[str] = MANDATORY_TOOLS) -> list[str]:
    """``tools ∪ mandatory``, canonicalised, filtered to the library, and
    returned in ``ALL_KNOWN_TOOLS`` order. Unknown ids are dropped; a
    mandatory tool absent from a sample's step4 still ends up "selected"
    here (the feature builder gates it to NaN downstream)."""
    merged = {_canonical(t) for t in tools} | {_canonical(t) for t in mandatory}
    merged = {t for t in merged if t in _LIB_INDEX}
    return sorted(merged, key=lambda t: _LIB_INDEX[t])
