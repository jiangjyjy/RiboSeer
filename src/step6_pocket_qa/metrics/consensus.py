"""q4 — cross_tool_consensus (formula 21 in the paper).

Idea
----
Multiple independent tools predicting the same binding residue is
stronger evidence than a single tool's claim. We measure that two
ways:

  1. **Vote ratio**: for each residue in the fused composite set,
     count how many active tools also predicted it; average the
     ``votes / n_tools`` ratio over the composite set.
  2. **Pairwise Jaccard**: for every pair of active tools, compute
     ``|A ∩ B| / |A ∪ B|`` over their full predicted binding sets;
     average across pairs.

The two are mixed with config weights (``vote_weight`` /
``jaccard_weight``, default 0.6 / 0.4 per the spec). Vote ratio
specifically rewards "the tools agree on what the composite reports";
Jaccard captures "the tools agree with each other regardless of the
composite". Mixing both keeps the score robust to fusion-threshold
quirks.

Edge cases
----------
- Tools with ``success=False`` are skipped.
- Tools with empty ``binding_protein_residues`` are skipped (e.g.
  EquiPNAS when every per-residue prob fell below threshold —
  contributing such a tool would dilute Jaccard with all-zero rows).
- 0 active tools → return ``None`` (the metric abstains; scorer drops
  it from the weighted total).
- 1 active tool → return 0.5 (neutral — there is no consensus to
  measure with one tool).
- ``min_tools`` config keeps both rules together: < min_tools but ≥ 1
  active falls back to the neutral 0.5.
- Composite empty but tools have predictions → vote_ratio = 0.0;
  Jaccard still computed. Score reflects "tools agreed somewhere but
  the fusion produced nothing" — partially credits tool-level
  agreement.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, Optional


def _round_or_none(x: Optional[float], places: int = 4) -> Optional[float]:
    return None if x is None else round(float(x), places)


def cross_tool_consensus(
    composite_binding: list[int],
    tool_predictions: Iterable,
    config: dict,
) -> tuple[Optional[float], dict]:
    """Score how strongly the surviving tools agree on the composite.

    Parameters
    ----------
    composite_binding
        Residue indices selected by step-5 fusion (after threshold).
    tool_predictions
        Iterable of step-4 ``ToolPrediction`` objects. Filtered to
        ``success=True`` AND non-empty ``binding_protein_residues``
        before any computation.
    config
        Sub-block ``pocket_qa.consensus``. Reads ``min_tools`` (default
        2), ``vote_weight`` (default 0.6), ``jaccard_weight`` (default
        0.4).

    Returns
    -------
    ``(score, info)`` where ``score`` is in [0, 1] when ≥ ``min_tools``
    active tools, 0.5 when 1 active tool (neutral), or None when 0
    active tools. ``info`` carries vote counts, pairwise Jaccard
    values, and the parameters actually used.
    """
    min_tools = int(config.get("min_tools", 2))
    vote_w = float(config.get("vote_weight", 0.6))
    jacc_w = float(config.get("jaccard_weight", 0.4))

    # Filter to active tools: success=True with non-empty binding list.
    active: list[tuple[str, set[int]]] = []
    skipped: dict[str, str] = {}
    for p in tool_predictions:
        tool_id = getattr(p, "tool_id", None) or "?"
        if not getattr(p, "success", False):
            skipped[tool_id] = "success=False"
            continue
        bps = getattr(p, "binding_protein_residues", None) or []
        if not bps:
            skipped[tool_id] = "empty binding_protein_residues"
            continue
        active.append((tool_id, set(int(i) for i in bps)))

    n_active = len(active)
    base_info: dict = {
        "n_active_tools": n_active,
        "active_tools": [tid for tid, _ in active],
        "skipped_tools": skipped,
        "min_tools": min_tools,
        "weights": {"vote_weight": vote_w, "jaccard_weight": jacc_w},
    }

    if n_active == 0:
        return None, {**base_info,
                      "reason": "no tools survived the success+non-empty filter"}

    if n_active < max(min_tools, 2):
        # 1 surviving tool (or min_tools coerced down) — neutral 0.5.
        return 0.5, {**base_info,
                     "reason": "only one active tool; consensus is undefined "
                               "→ neutral 0.5"}

    composite_set: set[int] = set(int(i) for i in (composite_binding or []))

    # Vote count per composite residue.
    vote_count: dict[int, int] = {
        i: sum(1 for _, s in active if i in s) for i in composite_set
    }
    if composite_set:
        avg_vote_ratio = (
            sum(vote_count.values()) / (n_active * len(composite_set))
        )
    else:
        avg_vote_ratio = 0.0

    # Pairwise Jaccard over the FULL per-tool predicted sets (not
    # restricted to composite — Jaccard is meant to capture tool-level
    # agreement everywhere).
    jaccard_values: list[float] = []
    pairwise_jaccard: dict[str, float] = {}
    for (tid_a, set_a), (tid_b, set_b) in combinations(active, 2):
        union = set_a | set_b
        if not union:
            continue  # defensive — empty sets were filtered upstream
        jacc = len(set_a & set_b) / len(union)
        jaccard_values.append(jacc)
        pairwise_jaccard[f"{tid_a}|{tid_b}"] = round(jacc, 4)

    avg_jaccard = (sum(jaccard_values) / len(jaccard_values)
                   if jaccard_values else 0.0)

    score = vote_w * avg_vote_ratio + jacc_w * avg_jaccard
    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0

    info = {
        **base_info,
        "n_composite_residues": len(composite_set),
        "avg_vote_ratio": _round_or_none(avg_vote_ratio),
        "avg_jaccard": _round_or_none(avg_jaccard),
        "vote_count": dict(sorted(vote_count.items())),
        "pairwise_jaccard": pairwise_jaccard,
    }
    return round(score, 6), info
