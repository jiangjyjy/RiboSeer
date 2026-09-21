"""Noisy-OR fusion engine — pure math, no LLM involvement.

Implements formula (11) from the paper:

    b_hat(i) = 1 - ∏_k (1 - c_k · b_k(i))

where:
  - i indexes a protein residue (or RNA nucleotide on the RNA side)
  - k indexes the tools that survived ``success=True`` filtering
  - c_k is the per-tool weight assigned by the LLM (or a fallback)
  - b_k(i) is tool k's per-residue probability for i, normalised to
    [0, 1] according to the tool's category

Per-residue probability normalisation (per Step 5 spec)
-------------------------------------------------------
- Cat A (Boltz-2, Chai-1): ``per_residue_confidence`` carries
  pLDDT in [0, 100] → divide by 100.
- Cat B (P2Rank): ligandability already in [0, 1] → pass through.
  (Fpocket is registered but currently disabled, see tool_registry.)
- Cat D (HADDOCK 3): docked complex — the adapter does not populate
  the per_residue_confidence field (only the eval-only distance
  score), so noisy-OR sees this tool only through its
  binding_protein_residues gate.
- Cat C (EquiPNAS): probability already in [0, 1] → pass through.
- Unknown / out-of-range values are clamped to [0, 1] defensively to
  avoid feeding noisy-OR with negative probabilities.

Per-residue source resolution
-----------------------------
``binding_protein_residues`` is the *gate*: a residue not in that list
contributes b_k(i) = 0 regardless of any per-residue map. This matters
for Cat A tools, where ``per_residue_confidence`` carries pLDDT — a
structure-quality score that is high for well-modelled residues
everywhere on the chain, *not* an indicator of being on the RNA
interface. Letting pLDDT through for non-binding residues blew up the
fused binding set (the bug that motivated this gating).

For residues that ARE in ``binding_protein_residues``:
  - If ``per_residue_confidence`` has the residue → use the (normalised)
    value as b_k(i).
  - If not → fall back to b_k(i) = 1.0 (binary indicator).

For RNA, ``ToolPrediction`` only carries ``binding_rna_nucleotides`` (a
list, no per-nucleotide score). We treat each listed nucleotide as
b_k(i) = 1.0; unlisted nucleotides contribute nothing to the noisy-OR
product.

Edge cases
----------
- success=False predictions are filtered upstream (in fusion.py) before
  this function is called. As a defensive measure we still ignore them
  here.
- 0 surviving tools → returns empty result (caller should detect this
  earlier and short-circuit).
- 1 surviving tool → still runs the formula with c_k = 1.0 (caller may
  short-circuit this case by passing weights={tool: 1.0}).
- A residue predicted by only one tool → b_hat(i) = c_k * b_k(i)
  (because the product over the other tools degenerates to 1).
- A residue predicted by no tools → not present in the output dict.
"""
from __future__ import annotations

from typing import Iterable, Optional

# Imported lazily / typed loosely so this module is independently importable.
# step4 schemas live in src/step4_tool_adapters/schemas.py.
try:
    from step4_tool_adapters.schemas import ToolPrediction  # type: ignore
except Exception:  # pragma: no cover — fallback for partial installs
    ToolPrediction = object  # type: ignore


def _normalize_confidence(value: float, category: str) -> float:
    """Map a tool's per-residue confidence to [0, 1].

    See module docstring for the per-category rules.
    """
    v = float(value)
    if category == "A":
        v = v / 100.0
    # Clamp into [0, 1] for downstream noisy-OR.
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _protein_residue_probs(pred) -> dict[int, float]:
    """Extract {residue_idx: prob_in_[0,1]} for one tool's protein side.

    Gating rule: ``binding_protein_residues`` decides which residues this
    tool predicts as binding. For each such residue, the value comes
    from ``per_residue_confidence`` (normalised per category) when
    available, otherwise a binary 1.0. Residues not in the binding list
    are simply omitted from the dict (i.e. b_k(i) = 0).
    """
    cat = getattr(pred, "category", "")
    binding = getattr(pred, "binding_protein_residues", None) or []
    if not binding:
        return {}
    prc = getattr(pred, "per_residue_confidence", None) or {}
    out: dict[int, float] = {}
    for raw_i in binding:
        i = int(raw_i)
        if i in prc:
            out[i] = _normalize_confidence(prc[i], cat)
        else:
            out[i] = 1.0
    return out


def _rna_nuc_probs(pred) -> dict[int, float]:
    """Extract {nucleotide_idx: 1.0} for one tool's RNA side.

    ``ToolPrediction`` does not carry per-nucleotide confidence, so we
    treat each listed nucleotide as a hard positive (b_k(i) = 1.0).
    Tools that didn't predict the RNA side return an empty dict.
    """
    brn = getattr(pred, "binding_rna_nucleotides", None) or []
    return {int(i): 1.0 for i in brn}


def _noisy_or_combine(
    per_tool_probs: list[tuple[float, dict[int, float]]],
) -> dict[int, float]:
    """Apply ``1 - ∏_k (1 - c_k · b_k(i))`` over every residue index.

    ``per_tool_probs`` is a list of ``(c_k, {i: b_k(i)})`` pairs. Only
    residues seen by at least one tool appear in the result.
    """
    # Collect every residue index any tool touched.
    all_indices: set[int] = set()
    for _, probs in per_tool_probs:
        all_indices.update(probs.keys())

    fused: dict[int, float] = {}
    for i in all_indices:
        # ∏_k (1 - c_k · b_k(i))  — start at 1.0, multiply per tool
        prod = 1.0
        for c_k, probs in per_tool_probs:
            b_ki = probs.get(i, 0.0)
            prod *= (1.0 - c_k * b_ki)
        prob_i = 1.0 - prod
        # Numerical guard — float errors can drag this slightly outside [0, 1].
        if prob_i < 0.0:
            prob_i = 0.0
        elif prob_i > 1.0:
            prob_i = 1.0
        fused[i] = prob_i
    return fused


def noisy_or_fusion(
    tool_predictions: Iterable,
    weights: dict[str, float],
    threshold: float = 0.5,
) -> dict:
    """Fuse multiple ``ToolPrediction`` objects with noisy-OR.

    Parameters
    ----------
    tool_predictions
        Iterable of ``ToolPrediction``-like objects. Items with
        ``success=False`` are skipped defensively even though the caller
        should already have filtered them out.
    weights
        ``{tool_id: c_k}`` map; missing keys default to 0.0 (effectively
        excluding that tool). Each c_k must be in [0, 1].
    threshold
        τ used to threshold ``b_hat(i)`` for the binding-set output.

    Returns
    -------
    dict with keys:
      - per_residue_probability: ``{int: float}``  (protein side)
      - per_nucleotide_probability: ``{int: float}`` (RNA side, may be empty)
      - binding_protein_residues: ``list[int]`` sorted
      - binding_rna_nucleotides: ``list[int]`` sorted
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold {threshold} outside [0, 1]")
    for tool_id, c in weights.items():
        if not (0.0 <= float(c) <= 1.0):
            raise ValueError(
                f"weight for '{tool_id}' = {c} outside [0, 1]"
            )

    # Build per-tool probability maps for both sides.
    protein_pairs: list[tuple[float, dict[int, float]]] = []
    rna_pairs: list[tuple[float, dict[int, float]]] = []

    for pred in tool_predictions:
        if not getattr(pred, "success", False):
            continue
        tool_id = getattr(pred, "tool_id", None)
        if tool_id is None:
            continue
        c_k = float(weights.get(tool_id, 0.0))
        if c_k <= 0.0:
            # A zero weight contributes (1 - 0 * b) = 1 to the product,
            # i.e. nothing — skip to keep ``all_indices`` tight.
            continue

        protein_pairs.append((c_k, _protein_residue_probs(pred)))
        rna_pairs.append((c_k, _rna_nuc_probs(pred)))

    per_residue = _noisy_or_combine(protein_pairs)
    per_nuc = _noisy_or_combine(rna_pairs)

    binding_protein = sorted(i for i, p in per_residue.items() if p > threshold)
    binding_rna = sorted(i for i, p in per_nuc.items() if p > threshold)

    return {
        "per_residue_probability": per_residue,
        "per_nucleotide_probability": per_nuc,
        "binding_protein_residues": binding_protein,
        "binding_rna_nucleotides": binding_rna,
    }


def equal_weights(tool_ids: Iterable[str], value: float = 1.0) -> dict[str, float]:
    """Convenience: ``{tool_id: value}`` map for fallback / single-tool cases.

    The default value 1.0 is intended for the ``single surviving tool``
    short-circuit (not specified in the paper; in the paper's cascade c_k=1 means simply adopting that tool's prediction).
    For the ``LLM failed`` fallback path, callers usually pass
    ``value=config.fallback_weight`` (e.g. 0.5).
    """
    return {tid: float(value) for tid in tool_ids}
