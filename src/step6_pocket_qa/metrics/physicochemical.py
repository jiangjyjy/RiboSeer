"""q2 — physicochemical_complementarity (formula 19 in the paper).

Idea
----
RNA carries a strongly negative phosphate backbone and exposes flat
aromatic bases. A *bona fide* protein binding interface tends to
mirror those properties: enriched in positively charged side chains
(electrostatics with the backbone), aromatics (base stacking), and
polar residues (H-bonding to bases / 2'-OH), while not being so
glycine/proline-heavy that the loop becomes either floppy or kinked
to the point of breaking complementarity. Four sub-scores capture
those expectations and are averaged.

Sub-scores (each ∈ [0, 1])
--------------------------
1. ``score_charge``   = ``min(positive_ratio  / positive_threshold,  1.0)``
   positive_ratio = fraction of K/R/H among standard binding residues.
2. ``score_aromatic`` = ``min(aromatic_ratio  / aromatic_threshold,  1.0)``
3. ``score_polar``    = ``min(polar_ratio     / polar_threshold,     1.0)``
4. ``score_gp``       = 1.0 if ``gp_ratio < gp_threshold`` else
                       ``max(0, 1 - (gp_ratio - gp_threshold) / gp_threshold)``

``q2 = mean(score_charge, score_aromatic, score_polar, score_gp)``

Edge cases
----------
- Empty ``binding_residues`` → return ``None`` (abstain; metric has
  no data, scorer drops it from the weighted total).
- Residue indices are 1-based (matching step-5 ``CompositeResult``).
  Out-of-range indices (``res < 1`` or ``res > len(sequence)``) are
  silently skipped and counted in ``info['n_out_of_range']``.
- Non-standard amino acids (anything outside the 20 canonical letters
  — X, B, Z, U, O, gap, lowercase, etc.) are skipped from the ratio
  numerators AND denominator, and counted in ``info['n_non_standard']``.
- If after filtering 0 standard AAs remain → return ``None``.
- ``rna_sequence`` / ``binding_nucleotides`` are not used in the score
  itself (sub-scores are protein-side); we surface their summary in
  ``info`` so step 7's LLM has the full picture.
"""
from __future__ import annotations

from typing import Optional


# Canonical 20 amino acids — anything else is non-standard.
STANDARD_AA: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Side-chain classes used by the four sub-scores. A residue may belong
# to multiple classes (e.g. H is positive and aromatic-ish; D/E are
# polar and acidic). We follow the spec's groupings literally — any
# overlap is intentional and consistent across samples.
POSITIVE_CHARGED: frozenset[str] = frozenset("KRH")
AROMATIC: frozenset[str] = frozenset("FYW")
POLAR: frozenset[str] = frozenset("STNQDE")
GLYCINE_PROLINE: frozenset[str] = frozenset("GP")


def _round_or_none(x: Optional[float], places: int = 4) -> Optional[float]:
    return None if x is None else round(float(x), places)


def physicochemical_complementarity(
    binding_residues: list[int],
    protein_sequence: str,
    rna_sequence: str,
    binding_nucleotides: Optional[list[int]],
    config: dict,
) -> tuple[Optional[float], dict]:
    """Score how chemically complementary the binding interface looks.

    Parameters
    ----------
    binding_residues
        1-based protein residue indices from step-5 fusion.
    protein_sequence
        Full protein sequence (single-letter codes).
    rna_sequence
        Partner RNA sequence (used only for context in ``info``).
    binding_nucleotides
        Step-5 fused RNA binding indices (info-only; the four
        sub-scores live on the protein side).
    config
        Sub-block ``pocket_qa.physicochemical``. Reads
        ``positive_threshold`` (default 0.3), ``aromatic_threshold``
        (default 0.1), ``polar_threshold`` (default 0.2),
        ``gp_threshold`` (default 0.3).

    Returns
    -------
    ``(score, info)`` — ``score`` ∈ [0, 1] when ≥ 1 standard amino
    acid survives the index/standard filter, otherwise ``None``.
    ``info`` exposes per-class counts and the four sub-scores so
    step 7's LLM can reason about which axis is failing.
    """
    pos_thr = float(config.get("positive_threshold", 0.3))
    aro_thr = float(config.get("aromatic_threshold", 0.1))
    pol_thr = float(config.get("polar_threshold", 0.2))
    gp_thr = float(config.get("gp_threshold", 0.3))

    # ----- input housekeeping (record everything in info, abstain on empty)
    n_input = len(binding_residues or [])
    seq_len = len(protein_sequence or "")

    if n_input == 0:
        return None, {
            "reason": "binding_residues is empty",
            "n_binding_input": 0,
            "n_standard_aa": 0,
            "thresholds": {
                "positive": pos_thr, "aromatic": aro_thr,
                "polar": pol_thr, "gp": gp_thr,
            },
        }

    # 1-based → 0-based; collect standard AA letters.
    standard_letters: list[str] = []
    out_of_range: list[int] = []
    non_standard: list[tuple[int, str]] = []
    for r in binding_residues:
        if r < 1 or r > seq_len:
            out_of_range.append(int(r))
            continue
        aa = protein_sequence[r - 1]
        if aa in STANDARD_AA:
            standard_letters.append(aa)
        else:
            non_standard.append((int(r), aa))

    n_std = len(standard_letters)

    base_info: dict = {
        "n_binding_input": n_input,
        "n_standard_aa": n_std,
        "n_out_of_range": len(out_of_range),
        "n_non_standard": len(non_standard),
        "out_of_range_residues": out_of_range[:20],
        "non_standard_residues": non_standard[:20],
        "thresholds": {
            "positive": pos_thr, "aromatic": aro_thr,
            "polar": pol_thr, "gp": gp_thr,
        },
        "rna_length": len(rna_sequence or ""),
        "n_binding_nucleotides": len(binding_nucleotides or []),
    }

    if n_std == 0:
        # Every binding residue was either out of range or non-standard.
        # No data → abstain (consistent with empty-input case).
        return None, {
            **base_info,
            "reason": "no standard amino acids among binding residues "
                      "(all out-of-range or non-canonical)",
        }

    # ----- per-class counts on the standard-AA subset
    n_positive = sum(1 for c in standard_letters if c in POSITIVE_CHARGED)
    n_aromatic = sum(1 for c in standard_letters if c in AROMATIC)
    n_polar = sum(1 for c in standard_letters if c in POLAR)
    n_gp = sum(1 for c in standard_letters if c in GLYCINE_PROLINE)

    pos_ratio = n_positive / n_std
    aro_ratio = n_aromatic / n_std
    pol_ratio = n_polar / n_std
    gp_ratio = n_gp / n_std

    # ----- four sub-scores (each ∈ [0, 1])
    # Threshold = 0 would divide-by-zero; treat it as "any presence is full
    # credit" (ratio > 0 → 1.0, else 0.0). Negative threshold is nonsense;
    # clamp to 0 for safety.
    def _ratio_score(ratio: float, thr: float) -> float:
        if thr <= 0.0:
            return 1.0 if ratio > 0.0 else 0.0
        return min(ratio / thr, 1.0)

    score_charge = _ratio_score(pos_ratio, pos_thr)
    score_aromatic = _ratio_score(aro_ratio, aro_thr)
    score_polar = _ratio_score(pol_ratio, pol_thr)

    # G/P penalty kicks in only above the threshold; full credit below.
    if gp_thr <= 0.0:
        score_gp = 1.0 if gp_ratio == 0.0 else 0.0
    elif gp_ratio < gp_thr:
        score_gp = 1.0
    else:
        score_gp = max(0.0, 1.0 - (gp_ratio - gp_thr) / gp_thr)

    q2 = (score_charge + score_aromatic + score_polar + score_gp) / 4.0
    # Defensive clamp (sub-scores already in [0, 1]; clamp guards against
    # future edits introducing rounding drift).
    if q2 < 0.0:
        q2 = 0.0
    elif q2 > 1.0:
        q2 = 1.0

    info = {
        **base_info,
        "counts": {
            "positive": n_positive,
            "aromatic": n_aromatic,
            "polar": n_polar,
            "glycine_proline": n_gp,
        },
        "ratios": {
            "positive": _round_or_none(pos_ratio),
            "aromatic": _round_or_none(aro_ratio),
            "polar": _round_or_none(pol_ratio),
            "glycine_proline": _round_or_none(gp_ratio),
        },
        "sub_scores": {
            "charge": _round_or_none(score_charge),
            "aromatic": _round_or_none(score_aromatic),
            "polar": _round_or_none(score_polar),
            "gp": _round_or_none(score_gp),
        },
    }
    return round(q2, 6), info
