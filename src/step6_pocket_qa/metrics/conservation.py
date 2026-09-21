"""q3 — evolutionary_conservation (formula 20 in the paper).

Idea
----
Truly functional binding sites tend to be evolutionarily conserved.
Without an MSA / ConSurf run we approximate that signal from the
single sequence + the step-1 features:

  1. **Rare-AA enrichment** — interfaces around small / aromatic
     anchor residues (C, W, M, H by default) are statistically
     conserved more often than glycine-rich loops. We measure the
     fraction of those "rare" residues among the binding letters and
     scale by a target ratio (default 0.15).
  2. **Positional plausibility** — sequence termini (the first /
     last ``terminal_fraction`` of the protein) are usually
     disordered / poorly conserved. Binding bleeding into the termini
     is a soft red flag; the score is ``1 - terminal_share``.
  3. **pI consistency** — the protein-wide pI predicts the dominant
     charge mode of any conserved interface. Strongly basic proteins
     (pI > ``positive_pi_threshold``) should have positively charged
     binding patches; strongly acidic proteins (pI < ``acidic_pi_threshold``)
     should *not* expose acidic patches at the RNA interface. In the
     neutral band we abstain from this axis (returns 0.5).

Sub-scores (each ∈ [0, 1])
--------------------------
- ``score_rare_aa = min(rare_ratio / rare_aa_threshold, 1.0)``
- ``score_position = 1.0 - terminal_share`` (clamped to [0, 1])
- ``score_pi_consistency`` — three-way:
    * ``pI > positive_pi_threshold`` → ``min(positive_ratio / charge_target, 1.0)``
    * ``pI < acidic_pi_threshold``   → ``max(0, 1 - acidic_ratio / charge_target)``
    * otherwise                      → ``0.5`` (neutral band)
    * ``pI is None``                 → drop this axis from the average

``q3 = mean(computed sub-scores)``.

Edge cases
----------
- Empty ``binding_residues`` → return ``None`` (abstain).
- Empty / missing protein sequence → return ``None``.
- Non-standard residues (X / B / Z / lowercase / out-of-range) are
  filtered out for the ratio numerators *and* denominator, just like
  q2. If 0 standard residues survive the filter → return ``None``.
- ``aa_composition`` is *not* required — the rare-AA target is a
  configurable absolute threshold, not a relative enrichment, so the
  score works even when step 1 features are missing.

Note: a real implementation should run ConSurf or a PSI-BLAST MSA and
score Shannon entropy at each binding position. This MVP keeps the
metric purely sequence-local so step 6 can run offline.
"""
from __future__ import annotations

from typing import Optional


# Canonical 20 amino acids — anything else is non-standard.
STANDARD_AA: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")
POSITIVE_AA: frozenset[str] = frozenset("KRH")
ACIDIC_AA: frozenset[str] = frozenset("DE")

# Sub-score defaults (mirror configs/step6_config.yaml::pocket_qa.conservation).
DEFAULT_RARE_AA = "CWMH"
DEFAULT_RARE_THRESHOLD = 0.15
DEFAULT_TERMINAL_FRACTION = 0.1
DEFAULT_POSITIVE_PI = 9.0
DEFAULT_ACIDIC_PI = 6.0
# Reuse q2's positive-charge target as the cross-check for pI > 9 case.
DEFAULT_CHARGE_TARGET = 0.3


def _round_or_none(x: Optional[float], places: int = 4) -> Optional[float]:
    return None if x is None else round(float(x), places)


def evolutionary_conservation(
    binding_residues: list[int],
    protein_sequence: str,
    sample_json: dict,
    config: dict,
) -> tuple[Optional[float], dict]:
    """Score how evolutionarily plausible the predicted interface looks.

    Parameters
    ----------
    binding_residues
        1-based protein residue indices from step-5 fusion.
    protein_sequence
        Full protein sequence (single-letter codes).
    sample_json
        Step-1 sample dict; we read ``protein.features.pI`` for the
        pI-consistency axis. Other paths are tolerated as missing.
    config
        Sub-block ``pocket_qa.conservation``. Reads ``rare_aa`` (str of
        AA letters; default "CWMH"), ``rare_aa_threshold`` (default
        0.15), ``terminal_fraction`` (default 0.1), ``positive_pi_threshold``
        (default 9.0), ``acidic_pi_threshold`` (default 6.0),
        ``charge_target`` (default 0.3).

    Returns
    -------
    ``(score, info)`` — ``score`` ∈ [0, 1] when at least one axis can
    be computed, otherwise ``None``. ``info`` exposes per-axis
    sub-scores, ratios, pI used, and the configured thresholds.
    """
    # ---------------- config with defaults ---------------------------------
    rare_aa_str = str(config.get("rare_aa", DEFAULT_RARE_AA) or "")
    rare_aa = frozenset(c.upper() for c in rare_aa_str if c.upper() in STANDARD_AA)
    rare_threshold = float(config.get("rare_aa_threshold", DEFAULT_RARE_THRESHOLD))
    terminal_fraction = float(
        config.get("terminal_fraction", DEFAULT_TERMINAL_FRACTION)
    )
    pos_pi_thr = float(config.get("positive_pi_threshold", DEFAULT_POSITIVE_PI))
    acid_pi_thr = float(config.get("acidic_pi_threshold", DEFAULT_ACIDIC_PI))
    charge_target = float(config.get("charge_target", DEFAULT_CHARGE_TARGET))

    n_input = len(binding_residues or [])
    seq_len = len(protein_sequence or "")

    base_info: dict = {
        "n_binding_input": n_input,
        "protein_length": seq_len,
        "config": {
            "rare_aa": "".join(sorted(rare_aa)),
            "rare_aa_threshold": rare_threshold,
            "terminal_fraction": terminal_fraction,
            "positive_pi_threshold": pos_pi_thr,
            "acidic_pi_threshold": acid_pi_thr,
            "charge_target": charge_target,
        },
    }

    # ---------------- abstain paths ----------------------------------------
    if n_input == 0:
        return None, {**base_info, "reason": "binding_residues is empty"}
    if seq_len == 0:
        return None, {**base_info, "reason": "protein_sequence is empty"}

    # ---------------- input filtering --------------------------------------
    standard_letters: list[str] = []
    in_range_indices: list[int] = []
    out_of_range: list[int] = []
    non_standard: list[tuple[int, str]] = []
    for r in binding_residues:
        ri = int(r)
        if ri < 1 or ri > seq_len:
            out_of_range.append(ri)
            continue
        aa = protein_sequence[ri - 1]
        if aa in STANDARD_AA:
            standard_letters.append(aa)
            in_range_indices.append(ri)
        else:
            non_standard.append((ri, aa))

    n_std = len(standard_letters)
    base_info["n_standard_aa"] = n_std
    base_info["n_out_of_range"] = len(out_of_range)
    base_info["n_non_standard"] = len(non_standard)
    base_info["out_of_range_residues"] = out_of_range[:20]
    base_info["non_standard_residues"] = non_standard[:20]

    if n_std == 0:
        return None, {
            **base_info,
            "reason": "no standard amino acids among binding residues "
                      "(all out-of-range or non-canonical)",
        }

    # ---------------- sub-score 1: rare-AA enrichment ----------------------
    n_rare = sum(1 for aa in standard_letters if aa in rare_aa)
    rare_ratio = n_rare / n_std
    if rare_threshold <= 0:
        # Threshold = 0 → "any rare residue is full credit" (matches q2 style).
        score_rare = 1.0 if rare_ratio > 0.0 else 0.0
    else:
        score_rare = min(rare_ratio / rare_threshold, 1.0)

    # ---------------- sub-score 2: positional (terminal penalty) -----------
    # terminal_size: at least 1 residue at each end so the rule still fires
    # on tiny proteins; capped at half the sequence so the two windows don't
    # overlap (would over-count).
    half = seq_len // 2 if seq_len > 1 else 1
    terminal_size = max(
        1, min(half, int(round(seq_len * max(0.0, terminal_fraction)))),
    )
    n_terminal = sum(
        1 for ri in in_range_indices
        if ri <= terminal_size or ri > seq_len - terminal_size
    )
    terminal_share = n_terminal / len(in_range_indices)
    score_position = max(0.0, min(1.0, 1.0 - terminal_share))

    # ---------------- sub-score 3: pI consistency --------------------------
    # Read pI from step-1 protein features; tolerate missing field / non-numeric.
    features = ((sample_json or {}).get("protein") or {}).get("features") or {}
    raw_pi = features.get("pI")
    pI: Optional[float]
    if raw_pi is None:
        pI = None
    else:
        try:
            pI = float(raw_pi)
        except (TypeError, ValueError):
            pI = None

    n_positive = sum(1 for aa in standard_letters if aa in POSITIVE_AA)
    n_acidic = sum(1 for aa in standard_letters if aa in ACIDIC_AA)
    pos_ratio = n_positive / n_std
    acid_ratio = n_acidic / n_std

    score_pi: Optional[float]
    pi_reason: str
    if pI is None:
        score_pi = None
        pi_reason = "pI not available; sub-score skipped"
    elif pI > pos_pi_thr:
        if charge_target <= 0.0:
            score_pi = 1.0 if pos_ratio > 0.0 else 0.0
        else:
            score_pi = min(pos_ratio / charge_target, 1.0)
        pi_reason = (
            f"pI={pI:.2f} > {pos_pi_thr}: expected positive enrichment; "
            f"positive_ratio={pos_ratio:.3f}"
        )
    elif pI < acid_pi_thr:
        if charge_target <= 0.0:
            score_pi = 0.0 if acid_ratio > 0.0 else 1.0
        else:
            score_pi = max(0.0, 1.0 - acid_ratio / charge_target)
        pi_reason = (
            f"pI={pI:.2f} < {acid_pi_thr}: expected low acidic; "
            f"acidic_ratio={acid_ratio:.3f}"
        )
    else:
        # Neutral pI band: no informative expectation, give 0.5 so this
        # axis pulls the average toward 0.5 (rather than abstaining and
        # letting the other two axes dominate).
        score_pi = 0.5
        pi_reason = (
            f"pI={pI:.2f} in neutral band [{acid_pi_thr}, {pos_pi_thr}]; "
            f"score=0.5"
        )

    # Defensive clamps before averaging.
    score_rare = max(0.0, min(1.0, score_rare))
    score_position = max(0.0, min(1.0, score_position))
    if score_pi is not None:
        score_pi = max(0.0, min(1.0, score_pi))

    sub_scores: dict[str, Optional[float]] = {
        "rare_aa": _round_or_none(score_rare),
        "position": _round_or_none(score_position),
        "pi_consistency": _round_or_none(score_pi),
    }
    computed = [s for s in sub_scores.values() if s is not None]
    # `score_rare` and `score_position` always compute past the abstain
    # checks, so `computed` has ≥ 2 entries here.
    q3 = sum(computed) / len(computed)
    if q3 < 0.0:
        q3 = 0.0
    elif q3 > 1.0:
        q3 = 1.0

    info = {
        **base_info,
        "counts": {
            "rare": n_rare,
            "positive": n_positive,
            "acidic": n_acidic,
        },
        "ratios": {
            "rare": _round_or_none(rare_ratio),
            "positive": _round_or_none(pos_ratio),
            "acidic": _round_or_none(acid_ratio),
        },
        "terminal_size": terminal_size,
        "n_terminal_binding": n_terminal,
        "terminal_share": _round_or_none(terminal_share),
        "pI": _round_or_none(pI),
        "pi_reason": pi_reason,
        "sub_scores": sub_scores,
        "n_sub_scores_computed": len(computed),
    }
    return round(q3, 6), info
