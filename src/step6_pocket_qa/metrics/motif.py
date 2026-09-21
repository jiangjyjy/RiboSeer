"""q5 — known_motif_consistency (formula 22 in the paper).

Idea
----
RNA-binding domains have well-characterised sequence motifs. If
step 2 predicted the protein domain (RRM / KH / zinc finger / dsRBD /
PUF / DEAD-box), we can scan the sequence for the canonical motif of
that family and ask: do the predicted binding residues actually
overlap the motif positions? A high overlap means "the prediction
respects the family's conserved binding apparatus".

For domains where step 2 abstains (``novel_fold``, ``unstructured``)
or where we cannot recognise the category, we return a neutral score
(``neutral_score``, default 0.5) — we have no informative prior, so
the metric should not push the aggregate either way.

Scoring
-------
For every regex hit in the protein sequence we look at the residue
indices it spans (1-based). Across all hits combined:

    score = (# motif residues that fall inside binding_residues)
            / (# motif residues total across all hits)

When zero motifs match the protein sequence the rule cannot fire →
return ``neutral_score``. Pure overlap fraction is intentionally
naive — the spec calls this an MVP based on domain knowledge; a
production version would weight conserved positions more heavily and
consult Pfam / InterPro.

Edge cases
----------
- Empty ``binding_residues`` → ``None`` (abstain).
- Empty / missing ``protein_sequence`` → ``None``.
- Empty / unrecognised ``target_category`` (and the special
  ``novel_fold`` / ``unstructured`` cases) → return ``neutral_score``;
  the metric still contributes to the weighted average.
- ``multi_domain`` → scan with every motif set; sum coverage across
  all matches.
- Out-of-range binding indices (``r < 1`` or ``r > len(sequence)``)
  are dropped before overlap counting.
"""
from __future__ import annotations

import re
from typing import Optional


# Canonical motif patterns per protein domain class.
#
# These regexes are deliberately simple, single-letter regexes — MVP
# heuristics derived from textbook descriptions. Each entry is
# ``(name, pattern)``; the same domain may have multiple motifs (RNP1
# + RNP2 for RRM).
#
# References:
#   - RRM RNP1 / RNP2: Maris et al., FEBS J 2005.
#   - KH GXXG loop:    Valverde et al., FEBS J 2008.
#   - C2H2 zinc finger spacing: Wolfe et al., Annu Rev Biophys 2000.
#   - DEAD-box: Linder & Jankowsky, Nat Rev Mol Cell Biol 2011 (DEAD/DEAH).
MOTIFS: dict[str, list[tuple[str, str]]] = {
    "RRM": [
        ("RNP1", r"[KR]G[FY][GA][FY][VILM].[FY]"),  # 8 residues
        ("RNP2", r"[ILV][FY][ILV].NL"),              # 6 residues
    ],
    "KH": [
        ("GXXG", r"G..G"),                            # 4 residues
    ],
    "zinc_finger": [
        # Classic C2H2: C-x(2,4)-C-x(12)-H-x(3,5)-H (~20-25 residues)
        ("C2H2", r"C.{2,4}C.{12}H.{3,5}H"),
    ],
    "dsRBD": [
        # dsRBD has no rigid sequence motif; the alpha-2 helix tends
        # to expose a basic patch. Use a "two consecutive K/R" proxy
        # — coarse but matches the conserved RNA-contact stretch.
        ("KR-pair", r"[KR][KR]"),                    # 2 residues
    ],
    "PUF": [
        # PUF repeats use an aromatic (F/Y) + asparagine pair to read
        # individual bases; the spacing varies. Loose proxy.
        ("PUF-repeat", r"[FY]N.{2,4}N"),
    ],
    "DEAD_box": [
        # The eponymous Walker B motif (DEAD or its DEAH variant).
        ("DEAD-motif", r"DEA[DH]"),
    ],
}

# Domains where we have no motif rule and consciously return neutral.
_NEUTRAL_DOMAINS: frozenset[str] = frozenset({"novel_fold", "unstructured", ""})


def _round_or_none(x: Optional[float], places: int = 4) -> Optional[float]:
    return None if x is None else round(float(x), places)


def _parse_domain(target_category: str) -> str:
    """Extract the protein-domain prefix from ``"{domain}_x_{rna}"``.

    Step-2 categories are formatted as ``RRM_x_stem_loop`` etc. We
    accept either the full canonical form or a bare domain name (in
    case upstream wrote just the domain — defensive).
    """
    if not target_category:
        return ""
    if "_x_" in target_category:
        return target_category.split("_x_", 1)[0]
    return target_category


def _select_patterns(domain: str) -> list[tuple[str, str]]:
    """Pick the motif list for a recognised domain.

    Returns an empty list when the domain has no rule (caller treats
    that as "neutral" and short-circuits).
    """
    if domain in MOTIFS:
        return list(MOTIFS[domain])
    if domain == "multi_domain":
        # Composite proteins: scan for every canonical family's motif.
        merged: list[tuple[str, str]] = []
        for d in ("RRM", "KH", "zinc_finger", "dsRBD", "PUF", "DEAD_box"):
            merged.extend(MOTIFS.get(d, []))
        return merged
    return []


def known_motif_consistency(
    binding_residues: list[int],
    protein_sequence: str,
    target_category: str,
    sample_json: dict,
    config: dict,
) -> tuple[Optional[float], dict]:
    """Score how well predicted binding residues cover the family motif.

    Parameters
    ----------
    binding_residues
        1-based protein residue indices from step-5 fusion.
    protein_sequence
        Full protein sequence (single-letter codes).
    target_category
        Step-2 category string ``"{protein_domain}_x_{rna_structure}"``;
        used to select the family motif. Bare domain names accepted.
    sample_json
        Step-1 sample dict (currently unused; kept for interface
        symmetry with q3 and future extensions).
    config
        Sub-block ``pocket_qa.motif``. Reads ``neutral_score`` (default
        0.5) used when no motif rule applies.

    Returns
    -------
    ``(score, info)`` — ``score`` ∈ [0, 1] when ≥ 1 motif matches the
    sequence and the binding set is non-empty; ``neutral_score`` for
    domains without a rule or motifs that don't match; ``None`` when
    the inputs preclude any scoring (empty binding / no sequence).
    """
    # `sample_json` accepted for interface symmetry with q3 / scorer.
    del sample_json
    neutral = float(config.get("neutral_score", 0.5))
    if neutral < 0.0:
        neutral = 0.0
    elif neutral > 1.0:
        neutral = 1.0

    n_input = len(binding_residues or [])
    seq_len = len(protein_sequence or "")
    domain = _parse_domain(target_category or "")

    base_info: dict = {
        "n_binding_input": n_input,
        "protein_length": seq_len,
        "target_category": target_category or "",
        "protein_domain": domain,
        "config": {"neutral_score": neutral},
    }

    # ---------------- abstain paths ----------------------------------------
    if n_input == 0:
        return None, {**base_info, "reason": "binding_residues is empty"}
    if seq_len == 0:
        return None, {**base_info, "reason": "protein_sequence is empty"}

    # ---------------- filter binding to the valid range --------------------
    binding_set: set[int] = set()
    out_of_range: list[int] = []
    for r in binding_residues:
        ri = int(r)
        if 1 <= ri <= seq_len:
            binding_set.add(ri)
        else:
            out_of_range.append(ri)
    base_info["n_binding_in_range"] = len(binding_set)
    base_info["n_out_of_range"] = len(out_of_range)
    base_info["out_of_range_residues"] = out_of_range[:20]

    if not binding_set:
        return None, {
            **base_info,
            "reason": "every binding residue is out of range",
        }

    # ---------------- pick motif set ---------------------------------------
    if domain in _NEUTRAL_DOMAINS:
        return neutral, {
            **base_info,
            "reason": (
                f"no motif rule for domain {domain!r}; "
                f"returning neutral {neutral}"
            ),
            "n_motifs_found": 0,
            "n_motifs_total_residues": 0,
        }

    patterns = _select_patterns(domain)
    if not patterns:
        # Domain string is non-empty and not a "neutral" sentinel, but
        # it doesn't match any known family — treat as neutral and
        # surface the unrecognised label so step 7 can flag it.
        return neutral, {
            **base_info,
            "reason": f"unrecognised domain {domain!r}; returning neutral {neutral}",
            "n_motifs_found": 0,
            "n_motifs_total_residues": 0,
        }

    # ---------------- scan the sequence ------------------------------------
    hits: list[dict] = []
    motif_residues_in_binding = 0
    motif_residues_total = 0
    regex_errors: list[str] = []
    for name, pattern in patterns:
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            regex_errors.append(f"{name}: invalid pattern {exc}")
            continue
        for m in compiled.finditer(protein_sequence):
            start_1 = m.start() + 1
            length = m.end() - m.start()
            residue_indices = set(range(start_1, start_1 + length))
            in_binding = len(residue_indices & binding_set)
            hits.append({
                "name": name,
                "match": m.group(),
                "start": start_1,
                "end": start_1 + length - 1,
                "length": length,
                "residues_in_binding": in_binding,
            })
            motif_residues_in_binding += in_binding
            motif_residues_total += length

    base_info["patterns_used"] = [name for name, _ in patterns]
    base_info["n_motifs_found"] = len(hits)
    base_info["motif_hits"] = hits[:20]
    base_info["n_motifs_total_residues"] = motif_residues_total
    base_info["motif_residues_in_binding"] = motif_residues_in_binding
    if regex_errors:
        base_info["regex_errors"] = regex_errors

    if motif_residues_total == 0:
        # Domain has rules but no instance was found in the sequence —
        # we lack evidence to support *or* contradict the prediction.
        return neutral, {
            **base_info,
            "reason": (
                f"no '{domain}' motif found in protein sequence; "
                f"returning neutral {neutral}"
            ),
        }

    score = motif_residues_in_binding / motif_residues_total
    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0

    info = {
        **base_info,
        "coverage_ratio": _round_or_none(score),
    }
    return round(score, 6), info
