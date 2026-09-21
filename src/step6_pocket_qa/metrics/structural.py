"""q1 — structural_plausibility (formula 18 in the paper).

Idea
----
A real RNA-protein binding interface looks like a *patch* on the
protein surface, not a scattered handful of residues. Three axes
capture that:

  1. **Sequence continuity** — binding residues should cluster on
     the sequence (1-3 stretches), not be sprinkled all over.
     ``score_continuity = 1 / (1 + max(0, n_clusters - max_clusters))``.
  2. **3D compactness** — the CA atoms of binding residues should
     have a small radius of gyration relative to what a random walk
     of the same length would give (``sqrt(N) * 3.8 Å``).
     ``score_compactness = 1 / (1 + Rg / expected_Rg)``.
  3. **Interface fraction** — a typical RNA-binding pocket spans
     10-40 % of the protein. Outside that band the score falls off
     symmetrically around the centre.

Sub-score availability
----------------------
- ``structure_path is None`` (no Cat A tool succeeded) → only
  continuity + ratio are averaged. The metric still returns a real
  score; ``compactness`` is reported as None in ``info`` for clarity.
- ``structure_path`` is set but the file is missing / unreadable, or
  no CA atoms can be matched to binding residues → log the reason in
  ``info['compactness_skip_reason']`` and fall back to the same
  2-axis average.
- ``binding_residues`` empty, or ``protein_length <= 0``, or every
  binding index is out-of-range → abstain (return ``None``); scorer
  drops the metric from the weighted total.

Residue indexing
----------------
Step-5 ``binding_protein_residues`` are 1-based positions on
``protein.sequence``. Cat A predicted structures (Boltz-2, Chai-1)
emit one residue per input sequence position with no gaps, so
``residue.seqid.num`` equals the sequence index — same convention as
``boltz2_adapter.read_protein_plddt_from_cif`` (see its docstring).
We look up CA coordinates by ``seqid.num`` instead of ``label_seq``
to avoid depending on gemmi's polymer auto-detection (which uses
peptide-bond geometry and would mis-number toy test structures).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

try:
    import gemmi  # type: ignore
except ImportError as _e:  # pragma: no cover — gemmi is in environment.yml
    gemmi = None  # type: ignore
    _GEMMI_IMPORT_ERROR: Optional[Exception] = _e
else:
    _GEMMI_IMPORT_ERROR = None


# CA-CA virtual bond length used as the "random walk step" when
# normalising Rg. 3.8 Å is the canonical trans-peptide CA distance.
_CA_BOND_ANGSTROM = 3.8


def _round_or_none(x: Optional[float], places: int = 4) -> Optional[float]:
    return None if x is None else round(float(x), places)


# ---------- pure-sequence helpers -----------------------------------------


def count_sequence_clusters(
    residues: list[int], gap_threshold: int = 5,
) -> int:
    """Count contiguous runs in a sorted unique residue list.

    Two residues are in the same cluster when the gap between
    consecutive sorted indices is ``<= gap_threshold``. Empty input
    returns 0 (caller is responsible for treating that as "no data").
    """
    if not residues:
        return 0
    uniq = sorted(set(int(r) for r in residues))
    clusters = 1
    for prev, cur in zip(uniq, uniq[1:]):
        if cur - prev > gap_threshold:
            clusters += 1
    return clusters


def _continuity_score(n_clusters: int, max_clusters: int) -> float:
    """``1 / (1 + max(0, n_clusters - max_clusters))``.

    Up to ``max_clusters`` runs of binding residues are tolerated for
    free; each additional cluster halves, thirds, etc. the score.
    """
    over = max(0, n_clusters - max(1, int(max_clusters)))
    return 1.0 / (1.0 + over)


def _ratio_score(
    ratio: float, low: float, high: float,
) -> float:
    """Plateau on ``[low, high]``, symmetric falloff outside.

    Inside the band: 1.0. Outside, follows the spec's literal formula
    ``max(0, 1 - |ratio - centre| / scale)`` with ``centre = (low+high)/2``
    and ``scale = centre`` — this reproduces the spec's hard-coded
    ``max(0, 1 - |r - 0.25| / 0.25)`` exactly when the default
    ``expected_interface_ratio = [0.1, 0.4]`` is used. The discontinuity
    at the band edge (1.0 just inside, < 1 just outside) is intentional
    MVP behaviour: the configured band is meant to be a *tolerance*,
    not a soft optimum.
    """
    if low <= ratio <= high:
        return 1.0
    centre = (low + high) / 2.0
    if low <= 0 and high <= 0:
        return 0.0
    scale = centre if centre > 0 else (high - low) / 2.0
    if scale <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(ratio - centre) / scale)


# ---------- 3D compactness ------------------------------------------------


def _radius_of_gyration(coords: list[tuple[float, float, float]]) -> float:
    """Standard mass-less Rg over a list of 3D points.

    ``Rg = sqrt(mean(||r_i - centroid||^2))``. Caller must ensure
    ``len(coords) >= 1``. With a single point, returns 0.
    """
    n = len(coords)
    if n == 0:
        return 0.0
    cx = sum(c[0] for c in coords) / n
    cy = sum(c[1] for c in coords) / n
    cz = sum(c[2] for c in coords) / n
    msd = sum(
        (c[0] - cx) ** 2 + (c[1] - cy) ** 2 + (c[2] - cz) ** 2
        for c in coords
    ) / n
    return math.sqrt(msd)


def _compactness_score(rg: float, n_points: int) -> float:
    """``1 / (1 + Rg / expected_Rg)`` with ``expected = sqrt(N) * 3.8``.

    Rg = expected → 0.5; Rg = 0 → 1.0; Rg → ∞ → 0. ``n_points`` is the
    number of CA atoms actually used (post-filter), not the input
    binding-residue count.
    """
    if n_points <= 0:
        return 0.0
    expected = math.sqrt(n_points) * _CA_BOND_ANGSTROM
    if expected <= 0:
        return 0.0
    return 1.0 / (1.0 + rg / expected)


def _pick_protein_chain(model):
    """Return the gemmi chain whose residues are mostly amino acids.

    Mirrors ``contact_extractor._classify_chain`` minimally — we only
    need the protein chain here. Returns None when no chain is
    predominantly protein (RNA-only structures, parsing oddities).
    """
    best = None
    best_count = 0
    for chain in model:
        n_aa = 0
        for res in chain:
            try:
                info = gemmi.find_tabulated_residue(res.name)
            except Exception:
                info = None
            if info is not None and "AA" in str(getattr(info, "kind", "")):
                n_aa += 1
        if n_aa > best_count:
            best_count = n_aa
            best = chain
    return best if best_count > 0 else None


def _collect_ca_coords(
    structure_path: str,
    binding_residues: list[int],
) -> tuple[list[tuple[float, float, float]], dict]:
    """Read CA coordinates for the requested binding residues.

    Returns ``(coords, info_dict)``. ``coords`` is empty when the file
    can't be parsed or no binding residue maps to a CA atom; ``info``
    always carries diagnostic fields (``protein_chain_id``,
    ``n_ca_found``, ``missing_residues`` first 20, optional
    ``parse_error``).
    """
    info: dict = {}
    if gemmi is None:
        info["parse_error"] = f"gemmi not available: {_GEMMI_IMPORT_ERROR}"
        return [], info

    path = Path(structure_path)
    if not path.is_file():
        info["parse_error"] = f"file not found: {path}"
        return [], info

    try:
        structure = gemmi.read_structure(
            str(path), merge_chain_parts=True,
        )
    except Exception as exc:
        info["parse_error"] = f"read_structure failed: {exc}"
        return [], info

    if len(structure) == 0:
        info["parse_error"] = "no models in file"
        return [], info

    chain = _pick_protein_chain(structure[0])
    if chain is None:
        info["parse_error"] = "no protein chain found"
        return [], info

    info["protein_chain_id"] = chain.name

    wanted = {int(r) for r in binding_residues}
    found_coords: dict[int, tuple[float, float, float]] = {}
    for res in chain:
        idx = int(res.seqid.num) if res.seqid is not None else None
        if idx is None or idx not in wanted:
            continue
        for atom in res:
            # Backbone CA only; skip alt-locs we already saw.
            if atom.name == "CA" and idx not in found_coords:
                p = atom.pos
                found_coords[idx] = (float(p.x), float(p.y), float(p.z))
                break

    info["n_ca_found"] = len(found_coords)
    missing = sorted(wanted - found_coords.keys())
    info["n_ca_missing"] = len(missing)
    info["missing_residues"] = missing[:20]

    return list(found_coords.values()), info


# ---------- public entry point --------------------------------------------


def structural_plausibility(
    binding_residues: list[int],
    structure_path: Optional[str],
    protein_length: int,
    config: dict,
) -> tuple[Optional[float], dict]:
    """Score how structurally plausible the predicted interface looks.

    Parameters
    ----------
    binding_residues
        1-based protein residue indices from step-5 fusion.
    structure_path
        Path to a Cat A predicted structure (PDB or mmCIF). When
        ``None`` (or unreadable) the compactness sub-score is skipped.
    protein_length
        Total length of the target protein (drives the interface
        ratio).
    config
        Sub-block ``pocket_qa.structural``. Reads
        ``expected_interface_ratio`` (default ``[0.1, 0.4]``),
        ``max_clusters`` (default 3), ``cluster_gap`` (default 5).

    Returns
    -------
    ``(score, info)`` — ``score`` ∈ [0, 1] when at least the continuity
    and ratio sub-scores can be computed, otherwise ``None``. ``info``
    exposes per-axis sub-scores, the cluster count, the interface
    ratio, and (when compactness ran) the Rg, expected Rg and CA-atom
    diagnostics.
    """
    # ----- config with defaults
    band = config.get("expected_interface_ratio") or [0.1, 0.4]
    try:
        ratio_low = float(band[0])
        ratio_high = float(band[1])
    except (TypeError, ValueError, IndexError):
        ratio_low, ratio_high = 0.1, 0.4
    if ratio_low > ratio_high:
        ratio_low, ratio_high = ratio_high, ratio_low
    max_clusters = int(config.get("max_clusters", 3))
    cluster_gap = int(config.get("cluster_gap", 5))

    base_info: dict = {
        "n_binding_input": len(binding_residues or []),
        "protein_length": int(protein_length or 0),
        "config": {
            "expected_interface_ratio": [ratio_low, ratio_high],
            "max_clusters": max_clusters,
            "cluster_gap": cluster_gap,
        },
    }

    # ----- abstain paths
    if not binding_residues:
        return None, {
            **base_info,
            "reason": "binding_residues is empty",
        }
    if protein_length is None or int(protein_length) <= 0:
        return None, {
            **base_info,
            "reason": "protein_length must be > 0",
        }

    # Drop residues outside [1, protein_length]; record for diagnostics.
    in_range: list[int] = []
    out_of_range: list[int] = []
    for r in binding_residues:
        ri = int(r)
        if 1 <= ri <= int(protein_length):
            in_range.append(ri)
        else:
            out_of_range.append(ri)
    base_info["n_out_of_range"] = len(out_of_range)
    base_info["out_of_range_residues"] = out_of_range[:20]
    base_info["n_in_range"] = len(in_range)

    if not in_range:
        return None, {
            **base_info,
            "reason": "every binding residue is out of range",
        }

    # ----- sub-score 1: continuity
    n_clusters = count_sequence_clusters(in_range, gap_threshold=cluster_gap)
    score_continuity = _continuity_score(n_clusters, max_clusters)

    # ----- sub-score 3: interface ratio
    interface_ratio = len(set(in_range)) / float(protein_length)
    score_ratio = _ratio_score(interface_ratio, ratio_low, ratio_high)

    sub_scores: dict[str, Optional[float]] = {
        "continuity": _round_or_none(score_continuity),
        "compactness": None,
        "ratio": _round_or_none(score_ratio),
    }

    info: dict = {
        **base_info,
        "n_clusters": n_clusters,
        "interface_ratio": _round_or_none(interface_ratio),
    }

    # ----- sub-score 2: compactness (only when a structure is available)
    if structure_path:
        coords, parse_info = _collect_ca_coords(str(structure_path), in_range)
        info["structure_path"] = str(structure_path)
        info.update({k: v for k, v in parse_info.items() if k != "parse_error"})
        if "parse_error" in parse_info:
            info["compactness_skip_reason"] = parse_info["parse_error"]
        elif len(coords) < 2:
            info["compactness_skip_reason"] = (
                f"need >= 2 CA atoms, got {len(coords)}"
            )
        else:
            rg = _radius_of_gyration(coords)
            score_compactness = _compactness_score(rg, len(coords))
            sub_scores["compactness"] = _round_or_none(score_compactness)
            info["rg_angstrom"] = _round_or_none(rg)
            info["expected_rg_angstrom"] = _round_or_none(
                math.sqrt(len(coords)) * _CA_BOND_ANGSTROM
            )
    else:
        info["compactness_skip_reason"] = "no structure_path provided"

    # ----- aggregate (mean of computed sub-scores)
    computed = [s for s in sub_scores.values() if s is not None]
    if not computed:
        # Should be unreachable (continuity + ratio always compute when
        # we got past the abstain checks), but defend just in case.
        return None, {
            **info,
            "sub_scores": sub_scores,
            "reason": "no sub-score computable",
        }
    q1 = sum(computed) / len(computed)
    if q1 < 0.0:
        q1 = 0.0
    elif q1 > 1.0:
        q1 = 1.0

    info["sub_scores"] = sub_scores
    info["n_sub_scores_computed"] = len(computed)
    return round(q1, 6), info
