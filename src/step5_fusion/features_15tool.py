"""Extended feature builder for paper Table 9 (LLM-module ablation).

Why a separate builder
----------------------
``step5_fusion.enriched_fusion.build_enriched_features`` is hard-wired to
the 6-tool ``TOOL_ORDER`` layout (79-D, the current headline). Four other
scripts and the saved ``enriched_v7_model`` bundle depend on that exact
contract, so we do NOT touch it (Table 9 plan §6.4 backward-compat).

Table 9 needs two things the 79-D builder can't express:

1. **Variable tool subset** over the *full* 15-tool library — MAESTRO's
   selection has to bite, so an un-selected (or unavailable) tool's
   columns must be **NaN** (LightGBM splits on missing natively), not 0.
2. **SCOPE profile block (Group 4)** — a sample-level descriptor
   broadcast onto every residue row, so SCOPE on/off changes the feature
   space.

This module supplies ``build_15tool_features`` for exactly that, reusing
the low-level transforms (`compute_rank`, `compute_zscore`,
`window_mean`, `binding_streak`, `_global_plddt`) from
``enriched_fusion`` so the per-tool numbers stay comparable.

Layout (column order is a stable contract — see ``table9_feature_names``)
-------------------------------------------------------------------------
* Base, per tool in ``ALL_KNOWN_TOOLS`` order: 6 cols each
  ``[main, gate, main_rank, main_zscore, secondary, global_quality]``
  → 15 × 6 = 90
* Summary: ``[vote_count, catA_agree, n_tools_active]`` → 3
* Cross-terms over ``CROSS_TOOLS`` pairs (score product + gate product),
  NaN when either tool is inactive → C(6,2) × 2 = 30
* Context (window means over ``CROSS_TOOLS``): 6 score-win + 6 gate-dens
  + ``[vote_win, max_score_win, binding_streak]`` → 15
* SCOPE Group 4: 16

Default (full library, context on) ⇒ 90 + 3 + 30 + 15 + 16 = **154**.

An *inactive* tool = not in ``selected_tools`` **or** absent/failed in
step4. Its 6 base cols are NaN; cross/context cols that reference it are
NaN. Summary counts only active tools.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from step3_tool_selection.tool_registry import (  # noqa: E402
    get_all_tool_ids, get_all_tools,
)
from .data_collector import per_residue_to_int_dict  # noqa: E402
from .enriched_fusion import (  # noqa: E402
    _global_plddt, binding_streak, compute_rank, compute_zscore, window_mean,
)


# ---------------------------------------------------------------------------
# Tool library
# ---------------------------------------------------------------------------

# The 15-tool library, taken straight from the registry (paper Table 1
# order). Order is the feature-column contract — every per-tool block in
# ``build_15tool_features`` follows ALL_KNOWN_TOOLS.
ALL_KNOWN_TOOLS: tuple[str, ...] = tuple(get_all_tool_ids())

TOOL_CATEGORY: dict[str, str] = {t.tool_id: t.category for t in get_all_tools()}

# Tools that emit a full 3D complex → a PAE/distance-derived [0,1]
# interface score in ``per_residue_pae_score`` plus a per-residue pLDDT
# in ``per_residue_confidence``. The rest score residues directly into
# ``per_residue_confidence``.
_STRUCTURE_CATS = ("A", "D")


def _is_structure(tool_id: str) -> bool:
    return TOOL_CATEGORY.get(tool_id, "C") in _STRUCTURE_CATS


def main_score_field(tool_id: str) -> str:
    """step4 field holding the tool's main per-residue signal."""
    return ("per_residue_pae_score" if _is_structure(tool_id)
            else "per_residue_confidence")


# Cross-term / context tools — bounded to the original 6-tool core so the
# pairwise block stays C(6,2)×2 = 30 cols (matching the 79-D design)
# instead of C(15,2)×2 = 210. New library tools still contribute through
# their 6 base columns; only the interaction/window blocks are core-only.
CROSS_TOOLS: tuple[str, ...] = (
    "boltz2", "chai1", "rosettafold2na", "equipnas", "p2rank", "fpocket",
)
_CROSS_PAIRS: list[tuple[str, str]] = [
    (CROSS_TOOLS[i], CROSS_TOOLS[j])
    for i in range(len(CROSS_TOOLS))
    for j in range(i + 1, len(CROSS_TOOLS))
]

_PER_TOOL = 6  # main, gate, main_rank, main_zscore, secondary, global_q
_PER_TOOL_COLS = ["main", "gate", "main_rank", "main_zscore",
                  "secondary", "global_q"]


# ---------------------------------------------------------------------------
# SCOPE profile (Group 4)
# ---------------------------------------------------------------------------

SCOPE_FAMILIES: tuple[str, ...] = (
    "RRM", "KH", "ZnF", "dsRBD", "Multi", "Novel",
)
SCOPE_RNA_CONTEXTS: tuple[str, ...] = (
    "single-strand", "stem-loop", "internal-loop", "junction",
    "G-quadruplex", "unstructured",
)
_DIFFICULTY = {"easy": 0.0, "medium": 0.5, "hard": 1.0}

# Length normalisers for the two automatic bins (residues / nucleotides).
# Chosen so typical complexes land in (0, 1); longer ones clamp at 1.
PROT_LEN_NORM = 500.0
RNA_LEN_NORM = 200.0

SCOPE_FEATURE_NAMES: list[str] = (
    [f"scope_family_{f}" for f in SCOPE_FAMILIES]
    + [f"scope_rna_{c}" for c in SCOPE_RNA_CONTEXTS]
    + ["scope_difficulty", "scope_protein_len_bin",
       "scope_rna_len_bin", "scope_confidence"]
)
SCOPE_FEATURE_DIM = len(SCOPE_FEATURE_NAMES)  # 6 + 6 + 4 = 16


def _one_hot(value: Optional[str], vocab: Sequence[str],
             default: str) -> list[float]:
    v = (value or "").strip()
    # Case-insensitive match against the vocabulary; unknown → default.
    lookup = {x.lower(): x for x in vocab}
    canonical = lookup.get(v.lower(), default)
    return [1.0 if x == canonical else 0.0 for x in vocab]


def encode_scope_profile(
    profile: Optional[dict],
    protein_len: int,
    rna_len: int,
) -> np.ndarray:
    """Encode a SCOPE profile dict into the 16-D Group-4 vector.

    ``profile`` may be ``None`` (→ all-zero block, the SCOPE-off arm) or a
    dict with any subset of ``protein_family`` / ``rna_context`` /
    ``difficulty`` / ``confidence``. The two length bins are always
    computed from ``protein_len`` / ``rna_len`` (they're automatic, not
    LLM-derived) — but when ``profile is None`` the whole block is zeroed
    so SCOPE-off truly removes the module's influence.
    """
    if profile is None:
        return np.zeros(SCOPE_FEATURE_DIM, dtype=np.float64)

    fam = _one_hot(profile.get("protein_family"), SCOPE_FAMILIES, "Novel")
    rna = _one_hot(profile.get("rna_context"), SCOPE_RNA_CONTEXTS,
                   "unstructured")
    diff_raw = str(profile.get("difficulty", "medium")).strip().lower()
    difficulty = _DIFFICULTY.get(diff_raw, 0.5)
    prot_bin = min(max(float(protein_len), 0.0) / PROT_LEN_NORM, 1.0)
    rna_bin = min(max(float(rna_len), 0.0) / RNA_LEN_NORM, 1.0)
    try:
        conf = float(profile.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = min(max(conf, 0.0), 1.0)

    vec = fam + rna + [difficulty, prot_bin, rna_bin, conf]
    return np.asarray(vec, dtype=np.float64)


def auto_scope_profile(sample: dict) -> dict:
    """Heuristic ``cauto`` profile — no LLM call (Table 9 plan §1.3).

    Used as the automated stand-in for the LLM profile: family/context fall
    back to the generic class (no Pfam/structure annotation is assumed to
    be present in the sample JSON), difficulty follows a simple
    length-based rule, and ``confidence`` is fixed low to mark it as
    non-LLM. The length bins downstream still carry real signal.
    """
    prot = sample.get("protein") or {}
    rna = sample.get("rna") or sample.get("ligand_rna") or {}
    plen = int(prot.get("length") or len(prot.get("sequence") or "") or 0)
    rlen = int(rna.get("length") or len(rna.get("sequence") or "") or 0)

    # Difficulty: long protein or long RNA or both present large → harder.
    score = 0
    if plen > 300:
        score += 1
    if rlen > 80:
        score += 1
    if plen > 150 and rlen > 40:
        score += 1
    difficulty = "easy" if score == 0 else "medium" if score == 1 else "hard"

    family = (prot.get("family") or prot.get("domain_class") or "Novel")
    rna_context = (rna.get("structure_context") or "unstructured")
    return {
        "sample_id": sample.get("sample_id"),
        "protein_family": family,
        "rna_context": rna_context,
        "difficulty": difficulty,
        "confidence": 0.30,  # marks this as the automated descriptor
        "source": "cauto",
    }


# ---------------------------------------------------------------------------
# Per-tool view (lightweight; works for all 15 tools)
# ---------------------------------------------------------------------------


class _Tool:
    """Per-residue maps for one tool on one sample, for any category."""

    __slots__ = ("active", "main", "main_rank", "main_zscore",
                 "secondary", "gate", "global_q", "is_struct")

    def __init__(self) -> None:
        self.active = False
        self.main: dict[int, float] = {}
        self.main_rank: dict[int, float] = {}
        self.main_zscore: dict[int, float] = {}
        self.secondary: dict[int, float] = {}
        self.gate: set[int] = set()
        self.global_q = 0.0
        self.is_struct = False

    @classmethod
    def build(cls, tool_id: str, pred: Optional[dict],
              selected: bool) -> "_Tool":
        v = cls()
        v.is_struct = _is_structure(tool_id)
        if not selected or pred is None or not pred.get("success"):
            return v  # inactive → all-NaN columns downstream
        v.active = True
        main = per_residue_to_int_dict(pred.get(main_score_field(tool_id)))
        conf = per_residue_to_int_dict(pred.get("per_residue_confidence"))
        # Non-structure tool with an empty pae field: fall back to conf so
        # the tool still contributes (mirrors enriched_fusion).
        if not main and not v.is_struct:
            main = conf
        v.main = main
        v.main_rank = compute_rank(main)
        v.main_zscore = compute_zscore(main)
        if v.is_struct:
            # secondary = per-residue pLDDT; global_q = mean pLDDT.
            v.secondary = conf
            v.global_q = _global_plddt(pred, conf)
        else:
            # secondary = the tool's own confidence again (rank-stable
            # second view); global_q = mean confidence.
            v.secondary = conf
            v.global_q = (float(np.mean(list(conf.values())))
                          if conf else 0.0)
        for r in (pred.get("binding_protein_residues") or []):
            try:
                v.gate.add(int(r))
            except (TypeError, ValueError):
                continue
        return v


def _canonical(tid: str) -> str:
    """Map step4 tool_id aliases to the library id."""
    aliases = {"rf2na": "rosettafold2na", "rosettafoldaa": "rfaa",
               "af3": "alphafold3"}
    return aliases.get(tid or "", tid or "")


def _index_predictions(step4_data: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in (step4_data or {}).get("predictions") or []:
        tid = _canonical(p.get("tool_id") or "")
        if tid in TOOL_CATEGORY and p.get("success"):
            out.setdefault(tid, p)
    return out


# ---------------------------------------------------------------------------
# Feature names (column contract)
# ---------------------------------------------------------------------------


def table9_feature_names(*, use_context: bool = True,
                         use_scope: bool = True) -> list[str]:
    names: list[str] = []
    for t in ALL_KNOWN_TOOLS:
        names += [f"{t}_{c}" for c in _PER_TOOL_COLS]
    names += ["vote_count", "catA_agree", "n_tools_active"]
    names += [f"cross_{a}_{b}" for a, b in _CROSS_PAIRS]
    names += [f"gate_{a}_{b}" for a, b in _CROSS_PAIRS]
    if use_context:
        names += [f"{t}_main_win5" for t in CROSS_TOOLS]
        names += [f"{t}_gate_density5" for t in CROSS_TOOLS]
        names += ["vote_count_win5", "max_score_win5", "binding_streak"]
    if use_scope:
        names += list(SCOPE_FEATURE_NAMES)
    return names


N_BASE_T9 = len(ALL_KNOWN_TOOLS) * _PER_TOOL + 3          # 93
N_CROSS_T9 = len(_CROSS_PAIRS) * 2                         # 30
N_CONTEXT_T9 = len(CROSS_TOOLS) * 2 + 3                    # 15


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------


def build_15tool_features(
    step4_data: dict,
    residue_ids: Iterable[int],
    selected_tools: Optional[Iterable[str]] = None,
    scope_vector: Optional[np.ndarray] = None,
    *,
    use_context: bool = True,
    use_scope: bool = True,
) -> np.ndarray:
    """Build the Table 9 per-residue feature matrix.

    Parameters
    ----------
    step4_data
        Parsed step4 JSONL record (``{"predictions": [...]}``).
    residue_ids
        1-based protein residue ids in output-row order.
    selected_tools
        MAESTRO's chosen subset. ``None`` → every library tool is
        eligible (still gated by step4 availability). A tool that is
        unselected *or* absent/failed gets all-NaN columns.
    scope_vector
        Pre-encoded 16-D SCOPE block (see ``encode_scope_profile``),
        broadcast to every row. ``None`` with ``use_scope=True`` → zeros.
    use_context / use_scope
        Toggle the context (15) / SCOPE (16) blocks.

    Returns
    -------
    ``(n_residues, D)`` float64 array. Inactive-tool cells are ``np.nan``.
    """
    residue_ids = list(residue_ids)
    n = len(residue_ids)
    sel = set(ALL_KNOWN_TOOLS if selected_tools is None else selected_tools)
    sel = {_canonical(t) for t in sel}

    preds = _index_predictions(step4_data)
    tools = {
        t: _Tool.build(t, preds.get(t), selected=(t in sel))
        for t in ALL_KNOWN_TOOLS
    }

    nan = float("nan")
    n_active = float(sum(1 for t in ALL_KNOWN_TOOLS if tools[t].active))

    base = np.full((n, N_BASE_T9), nan, dtype=np.float64)
    cross = np.full((n, N_CROSS_T9), nan, dtype=np.float64)

    for row, rid in enumerate(residue_ids):
        # ---- per-tool base block ----
        col = 0
        gates: dict[str, float] = {}
        scores: dict[str, float] = {}
        for t in ALL_KNOWN_TOOLS:
            tv = tools[t]
            if tv.active:
                g = 1.0 if rid in tv.gate else 0.0
                s = tv.main.get(rid, 0.0)
                base[row, col:col + _PER_TOOL] = [
                    s, g,
                    tv.main_rank.get(rid, 0.0),
                    tv.main_zscore.get(rid, 0.0),
                    tv.secondary.get(rid, 0.0),
                    tv.global_q,
                ]
                gates[t] = g
                scores[t] = s
            # inactive → leave NaN (already filled)
            col += _PER_TOOL

        # ---- summary block (active tools only) ----
        vote_count = float(sum(gates.values()))
        cat_a_votes = sum(gates.get(t, 0.0) for t in
                          ("boltz2", "chai1", "rosettafold2na"))
        base[row, col] = vote_count
        base[row, col + 1] = 1.0 if cat_a_votes >= 2 else 0.0
        base[row, col + 2] = n_active

        # ---- cross-term block (core tools) ----
        ci = 0
        for a, b in _CROSS_PAIRS:
            if a in scores and b in scores:
                cross[row, ci] = scores[a] * scores[b]
            ci += 1
        for a, b in _CROSS_PAIRS:
            if a in gates and b in gates:
                cross[row, ci] = gates[a] * gates[b]
            ci += 1

    blocks = [base, cross]

    # ---- context block ----
    if use_context:
        ctx = _build_context(base, residue_ids, tools)
        blocks.append(ctx)

    # ---- SCOPE block ----
    if use_scope:
        if scope_vector is None:
            sv = np.zeros(SCOPE_FEATURE_DIM, dtype=np.float64)
        else:
            sv = np.asarray(scope_vector, dtype=np.float64).ravel()
            if sv.size != SCOPE_FEATURE_DIM:
                raise ValueError(
                    f"scope_vector has {sv.size} dims, "
                    f"expected {SCOPE_FEATURE_DIM}")
        blocks.append(np.tile(sv, (n, 1)))

    return np.hstack(blocks)


def _build_context(base: np.ndarray, residue_ids: list[int],
                   tools: dict[str, "_Tool"]) -> np.ndarray:
    """±5 window means over the CROSS_TOOLS core. Inactive tool → its
    window columns are NaN (consistent with its NaN base columns)."""
    n = len(residue_ids)
    ctx = np.full((n, N_CONTEXT_T9), float("nan"), dtype=np.float64)
    # Map each core tool to its (main_col, gate_col) in ``base``.
    main_col = {t: ALL_KNOWN_TOOLS.index(t) * _PER_TOOL for t in CROSS_TOOLS}
    gate_col = {t: ALL_KNOWN_TOOLS.index(t) * _PER_TOOL + 1
                for t in CROSS_TOOLS}
    vote_col = len(ALL_KNOWN_TOOLS) * _PER_TOOL  # first summary col

    order = sorted(range(n), key=lambda p: residue_ids[p])
    active_core = [t for t in CROSS_TOOLS if tools[t].active]

    # Sequence-ordered arrays (only meaningful for active tools).
    seq_main = {t: [base[order[k], main_col[t]] for k in range(n)]
                for t in active_core}
    seq_gate = {t: [base[order[k], gate_col[t]] for k in range(n)]
                for t in active_core}
    seq_vote = [base[order[k], vote_col] for k in range(n)]
    if active_core:
        seq_max = [max(seq_main[t][k] for t in active_core)
                   for k in range(n)]
    else:
        seq_max = [0.0] * n

    for k in range(n):
        vals: list[float] = []
        for ti, t in enumerate(CROSS_TOOLS):
            vals.append(window_mean(seq_main[t], k, 5)
                        if t in active_core else float("nan"))
        for t in CROSS_TOOLS:
            vals.append(window_mean(seq_gate[t], k, 5)
                        if t in active_core else float("nan"))
        vals.append(window_mean(seq_vote, k, 5))
        lo, hi = max(0, k - 5), min(n, k + 5 + 1)
        vals.append(max(seq_max[lo:hi]) if active_core else 0.0)
        vals.append(float(binding_streak(seq_vote, k)))
        ctx[order[k]] = vals
    return ctx
