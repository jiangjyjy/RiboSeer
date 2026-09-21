"""Enriched multi-metric + cross-tool fusion (6-tool design).

Why this module exists alongside ``xgb_fusion``
-----------------------------------------------
``XGBFusion`` reuses the WeightOptimizer feature row: 1 score + 1 gate
per tool, a vote count and an AA one-hot block (~31 cols). That throws
away signal each tool already carries — pLDDT, global structure
quality, iptm, and the *rank / z-score* of a residue's score within
its own chain. It also can't see tool-vs-tool agreement except through
shallow tree splits.

This module implements the user's "Cross-Tool Interaction + multi-metric
feature extraction + Fpocket replacing HADDOCK3" spec:

  - 6 fixed tools: boltz2, chai1, rosettafold2na, equipnas, p2rank,
    fpocket (RF2NA re-enabled in single-seq mode — 3rd Cat A tool;
    HADDOCK3 dropped; Fpocket re-enabled — see tool_registry).
  - 34 base per-residue features (multi-metric extraction).
  - 15 score cross-terms + 15 gate cross-terms → 64 total.
  - Missing tool ⇒ all of that tool's columns (incl. gate / rank) = 0,
    same convention as the noisy-OR / WeightOptimizer paths.

Feature layout (column order is the contract — pinned in
``FEATURE_NAMES``; ``feature_set='base'`` is exactly the first
``N_BASE`` columns so the ablation slice is a no-op):

    Boltz-2 (7): dist_score, plddt, gate, plddt_rank, dist_rank,
                 global_plddt, iptm
    Chai-1  (6): dist_score, plddt, gate, plddt_rank, dist_rank,
                 global_plddt
    RF2NA   (6): rf2na_dist_score, rf2na_plddt, rf2na_gate,
                 rf2na_plddt_rank, rf2na_dist_rank, rf2na_global_plddt
                 (tool_id rosettafold2na; ``rf2na`` alias accepted)
    EquiPNAS(4): conf, gate, conf_rank, conf_zscore
    P2Rank  (4): score, gate, score_rank, score_zscore
    Fpocket (4): score, gate, score_rank, score_zscore
    Summary (3): vote_count, catA_agree (≥2 of 3 Cat A gates),
                 n_tools_available
    Cross score (15): pairwise products of the 6 main scores
    Cross gate  (15): pairwise products of the 6 gates

The class mirrors :class:`step5_fusion.xgb_fusion.XGBFusion`'s public
surface (``train`` / ``predict_sample`` / ``save`` / ``load``) so the
evaluate harness can hold either behind one interface.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from .data_collector import (
    load_sample_json,
    per_residue_to_int_dict,
    read_jsonl_record,
)

try:
    import xgboost as xgb  # type: ignore
    XGB_OK = True
    XGB_IMPORT_ERROR: Optional[Exception] = None
except ImportError as _e:  # pragma: no cover — only on a host w/o xgboost
    xgb = None  # type: ignore
    XGB_OK = False
    XGB_IMPORT_ERROR = _e

# LightGBM is an optional learner — same lazy-import pattern as xgboost.
# Used by ``model='lightgbm'``; nothing else in the module depends on it.
try:
    import lightgbm as lgb  # type: ignore
    LGBM_OK = True
    LGBM_IMPORT_ERROR: Optional[Exception] = None
except ImportError as _e:  # pragma: no cover — local dev box may skip this
    lgb = None  # type: ignore
    LGBM_OK = False
    LGBM_IMPORT_ERROR = _e


# ---------------------------------------------------------------------------
# Layout constants — the column order IS the wire contract. Do not reorder
# without bumping the meta version; evaluate_enriched_fusion relies on it.
# ---------------------------------------------------------------------------

# 6 tools. Three Cat A structure predictors (boltz2 / chai1 /
# rosettafold2na — RF2NA re-enabled in single-seq mode) lead, then the
# Cat B/C tools. Order IS the column contract.
TOOL_ORDER: tuple[str, ...] = (
    "boltz2", "chai1", "rosettafold2na", "equipnas", "p2rank", "fpocket",
)

# step4 records use the registry id ``rosettafold2na``; the user's spec
# also refers to it as ``rf2na``. Accept both on read.
TOOL_ID_ALIASES: dict[str, str] = {"rf2na": "rosettafold2na"}

# Short prefix used in FEATURE_NAMES (RF2NA's columns are ``rf2na_*``
# even though its tool_id is ``rosettafold2na``).
NAME_PREFIX: dict[str, str] = {
    "boltz2": "boltz2", "chai1": "chai1",
    "rosettafold2na": "rf2na", "equipnas": "equipnas",
    "p2rank": "p2rank", "fpocket": "fpocket",
}

# Which step4 field holds each tool's *main* score (used for cross-terms
# and the primary per-residue signal). Cat A tools → the PAE/distance
# derived [0,1] interface score; Cat B/C → per_residue_confidence.
MAIN_SCORE_FIELD: dict[str, str] = {
    "boltz2":         "per_residue_pae_score",
    "chai1":          "per_residue_pae_score",
    "rosettafold2na": "per_residue_pae_score",
    "equipnas":       "per_residue_confidence",
    "p2rank":         "per_residue_confidence",
    "fpocket":        "per_residue_confidence",
}

# Cat A tools also carry a pLDDT-style per_residue_confidence + global
# quality (plddt_mean). Only Boltz-2 emits ipTM. Non-Cat-A tools don't.
_CAT_A = ("boltz2", "chai1", "rosettafold2na")


def _canonical_tool_id(tid: str) -> str:
    """Map a step4 tool_id (incl. the ``rf2na`` alias) to its canonical
    TOOL_ORDER id; pass-through for anything else."""
    return TOOL_ID_ALIASES.get(tid or "", tid or "")


def _main_suffix(tool_id: str) -> str:
    """Main-score column suffix for a tool (Cat A → dist_score,
    EquiPNAS → conf, P2Rank/Fpocket → score)."""
    if tool_id in _CAT_A:
        return "dist_score"
    return "conf" if tool_id == "equipnas" else "score"


def _base_names_for(tool_id: str) -> list[str]:
    """Per-tool base feature names (column order = build order)."""
    pfx = NAME_PREFIX[tool_id]
    if tool_id in _CAT_A:
        cols = [f"{pfx}_dist_score", f"{pfx}_plddt", f"{pfx}_gate",
                f"{pfx}_plddt_rank", f"{pfx}_dist_rank",
                f"{pfx}_global_plddt"]
        if tool_id == "boltz2":
            cols.append(f"{pfx}_iptm")  # only Boltz-2 emits ipTM
        return cols
    sfx = _main_suffix(tool_id)
    return [f"{pfx}_{sfx}", f"{pfx}_gate",
            f"{pfx}_{sfx}_rank", f"{pfx}_{sfx}_zscore"]


# Cross-term pair order: every unordered tool pair, in TOOL_ORDER index
# order. C(6, 2) = 15 score products + 15 gate products.
_PAIRS: list[tuple[str, str]] = [
    (TOOL_ORDER[i], TOOL_ORDER[j])
    for i in range(len(TOOL_ORDER))
    for j in range(i + 1, len(TOOL_ORDER))
]

# Base block (per-tool, then 3 summary cols), generated so the names
# can't drift from the build loop / _PAIRS.
_BASE_NAMES: list[str] = []
for _t in TOOL_ORDER:
    _BASE_NAMES += _base_names_for(_t)
_BASE_NAMES += ["vote_count", "catA_agree", "n_tools_available"]

FEATURE_NAMES: list[str] = (
    _BASE_NAMES
    + [f"cross_{NAME_PREFIX[a]}_{NAME_PREFIX[b]}" for a, b in _PAIRS]
    + [f"gate_{NAME_PREFIX[a]}_{NAME_PREFIX[b]}" for a, b in _PAIRS]
)

# 7(boltz2)+6(chai1)+6(rf2na)+4+4+4 + 3 summary = 34
N_BASE = sum(len(_base_names_for(t)) for t in TOOL_ORDER) + 3
N_FULL = len(FEATURE_NAMES)  # 34 + 15 + 15 = 64
assert N_BASE == 34, f"base layout drifted: {N_BASE} != 34"
assert N_FULL == 64, f"feature layout drifted: {N_FULL} != 64"

# ---- neighbourhood context block (appended AFTER the 64 full cols) ------
#
# binding residues cluster along the sequence; a ±W sliding window
# injects that spatial-continuity prior. The context columns are
# computed from the already-built full row (score / gate / vote_count
# columns) so they stay consistent with the base design by construction.
CONTEXT_W = 5  # window half-width → 2*W+1 = 11 positions

CONTEXT_FEATURE_NAMES: list[str] = (
    # per-tool main-score window means (6)
    [f"{NAME_PREFIX[t]}_{_main_suffix(t)}_win5" for t in TOOL_ORDER]
    # per-tool gate window means = local binding density (6)
    + [f"{NAME_PREFIX[t]}_gate_density5" for t in TOOL_ORDER]
    # cross-tool summary window features (3)
    + ["vote_count_win5", "max_score_win5", "binding_streak"]
)
N_CONTEXT = len(CONTEXT_FEATURE_NAMES)  # 6 + 6 + 3 = 15
assert N_CONTEXT == 15, f"context layout drifted: {N_CONTEXT} != 15"

# Column indices into the full row that the context block reads.
_MAIN_COLS = [FEATURE_NAMES.index(f"{NAME_PREFIX[t]}_{_main_suffix(t)}")
              for t in TOOL_ORDER]
_GATE_COLS = [FEATURE_NAMES.index(f"{NAME_PREFIX[t]}_gate")
              for t in TOOL_ORDER]
_VOTE_COL = FEATURE_NAMES.index("vote_count")


def feature_names(feature_set: str = "full",
                  use_context: bool = True) -> list[str]:
    """Column names for the chosen design.

    ``feature_set='base'`` → 34 (no cross-terms, no context — ablation).
    ``feature_set='full'`` → 64; ``+context`` appends 15 → 79.
    """
    if feature_set == "base":
        return list(FEATURE_NAMES[:N_BASE])
    names = list(FEATURE_NAMES)
    if use_context:
        names += CONTEXT_FEATURE_NAMES
    return names


def window_mean(values: list, center: int, w: int) -> float:
    """Mean of ``values[center-w : center+w+1]``, clipped to bounds.

    ``values`` is in sequence order; ``center`` is its list index.
    """
    start = max(0, center - w)
    end = min(len(values), center + w + 1)
    window = values[start:end]
    return sum(window) / len(window) if window else 0.0


def binding_streak(vote_counts: list, center: int) -> int:
    """Length of the contiguous ``vote_count > 0`` run containing
    ``center`` (0 if the centre residue itself has no votes)."""
    if not vote_counts or vote_counts[center] <= 0:
        return 0
    streak = 1
    for d in range(1, len(vote_counts)):
        left_ok = (center - d >= 0) and (vote_counts[center - d] > 0)
        right_ok = (center + d < len(vote_counts)) and (
            vote_counts[center + d] > 0)
        if left_ok:
            streak += 1
        if right_ok:
            streak += 1
        if not left_ok and not right_ok:
            break
    return streak


# ---------------------------------------------------------------------------
# Score-vector transforms (per the spec's compute_rank / compute_zscore)
# ---------------------------------------------------------------------------


def compute_rank(scores: dict[int, float]) -> dict[int, float]:
    """{res_id: score} → {res_id: normalized_rank in [0,1]}.

    Higher score ⇒ higher rank. Single-residue / empty dicts map every
    entry to 0.0 (no rank information).
    """
    if not scores:
        return {}
    ids = sorted(scores.keys(), key=lambda k: scores[k])
    n = len(ids)
    if n == 1:
        return {ids[0]: 0.0}
    return {rid: i / (n - 1) for i, rid in enumerate(ids)}


def compute_zscore(scores: dict[int, float]) -> dict[int, float]:
    """{res_id: score} → {res_id: (score-μ)/σ}; σ≈0 ⇒ all zeros."""
    if not scores:
        return {}
    vals = np.fromiter(scores.values(), dtype=np.float64)
    mu = float(vals.mean())
    sigma = float(vals.std())
    if sigma < 1e-8:
        return {k: 0.0 for k in scores}
    return {k: (float(v) - mu) / sigma for k, v in scores.items()}


# ---------------------------------------------------------------------------
# Per-tool extraction
# ---------------------------------------------------------------------------


def _global_plddt(pred: dict, conf: dict[int, float]) -> float:
    """plddt_mean if the adapter wrote it; else mean of the per-residue
    confidence (pLDDT) map; 0.0 when the tool gave neither."""
    pm = pred.get("plddt_mean")
    if pm is not None:
        try:
            return float(pm)
        except (TypeError, ValueError):
            pass
    if conf:
        return float(np.mean(list(conf.values())))
    return 0.0


class _ToolView:
    """Pre-computed per-residue maps for one tool on one sample.

    Everything a residue row needs is O(1) once this is built:
    main-score dict, pLDDT dict (Cat A), gate set, the rank / z-score
    transforms, and the two shared globals (global_plddt, iptm).
    """

    __slots__ = (
        "present", "main", "main_rank", "plddt", "plddt_rank",
        "main_zscore", "gate", "global_plddt", "iptm",
    )

    def __init__(self) -> None:
        self.present = False
        self.main: dict[int, float] = {}
        self.main_rank: dict[int, float] = {}
        self.main_zscore: dict[int, float] = {}
        self.plddt: dict[int, float] = {}
        self.plddt_rank: dict[int, float] = {}
        self.gate: set[int] = set()
        self.global_plddt = 0.0
        self.iptm = 0.0

    @classmethod
    def from_prediction(cls, tool_id: str, pred: Optional[dict]) -> "_ToolView":
        v = cls()
        if pred is None or not pred.get("success"):
            return v
        v.present = True
        main = per_residue_to_int_dict(pred.get(MAIN_SCORE_FIELD[tool_id]))
        conf = per_residue_to_int_dict(pred.get("per_residue_confidence"))
        # Cat A: main = PAE/dist score, plddt = per_residue_confidence.
        # Cat B/C: main = per_residue_confidence; if MAIN_SCORE_FIELD's
        # field was empty (older record) fall back to confidence so the
        # tool still contributes instead of vanishing.
        if not main and tool_id not in _CAT_A:
            main = conf
        v.main = main
        v.main_rank = compute_rank(main)
        v.main_zscore = compute_zscore(main)
        if tool_id in _CAT_A:
            v.plddt = conf
            v.plddt_rank = compute_rank(conf)
            v.global_plddt = _global_plddt(pred, conf)
            iptm = pred.get("iptm_score")
            try:
                v.iptm = float(iptm) if iptm is not None else 0.0
            except (TypeError, ValueError):
                v.iptm = 0.0
        for r in (pred.get("binding_protein_residues") or []):
            try:
                v.gate.add(int(r))
            except (TypeError, ValueError):
                continue
        return v


# ---------------------------------------------------------------------------
# Feature builder (the spec's build_enriched_features)
# ---------------------------------------------------------------------------


def build_enriched_features(
    step4_data: dict,
    residue_ids: Iterable[int],
    *,
    feature_set: str = "full",
    use_context: bool = True,
) -> np.ndarray:
    """Extract the per-residue feature matrix.

    Shape: ``(n_residues, D)`` where ``D`` is

      - 34  ``feature_set='base'``      (no cross-terms, no context)
      - 64  ``feature_set='full', use_context=False``
      - 79  ``feature_set='full', use_context=True`` (default)

    Parameters
    ----------
    step4_data
        Parsed step4 JSONL record (``{"predictions": [...]}``).
    residue_ids
        Every protein residue id (1-based) for the sample, in output row
        order. Context windows are computed in ascending-residue-id
        (sequence) order regardless of input ordering, then scattered
        back so rows stay aligned with ``residue_ids``.
    feature_set
        ``'full'`` (default) or ``'base'``.
    use_context
        Append the 15-col ±5 neighbourhood block (ignored for
        ``'base'``). Off → 64-col output for the cross-term ablation.

    Missing / failed tools contribute all-zero columns (gate, rank,
    z-score AND their window means) — identical to the noisy-OR /
    WeightOptimizer convention so A/B numbers stay comparable.
    """
    residue_ids = list(residue_ids)
    preds_by_tool: dict[str, dict] = {}
    for p in (step4_data or {}).get("predictions") or []:
        tid = _canonical_tool_id(p.get("tool_id") or "")
        if tid in TOOL_ORDER and p.get("success"):
            # First successful record wins (batch_predict writes one per
            # tool; defensive against accidental dupes).
            preds_by_tool.setdefault(tid, p)

    views = {
        tid: _ToolView.from_prediction(tid, preds_by_tool.get(tid))
        for tid in TOOL_ORDER
    }
    n_tools_available = float(sum(1 for v in views.values() if v.present))

    # Always build the full 64-col row; slice for 'base', append the
    # 15-col context block for 'full'+context. Keeps the column
    # contract a strict prefix relationship across all variants.
    full = np.zeros((len(residue_ids), N_FULL), dtype=np.float64)

    b = views["boltz2"]
    c = views["chai1"]
    rf = views["rosettafold2na"]
    e, p2, fp = views["equipnas"], views["p2rank"], views["fpocket"]

    for row, rid in enumerate(residue_ids):
        # gates first — reused by vote_count / catA_agree / cross gates.
        g = {
            "boltz2":         1.0 if rid in b.gate else 0.0,
            "chai1":          1.0 if rid in c.gate else 0.0,
            "rosettafold2na": 1.0 if rid in rf.gate else 0.0,
            "equipnas":       1.0 if rid in e.gate else 0.0,
            "p2rank":         1.0 if rid in p2.gate else 0.0,
            "fpocket":        1.0 if rid in fp.gate else 0.0,
        }
        # main scores (also the cross-term operands).
        s = {
            "boltz2":         b.main.get(rid, 0.0),
            "chai1":          c.main.get(rid, 0.0),
            "rosettafold2na": rf.main.get(rid, 0.0),
            "equipnas":       e.main.get(rid, 0.0),
            "p2rank":         p2.main.get(rid, 0.0),
            "fpocket":        fp.main.get(rid, 0.0),
        }
        # catA_agree: ≥2 of the 3 Cat A structure tools flag this
        # residue. sum>=2 (not the strict triple product) so the
        # signal survives one Cat A tool being missing/failed.
        cat_a_votes = (g["boltz2"] + g["chai1"]
                       + g["rosettafold2na"])
        f = [
            # Boltz-2 (7)
            s["boltz2"], b.plddt.get(rid, 0.0), g["boltz2"],
            b.plddt_rank.get(rid, 0.0), b.main_rank.get(rid, 0.0),
            b.global_plddt, b.iptm,
            # Chai-1 (6)
            s["chai1"], c.plddt.get(rid, 0.0), g["chai1"],
            c.plddt_rank.get(rid, 0.0), c.main_rank.get(rid, 0.0),
            c.global_plddt,
            # RF2NA (6) — Cat A, same shape as Chai-1 (no ipTM)
            s["rosettafold2na"], rf.plddt.get(rid, 0.0),
            g["rosettafold2na"], rf.plddt_rank.get(rid, 0.0),
            rf.main_rank.get(rid, 0.0), rf.global_plddt,
            # EquiPNAS (4)
            s["equipnas"], g["equipnas"],
            e.main_rank.get(rid, 0.0), e.main_zscore.get(rid, 0.0),
            # P2Rank (4)
            s["p2rank"], g["p2rank"],
            p2.main_rank.get(rid, 0.0), p2.main_zscore.get(rid, 0.0),
            # Fpocket (4)
            s["fpocket"], g["fpocket"],
            fp.main_rank.get(rid, 0.0), fp.main_zscore.get(rid, 0.0),
            # Summary (3)
            sum(g.values()),
            1.0 if cat_a_votes >= 2 else 0.0,
            n_tools_available,
        ]
        f.extend(s[a] * s[bb] for a, bb in _PAIRS)   # cross score (15)
        f.extend(g[a] * g[bb] for a, bb in _PAIRS)   # cross gate (15)
        full[row] = f

    if feature_set == "base":
        return full[:, :N_BASE].copy()
    if not use_context:
        return full

    # ---- neighbourhood context block (15 cols) ----------------------
    # Sequence order = ascending residue id. Compute the windows in
    # that order, then scatter results back to each residue's original
    # output row so the matrix stays aligned with ``residue_ids``.
    n = len(residue_ids)
    order = sorted(range(n), key=lambda p: residue_ids[p])  # row idxs
    ctx = np.zeros((n, N_CONTEXT), dtype=np.float64)

    # Per-tool score / gate arrays in sequence order.
    seq_main = [[full[order[k], col] for k in range(n)]
                for col in _MAIN_COLS]
    seq_gate = [[full[order[k], col] for k in range(n)]
                for col in _GATE_COLS]
    seq_vote = [full[order[k], _VOTE_COL] for k in range(n)]
    seq_maxscore = [max(seq_main[t][k] for t in range(len(TOOL_ORDER)))
                    for k in range(n)]

    for k in range(n):
        vals = []
        vals += [window_mean(seq_main[t], k, CONTEXT_W)
                 for t in range(len(TOOL_ORDER))]          # 6 score win
        vals += [window_mean(seq_gate[t], k, CONTEXT_W)
                 for t in range(len(TOOL_ORDER))]          # 6 gate dens
        vals.append(window_mean(seq_vote, k, CONTEXT_W))   # vote_count_win5
        lo = max(0, k - CONTEXT_W)
        hi = min(n, k + CONTEXT_W + 1)
        vals.append(max(seq_maxscore[lo:hi]))              # max_score_win5
        vals.append(float(binding_streak(seq_vote, k)))    # binding_streak
        ctx[order[k]] = vals

    return np.hstack([full, ctx])


# ---------------------------------------------------------------------------
# Stats helpers (byte-identical to xgb_fusion / evaluate so train-set and
# test-set numbers are directly comparable)
# ---------------------------------------------------------------------------


def _pearson(xs, ys) -> Optional[float]:
    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    if xs.size < 2 or xs.shape != ys.shape:
        return None
    sx, sy = xs.std(), ys.std()
    if sx == 0 or sy == 0:
        return None
    return float(((xs - xs.mean()) * (ys - ys.mean())).mean() / (sx * sy))


def _spearman(xs, ys) -> Optional[float]:
    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    if xs.size < 2 or xs.shape != ys.shape:
        return None

    def _avg_ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        i, n = 0, len(v)
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return ranks

    return _pearson(_avg_ranks(xs), _avg_ranks(ys))


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class EnrichedFusion:
    """XGBoost (default), Ridge, or LightGBM over the enriched feature
    design (34 base / 64 +cross / 79 +context — see
    build_enriched_features)."""

    _META_NAME = "enriched_meta.json"
    _REPORT_NAME = "training_report.json"
    # Per-model artefact name. XGBoost / Ridge share ``model.json`` (one
    # is native xgb json, the other is a small dict of ridge weights —
    # both human-readable). LightGBM is a pickle (full sklearn estimator
    # plus fitted booster + state) because the official Booster
    # round-trip via text file would lose the LGBMRegressor wrapper.
    _MODEL_NAMES: dict = {
        "xgboost": "model.json",
        "ridge": "model.json",
        "lightgbm": "model.pkl",
    }

    # Kept as a class attribute for backwards-compat with anything
    # that grepped ``EnrichedFusion._MODEL_NAME`` (older evaluate /
    # ablation scripts). The dispatch above is the authoritative source.
    _MODEL_NAME = "model.json"

    _DEFAULT_XGB_PARAMS: dict = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "seed": 42,
    }

    # LightGBM defaults pinned to ``ablation_fusion_method.py``'s
    # Table-7 regressor recipe (n_estimators=100, max_depth=4,
    # learning_rate=0.1) — that variant scored Pearson R = 0.555 vs
    # the LGBMClassifier+binary-CE variant's 0.530 on the same 107 test
    # samples. Pearson is rank/scale-invariant, so MSE-fit
    # ``LGBMRegressor`` on {0,1} labels keeps more dynamic range in the
    # predictions than a sigmoid-squashed ``predict_proba``; that's
    # what the 0.025 gap is. We carry the regressor recipe as the
    # single source of truth so ``train_enriched_fusion.py`` + Table 7
    # land at the same number for the LightGBM row.
    _DEFAULT_LGBM_PARAMS: dict = {
        "objective": "regression",
        "metric": "rmse",
        "max_depth": 4,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    SUPPORTED_MODELS: tuple[str, ...] = ("xgboost", "ridge", "lightgbm")

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = dict(config or {})
        self.config = cfg
        self.model_type = str(cfg.get("model", "xgboost")).lower()
        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"model must be one of {self.SUPPORTED_MODELS}; got "
                f"{self.model_type!r}"
            )
        self.feature_set = str(cfg.get("feature_set", "full")).lower()
        if self.feature_set not in ("full", "base"):
            raise ValueError(
                f"feature_set must be 'full' or 'base'; got "
                f"{self.feature_set!r}"
            )
        # Context only applies to the full design; 'base' is always 28.
        self.use_context = (bool(cfg.get("use_context", True))
                            and self.feature_set == "full")
        self.feature_names = feature_names(self.feature_set,
                                           self.use_context)

        # Both param dicts are always built (cheap) but only the active
        # one is used; this keeps save/load round-trip simple — the
        # meta carries whichever block matches the chosen model_type.
        self.xgb_params = dict(self._DEFAULT_XGB_PARAMS)
        for k in ("max_depth", "learning_rate", "n_estimators",
                  "subsample", "colsample_bytree", "min_child_weight",
                  "seed"):
            if k in cfg:
                self.xgb_params[k] = cfg[k]
        self.lgbm_params = dict(self._DEFAULT_LGBM_PARAMS)
        for k in ("max_depth", "learning_rate", "n_estimators",
                  "subsample", "colsample_bytree", "min_child_weight",
                  "random_state", "num_leaves", "n_jobs"):
            if k in cfg:
                self.lgbm_params[k] = cfg[k]
        self.ridge_lambda = float(cfg.get("ridge_lambda", 1.0))

        self.model = None
        self._ridge_w: Optional[np.ndarray] = None
        self._ridge_b: float = 0.0
        self.training_report: dict = {}

    # ---- data collection ------------------------------------------------

    def _collect(
        self,
        step4_dir: Path,
        processed_dir: Path,
        sample_ids: Optional[Iterable[str]],
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        step4_dir = Path(step4_dir)
        processed_dir = Path(processed_dir)
        if sample_ids is not None:
            sids = list(sample_ids)
        else:
            sids = sorted(p.stem for p in step4_dir.glob("*.jsonl"))

        X_blocks: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        n_used = n_skip = 0
        for sid in sids:
            s4 = read_jsonl_record(step4_dir / f"{sid}.jsonl")
            if s4 is None:
                n_skip += 1
                continue
            sample = load_sample_json(processed_dir, sid)
            if sample is None:
                n_skip += 1
                continue
            prot = sample.get("protein") or {}
            length = prot.get("length") or len(prot.get("sequence") or "")
            gt = set(
                (sample.get("interaction") or {}).get(
                    "binding_protein_residues") or []
            )
            if not length or not gt:
                n_skip += 1
                continue
            if not [p for p in (s4.get("predictions") or [])
                    if p.get("success") and p.get("tool_id") in TOOL_ORDER]:
                n_skip += 1
                continue
            residue_ids = list(range(1, int(length) + 1))
            X_blocks.append(build_enriched_features(
                s4, residue_ids, feature_set=self.feature_set,
                use_context=self.use_context))
            y_parts.append(np.fromiter(
                (1.0 if r in gt else 0.0 for r in residue_ids),
                dtype=np.float64, count=len(residue_ids)))
            n_used += 1

        if not X_blocks:
            raise RuntimeError(
                "no usable training rows — check step4_dir has JSONLs "
                "with successful predictions for the 5-tool set and "
                "processed_dir samples carry "
                "interaction.binding_protein_residues."
            )
        X = np.vstack(X_blocks)
        y = np.concatenate(y_parts)
        return X, y, {"n_samples_used": n_used,
                      "n_samples_skipped": n_skip}

    # ---- training -------------------------------------------------------

    def train(
        self,
        *,
        step4_dir: Path,
        processed_dir: Path,
        sample_ids: Optional[Iterable[str]] = None,
        verbose: bool = True,
    ) -> dict:
        X, y, prov = self._collect(step4_dir, processed_dir, sample_ids)
        pos = int((y > 0.5).sum())
        neg = int(y.size - pos)

        if self.model_type == "xgboost":
            if not XGB_OK:
                raise ImportError(
                    f"xgboost required but failed to import "
                    f"({XGB_IMPORT_ERROR}). conda run -n riboseer pip "
                    f"install xgboost"
                )
            spw = float(neg / max(pos, 1))
            self.model = xgb.XGBClassifier(
                **self.xgb_params, scale_pos_weight=spw)
            self.model.fit(X, y, verbose=False)
            y_pred = self.model.predict_proba(X)[:, 1]
            importances = self.model.feature_importances_.tolist()
        elif self.model_type == "lightgbm":
            if not LGBM_OK:
                raise ImportError(
                    f"lightgbm required but failed to import "
                    f"({LGBM_IMPORT_ERROR}). conda run -n riboseer pip "
                    f"install lightgbm"
                )
            # MSE regressor on the {0, 1} target. NOT a classifier —
            # see ``_DEFAULT_LGBM_PARAMS``'s comment for why. No
            # ``scale_pos_weight`` here (regressor doesn't take it);
            # MSE on imbalanced targets handles itself by squashing
            # toward the majority value, and Pearson R doesn't care
            # about that bias since it's location-invariant.
            spw = None
            self.model = lgb.LGBMRegressor(**self.lgbm_params)
            self.model.fit(X, y)
            y_pred = self.model.predict(X)
            importances = self.model.feature_importances_.tolist()
        else:
            spw = None
            self._fit_ridge(X, y, balanced=True)
            y_pred = self._ridge_predict(X)
            # |weight| as a cheap importance proxy for Ridge.
            importances = np.abs(self._ridge_w).tolist()

        pr = _pearson(y_pred, y)
        sr = _spearman(y_pred, y)
        r2 = (pr ** 2) if pr is not None else None
        rmse = float(math.sqrt(((y_pred - y) ** 2).mean()))

        ranked = sorted(zip(self.feature_names, importances),
                        key=lambda kv: -kv[1])
        report = {
            "model_type": self.model_type,
            "feature_set": self.feature_set,
            "use_context": self.use_context,
            "n_rows": int(y.size),
            "n_features": int(X.shape[1]),
            "pos_count": pos,
            "neg_count": neg,
            "pos_rate": float(y.mean()),
            "scale_pos_weight": spw,
            "rmse": rmse,
            "pearson_r": pr,
            "spearman_r": sr,
            "r2": r2,
            "feature_importances": [
                {"feature": f, "importance": float(i)} for f, i in ranked
            ],
            "feature_importances_top": [
                {"feature": f, "importance": float(i)}
                for f, i in ranked[:15]
            ],
            "xgb_params": dict(self.xgb_params)
            if self.model_type == "xgboost" else None,
            "lgbm_params": dict(self.lgbm_params)
            if self.model_type == "lightgbm" else None,
            "ridge_lambda": self.ridge_lambda
            if self.model_type == "ridge" else None,
            **prov,
        }
        self.training_report = report
        if verbose:
            print(f"[enriched/{self.model_type}] n={y.size} "
                  f"d={X.shape[1]} pos={pos} neg={neg} "
                  f"feature_set={self.feature_set} "
                  f"context={self.use_context}")
            print(f"[enriched/{self.model_type}] train Pearson "
                  f"R={pr:.4f} Spearman R={sr:.4f} R²={r2:.4f} "
                  f"RMSE={rmse:.4f}")
            print("[enriched] top features:")
            for e in report["feature_importances_top"][:10]:
                print(f"  {e['feature']:24s}: {e['importance']:.4f}")
        return report

    # ---- ridge (numpy closed form, balanced row weighting) -------------

    def _fit_ridge(self, X: np.ndarray, y: np.ndarray,
                   *, balanced: bool) -> None:
        n, d = X.shape
        if balanced:
            pos = max(int((y > 0.5).sum()), 1)
            neg = max(int(n - pos), 1)
            w = np.where(y > 0.5, n / (2.0 * pos), n / (2.0 * neg))
        else:
            w = np.ones(n)
        Xb = np.hstack([X, np.ones((n, 1))])
        lam = np.full(d + 1, self.ridge_lambda)
        lam[-1] = 0.0
        sw = np.sqrt(w)[:, None]
        Xe, ye = Xb * sw, y * sw.ravel()
        gram = Xe.T @ Xe + np.diag(lam)
        try:
            wb = np.linalg.solve(gram, Xe.T @ ye)
        except np.linalg.LinAlgError:
            wb, *_ = np.linalg.lstsq(gram, Xe.T @ ye, rcond=None)
        self._ridge_w = wb[:-1].astype(np.float64)
        self._ridge_b = float(wb[-1])

    def _ridge_predict(self, X: np.ndarray) -> np.ndarray:
        z = X @ self._ridge_w + self._ridge_b
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    # ---- inference ------------------------------------------------------

    def predict_sample(
        self,
        tool_predictions: list[dict],
        protein_sequence: str,
        protein_length: int,
    ) -> dict[int, float]:
        """Per-residue binding probability for one sample.

        ``protein_sequence`` is accepted for signature parity with the
        other fusion classes; the enriched design is sequence-agnostic
        (no AA one-hot) so it's unused.
        """
        residue_ids = list(range(1, int(protein_length) + 1))
        if not residue_ids:
            return {}
        X = build_enriched_features(
            {"predictions": tool_predictions or []}, residue_ids,
            feature_set=self.feature_set, use_context=self.use_context)
        if self.model_type == "xgboost":
            if self.model is None:
                raise RuntimeError("predict_sample before train()/load()")
            probs = self.model.predict_proba(X)[:, 1]
        elif self.model_type == "lightgbm":
            if self.model is None:
                raise RuntimeError("predict_sample before train()/load()")
            # Regressor predicts raw MSE-fit values. Pearson R is
            # rank/scale-invariant so we don't clamp; downstream
            # consumers (table04_main_results / ablation eval) treat the
            # vector as a relative score, not a probability.
            probs = self.model.predict(X)
        else:
            if self._ridge_w is None:
                raise RuntimeError("predict_sample before train()/load()")
            probs = self._ridge_predict(X)
        return {rid: float(round(float(p), 6))
                for rid, p in zip(residue_ids, probs)}

    # ---- I/O ------------------------------------------------------------

    def save(self, dir_path: Union[str, Path]) -> Path:
        dir_path = Path(dir_path)
        if (self.model_type in ("xgboost", "lightgbm")
                and self.model is None):
            raise RuntimeError("save() before train()/load()")
        if self.model_type == "ridge" and self._ridge_w is None:
            raise RuntimeError("save() before train()/load()")
        dir_path.mkdir(parents=True, exist_ok=True)
        model_file = dir_path / self._MODEL_NAMES[self.model_type]
        if self.model_type == "xgboost":
            self.model.save_model(str(model_file))
        elif self.model_type == "lightgbm":
            # Pickle the full sklearn wrapper (LGBMRegressor) so load()
            # restores both the booster and the feature_importances_ /
            # predict surface without manual re-wiring. The task spec
            # explicitly OK's pickle/joblib here.
            import pickle
            with model_file.open("wb") as f:
                pickle.dump(self.model, f)
        else:
            model_file.write_text(
                json.dumps({"weights": self._ridge_w.tolist(),
                            "bias": self._ridge_b,
                            "ridge_lambda": self.ridge_lambda},
                           indent=2), encoding="utf-8")
        meta = {
            "version": 1,
            "model_type": self.model_type,
            "feature_set": self.feature_set,
            "use_context": self.use_context,
            "feature_names": list(self.feature_names),
            "tool_order": list(TOOL_ORDER),
            "xgb_params": dict(self.xgb_params),
            "lgbm_params": dict(self.lgbm_params),
            "ridge_lambda": self.ridge_lambda,
        }
        (dir_path / self._META_NAME).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8")
        if self.training_report:
            (dir_path / self._REPORT_NAME).write_text(
                json.dumps(self.training_report, indent=2,
                           ensure_ascii=False), encoding="utf-8")
        return dir_path

    @classmethod
    def load(cls, dir_path: Union[str, Path]) -> "EnrichedFusion":
        dir_path = Path(dir_path)
        meta = json.loads(
            (dir_path / cls._META_NAME).read_text(encoding="utf-8"))
        model_type = meta.get("model_type", "xgboost")
        # Only feed the active model's param block into __init__ so the
        # opposite block's defaults aren't accidentally overridden by
        # disk values from an unrelated training run.
        active_params: dict = {}
        if model_type == "xgboost":
            active_params = dict(meta.get("xgb_params") or {})
        elif model_type == "lightgbm":
            active_params = dict(meta.get("lgbm_params") or {})
        out = cls({
            "model": model_type,
            "feature_set": meta.get("feature_set", "full"),
            "use_context": meta.get("use_context", False),
            **active_params,
            "ridge_lambda": meta.get("ridge_lambda", 1.0),
        })
        model_file = dir_path / cls._MODEL_NAMES[out.model_type]
        if out.model_type == "xgboost":
            if not XGB_OK:
                raise ImportError(
                    f"xgboost required to load this bundle "
                    f"({XGB_IMPORT_ERROR})")
            out.model = xgb.XGBClassifier(**out.xgb_params)
            out.model.load_model(str(model_file))
        elif out.model_type == "lightgbm":
            if not LGBM_OK:
                raise ImportError(
                    f"lightgbm required to load this bundle "
                    f"({LGBM_IMPORT_ERROR})")
            import pickle
            with model_file.open("rb") as f:
                out.model = pickle.load(f)
        else:
            d = json.loads(
                model_file.read_text(encoding="utf-8"))
            out._ridge_w = np.asarray(d["weights"], dtype=np.float64)
            out._ridge_b = float(d["bias"])
        rp = dir_path / cls._REPORT_NAME
        if rp.is_file():
            try:
                out.training_report = json.loads(
                    rp.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                out.training_report = {}
        return out
