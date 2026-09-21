"""Shared retrain-from-scratch evaluation for Table 23 / 24.

Both tables retrain the LightGBM fusion model (the LLM modules are frozen —
no SCOPE/MAESTRO/POLISH, i.e. the Table-9 off/off/off deterministic
pipeline: the 5 default tools + LightGBM) under different train/test splits
(Table 23) or seeds (Table 24), and never re-call the LLM or re-run tools.

Reuses the project pipeline:
  * GT + resolved-residue mask: ``interaction.binding_protein_residues`` /
    ``protein.resolved_residues`` (same as ``collect_sample_data``).
  * Features: ``features_15tool.build_15tool_features`` with the default
    tool subset, SCOPE block zeroed (scope off) — the off/off/off design.
  * Model: LightGBM regressor with the project recipe
    (n_estimators=100, max_depth=4, lr=0.1), seedable.
  * Metric: per-sample Pearson/Spearman/R² on the resolved subset
    (``ablation_fusion_method.per_sample_corr``).

step4 records are looked up across MULTIPLE dirs (train + test) because a
re-split moves samples between them — ``find_step4`` probes each in order.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step5_fusion.data_collector import (  # noqa: E402
    load_sample_json, read_jsonl_record,
)
from step5_fusion.features_15tool import (  # noqa: E402
    build_15tool_features, encode_scope_profile, _canonical, _index_predictions,
)
from scripts.riboseer.ablation_fusion_method import (  # noqa: E402
    per_sample_corr, _tool_raw_scores,
)

try:
    import lightgbm as lgb  # type: ignore
    LGBM_OK = True
except ImportError:  # pragma: no cover
    lgb = None  # type: ignore
    LGBM_OK = False

DEFAULT_TOOLS = ["boltz2", "chai1", "rosettafold2na", "equipnas", "p2rank"]


# ---------------------------------------------------------------------------
# Sample bundle
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    sid: str
    X: np.ndarray            # (L, D) off/off/off feature matrix
    y: np.ndarray            # (L,) binary GT over residue_ids
    eval_mask: np.ndarray    # (L,) resolved-residue subset
    residue_ids: list[int]
    boltz: np.ndarray        # (L,) raw boltz2 per-residue score


def find_step4(step4_dirs: list[Path], sid: str) -> Optional[dict]:
    """First step4 record for ``sid`` across the given dirs (train then
    test), so a re-split that moves a sample still resolves its tools."""
    for d in step4_dirs:
        rec = read_jsonl_record(d / f"{sid}.jsonl")
        if rec is not None:
            return rec
    return None


def _rna_len(sample: dict) -> int:
    rna = sample.get("rna") or sample.get("ligand_rna") or {}
    return int(rna.get("length") or len(rna.get("sequence") or "") or 0)


def _vec(scores: dict[int, float], residue_ids: list[int]) -> np.ndarray:
    return np.fromiter((float(scores.get(r, 0.0)) for r in residue_ids),
                       dtype=np.float64, count=len(residue_ids))


def collect_samples(step4_dirs: list[Path], data_dir: Path,
                    sample_ids: list[str],
                    default_tools: list[str]) -> dict[str, Sample]:
    """``{sid: Sample}`` for every usable sample (GT + length present + at
    least one default tool succeeded). Collected ONCE over all 332 ids; the
    callers slice it per split/seed."""
    tools = [_canonical(t) for t in default_tools]
    out: dict[str, Sample] = {}
    for sid in sample_ids:
        s4 = find_step4(step4_dirs, sid)
        if s4 is None:
            continue
        sample = load_sample_json(data_dir, sid)
        if sample is None:
            continue
        prot = sample.get("protein") or {}
        length = int(prot.get("length") or len(prot.get("sequence") or "") or 0)
        gt = {int(r) for r in (sample.get("interaction") or {})
              .get("binding_protein_residues") or []}
        if not length or not gt:
            continue
        preds = _index_predictions(s4)
        if not any(t in preds for t in tools):
            continue

        residue_ids = list(range(1, length + 1))
        y = np.fromiter((1.0 if r in gt else 0.0 for r in residue_ids),
                        dtype=np.float64, count=length)
        resolved = prot.get("resolved_residues") or []
        if resolved:
            rs = {int(r) for r in resolved}
            mask = np.fromiter((r in rs for r in residue_ids),
                               dtype=bool, count=length)
        else:
            mask = np.ones(length, dtype=bool)

        scope_vec = encode_scope_profile(None, length, _rna_len(sample))
        X = build_15tool_features(s4, residue_ids, selected_tools=tools,
                                  scope_vector=scope_vec, use_context=True,
                                  use_scope=True)
        boltz_pred = preds.get("boltz2")
        boltz = _vec(_tool_raw_scores(boltz_pred) if boltz_pred else {},
                     residue_ids)
        out[sid] = Sample(sid, X, y, mask, residue_ids, boltz)
    return out


# ---------------------------------------------------------------------------
# Train / predict
# ---------------------------------------------------------------------------


def make_lgbm(seed: int):
    """LightGBM regressor with the project recipe (matches
    table09_llm_modules / enriched_fusion), seeded."""
    if not LGBM_OK:
        raise RuntimeError("lightgbm not installed (pip install lightgbm)")
    return lgb.LGBMRegressor(  # type: ignore[union-attr]
        n_estimators=100, max_depth=4, learning_rate=0.1,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        objective="regression", random_state=seed, verbose=-1)


def train_predict(train: list[Sample], test: list[Sample],
                  seed: int) -> dict[str, np.ndarray]:
    """Train one LightGBM on ``train`` and return ``{sid: pred_vector}``
    over ``test`` (full residue range, pre-mask)."""
    X = np.vstack([s.X for s in train])
    y = np.concatenate([s.y for s in train])
    model = make_lgbm(seed)
    model.fit(X, y)
    return {s.sid: np.asarray(model.predict(s.X), dtype=np.float64)
            for s in test}


# ---------------------------------------------------------------------------
# Per-sample correlations (masked to resolved residues)
# ---------------------------------------------------------------------------


def riboseer_corrs(test: list[Sample],
                   preds: dict[str, np.ndarray]) -> list[dict]:
    out = []
    for s in test:
        m = s.eval_mask
        c = per_sample_corr(np.asarray(preds[s.sid])[m], s.y[m])
        if c is not None:
            out.append(c)
    return out


def boltz_corrs(test: list[Sample]) -> list[dict]:
    out = []
    for s in test:
        m = s.eval_mask
        c = per_sample_corr(s.boltz[m], s.y[m])
        if c is not None:
            out.append(c)
    return out


def mean_metric(corrs: list[dict], key: str) -> Optional[float]:
    vals = [c[key] for c in corrs if c.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def load_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"sample list not found: {path}")
    return [ln.split()[0] for ln in
            path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def all_ids(train_list: Path, test_list: Path) -> list[str]:
    """Union of the two splits, order-preserving (train then test)."""
    seen: set[str] = set()
    out: list[str] = []
    for p in (train_list, test_list):
        for sid in load_ids(p):
            if sid not in seen:
                seen.add(sid)
                out.append(sid)
    return out
