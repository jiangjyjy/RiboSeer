"""Shared IO for the RiboSeer **full-system (on/on/on)** per-sample
predictions — the headline 0.565 pipeline (SCOPE + MAESTRO + POLISH).

The headline number stopped coming from a model bundle on disk: it is the
``table09_llm_modules`` on/on/on row (15-tool variable selection + SCOPE
G4 block + POLISH soft edits). To let every downstream table read that
*exact* prediction instead of re-deriving an off/off/off model output,
``generate_fullsystem_predictions`` (or ``table09_llm_modules
--save-predictions-dir``) dumps one JSON per test sample and the table
scripts read it back with ``--predictions-dir``.

On-disk format — ``<dir>/<sample_id>.json``::

    {"residue_ids": [1, 2, 3, ...], "probabilities": [0.1, 0.9, ...]}

``residue_ids`` are 1-based protein residue ids in the same order the
table scripts iterate (``range(1, length+1)``); ``probabilities`` is the
post-POLISH fused probability for each.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def save_predictions(out_dir: Path, sid: str, residue_ids: Iterable[int],
                     probabilities: Iterable[float]) -> Path:
    """Write one sample's prediction to ``<out_dir>/<sid>.json`` and return
    the path. ``residue_ids`` / ``probabilities`` are aligned sequences."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rids = [int(r) for r in residue_ids]
    probs = [float(p) for p in probabilities]
    if len(rids) != len(probs):
        raise ValueError(
            f"{sid}: residue_ids ({len(rids)}) != probabilities ({len(probs)})")
    path = out_dir / f"{sid}.json"
    path.write_text(json.dumps({"residue_ids": rids, "probabilities": probs}),
                    encoding="utf-8")
    return path


def save_prob_dict(out_dir: Path, sid: str,
                   prob: dict[int, float],
                   residue_ids: Optional[Iterable[int]] = None) -> Path:
    """Convenience wrapper: dump a ``{res_id: prob}`` dict. When
    ``residue_ids`` is given the output keeps that order (and fills 0.0 for
    any residue absent from ``prob``); otherwise the dict's own sorted
    key order is used."""
    if residue_ids is None:
        rids = sorted(prob)
        probs = [prob[r] for r in rids]
    else:
        rids = [int(r) for r in residue_ids]
        probs = [float(prob.get(r, 0.0)) for r in rids]
    return save_predictions(out_dir, sid, rids, probs)


def load_predictions_dir(directory: Optional[Path]) -> dict[str, dict[int, float]]:
    """Load ``{sample_id: {res_id: prob}}`` from ``<dir>/<sid>.json``.

    Missing dir / unreadable file → that sample is simply absent (callers
    treat an absent sample as "no RiboSeer prediction"). Never raises on a
    malformed file; it's skipped."""
    out: dict[str, dict[int, float]] = {}
    if directory is None or not Path(directory).is_dir():
        return out
    for f in Path(directory).glob("*.json"):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rids = doc.get("residue_ids")
        probs = doc.get("probabilities")
        if not isinstance(rids, list) or not isinstance(probs, list):
            continue
        if len(rids) != len(probs):
            continue
        try:
            out[f.stem] = {int(r): float(p) for r, p in zip(rids, probs)}
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Saved on/on/on LightGBM model (for Table 27 feature importance)
# ---------------------------------------------------------------------------

MODEL_FILE = "model.txt"                  # raw LightGBM booster dump
MODEL_FEATURES_FILE = "model_feature_names.json"   # real column names sidecar


def save_lightgbm_model(out_dir: Path, model,
                        feature_names: Iterable[str]) -> Path:
    """Dump a fitted LightGBM model (``LGBMRegressor`` or raw ``Booster``)
    to ``<out_dir>/model.txt`` plus the real ``feature_names`` sidecar.

    The booster only remembers generic ``Column_i`` names when trained on a
    numpy array, so the design names (``table9_feature_names``) are stored
    alongside and zipped back by gain index in ``load_lightgbm_gain``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    booster = getattr(model, "booster_", None) or model  # LGBMRegressor / Booster
    booster.save_model(str(out_dir / MODEL_FILE))
    names = [str(n) for n in feature_names]
    (out_dir / MODEL_FEATURES_FILE).write_text(
        json.dumps({"feature_names": names, "model_type": "lightgbm"}),
        encoding="utf-8")
    return out_dir / MODEL_FILE


def has_lightgbm_model(model_dir: Optional[Path]) -> bool:
    return model_dir is not None and (Path(model_dir) / MODEL_FILE).is_file()


def load_lightgbm_model(model_dir: Path):
    """Load a booster saved by :func:`save_lightgbm_model`.

    Returns the raw ``lightgbm.Booster`` (``predict`` on a 2-D array gives the
    per-row scores the fusion expects)."""
    import lightgbm as lgb  # lazy: the model file is the only hard dependency

    md = Path(model_dir)
    model_file = md / MODEL_FILE
    if not model_file.is_file():
        raise FileNotFoundError(f"no LightGBM model under {md}")
    return lgb.Booster(model_file=str(model_file))


def load_lightgbm_gain(model_dir: Path) -> tuple[list[str], list[float]]:
    """``(feature_names, gain_importances)`` for the saved on/on/on
    booster, aligned by column index. Feature names come from the sidecar
    (falling back to the booster's own names, then generic ``f{i}``)."""
    import lightgbm as lgb  # lazy: not installed on the local dev box

    md = Path(model_dir)
    booster = lgb.Booster(model_file=str(md / MODEL_FILE))
    gains = [float(g) for g in
             booster.feature_importance(importance_type="gain")]
    names: list[str] = []
    sidecar = md / MODEL_FEATURES_FILE
    if sidecar.is_file():
        try:
            names = list(json.loads(sidecar.read_text(encoding="utf-8"))
                         .get("feature_names") or [])
        except (OSError, json.JSONDecodeError):
            names = []
    if len(names) != len(gains):
        names = [str(n) for n in booster.feature_name()]
    if len(names) != len(gains):
        names = [f"f{i}" for i in range(len(gains))]
    return names, gains


def predictions_to_vector(pred_by_res: Optional[dict[int, float]],
                          residue_ids: Iterable[int]) -> np.ndarray:
    """Full-length score vector aligned to ``residue_ids`` (1-based, in row
    order). Residues missing from ``pred_by_res`` (or an absent sample,
    ``pred_by_res is None``) default to 0.0 — the same convention
    ``_tool_raw_vector`` uses, so the masked correlation stays well
    defined."""
    pred_by_res = pred_by_res or {}
    rids = list(residue_ids)
    return np.fromiter((float(pred_by_res.get(int(r), 0.0)) for r in rids),
                       dtype=np.float64, count=len(rids))
