"""Per-sample correlation metrics used across the fusion code.

One implementation, imported by both the pipeline (HARMONY, VERDICT) and
the experiment scripts, so a "Table 4 row" means the same thing
everywhere.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .enriched_fusion import _pearson, _spearman


def per_sample_corr(pred: np.ndarray, gt: np.ndarray
                    ) -> Optional[dict]:
    """Pearson / Spearman / R² for one (sample, method). None when the
    correlation is undefined (constant pred or constant GT)."""
    if pred.size < 2 or pred.shape != gt.shape:
        return None
    if pred.max() - pred.min() <= 1e-12:
        return None
    if gt.max() - gt.min() <= 1e-12:
        return None
    pr = _pearson(pred, gt)
    if pr is None:
        return None
    sr = _spearman(pred, gt)
    return {
        "pearson_r": round(pr, 4),
        "spearman_r": None if sr is None else round(sr, 4),
        "r_squared": round(pr * pr, 4),
    }
