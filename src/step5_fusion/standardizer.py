"""Per-tool z-score → sigmoid normaliser for the learned-fusion path.

Why standardise per tool
------------------------
Each Cat A/B/C tool reports a per-residue score on its own scale and
distribution: EquiPNAS probabilities cluster low (median ~0.05), Boltz-2
PAE-derived scores cover (0.07, 1.0], P2Rank's ligandability sits near 0
for most residues with a thin upper tail, etc. Feeding the raw values
straight into a Ridge regression makes the learned weights coupled to
each tool's idiosyncratic mean / variance instead of its actual
discriminative signal — a tool whose distribution is centred at 0.05
gets a tiny coefficient even when it ranks residues well.

Z-score per tool removes the mean/variance bias; the sigmoid then maps
the standardised value back into [0, 1] so the optimiser sees features
in a comparable range (and the downstream "fused probability" stays
interpretable as a probability).

Persistence
-----------
``ScoreStandardizer`` round-trips through plain JSON so the model
artefact lands next to the optimiser's weights — same dir, two files,
no pickle / no version coupling.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Optional, Union


# Lower bound on σ. Tools with effectively-constant scores would
# otherwise blow up the (x-μ)/σ division. The bound is tight enough
# (1e-8) that real distributions are unaffected.
_MIN_STD = 1e-8


def _sigmoid(z: float) -> float:
    """Numerically stable σ(z) = 1/(1+exp(-z)).

    Plain ``1/(1+exp(-z))`` overflows for very negative z. The split
    keeps ``exp`` argument non-positive on both branches.
    """
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


class ScoreStandardizer:
    """Per-tool z-score normaliser, sigmoid-mapped into [0, 1].

    Usage
    -----
    >>> std = ScoreStandardizer()
    >>> std.fit({"boltz2": [0.1, 0.2, 0.3], "p2rank": [0.0, 0.5, 1.0]})
    >>> std.transform("boltz2", 0.2)   # near the mean → ~0.5
    0.5
    >>> std.transform("p2rank", 1.0)   # high tail → > 0.5
    """

    def __init__(self) -> None:
        # tool_id → {"mean": float, "std": float, "n": int}
        # ``n`` is informational (training-set size) and not used by
        # transform — kept so save/load preserves provenance.
        self.params: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------ fit

    def fit(self, train_scores: dict[str, Iterable[float]]) -> None:
        """Learn per-tool mean / std on the training set.

        ``train_scores[tool_id]`` is the flat list of every per-residue
        score for that tool across all training samples (residues from
        every sample concatenated). Tools missing from the dict are
        simply not standardised; ``transform`` for them is a passthrough.

        Non-finite entries (NaN / inf) are dropped before the moments
        are computed — they'd otherwise poison μ and σ. A tool with no
        finite scores at all is recorded with mean=0, std=1 so
        downstream still gets a deterministic transform; this is the
        safe degenerate case (sigmoid(0) = 0.5 for every input).
        """
        new_params: dict[str, dict[str, float]] = {}
        for tool_id, scores in train_scores.items():
            finite = [
                float(x) for x in scores
                if x is not None
                and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
            ]
            n = len(finite)
            if n == 0:
                new_params[tool_id] = {"mean": 0.0, "std": 1.0, "n": 0}
                continue
            mean = sum(finite) / n
            # Population std (matches numpy default ddof=0 for parity
            # with anyone re-fitting in numpy / pandas downstream).
            var = sum((x - mean) ** 2 for x in finite) / n
            std = max(math.sqrt(var), _MIN_STD)
            new_params[tool_id] = {"mean": mean, "std": std, "n": n}
        self.params = new_params

    # ------------------------------------------------------------------ transform

    def transform(self, tool_id: str, score: float) -> float:
        """Standardise one score; passthrough for unknown tools.

        Unknown ``tool_id`` returns the input unchanged (clamped to
        [0, 1]) — this keeps the inference path safe when a new tool
        is dropped into the pipeline before the standardiser has been
        re-fit. Callers that want strict failure should check
        ``tool_id in self.params`` themselves.
        """
        p = self.params.get(tool_id)
        if p is None:
            # Unknown tool — best-effort clamp.
            try:
                v = float(score)
            except (TypeError, ValueError):
                return 0.0
            if v < 0.0:
                return 0.0
            if v > 1.0:
                return 1.0
            return v
        try:
            v = float(score)
        except (TypeError, ValueError):
            return 0.5  # missing → neutral
        z = (v - p["mean"]) / p["std"]
        return _sigmoid(z)

    def transform_many(
        self, tool_id: str, scores: Iterable[float],
    ) -> list[float]:
        """Vectorised wrapper around ``transform`` for one tool."""
        return [self.transform(tool_id, s) for s in scores]

    # ------------------------------------------------------------------ I/O

    def to_dict(self) -> dict:
        """Serialisable form (plain dict)."""
        return {
            "version": 1,
            "method": "z-score-sigmoid",
            "min_std": _MIN_STD,
            "params": self.params,
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Atomic write to ``path`` (``.tmp + replace``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ScoreStandardizer":
        """Load from a file written by ``save``.

        Tolerates older payloads that lack ``version`` / ``method``;
        only the ``params`` block is structurally required.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "params" not in data:
            raise ValueError(
                f"standardizer file {path} has no 'params' block"
            )
        out = cls()
        params = data["params"] or {}
        cleaned: dict[str, dict[str, float]] = {}
        for tool_id, p in params.items():
            if not isinstance(p, dict):
                continue
            cleaned[tool_id] = {
                "mean": float(p.get("mean", 0.0)),
                "std":  max(float(p.get("std",  1.0)), _MIN_STD),
                "n":    int(p.get("n", 0)),
            }
        out.params = cleaned
        return out
