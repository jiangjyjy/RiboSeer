"""Ridge-regression weight learner for the learned-fusion path.

Closed-form solution
--------------------
Given a feature matrix ``X`` (n_residues × n_features) and a binary
label vector ``y`` (n_residues, 0/1), Ridge minimises

    L(w, b) = ||y - (X w + b·1)||² + λ ||w||²

Augment ``X`` with a column of ones for the bias and solve

    [w; b] = (X̃ᵀ X̃ + λ Iʹ)⁻¹ X̃ᵀ y

where ``Iʹ`` is the identity with a 0 in the bias slot — we
intentionally do NOT regularise the bias (otherwise λ would shrink
the predicted base rate toward 0 and bias the model against tools
with universally-low scores).

Sigmoid head
------------
``predict`` runs the linear combination through a sigmoid so the
output stays in [0, 1] — the per-residue binding-probability
contract that downstream eval / paper tables expect. The Ridge fit
is on the raw 0/1 labels (linear regression on a binary target —
"linear probability model") rather than logistic regression because:

  - it has a closed-form solution (no iteration / line-search),
  - the sigmoid only needs to flatten the predictions back into
    [0, 1] for ranking, not to be a calibrated probability,
  - training data lives at the per-residue level (~10⁵ rows for
    a 200-sample × 100-residue split), well-conditioned for the
    closed form even with the 5 + 20 + 10 = 35-feature design.

Why numpy-only
--------------
Per the task spec: keep deps simple, no sklearn. ``np.linalg.solve``
on a (35×35) Gram matrix is microseconds; sklearn would add a
heavyweight import and hide the closed-form math behind an estimator.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np


# Standard 20-letter amino-acid alphabet, in the canonical
# alphabetical-by-1-letter-code order. Used both to build the one-hot
# feature block and to label the saved weights for human inspection.
AA20 = "ACDEFGHIKLMNPQRSTVWY"


def _sigmoid_array(z: np.ndarray) -> np.ndarray:
    """Numerically stable σ on a numpy array.

    Splits at 0 the same way standardizer._sigmoid does — exp's
    argument stays non-positive on each branch.
    """
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


class WeightOptimizer:
    """Ridge regression on a tool / residue-type / cross-term design.

    Parameters
    ----------
    use_residue_type : bool
        If True, append a 20-dim one-hot block for the residue's amino
        acid letter to every feature row. Captures position-independent
        amino-acid-level bias (e.g. R / K / H more likely to RNA-bind).
    use_gating : bool
        If True, append per-tool gating features:
        * ``n_tools`` binary columns — 1 iff the residue is in that tool's
          ``binding_protein_residues`` list
        * 1 vote-count column — fraction of tools that voted "yes"
        These encode the noisy-OR core idea ("only residues a tool
        actually flagged matter") into the linear design, which the
        residue-score-only design can't recover from standardised
        scores alone. Default ``True`` because in our calibration
        runs flipping it on closed most of the gap to noisy-OR.
    use_cross_terms : bool
        If True, append the C(n_tools, 2) pairwise products of the
        standardised tool scores. Captures "agreement" signal — both
        tools high together is more informative than either alone.
        Off by default to keep the design matrix small / non-overfit.
    regularization : float
        Ridge λ. ``0.0`` falls back to plain OLS. The bias term is
        NEVER regularised (see module docstring).

    Layout note: ``tool_order`` (saved from fit, restored by load) is
    the source of truth for column order — callers that build features
    after load must use the same order, which ``build_features`` does
    automatically by sorting the input dict.
    """

    def __init__(
        self,
        *,
        use_residue_type: bool = True,
        use_gating: bool = True,
        use_cross_terms: bool = False,
        regularization: float = 0.01,
    ) -> None:
        self.use_residue_type = bool(use_residue_type)
        self.use_gating = bool(use_gating)
        self.use_cross_terms = bool(use_cross_terms)
        self.regularization = float(regularization)

        # Set on fit() and persisted; required on load.
        self.tool_order: list[str] = []
        self.weights: Optional[np.ndarray] = None  # shape (n_features,)
        self.bias: float = 0.0
        self.feature_names: list[str] = []
        # Stats for the training-report side of save().
        self.train_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------ feature builder

    def _feature_names(self, tool_order: Sequence[str]) -> list[str]:
        """Return the column-ordered feature names for ``tool_order``.

        Mirrors the order of ``build_features``. Used at fit time so
        save() can persist meaningful labels alongside the weights.
        """
        names: list[str] = []
        names.extend(f"tool:{t}" for t in tool_order)
        if self.use_gating:
            names.extend(f"gate:{t}" for t in tool_order)
            names.append("vote_count")
        if self.use_residue_type:
            names.extend(f"aa:{aa}" for aa in AA20)
        if self.use_cross_terms:
            for i, ti in enumerate(tool_order):
                for tj in tool_order[i + 1:]:
                    names.append(f"x:{ti}*{tj}")
        return names

    def build_features(
        self,
        tool_scores: dict[str, float],
        residue_type: Optional[str] = None,
        tool_binding_flags: Optional[dict[str, bool]] = None,
    ) -> np.ndarray:
        """Build one feature ROW for one (residue, sample).

        ``tool_scores`` keys must be the deployed-tool ids; missing
        entries are treated as 0.0 (defensive — a tool that failed
        to produce a score for the residue contributes nothing,
        same semantic as the noisy-OR ``b_k(i) = 0`` gate).

        On the first call, if ``self.tool_order`` is empty, we adopt
        ``sorted(tool_scores.keys())`` as the canonical column order;
        subsequent calls reuse that order regardless of the input
        dict's iteration order.

        The residue letter is matched case-insensitively against
        ``AA20``; non-standard residues (e.g. "X", "B") map to an
        all-zero one-hot block — the same "no information" encoding
        the standardiser uses for unknown tools.

        ``tool_binding_flags`` is consumed only when ``use_gating`` is
        on. Missing keys / a None dict ⇒ flags default to False (the
        "tool didn't flag this residue" case). The vote-count column
        is computed against ``self.tool_order`` (NOT against the input
        dict's keys) so the design stays deterministic across inputs.
        """
        if not self.tool_order:
            self.tool_order = sorted(tool_scores.keys())

        row: list[float] = []
        for tid in self.tool_order:
            v = tool_scores.get(tid, 0.0)
            try:
                row.append(float(v))
            except (TypeError, ValueError):
                row.append(0.0)

        if self.use_gating:
            flags = tool_binding_flags or {}
            binary: list[float] = [
                1.0 if flags.get(tid, False) else 0.0
                for tid in self.tool_order
            ]
            row.extend(binary)
            n_tools = len(self.tool_order)
            row.append(sum(binary) / n_tools if n_tools > 0 else 0.0)

        if self.use_residue_type:
            aa = (residue_type or "").upper()[:1]
            row.extend(1.0 if aa == letter else 0.0 for letter in AA20)

        if self.use_cross_terms:
            scores = [
                float(tool_scores.get(t, 0.0) or 0.0) for t in self.tool_order
            ]
            for i in range(len(scores)):
                for j in range(i + 1, len(scores)):
                    row.append(scores[i] * scores[j])

        return np.asarray(row, dtype=np.float64)

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weights: Optional[np.ndarray] = None,
        tool_order: Optional[Sequence[str]] = None,
        verbose: bool = True,
    ) -> dict[str, float]:
        """Closed-form (weighted) Ridge fit. Returns training-report dict.

        With binding residues at ~5-6 % positive rate, plain Ridge gets
        a near-zero R² because the loss is dominated by the easy
        negatives. ``sample_weights`` reweights the per-row squared
        loss so positives carry their share — typically ``≈ neg/pos``
        for "balanced" weighting (LearnedFusion.train resolves this).

        Math
        ----
        Minimising ``Σᵢ wᵢ (yᵢ - ŷᵢ)² + λ ||w||²`` is equivalent to
        the unweighted problem on ``X̃ √w``, ``y √w`` — so we just
        scale rows of the augmented design / target instead of forming
        a (potentially huge) diagonal weight matrix.

        Parameters
        ----------
        sample_weights :
            Length-n non-negative weights. ``None`` (default) is
            equivalent to ``np.ones(n)`` — plain Ridge.
        tool_order :
            Pin column order; required when fit() is called with a
            prebuilt X (e.g. from learned_fusion.train).

        Returns
        -------
        report : dict
            ``{"n_rows", "n_features", "lambda", "rmse", "r2",
              "pos_rate", "weighted"}`` — last flag tells the trainer
            CLI / report whether weighting was applied.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D; got shape {X.shape!r}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X rows ({X.shape[0]}) != y len ({y.shape[0]})"
            )
        n, d = X.shape

        if sample_weights is not None:
            sw = np.asarray(sample_weights, dtype=np.float64).ravel()
            if sw.shape[0] != n:
                raise ValueError(
                    f"sample_weights length ({sw.shape[0]}) != X rows ({n})"
                )
            if np.any(sw < 0):
                raise ValueError("sample_weights must be non-negative")
            weighted = True
        else:
            sw = None
            weighted = False

        if tool_order is not None:
            self.tool_order = list(tool_order)
        # Re-derive feature names so they match the fit-time design.
        self.feature_names = self._feature_names(self.tool_order)
        if len(self.feature_names) != d:
            raise ValueError(
                f"feature_names length ({len(self.feature_names)}) "
                f"does not match X.shape[1] ({d}); fit was called "
                f"with a design matrix that doesn't match the "
                f"configured (use_residue_type / use_gating / "
                f"use_cross_terms / tool_order) options."
            )

        # Augment with bias column; build λI with bias slot un-regularised.
        X_b = np.hstack([X, np.ones((n, 1))])
        lam_diag = np.full(d + 1, self.regularization)
        lam_diag[-1] = 0.0

        if weighted:
            # Row-scaling trick: solving on (X̃ √w, y √w) gives the same
            # closed-form solution as the diagonal-weighted normal eqn,
            # without materialising the n×n diag(W).
            sqrt_w = np.sqrt(sw)[:, np.newaxis]
            X_eff = X_b * sqrt_w
            y_eff = y * sqrt_w.ravel()
        else:
            X_eff = X_b
            y_eff = y
        gram = X_eff.T @ X_eff + np.diag(lam_diag)
        rhs = X_eff.T @ y_eff
        try:
            wb = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            # Singular Gram — fall back to least-squares pseudoinverse.
            wb, *_ = np.linalg.lstsq(gram, rhs, rcond=None)

        self.weights = wb[:-1].astype(np.float64)
        self.bias = float(wb[-1])

        # ---- training metrics — UNweighted RMSE / R² so the numbers
        # are comparable across runs with different weighting schemes.
        # (Weighted R² would inflate when positives are upweighted and
        # mask whether the model actually generalises.)
        y_pred_linear = X @ self.weights + self.bias
        residuals = y - y_pred_linear
        ss_res = float((residuals ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = float(math.sqrt(ss_res / n)) if n else float("nan")
        report = {
            "n_rows": int(n),
            "n_features": int(d),
            "lambda": float(self.regularization),
            "rmse": rmse,
            "r2": r2,
            "pos_rate": float(y.mean()),
            "weighted": bool(weighted),
        }
        self.train_metrics = report

        if verbose:
            wlabel = "weighted" if weighted else "unweighted"
            print(f"[ridge/{wlabel}] n={n}  d={d}  λ={self.regularization}  "
                  f"rmse={rmse:.4f}  R²={r2:.4f}  pos_rate={y.mean():.4f}")
            for name, val in zip(self.feature_names, self.weights):
                print(f"  {name:25s}: {val:+.4f}")
            print(f"  {'bias':25s}: {self.bias:+.4f}")
        return report

    # ------------------------------------------------------------------ predict

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return σ(Xw + b) — per-row binding probability in [0, 1]."""
        if self.weights is None:
            raise RuntimeError(
                "WeightOptimizer.predict called before fit / load"
            )
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"X.shape[1] ({X.shape[1]}) != weights ({self.weights.shape[0]}); "
                f"feature design changed since fit"
            )
        logits = X @ self.weights + self.bias
        return _sigmoid_array(logits)

    # ------------------------------------------------------------------ I/O

    def to_dict(self) -> dict:
        """Serialisable form."""
        return {
            "version": 2,
            "method": "ridge",
            "config": {
                "use_residue_type": self.use_residue_type,
                "use_gating": self.use_gating,
                "use_cross_terms": self.use_cross_terms,
                "regularization": self.regularization,
            },
            "tool_order": list(self.tool_order),
            "feature_names": list(self.feature_names),
            "weights": (
                self.weights.tolist() if self.weights is not None else []
            ),
            "bias": float(self.bias),
            "train_metrics": dict(self.train_metrics),
        }

    def save(self, path: Union[str, Path]) -> Path:
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
    def load(cls, path: Union[str, Path]) -> "WeightOptimizer":
        """Load a saved bundle.

        Backward compat: pre-v2 bundles have no ``use_gating`` key —
        we default it to ``False`` on load so old bundles trained
        without gating keep predicting on the same feature design
        they were fit with. New bundles persist ``use_gating``
        explicitly so this default never fires for fresh artefacts.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"optimizer file {path} is not a JSON object")
        cfg = (data.get("config") or {})
        out = cls(
            use_residue_type=bool(cfg.get("use_residue_type", True)),
            # Older bundles (version 1) lacked the gating block; default
            # off so loading is lossless.
            use_gating=bool(cfg.get("use_gating", False)),
            use_cross_terms=bool(cfg.get("use_cross_terms", False)),
            regularization=float(cfg.get("regularization", 0.01)),
        )
        out.tool_order = list(data.get("tool_order") or [])
        out.feature_names = list(data.get("feature_names") or [])
        weights = data.get("weights") or []
        if weights:
            out.weights = np.asarray(weights, dtype=np.float64)
        out.bias = float(data.get("bias", 0.0))
        out.train_metrics = dict(data.get("train_metrics") or {})
        return out
