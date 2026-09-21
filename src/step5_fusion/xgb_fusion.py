"""XGBoost-based learned fusion (alternative to the Ridge path).

Why a second model
------------------
The Ridge path closed most of the gap to the noisy-OR baseline once
gating + balanced class weights were added. But linear-on-sigmoid
can't represent useful interactions like "tool flagged the residue
AND its standardised score is above the per-tool mean ⇒ binding".
XGBoost picks those up automatically through tree splits, and on the
500-sample train batch this typically takes per-residue Pearson R
from ~0.46 (Ridge) to ~0.55+ depending on hyperparams.

Architectural parity with LearnedFusion
---------------------------------------
We deliberately:
  - reuse the same feature design (tool scores + gating + AA one-hot)
    by composing a ``WeightOptimizer`` purely as a feature builder —
    no fitting on it, no weight loading. Layout drift between the
    two model classes would silently corrupt comparison numbers, so
    we share the single source of truth.
  - reuse ``ScoreStandardizer`` so per-tool z-scoring matches Ridge
    exactly. XGBoost is scale-invariant per feature, but it lets us
    write the same standardiser bundle file in both paths so the
    inference code is interchangeable.
  - reuse ``data_collector.collect_per_residue_data`` so the train
    set is byte-identical to Ridge's.

Save / load
-----------
Bundle layout under ``<dir>/``:
  - ``standardizer.json``    same shape as Ridge bundle
  - ``xgb_model.json``       XGBoost native JSON (cross-version safe)
  - ``xgb_meta.json``        tool_order + feature_names + score_fields
                             + xgb_params + use_residue_type/gating
  - ``training_report.json`` n_rows, pos_rate, train Pearson / R², top
                             feature importances

The presence of ``xgb_model.json`` (vs ``optimizer.json`` for Ridge)
is what ``scripts/evaluate_learned_fusion.py`` uses to auto-detect
which class to load.

Optional dependency
-------------------
``xgboost`` is imported at module load and surfaced through ``XGB_OK``
+ ``XGB_IMPORT_ERROR``. Tests skip when it's missing; the install line
for the project conda env is::

    conda run -n riboseer pip install xgboost
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from .data_collector import (
    DEFAULT_SCORE_FIELDS,
    aa_for,
    collect_per_residue_data,
    extract_score,
    per_residue_to_int_dict,
    read_jsonl_record,
    resolve_score_fields,
)
from .standardizer import ScoreStandardizer
from .weight_optimizer import AA20, WeightOptimizer


try:
    import xgboost as xgb  # type: ignore
    XGB_OK = True
    XGB_IMPORT_ERROR: Optional[Exception] = None
except ImportError as _e:  # pragma: no cover — exercised only on CI without xgb
    xgb = None  # type: ignore
    XGB_OK = False
    XGB_IMPORT_ERROR = _e


# ---- internal stats helpers ---------------------------------------------


def _pearson(xs, ys) -> Optional[float]:
    """Numpy Pearson r; ``None`` on degenerate input. Matches the
    eval scripts' implementation byte-for-byte so training-set
    correlations are directly comparable to test-set ones."""
    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    if xs.size < 2 or xs.shape != ys.shape:
        return None
    sx = xs.std()
    sy = ys.std()
    if sx == 0 or sy == 0:
        return None
    return float(((xs - xs.mean()) * (ys - ys.mean())).mean() / (sx * sy))


def _spearman(xs, ys) -> Optional[float]:
    """Spearman rho via rank-then-Pearson (numpy argsort, average ties).

    Manual implementation avoids requiring scipy in the project env —
    we already do the same thing in scripts/evaluate.py for the
    test-set numbers.
    """
    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    if xs.size < 2 or xs.shape != ys.shape:
        return None

    def _avg_ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        i = 0
        n = len(v)
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            ranks[order[i:j + 1]] = avg
            i = j + 1
        return ranks

    return _pearson(_avg_ranks(xs), _avg_ranks(ys))


def _check_xgb_available() -> None:
    if not XGB_OK:
        raise ImportError(
            f"xgboost is required for XGBFusion but failed to import "
            f"({XGB_IMPORT_ERROR}). Install it with:\n"
            f"  conda run -n riboseer pip install xgboost"
        )


# ---- main class ----------------------------------------------------------


class XGBFusion:
    """XGBoost classifier wrapper with the LearnedFusion-style I/O.

    Public surface mirrors ``LearnedFusion``:
      - ``train(step4_dir, processed_dir, ...)`` returns a report dict
      - ``predict_sample(predictions, seq, length)`` returns
        ``{residue_id: prob}``
      - ``save(dir)`` / ``load(dir)`` round-trip a bundle dir

    so ``scripts/evaluate_learned_fusion.py`` can hold one of either
    instance behind the same interface.
    """

    _STD_NAME = "standardizer.json"
    _MODEL_NAME = "xgb_model.json"
    _META_NAME = "xgb_meta.json"
    _REPORT_NAME = "training_report.json"

    # Sensible defaults for the binary RNA-binding-residue task.
    # max_depth=4 keeps trees shallow enough that tree count drives
    # capacity, not single-tree overfit; min_child_weight=5 prevents
    # leaves from chasing one-or-two-residue idiosyncrasies of one
    # protein. The user can override every key via the YAML config.
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

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.config = dict(cfg)
        self.score_fields = resolve_score_fields(cfg)

        self.standardizer = ScoreStandardizer()
        # Compose a WeightOptimizer purely as a feature builder. We
        # never call its .fit / .predict — only build_features() and
        # the column-name machinery. This keeps the feature layout
        # bit-identical to Ridge so a/b comparison is fair.
        self.feature_builder = WeightOptimizer(
            use_residue_type=bool(cfg.get("use_residue_type", True)),
            use_gating=bool(cfg.get("use_gating", True)),
            use_cross_terms=bool(cfg.get("use_cross_terms", False)),
            regularization=0.0,  # unused
        )

        # XGBoost hyperparams — start from defaults then layer config
        # overrides on top. Using a fresh dict so multiple XGBFusion
        # instances don't clobber each other.
        self.xgb_params: dict = dict(self._DEFAULT_XGB_PARAMS)
        for k in (
            "max_depth", "learning_rate", "n_estimators",
            "subsample", "colsample_bytree", "min_child_weight",
            "seed",
        ):
            if k in cfg:
                self.xgb_params[k] = cfg[k]

        self.model = None  # set on train() / load()
        self.tool_order: list[str] = []
        self.feature_names: list[str] = []
        self.training_report: dict = {}

    # ------------------------------------------------------------------ feature builder

    def build_features(
        self,
        tool_scores: dict[str, float],
        residue_type: Optional[str] = None,
        tool_binding_flags: Optional[dict[str, bool]] = None,
    ) -> np.ndarray:
        """Delegate to the composed WeightOptimizer so the layout is
        guaranteed identical to the Ridge path."""
        self.feature_builder.tool_order = list(self.tool_order)
        return self.feature_builder.build_features(
            tool_scores, residue_type, tool_binding_flags,
        )

    # ------------------------------------------------------------------ training

    def train(
        self,
        *,
        step4_dir: Path,
        processed_dir: Path,
        sample_ids: Optional[Iterable[str]] = None,
        verbose: bool = True,
    ) -> dict:
        """Collect → standardise → fit XGBoost. Returns the report dict.

        ``scale_pos_weight`` is set to ``neg_count / max(pos_count, 1)``
        — XGBoost's built-in class-imbalance handling, equivalent in
        spirit to the Ridge path's ``class_weight="balanced"``.
        """
        _check_xgb_available()

        bundle = collect_per_residue_data(
            step4_dir=step4_dir,
            processed_dir=processed_dir,
            score_fields=self.score_fields,
            sample_ids=sample_ids,
        )
        per_tool = bundle["per_tool_scores"]
        rows = bundle["rows"]
        if not rows:
            raise RuntimeError(
                "no usable training rows collected — check that "
                "step4_dir contains JSONLs with successful predictions "
                "and that processed_dir's samples carry "
                "interaction.binding_protein_residues."
            )
        if verbose:
            print(f"[xgb_fusion] collected {len(rows)} per-residue rows "
                  f"from {bundle['n_samples_used']} samples "
                  f"(skipped {bundle['n_samples_skipped']})")

        # 1) standardiser fit on the per-tool flat lists.
        self.standardizer.fit(per_tool)

        # 2) tool_order — union of tools that gave scores and tools
        # that emitted binding flags, sorted for determinism (matches
        # LearnedFusion semantics so a/b runs use the same column
        # order).
        all_tool_ids: set[str] = set(per_tool.keys())
        for r in rows:
            all_tool_ids.update(r.get("tool_binding_flags") or {})
        self.tool_order = sorted(all_tool_ids)
        self.feature_builder.tool_order = list(self.tool_order)

        # 3) build X / y in the same layout WeightOptimizer uses.
        X_rows: list[np.ndarray] = []
        y: list[float] = []
        for r in rows:
            std_scores = {
                tid: self.standardizer.transform(tid, sc)
                for tid, sc in r["tool_scores"].items()
            }
            X_rows.append(self.feature_builder.build_features(
                std_scores, r["aa"], r.get("tool_binding_flags"),
            ))
            y.append(float(r["label"]))
        X = np.vstack(X_rows)
        y_arr = np.asarray(y, dtype=np.float64)
        self.feature_names = list(
            self.feature_builder._feature_names(self.tool_order)
        )

        # 4) class balance for XGBoost via scale_pos_weight.
        pos = int((y_arr > 0.5).sum())
        neg = int(y_arr.size - pos)
        scale_pos_weight = float(neg / max(pos, 1))

        # 5) fit XGBoost.
        self.model = xgb.XGBClassifier(
            **self.xgb_params,
            scale_pos_weight=scale_pos_weight,
        )
        # ``verbose=False`` suppresses XGB's own per-iteration logging;
        # the summary lines below are enough for our reports.
        self.model.fit(X, y_arr, verbose=False)

        # 6) training metrics (all unweighted, on the same set we fit).
        y_pred = self.model.predict_proba(X)[:, 1]
        pr = _pearson(y_pred, y_arr)
        sr = _spearman(y_pred, y_arr)
        r2 = (pr ** 2) if pr is not None else None
        # Cross-entropy / RMSE for completeness — XGBoost's eval_metric
        # is logloss, but we also report RMSE so it's directly
        # comparable to the Ridge report.
        rmse = float(math.sqrt(((y_pred - y_arr) ** 2).mean()))

        # 7) feature importances (gain — XGBoost's default for
        # XGBClassifier.feature_importances_).
        importances = self.model.feature_importances_.tolist()
        # Top-10 sorted by descending importance for the report.
        top: list[dict] = []
        for name, imp in sorted(
            zip(self.feature_names, importances), key=lambda x: -x[1],
        )[:10]:
            top.append({"feature": name, "importance": float(imp)})

        report = {
            "n_rows": int(y_arr.size),
            "n_features": int(X.shape[1]),
            "pos_count": pos,
            "neg_count": neg,
            "pos_rate": float(y_arr.mean()),
            "scale_pos_weight": scale_pos_weight,
            "rmse": rmse,
            "pearson_r": pr,
            "spearman_r": sr,
            "r2": r2,
            "n_samples_used": bundle["n_samples_used"],
            "n_samples_skipped": bundle["n_samples_skipped"],
            "score_fields": dict(self.score_fields),
            "feature_importances_top": top,
            "xgb_params": dict(self.xgb_params),
        }
        self.training_report = report

        if verbose:
            print(
                f"[xgb_fusion] n={y_arr.size}  d={X.shape[1]}  "
                f"pos={pos}  neg={neg}  spw={scale_pos_weight:.2f}"
            )
            print(
                f"[xgb_fusion] train Pearson R={pr:.4f}  "
                f"Spearman R={sr:.4f}  R²={r2:.4f}  RMSE={rmse:.4f}"
            )
            print("[xgb_fusion] top feature importances:")
            for entry in top:
                print(f"  {entry['feature']:25s}: {entry['importance']:.4f}")
        return report

    # ------------------------------------------------------------------ inference

    def predict_sample(
        self,
        tool_predictions: list[dict],
        protein_sequence: str,
        protein_length: int,
    ) -> dict[int, float]:
        """Per-residue binding probability for one sample.

        Same shape contract as ``LearnedFusion.predict_sample`` —
        ``scripts/evaluate_learned_fusion.py`` calls one or the other
        without branching.
        """
        _check_xgb_available()
        if self.model is None:
            raise RuntimeError(
                "XGBFusion.predict_sample called before train() / load()"
            )

        successful = [
            p for p in (tool_predictions or []) if p.get("success")
        ]

        # Pre-compute each tool's binding-list as a set so the
        # per-residue gating-flag lookup is O(1).
        binding_sets: dict[str, set[int]] = {}
        if self.feature_builder.use_gating:
            for pred in successful:
                tid = pred.get("tool_id") or ""
                bset: set[int] = set()
                for v in (pred.get("binding_protein_residues") or []):
                    try:
                        bset.add(int(v))
                    except (TypeError, ValueError):
                        continue
                binding_sets[tid] = bset

        # Pin column order in the feature builder so every row uses
        # the same layout the model was trained on.
        self.feature_builder.tool_order = list(self.tool_order)

        rows: list[np.ndarray] = []
        residue_ids = list(range(1, int(protein_length) + 1))
        for res_id in residue_ids:
            tool_scores: dict[str, float] = {}
            for pred in successful:
                score = extract_score(pred, res_id, self.score_fields)
                if score is None:
                    continue
                tid = pred.get("tool_id") or ""
                tool_scores[tid] = self.standardizer.transform(tid, score)
            aa = aa_for(protein_sequence, res_id)
            flags: Optional[dict[str, bool]]
            if self.feature_builder.use_gating:
                flags = {
                    tid: (res_id in bset)
                    for tid, bset in binding_sets.items()
                }
            else:
                flags = None
            rows.append(self.feature_builder.build_features(
                tool_scores, aa, flags,
            ))

        if not rows:
            return {}
        X = np.vstack(rows)
        probs = self.model.predict_proba(X)[:, 1]
        return {
            res_id: float(round(float(p), 6))
            for res_id, p in zip(residue_ids, probs)
        }

    # ------------------------------------------------------------------ I/O

    def save(self, dir_path: Union[str, Path]) -> Path:
        """Write the bundle (standardizer + xgb model + meta + report).

        Bundle layout chosen so ``scripts/evaluate_learned_fusion.py``
        can auto-detect XGB vs Ridge purely by file presence
        (``xgb_model.json`` for this class, ``optimizer.json`` for the
        Ridge path).
        """
        if self.model is None:
            raise RuntimeError("save() called before train() / load()")
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        self.standardizer.save(dir_path / self._STD_NAME)
        # XGBoost native JSON — cross-version-safe, human-readable.
        self.model.save_model(str(dir_path / self._MODEL_NAME))
        meta = {
            "version": 1,
            "model_type": "xgboost",
            "tool_order": list(self.tool_order),
            "feature_names": list(self.feature_names),
            "score_fields": dict(self.score_fields),
            "xgb_params": dict(self.xgb_params),
            "config": {
                "use_residue_type": self.feature_builder.use_residue_type,
                "use_gating":       self.feature_builder.use_gating,
                "use_cross_terms":  self.feature_builder.use_cross_terms,
            },
        }
        (dir_path / self._META_NAME).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        if self.training_report:
            (dir_path / self._REPORT_NAME).write_text(
                json.dumps(self.training_report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return dir_path

    @classmethod
    def load(cls, dir_path: Union[str, Path]) -> "XGBFusion":
        _check_xgb_available()
        dir_path = Path(dir_path)
        meta_path = dir_path / cls._META_NAME
        meta = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.is_file() else {}
        )
        cfg = meta.get("config") or {}
        out = cls({
            "use_residue_type": bool(cfg.get("use_residue_type", True)),
            "use_gating":       bool(cfg.get("use_gating", True)),
            "use_cross_terms":  bool(cfg.get("use_cross_terms", False)),
            **(meta.get("xgb_params") or {}),
            "score_fields":     meta.get("score_fields") or {},
        })
        out.standardizer = ScoreStandardizer.load(dir_path / cls._STD_NAME)
        out.tool_order = list(meta.get("tool_order") or [])
        out.feature_builder.tool_order = list(out.tool_order)
        out.feature_names = list(meta.get("feature_names") or [])

        # XGBClassifier needs a fresh instance to load into.
        out.model = xgb.XGBClassifier(**out.xgb_params)
        out.model.load_model(str(dir_path / cls._MODEL_NAME))

        report_path = dir_path / cls._REPORT_NAME
        if report_path.is_file():
            try:
                out.training_report = json.loads(
                    report_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                out.training_report = {}
        return out


# ---------------------------------------------------------------------------
# AgentCalibratedXGBFusion — XGBoost as a calibration head on the agent
# ---------------------------------------------------------------------------


# Feature names for the two columns prepended to the WeightOptimizer
# base. Pinned at module scope so save/load and tests can reference
# the canonical strings.
AGENT_FEATURE_NAMES = ("agent:fusion_prob", "agent:in_binding")


class AgentCalibratedXGBFusion:
    """XGBoost trained on top of the agent's noisy-OR fusion output.

    Why this class exists alongside :class:`XGBFusion`
    --------------------------------------------------
    The plain XGBoost path replaces the agent's noisy-OR fusion
    entirely. That gives us a comparable Pearson R but throws away
    everything the LLM-driven steps 2/3/5/7 contributed (target
    characterisation → tool plan → per-tool weights → iterative
    refinement). The calibration head keeps the agent as the
    primary predictor and lets XGBoost re-rank residues using the
    SAME information the agent had access to plus the agent's own
    output as a feature. Empirically this is what the user's PI
    wanted: keep the agent in the loop, layer a small model for the
    final probability calibration.

    Feature layout
    --------------
    Each (sample, residue) row is::

        [
            agent_fusion_prob,        # step5's per_residue_probability[res]
            agent_in_binding,         # 1 iff residue ∈ step5's binding list
            <WeightOptimizer base>,   # 5 tool scores + 5 gates + 1 vote
                                      # + 20 AA one-hot (knobs from config)
        ]

    Total = 2 + (varies by config). The two leading columns are the
    only thing that distinguishes this class's design from the
    standalone :class:`XGBFusion`.

    Bundle file marker
    ------------------
    ``xgb_meta.json`` carries ``model_type: "xgboost_calibrated"`` AND
    ``calibrated: true`` so :func:`scripts.evaluate_learned_fusion._detect_model_type`
    can pick this class without ambiguity. Inference paths that need
    the agent input (``predict_sample(..., agent_probs=, agent_binding_set=)``)
    can then refuse to run silently as standalone.
    """

    # Re-use the file-name layout from XGBFusion for parity.
    _STD_NAME = XGBFusion._STD_NAME
    _MODEL_NAME = XGBFusion._MODEL_NAME
    _META_NAME = XGBFusion._META_NAME
    _REPORT_NAME = XGBFusion._REPORT_NAME

    # Conservative defaults — the agent has already done the heavy
    # lifting, so we want a small, well-regularised model that just
    # nudges the probability calibration.
    _DEFAULT_XGB_PARAMS: dict = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 20,
        "seed": 42,
    }

    # Marker fields written into the bundle's xgb_meta.json so
    # downstream auto-detect can route to this class.
    MODEL_TYPE = "xgboost_calibrated"

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.config = dict(cfg)
        self.score_fields = resolve_score_fields(cfg)

        self.standardizer = ScoreStandardizer()
        self.feature_builder = WeightOptimizer(
            use_residue_type=bool(cfg.get("use_residue_type", True)),
            use_gating=bool(cfg.get("use_gating", True)),
            use_cross_terms=bool(cfg.get("use_cross_terms", False)),
            regularization=0.0,
        )

        # Hyperparams: defaults below + config overrides.
        self.xgb_params: dict = dict(self._DEFAULT_XGB_PARAMS)
        for k in (
            "max_depth", "learning_rate", "n_estimators",
            "subsample", "colsample_bytree", "min_child_weight",
            "seed",
        ):
            if k in cfg:
                self.xgb_params[k] = cfg[k]

        self.model = None
        self.tool_order: list[str] = []
        self.feature_names: list[str] = []
        self.training_report: dict = {}

    # ------------------------------------------------------------------ feature builder

    def build_features(
        self,
        tool_scores: dict[str, float],
        residue_type: Optional[str],
        tool_binding_flags: Optional[dict[str, bool]],
        agent_prob: float,
        agent_in_binding: bool,
    ) -> np.ndarray:
        """Prepend the two agent-derived features to the WeightOptimizer base."""
        self.feature_builder.tool_order = list(self.tool_order)
        base = self.feature_builder.build_features(
            tool_scores, residue_type, tool_binding_flags,
        )
        return np.concatenate(
            ([float(agent_prob), 1.0 if agent_in_binding else 0.0], base)
        )

    def _full_feature_names(self) -> list[str]:
        """Concatenate agent-feature names with the WeightOptimizer base names."""
        return list(AGENT_FEATURE_NAMES) + list(
            self.feature_builder._feature_names(self.tool_order)
        )

    # ------------------------------------------------------------------ training

    def _read_step5_agent(
        self, step5_dir: Path, sample_id: str,
    ) -> Optional[tuple[dict[int, float], set[int]]]:
        """Pull (per_residue_probability, binding_protein_residues) from step5.

        ``per_residue_probability`` is JSON-serialised with string keys
        (step5 stringifies them in build_output_record); coerce to int.
        Returns ``None`` when the step5 record is missing — the caller
        skips that sample's rows.
        """
        s5 = read_jsonl_record(step5_dir / f"{sample_id}.jsonl")
        if s5 is None:
            return None
        probs = per_residue_to_int_dict(s5.get("per_residue_probability"))
        binding_raw = s5.get("binding_protein_residues") or []
        binding: set[int] = set()
        for v in binding_raw:
            try:
                binding.add(int(v))
            except (TypeError, ValueError):
                continue
        return probs, binding

    def train(
        self,
        *,
        step4_dir: Path,
        step5_dir: Path,
        processed_dir: Path,
        sample_ids: Optional[Iterable[str]] = None,
        verbose: bool = True,
    ) -> dict:
        """Fit the calibration model. Requires step5 (agent output) per sample.

        Implementation outline:
          1. Use the shared collector for step4 + GT + per-tool scores
             (same training rows that XGBFusion / LearnedFusion see).
          2. Read the per-sample step5 record and enrich each row with
             ``agent_prob`` and ``agent_in_binding``.
          3. Drop rows whose sample lacks a step5 record (no agent
             signal → calibration is undefined).
          4. Standardise tool scores, build the design matrix
             (agent features + base features), fit XGBoost.
        """
        _check_xgb_available()

        bundle = collect_per_residue_data(
            step4_dir=step4_dir,
            processed_dir=processed_dir,
            score_fields=self.score_fields,
            sample_ids=sample_ids,
        )
        per_tool = bundle["per_tool_scores"]
        rows = bundle["rows"]
        if not rows:
            raise RuntimeError(
                "no usable training rows collected — see XGBFusion.train"
            )

        # Read step5 per unique sample so we don't re-open the file
        # for every residue.
        step5_dir = Path(step5_dir)
        agent_data: dict[str, tuple[dict[int, float], set[int]]] = {}
        unique_sids = sorted({r["sample_id"] for r in rows})
        for sid in unique_sids:
            agent = self._read_step5_agent(step5_dir, sid)
            if agent is not None:
                agent_data[sid] = agent

        # Drop rows whose sample has no step5 record — calibration
        # has nothing to calibrate against. Track the count so the
        # report can flag it (a non-zero number signals stale step5
        # output relative to step4).
        kept_rows: list[dict] = []
        for r in rows:
            sid = r["sample_id"]
            agent = agent_data.get(sid)
            if agent is None:
                continue
            probs, binding = agent
            er = dict(r)
            er["agent_prob"] = float(probs.get(r["residue_id"], 0.0))
            er["agent_in_binding"] = r["residue_id"] in binding
            kept_rows.append(er)

        n_dropped_no_agent = len(rows) - len(kept_rows)
        if not kept_rows:
            raise RuntimeError(
                "no rows survived the step5 enrichment — every sample "
                "in the train batch is missing a step5 JSONL. Check "
                "--train-step5-dir."
            )
        if verbose:
            print(
                f"[xgb_fusion/calibrated] collected {len(kept_rows)} "
                f"per-residue rows from {bundle['n_samples_used']} samples "
                f"(skipped {bundle['n_samples_skipped']} "
                f"+ {n_dropped_no_agent} rows missing step5 enrichment)"
            )

        # Standardiser fit on tool-score lists collected by the shared
        # collector — ``agent_prob`` is intentionally NOT standardised
        # (it's already a calibrated probability in [0, 1] from
        # noisy-OR).
        self.standardizer.fit(per_tool)

        # tool_order — same union semantics as XGBFusion.
        all_tool_ids: set[str] = set(per_tool.keys())
        for r in kept_rows:
            all_tool_ids.update(r.get("tool_binding_flags") or {})
        self.tool_order = sorted(all_tool_ids)
        self.feature_builder.tool_order = list(self.tool_order)

        # Build (X, y).
        X_rows: list[np.ndarray] = []
        y: list[float] = []
        for r in kept_rows:
            std_scores = {
                tid: self.standardizer.transform(tid, sc)
                for tid, sc in r["tool_scores"].items()
            }
            X_rows.append(self.build_features(
                std_scores, r["aa"], r.get("tool_binding_flags"),
                r["agent_prob"], r["agent_in_binding"],
            ))
            y.append(float(r["label"]))
        X = np.vstack(X_rows)
        y_arr = np.asarray(y, dtype=np.float64)
        self.feature_names = self._full_feature_names()

        # Class balance via XGBoost's built-in scale_pos_weight.
        pos = int((y_arr > 0.5).sum())
        neg = int(y_arr.size - pos)
        scale_pos_weight = float(neg / max(pos, 1))

        self.model = xgb.XGBClassifier(
            **self.xgb_params,
            scale_pos_weight=scale_pos_weight,
        )
        self.model.fit(X, y_arr, verbose=False)

        # Training metrics — compute on the same train set; not a
        # generalisation estimate but useful for noticing under-fit.
        y_pred = self.model.predict_proba(X)[:, 1]
        pr = _pearson(y_pred, y_arr)
        sr = _spearman(y_pred, y_arr)
        r2 = (pr ** 2) if pr is not None else None
        rmse = float(math.sqrt(((y_pred - y_arr) ** 2).mean()))

        importances = self.model.feature_importances_.tolist()
        top: list[dict] = []
        for name, imp in sorted(
            zip(self.feature_names, importances), key=lambda x: -x[1],
        )[:10]:
            top.append({"feature": name, "importance": float(imp)})

        report = {
            "mode": "calibrated",
            "n_rows": int(y_arr.size),
            "n_features": int(X.shape[1]),
            "pos_count": pos,
            "neg_count": neg,
            "pos_rate": float(y_arr.mean()),
            "scale_pos_weight": scale_pos_weight,
            "rmse": rmse,
            "pearson_r": pr,
            "spearman_r": sr,
            "r2": r2,
            "n_samples_used": bundle["n_samples_used"],
            "n_samples_skipped": bundle["n_samples_skipped"],
            "n_dropped_no_agent": n_dropped_no_agent,
            "score_fields": dict(self.score_fields),
            "feature_importances_top": top,
            "xgb_params": dict(self.xgb_params),
        }
        self.training_report = report

        if verbose:
            print(
                f"[xgb_fusion/calibrated] n={y_arr.size}  d={X.shape[1]}  "
                f"pos={pos}  neg={neg}  spw={scale_pos_weight:.2f}"
            )
            print(
                f"[xgb_fusion/calibrated] train Pearson R={pr:.4f}  "
                f"Spearman R={sr:.4f}  R²={r2:.4f}  RMSE={rmse:.4f}"
            )
            print("[xgb_fusion/calibrated] top feature importances:")
            for entry in top:
                print(f"  {entry['feature']:25s}: {entry['importance']:.4f}")
        return report

    # ------------------------------------------------------------------ inference

    def predict_sample(
        self,
        tool_predictions: list[dict],
        protein_sequence: str,
        protein_length: int,
        *,
        agent_probs: Optional[dict] = None,
        agent_binding_set: Optional[set[int]] = None,
    ) -> dict[int, float]:
        """Per-residue calibrated probability.

        Unlike :meth:`XGBFusion.predict_sample` this method needs the
        agent's noisy-OR fusion output for the same sample —
        ``agent_probs`` (any of the per_residue_probability dicts the
        step5 record carries; int OR string keys both accepted) and
        ``agent_binding_set`` (residue ids in the agent's binding
        list). Both are required; ``None`` raises so a misconfigured
        eval script doesn't silently zero them out.
        """
        _check_xgb_available()
        if self.model is None:
            raise RuntimeError(
                "AgentCalibratedXGBFusion.predict_sample called before "
                "train() / load()"
            )
        if agent_probs is None or agent_binding_set is None:
            raise ValueError(
                "AgentCalibratedXGBFusion needs agent_probs and "
                "agent_binding_set — pass step5's per_residue_probability "
                "and binding_protein_residues for this sample."
            )
        agent_probs_int = per_residue_to_int_dict(agent_probs)
        agent_set: set[int] = set()
        for v in agent_binding_set:
            try:
                agent_set.add(int(v))
            except (TypeError, ValueError):
                continue

        successful = [
            p for p in (tool_predictions or []) if p.get("success")
        ]

        binding_sets: dict[str, set[int]] = {}
        if self.feature_builder.use_gating:
            for pred in successful:
                tid = pred.get("tool_id") or ""
                bset: set[int] = set()
                for v in (pred.get("binding_protein_residues") or []):
                    try:
                        bset.add(int(v))
                    except (TypeError, ValueError):
                        continue
                binding_sets[tid] = bset

        self.feature_builder.tool_order = list(self.tool_order)

        rows: list[np.ndarray] = []
        residue_ids = list(range(1, int(protein_length) + 1))
        for res_id in residue_ids:
            tool_scores: dict[str, float] = {}
            for pred in successful:
                score = extract_score(pred, res_id, self.score_fields)
                if score is None:
                    continue
                tid = pred.get("tool_id") or ""
                tool_scores[tid] = self.standardizer.transform(tid, score)
            aa = aa_for(protein_sequence, res_id)
            flags: Optional[dict[str, bool]]
            if self.feature_builder.use_gating:
                flags = {
                    tid: (res_id in bset)
                    for tid, bset in binding_sets.items()
                }
            else:
                flags = None
            agent_prob = float(agent_probs_int.get(res_id, 0.0))
            agent_in_binding = res_id in agent_set
            rows.append(self.build_features(
                tool_scores, aa, flags, agent_prob, agent_in_binding,
            ))

        if not rows:
            return {}
        X = np.vstack(rows)
        probs = self.model.predict_proba(X)[:, 1]
        return {
            res_id: float(round(float(p), 6))
            for res_id, p in zip(residue_ids, probs)
        }

    # ------------------------------------------------------------------ I/O

    def save(self, dir_path: Union[str, Path]) -> Path:
        """Persist the bundle. ``xgb_meta.json`` carries the
        ``calibrated`` flag + ``model_type: "xgboost_calibrated"`` so
        :mod:`scripts.evaluate_learned_fusion` auto-detect routes here."""
        if self.model is None:
            raise RuntimeError("save() called before train() / load()")
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        self.standardizer.save(dir_path / self._STD_NAME)
        self.model.save_model(str(dir_path / self._MODEL_NAME))
        meta = {
            "version": 1,
            "model_type": self.MODEL_TYPE,
            "calibrated": True,
            "tool_order": list(self.tool_order),
            "feature_names": list(self.feature_names),
            "agent_feature_names": list(AGENT_FEATURE_NAMES),
            "score_fields": dict(self.score_fields),
            "xgb_params": dict(self.xgb_params),
            "config": {
                "use_residue_type": self.feature_builder.use_residue_type,
                "use_gating":       self.feature_builder.use_gating,
                "use_cross_terms":  self.feature_builder.use_cross_terms,
            },
        }
        (dir_path / self._META_NAME).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        if self.training_report:
            (dir_path / self._REPORT_NAME).write_text(
                json.dumps(self.training_report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return dir_path

    @classmethod
    def load(cls, dir_path: Union[str, Path]) -> "AgentCalibratedXGBFusion":
        _check_xgb_available()
        dir_path = Path(dir_path)
        meta_path = dir_path / cls._META_NAME
        meta = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.is_file() else {}
        )
        cfg = meta.get("config") or {}
        out = cls({
            "use_residue_type": bool(cfg.get("use_residue_type", True)),
            "use_gating":       bool(cfg.get("use_gating", True)),
            "use_cross_terms":  bool(cfg.get("use_cross_terms", False)),
            **(meta.get("xgb_params") or {}),
            "score_fields":     meta.get("score_fields") or {},
        })
        out.standardizer = ScoreStandardizer.load(dir_path / cls._STD_NAME)
        out.tool_order = list(meta.get("tool_order") or [])
        out.feature_builder.tool_order = list(out.tool_order)
        out.feature_names = list(meta.get("feature_names") or [])
        out.model = xgb.XGBClassifier(**out.xgb_params)
        out.model.load_model(str(dir_path / cls._MODEL_NAME))

        report_path = dir_path / cls._REPORT_NAME
        if report_path.is_file():
            try:
                out.training_report = json.loads(
                    report_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                out.training_report = {}
        return out
