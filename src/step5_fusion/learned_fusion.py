"""Learned fusion: data-driven replacement for LLM weights + noisy-OR.

Pipeline
--------
Training (offline, on a labelled batch):
    1. Walk the training-set step4/ JSONLs; for each (sample, residue):
        - read the configured score field per tool (PAE for boltz2/chai1,
          confidence for p2rank/fpocket/equipnas)
        - emit (per-tool score vector, residue_type, GT_label)
    2. Fit ScoreStandardizer on the per-tool score lists.
    3. Standardise everything; build the design matrix X.
    4. Fit WeightOptimizer (Ridge) → {weights, bias}.
    5. Save the bundle (standardizer.json + optimizer.json + report).

Inference (per sample):
    1. Load standardizer + optimizer.
    2. For each protein residue: gather per-tool scores → standardise →
       build_features(... amino-acid letter) → predict.
    3. Return ``{residue_idx: probability}`` — the same shape evaluate.py
       already consumes for the per-residue correlation table.

Design choices
--------------
- ``predict_sample`` returns dense scores for residues 1..protein_length
  (every position gets a probability), matching the evaluate.py
  expectation that PAE-style scores cover the whole sequence rather than
  only the noisy-OR-gated binding list. A residue with NO tool score is
  predicted from the bias + amino-acid-only features (residual signal).
- This module never touches fusion.py / noisy_or.py — the learned model
  is purely additive, used only by the new train / evaluate scripts.
- All artefacts round-trip through plain JSON (no pickle / no numpy
  ``.npy`` blobs) so a model trained on one host loads on another.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from .data_collector import (
    DEFAULT_SCORE_FIELDS,
    aa_for as _aa_for,
    collect_per_residue_data,
    extract_score as _extract_score_free,
    load_sample_json as _load_sample_json,
    per_residue_to_int_dict as _per_residue_to_int_dict,
    read_jsonl_record as _read_jsonl_record,
    resolve_score_fields,
)
from .standardizer import ScoreStandardizer
from .weight_optimizer import AA20, WeightOptimizer


# Underscore aliases above keep backward compat with scripts that
# imported these helpers from learned_fusion before the refactor
# (notably scripts/evaluate_learned_fusion.py).


# ---- main class ----------------------------------------------------------


class LearnedFusion:
    """Data-driven fusion model — standardiser + Ridge optimiser bundle.

    Parameters
    ----------
    config : dict
        Optional. Keys (with defaults):
          - ``use_residue_type`` (True)
          - ``use_cross_terms``  (False)
          - ``regularization``   (0.01)
          - ``score_fields``     (DEFAULT_SCORE_FIELDS)
          - ``tools``            optional dict for per-tool overrides
            of ``score_field`` (matches the YAML schema in
            configs/learned_fusion_config.yaml)
    """

    # File names inside the saved bundle directory.
    _STD_NAME = "standardizer.json"
    _OPT_NAME = "optimizer.json"
    _META_NAME = "feature_names.json"  # mirrors optimizer.feature_names + score_fields
    _REPORT_NAME = "training_report.json"

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.config = dict(cfg)
        self.score_fields = self._resolve_score_fields(cfg)
        self.standardizer = ScoreStandardizer()
        self.optimizer = WeightOptimizer(
            use_residue_type=bool(cfg.get("use_residue_type", True)),
            use_gating=bool(cfg.get("use_gating", True)),
            use_cross_terms=bool(cfg.get("use_cross_terms", False)),
            regularization=float(cfg.get("regularization", 0.01)),
        )
        # Class weighting is a training-time setting only — the trained
        # weights already encode the effect, so it isn't part of the
        # WeightOptimizer's persisted config. Stored here so the
        # training_report can record what was used.
        # Accepted values: "balanced" | "none" | None | numeric (float-like).
        self.class_weight = cfg.get("class_weight", None)
        # Set on train(); used by predict_sample to produce stable
        # column ordering across calls.
        self.tool_order: list[str] = []
        self.training_report: dict = {}

    # ------------------------------------------------------------------ config

    @staticmethod
    def _resolve_score_fields(cfg: dict) -> dict[str, str]:
        """Thin wrapper around data_collector.resolve_score_fields kept
        as a static method for backward-compat with old call sites."""
        return resolve_score_fields(cfg or {})

    # ------------------------------------------------------------------ data collection

    def _extract_score(
        self, prediction: dict, residue_id: int,
    ) -> Optional[float]:
        """Thin wrapper around data_collector.extract_score; kept on
        the class for the predict_sample() inference path."""
        return _extract_score_free(prediction, residue_id, self.score_fields)

    def collect_training_data(
        self,
        *,
        step4_dir: Path,
        processed_dir: Path,
        sample_ids: Optional[Iterable[str]] = None,
    ) -> dict:
        """Walk step4/ JSONLs + sample JSONs → per-residue training rows.

        ``sample_ids`` restricts to a known subset (typical: the train
        split's ``train.txt``); if None, every step4 JSONL under
        ``step4_dir`` is processed.

        Returns
        -------
        bundle : dict with keys:
          - ``per_tool_scores`` : ``{tool_id: [score, score, ...]}``
            flat lists for the standardiser to fit on.
          - ``rows`` : list of per-(sample, residue) entries — see
            :func:`data_collector.collect_per_residue_data` for the
            row schema.
          - ``n_samples_used`` / ``n_samples_skipped`` : provenance.

        Implementation note: this method is now a thin wrapper around
        ``data_collector.collect_per_residue_data`` so XGBFusion can
        share the exact same training input without inheriting from
        LearnedFusion.
        """
        return collect_per_residue_data(
            step4_dir=step4_dir,
            processed_dir=processed_dir,
            score_fields=self.score_fields,
            sample_ids=sample_ids,
        )

    # ------------------------------------------------------------------ training

    @staticmethod
    def _resolve_class_weight(
        class_weight, y: np.ndarray,
    ) -> tuple[Optional[np.ndarray], dict]:
        """Translate the ``class_weight`` setting → per-row sample weights.

        Accepted forms:
          * ``None`` / ``"none"`` (case-insensitive) — no weighting; fit
            falls back to plain Ridge (returns ``(None, info)``).
          * ``"balanced"`` — pos_weight = neg_count / max(pos_count, 1);
            negatives get weight 1, positives get ``pos_weight``. With
            ~5-6 % positive rate this is ~17×, which is what closed
            most of the gap to noisy-OR in calibration.
          * a numeric value (int / float / numeric str) — used directly
            as the positive-class weight; useful for grid-searching
            something other than the auto-balanced default.

        Returns ``(sample_weights, info_dict)``. The info dict goes
        into the training report so the bundle records exactly what
        was applied.
        """
        # Normalise input.
        cw = class_weight
        if isinstance(cw, str):
            cw = cw.strip().lower()
            if cw in ("none", ""):
                cw = None

        info: dict = {"class_weight": class_weight, "pos_weight": None,
                      "neg_weight": 1.0}
        if cw is None:
            info["mode"] = "none"
            return None, info

        n = int(y.size)
        pos_count = int((y > 0.5).sum())
        neg_count = n - pos_count

        if cw == "balanced":
            pos_weight = neg_count / max(pos_count, 1)
            info["mode"] = "balanced"
        else:
            try:
                pos_weight = float(cw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"unsupported class_weight={class_weight!r}; expected "
                    f"'balanced' / 'none' / numeric"
                )
            info["mode"] = "manual"
        info["pos_weight"] = float(pos_weight)
        info["pos_count"] = pos_count
        info["neg_count"] = neg_count

        weights = np.where(y > 0.5, pos_weight, 1.0).astype(np.float64)
        return weights, info

    def train(
        self,
        *,
        step4_dir: Path,
        processed_dir: Path,
        sample_ids: Optional[Iterable[str]] = None,
        verbose: bool = True,
    ) -> dict:
        """Full training loop. Returns the report dict (also stashed)."""
        bundle = self.collect_training_data(
            step4_dir=step4_dir, processed_dir=processed_dir,
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
            print(f"[learned_fusion] collected {len(rows)} per-residue rows "
                  f"from {bundle['n_samples_used']} samples "
                  f"(skipped {bundle['n_samples_skipped']})")
            print(f"[learned_fusion] per-tool training counts: "
                  f"{ {t: len(v) for t, v in per_tool.items()} }")

        # 1) standardiser
        self.standardizer.fit(per_tool)

        # 2) freeze tool_order (sorted for determinism / save-load parity).
        # Use the union of tools that produced scores AND tools that
        # produced binding flags so a Cat A tool that emitted only a
        # binding list (no per-residue score field) still gets a
        # gating column. The gating columns it owns will be 0 for
        # residues outside its binding list and 1 inside — exactly
        # what we want.
        all_tool_ids: set[str] = set(per_tool.keys())
        for r in rows:
            all_tool_ids.update(r.get("tool_binding_flags") or {})
        self.tool_order = sorted(all_tool_ids)

        # 3) build X / y
        X_rows: list[np.ndarray] = []
        y: list[float] = []
        for r in rows:
            std_scores = {
                tid: self.standardizer.transform(tid, sc)
                for tid, sc in r["tool_scores"].items()
            }
            self.optimizer.tool_order = self.tool_order  # pin column order
            X_rows.append(self.optimizer.build_features(
                std_scores, r["aa"], r.get("tool_binding_flags"),
            ))
            y.append(float(r["label"]))
        X = np.vstack(X_rows)
        y_arr = np.asarray(y, dtype=np.float64)

        # 4) class weighting (from config)
        sample_weights, weight_info = self._resolve_class_weight(
            self.class_weight, y_arr,
        )
        if verbose:
            if weight_info["mode"] == "none":
                print("[learned_fusion] class_weight: none (plain Ridge)")
            else:
                print(
                    f"[learned_fusion] class_weight: "
                    f"mode={weight_info['mode']} "
                    f"pos_weight={weight_info['pos_weight']:.2f} "
                    f"(pos={weight_info.get('pos_count')}, "
                    f"neg={weight_info.get('neg_count')})"
                )

        # 5) Ridge fit
        report = self.optimizer.fit(
            X, y_arr,
            sample_weights=sample_weights,
            tool_order=self.tool_order,
            verbose=verbose,
        )
        report["n_samples_used"] = bundle["n_samples_used"]
        report["n_samples_skipped"] = bundle["n_samples_skipped"]
        report["per_tool_train_counts"] = {
            t: len(v) for t, v in per_tool.items()
        }
        report["score_fields"] = dict(self.score_fields)
        report["class_weight"] = weight_info
        self.training_report = report
        return report

    # ------------------------------------------------------------------ inference

    def predict_sample(
        self,
        tool_predictions: list[dict],
        protein_sequence: str,
        protein_length: int,
    ) -> dict[int, float]:
        """Predict per-residue binding probability for one sample.

        ``tool_predictions`` are dicts in the same shape ``ToolPrediction``
        serialises to (the step4 JSONL ``predictions[]`` entries). This
        keeps the inference path purely data-driven — no pydantic
        coupling — so the function works with anything that round-trips
        through JSON.

        Returns ``{residue_id_1_based: probability}`` for residues
        ``1..protein_length``. Residues that no tool scored still get a
        prediction (driven by the bias + amino-acid one-hot block);
        callers can decide to ignore those by intersecting the keys
        with the union of per-tool score keys themselves.
        """
        if self.optimizer.weights is None:
            raise RuntimeError(
                "LearnedFusion.predict_sample called before train() / load()"
            )
        successful = [
            p for p in (tool_predictions or []) if p.get("success")
        ]
        # Pin column order so build_features doesn't re-derive it
        # from a partial input dict.
        self.optimizer.tool_order = list(self.tool_order)

        # Pre-compute each tool's binding-list as a set so the per-
        # residue gating-flag lookup is O(1). Only rebuild when
        # use_gating is on (otherwise the flags are unused).
        binding_sets: dict[str, set[int]] = {}
        if self.optimizer.use_gating:
            for pred in successful:
                tid = pred.get("tool_id") or ""
                br = pred.get("binding_protein_residues") or []
                bset: set[int] = set()
                for v in br:
                    try:
                        bset.add(int(v))
                    except (TypeError, ValueError):
                        continue
                binding_sets[tid] = bset

        rows: list[np.ndarray] = []
        residue_ids = list(range(1, int(protein_length) + 1))
        for res_id in residue_ids:
            tool_scores: dict[str, float] = {}
            for pred in successful:
                score = self._extract_score(pred, res_id)
                if score is None:
                    continue
                tid = pred.get("tool_id") or ""
                tool_scores[tid] = self.standardizer.transform(tid, score)
            aa = _aa_for(protein_sequence, res_id)
            flags: Optional[dict[str, bool]]
            if self.optimizer.use_gating:
                flags = {
                    tid: (res_id in bset)
                    for tid, bset in binding_sets.items()
                }
            else:
                flags = None
            rows.append(self.optimizer.build_features(tool_scores, aa, flags))

        X = np.vstack(rows) if rows else np.zeros((0, len(self.optimizer.weights)))
        probs = self.optimizer.predict(X) if rows else np.zeros((0,))
        return {
            res_id: float(round(p, 6)) for res_id, p in zip(residue_ids, probs)
        }

    # ------------------------------------------------------------------ I/O

    def save(self, dir_path: Union[str, Path]) -> Path:
        """Persist standardiser + optimiser + report into ``dir_path/``.

        Three files land underneath:
          - ``standardizer.json``     ScoreStandardizer.to_dict()
          - ``optimizer.json``        WeightOptimizer.to_dict()
          - ``feature_names.json``    {tool_order, feature_names,
                                       score_fields, config}
          - ``training_report.json``  metrics from the last train() call

        Returns the directory path.
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        self.standardizer.save(dir_path / self._STD_NAME)
        self.optimizer.save(dir_path / self._OPT_NAME)
        meta = {
            "tool_order": list(self.tool_order),
            "feature_names": list(self.optimizer.feature_names),
            "score_fields": dict(self.score_fields),
            "config": {
                "use_residue_type": self.optimizer.use_residue_type,
                "use_gating":       self.optimizer.use_gating,
                "use_cross_terms":  self.optimizer.use_cross_terms,
                "regularization":   self.optimizer.regularization,
                # Class weighting affects training only; the trained
                # weights already encode it. Persist it for provenance
                # so the saved bundle is fully self-describing.
                "class_weight":     self.class_weight,
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
    def load(cls, dir_path: Union[str, Path]) -> "LearnedFusion":
        """Load a bundle saved by :meth:`save`."""
        dir_path = Path(dir_path)
        meta_path = dir_path / cls._META_NAME
        meta = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.is_file() else {}
        )
        cfg = meta.get("config") or {}
        # Re-inject the per-tool score field overrides so inference
        # uses the same ``per_residue_pae_score`` vs
        # ``per_residue_confidence`` map the trainer used. Backward
        # compat: pre-v2 bundles have no ``use_gating`` key, default
        # off so the loaded design matches what was saved.
        score_fields = meta.get("score_fields") or {}
        out = cls({
            "use_residue_type": bool(cfg.get("use_residue_type", True)),
            "use_gating":       bool(cfg.get("use_gating", False)),
            "use_cross_terms":  bool(cfg.get("use_cross_terms", False)),
            "regularization":   float(cfg.get("regularization", 0.01)),
            "class_weight":     cfg.get("class_weight"),
            "score_fields":     score_fields,
        })
        out.standardizer = ScoreStandardizer.load(dir_path / cls._STD_NAME)
        out.optimizer = WeightOptimizer.load(dir_path / cls._OPT_NAME)
        out.tool_order = list(meta.get("tool_order")
                              or out.optimizer.tool_order)
        # Pin the optimiser's tool_order to the saved meta in case the
        # config dict shipped a different order on load.
        out.optimizer.tool_order = list(out.tool_order)
        report_path = dir_path / cls._REPORT_NAME
        if report_path.is_file():
            try:
                out.training_report = json.loads(
                    report_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                out.training_report = {}
        return out
