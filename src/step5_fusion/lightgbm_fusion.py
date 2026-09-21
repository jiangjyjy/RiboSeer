"""HARMONY fusion — the system reported in the paper.

This is the final fusion in the RiboSeer pipeline: a LightGBM model over the
per-residue feature space of :mod:`step5_fusion.features_15tool` (the
per-tool blocks, cross-tool terms, window context and the SCOPE profile
block), plus the optional POLISH post-edit of the fused probability.

Three things make it the *full-system* configuration the paper reports
(paper Table 4, Pearson r = 0.593 on the 107-sample test split):

* every one of the 15 library tools contributes its own feature block, with
  unselected tools left as NaN so LightGBM splits on missingness;
* the SCOPE profile enters as a sample-level block, so the model conditions
  on the target profile;
* MAESTRO's per-sample selection is what decides which tools are active,
  with the MANDATORY core unioned in by the caller.

Training and inference are separate so a run can be reproduced exactly:
:func:`train_fullsystem_model` fits on a split and returns the model (or
``None`` when LightGBM is unavailable), and
:func:`compute_fullsystem_predictions` re-applies a fitted model to a set of
samples. :func:`fullsystem_feature_names` is the column contract, and every
downstream table that reads a prediction goes through
:mod:`step5_fusion.prediction_io`.
"""
from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from step3_tool_selection.tool_registry import MANDATORY_TOOLS
from .data_collector import load_sample_json, read_jsonl_record
from .features_15tool import (
    ALL_KNOWN_TOOLS, _canonical, _index_predictions, auto_scope_profile,
    build_15tool_features, encode_scope_profile, table9_feature_names,
)
from .metrics import per_sample_corr
from step7_iteration.polish_ops import apply_actions, auto_polish_action
from .prediction_io import save_lightgbm_model, save_prob_dict

try:
    import lightgbm as lgb  # type: ignore
    LGBM_OK = True
except ImportError:  # pragma: no cover - exercised on machines without lgbm
    lgb = None  # type: ignore
    LGBM_OK = False

logger = logging.getLogger("lightgbm_fusion")


def inject_mandatory(tools: Iterable[str],
                     mandatory: Iterable[str] = MANDATORY_TOOLS) -> list[str]:
    """``tools`` ∪ ``mandatory``, canonicalised, filtered to the library and
    returned in library order. Unknown ids are dropped; a mandatory tool the
    sample has no step-4 record for still ends up listed here (the feature
    builder gates it to NaN downstream)."""
    merged = {_canonical(t) for t in tools} | {_canonical(t) for t in mandatory}
    merged = {t for t in merged if t in ALL_KNOWN_TOOLS}
    return sorted(merged, key=ALL_KNOWN_TOOLS.index)




FIXED5 = ("boltz2", "chai1", "rosettafold2na", "equipnas", "p2rank")

# Combo order matches the plan's Table 9 (§4.3): all-off first, single
# modules, pairs, full system last. Tuple = (scope, maestro, polish).
COMBOS: list[tuple[bool, bool, bool]] = [
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
]


# ---------------------------------------------------------------------------
# Sample bundle
# ---------------------------------------------------------------------------


@dataclass
class SampleT9:
    sid: str
    step4_data: dict
    sample: dict
    residue_ids: list[int]
    y: np.ndarray
    eval_mask: np.ndarray
    protein_len: int
    rna_len: int
    available_tools: list[str]
    # filled lazily by resolvers / external JSON:
    cauto_profile: dict = field(default_factory=dict)


def _rna_len(sample: dict) -> int:
    rna = sample.get("rna") or sample.get("ligand_rna") or {}
    return int(rna.get("length") or len(rna.get("sequence") or "") or 0)


def collect_samples(step4_dir: Path, processed_dir: Path,
                    sample_ids: list[str]) -> list[SampleT9]:
    """One SampleT9 per usable sample. Skip reasons mirror
    EnrichedFusion._collect: missing step4 / sample JSON, zero GT, or no
    successful library-tool prediction."""
    out: list[SampleT9] = []
    for sid in sample_ids:
        s4 = read_jsonl_record(step4_dir / f"{sid}.jsonl")
        if s4 is None:
            continue
        sample = load_sample_json(processed_dir, sid)
        if sample is None:
            continue
        prot = sample.get("protein") or {}
        length = int(prot.get("length") or len(prot.get("sequence") or "") or 0)
        gt = {int(r) for r in (sample.get("interaction") or {})
              .get("binding_protein_residues") or []}
        if not length or not gt:
            continue
        avail = sorted(_index_predictions(s4).keys(),
                       key=lambda t: ALL_KNOWN_TOOLS.index(t))
        if not avail:
            continue

        residue_ids = list(range(1, length + 1))
        y = np.fromiter((1.0 if r in gt else 0.0 for r in residue_ids),
                        dtype=np.float64, count=length)
        resolved = prot.get("resolved_residues") or []
        if resolved:
            rs = {int(r) for r in resolved}
            eval_mask = np.fromiter((r in rs for r in residue_ids),
                                    dtype=bool, count=length)
        else:
            eval_mask = np.ones(length, dtype=bool)

        s = SampleT9(
            sid=sid, step4_data=s4, sample=sample,
            residue_ids=residue_ids, y=y, eval_mask=eval_mask,
            protein_len=length, rna_len=_rna_len(sample),
            available_tools=avail,
        )
        s.cauto_profile = auto_scope_profile(sample)
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Optional external (LLM) inputs
# ---------------------------------------------------------------------------


def _load_json_dir(directory: Optional[Path]) -> dict[str, dict]:
    """Load ``{sample_id: parsed_json}`` from ``<dir>/<sid>.json``.
    Missing dir → empty (Phase-1 control path)."""
    out: dict[str, dict] = {}
    if directory is None or not directory.is_dir():
        return out
    for f in directory.glob("*.json"):
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return out


# ---------------------------------------------------------------------------
# Per-sample resolvers (mode → concrete values)
# ---------------------------------------------------------------------------


def resolve_selected_tools(sample: SampleT9, maestro_on: bool,
                           selections: dict[str, dict],
                           mandatory: Optional[Iterable[str]] = None
                           ) -> list[str]:
    """MAESTRO's tool subset for one sample.

    ON  → LLM selection JSON if present, else all available tools.
    OFF → the fixed 5-tool baseline (``mandatory`` is ignored).

    ``mandatory`` (the headline guardrail, ``config.MANDATORY_TOOLS``) is
    unioned into the MAESTRO-on subset and the result is returned in library
    order; ``None`` (default) leaves the raw selection untouched — used by
    the "free LLM selection" arms (Table 11 v2) and the UCB online selection
    (Table 15), which must NOT be force-augmented.
    """
    if not maestro_on:
        return list(FIXED5)
    sel = selections.get(sample.sid)
    tools: list[str] = []
    if sel:
        tools = [_canonical(t) for t in (sel.get("selected_tools") or [])]
        tools = [t for t in tools if t in ALL_KNOWN_TOOLS]
    if not tools:
        tools = list(sample.available_tools)
    if mandatory:
        tools = inject_mandatory(tools, mandatory)
    return tools


def resolve_scope_vector(sample: SampleT9, scope_on: bool,
                         profiles: dict[str, dict]) -> np.ndarray:
    """SCOPE Group-4 vector for one sample.

    ON  → LLM profile JSON if present, else the cauto heuristic profile.
    OFF → zero block (``encode_scope_profile(None, ...)``).
    """
    if not scope_on:
        return encode_scope_profile(None, sample.protein_len, sample.rna_len)
    profile = profiles.get(sample.sid) or sample.cauto_profile
    return encode_scope_profile(profile, sample.protein_len, sample.rna_len)


# ---------------------------------------------------------------------------
# Feature matrices + LightGBM
# ---------------------------------------------------------------------------


def build_sample_matrix(sample: SampleT9, scope_on: bool, maestro_on: bool,
                        profiles: dict[str, dict],
                        selections: dict[str, dict],
                        mandatory: Optional[Iterable[str]] = None) -> np.ndarray:
    """Per-residue feature matrix for one (scope, maestro) setting.

    ``mandatory`` (the headline guardrail) is force-unioned into the
    MAESTRO-on selection; headline callers pass ``config.MANDATORY_TOOLS``,
    while the Table-15 UCB arm leaves it ``None``.
    """
    selected = resolve_selected_tools(sample, maestro_on, selections,
                                      mandatory=mandatory)
    scope_vec = resolve_scope_vector(sample, scope_on, profiles)
    return build_15tool_features(
        sample.step4_data, sample.residue_ids,
        selected_tools=selected, scope_vector=scope_vec,
        use_context=True, use_scope=True)


def _make_lightgbm():
    """LGBMRegressor with the project's standard recipe (matches
    enriched_fusion / ablation_fusion_method so rows stay comparable)."""
    return lgb.LGBMRegressor(  # type: ignore[union-attr]
        n_estimators=100, max_depth=4, learning_rate=0.1,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1)


# Exposed so tests can inject a deterministic fake without LightGBM.
def _train_model(X: np.ndarray, y: np.ndarray):
    model = _make_lightgbm()
    model.fit(X, y)
    return model


def predict_setting(
    scope_on: bool, maestro_on: bool,
    train: list[SampleT9], test: list[SampleT9],
    profiles_train: dict[str, dict], profiles_test: dict[str, dict],
    selections_train: dict[str, dict], selections_test: dict[str, dict],
) -> dict[str, dict[int, float]]:
    """Train one LightGBM for a (scope, maestro) setting and return
    ``{sid: {res_id: prob}}`` over the test set (pre-POLISH)."""
    X_train = np.vstack([
        build_sample_matrix(s, scope_on, maestro_on,
                            profiles_train, selections_train,
                            mandatory=MANDATORY_TOOLS)
        for s in train])
    y_train = np.concatenate([s.y for s in train])
    model = _train_model(X_train, y_train)

    preds: dict[str, dict[int, float]] = {}
    for s in test:
        X = build_sample_matrix(s, scope_on, maestro_on,
                                profiles_test, selections_test,
                                mandatory=MANDATORY_TOOLS)
        vec = np.asarray(model.predict(X), dtype=np.float64)
        preds[s.sid] = {rid: float(vec[i])
                        for i, rid in enumerate(s.residue_ids)}
    return preds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _corr_from_probdict(prob: dict[int, float],
                        sample: SampleT9) -> Optional[dict]:
    vec = np.fromiter((prob.get(r, 0.0) for r in sample.residue_ids),
                      dtype=np.float64, count=len(sample.residue_ids))
    mask = sample.eval_mask
    if mask.size != vec.size:
        return None
    return per_sample_corr(vec[mask], sample.y[mask])


def apply_polish(prob: dict[int, float], sample: SampleT9,
                 polish_on: bool, actions_by_sid: dict[str, dict]
                 ) -> dict[int, float]:
    if not polish_on:
        return prob
    saved = actions_by_sid.get(sample.sid)
    if saved:
        acts = None
        if isinstance(saved, dict):
            if isinstance(saved.get("rounds"), list):
                acts = saved["rounds"]       # iterative record (Table 12/14)
            elif isinstance(saved.get("actions"), list):
                acts = saved["actions"]      # multi-action wrapper
            else:
                acts = [saved]               # single-action record
        return apply_actions(prob, acts or [])
    return apply_actions(prob, [auto_polish_action(prob)])


# ---------------------------------------------------------------------------
# Full-system (on/on/on) prediction export — the headline pipeline
# (SCOPE + MAESTRO ∪ MANDATORY core + POLISH; R = 0.593)
# ---------------------------------------------------------------------------


def train_fullsystem_model(
    train: list[SampleT9],
    profiles_train: dict[str, dict], selections_train: dict[str, dict]):
    """Fit the on/on/on (SCOPE on + MAESTRO on, 15-tool) LightGBM on the
    train split and return the fitted estimator. Same recipe / feature
    space as the Table 9 on/on/on arm."""
    X_train = np.vstack([
        build_sample_matrix(s, True, True, profiles_train, selections_train,
                            mandatory=MANDATORY_TOOLS)
        for s in train])
    y_train = np.concatenate([s.y for s in train])
    return _train_model(X_train, y_train)


def fullsystem_predict_with_model(
    model, test: list[SampleT9],
    profiles_test: dict[str, dict], selections_test: dict[str, dict],
    polish_actions: dict[str, dict],
    *,
    polish_on: bool = True,
) -> dict[str, dict[int, float]]:
    """Apply a trained on/on/on model to the test set and return the
    per-residue probabilities ``{sid: {res_id: prob}}``.

    With ``polish_on`` (default) the saved POLISH actions are replayed — the
    Table 9/10/11 convention, where an absent action falls back to the
    deterministic control. Pass ``polish_on=False`` for the raw fusion."""
    out: dict[str, dict[int, float]] = {}
    for s in test:
        X = build_sample_matrix(s, True, True,
                                profiles_test, selections_test,
                                mandatory=MANDATORY_TOOLS)
        vec = np.asarray(model.predict(X), dtype=np.float64)
        prob = {rid: float(vec[i]) for i, rid in enumerate(s.residue_ids)}
        out[s.sid] = apply_polish(prob, s, polish_on, polish_actions)
    return out


def compute_fullsystem_predictions(
    *,
    train: list[SampleT9], test: list[SampleT9],
    profiles_train: dict[str, dict], profiles_test: dict[str, dict],
    selections_train: dict[str, dict], selections_test: dict[str, dict],
    polish_actions: dict[str, dict],
    return_model: bool = False,
):
    """Train the on/on/on LightGBM (SCOPE on + MAESTRO on) and return the
    **post-POLISH** per-residue probabilities ``{sid: {res_id: prob}}`` over
    the test set — i.e. exactly the prediction behind the Table 9 on/on/on
    row (RiboSeer 0.593). With ``return_model=True`` also returns the fitted
    LightGBM (for Table 27 feature importance / on-disk export)."""
    model = train_fullsystem_model(train, profiles_train, selections_train)
    out = fullsystem_predict_with_model(
        model, test, profiles_test, selections_test, polish_actions)
    if return_model:
        return out, model
    return out


def fullsystem_feature_names() -> list[str]:
    """Column-name contract of the on/on/on feature matrix (15-tool base +
    cross + context + SCOPE = 154-D), used when exporting the model."""
    return table9_feature_names(use_context=True, use_scope=True)


def save_fullsystem_predictions(
    out_dir: Path, test: list[SampleT9],
    preds: dict[str, dict[int, float]]) -> int:
    """Dump ``preds`` to ``<out_dir>/<sid>.json`` (residue_ids order =
    each sample's ``residue_ids``). Returns the count written."""
    n = 0
    for s in test:
        prob = preds.get(s.sid)
        if prob is None:
            continue
        save_prob_dict(out_dir, s.sid, prob, residue_ids=s.residue_ids)
        n += 1
    return n

# ---------------------------------------------------------------------------
# Single-sample inference (what the pipeline's step 5 calls)
# ---------------------------------------------------------------------------


def sample_for_prediction(
    sample_id: str,
    sample_json: dict,
    step4_record: dict,
    *,
    require_ground_truth: bool = False,
) -> Optional[SampleT9]:
    """Build the per-sample bundle for **inference**.

    Ground truth is optional here: ``y`` / ``eval_mask`` are only read when a
    sample is *scored*, so a prediction-only run can leave them empty. Returns
    ``None`` when the sample has no usable step-4 record or no protein length.
    """
    if step4_record is None:
        return None
    prot = sample_json.get("protein") or {}
    length = int(prot.get("length") or len(prot.get("sequence") or "") or 0)
    if not length:
        return None

    gt = {int(r) for r in (sample_json.get("interaction") or {})
          .get("binding_protein_residues") or []}
    if require_ground_truth and not gt:
        return None

    avail = sorted(_index_predictions(step4_record).keys(),
                   key=lambda t: ALL_KNOWN_TOOLS.index(t))
    if not avail:
        return None

    residue_ids = list(range(1, length + 1))
    resolved = prot.get("resolved_residues") or []
    if resolved:
        rs = {int(r) for r in resolved}
        eval_mask = np.fromiter((r in rs for r in residue_ids),
                                dtype=bool, count=length)
    else:
        eval_mask = np.ones(length, dtype=bool)

    sample = SampleT9(
        sid=sample_id, step4_data=step4_record, sample=sample_json,
        residue_ids=residue_ids,
        y=np.fromiter((1.0 if r in gt else 0.0 for r in residue_ids),
                      dtype=np.float64, count=length),
        eval_mask=eval_mask,
        protein_len=length, rna_len=_rna_len(sample_json),
        available_tools=avail,
    )
    sample.cauto_profile = auto_scope_profile(sample_json)
    return sample


def predict_one(
    model,
    sample: SampleT9,
    *,
    scope_profiles: Optional[dict[str, dict]] = None,
    selections: Optional[dict[str, dict]] = None,
    polish_actions: Optional[dict[str, dict]] = None,
    apply_polish_edit: bool = True,
) -> dict[int, float]:
    """Per-residue probabilities for one sample from a fitted model.

    ``scope_profiles`` / ``selections`` are the frozen SCOPE / MAESTRO outputs
    keyed by sample id; when a sample is absent from them the deterministic
    controls are used (the ``cauto`` profile and the 15-tool library), which is
    what the Phase-1 rows of Table 9 do.
    """
    actions = dict(polish_actions or {})
    # POLISH is the pipeline's refinement step (VERDICT): apply it only when
    # saved actions exist, or when the caller explicitly asks for the
    # deterministic control. Otherwise the raw fusion probability is returned.
    polish_on = bool(actions) or apply_polish_edit
    prob = fullsystem_predict_with_model(
        model, [sample], scope_profiles or {}, selections or {}, actions,
        polish_on=polish_on)
    return prob[sample.sid]


def fuse_sample(
    model,
    *,
    sample_id: str,
    sample_json: dict,
    step4_record: dict,
    scope_profiles: Optional[dict[str, dict]] = None,
    selections: Optional[dict[str, dict]] = None,
    polish_actions: Optional[dict[str, dict]] = None,
    threshold: float = 0.5,
):
    """Fuse one sample with a fitted model and return a ``CompositeResult``.

    Same record shape the rest of the pipeline already consumes, so VERDICT
    (step 6/7) and MEMORY (step 8) work unchanged. LightGBM assigns no
    explicit per-tool weight, so ``tool_weights`` reports the uniform
    contribution of the active tools and the rationale names the model.
    """
    from .schemas import CompositeResult  # local import: avoid a cycle at module load

    sample = sample_for_prediction(sample_id, sample_json, step4_record)
    if sample is None:
        raise ValueError(
            f"{sample_id}: no usable step-4 record for LightGBM fusion")

    # Pipeline semantics: replay saved POLISH actions when they exist, but do
    # not invent the deterministic control — refinement is VERDICT's job
    # (step 7), not the fusion's.
    prob = predict_one(model, sample, scope_profiles=scope_profiles,
                       selections=selections, polish_actions=polish_actions,
                       apply_polish_edit=bool(polish_actions))
    active = sample.available_tools
    return CompositeResult(
        sample_id=sample_id,
        tool_weights={t: 1.0 for t in active},
        binding_protein_residues=sorted(
            r for r, p in prob.items() if p > threshold),
        per_residue_probability={int(r): float(p) for r, p in prob.items()},
        threshold=threshold,
        fusion_rationale=(
            "HARMONY: LightGBM over the 154-D 15-tool feature space "
            f"({len(active)} tools active for this sample)"
            + (", POLISH edits applied" if polish_actions else "")),
        tools_fused=list(active),
    )

