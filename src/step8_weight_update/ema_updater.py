"""EMA update of W[K, M, J] from one PocketQAResult (Section 3.6, Phase 2, of the paper; formula 14).

Two public functions:

  - ``compute_per_tool_scores(...)`` decomposes the 5 fused PocketQA
    sub-scores back to per-tool, per-metric scores. The tensor W is
    per-tool but the QA scores are computed on the *fused* prediction;
    we need a heuristic that distributes credit reasonably. See the
    docstring for the per-metric strategy.

  - ``ema_update(...)`` applies the EMA rule
        W'[k, m, j] = (1-η) W[k, m, j] + η s_k^(m)
    in-place on a ``WeightTensor``, returns the per-cell delta dict, and
    bumps the per-(tool, category) evaluation counter so step 3's UCB
    exploration bonus shrinks for tools we've actually run.

Both are pure code (no LLM, no I/O). Phase 8.1's CLI calls them in
sequence and persists the resulting tensor.

Design choices
--------------
1. **Failed tools are not updated.** A success=False ToolPrediction has
   no signal to attribute — penalising an EMA cell would be wrong (the
   tool didn't get a chance to perform).
2. **q4 (consensus) is the only naturally per-tool metric.** We
   decompose it from ``info.pairwise_jaccard`` (each tool's avg Jaccard
   with the others is its consensus contribution). All other q_m are
   shared by surviving tools.
3. **Sharing weighted by tool_weights when present.** The fused score
   already credits high-weight tools more in the prediction; mirroring
   that in the W update keeps the two consistent. When no weights are
   available (e.g. equal-weights fallback), every surviving tool gets
   the raw fused sub-score.
4. **Cat A bonus on q1 (structural).** Only Cat A tools predict
   structures; Cat B/C tools "borrow" the fused q1 at half scale
   because they didn't contribute to it. This avoids penalising Cat A
   relative to Cat C just because Cat C piggy-backed on the structure.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from step3_tool_selection.weight_tensor import METRICS, WeightTensor
from step4_tool_adapters.schemas import ToolPrediction
from step5_fusion.schemas import CompositeResult
from step6_pocket_qa.schemas import PocketQAResult


# ---------- helpers --------------------------------------------------------


def _success_tool_ids(predictions: Iterable[ToolPrediction]) -> list[str]:
    return [p.tool_id for p in predictions if getattr(p, "success", False)]


def _normalised_share(weights: dict[str, float], tool_ids: list[str]) -> dict[str, float]:
    """Project ``weights`` onto ``tool_ids`` and renormalise to sum=1.

    When the projection is empty / sum<=0, fall back to equal shares.
    Used to convert step-5's tool_weights into "share of credit" for
    sub-scores that don't decompose naturally per tool.
    """
    if not tool_ids:
        return {}
    sliced = {t: max(0.0, float(weights.get(t, 0.0))) for t in tool_ids}
    total = sum(sliced.values())
    if total <= 0.0:
        share = 1.0 / len(tool_ids)
        return {t: share for t in tool_ids}
    return {t: sliced[t] / total for t in tool_ids}


def _per_tool_consensus(
    info: dict[str, Any], tool_ids: list[str], fused_score: float,
) -> dict[str, float]:
    """Decompose q4 from pairwise_jaccard.

    For each tool, score = mean(jaccard with each other active tool).
    If only one tool is active or pairwise dict is empty, fall back to
    the fused score (it's the closest proxy to "this tool agrees with
    itself"). Tools NOT in info.active_tools score 0 — they didn't
    participate in the consensus computation.
    """
    pj: dict[str, float] = info.get("pairwise_jaccard") or {}
    active = list(info.get("active_tools") or [])
    if not pj or len(active) < 2:
        return {t: float(fused_score) for t in tool_ids}

    out: dict[str, float] = {}
    for t in tool_ids:
        if t not in active:
            out[t] = 0.0
            continue
        sims = []
        for other in active:
            if other == t:
                continue
            key1 = f"{t}|{other}"
            key2 = f"{other}|{t}"
            v = pj.get(key1, pj.get(key2))
            if v is not None:
                sims.append(float(v))
        out[t] = (sum(sims) / len(sims)) if sims else 0.0
    return out


# ---------- per-metric decomposition --------------------------------------


def compute_per_tool_scores(
    qa_result: PocketQAResult,
    tool_predictions: list[ToolPrediction],
    composite_result: CompositeResult,
    config: Optional[dict] = None,
) -> dict[str, dict[str, float]]:
    """Decompose 5 fused sub-scores into per-tool scores ``s_k^(m)``.

    Returns ``{tool_id: {metric: score}}`` containing only surviving
    tools (success=True). Metric keys are exactly the 5 names from
    ``step3_tool_selection.weight_tensor.METRICS``. Cells where the
    corresponding sub-score *abstained* (qa_result.<metric> is None)
    are absent — abstained metrics carry no signal and shouldn't move
    the EMA.

    Per-metric policy
    -----------------
    - q1 structural_plausibility : Cat A tools get the full fused
      score; Cat B/C/D get half-credit (they didn't predict the
      structure but their binding sites overlap the same residues).
    - q2 physicochemical          : shared, weighted by tool_weights.
    - q3 evolutionary_conservation: shared, weighted by tool_weights.
    - q4 cross_tool_consensus     : decomposed via pairwise Jaccard.
    - q5 known_motif_consistency  : shared, weighted by tool_weights.

    The returned scores are NOT clamped here — clamping happens at the
    EMA step (W stays in [0, 1]).
    """
    cfg = config or {}
    cat_a_full = float(((cfg.get("decompose") or {}).get("cat_a_q1_full", 1.0)))
    other_q1   = float(((cfg.get("decompose") or {}).get("other_q1_share", 0.5)))

    survivors = [p for p in tool_predictions if getattr(p, "success", False)]
    survivor_ids = [p.tool_id for p in survivors]
    cat_by_tool = {p.tool_id: getattr(p, "category", "C") for p in survivors}

    if not survivor_ids:
        return {}

    # Step 5 weights → "share of credit" on shared metrics.
    weights = dict(getattr(composite_result, "tool_weights", {}) or {})
    share = _normalised_share(weights, survivor_ids)
    # When LLM weights are present we use them *as ratios* (not shares):
    # a tool with weight 0.7 gets 0.7 of the fused score, capped at 1.0.
    # This keeps the absolute scale comparable to the Cat A direct path.
    use_weight_directly = bool(weights)

    def _shared(score: float, info_tool: str) -> float:
        if use_weight_directly:
            w = max(0.0, float(weights.get(info_tool, 0.0)))
            return min(1.0, score * w) if w > 0 else 0.0
        return float(score) * float(share.get(info_tool, 0.0)) * len(survivor_ids)
        # ↑ multiplying by len(survivors) reverses the share normalisation
        #   so equal-weights-fallback also gets the raw fused score.

    out: dict[str, dict[str, float]] = {t: {} for t in survivor_ids}

    # ---- q1 structural --------------------------------------------------
    q1 = qa_result.structural_plausibility
    if q1 is not None:
        for t in survivor_ids:
            cat = cat_by_tool.get(t, "C")
            scale = cat_a_full if cat == "A" else other_q1
            out[t]["structural_plausibility"] = float(q1) * scale

    # ---- q2 physicochemical, q3 conservation, q5 motif (all shared) -----
    for metric, value in (
        ("physicochemical_complementarity", qa_result.physicochemical_complementarity),
        ("evolutionary_conservation", qa_result.evolutionary_conservation),
        ("known_motif_consistency", qa_result.known_motif_consistency),
    ):
        if value is None:
            continue
        for t in survivor_ids:
            out[t][metric] = _shared(float(value), t)

    # ---- q4 consensus (per-tool from pairwise Jaccard) ------------------
    q4 = qa_result.cross_tool_consensus
    if q4 is not None:
        info = (qa_result.details.get("cross_tool_consensus")
                .info if "cross_tool_consensus" in qa_result.details else {})
        per_tool_q4 = _per_tool_consensus(info or {}, survivor_ids, float(q4))
        for t, v in per_tool_q4.items():
            out[t]["cross_tool_consensus"] = float(v)

    # Drop any tool that ended up with no metric scores at all (shouldn't
    # happen unless every q_m abstained — defensive).
    return {t: m for t, m in out.items() if m}


# ---------- EMA pass -------------------------------------------------------


def _clip_unit(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def ema_update(
    weight_tensor: WeightTensor,
    category: str,
    per_tool_scores: dict[str, dict[str, float]],
    learning_rate: float = 0.1,
    *,
    increment_counts: bool = True,
) -> dict[str, dict[str, float]]:
    """Apply ``W' = (1-η)W + η s`` in-place on ``weight_tensor``.

    Parameters
    ----------
    weight_tensor
        The ``WeightTensor`` to mutate. Cells not yet written start at
        their cold-start default (per ``WeightTensor._default_for``).
    category
        Target category j.
    per_tool_scores
        ``{tool_id: {metric: score}}`` from ``compute_per_tool_scores``.
        Tools / metrics absent here are NOT touched (failed tools and
        abstained metrics).
    learning_rate
        η ∈ (0, 1]. The EMA weight on the new sample.
    increment_counts
        When True, also bumps the per-(tool, category) eval counter so
        step 3's UCB exploration bonus shrinks for tools we ran.

    Returns
    -------
    ``{tool_id: {metric: delta}}`` with delta = W_after - W_before for
    every cell that actually moved (matches the schema's ``ema_deltas``).
    """
    if not (0.0 < learning_rate <= 1.0):
        raise ValueError(
            f"learning_rate must be in (0, 1], got {learning_rate!r}"
        )
    if not category:
        raise ValueError("category must be a non-empty string")

    eta = float(learning_rate)
    deltas: dict[str, dict[str, float]] = {}

    for tool_id, mdict in per_tool_scores.items():
        if not mdict:
            continue
        per_tool_deltas: dict[str, float] = {}
        for metric, raw_score in mdict.items():
            if metric not in METRICS:
                # Defensive: skip unknown metrics rather than crash.
                continue
            score = _clip_unit(float(raw_score))
            before = weight_tensor.get_weight(tool_id, metric, category)
            after = _clip_unit((1.0 - eta) * before + eta * score)
            delta = after - before
            weight_tensor.update(tool_id, category, metric, after)
            per_tool_deltas[metric] = delta
        if per_tool_deltas:
            deltas[tool_id] = per_tool_deltas
            if increment_counts:
                weight_tensor.increment_count(tool_id, category)

    return deltas


# ---------- snapshot helpers (used by run.py + tests) ---------------------


def slice_for_snapshot(
    weight_tensor: WeightTensor,
    category: str,
    tool_ids: Iterable[str],
) -> dict[str, dict[str, float]]:
    """Read the current ``W[:, :, j]`` slice for the named tools.

    Used to capture before / after snapshots for ``WeightUpdateResult``.
    Reads only the 5 canonical metrics, so the snapshot dict has a
    fixed shape regardless of W's internal sparsity.
    """
    return {
        t: {m: weight_tensor.get_weight(t, m, category) for m in METRICS}
        for t in tool_ids
    }
