"""Step 6 main scoring logic — compute 5 sub-scores and a weighted total.

Pipeline (Section 3.7)
----------------------
  1. Pull the inputs each metric needs from the upstream artefacts:
     - ``CompositeResult`` (step 5): fused binding sets + tool weights.
     - ``ToolPrediction`` list (step 4): per-tool predictions for the
       consensus calculation; the first Cat A tool's predicted
       structure path feeds the structural metric.
     - ``sample_json`` (step 1): protein/RNA sequences and features.
  2. Run each q_m independently inside a try/except so a single bad
     metric never aborts the others. Successful metrics return
     ``(score, info)``; failures (raised exceptions OR a ``score=None``
     return) drop out of the weighted sum.
  3. Aggregate (formula 17 in the paper) with **renormalised** weights — when a
     metric drops out, its weight is excluded from the denominator so
     "missing data" does not penalise the score. ``total_score = 0.0``
     when zero metrics computed.
  4. Pack everything into a ``PocketQAResult``. Never raises; bad input
     just produces a result with ``n_metrics_computed == 0``.

No LLM calls; pure Python.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

from step4_tool_adapters.schemas import ToolPrediction
from step5_fusion.schemas import CompositeResult

from .metrics.consensus import cross_tool_consensus
from .metrics.conservation import evolutionary_conservation
from .metrics.motif import known_motif_consistency
from .metrics.physicochemical import physicochemical_complementarity
from .metrics.structural import structural_plausibility
from .schemas import METRIC_NAMES, MetricDetail, PocketQAResult


# ---------- helpers --------------------------------------------------------


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_metric_call(fn: Callable, **kwargs) -> MetricDetail:
    """Run one metric, trapping any exception into ``MetricDetail.error``.

    Two outcome paths:
      - ``fn`` returns ``(None, info)`` → metric chose to abstain
        (e.g. missing input data). ``computed=False``, ``error=None``.
      - ``fn`` raises → ``computed=False``, ``error=str(exc)``.
      - ``fn`` returns ``(score, info)`` with score in [0, 1] →
        ``computed=True``, ``score`` set, ``error=None``.
    Out-of-range scores are clamped to [0, 1] defensively.
    """
    try:
        result = fn(**kwargs)
    except Exception as exc:  # noqa: BLE001 — we want every failure mode
        return MetricDetail(
            score=None, computed=False,
            error=f"{type(exc).__name__}: {exc}"[:1500],
            info={},
        )

    if not (isinstance(result, tuple) and len(result) == 2):
        return MetricDetail(
            score=None, computed=False,
            error=f"metric returned {type(result).__name__}, expected (score, info) tuple",
            info={},
        )
    score, info = result
    if not isinstance(info, dict):
        info = {"_raw_info_type": type(info).__name__}

    if score is None:
        return MetricDetail(score=None, computed=False, info=info)

    try:
        s = float(score)
    except (TypeError, ValueError) as exc:
        return MetricDetail(
            score=None, computed=False,
            error=f"non-numeric score {score!r}: {exc}",
            info=info,
        )
    if s < 0.0:
        s = 0.0
    elif s > 1.0:
        s = 1.0
    return MetricDetail(score=s, computed=True, info=info)


def _pick_structure_path(tool_predictions: Iterable[ToolPrediction]) -> Optional[str]:
    """Return the first Cat A success's predicted structure path, if any.

    Cat A tools (Boltz-2, Chai-1) emit a PDB/CIF for the complex.
    Step 6's structural metric only needs one such file — pick the
    first available so we don't double-count tools that happened to
    run on the same sample.
    """
    for p in tool_predictions:
        if (getattr(p, "success", False)
                and getattr(p, "category", "") == "A"
                and getattr(p, "predicted_structure_path", None)):
            return p.predicted_structure_path
    return None


def _aggregate_weighted(
    details: dict[str, MetricDetail],
    cfg_weights: dict[str, float],
) -> tuple[float, dict[str, float], int]:
    """Weighted sum with renormalisation over computed metrics only.

    Returns ``(total_score, weights_used, n_computed)`` where
    ``weights_used`` carries only the metrics that contributed (i.e.
    ``computed=True`` AND the configured weight is positive).
    ``total_score`` is clamped to [0, 1] after the divide.
    """
    num = 0.0
    den = 0.0
    used: dict[str, float] = {}
    n = 0
    for name in METRIC_NAMES:
        d = details.get(name)
        if d is None or not d.computed or d.score is None:
            continue
        w = float(cfg_weights.get(name, 0.0))
        if w <= 0.0:
            continue
        num += w * d.score
        den += w
        used[name] = w
        n += 1
    total = (num / den) if den > 0 else 0.0
    if total < 0.0:
        total = 0.0
    elif total > 1.0:
        total = 1.0
    return round(total, 6), used, n


# ---------- main entry point -----------------------------------------------


def score_prediction(
    sample_json: dict,
    composite_result: CompositeResult,
    tool_predictions: list[ToolPrediction],
    config: dict,
) -> PocketQAResult:
    """Compute the 5 unsupervised QA sub-scores and the weighted total.

    Parameters
    ----------
    sample_json
        Step 1 sample dict (provides protein/RNA sequence + features).
    composite_result
        Step 5 fused output for the same sample.
    tool_predictions
        Step 4 ``ToolPrediction`` list — used by the consensus metric
        and to locate a predicted structure for the structural metric.
    config
        Loaded ``step6_config.yaml`` contents. Reads
        ``pocket_qa.weights`` and per-metric sub-blocks.

    Never raises. Bad inputs surface as ``n_metrics_computed=0`` and a
    ``total_score`` of 0.0 with the per-metric error captured in
    ``details``.
    """
    sample_id = composite_result.sample_id

    qa_cfg: dict[str, Any] = (config.get("pocket_qa") or {})
    cfg_weights: dict[str, float] = (qa_cfg.get("weights") or {})

    # Pull sequences from sample_json (step 1 layout).
    protein = (sample_json.get("protein") or {})
    rna = (sample_json.get("rna") or {})
    protein_seq: str = protein.get("sequence") or ""
    rna_seq: str = rna.get("sequence") or ""
    protein_length: int = int(protein.get("length") or len(protein_seq) or 0)

    # Step 2 category may live in the sample (joined upstream) or be
    # absent here — accept both shapes.
    target_category: str = (
        (sample_json.get("target_char") or {}).get("category")
        or (sample_json.get("step2") or {}).get("category")
        or ""
    )

    binding_residues = list(composite_result.binding_protein_residues or [])
    binding_nuc = list(composite_result.binding_rna_nucleotides or [])
    structure_path = _pick_structure_path(tool_predictions)

    details: dict[str, MetricDetail] = {}

    details["structural_plausibility"] = _safe_metric_call(
        structural_plausibility,
        binding_residues=binding_residues,
        structure_path=structure_path,
        protein_length=protein_length,
        config=qa_cfg.get("structural") or {},
    )

    details["physicochemical_complementarity"] = _safe_metric_call(
        physicochemical_complementarity,
        binding_residues=binding_residues,
        protein_sequence=protein_seq,
        rna_sequence=rna_seq,
        binding_nucleotides=binding_nuc,
        config=qa_cfg.get("physicochemical") or {},
    )

    details["evolutionary_conservation"] = _safe_metric_call(
        evolutionary_conservation,
        binding_residues=binding_residues,
        protein_sequence=protein_seq,
        sample_json=sample_json,
        config=qa_cfg.get("conservation") or {},
    )

    details["cross_tool_consensus"] = _safe_metric_call(
        cross_tool_consensus,
        composite_binding=binding_residues,
        tool_predictions=list(tool_predictions),
        config=qa_cfg.get("consensus") or {},
    )

    details["known_motif_consistency"] = _safe_metric_call(
        known_motif_consistency,
        binding_residues=binding_residues,
        protein_sequence=protein_seq,
        target_category=target_category,
        sample_json=sample_json,
        config=qa_cfg.get("motif") or {},
    )

    total_score, weights_used, n_computed = _aggregate_weighted(details, cfg_weights)

    return PocketQAResult(
        sample_id=sample_id,
        structural_plausibility=details["structural_plausibility"].score,
        physicochemical_complementarity=details["physicochemical_complementarity"].score,
        evolutionary_conservation=details["evolutionary_conservation"].score,
        cross_tool_consensus=details["cross_tool_consensus"].score,
        known_motif_consistency=details["known_motif_consistency"].score,
        total_score=total_score,
        weights_used=weights_used,
        n_metrics_computed=n_computed,
        details=details,
        timestamp=_timestamp(),
    )
