"""CLI entry point for step 8 — weight tensor update (Section 3.6, Phase 2, of the paper).

Reads existing step 2 / 4 / 5 / 6 / 7 JSONL outputs + the current
``WeightTensor`` JSON and produces:
  - One ``WeightUpdateResult`` JSONL line per sample under ``--output``.
  - A refreshed ``WeightTensor`` JSON at ``--weight-tensor`` (atomic
    replace; the same path is loaded if it exists, else cold-start).

Typical invocation::

    python -m step8_weight_update.run \\
        --processed-dir data/processed \\
        --step2-output data/step2_outputs/e2e_smoke.jsonl \\
        --step4-output data/step4_outputs/1un6_B_F.jsonl \\
        --step5-output data/step5_outputs/e2e_smoke.jsonl \\
        --step6-output data/step6_outputs/e2e_smoke.jsonl \\
        --step7-output data/step7_outputs/e2e_smoke.jsonl \\
        --weight-tensor data/weights/weight_tensor.json \\
        --history data/history/prediction_history.jsonl \\
        --config configs/step8_config.yaml \\
        --output data/step8_outputs/

Pipeline per sample
-------------------
  1. Reconstruct step 4 / 5 / 6 records into Pydantic objects.
  2. ``compute_per_tool_scores`` → ``{tool_id: {metric: score}}``.
  3. Snapshot W slice → ``weights_before``.
  4. ``ema_update`` mutates W in-place; capture deltas + ``weights_after``.
  5. Optional ``meta_correct`` → multiplies γ_k into W; refreshes
     ``weights_after`` snapshot if applied.
  6. Append a ``PredictionHistory`` record (score / tools / category)
     and persist history JSONL.
  7. Save the updated ``WeightTensor``.
  8. Emit one ``WeightUpdateResult`` JSONL line.

Exit codes
----------
- 0  every requested sample produced an update
- 1  setup error (missing inputs, malformed records, LLM client build)
- 2  at least one sample was skipped
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import yaml

from step2_target_char.llm_client import LLMClient
from step2_target_char.history import PredictionHistory
from step2_target_char.run import build_client, load_config as load_step2_config
from step3_tool_selection.tool_registry import get_all_tool_ids
from step3_tool_selection.weight_tensor import METRICS, WeightTensor
# Re-use step 6's CLI helpers for the step 4 / 5 record-rebuild path.
from step6_pocket_qa.run import (
    _category_for as _step6_category_for,
    _step4_record_to_predictions,
    _step5_record_to_composite,
)
from step7_iteration.run import _step6_record_to_qa
from step7_iteration.schemas import IterationResult

from .ema_updater import (
    compute_per_tool_scores,
    ema_update,
    slice_for_snapshot,
)
from .meta_correction import (
    MetaCorrectionResult,
    apply_correction_factors,
    meta_correct,
)
from .schemas import WeightUpdateResult


# ---------- IO helpers -----------------------------------------------------


def load_config(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def _index_by_sid(rows: Iterable[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        sid = r.get("sample_id")
        if sid:
            out[sid] = r
    return out


def _step7_record_to_iter_result(record: dict) -> IterationResult:
    """Reverse of ``IterationResult.model_dump(mode='json')``."""
    return IterationResult.model_validate(record)


# ---------- output writer --------------------------------------------------


def _resolve_output_path(out_arg: Optional[Path], sample_id: str) -> Optional[Path]:
    if out_arg is None:
        return None
    if out_arg.suffix in (".jsonl", ".json"):
        out_arg.parent.mkdir(parents=True, exist_ok=True)
        return out_arg
    out_arg.mkdir(parents=True, exist_ok=True)
    return out_arg / f"{sample_id}.jsonl"


def _write_record_atomic(path: Path, record: dict, *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if append and path.is_file():
        existing = path.read_text(encoding="utf-8")
    else:
        existing = ""
    body = existing + json.dumps(record, ensure_ascii=False) + "\n"
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


def _save_tensor_atomic(weight_tensor: WeightTensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    weight_tensor.save(tmp)
    tmp.replace(path)


# ---------- summary printers ----------------------------------------------


def _summarize(record: dict) -> str:
    sid = record.get("sample_id", "?")
    cat = record.get("category", "?")
    tools = record.get("tools_updated") or []
    n_meta = (
        len(record.get("correction_factors") or {})
        if record.get("meta_correction_applied") else 0
    )
    return (
        f"{sid:<16}  cat={cat:<28}  "
        f"updated={','.join(tools) or '(none)'}  meta_factors={n_meta}"
    )


def _build_history_record(
    sid: str, category: str, qa_record: dict, iter_record: dict,
    tool_predictions, ts: str,
) -> dict:
    """One PredictionHistory entry for this sample.

    Used by future meta-correction calls (across-sample analysis). Keys
    chosen to match what ``prompts.summarise_history`` looks for.
    """
    return {
        "sample_id": sid,
        "category": category,
        "timestamp": ts,
        "scores": {
            "structural_plausibility": qa_record.get("structural_plausibility"),
            "physicochemical_complementarity":
                qa_record.get("physicochemical_complementarity"),
            "evolutionary_conservation":
                qa_record.get("evolutionary_conservation"),
            "cross_tool_consensus": qa_record.get("cross_tool_consensus"),
            "known_motif_consistency":
                qa_record.get("known_motif_consistency"),
            "total": qa_record.get("total_score"),
        },
        "tools_used": [
            p.tool_id for p in tool_predictions
            if getattr(p, "success", False)
        ],
        "final_action": iter_record.get("final_action"),
        "final_score": iter_record.get("final_score"),
    }


# ---------- main -----------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Step 8 — weight tensor update (EMA + optional LLM γ correction)",
    )
    ap.add_argument("--processed-dir", type=Path, required=True,
                    help="step 1 output dir containing samples/<id>.json")
    ap.add_argument("--step4-output", type=Path, required=True,
                    help="JSONL from step 4")
    ap.add_argument("--step5-output", type=Path, required=True,
                    help="JSONL from step 5 (CompositeResult per line)")
    ap.add_argument("--step6-output", type=Path, required=True,
                    help="JSONL from step 6 (PocketQAResult per line)")
    ap.add_argument("--step7-output", type=Path, required=True,
                    help="JSONL from step 7 (IterationResult per line)")
    ap.add_argument("--step2-output", type=Path, default=None,
                    help="JSONL from step 2 (optional; for category lookup)")
    ap.add_argument("--weight-tensor", type=Path, required=True,
                    help="path to the WeightTensor JSON; created if missing")
    ap.add_argument("--history", type=Path, default=None,
                    help="PredictionHistory JSONL; appended in-place. "
                         "When omitted, no history is loaded or saved.")
    ap.add_argument("--config", type=Path, required=True,
                    help="step8_config.yaml")
    ap.add_argument("--step2-config", type=Path, default=None,
                    help="step2_config.yaml (for LLMClient transport)")
    ap.add_argument("--sample-id", type=str, action="append", default=None,
                    help="restrict to this sample id (repeatable)")
    ap.add_argument("--output", type=Path, default=None,
                    help="output dir or single .jsonl file")
    ap.add_argument("--no-llm", action="store_true",
                    help="force-disable meta-correction even if config has "
                         "enable_meta_correction=true")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # ----- configs --------------------------------------------------------
    config = load_config(args.config)
    cfg_dir = args.config.parent
    step2_cfg_path = args.step2_config or (cfg_dir / "step2_config.yaml")
    step2_cfg = load_step2_config(step2_cfg_path) if step2_cfg_path.is_file() else {}

    wu_cfg = (config.get("weight_update") or {})
    learning_rate = float(wu_cfg.get("learning_rate", 0.1))
    enable_meta = bool(wu_cfg.get("enable_meta_correction", False))
    if args.no_llm:
        enable_meta = False  # CLI override

    # ----- client (only built if meta-correction enabled) -----------------
    client: Optional[LLMClient]
    if enable_meta:
        try:
            client = build_client(step2_cfg)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR building LLMClient: {e}", file=sys.stderr)
            return 1
    else:
        client = None

    # ----- load weight tensor + history ----------------------------------
    weight_tensor = WeightTensor.load(args.weight_tensor)
    history = (
        PredictionHistory.load(args.history)
        if args.history is not None else PredictionHistory()
    )

    # ----- load JSONL outputs --------------------------------------------
    s4_index = _index_by_sid(_load_jsonl(args.step4_output))
    s5_index = _index_by_sid(_load_jsonl(args.step5_output))
    s6_index = _index_by_sid(_load_jsonl(args.step6_output))
    s7_index = _index_by_sid(_load_jsonl(args.step7_output))
    s2_index = (
        _index_by_sid(_load_jsonl(args.step2_output))
        if args.step2_output is not None and args.step2_output.is_file()
        else {}
    )

    sample_ids = args.sample_id or sorted(s7_index.keys())
    if not sample_ids:
        print("ERROR: no sample ids resolved from --step7-output",
              file=sys.stderr)
        return 1

    # If --output is a single file, truncate once at start.
    write_to_one_file = (
        args.output is not None and args.output.suffix in (".jsonl", ".json")
    )
    one_file_path: Optional[Path] = None
    if write_to_one_file:
        one_file_path = args.output
        one_file_path.parent.mkdir(parents=True, exist_ok=True)
        one_file_path.write_text("", encoding="utf-8")

    # Pre-pull tool ids once — they don't change between samples.
    try:
        all_tool_ids = get_all_tool_ids()
    except Exception:  # pragma: no cover
        all_tool_ids = []

    n_done = 0
    n_skipped = 0
    n_with_meta = 0
    total_tokens = 0

    for sid in sample_ids:
        s4_record = s4_index.get(sid)
        s5_record = s5_index.get(sid)
        s6_record = s6_index.get(sid)
        s7_record = s7_index.get(sid)
        if not s4_record or not s5_record or not s6_record or not s7_record:
            missing = []
            if not s4_record: missing.append("step4")
            if not s5_record: missing.append("step5")
            if not s6_record: missing.append("step6")
            if not s7_record: missing.append("step7")
            print(f"[SKIP] {sid}: missing {','.join(missing)}",
                  file=sys.stderr)
            n_skipped += 1
            continue

        try:
            composite = _step5_record_to_composite(s5_record)
            predictions = _step4_record_to_predictions(s4_record)
            qa_result = _step6_record_to_qa(s6_record)
            iter_result = _step7_record_to_iter_result(s7_record)
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {sid}: record invalid ({e})", file=sys.stderr)
            n_skipped += 1
            continue

        category = _step6_category_for(sid, s2_index, s5_record) or "unknown"

        # ----- per-tool decomposition ------------------------------------
        per_tool_scores = compute_per_tool_scores(
            qa_result=qa_result,
            tool_predictions=predictions,
            composite_result=composite,
            config=wu_cfg,
        )
        updated_tool_ids = list(per_tool_scores.keys())

        # ----- snapshot before -------------------------------------------
        weights_before = slice_for_snapshot(
            weight_tensor, category, updated_tool_ids,
        )

        # ----- EMA pass --------------------------------------------------
        deltas = ema_update(
            weight_tensor=weight_tensor,
            category=category,
            per_tool_scores=per_tool_scores,
            learning_rate=learning_rate,
        )
        weights_after = slice_for_snapshot(
            weight_tensor, category, updated_tool_ids,
        )

        # ----- meta-correction (optional) --------------------------------
        meta = MetaCorrectionResult(status="disabled_by_config")
        meta_factors: Optional[dict[str, float]] = None
        if enable_meta and client is not None:
            meta = meta_correct(
                weight_tensor=weight_tensor,
                category=category,
                history=history.records,
                client=client,
                config=config,
            )
            if meta.factors:
                # Apply γ_k to the just-updated tools (and any other
                # tools the LLM names) — refresh weights_after snapshot.
                _ = apply_correction_factors(
                    weight_tensor=weight_tensor,
                    category=category,
                    factors=meta.factors,
                    tool_ids=updated_tool_ids,
                )
                weights_after = slice_for_snapshot(
                    weight_tensor, category, updated_tool_ids,
                )
                # Update deltas to reflect the new W (after - before).
                for t in updated_tool_ids:
                    for m in METRICS:
                        b = weights_before.get(t, {}).get(m)
                        a = weights_after.get(t, {}).get(m)
                        if b is None or a is None:
                            continue
                        d = a - b
                        if abs(d) > 1e-12 or t in deltas:
                            deltas.setdefault(t, {})[m] = d
                meta_factors = dict(meta.factors)
                n_with_meta += 1

        # ----- append history --------------------------------------------
        ts = datetime.now(timezone.utc).isoformat()
        history.add_record(_build_history_record(
            sid=sid, category=category,
            qa_record=qa_result.model_dump(mode="json"),
            iter_record=iter_result.model_dump(mode="json"),
            tool_predictions=predictions,
            ts=ts,
        ))

        # ----- build & validate result -----------------------------------
        try:
            result = WeightUpdateResult.model_validate({
                "sample_id": sid,
                "category": category,
                "tools_updated": updated_tool_ids,
                "ema_deltas": {t: dict(d) for t, d in deltas.items()},
                "learning_rate": learning_rate,
                "meta_correction_applied": bool(meta_factors),
                "correction_factors": meta_factors,
                "meta_rationale": meta.rationale if meta_factors else None,
                "weights_before": weights_before,
                "weights_after": weights_after,
                "api_usage": {**meta.api_usage, "status": meta.status},
                "timestamp": ts,
            })
        except Exception as e:  # noqa: BLE001 — validation should not happen, but defensive
            print(f"[SKIP] {sid}: WeightUpdateResult validation failed: {e}",
                  file=sys.stderr)
            n_skipped += 1
            continue

        record = result.model_dump(mode="json")
        print(_summarize(record))

        out_path = (one_file_path if write_to_one_file
                    else _resolve_output_path(args.output, sid))
        if out_path is not None:
            _write_record_atomic(out_path, record, append=write_to_one_file)

        n_done += 1
        # Sum tokens across the meta_correction api_usage.
        t = (meta.api_usage or {}).get("total_tokens")
        if isinstance(t, int):
            total_tokens += t

    # ----- persist tensor + history --------------------------------------
    _save_tensor_atomic(weight_tensor, args.weight_tensor)
    if args.history is not None:
        history.save(args.history)

    print()
    print(
        f"Processed: {n_done}  skipped: {n_skipped}  "
        f"meta_corrected: {n_with_meta}"
    )
    print(f"Total tokens: {total_tokens}")
    print(f"Wrote tensor: {args.weight_tensor}")
    if args.output:
        print(f"Wrote results: {args.output}")
    if args.history is not None:
        print(f"Wrote history: {args.history}")
    return 0 if n_skipped == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
