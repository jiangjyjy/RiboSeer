"""In-process batch runner for the full RiboSeer pipeline.

Reads a sample-id list (one per line) and runs steps 2 → 8 against each
sample, calling the per-step core functions DIRECTLY (no subprocess
fan-out — that would re-import every module 4500 times). On every
sample it writes a per-step JSONL under ``--output-dir/step{N}/`` plus
one summary line under ``summary/results.jsonl`` (or
``summary/failures.jsonl`` on crash).

Two operating modes
-------------------
1. **Train**:
       python scripts/batch_predict.py --sample-list train_500.txt ...
   Steps 2–8 run; W tensor accumulates updates across samples.

2. **Eval**:
       python scripts/batch_predict.py --sample-list test.txt --no-weight-update ...
   Steps 2–7 run; step 8 is skipped (W stays frozen). The same
   ``--weight-tensor`` is loaded read-only so step 3 sees the
   train-tuned weights.

Resume semantics
----------------
``--resume`` reads ``summary/results.jsonl`` AND
``summary/failures.jsonl`` once at startup, builds a set of sample ids
that already have a final outcome, and skips them. It does NOT inspect
per-step JSONL files: a sample that crashed mid-step 5 will retry from
step 2 next time (cheap because LLM caches don't apply, but
re-running the structure tools wastes hours — pre-stage their outputs
externally if you care).

Slicing for parallel batch runs
-------------------------------
``--start N --end M`` slices the sample list before processing. Use
this to parallelise across machines: e.g. on 4 hosts, slice
``[0:1133)``, ``[1133:2267)``, ``[2267:3400)``, ``[3400:4534)``. Each
host writes to the SAME output dir; ``--resume`` makes that idempotent
and the per-step JSONL writes are atomic so partial runs don't
corrupt anything.

Failure handling
----------------
Any uncaught exception in any step → that sample is logged to
``summary/failures.jsonl`` and the loop continues. The traceback is
captured (truncated to 2000 chars) so we can grep the JSONL afterward
for failure patterns without re-running.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---- step-2/3/5/7/8 LLM transport ---------------------------------------
from step2_target_char.llm_client import LLMClient  # noqa: E402
from step2_target_char.history import PredictionHistory  # noqa: E402
from step2_target_char.run import build_client, load_config as load_step2_config  # noqa: E402
from step2_target_char.target_char import characterize_target  # noqa: E402

# ---- step 3 -----
from step3_tool_selection.tool_selector import select_tools  # noqa: E402
from step3_tool_selection.weight_tensor import WeightTensor  # noqa: E402

# ---- step 4 -----
from step4_tool_adapters.run import run_sample as run_step4_sample  # noqa: E402
from step4_tool_adapters.run_all import (  # noqa: E402
    extract_planned_tools, filter_deployed,
)
from step4_tool_adapters.schemas import ToolPrediction, ToolPredictionSet  # noqa: E402

# ---- step 5 -----
from step5_fusion.fusion import build_output_record, fuse_predictions  # noqa: E402
from step5_fusion.lightgbm_fusion import (  # noqa: E402
    collect_samples, fullsystem_feature_names, fuse_sample,
    train_fullsystem_model,
)
from step5_fusion.prediction_io import (  # noqa: E402
    has_lightgbm_model, load_lightgbm_model, save_lightgbm_model,
)

# ---- step 6 -----
from step6_pocket_qa.scorer import score_prediction  # noqa: E402

# ---- step 7 -----
from step7_iteration.iterator import run_iteration_loop  # noqa: E402

# ---- step 8 -----
from step8_weight_update.ema_updater import (  # noqa: E402
    compute_per_tool_scores, ema_update, slice_for_snapshot,
)
from step8_weight_update.schemas import WeightUpdateResult  # noqa: E402


# ---------- IO -------------------------------------------------------------


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_sample(processed_dir: Path, sample_id: str) -> dict:
    """Read step-1 sample JSON. Looks first under ``samples/<id>.json``
    then directly under processed-dir (matches step 4's behaviour).

    Falls back to a case-insensitive scan of ``samples/`` when the exact
    match misses — splits.json sometimes carries an id whose case differs
    from the actual filename (e.g. ``3j46_y_1`` in splits vs
    ``3j46_Y_1.json`` on disk)."""
    for p in (processed_dir / "samples" / f"{sample_id}.json",
              processed_dir / f"{sample_id}.json"):
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))

    samples_dir = processed_dir / "samples"
    if samples_dir.is_dir():
        target = sample_id.lower()
        for f in samples_dir.iterdir():
            if f.suffix == ".json" and f.stem.lower() == target:
                return json.loads(f.read_text(encoding="utf-8"))

    raise FileNotFoundError(
        f"sample {sample_id!r} not found under {processed_dir}"
    )


def load_sample_list(path: Path) -> list[str]:
    """One sample id per line; blank / # lines skipped."""
    if not path.is_file():
        raise FileNotFoundError(f"sample list not found: {path}")
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def append_jsonl(path: Path, record: dict) -> None:
    """Append-mode write of one JSONL line. Creates parent dir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_step_record(
    output_dir: Path, step: str, sample_id: str, record: dict,
) -> Path:
    """Write a per-step JSONL file (one line per sample) under
    ``<output_dir>/<step>/<sample_id>.jsonl``. Atomic .tmp + replace.
    """
    out = Path(output_dir) / step / f"{sample_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(out)
    return out


def load_completed_ids(output_dir: Path) -> set[str]:
    """Resume: read existing summary JSONLs and return seen sample ids.

    Both ``results.jsonl`` (success) and ``failures.jsonl`` (terminal
    failure) count as "done" — re-running a known failure would just
    reproduce it, so skip it. To force a retry, delete the relevant
    line from failures.jsonl.
    """
    done: set[str] = set()
    for fname in ("results.jsonl", "failures.jsonl"):
        p = Path(output_dir) / "summary" / fname
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("sample_id")
            if sid:
                done.add(sid)
    return done


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------- step glue helpers ---------------------------------------------


def _make_predictions_set(
    sample_id: str, predictions: list[ToolPrediction], tools_run: list[str],
    runtime_s: float,
) -> dict:
    """Mirror what step 4's ``run.py`` would have written, in dict form."""
    return ToolPredictionSet(
        sample_id=sample_id,
        tools_run=tools_run,
        predictions=predictions,
        total_runtime_seconds=round(runtime_s, 3),
        timestamp=utc_iso(),
    ).model_dump(mode="json")


def _step4_record_to_predictions(record: dict) -> list[ToolPrediction]:
    """Re-hydrate the per-step JSONL we just wrote into ToolPrediction
    objects. Defensive: in-process we already have them but the helper
    works the same as step 6's CLI variant."""
    return [ToolPrediction.model_validate(p)
            for p in (record.get("predictions") or [])]


def _summary_row(
    sample_id: str,
    sample_json: dict,
    s2_record: dict,
    s4_record: dict,
    s5_record: dict,
    s6_record: dict,
    s7_record: dict,
    elapsed: float,
    total_tokens: int,
) -> dict:
    """Flatten the per-step records into one row for ``results.jsonl``.

    Field choices match the spec; missing data renders as None rather
    than crashes (e.g. samples without ground truth lack precision /
    recall / f1).
    """
    s2_out = s2_record.get("output") or {}
    s4_preds = s4_record.get("predictions") or []
    s4_tools_attempted = len(s4_preds)
    s4_tools_ok = sum(1 for p in s4_preds if p.get("success"))
    s4_tools_used = [p.get("tool_id") for p in s4_preds if p.get("success")]

    s5_status = (s5_record.get("api_usage") or {}).get("status")

    interaction = sample_json.get("interaction") or {}
    gt = interaction.get("binding_protein_residues") or []
    quality_tier = (
        (sample_json.get("data_availability") or {}).get("quality_tier")
        or sample_json.get("quality_tier")
    )

    s7_iters = s7_record.get("iterations") or []
    last_action = (
        (s7_iters[-1].get("action") or {}).get("action") if s7_iters else None
    )

    return {
        "sample_id": sample_id,
        "category": s2_out.get("category"),
        "quality_tier": quality_tier,
        "protein_length": (sample_json.get("protein") or {}).get("length"),
        "rna_length": (sample_json.get("rna") or {}).get("length"),
        "tools_attempted": s4_tools_attempted,
        "tools_succeeded": s4_tools_ok,
        "tools_used": s4_tools_used,
        "fusion_status": s5_status,
        "binding_protein_predicted":
            len(s5_record.get("binding_protein_residues") or []),
        "binding_protein_gt": len(gt),
        "precision": s5_record.get("precision"),
        "recall": s5_record.get("recall"),
        "f1": s5_record.get("f1"),
        "qa_total": s6_record.get("total_score"),
        "qa_final": s7_record.get("final_score"),
        "iteration_action": last_action,
        "total_iterations": s7_record.get("total_iterations"),
        "termination_reason": s7_record.get("termination_reason"),
        "runtime_seconds": round(elapsed, 3),
        "total_tokens": total_tokens,
        "timestamp": utc_iso(),
    }


def _sum_tokens(*records: dict) -> int:
    """Sum total_tokens across every per-step record's api_usage."""
    total = 0
    for r in records:
        usage = (r or {}).get("api_usage") or {}
        t = usage.get("total_tokens")
        if isinstance(t, int):
            total += t
    return total


# ---------- per-sample driver ---------------------------------------------


def resolve_tool_policy(
    tools: Optional[str] = None,
    *,
    skip_external: bool = False,
):
    """Build the filter applied to each sample's planned tool list.

    ``tools`` is ``None``/``"all"`` (run everything the plan asks for),
    ``"local"`` (drop the tools that need a manual web submission) or a
    comma-separated id list (run only those). ``skip_external`` drops the
    web-submission tools on top of whatever ``tools`` says.
    """
    from step3_tool_selection.tool_registry import (  # noqa: PLC0415
        WEB_SUBMISSION_TOOLS, get_all_tool_ids,
    )

    spec = (tools or "all").strip().lower()
    allowed: Optional[set[str]] = None
    if spec not in ("all", ""):
        if spec == "local":
            allowed = set(get_all_tool_ids()) - set(WEB_SUBMISSION_TOOLS)
        else:
            requested = {t.strip() for t in spec.split(",") if t.strip()}
            unknown = requested - set(get_all_tool_ids())
            if unknown:
                raise ValueError(
                    f"--tools names unknown tool ids: {sorted(unknown)}")
            allowed = requested
    elif skip_external:
        allowed = set(get_all_tool_ids()) - set(WEB_SUBMISSION_TOOLS)

    if allowed is None and not skip_external:
        return None

    def _policy(planned: list[str]) -> list[str]:
        kept = [t for t in planned if allowed is None or t in allowed]
        if skip_external:
            kept = [t for t in kept if t not in WEB_SUBMISSION_TOOLS]
        return kept

    return _policy


def resolve_fusion_model(args, output_dir: Path):
    """Return a fitted HARMONY model, or ``None`` for the prototype path.

    Order: an explicit ``--fusion-model-dir`` bundle; a bundle already in the
    output dir; otherwise — when training inputs were given — fit one and
    cache it. ``--fusion noisy-or`` skips all of this.
    """
    if args.fusion == "noisy-or":
        return None

    model_dir = args.fusion_model_dir or (output_dir / "harmony_model")
    if has_lightgbm_model(model_dir):
        print(f"harmony: loading LightGBM model from {model_dir}")
        return load_lightgbm_model(model_dir)

    if args.fusion_train_step4_dir and args.fusion_train_list:
        print("harmony: training the LightGBM fusion on "
              f"{args.fusion_train_list} …")
        train_ids = load_sample_list(args.fusion_train_list)
        train = collect_samples(args.fusion_train_step4_dir,
                                args.processed_dir, train_ids)
        if not train:
            raise RuntimeError(
                "harmony: no usable training samples "
                f"({args.fusion_train_step4_dir})")
        model = train_fullsystem_model(
            train, _load_json_dir(args.fusion_scope_dir),
            _load_json_dir(args.fusion_maestro_dir))
        try:
            save_lightgbm_model(model_dir, model,
                                fullsystem_feature_names())
            print(f"harmony: model cached in {model_dir}")
        except Exception as e:  # noqa: BLE001 - caching is best-effort
            print(f"harmony: could not cache the model ({e})")
        return model

    if args.fusion == "lightgbm":
        raise RuntimeError(
            "harmony: --fusion lightgbm needs --fusion-model-dir, or "
            "--fusion-train-step4-dir + --fusion-train-list")
    print("harmony: no LightGBM model or training inputs — using the "
          "prototype noisy-OR fusion")
    return None


def _load_json_dir(directory) -> dict:
    """``{sample_id: parsed_json}`` for the frozen SCOPE / MAESTRO outputs."""
    import json as _json  # noqa: PLC0415

    out: dict = {}
    if directory is None:
        return out
    path = Path(directory)
    if not path.is_dir():
        return out
    for f in sorted(path.glob("*.json")):
        try:
            out[f.stem] = _json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
    return out


def run_one_sample(
    sample_id: str,
    *,
    sample_json: dict,
    output_dir: Path,
    work_dir: Path,
    weight_tensor: WeightTensor,
    history: Optional[PredictionHistory],
    client: Optional[LLMClient],
    cfgs: dict[str, dict],
    no_weight_update: bool,
    tool_policy=None,
    fusion_model=None,
) -> dict:
    """Run steps 2–8 for one sample. Returns the summary row.

    ``tool_policy`` is an optional ``list[str] -> list[str]`` filter (see
    :func:`resolve_tool_policy`) applied to the step-3 plan before RELAY
    dispatches it.

    Raises on step-level failure so the outer loop can record it in
    ``failures.jsonl``. Per-step records are still written for whatever
    succeeded before the crash, which simplifies post-mortem.
    """
    # ---------- step 2 -----------------------------------------------------
    s2 = characterize_target(
        sample_json=sample_json, client=client, config=cfgs["step2"],
        history=history,
    )
    write_step_record(output_dir, "step2", sample_id, s2)
    target_char = s2.get("output") or {}
    target_features = s2.get("input_features") or {}
    category = target_char.get("category") or "novel_fold_x_unstructured"

    # ---------- step 3 -----------------------------------------------------
    s3 = select_tools(
        sample_id=sample_id, target_char=target_char,
        target_features=target_features, client=client,
        weight_tensor=weight_tensor, config=cfgs["step3"], history=history,
    )
    write_step_record(output_dir, "step3", sample_id, s3)

    planned = extract_planned_tools(s3)
    deployed, dropped = filter_deployed(planned)
    if tool_policy is not None:
        filtered = tool_policy(deployed)
        if filtered != deployed:
            skipped = sorted(set(deployed) - set(filtered))
            print(f"    tool policy dropped {skipped} for {sample_id}")
        deployed = filtered
    if not deployed:
        raise RuntimeError(
            f"step3 produced no runnable tools (planned={planned}, "
            f"dropped={dropped}). Cannot continue."
        )

    # ---------- step 4 -----------------------------------------------------
    t0 = time.monotonic()
    s4_pred_set = run_step4_sample(
        sample_json, deployed, cfgs["step4"], work_dir,
    )
    s4_runtime = time.monotonic() - t0
    s4_record = s4_pred_set.model_dump(mode="json")
    write_step_record(output_dir, "step4", sample_id, s4_record)
    if not any(p.success for p in s4_pred_set.predictions):
        raise RuntimeError(
            f"step4: all {len(s4_pred_set.predictions)} tool runs failed; "
            "fusion has no signal to work with."
        )

    # ---------- step 5 -----------------------------------------------------
    if fusion_model is not None:
        # HARMONY (the model reported in the paper): LightGBM over the 15-tool
        # feature space, conditioned on this sample's SCOPE profile and
        # MAESTRO selection — both already in hand from steps 2 and 3.
        composite = fuse_sample(
            fusion_model,
            sample_id=sample_id, sample_json=sample_json,
            step4_record=s4_record,
            scope_profiles={sample_id: target_char},
            selections={sample_id: s3.get("tool_plan") or {}},
            polish_actions=None,
        )
    else:
        composite = fuse_predictions(
            sample_json=sample_json, target_char=target_char,
            tool_predictions=s4_pred_set.predictions, client=client,
            weight_tensor=weight_tensor, config=cfgs["step5"], history=history,
        )
    s5 = build_output_record(composite, sample_json, target_char)
    write_step_record(output_dir, "step5", sample_id, s5)

    # ---------- step 6 -----------------------------------------------------
    qa = score_prediction(
        sample_json=sample_json, composite_result=composite,
        tool_predictions=list(s4_pred_set.predictions), config=cfgs["step6"],
    )
    s6 = qa.model_dump(mode="json")
    write_step_record(output_dir, "step6", sample_id, s6)

    # ---------- step 7 -----------------------------------------------------
    iter_result = run_iteration_loop(
        sample_json=sample_json, target_char=target_char,
        tool_predictions=list(s4_pred_set.predictions),
        composite_result=composite, qa_result=qa,
        client=client, config=cfgs["step7"],
        fusion_config=cfgs["step5"], qa_config=cfgs["step6"],
    )
    s7 = iter_result.model_dump(mode="json")
    write_step_record(output_dir, "step7", sample_id, s7)

    # ---------- step 8 (train only) ---------------------------------------
    s8: dict = {}
    if not no_weight_update:
        wu_cfg = (cfgs["step8"].get("weight_update") or {})
        learning_rate = float(wu_cfg.get("learning_rate", 0.1))

        per_tool_scores = compute_per_tool_scores(
            qa_result=qa, tool_predictions=list(s4_pred_set.predictions),
            composite_result=composite, config=wu_cfg,
        )
        updated_ids = list(per_tool_scores.keys())
        snap_before = slice_for_snapshot(
            weight_tensor, category, updated_ids,
        )
        deltas = ema_update(
            weight_tensor=weight_tensor, category=category,
            per_tool_scores=per_tool_scores, learning_rate=learning_rate,
        )
        snap_after = slice_for_snapshot(
            weight_tensor, category, updated_ids,
        )
        s8 = WeightUpdateResult.model_validate({
            "sample_id": sample_id, "category": category,
            "tools_updated": updated_ids,
            "ema_deltas": {t: dict(d) for t, d in deltas.items()},
            "learning_rate": learning_rate,
            "meta_correction_applied": False,
            "correction_factors": None, "meta_rationale": None,
            "weights_before": snap_before, "weights_after": snap_after,
            "api_usage": {"status": "ema_only"},
            "timestamp": utc_iso(),
        }).model_dump(mode="json")
        write_step_record(output_dir, "step8", sample_id, s8)

        # Append to in-process history so the next sample's step 2/3/5/7
        # prompts can see this run's outcome.
        if history is not None:
            history.add_record({
                "sample_id": sample_id, "category": category,
                "timestamp": utc_iso(),
                "scores": {
                    "structural_plausibility": qa.structural_plausibility,
                    "physicochemical_complementarity": qa.physicochemical_complementarity,
                    "evolutionary_conservation": qa.evolutionary_conservation,
                    "cross_tool_consensus": qa.cross_tool_consensus,
                    "known_motif_consistency": qa.known_motif_consistency,
                    "total": qa.total_score,
                },
                "tools_used": [
                    p.tool_id for p in s4_pred_set.predictions if p.success
                ],
                "final_action": iter_result.final_action,
                "final_score": iter_result.final_score,
            })

    # ---------- summary row -----------------------------------------------
    total_tokens = _sum_tokens(s2, s3, s5, s7, s8)
    elapsed = (
        s2.get("api_usage", {}).get("total_tokens", 0) and 0
    )  # unused — real value computed by caller
    return _summary_row(
        sample_id=sample_id, sample_json=sample_json,
        s2_record=s2, s4_record=s4_record, s5_record=s5,
        s6_record=s6, s7_record=s7,
        elapsed=elapsed,  # caller will overwrite
        total_tokens=total_tokens,
    )


# ---------- main -----------------------------------------------------------


def _build_cfgs(config_dir: Path) -> dict[str, dict]:
    """Load every per-step yaml. Missing files → empty dict (the
    inner functions are tolerant of empty configs)."""
    return {
        "step2": load_yaml(config_dir / "step2_config.yaml"),
        "step3": load_yaml(config_dir / "step3_config.yaml"),
        "step4": load_yaml(config_dir / "step4_config.yaml"),
        "step5": load_yaml(config_dir / "step5_config.yaml"),
        "step6": load_yaml(config_dir / "step6_config.yaml"),
        "step7": load_yaml(config_dir / "step7_config.yaml"),
        "step8": load_yaml(config_dir / "step8_config.yaml"),
    }


def main(argv: Optional[list[str]] = None,
         description: Optional[str] = None,
         prog: Optional[str] = None) -> int:
    p = argparse.ArgumentParser(
        prog=prog, description=description or __doc__)
    p.add_argument("--processed-dir", type=Path, required=True,
                   help="step 1 output dir containing samples/<id>.json")
    p.add_argument("--sample-list", type=Path, required=True,
                   help="text file, one sample id per line")
    p.add_argument("--config-dir", type=Path, default=REPO_ROOT / "configs",
                   help="directory containing step*_config.yaml")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="root for per-step JSONLs + summary + weights")
    p.add_argument("--weight-tensor", type=Path, required=True,
                   help="W tensor JSON path (created if missing)")
    p.add_argument("--history", type=Path, default=None,
                   help="PredictionHistory JSONL; appended in-place. "
                        "Default: <output-dir>/history/history.jsonl")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"),
                   help="raw PDB dir (passed through to step 4 adapters)")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="step 4 staging dir (default: <output-dir>/work)")
    p.add_argument("--resume", action="store_true",
                   help="skip samples already in summary/results.jsonl "
                        "or summary/failures.jsonl")
    p.add_argument("--no-weight-update", action="store_true",
                   help="skip step 8 (eval mode — W stays frozen)")
    p.add_argument("--no-llm", action="store_true",
                   help="skip every LLM call; steps 2/3/5/7 fall back to "
                        "their built-in offline path")
    p.add_argument("--start", type=int, default=None,
                   help="slice the sample list at this index (inclusive)")
    p.add_argument("--end", type=int, default=None,
                   help="slice the sample list at this index (exclusive)")
    p.add_argument("--tools", default="all",
                   help="which tools RELAY may run: 'all' (default), 'local' "
                        "(everything except the manual web-submission tools), "
                        "or a comma-separated id list")
    p.add_argument("--fusion", default="auto",
                   choices=["auto", "lightgbm", "noisy-or"],
                   help="HARMONY fusion strategy: 'auto' uses the LightGBM "
                        "model when one is available and falls back to the "
                        "prototype noisy-OR otherwise; 'lightgbm' requires it")
    p.add_argument("--fusion-model-dir", type=Path, default=None,
                   help="saved LightGBM bundle (default: <output-dir>/harmony_model)")
    p.add_argument("--fusion-train-step4-dir", type=Path, default=None,
                   help="step-4 dir of the training split; with "
                        "--fusion-train-list this fits the fusion model")
    p.add_argument("--fusion-train-list", type=Path, default=None,
                   help="sample list for the training split")
    p.add_argument("--fusion-scope-dir", type=Path, default=None,
                   help="frozen SCOPE profiles for the training split")
    p.add_argument("--fusion-maestro-dir", type=Path, default=None,
                   help="frozen MAESTRO selections for the training split")
    p.add_argument("--skip-external", action="store_true",
                   help="drop the tools whose results must be submitted by "
                        "hand through a web server (they are then zero-filled "
                        "by HARMONY, as any failed tool would be)")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # ---- configs ---------------------------------------------------------
    cfgs = _build_cfgs(args.config_dir)
    try:
        tool_policy = resolve_tool_policy(args.tools,
                                          skip_external=args.skip_external)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    try:
        fusion_model = resolve_fusion_model(args, args.output_dir)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR resolving the fusion model: {e}", file=sys.stderr)
        return 2
    work_dir = args.work_dir or (args.output_dir / "work")
    history_path = args.history or (args.output_dir / "history" / "history.jsonl")

    # ---- client ----------------------------------------------------------
    client: Optional[LLMClient]
    if args.no_llm:
        client = None
    else:
        if not os.environ.get("LLM_API_KEY"):
            print("ERROR: LLM_API_KEY env var not set; pass --no-llm for "
                  "an offline (LLM-disabled) run.", file=sys.stderr)
            return 1
        try:
            client = build_client(cfgs["step2"])
        except Exception as e:  # noqa: BLE001
            print(f"ERROR building LLMClient: {e}", file=sys.stderr)
            return 1

    # ---- weight tensor + history ----------------------------------------
    weight_tensor = WeightTensor.load(args.weight_tensor)
    history = (
        PredictionHistory.load(history_path)
        if args.history is not None or history_path.is_file()
        else PredictionHistory()
    )

    # ---- sample list -----------------------------------------------------
    try:
        sample_ids = load_sample_list(args.sample_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if args.start is not None or args.end is not None:
        start = args.start or 0
        end = args.end if args.end is not None else len(sample_ids)
        sample_ids = sample_ids[start:end]

    done: set[str] = set()
    if args.resume:
        done = load_completed_ids(args.output_dir)
        print(f"resume: {len(done)} samples already done — skipping")

    if not sample_ids:
        print("ERROR: no samples to run", file=sys.stderr)
        return 1

    # ---- progress bar ----------------------------------------------------
    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:
        def tqdm(it, **kwargs):  # type: ignore
            return it

    summary_path = args.output_dir / "summary" / "results.jsonl"
    failures_path = args.output_dir / "summary" / "failures.jsonl"

    n_ok = n_fail = n_skip = 0
    for sid in tqdm(sample_ids, desc="samples"):
        if sid in done:
            n_skip += 1
            continue
        t0 = time.monotonic()
        try:
            sample_json = load_sample(args.processed_dir, sid)
        except FileNotFoundError as e:
            n_fail += 1
            append_jsonl(failures_path, {
                "sample_id": sid,
                "error": f"sample json missing: {e}",
                "phase": "load_sample",
                "elapsed_seconds": round(time.monotonic() - t0, 3),
                "timestamp": utc_iso(),
            })
            continue

        try:
            row = run_one_sample(
                sid, sample_json=sample_json,
                output_dir=args.output_dir, work_dir=work_dir,
                weight_tensor=weight_tensor, history=history,
                client=client, cfgs=cfgs,
                no_weight_update=args.no_weight_update,
                tool_policy=tool_policy,
                fusion_model=fusion_model,
            )
            row["runtime_seconds"] = round(time.monotonic() - t0, 3)
            append_jsonl(summary_path, row)
            n_ok += 1
        except Exception as e:  # noqa: BLE001 — never let one bad sample kill the batch
            tb = traceback.format_exc(limit=8)
            if len(tb) > 2000:
                tb = tb[:2000] + " ... (truncated)"
            append_jsonl(failures_path, {
                "sample_id": sid,
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb,
                "elapsed_seconds": round(time.monotonic() - t0, 3),
                "timestamp": utc_iso(),
            })
            n_fail += 1

        # Persist the W tensor on every successful train sample so a
        # crash mid-batch doesn't lose all training. Cheap (~few KB).
        if not args.no_weight_update:
            try:
                weight_tensor.save(args.weight_tensor)
            except Exception as e:  # noqa: BLE001
                print(f"WARN: failed to save weight tensor: {e}",
                      file=sys.stderr)

    # ---- final history persistence --------------------------------------
    try:
        history.save(history_path)
    except Exception as e:  # noqa: BLE001
        print(f"WARN: failed to save history: {e}", file=sys.stderr)

    print()
    print(f"Done: ok={n_ok}  fail={n_fail}  skip={n_skip}  "
          f"total={n_ok + n_fail + n_skip}")
    print(f"summary: {summary_path}")
    if n_fail:
        print(f"failures: {failures_path}")
    print(f"weight tensor: {args.weight_tensor}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
