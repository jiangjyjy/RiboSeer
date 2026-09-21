"""Rerun a single step-4 tool over an existing batch, then refresh step 5–8.

Motivation
----------
A full ``batch_predict.py`` run on the train split costs hours; if only one
tool failed (e.g. HADDOCK 3 with a broken conda env path) we want to patch
just that tool's per-sample record in the existing step 4 JSONLs, then
re-run the downstream fusion / QA / iteration / weight-update on top of
the now-complete tool set.

What this script does
---------------------
For each sample id in ``--sample-list``:

  1. Load ``<output-dir>/step4/<sample_id>.jsonl`` (one ``ToolPredictionSet``
     line). If it doesn't exist, skip — we only patch existing batches.
  2. Inspect the record for an existing entry on ``--tool``:
       * present + ``success=True`` AND ``--resume`` → skip (nothing to do)
       * present + ``success=False`` → drop the failed entry, run the
         adapter, splice the result in its place
       * absent → run the adapter and append a new entry
  3. Re-write the JSONL line atomically (.tmp → replace).
  4. Re-run steps 5 → 8 (or 5 → 7 if ``--no-weight-update``) and overwrite
     the corresponding per-step JSONL files. Step 2 + step 3 records are
     re-read from disk and used as inputs — they don't change just because
     we added a tool.
  5. Append a summary row to ``<output-dir>/summary/rerun_results.jsonl``
     so the run is auditable. Failures go to ``rerun_failures.jsonl``.

The W tensor is updated in-place on every successful train sample (same
discipline as ``batch_predict.py``); pass ``--no-weight-update`` to keep
it frozen.

CNS_SOLVE resolution
--------------------
HADDOCK 3 needs ``CNS_SOLVE`` in its environment. Resolution order:

  1. ``--cns-solve`` CLI flag (wins everything)
  2. ``configs/step4_config.yaml`` → ``tools.haddock3.cns_solve``
  3. ``os.environ["CNS_SOLVE"]``

If none of the three is set, the adapter still runs (the haddock3 conda
env may already export it); we just don't inject anything.

Usage
-----
::

    python scripts/rerun_single_tool.py \\
        --tool haddock3 \\
        --processed-dir data/processed_filtered \\
        --sample-list data/processed_filtered/splits/train_200.txt \\
        --config-dir configs/ \\
        --output-dir data/batch_train_v4/ \\
        --weight-tensor data/batch_train_v4/W.json \\
        --raw-dir data/raw \\
        --resume
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
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---- step transports (LLM is optional, same as batch_predict) ----
from step2_target_char.llm_client import LLMClient  # noqa: E402
from step2_target_char.history import PredictionHistory  # noqa: E402
from step2_target_char.run import build_client  # noqa: E402

from step3_tool_selection.weight_tensor import WeightTensor  # noqa: E402
from step3_tool_selection.tool_registry import get_tool  # noqa: E402

from step4_tool_adapters.run import get_adapter  # noqa: E402
from step4_tool_adapters.schemas import (  # noqa: E402
    ToolPrediction, ToolPredictionSet, make_failure_prediction,
)

from step5_fusion.fusion import build_output_record, fuse_predictions  # noqa: E402

from step6_pocket_qa.scorer import score_prediction  # noqa: E402

from step7_iteration.iterator import run_iteration_loop  # noqa: E402

from step8_weight_update.ema_updater import (  # noqa: E402
    compute_per_tool_scores, ema_update, slice_for_snapshot,
)
from step8_weight_update.schemas import WeightUpdateResult  # noqa: E402


# ---------- IO helpers ------------------------------------------------------


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_sample(processed_dir: Path, sample_id: str) -> dict:
    """Same lookup as batch_predict.load_sample (case-insensitive fallback)."""
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
    if not path.is_file():
        raise FileNotFoundError(f"sample list not found: {path}")
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def read_single_jsonl(path: Path) -> Optional[dict]:
    """Per-step records are written as one JSONL line per sample. Some
    files have a trailing newline, some don't — strip blanks defensively
    and take the first non-blank line."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        return json.loads(line)
    return None


def write_step_record(
    output_dir: Path, step: str, sample_id: str, record: dict,
) -> Path:
    out = Path(output_dir) / step / f"{sample_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(out)
    return out


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------- CNS_SOLVE plumbing ---------------------------------------------


def inject_cns_solve(cfgs: dict, override: Optional[str]) -> Optional[str]:
    """Resolve a CNS_SOLVE path and stamp it into the haddock3 config block.

    Returns the resolved path (or None if nothing was found). The mutated
    ``cfgs['step4']['tools']['haddock3']['cns_solve']`` is what the
    adapter actually reads.
    """
    step4_cfg = cfgs.setdefault("step4", {})
    tools = step4_cfg.setdefault("tools", {})
    h3 = tools.setdefault("haddock3", {})

    resolved = override or h3.get("cns_solve") or os.environ.get("CNS_SOLVE")
    if resolved:
        h3["cns_solve"] = str(resolved)
    return resolved


# ---------- tool-id / category resolution ----------------------------------

# step4 records use the registry id ``rosettafold2na``; the spec / CLI
# may pass ``rf2na``. Canonicalise before adapter + registry lookup.
_TOOL_ID_ALIASES = {"rf2na": "rosettafold2na"}


def _canonical_tool_id(tool_id: str) -> str:
    return _TOOL_ID_ALIASES.get(tool_id, tool_id)


def _registry_category(tool_id: str, default: str = "D") -> str:
    """Authoritative category from the tool registry (e.g.
    rosettafold2na → 'A'). Falls back to ``default`` only for ids the
    registry doesn't know — never hard-code 'D' for a known tool."""
    try:
        return get_tool(_canonical_tool_id(tool_id)).category
    except KeyError:
        return default


# ---------- step 4 patch ---------------------------------------------------


def _strip_tool_entries(record: dict, tool_id: str) -> dict:
    """Drop every prediction whose ``tool_id`` matches. Returns a NEW dict
    so we never mutate the input in place."""
    out = dict(record)
    preds = out.get("predictions") or []
    out["predictions"] = [p for p in preds if p.get("tool_id") != tool_id]

    tools_run = out.get("tools_run") or []
    out["tools_run"] = [t for t in tools_run if t != tool_id]
    return out


def _existing_tool_entry(record: dict, tool_id: str) -> Optional[dict]:
    """Find the (assumed-unique) prediction for ``tool_id`` in a step 4
    record. None if not present."""
    for p in record.get("predictions") or []:
        if p.get("tool_id") == tool_id:
            return p
    return None


def run_single_tool(
    tool_id: str,
    sample_json: dict,
    work_dir: Path,
    config: dict,
) -> ToolPrediction:
    """Run one adapter, isolated from the parallel run_sample path.

    BaseAdapter.predict already swallows every exception, so this never
    raises — a crash is folded into ``success=False`` and we still write
    that to step 4 (matching the original batch's failure semantics).
    """
    sample_id = sample_json.get("sample_id", "<unknown>")
    sample_work = Path(work_dir) / sample_id
    sample_work.mkdir(parents=True, exist_ok=True)
    try:
        adapter = get_adapter(_canonical_tool_id(tool_id))
    except ValueError as e:
        # Category MUST come from the registry, not a hard-coded 'D'
        # — RF2NA (rosettafold2na) is a Cat A structure predictor;
        # mislabelling it 'D' poisons fusion / eval which key off
        # category.
        return make_failure_prediction(
            tool_id=tool_id,
            category=_registry_category(tool_id),
            sample_id=sample_id,
            error_message=f"adapter lookup failed: {e}",
        )
    return adapter.predict(sample_json, sample_work, config)


def patch_step4_record(
    sample_id: str,
    tool_id: str,
    record: dict,
    new_pred: ToolPrediction,
    runtime_delta: float,
) -> dict:
    """Splice ``new_pred`` into the step 4 record:
      - replace any existing entry for ``tool_id`` (success or failure)
      - keep every other tool's prediction untouched
      - append ``tool_id`` to ``tools_run`` if not already there
      - bump ``total_runtime_seconds`` by the wallclock spent on the rerun
    """
    base = _strip_tool_entries(record, tool_id)
    preds = list(base["predictions"])
    preds.append(new_pred.model_dump(mode="json"))

    tools_run = list(base["tools_run"])
    if tool_id not in tools_run:
        tools_run.append(tool_id)

    prior_runtime = float(record.get("total_runtime_seconds") or 0.0)

    out = ToolPredictionSet(
        sample_id=sample_id,
        tools_run=tools_run,
        predictions=[ToolPrediction.model_validate(p) for p in preds],
        total_runtime_seconds=round(prior_runtime + runtime_delta, 3),
        timestamp=utc_iso(),
    )
    return out.model_dump(mode="json")


# ---------- step 5–8 rerun --------------------------------------------------


def _hydrate_predictions(record: dict) -> list[ToolPrediction]:
    return [ToolPrediction.model_validate(p)
            for p in (record.get("predictions") or [])]


def rerun_downstream(
    sample_id: str,
    *,
    sample_json: dict,
    step2_record: Optional[dict],
    step4_record: dict,
    output_dir: Path,
    weight_tensor: WeightTensor,
    weight_tensor_path: Path,
    history: PredictionHistory,
    client: Optional[LLMClient],
    cfgs: dict[str, dict],
    no_weight_update: bool,
) -> dict:
    """Run steps 5 → 8 with the patched step 4. Writes per-step JSONLs
    and returns a small summary row (mirrors batch_predict._summary_row
    but trimmed — step 2/3 we don't touch)."""
    predictions = _hydrate_predictions(step4_record)
    if not any(p.success for p in predictions):
        # Step 5 itself returns an empty result here; we still continue so
        # the downstream files exist and the audit row is honest.
        pass

    target_char = (step2_record or {}).get("output") or {}
    category = target_char.get("category") or "novel_fold_x_unstructured"

    # ---- step 5 ----
    composite = fuse_predictions(
        sample_json=sample_json, target_char=target_char,
        tool_predictions=predictions, client=client,
        weight_tensor=weight_tensor, config=cfgs["step5"], history=history,
    )
    s5 = build_output_record(composite, sample_json, target_char)
    write_step_record(output_dir, "step5", sample_id, s5)

    # ---- step 6 ----
    qa = score_prediction(
        sample_json=sample_json, composite_result=composite,
        tool_predictions=predictions, config=cfgs["step6"],
    )
    s6 = qa.model_dump(mode="json")
    write_step_record(output_dir, "step6", sample_id, s6)

    # ---- step 7 ----
    iter_result = run_iteration_loop(
        sample_json=sample_json, target_char=target_char,
        tool_predictions=predictions,
        composite_result=composite, qa_result=qa,
        client=client, config=cfgs["step7"],
        fusion_config=cfgs["step5"], qa_config=cfgs["step6"],
    )
    s7 = iter_result.model_dump(mode="json")
    write_step_record(output_dir, "step7", sample_id, s7)

    # ---- step 8 (train only) ----
    s8: dict = {}
    if not no_weight_update:
        wu_cfg = (cfgs["step8"].get("weight_update") or {})
        learning_rate = float(wu_cfg.get("learning_rate", 0.1))

        per_tool_scores = compute_per_tool_scores(
            qa_result=qa, tool_predictions=predictions,
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

        # Persist W after every sample — cheap and survives a crash.
        try:
            weight_tensor.save(weight_tensor_path)
        except Exception as e:  # noqa: BLE001
            print(f"WARN: failed to save weight tensor: {e}", file=sys.stderr)

    return {
        "sample_id": sample_id,
        "category": category,
        "tools_in_step4": [p.tool_id for p in predictions],
        "tools_succeeded": [p.tool_id for p in predictions if p.success],
        "fusion_status": (s5.get("api_usage") or {}).get("status"),
        "qa_total": s6.get("total_score"),
        "qa_final": s7.get("final_score"),
        "step8_run": not no_weight_update,
    }


# ---------- per-sample driver ----------------------------------------------


def _decide_action(
    record: dict, tool_id: str, resume: bool,
) -> tuple[str, Optional[dict]]:
    """Return ``(action, existing_entry)`` where action is one of:
      - ``skip``    : success already on disk and --resume
      - ``replace`` : existing entry (success or failure) needs to be
                      swapped for a fresh run
      - ``append``  : no entry for this tool yet
    """
    existing = _existing_tool_entry(record, tool_id)
    if existing is None:
        return "append", None
    if resume and existing.get("success") is True:
        return "skip", existing
    return "replace", existing


def process_sample(
    sample_id: str,
    *,
    tool_id: str,
    processed_dir: Path,
    output_dir: Path,
    work_dir: Path,
    weight_tensor: WeightTensor,
    weight_tensor_path: Path,
    history: PredictionHistory,
    client: Optional[LLMClient],
    cfgs: dict[str, dict],
    resume: bool,
    no_weight_update: bool,
) -> dict:
    """Patch one sample's step 4, then rerun steps 5–8. Returns the
    summary row."""
    s4_path = output_dir / "step4" / f"{sample_id}.jsonl"
    s4_record = read_single_jsonl(s4_path)
    if s4_record is None:
        raise FileNotFoundError(
            f"step4 record missing for {sample_id!r} at {s4_path}; "
            f"this script only patches existing batches — run "
            f"batch_predict.py first."
        )

    action, existing = _decide_action(s4_record, tool_id, resume)

    if action == "skip":
        return {
            "sample_id": sample_id,
            "tool": tool_id,
            "action": "skip_existing_success",
            "step4_patched": False,
            "downstream_rerun": False,
            "timestamp": utc_iso(),
        }

    # Need step 2 for category. The sample json is needed by both step 4
    # adapters and downstream.
    sample_json = load_sample(processed_dir, sample_id)
    step2_record = read_single_jsonl(output_dir / "step2" / f"{sample_id}.jsonl")

    # ---- run the tool ----
    t0 = time.monotonic()
    new_pred = run_single_tool(tool_id, sample_json, work_dir, cfgs["step4"])
    runtime = time.monotonic() - t0

    patched = patch_step4_record(
        sample_id, tool_id, s4_record, new_pred, runtime,
    )
    write_step_record(output_dir, "step4", sample_id, patched)

    # ---- step 5–8 ----
    downstream = rerun_downstream(
        sample_id,
        sample_json=sample_json, step2_record=step2_record,
        step4_record=patched, output_dir=output_dir,
        weight_tensor=weight_tensor,
        weight_tensor_path=weight_tensor_path,
        history=history, client=client, cfgs=cfgs,
        no_weight_update=no_weight_update,
    )

    return {
        "sample_id": sample_id,
        "tool": tool_id,
        "action": action,                   # "append" or "replace"
        "tool_success": bool(new_pred.success),
        "tool_error": new_pred.error_message,
        "tool_runtime_seconds": round(runtime, 3),
        "step4_patched": True,
        "downstream_rerun": True,
        **downstream,
        "timestamp": utc_iso(),
    }


# ---------- argparse main --------------------------------------------------


def _build_cfgs(config_dir: Path) -> dict[str, dict]:
    return {
        "step2": load_yaml(config_dir / "step2_config.yaml"),
        "step3": load_yaml(config_dir / "step3_config.yaml"),
        "step4": load_yaml(config_dir / "step4_config.yaml"),
        "step5": load_yaml(config_dir / "step5_config.yaml"),
        "step6": load_yaml(config_dir / "step6_config.yaml"),
        "step7": load_yaml(config_dir / "step7_config.yaml"),
        "step8": load_yaml(config_dir / "step8_config.yaml"),
    }


def _inject_raw_dir(cfgs: dict, raw_dir: Optional[Path]) -> None:
    """Adapter reads ``step4.structure_source.raw_dir`` to locate raw
    PDBs. Allow --raw-dir to override the config so the CLI works
    standalone."""
    if raw_dir is None:
        return
    step4 = cfgs.setdefault("step4", {})
    ss = step4.setdefault("structure_source", {})
    ss["raw_dir"] = str(raw_dir)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tool", type=str, required=True,
                   help="single tool_id to rerun (e.g. haddock3)")
    p.add_argument("--processed-dir", type=Path, required=True,
                   help="step 1 output dir containing samples/<id>.json")
    p.add_argument("--sample-list", type=Path, required=True,
                   help="text file, one sample id per line")
    p.add_argument("--config-dir", type=Path, default=REPO_ROOT / "configs",
                   help="directory containing step*_config.yaml")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="root holding existing step{2..8}/ JSONLs")
    p.add_argument("--weight-tensor", type=Path, required=True,
                   help="W tensor JSON path (read + written when "
                        "--no-weight-update is absent)")
    p.add_argument("--history", type=Path, default=None,
                   help="PredictionHistory JSONL (default: "
                        "<output-dir>/history/history.jsonl)")
    p.add_argument("--raw-dir", type=Path, default=None,
                   help="raw PDB dir; overrides "
                        "step4.structure_source.raw_dir")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="step 4 staging dir (default: <output-dir>/work)")
    p.add_argument("--cns-solve", type=str, default=None,
                   help="CNS_SOLVE path; overrides config and env var")
    p.add_argument("--resume", action="store_true",
                   help="skip samples whose step 4 already has a "
                        "success=True entry for --tool")
    p.add_argument("--no-weight-update", action="store_true",
                   help="skip step 8 (eval mode — W stays frozen)")
    p.add_argument("--no-llm", action="store_true",
                   help="skip every LLM call; step 5/7 fall back to "
                        "their built-in offline path")
    p.add_argument("--start", type=int, default=None,
                   help="slice the sample list at this index (inclusive)")
    p.add_argument("--end", type=int, default=None,
                   help="slice the sample list at this index (exclusive)")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # ---- configs ---------------------------------------------------------
    cfgs = _build_cfgs(args.config_dir)
    _inject_raw_dir(cfgs, args.raw_dir)
    resolved_cns = inject_cns_solve(cfgs, args.cns_solve)
    if args.tool == "haddock3":
        if resolved_cns:
            print(f"CNS_SOLVE -> {resolved_cns}")
        else:
            print("WARN: no CNS_SOLVE resolved (CLI/config/env); "
                  "relying on haddock3 conda env defaults",
                  file=sys.stderr)

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

    # ---- W + history ----------------------------------------------------
    weight_tensor = WeightTensor.load(args.weight_tensor)
    if args.history is not None or history_path.is_file():
        history = PredictionHistory.load(history_path)
    else:
        history = PredictionHistory()

    # ---- sample list ----------------------------------------------------
    try:
        sample_ids = load_sample_list(args.sample_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if args.start is not None or args.end is not None:
        start = args.start or 0
        end = args.end if args.end is not None else len(sample_ids)
        sample_ids = sample_ids[start:end]

    if not sample_ids:
        print("ERROR: no samples to run", file=sys.stderr)
        return 1

    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:
        def tqdm(it, **kwargs):  # type: ignore
            return it

    summary_path = args.output_dir / "summary" / "rerun_results.jsonl"
    failures_path = args.output_dir / "summary" / "rerun_failures.jsonl"

    n_ok = n_fail = n_skip = 0
    for sid in tqdm(sample_ids, desc=f"rerun {args.tool}"):
        t0 = time.monotonic()
        try:
            row = process_sample(
                sid, tool_id=args.tool,
                processed_dir=args.processed_dir,
                output_dir=args.output_dir,
                work_dir=work_dir,
                weight_tensor=weight_tensor,
                weight_tensor_path=args.weight_tensor,
                history=history, client=client, cfgs=cfgs,
                resume=args.resume,
                no_weight_update=args.no_weight_update,
            )
            row["elapsed_seconds"] = round(time.monotonic() - t0, 3)
            if row.get("action") == "skip_existing_success":
                n_skip += 1
            else:
                n_ok += 1
            append_jsonl(summary_path, row)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc(limit=8)
            if len(tb) > 2000:
                tb = tb[:2000] + " ... (truncated)"
            append_jsonl(failures_path, {
                "sample_id": sid,
                "tool": args.tool,
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb,
                "elapsed_seconds": round(time.monotonic() - t0, 3),
                "timestamp": utc_iso(),
            })
            n_fail += 1

    # ---- final history persistence --------------------------------------
    try:
        history.save(history_path)
    except Exception as e:  # noqa: BLE001
        print(f"WARN: failed to save history: {e}", file=sys.stderr)

    print()
    print(f"Done: rerun={n_ok}  skip={n_skip}  fail={n_fail}  "
          f"total={n_ok + n_skip + n_fail}")
    print(f"summary: {summary_path}")
    if n_fail:
        print(f"failures: {failures_path}")
    if not args.no_weight_update:
        print(f"weight tensor: {args.weight_tensor}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
