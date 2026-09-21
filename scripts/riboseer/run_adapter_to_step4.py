#!/usr/bin/env python3
"""Run ONE step-4 tool adapter over a sample list and merge it into step4.

Why this exists
---------------
``rerun_single_tool.py`` patches a tool into step4 **and then re-runs steps
5–8 + updates the W tensor** (needs ``--weight-tensor`` and, unless
``--no-llm``, a LLM key). When all you want is to (re)generate one tool's
per-sample predictions and drop them into the step4 JSONLs — e.g. backfill
``fpocket`` / ``rfaa`` on the train split so the Table 9 / Table 10 feature
matrices have them — that whole downstream rerun is wasted work.

This script does only the minimal thing:

  for each sample id:
    1. load the step-1 sample JSON
    2. run ``get_adapter(tool).predict(sample_json, work_dir, step4_cfg)``
       (BaseAdapter.predict never raises — a crash becomes success=False)
    3. merge the result into ``<step4-dir>/<sample_id>.jsonl``: replace any
       existing entry for this tool, keep every other tool, refresh
       ``tools_run`` (reuses ``merge_external_step4.merge_one``). If the
       file doesn't exist yet it's created with just this tool.

No step 5–8, no weight tensor, no LLM. Idempotent with ``--resume``.

Config
------
Reads the same ``step4_config.yaml`` the pipeline uses, so the tool's
settings (``tools.fpocket`` / ``tools.rfaa.{conda_env,install_dir,device,…}``
and ``structure_source.raw_dir``) come from there. ``--raw-dir`` overrides
the raw-structure dir; ``--device`` overrides ``tools.<tool>.device`` (RFAA's
GPU pin, via CUDA_VISIBLE_DEVICES).

Usage (server)
--------------
::

    # Fpocket (CPU)
    python scripts/riboseer/run_adapter_to_step4.py \\
        --tool fpocket \\
        --processed-dir data/processed_quality \\
        --sample-list   data/processed_quality/splits_tmscore_035/train.txt \\
        --step4-config  configs/step4_config.yaml \\
        --step4-dir     data/batch_train_v7/step4 \\
        --raw-dir       data/raw \\
        --resume

    # RFAA (GPU) — pin a card with --device (or CUDA_VISIBLE_DEVICES)
    python scripts/riboseer/run_adapter_to_step4.py \\
        --tool rfaa \\
        --processed-dir data/processed_quality \\
        --sample-list   data/processed_quality/splits_tmscore_035/train.txt \\
        --step4-config  configs/step4_config.yaml \\
        --step4-dir     data/batch_train_v7/step4 \\
        --raw-dir       data/raw \\
        --device        cuda:0 \\
        --resume
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step4_tool_adapters.run import get_adapter  # noqa: E402
from step4_tool_adapters.schemas import (  # noqa: E402
    ToolPrediction, ToolPredictionSet, make_failure_prediction,
)
from scripts.riboseer.merge_external_step4 import (  # noqa: E402
    merge_one, read_set, write_set,
)

# step4 records use the registry id ``rosettafold2na``; accept ``rf2na``.
_TOOL_ID_ALIASES = {"rf2na": "rosettafold2na"}


def _canonical(tool_id: str) -> str:
    return _TOOL_ID_ALIASES.get(tool_id, tool_id)


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_sample(processed_dir: Path, sample_id: str) -> Optional[dict]:
    """Case-insensitive sample-JSON lookup (mirrors batch_predict)."""
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
    return None


def load_sample_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"sample list not found: {path}")
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s.split()[0])
    return out


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _inject_overrides(step4_cfg: dict, raw_dir: Optional[Path],
                      tool_id: str, device: Optional[str]) -> None:
    if raw_dir is not None:
        ss = step4_cfg.setdefault("structure_source", {})
        ss["raw_dir"] = str(raw_dir)
    if device is not None:
        tools = step4_cfg.setdefault("tools", {})
        tcfg = tools.setdefault(tool_id, {})
        tcfg["device"] = device


def _existing_success(record: Optional[dict], tool_id: str) -> bool:
    if not record:
        return False
    for p in record.get("predictions") or []:
        if p.get("tool_id") == tool_id and p.get("success") is True:
            return True
    return False


def run_one(tool_id: str, sample_json: dict, work_dir: Path,
            step4_cfg: dict) -> ToolPrediction:
    """Run a single adapter. Never raises (BaseAdapter.predict swallows)."""
    sid = sample_json.get("sample_id", "<unknown>")
    sw = Path(work_dir) / sid
    sw.mkdir(parents=True, exist_ok=True)
    try:
        adapter = get_adapter(_canonical(tool_id))
    except ValueError as e:
        return make_failure_prediction(
            tool_id=tool_id, category="D", sample_id=sid,
            error_message=f"adapter lookup failed: {e}")
    return adapter.predict(sample_json, sw, step4_cfg)


def merge_into_step4(step4_dir: Path, sample_id: str, tool_id: str,
                     pred: ToolPrediction, runtime: float) -> tuple[dict, str]:
    """Merge one prediction into <step4-dir>/<sid>.jsonl. Returns
    ``(merged_record, action)`` where action ∈ {create, update}."""
    tgt_path = step4_dir / f"{sample_id}.jsonl"
    target = read_set(tgt_path)
    source = ToolPredictionSet(
        sample_id=sample_id,
        tools_run=[tool_id],
        predictions=[pred],
        total_runtime_seconds=round(runtime, 3),
        timestamp=utc_iso(),
    ).model_dump(mode="json")
    merged, _src_tools, _replaced = merge_one(source, target, sample_id)
    action = "create" if target is None else "update"
    write_set(tgt_path, merged)
    return merged, action


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tool", required=True, help="tool_id (e.g. fpocket, rfaa)")
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--sample-list", type=Path, required=True)
    p.add_argument("--step4-config", type=Path,
                   default=REPO_ROOT / "configs" / "step4_config.yaml")
    p.add_argument("--step4-dir", type=Path, required=True,
                   help="target step4 JSONL dir (e.g. data/batch_train_v7/step4)")
    p.add_argument("--raw-dir", type=Path, default=None,
                   help="override structure_source.raw_dir")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="adapter staging dir (default <step4-dir>/../work_<tool>)")
    p.add_argument("--device", default=None,
                   help="override tools.<tool>.device, e.g. cuda:0 (RFAA GPU pin)")
    p.add_argument("--resume", action="store_true",
                   help="skip samples whose step4 already has success=True "
                        "for this tool")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    tool_id = args.tool
    step4_cfg = load_yaml(args.step4_config)
    if not step4_cfg:
        print(f"ERROR: empty/missing step4 config: {args.step4_config}",
              file=sys.stderr)
        return 1
    _inject_overrides(step4_cfg, args.raw_dir, _canonical(tool_id), args.device)

    work_dir = args.work_dir or (args.step4_dir.parent / f"work_{tool_id}")
    args.step4_dir.mkdir(parents=True, exist_ok=True)

    try:
        sample_ids = load_sample_list(args.sample_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if args.start is not None or args.end is not None:
        sample_ids = sample_ids[(args.start or 0):
                                (args.end if args.end is not None
                                 else len(sample_ids))]
    if not sample_ids:
        print("ERROR: no samples to run", file=sys.stderr)
        return 1

    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:
        def tqdm(it, **kw):  # type: ignore
            return it

    summary_path = args.step4_dir.parent / "summary" / f"run_{tool_id}_results.jsonl"
    n_ok = n_fail_tool = n_missing = n_skip = 0
    for sid in tqdm(sample_ids, desc=f"{tool_id}->step4"):
        if args.resume and _existing_success(
                read_set(args.step4_dir / f"{sid}.jsonl"), tool_id):
            n_skip += 1
            continue
        sample = load_sample(args.processed_dir, sid)
        if sample is None:
            n_missing += 1
            append_jsonl(summary_path, {
                "sample_id": sid, "tool": tool_id, "status": "no_sample_json",
                "timestamp": utc_iso()})
            print(f"WARNING: no sample JSON for {sid}; skipped", file=sys.stderr)
            continue
        sample.setdefault("sample_id", sid)
        t0 = time.monotonic()
        try:
            pred = run_one(tool_id, sample, work_dir, step4_cfg)
        except Exception as e:  # noqa: BLE001 — defensive; predict shouldn't raise
            tb = traceback.format_exc(limit=6)
            append_jsonl(summary_path, {
                "sample_id": sid, "tool": tool_id, "status": "crash",
                "error": f"{type(e).__name__}: {e}", "traceback": tb,
                "timestamp": utc_iso()})
            n_fail_tool += 1
            continue
        runtime = time.monotonic() - t0
        _merged, action = merge_into_step4(
            args.step4_dir, sid, tool_id, pred, runtime)
        ok = bool(pred.success)
        n_ok += int(ok)
        n_fail_tool += int(not ok)
        append_jsonl(summary_path, {
            "sample_id": sid, "tool": tool_id,
            "status": "ok" if ok else "tool_failed",
            "action": action, "error": pred.error_message,
            "runtime_seconds": round(runtime, 3), "timestamp": utc_iso()})

    print()
    print(f"Done [{tool_id}]: success={n_ok}  tool_failed={n_fail_tool}  "
          f"missing_json={n_missing}  skipped={n_skip}  "
          f"total={len(sample_ids)}")
    print(f"step4 dir: {args.step4_dir}")
    print(f"summary  : {summary_path}")
    return 0 if n_fail_tool == 0 and n_missing == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
