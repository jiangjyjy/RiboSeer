"""Step 4 — batch orchestrator driven by step 3's tool selection plan.

Reads a step-3 JSONL output (one record per sample, each carrying a
``tool_plan.selected_tools`` list) and runs each plan with the
deployed-tool subset.

Usage
-----
::

    python -m step4_tool_adapters.run_all \\
        --processed-dir data/processed \\
        --step3-output data/step3_outputs/e2e_smoke.jsonl \\
        --config configs/step4_config.yaml \\
        --output data/step4_outputs/

Output layout
-------------
One JSONL file per sample under ``--output``:

    data/step4_outputs/
      1un6_B_F.jsonl
      2bgg_A_P.jsonl
      ...

Each file holds a single ToolPredictionSet line — same shape as
``run.py``'s output.

Step-3 records the plan can include tool IDs that aren't deployed yet
(see ``step3_tool_selection.tool_registry``). By default we silently
filter those out and run only the deployed subset; pass
``--strict-plan`` to make a sample fail when its plan asks for an
undeployed tool.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .run import (
    ADAPTER_REGISTRY, _summarize, load_config, load_sample, run_sample,
    write_jsonl_record,
)
from .schemas import ToolPrediction


# ---------- step 3 record helpers -----------------------------------------


def load_step3_records(path: Path) -> list[dict]:
    """Read a JSONL of step-3 records. Skips blank / comment lines."""
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            records.append(json.loads(line))
    return records


def extract_planned_tools(record: dict) -> list[str]:
    """Pull ``tool_plan.selected_tools`` out of a step-3 record.

    Returns empty list if the record is missing the field — callers
    can decide whether that's a skip or an error.
    """
    plan = record.get("tool_plan") or {}
    tools = plan.get("selected_tools") or []
    if not isinstance(tools, list):
        return []
    return [str(t) for t in tools]


def filter_deployed(
    tools: list[str],
    *,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Split ``tools`` into ``(deployed, dropped)`` against
    ``ADAPTER_REGISTRY``. With ``strict=True`` the second list being
    non-empty signals an error to the caller."""
    deployed: list[str] = []
    dropped: list[str] = []
    for t in tools:
        if t in ADAPTER_REGISTRY:
            deployed.append(t)
        else:
            dropped.append(t)
    return deployed, dropped


# ---------- batch loop -----------------------------------------------------


def _make_skip_record(sample_id: str, reason: str) -> dict:
    """Tiny diagnostic dict for skipped samples — printed, not written
    to disk (we don't want skip records cluttering step4_outputs)."""
    return {"sample_id": sample_id, "skipped": True, "reason": reason}


def run_batch(
    records: list[dict],
    *,
    processed_dir: Path,
    config: dict,
    work_dir: Path,
    out_dir: Path,
    strict_plan: bool = False,
    only_successful_plans: bool = True,
    sample_filter: Optional[set[str]] = None,
) -> dict:
    """Process a list of step-3 records. Returns aggregate counters.

    ``only_successful_plans`` skips records whose ``success`` flag is
    False (step-3 fallback or hard error). ``sample_filter`` restricts
    to specific sample_ids if provided.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(records)
    n_ok = n_partial = n_fail = n_skip = 0
    skip_log: list[dict] = []

    for i, record in enumerate(records, start=1):
        sid = record.get("sample_id", "?")
        if sample_filter is not None and sid not in sample_filter:
            continue

        if only_successful_plans and not record.get("success", True):
            print(f"[{i:>3}/{n_total}] SKIP {sid}: step3 record success=False")
            n_skip += 1
            skip_log.append(_make_skip_record(sid, "step3 success=False"))
            continue

        planned = extract_planned_tools(record)
        if not planned:
            print(f"[{i:>3}/{n_total}] SKIP {sid}: no tool_plan.selected_tools")
            n_skip += 1
            skip_log.append(_make_skip_record(sid, "no selected_tools"))
            continue

        deployed, dropped = filter_deployed(planned)
        if dropped and strict_plan:
            print(f"[{i:>3}/{n_total}] FAIL {sid}: undeployed tools in plan "
                  f"{dropped} (strict mode)")
            n_fail += 1
            continue
        if dropped:
            print(f"[{i:>3}/{n_total}] note {sid}: dropping undeployed "
                  f"tools {dropped} (deployed: {deployed})")
        if not deployed:
            print(f"[{i:>3}/{n_total}] SKIP {sid}: plan has no deployed tools "
                  f"(planned={planned})")
            n_skip += 1
            skip_log.append(_make_skip_record(sid, f"no deployed tools in {planned}"))
            continue

        try:
            sample = load_sample(processed_dir, sid)
        except FileNotFoundError as e:
            print(f"[{i:>3}/{n_total}] SKIP {sid}: {e}")
            n_skip += 1
            skip_log.append(_make_skip_record(sid, str(e)))
            continue

        pred_set = run_sample(sample, deployed, config, work_dir)
        out_path = out_dir / f"{sid}.jsonl"
        write_jsonl_record(pred_set, out_path)

        n_tool_ok = sum(1 for p in pred_set.predictions if p.success)
        n_tool_fail = len(pred_set.predictions) - n_tool_ok
        if n_tool_fail == 0:
            n_ok += 1
            tag = "OK  "
        elif n_tool_ok == 0:
            n_fail += 1
            tag = "FAIL"
        else:
            n_partial += 1
            tag = "PART"
        print(f"[{i:>3}/{n_total}] {tag} {_summarize(pred_set)} -> {out_path.name}")

    return {
        "total": n_total,
        "ok": n_ok,
        "partial": n_partial,
        "fail": n_fail,
        "skip": n_skip,
        "skip_log": skip_log,
    }


# ---------- argparse main --------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, required=True,
                   help="step 1 output dir containing samples/<id>.json")
    p.add_argument("--step3-output", type=Path, required=True,
                   help="JSONL output from step 3 (one record per sample)")
    p.add_argument("--config", type=Path, required=True,
                   help="step4_config.yaml")
    p.add_argument("--output", type=Path, default=None,
                   help="output dir for per-sample JSONL files (default: "
                        "<config.output.batch_jsonl_dir>)")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="staging dir (default: <config.work_dir>)")
    p.add_argument("--strict-plan", action="store_true",
                   help="fail samples whose plan references undeployed tools")
    p.add_argument("--include-failed-plans", action="store_true",
                   help="also run records where step3 success=False (default: skip)")
    p.add_argument("--sample-id", type=str, action="append", default=None,
                   help="restrict to one or more sample_ids "
                        "(repeat flag for multiple)")
    p.add_argument("--limit", type=int, default=None,
                   help="cap number of samples processed")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    records = load_step3_records(args.step3_output)
    if args.limit is not None:
        records = records[:args.limit]

    work_dir = args.work_dir or Path(config.get("work_dir", "data/step4_workdir"))
    out_cfg = config.get("output") or {}
    out_dir = args.output or Path(
        out_cfg.get("batch_jsonl_dir", "data/step4_outputs"),
    )
    sample_filter = set(args.sample_id) if args.sample_id else None

    print(f"Loaded {len(records)} step-3 record(s) from {args.step3_output}")
    print(f"Output dir   : {out_dir}")
    print(f"Work dir     : {work_dir}")
    print(f"Strict plan  : {args.strict_plan}")
    print(f"Include fail : {args.include_failed_plans}")
    print()

    summary = run_batch(
        records,
        processed_dir=args.processed_dir,
        config=config,
        work_dir=work_dir,
        out_dir=out_dir,
        strict_plan=args.strict_plan,
        only_successful_plans=not args.include_failed_plans,
        sample_filter=sample_filter,
    )

    print()
    print("=" * 72)
    print(f"Total: {summary['total']}  "
          f"ok={summary['ok']}  partial={summary['partial']}  "
          f"fail={summary['fail']}  skip={summary['skip']}")

    return 0 if summary["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
