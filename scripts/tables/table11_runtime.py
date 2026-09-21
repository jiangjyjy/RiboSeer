#!/usr/bin/env python3
"""Table 11 — per-sample average runtime (minutes) per selection policy.

Each tool has a typical per-sample runtime. A policy's runtime for a sample
is the cost of the tools that policy selected for it; the reported number is
the mean over samples.

Runtime source (per tool, in this priority):

1. **Real** — mean ``runtime_seconds`` over *successful* predictions in the
   step4 dir (``--step4-dir``). Only numeric values count; external-service
   tools that log no timing fall through to the estimate.
2. **Estimate** — ``DEFAULT_RUNTIME_SECONDS`` below (the task's observed
   values), used for any tool with no numeric step4 timing.

Aggregation across a sample's selected tools:

* ``sequential`` (default, **conservative**) — sum of tool runtimes, i.e.
  tools run one after another.
* ``parallel`` — max of tool runtimes, i.e. all tools run concurrently
  (Cat-A structure tools are GPU-parallel; others may be too).

Both are written; the printed headline is ``sequential``.

Usage::

    python scripts/tables/table11_runtime.py \\
        --free-sel      data/batch_test_v7/maestro_selections_llm_v4/ \\
        --mandatory-sel data/batch_test_v7/maestro_selections_mandatory/ \\
        --all-sel       data/batch_test_v7/maestro_selections_all/ \\
        --step4-dir     data/batch_test_v7/step4/ \\
        --output        data/batch_test_v7/table11_runtime.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scripts.riboseer.make_table11_selections import (  # noqa: E402
    clean_tools, load_selection_dir,
)
from step5_fusion.features_15tool import (  # noqa: E402
    ALL_KNOWN_TOOLS, _canonical,
)

# Observed per-sample runtimes (seconds). Used when step4 has no numeric
# timing for a tool (e.g. external services that log none).
DEFAULT_RUNTIME_SECONDS: dict[str, float] = {
    "boltz2": 300, "chai1": 300, "rosettafold2na": 600, "rfaa": 300,
    "alphafold3": 600,
    "p2rank": 10, "fpocket": 5, "deeppocket": 30,
    "equipnas": 30, "nucleicnet": 180, "graphbind": 120,
    "rnabindrplus": 600, "bindup": 300,
    "hdock": 300, "haddock3": 600,
}


def extract_tool_runtimes(step4_dir: Optional[Path]) -> dict[str, float]:
    """Mean ``runtime_seconds`` per tool over successful, numerically-timed
    step4 predictions. Tools with no numeric timing are omitted."""
    if step4_dir is None or not step4_dir.is_dir():
        return {}
    acc: dict[str, list[float]] = {}
    for f in sorted(step4_dir.glob("*.jsonl")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8").splitlines()[0])
        except (OSError, json.JSONDecodeError, IndexError):
            continue
        for p in rec.get("predictions") or []:
            if not p.get("success"):
                continue
            rt = p.get("runtime_seconds")
            tid = _canonical(p.get("tool_id") or "")
            if tid in DEFAULT_RUNTIME_SECONDS and isinstance(rt, (int, float)):
                acc.setdefault(tid, []).append(float(rt))
    return {t: statistics.fmean(vs) for t, vs in acc.items() if vs}


def resolve_runtimes(step4_dir: Optional[Path]
                     ) -> tuple[dict[str, float], dict[str, str]]:
    """Per-tool runtime (real > estimate) plus a per-tool source tag."""
    real = extract_tool_runtimes(step4_dir)
    runtimes: dict[str, float] = {}
    source: dict[str, str] = {}
    for t in ALL_KNOWN_TOOLS:
        if t in real:
            runtimes[t] = round(real[t], 1)
            source[t] = "real"
        else:
            runtimes[t] = float(DEFAULT_RUNTIME_SECONDS[t])
            source[t] = "estimate"
    return runtimes, source


def sample_runtime(tools: list[str], runtimes: dict[str, float],
                   mode: str) -> float:
    """Runtime (seconds) for one sample's selected tools."""
    costs = [runtimes.get(t, 0.0) for t in clean_tools(tools)]
    if not costs:
        return 0.0
    return max(costs) if mode == "parallel" else sum(costs)


def policy_runtime_minutes(records: dict[str, dict],
                           runtimes: dict[str, float],
                           mode: str) -> Optional[float]:
    secs = [sample_runtime(r.get("selected_tools") or [], runtimes, mode)
            for r in records.values()]
    if not secs:
        return None
    return round(statistics.fmean(secs) / 60.0, 2)


_COLUMNS = ["policy", "n_samples", "avg_tools",
            "runtime_sequential_min", "runtime_parallel_min"]

_LABELS = {
    "free": "Free LLM selection",
    "mandatory": "MANDATORY A tools + LLM",
    "all": "All 15 tools always",
}


def build_row(policy: str, sel_dir: Path,
              runtimes: dict[str, float]) -> dict:
    records = load_selection_dir(sel_dir) if sel_dir.is_dir() else {}
    tool_counts = [len(clean_tools(r.get("selected_tools") or []))
                   for r in records.values()]
    return {
        "policy": policy,
        "n_samples": len(records),
        "avg_tools": round(statistics.fmean(tool_counts), 2)
        if tool_counts else None,
        "runtime_sequential_min":
            policy_runtime_minutes(records, runtimes, "sequential"),
        "runtime_parallel_min":
            policy_runtime_minutes(records, runtimes, "parallel"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in _COLUMNS})


def print_runtimes(runtimes: dict[str, float], source: dict[str, str]) -> None:
    print("Per-tool runtime (seconds):")
    for t in ALL_KNOWN_TOOLS:
        print(f"  {t:16s} {runtimes[t]:8.1f}s  ({source[t]})")
    print()


def print_table(rows: list[dict]) -> None:
    hdr = (f"{'Selection policy':<26s} {'#tools':>6s} "
           f"{'seq(min)':>9s} {'par(min)':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{_LABELS.get(r['policy'], r['policy']):<26s} "
              f"{str(r['avg_tools']):>6s} "
              f"{str(r['runtime_sequential_min']):>9s} "
              f"{str(r['runtime_parallel_min']):>9s}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--free-sel", type=Path, required=True)
    p.add_argument("--mandatory-sel", type=Path, required=True)
    p.add_argument("--all-sel", type=Path, required=True)
    p.add_argument("--step4-dir", type=Path, default=None,
                   help="step4 dir for real runtime_seconds (else estimates)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    runtimes, source = resolve_runtimes(args.step4_dir)
    rows = [
        build_row("free", args.free_sel, runtimes),
        build_row("mandatory", args.mandatory_sel, runtimes),
        build_row("all", args.all_sel, runtimes),
    ]
    write_csv(args.output, rows)
    print_runtimes(runtimes, source)
    print(f"wrote {args.output}")
    print()
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
