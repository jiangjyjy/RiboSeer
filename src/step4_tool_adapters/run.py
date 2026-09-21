"""Step 4 — single-sample CLI entry point.

Runs one or more deployed tool adapters on a single sample and writes a
``ToolPredictionSet`` JSONL record (one line per file, one record per
sample).

Usage
-----
Single tool::

    python -m step4_tool_adapters.run \\
        --processed-dir data/processed \\
        --sample-id 1un6_B_F \\
        --tool p2rank \\
        --config configs/step4_config.yaml \\
        --output data/step4_outputs/1un6_B_F.jsonl

Multiple tools (comma-separated)::

    python -m step4_tool_adapters.run \\
        --processed-dir data/processed \\
        --sample-id 1un6_B_F \\
        --tools p2rank,boltz2,equipnas \\
        --config configs/step4_config.yaml

Output layout
-------------
One JSONL line per invocation:

    {
      "sample_id": "...",
      "tools_run": ["p2rank", "boltz2", ...],
      "predictions": [<ToolPrediction>, ...],
      "total_runtime_seconds": 180.4,
      "timestamp": "2026-04-29T12:34:56Z"
    }

Each entry in ``predictions`` is a fully-typed ``ToolPrediction``;
failures appear with ``success=False`` and an ``error_message`` rather
than aborting the batch — see ``schemas.py`` for the failure contract.

Cross-tool structure passing
----------------------------
The orchestrator runs tools in a fixed order (Cat A → B → C → D) but
does NOT yet hand a Cat A tool's predicted structure to subsequent
Cat B/C tools. Samples without a raw structure file must either have
their Cat A output staged into ``structure_source.raw_dir`` externally
before re-running, or have Cat B/C skipped via the ``--tools`` flag.
This is a planned step-5 enhancement; for now it keeps the orchestrator
deterministic and easy to debug.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import yaml

from .adapters import (
    AlphaFold3Adapter, BindUPAdapter, Boltz2Adapter, Chai1Adapter,
    DeepPocketAdapter, EquiPNASAdapter, FpocketAdapter, GraphBindAdapter,
    Haddock3Adapter, HdockAdapter, NucleicNetAdapter, P2RankAdapter,
    RF2NAAdapter, RFAAAdapter, RNABindRPlusAdapter,
)
from .base_adapter import BaseAdapter
from .schemas import ToolPrediction, ToolPredictionSet


# ---------- registry -------------------------------------------------------


# tool_id → adapter class, one entry per tool of the library (paper Table 1).
# Keys must match step3's tool_registry IDs so run_all.py can map
# plan.selected_tools → adapter without translation. tests/test_tool_registry.py
# asserts that this covers every registered tool_id.
ADAPTER_REGISTRY: dict[str, type[BaseAdapter]] = {
    # Category A — complex structure prediction
    "boltz2": Boltz2Adapter,
    "chai1": Chai1Adapter,
    "rosettafold2na": RF2NAAdapter,
    "rfaa": RFAAAdapter,
    "alphafold3": AlphaFold3Adapter,
    # Category B — pocket detection
    "p2rank": P2RankAdapter,
    "fpocket": FpocketAdapter,
    "deeppocket": DeepPocketAdapter,
    # Category C — binding residue prediction
    "equipnas": EquiPNASAdapter,
    "nucleicnet": NucleicNetAdapter,
    "graphbind": GraphBindAdapter,
    "rnabindrplus": RNABindRPlusAdapter,
    "bindup": BindUPAdapter,
    # Category D — docking
    "hdock": HdockAdapter,
    "haddock3": Haddock3Adapter,
}


# Deterministic execution order: Cat A first (longest, but also produces
# structures we may eventually feed back into B/C), then C/B (per-residue
# / pocket), then D (docking — currently no D tools deployed).
_CATEGORY_ORDER = {"A": 0, "C": 1, "B": 2, "D": 3}


def get_adapter(tool_id: str) -> BaseAdapter:
    cls = ADAPTER_REGISTRY.get(tool_id)
    if cls is None:
        raise ValueError(
            f"unknown tool_id {tool_id!r}; registered adapters: "
            f"{sorted(ADAPTER_REGISTRY)}"
        )
    return cls()


def order_tools(tool_ids: Iterable[str]) -> list[str]:
    """Sort tool_ids so Cat A runs before C / B. Stable within category."""
    seen: list[str] = []
    for t in tool_ids:
        if t not in seen:
            seen.append(t)
    return sorted(
        seen,
        key=lambda t: (
            _CATEGORY_ORDER.get(get_adapter(t).category, 99),
            seen.index(t),
        ),
    )


# ---------- IO helpers -----------------------------------------------------


def load_config(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_sample(processed_dir: Path, sample_id: str) -> dict:
    """Load a step-1 sample JSON. Searches ``samples/<id>.json`` then
    ``<id>.json`` directly under processed_dir."""
    candidates = [
        processed_dir / "samples" / f"{sample_id}.json",
        processed_dir / f"{sample_id}.json",
    ]
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"sample {sample_id!r} not found in "
        f"{[str(p) for p in candidates]}"
    )


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------- core runner ----------------------------------------------------


def _resolve_parallel_workers(
    config: dict, override: Optional[int] = None,
) -> int:
    """Decide how many adapters to run in parallel.

    Resolution order: explicit ``override`` (CLI --parallel/--no-parallel)
    → ``config.execution.parallel_workers`` → default 4. Values <= 1
    select serial mode (used for offline smoke tests + single-tool runs).
    """
    if override is not None:
        return int(override)
    exec_cfg = config.get("execution") or {}
    return int(exec_cfg.get("parallel_workers", 4))


def run_sample(
    sample: dict,
    tool_ids: list[str],
    config: dict,
    work_dir: Path,
    *,
    on_prediction: Optional[callable] = None,
    parallel_workers: Optional[int] = None,
) -> ToolPredictionSet:
    """Run all requested tools on one sample, collect predictions.

    Predictions in the returned ``ToolPredictionSet`` are sorted by the
    canonical category order (Cat A → C → B → D) regardless of the
    underlying execution mode. Each adapter's ``predict`` swallows its
    own exceptions, so this function never raises on adapter failure —
    a bad tool just shows up with ``success=False``.

    Execution
    ---------
    - ``parallel_workers <= 1`` (or ``--no-parallel``): serial loop, in
      category order. Behaves exactly like the pre-optimisation code.
    - ``parallel_workers >= 2`` (default 4 from
      ``config.execution.parallel_workers``): a ``ThreadPoolExecutor``
      dispatches every adapter at once. ``BaseAdapter.predict`` is
      thread-safe (each adapter writes to its own subdir under
      ``work_dir/<sample_id>/``), so concurrent runs don't trample each
      other's files. CUDA models share the GPU; if VRAM is tight, dial
      this down.

    The optional ``on_prediction`` callback fires once per tool with
    the completed ``ToolPrediction``. **In parallel mode, callback
    order follows completion order (non-deterministic), not category
    order.** Callers that depend on a specific firing order should set
    ``parallel_workers=1`` or sort downstream.
    """
    sample_id = sample["sample_id"]
    sample_work = Path(work_dir) / sample_id
    sample_work.mkdir(parents=True, exist_ok=True)

    ordered = order_tools(tool_ids)
    workers = _resolve_parallel_workers(config, parallel_workers)
    # Cap workers at the actual tool count — a 4-worker pool for a
    # single tool is wasteful and confuses test mocks counting threads.
    effective_workers = max(1, min(workers, len(ordered)))

    predictions: list[ToolPrediction] = []
    start = time.monotonic()

    if effective_workers <= 1 or len(ordered) <= 1:
        # Serial fallback: keeps callback order = category order, which
        # the existing on_prediction contract historically promised.
        for tool_id in ordered:
            adapter = get_adapter(tool_id)
            pred = adapter.predict(sample, sample_work, config)
            predictions.append(pred)
            if on_prediction is not None:
                on_prediction(pred)
    else:
        # Parallel dispatch. Submit all adapters then drain via
        # as_completed so a slow tool doesn't gate the callback for the
        # ones that already finished.
        adapters_by_id = {tid: get_adapter(tid) for tid in ordered}
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            future_to_tool = {
                pool.submit(
                    adapters_by_id[tid].predict, sample, sample_work, config,
                ): tid
                for tid in ordered
            }
            for future in as_completed(future_to_tool):
                # BaseAdapter.predict never raises — its try/except
                # folds anything into success=False. Defensive guard
                # below covers a hypothetical contract violation.
                try:
                    pred = future.result()
                except Exception as e:  # noqa: BLE001
                    bad_tid = future_to_tool[future]
                    pred = ToolPrediction(
                        tool_id=bad_tid,
                        category=adapters_by_id[bad_tid].category,
                        sample_id=sample_id,
                        success=False,
                        error_message=(
                            f"thread pool returned exception: "
                            f"{type(e).__name__}: {e}"
                        ),
                    )
                predictions.append(pred)
                if on_prediction is not None:
                    on_prediction(pred)
        # Sort back into canonical category order so the JSONL record
        # is reproducible across serial / parallel modes.
        cat_index = {tid: i for i, tid in enumerate(ordered)}
        predictions.sort(key=lambda p: cat_index.get(p.tool_id, 99))

    elapsed = time.monotonic() - start

    return ToolPredictionSet(
        sample_id=sample_id,
        tools_run=ordered,
        predictions=predictions,
        total_runtime_seconds=round(elapsed, 3),
        timestamp=_utc_iso(),
    )


def write_jsonl_record(prediction_set: ToolPredictionSet, output_path: Path) -> Path:
    """Write a single ToolPredictionSet as one JSONL line, atomically."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    payload = prediction_set.model_dump_json()
    with tmp.open("w", encoding="utf-8") as f:
        f.write(payload + "\n")
    tmp.replace(output_path)
    return output_path


# ---------- summary --------------------------------------------------------


def _summarize(pred_set: ToolPredictionSet) -> str:
    n_ok = sum(1 for p in pred_set.predictions if p.success)
    n_fail = len(pred_set.predictions) - n_ok
    parts = []
    for p in pred_set.predictions:
        flag = "OK" if p.success else "FAIL"
        parts.append(f"{p.tool_id}={flag}")
    return (
        f"sample={pred_set.sample_id}  "
        f"tools={len(pred_set.predictions)}  "
        f"ok={n_ok}  fail={n_fail}  "
        f"runtime={pred_set.total_runtime_seconds}s  "
        f"[{' '.join(parts)}]"
    )


# ---------- argparse main --------------------------------------------------


def _parse_tool_ids(args: argparse.Namespace) -> list[str]:
    if args.tool:
        return [args.tool]
    return [t.strip() for t in args.tools.split(",") if t.strip()]


def _default_output_path(config: dict, sample_id: str) -> Path:
    out_cfg = config.get("output") or {}
    out_dir = Path(out_cfg.get("batch_jsonl_dir", "data/step4_outputs"))
    return out_dir / f"{sample_id}.jsonl"


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, required=True,
                   help="step 1 output dir containing samples/<id>.json")
    p.add_argument("--sample-id", type=str, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--tool", type=str,
                       help="single tool_id to run")
    group.add_argument("--tools", type=str,
                       help="comma-separated list of tool_ids")
    p.add_argument("--config", type=Path, required=True,
                   help="step4_config.yaml")
    p.add_argument("--output", type=Path, default=None,
                   help="path to write JSONL (default: "
                        "<config.output.batch_jsonl_dir>/<sample-id>.jsonl)")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="staging dir for per-sample inputs/outputs "
                        "(default: <config.work_dir>)")
    parallel_grp = p.add_mutually_exclusive_group()
    parallel_grp.add_argument(
        "--parallel", type=int, default=None, metavar="N",
        help="run up to N adapters concurrently (overrides "
             "config.execution.parallel_workers; pass 1 for serial)",
    )
    parallel_grp.add_argument(
        "--no-parallel", action="store_true",
        help="run adapters serially (equivalent to --parallel 1)",
    )
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    sample = load_sample(args.processed_dir, args.sample_id)
    tool_ids = _parse_tool_ids(args)
    work_dir = args.work_dir or Path(config.get("work_dir", "data/step4_workdir"))
    out_path = args.output or _default_output_path(config, args.sample_id)

    parallel_override: Optional[int] = None
    if args.no_parallel:
        parallel_override = 1
    elif args.parallel is not None:
        parallel_override = args.parallel

    pred_set = run_sample(
        sample, tool_ids, config, work_dir,
        parallel_workers=parallel_override,
    )
    write_jsonl_record(pred_set, out_path)

    print(_summarize(pred_set))
    print(f"wrote: {out_path}")
    n_fail = sum(1 for p in pred_set.predictions if not p.success)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
