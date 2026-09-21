#!/usr/bin/env python3
"""Search for the tool subset that maximises test-set Pearson R.

Given the 15-tool library (``features_15tool.ALL_KNOWN_TOOLS``), find the
subset whose fused LightGBM prediction scores the highest per-residue
Pearson R on the test split. Three strategies:

1. **Backward elimination** — start from all 15 tools; each round try
   removing every remaining tool and keep the removal that scores highest;
   stop when no single removal strictly improves Pearson R.
2. **Forward selection** — start from the empty set; each round try adding
   every absent tool and keep the addition that scores highest; stop when
   no single addition improves Pearson R.
3. **Exhaustive** (``--exhaustive``) — score every non-empty subset within
   ``[--min-tools, --max-tools]`` and report the top-K. Full 2^15 is ~32K
   trainings (≈3-5 s each ⇒ ~30-45 h); the bounds keep it tractable.

This is a **pure tool search**: features are built with the SCOPE block
off (``use_scope=False``), no MAESTRO, no POLISH — only the tool subset
varies. It reuses the Table-10 machinery (``ablation_leave_one_out.
train_and_eval`` + ``_aggregate``) so a "subset score" here is exactly a
Table-10 row, and the LightGBM recipe matches Table 4 / 9 / 10.

The search algorithms take an injectable ``score_fn(tools) -> dict|None``
(``None`` when the correlation is undefined), so the greedy logic is
testable without any ML dependency.

Usage
-----
::

    python scripts/riboseer/search_best_tool_subset.py \\
        --train-step4-dir data/batch_train_v7/step4/ \\
        --test-step4-dir  data/batch_test_v7/step4/ \\
        --processed-dir   data/processed_quality \\
        --train-list      data/processed_quality/splits_tmscore_035/train.txt \\
        --test-list       data/processed_quality/splits_tmscore_035/test.txt \\
        --output          data/batch_test_v7/best_tool_subset.csv
        # optional: --exhaustive --min-tools 3 --max-tools 12 --top-k 20
"""
from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step5_fusion.lightgbm_fusion import (  # noqa: E402
    LGBM_OK,
    SampleT9,
    collect_samples,
)
from scripts.tables.table09_llm_modules import (  # noqa: E402
    _load_sample_ids,
)
from scripts.riboseer.ablation_leave_one_out import (  # noqa: E402
    DISPLAY, _aggregate, train_and_eval,
)
from step5_fusion.features_15tool import ALL_KNOWN_TOOLS  # noqa: E402

# score_fn maps a tool tuple → an aggregate dict (pearson_r/spearman_r/
# r_squared/n_samples) or None when the correlation is undefined.
ScoreFn = Callable[[tuple[str, ...]], Optional[dict]]


def _pretty(tools) -> str:
    return "+".join(tools) if tools else "(empty)"


def _pr(d: Optional[dict]) -> float:
    """Pearson R as a sortable scalar (undefined → -inf)."""
    if d is None or d.get("pearson_r") is None:
        return float("-inf")
    return float(d["pearson_r"])


# ---------------------------------------------------------------------------
# Subset scoring (the only part that needs LightGBM / real data)
# ---------------------------------------------------------------------------


def score_subset(train: list[SampleT9], test: list[SampleT9],
                 tools: tuple[str, ...]) -> Optional[dict]:
    """Train on ``tools`` and return the Table-10 aggregate, or None when
    empty / the per-sample correlation is everywhere undefined."""
    if not tools:
        return None
    agg = _aggregate(_pretty(tools), train_and_eval(train, test, tuple(tools)))
    return None if agg["pearson_r"] is None else agg


def make_cached_scorer(train: list[SampleT9],
                       test: list[SampleT9]) -> tuple[ScoreFn, dict]:
    """Wrap ``score_subset`` with a frozenset cache so repeated subsets
    (common across both greedy passes) train only once. Returns the
    scorer and its cache (cache size = number of distinct trainings)."""
    cache: dict[frozenset, Optional[dict]] = {}

    def scorer(tools: tuple[str, ...]) -> Optional[dict]:
        key = frozenset(tools)
        if key not in cache:
            cache[key] = score_subset(train, test, tools)
        return cache[key]

    return scorer, cache


# ---------------------------------------------------------------------------
# Search strategies (model-agnostic; driven by an injectable score_fn)
# ---------------------------------------------------------------------------


def backward_elimination(all_tools: tuple[str, ...], score_fn: ScoreFn, *,
                         min_tools: int = 1,
                         log: Callable[[str], None] = print) -> list[dict]:
    """Greedy removal. Returns a history list of accepted steps; the last
    entry is the chosen subset."""
    current = list(all_tools)
    base = score_fn(tuple(current))
    history = [{"action": "baseline", "tool": None, "tools": list(current),
                "score": base}]
    log(f"[backward] baseline {len(current)} tools  "
        f"PearsonR={_fmt(base)}")

    while len(current) > min_tools:
        best = None  # (dict, removed_tool, subset)
        for t in current:
            subset = tuple(x for x in current if x != t)
            sc = score_fn(subset)
            if best is None or _pr(sc) > _pr(best[0]):
                best = (sc, t, subset)
        sc, t, subset = best
        if _pr(sc) > _pr(base):
            current = list(subset)
            base = sc
            history.append({"action": "remove", "tool": t,
                            "tools": list(current), "score": sc})
            log(f"[backward] − {DISPLAY.get(t, t):<14s} → {len(current)} tools "
                f" PearsonR={_fmt(sc)}")
        else:
            log(f"[backward] no removal improves (best try "
                f"−{DISPLAY.get(t, t)} → {_fmt(sc)}); stop")
            break
    return history


def forward_selection(all_tools: tuple[str, ...], score_fn: ScoreFn, *,
                      max_tools: Optional[int] = None,
                      log: Callable[[str], None] = print) -> list[dict]:
    """Greedy addition. Returns a history list of accepted steps; the last
    entry is the chosen subset."""
    current: list[str] = []
    remaining = list(all_tools)
    base: Optional[dict] = None
    cap = max_tools or len(all_tools)
    history = [{"action": "baseline", "tool": None, "tools": [],
                "score": None}]
    log("[forward] baseline 0 tools")

    while remaining and len(current) < cap:
        best = None  # (dict, added_tool, subset)
        for t in remaining:
            subset = tuple(current + [t])
            sc = score_fn(subset)
            if best is None or _pr(sc) > _pr(best[0]):
                best = (sc, t, subset)
        sc, t, subset = best
        # accept iff defined AND (first pick OR strict improvement)
        if sc is not None and (base is None or _pr(sc) > _pr(base)):
            current = list(subset)
            remaining.remove(t)
            base = sc
            history.append({"action": "add", "tool": t,
                            "tools": list(current), "score": sc})
            log(f"[forward] + {DISPLAY.get(t, t):<14s} → {len(current)} tools "
                f" PearsonR={_fmt(sc)}")
        else:
            log(f"[forward] no addition improves (best try "
                f"+{DISPLAY.get(t, t)} → {_fmt(sc)}); stop")
            break
    return history


def exhaustive_search(all_tools: tuple[str, ...], score_fn: ScoreFn, *,
                      min_tools: int, max_tools: int, top_k: int = 20,
                      log: Callable[[str], None] = print) -> list[dict]:
    """Score every subset of size in ``[min_tools, max_tools]`` and return
    the top-K by Pearson R (descending)."""
    total = sum(comb(len(all_tools), k)
                for k in range(min_tools, max_tools + 1))
    log(f"[exhaustive] sizes {min_tools}-{max_tools}: {total} subsets")
    scored: list[dict] = []
    seen = 0
    for k in range(min_tools, max_tools + 1):
        for combo in combinations(all_tools, k):
            sc = score_fn(combo)
            seen += 1
            if sc is not None:
                scored.append({"tools": list(combo), "score": sc})
            if seen % 1000 == 0:
                log(f"[exhaustive] {seen}/{total} scored")
    scored.sort(key=lambda r: _pr(r["score"]), reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_COLUMNS = ["method", "rank", "action", "tool", "n_tools", "tools",
            "pearson_r", "spearman_r", "r_squared", "n_samples"]


def _fmt(d: Optional[dict]) -> str:
    return "n/a" if d is None or d.get("pearson_r") is None \
        else f"{d['pearson_r']:.4f}"


def _row(method: str, rank: int, action: str, tool: Optional[str],
         tools: list[str], score: Optional[dict]) -> dict:
    s = score or {}
    return {"method": method, "rank": rank, "action": action,
            "tool": DISPLAY.get(tool, tool) if tool else "",
            "n_tools": len(tools), "tools": "+".join(tools),
            "pearson_r": s.get("pearson_r"), "spearman_r": s.get("spearman_r"),
            "r_squared": s.get("r_squared"), "n_samples": s.get("n_samples")}


def build_output_rows(backward: list[dict], forward: list[dict],
                      exhaustive: Optional[list[dict]]) -> list[dict]:
    rows: list[dict] = []
    for i, h in enumerate(backward):
        rows.append(_row("backward", i, h["action"], h["tool"],
                         h["tools"], h["score"]))
    for i, h in enumerate(forward):
        rows.append(_row("forward", i, h["action"], h["tool"],
                         h["tools"], h["score"]))
    if exhaustive is not None:
        for i, h in enumerate(exhaustive, 1):
            rows.append(_row("exhaustive", i, "", None,
                             h["tools"], h["score"]))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in _COLUMNS})


def _best(history: list[dict]) -> dict:
    """The chosen (final) step of a greedy history."""
    return history[-1]


def print_summary(backward: list[dict], forward: list[dict],
                  exhaustive: Optional[list[dict]]) -> None:
    print("\n=== Best subsets ===")
    for name, hist in (("Backward", backward), ("Forward", forward)):
        b = _best(hist)
        print(f"{name:<9s}: {len(b['tools'])} tools  "
              f"PearsonR={_fmt(b['score'])}  [{_pretty(b['tools'])}]")
    if exhaustive:
        print("\nExhaustive top results:")
        print(f"  {'rank':>4s} {'n':>3s} {'PearsonR':>9s}  tools")
        for i, h in enumerate(exhaustive, 1):
            print(f"  {i:>4d} {len(h['tools']):>3d} {_fmt(h['score']):>9s}  "
                  f"{_pretty(h['tools'])}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-step4-dir", type=Path, required=True)
    p.add_argument("--test-step4-dir", type=Path, required=True)
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--train-list", type=Path, required=True)
    p.add_argument("--test-list", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--exhaustive", action="store_true",
                   help="also brute-force all subsets in the size window")
    p.add_argument("--min-tools", type=int, default=3,
                   help="exhaustive: smallest subset size (default 3)")
    p.add_argument("--max-tools", type=int, default=12,
                   help="exhaustive: largest subset size (default 12)")
    p.add_argument("--top-k", type=int, default=20,
                   help="exhaustive: how many best subsets to report")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not LGBM_OK:
        print("ERROR: lightgbm not installed (pip install lightgbm)",
              file=sys.stderr)
        return 1
    for d in (args.train_step4_dir, args.test_step4_dir, args.processed_dir):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 1
    try:
        train_ids = _load_sample_ids(args.train_list)
        test_ids = _load_sample_ids(args.test_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    train = collect_samples(args.train_step4_dir, args.processed_dir,
                            train_ids)
    test = collect_samples(args.test_step4_dir, args.processed_dir, test_ids)
    print(f"train usable: {len(train)}, test usable: {len(test)}")
    if not train or not test:
        print("ERROR: no usable train/test samples", file=sys.stderr)
        return 1

    scorer, cache = make_cached_scorer(train, test)

    print("\n--- Backward elimination ---")
    backward = backward_elimination(ALL_KNOWN_TOOLS, scorer)
    print("\n--- Forward selection ---")
    forward = forward_selection(ALL_KNOWN_TOOLS, scorer)

    exhaustive = None
    if args.exhaustive:
        lo = max(1, args.min_tools)
        hi = min(len(ALL_KNOWN_TOOLS), args.max_tools)
        print("\n--- Exhaustive search ---")
        exhaustive = exhaustive_search(ALL_KNOWN_TOOLS, scorer,
                                       min_tools=lo, max_tools=hi,
                                       top_k=args.top_k)

    rows = build_output_rows(backward, forward, exhaustive)
    write_csv(args.output, rows)
    print_summary(backward, forward, exhaustive)
    print(f"\nwrote {args.output}  ({len(rows)} rows; "
          f"{len(cache)} distinct trainings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
