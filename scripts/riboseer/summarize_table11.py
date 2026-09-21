#!/usr/bin/env python3
"""Table 11 — assemble the policy-comparison table.

Pulls the **scope=on / maestro=on / polish=off** row from each policy's
``table09_llm_modules.py`` output CSV and the avg tools/sample from each
policy's TEST selection dir, then emits::

    Selection policy              Pearson R   R2    # tools/sample
    Free LLM selection            0.569       0.401 5.7
    MANDATORY A tools + LLM       X.XXX       X.XXX X.X
    All 15 tools always           X.XXX       X.XXX 15.0

The three ablation CSVs differ only by ``--maestro-selections-{train,test}``;
everything else (SCOPE=on, POLISH=off) is held fixed, so the only variable
across rows is the tool-selection policy.

Usage::

    python scripts/riboseer/summarize_table11.py \\
        --free-csv      data/batch_test_v7/table09_llm_modules.csv \\
        --free-sel      data/batch_test_v7/maestro_selections_llm_v4/ \\
        --mandatory-csv data/batch_test_v7/table09_llm_modules_mandatory.csv \\
        --mandatory-sel data/batch_test_v7/maestro_selections_mandatory/ \\
        --all-csv       data/batch_test_v7/table09_llm_modules_all.csv \\
        --all-sel       data/batch_test_v7/maestro_selections_all/ \\
        --output        data/batch_test_v7/table11_mandatory_tools.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.riboseer.make_table11_selections import (  # noqa: E402
    avg_tools_per_sample, load_selection_dir,
)

# The row that defines every Table 11 cell.
ROW = {"scope": "on", "maestro": "on", "polish": "off"}

_COLUMNS = ["policy", "pearson_r", "r_squared", "tools_per_sample",
            "runtime_min", "n_samples"]


def read_runtime_csv(csv_path: Optional[Path]) -> dict[str, float]:
    """``{policy: runtime_sequential_min}`` from table11_runtime.py
    output. Missing/absent file → empty (runtime column stays blank)."""
    out: dict[str, float] = {}
    if csv_path is None or not csv_path.is_file():
        return out
    with csv_path.open(encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            val = _num(rec.get("runtime_sequential_min"))
            if rec.get("policy") and val is not None:
                out[rec["policy"]] = val
    return out


def read_ablation_row(csv_path: Path,
                      row=ROW) -> Optional[dict]:
    """Return the (scope, maestro, polish) row's metrics, or None."""
    if not csv_path.is_file():
        return None
    with csv_path.open(encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            if all(rec.get(k) == v for k, v in row.items()):
                return rec
    return None


def _num(s):
    try:
        return round(float(s), 4)
    except (TypeError, ValueError):
        return None


def build_row(policy: str, csv_path: Path,
              sel_dir: Path, runtime_min: Optional[float] = None) -> dict:
    rec = read_ablation_row(csv_path)
    sel = load_selection_dir(sel_dir) if sel_dir.is_dir() else {}
    return {
        "policy": policy,
        "pearson_r": _num(rec.get("pearson_r_mean")) if rec else None,
        "r_squared": _num(rec.get("r2_mean")) if rec else None,
        "tools_per_sample": avg_tools_per_sample(sel),
        "runtime_min": runtime_min,
        "n_samples": int(rec["n_samples"]) if rec and rec.get("n_samples")
        else None,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in _COLUMNS})


_LABELS = {
    "free": "Free LLM selection",
    "mandatory": "MANDATORY A tools + LLM",
    "all": "All 15 tools always",
}


def print_table(rows: list[dict]) -> None:
    hdr = (f"{'Selection policy':<26s} {'PearsonR':>9s} {'R2':>7s} "
           f"{'#tools/sample':>14s} {'runtime(min)':>13s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{_LABELS.get(r['policy'], r['policy']):<26s} "
              f"{str(r['pearson_r']):>9s} {str(r['r_squared']):>7s} "
              f"{str(r['tools_per_sample']):>14s} "
              f"{str(r['runtime_min']):>13s}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--free-csv", type=Path, required=True)
    p.add_argument("--free-sel", type=Path, required=True)
    p.add_argument("--mandatory-csv", type=Path, required=True)
    p.add_argument("--mandatory-sel", type=Path, required=True)
    p.add_argument("--all-csv", type=Path, required=True)
    p.add_argument("--all-sel", type=Path, required=True)
    p.add_argument("--runtime-csv", type=Path, default=None,
                   help="table11_runtime.py output (optional)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rtm = read_runtime_csv(args.runtime_csv)
    rows = [
        build_row("free", args.free_csv, args.free_sel, rtm.get("free")),
        build_row("mandatory", args.mandatory_csv, args.mandatory_sel,
                  rtm.get("mandatory")),
        build_row("all", args.all_csv, args.all_sel, rtm.get("all")),
    ]
    write_csv(args.output, rows)
    print(f"wrote {args.output}")
    print()
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
