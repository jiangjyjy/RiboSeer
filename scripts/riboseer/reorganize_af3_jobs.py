"""Bucket the AF3 Server JSON jobs into day-folders by priority.

After ``af3_inputs.py`` writes 105 JSON jobs and a separate
``af3_submission_order.csv``, the JSON file names give no hint about
which one to upload next. This script reorganises ``af3_jobs/`` into
``af3_jobs/day{N}/{priority:03d}_{sample_id}.json`` so the daily upload
workflow is "open day{N}, drag everything inside up to AF3 Server".

``--per-day`` controls the bucket size (default 30 → 4 day-folders
for 105 jobs: day1/day2/day3 with 30 each, day4 with 15).

The mover is idempotent: a re-run picks files up from the flat layout
(``{sample_id}.json`` at root) AND from any existing day-bucket
location, so a partial / aborted run is recoverable.

Usage
-----
::

    python scripts/riboseer/reorganize_af3_jobs.py \\
        --order-csv data/batch_test_v7/af3_submission_order.csv \\
        --jobs-dir  data/batch_test_v7/af3_jobs/

Add ``--dry-run`` to print the planned moves without touching anything.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Optional


def read_priority_order(path: Path) -> list[tuple[int, str]]:
    """Parses ``af3_submission_order.csv`` → ``[(priority, sample_id), ...]``
    sorted by priority asc. Required columns: ``priority``, ``sample_id``.
    Raises ValueError on a missing column (fail loud — silent skip
    would put the wrong file in the wrong day-folder).
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"priority", "sample_id"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path}: order CSV missing columns "
                f"{sorted(missing)} (saw {reader.fieldnames})")
        rows: list[tuple[int, str]] = []
        for r in reader:
            try:
                p = int(r["priority"])
            except (TypeError, ValueError):
                continue
            sid = (r.get("sample_id") or "").strip()
            if sid:
                rows.append((p, sid))
    rows.sort(key=lambda kv: kv[0])
    return rows


def day_for_priority(priority: int, per_day: int) -> int:
    """1-based day bucket — priorities 1..per_day → day 1, etc.
    Per-day must be ≥ 1; non-positive raises so a typo in the CLI
    doesn't divide-by-zero deep in the loop."""
    if per_day < 1:
        raise ValueError(f"per_day must be ≥ 1, got {per_day}")
    return (priority - 1) // per_day + 1


def target_relative_path(priority: int, sample_id: str,
                         per_day: int) -> Path:
    """``day{N}/{priority:03d}_{sample_id}.json`` — the on-disk
    location of one job AFTER reorganisation, relative to the jobs
    root dir."""
    day = day_for_priority(priority, per_day)
    return Path(f"day{day}") / f"{priority:03d}_{sample_id}.json"


def find_source(jobs_dir: Path, sample_id: str,
                priority: int, per_day: int) -> Optional[Path]:
    """Locate the current on-disk path of ``sample_id``'s JSON. Tries
    the canonical "already-reorganised" target first (idempotent
    re-runs do zero work), then the original flat
    ``{sample_id}.json``, then every ``day*/*.json`` in case an
    earlier run bucketed under a different ``--per-day``.

    Returns ``None`` if no candidate exists — caller logs and skips."""
    expected = jobs_dir / target_relative_path(priority, sample_id,
                                               per_day)
    if expected.is_file():
        return expected
    flat = jobs_dir / f"{sample_id}.json"
    if flat.is_file():
        return flat
    # Fallback: scan day-buckets — covers re-running with a different
    # --per-day (e.g. switching from 30/day to 25/day mid-project).
    suffix = f"_{sample_id}.json"
    for sub in jobs_dir.glob("day*"):
        if not sub.is_dir():
            continue
        for cand in sub.glob(f"*{suffix}"):
            if cand.name.endswith(suffix):
                return cand
    return None


def plan_moves(order: list[tuple[int, str]],
               jobs_dir: Path,
               per_day: int,
               ) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Returns ``(moves, missing)``:

    - ``moves``: ``(src, dst)`` pairs to execute, where src != dst
      (already-correct files contribute nothing). Both absolute paths.
    - ``missing``: sample_ids in the order CSV with no JSON found on
      disk. Reported but the script continues (e.g. samples that were
      cleaned away to empty are legitimately absent).
    """
    moves: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for prio, sid in order:
        src = find_source(jobs_dir, sid, prio, per_day)
        if src is None:
            missing.append(sid)
            continue
        dst = jobs_dir / target_relative_path(prio, sid, per_day)
        if src.resolve() != dst.resolve():
            moves.append((src, dst))
    return moves, missing


def execute_moves(moves: list[tuple[Path, Path]],
                  *, dry_run: bool = False) -> int:
    """Atomic per-file rename. ``Path.rename`` is atomic on the same
    filesystem so a crash mid-batch never leaves a half-written file.
    Returns the number of moves performed (0 in dry-run)."""
    n = 0
    for src, dst in moves:
        if dry_run:
            print(f"  DRY-RUN  {src.name:48s} → "
                  f"{dst.parent.name}/{dst.name}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)            # atomic; overwrites if dst exists
        n += 1
    return n


# ---- main ---------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--order-csv", type=Path, required=True,
                   help="path to af3_submission_order.csv "
                        "(must have priority + sample_id columns).")
    p.add_argument("--jobs-dir", type=Path, required=True,
                   help="dir holding the AF3 Server JSON job files. "
                        "Files are reorganised in place into day-N/ "
                        "subdirs of this same dir.")
    p.add_argument("--per-day", type=int, default=30,
                   help="how many jobs per day-bucket (default 30).")
    p.add_argument("--dry-run", action="store_true",
                   help="print the planned moves without executing.")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not args.order_csv.is_file():
        print(f"ERROR: --order-csv not a file: {args.order_csv}",
              file=sys.stderr)
        return 1
    if not args.jobs_dir.is_dir():
        print(f"ERROR: --jobs-dir not a directory: {args.jobs_dir}",
              file=sys.stderr)
        return 1

    try:
        order = read_priority_order(args.order_csv)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    moves, missing = plan_moves(order, args.jobs_dir, args.per_day)
    n_total = len(order)
    n_days = (math.ceil(n_total / args.per_day) if n_total else 0)
    print(f"order csv  : {args.order_csv}  ({n_total} entries)")
    print(f"jobs dir   : {args.jobs_dir}")
    print(f"per-day    : {args.per_day}  → day1..day{n_days}")
    print(f"to move    : {len(moves)} (skip {n_total - len(moves) - len(missing)} already in place)")
    if missing:
        print(f"missing    : {len(missing)} sample(s) with no JSON on "
              f"disk (likely cleaned-away empty samples):",
              file=sys.stderr)
        for sid in missing[:20]:
            print(f"  {sid}", file=sys.stderr)
        if len(missing) > 20:
            print(f"  ... +{len(missing) - 20} more",
                  file=sys.stderr)

    n_done = execute_moves(moves, dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry-run: would move {len(moves)} files")
    else:
        print(f"moved   : {n_done} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
