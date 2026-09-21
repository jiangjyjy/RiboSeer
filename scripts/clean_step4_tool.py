"""Strip a tool's predictions out of step4 JSONL records, in place.

Use cases
---------
- HADDOCK3 is disabled (Pearson R≈0.055, 34 % success) — its
  back-filled predictions must not reach fusion / eval::

      python scripts/clean_step4_tool.py \\
          --step4-dir data/batch_train_v4/step4/ --remove-tool haddock3

- RF2NA was accidentally run under the wrong tool_id ``rf2na`` (the
  registry / adapter id is ``rosettafold2na``). Those records are junk
  — drop them (they happen to all be ``success=false``)::

      python scripts/clean_step4_tool.py \\
          --step4-dir data/batch_train_v4/step4/ --remove-tool rf2na

For every ``*.jsonl`` under ``--step4-dir``, for every JSON record in
the file, this:

  1. drops every ``predictions[*]`` whose ``tool_id`` == --remove-tool
     (or only the ``success=false`` ones with ``--only-failed``),
  2. removes that id from ``tools_run``,
  3. decrements ``total_runtime_seconds`` by the sum of the removed
     entries' ``runtime_seconds`` (clamped at 0),
  4. rewrites the file atomically (``.tmp`` → replace).

``tool_id`` is matched EXACTLY — ``rf2na`` and ``rosettafold2na`` are
deliberately distinct here (the former is the wrong-id junk, the
latter the real Cat A tool). ``--dry-run`` reports what would change
and writes nothing. Non-JSON lines are passed through untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def _clean_record(
    rec: dict, tool_id: str, only_failed: bool,
) -> tuple[dict, int, float, bool]:
    """Return (new_record, n_removed, runtime_removed, tools_run_changed).

    Only builds a new dict when something actually changes; otherwise
    returns the input object so unchanged files can be left as-is.
    """
    preds = rec.get("predictions")
    if not isinstance(preds, list):
        return rec, 0, 0.0, False

    kept: list = []
    removed: list = []
    for p in preds:
        is_target = (
            isinstance(p, dict)
            and p.get("tool_id") == tool_id
            and (not only_failed or p.get("success") is False)
        )
        (removed if is_target else kept).append(p)

    if not removed:
        return rec, 0, 0.0, False

    runtime_removed = 0.0
    for p in removed:
        rt = p.get("runtime_seconds")
        if isinstance(rt, (int, float)):
            runtime_removed += float(rt)

    out = dict(rec)
    out["predictions"] = kept

    tools_run_changed = False
    tr = rec.get("tools_run")
    if isinstance(tr, list):
        # Only drop the id from tools_run if NO entry of that tool
        # survives (with --only-failed a success record may remain).
        survivors = {p.get("tool_id") for p in kept
                     if isinstance(p, dict)}
        if tool_id in tr and tool_id not in survivors:
            out["tools_run"] = [t for t in tr if t != tool_id]
            tools_run_changed = True

    if "total_runtime_seconds" in rec:
        prior = rec.get("total_runtime_seconds")
        if isinstance(prior, (int, float)):
            out["total_runtime_seconds"] = round(
                max(0.0, float(prior) - runtime_removed), 3)

    return out, len(removed), runtime_removed, tools_run_changed


def _process_file(
    path: Path, tool_id: str, only_failed: bool, dry_run: bool,
) -> dict:
    """Clean one JSONL file. Returns per-file counters."""
    n_removed = 0
    runtime_removed = 0.0
    changed = False
    out_lines: list[str] = []

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        return {"file": str(path), "error": f"read failed: {e}",
                "n_removed": 0, "runtime_removed": 0.0,
                "changed": False}

    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            rec = json.loads(s)
        except json.JSONDecodeError:
            out_lines.append(line)          # pass through verbatim
            continue
        new_rec, n, rt, tr_changed = _clean_record(
            rec, tool_id, only_failed)
        if n or tr_changed:
            changed = True
            n_removed += n
            runtime_removed += rt
        out_lines.append(json.dumps(new_rec, ensure_ascii=False))

    if changed and not dry_run:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        tmp.replace(path)

    return {"file": str(path), "n_removed": n_removed,
            "runtime_removed": round(runtime_removed, 3),
            "changed": changed}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step4-dir", type=Path, required=True,
                   help="dir of *.jsonl step4 records to clean.")
    p.add_argument("--remove-tool", type=str, required=True,
                   help="exact tool_id to strip (e.g. haddock3, rf2na).")
    p.add_argument("--only-failed", action="store_true",
                   help="only drop success=false entries of that tool "
                        "(default: drop all of them).")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would change; write nothing.")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not args.step4_dir.is_dir():
        print(f"ERROR: --step4-dir not a directory: {args.step4_dir}",
              file=sys.stderr)
        return 1

    files = sorted(args.step4_dir.glob("*.jsonl"))
    if not files:
        print(f"ERROR: no *.jsonl under {args.step4_dir}",
              file=sys.stderr)
        return 1

    n_files_changed = 0
    total_removed = 0
    total_runtime = 0.0
    errors = 0
    for f in files:
        r = _process_file(f, args.remove_tool, args.only_failed,
                          args.dry_run)
        if r.get("error"):
            errors += 1
            print(f"WARN: {r['file']}: {r['error']}", file=sys.stderr)
            continue
        if r["changed"]:
            n_files_changed += 1
            total_removed += r["n_removed"]
            total_runtime += r["runtime_removed"]

    mode = "DRY-RUN (no writes)" if args.dry_run else "applied"
    flt = " success=false only" if args.only_failed else ""
    print(f"clean_step4_tool [{mode}] remove-tool="
          f"{args.remove_tool!r}{flt}")
    print(f"  scanned files          : {len(files)}")
    print(f"  files {'to change' if args.dry_run else 'changed':<16}: "
          f"{n_files_changed}")
    print(f"  predictions removed    : {total_removed}")
    print(f"  runtime reclaimed (s)  : {round(total_runtime, 3)}")
    if errors:
        print(f"  unreadable files       : {errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
