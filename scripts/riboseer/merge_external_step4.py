#!/usr/bin/env python3
"""Merge externally-produced step-4 JSONL predictions into the main step-4 dir.

Some baselines are run off the main pipeline (e.g. BindUP parsed on a laptop)
and their per-sample ``ToolPrediction`` records land in a separate directory.
This script folds those into the canonical ``step4/`` directory: for each
``<sample_id>.jsonl`` in ``--source-dir`` it loads the matching target file (if
any), **drops the target's predictions for the tool(s) present in the source**,
appends the source predictions, refreshes ``tools_run``, and rewrites the file
atomically. Every other tool's prediction in the target is preserved.

It is intentionally schema-light (plain ``json``, no pydantic import) so it runs
anywhere, and tool-agnostic — it merges whatever ``tool_id``s the source files
carry, not just one baseline.

Usage
-----
::

    # preview
    python scripts/riboseer/merge_external_step4.py \
        --source-dir data/batch_test_v7/bindup_step4 \
        --target-dir data/batch_test_v7/step4 --dry-run

    # apply
    python scripts/riboseer/merge_external_step4.py \
        --source-dir data/batch_test_v7/bindup_step4 \
        --target-dir data/batch_test_v7/step4
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("merge_external_step4")


def read_set(path: Path) -> Optional[dict]:
    """Read the first non-empty JSONL record (a ToolPredictionSet) as a dict."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None
    return None


def merge_one(source: dict, target: Optional[dict], sample_id: str) -> tuple:
    """Return ``(merged_set, source_tools, replaced_tools)``.

    ``source`` predictions replace any same-``tool_id`` predictions in
    ``target``; all other target predictions are kept. Target's other top-level
    fields (timestamp, total_runtime_seconds, ...) are preserved."""
    src_preds = source.get("predictions") or []
    src_tools = sorted({p.get("tool_id") for p in src_preds if p.get("tool_id")})

    base = dict(target) if target else {}
    tgt_preds = base.get("predictions") or []
    src_tool_set = set(src_tools)
    replaced = sorted({p.get("tool_id") for p in tgt_preds
                       if p.get("tool_id") in src_tool_set})

    kept = [p for p in tgt_preds if p.get("tool_id") not in src_tool_set]
    merged_preds = kept + src_preds

    base["sample_id"] = sample_id
    base["predictions"] = merged_preds
    base["tools_run"] = sorted({p.get("tool_id") for p in merged_preds
                                if p.get("tool_id")})
    return base, src_tools, replaced


def write_set(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text(json.dumps(record) + "\n", encoding="utf-8")
    tmp.replace(path)


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dir", required=True, type=Path,
                   help="dir of external <sample_id>.jsonl to merge in")
    p.add_argument("--target-dir", required=True, type=Path,
                   help="main step4 dir to merge into")
    p.add_argument("--dry-run", action="store_true",
                   help="report changes without writing")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(message)s")

    source_dir = args.source_dir.expanduser().resolve()
    target_dir = args.target_dir.expanduser().resolve()
    if not source_dir.is_dir():
        logger.error("source dir not found: %s", source_dir)
        return 1

    src_files = sorted(source_dir.glob("*.jsonl"))
    if not src_files:
        logger.error("no *.jsonl files under %s", source_dir)
        return 1

    n_new = n_updated = n_skip = 0
    for sf in src_files:
        sample_id = sf.stem
        source = read_set(sf)
        if source is None:
            n_skip += 1
            logger.warning("SKIP %s (unreadable/empty)", sf.name)
            continue
        tgt_path = target_dir / f"{sample_id}.jsonl"
        target = read_set(tgt_path)
        merged, src_tools, replaced = merge_one(source, target, sample_id)

        action = "create" if target is None else "update"
        note = ("new file" if target is None
                else ("replaced " + ",".join(replaced) if replaced else "added"))
        logger.info("%-7s %s : +%s (%s)", action, sample_id,
                    ",".join(src_tools), note)
        if not args.dry_run:
            write_set(tgt_path, merged)
        if target is None:
            n_new += 1
        else:
            n_updated += 1

    verb = "would merge" if args.dry_run else "merged"
    logger.info("%s %d source files -> %s (%d new, %d updated, %d skipped)",
                verb, len(src_files), target_dir, n_new, n_updated, n_skip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
