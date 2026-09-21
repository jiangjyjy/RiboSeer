#!/usr/bin/env python
"""Stage 1.5b — merge splits.json back into every sample JSON's split_info.

⚠️  LINUX SERVER TASK in theory (ships alongside `cluster_and_split.py`),
    but this script is pure Python — no MMseqs2 dependency — so it can
    technically run anywhere the sample JSONs live. Kept under
    `step1_server/` to preserve the "1.5 is a server stage" grouping.

What it does
------------
Reads `data/processed/splits.json` and writes, for every sample:

  sample["split_info"]["protein_cluster_id"] = "prot_000042"
  sample["split_info"]["rna_cluster_id"]     = "rna_000017"
  sample["split_info"]["split"]              = "train" | "val" | "test" | null

`null` split means the sample is in `splits.json` but tagged unclusterable
(e.g. the 122 all-X protein samples → `protein_cluster_id =
"unclusterable_X_only"`).

Samples referenced in `index.csv` but **absent** from `splits.json` are a
config error — this script fails loudly instead of silently tagging them
unclusterable, because the clustering stage should have seen every sample.
Pass `--allow-missing` to downgrade that to a warning.

Idempotent + atomic write + resume
----------------------------------
  - Resume: if every key already equals what `splits.json` says, the sample
    file is not touched. Use `--force` to rewrite every sample unconditionally.
  - Atomic: tmp file + `os.replace`, same pattern as the local stage 1.3a /
    1.4 scripts.

Usage
-----
    python src/step1_server/apply_splits.py \\
        --processed-dir /srv/pocket/data/processed

    # dry run (compute would-be changes, don't write)
    python src/step1_server/apply_splits.py \\
        --processed-dir /srv/pocket/data/processed --dry-run

    # force rewrite every sample (e.g. after re-running clustering)
    python src/step1_server/apply_splits.py \\
        --processed-dir /srv/pocket/data/processed --force
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path


# ---------- sid_to_filename (kept in sync with step1_local/_common.py) ------


def _encode_case_marked(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalpha() and ch.islower():
            out.append("-")
        out.append(ch)
    return "".join(out)


def sid_to_filename(sample_id: str) -> str:
    pdb, _, tail = sample_id.partition("_")
    if not tail:
        return _encode_case_marked(sample_id)
    return f"{pdb}_{_encode_case_marked(tail)}"


# ---------- main ------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1.5b: merge splits.json into every sample JSON."
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="don't write samples, just report what would change")
    parser.add_argument("--force", action="store_true",
                        help="rewrite every sample even if split_info already matches")
    parser.add_argument("--allow-missing", action="store_true",
                        help="don't fail when index.csv has samples absent from splits.json")
    args = parser.parse_args()

    index_csv = args.processed_dir / "index.csv"
    splits_json_path = args.processed_dir / "splits.json"
    samples_dir = args.processed_dir / "samples"

    if not splits_json_path.exists():
        sys.exit(f"{splits_json_path} not found — run cluster_and_split.py first")
    if not index_csv.exists():
        sys.exit(f"{index_csv} not found")

    with splits_json_path.open("r", encoding="utf-8") as f:
        splits = json.load(f)
    samples_map: dict[str, dict] = splits["samples"]
    print(f"Loaded splits.json: {len(samples_map)} sample entries")

    with index_csv.open("r", encoding="utf-8") as f:
        index_rows = list(csv.DictReader(f))
    print(f"Loaded index.csv: {len(index_rows)} samples")

    missing_in_splits = [r["sample_id"] for r in index_rows
                         if r["sample_id"] not in samples_map]
    if missing_in_splits:
        msg = (f"{len(missing_in_splits)} samples in index.csv are missing from "
               f"splits.json (first 5: {missing_in_splits[:5]})")
        if args.allow_missing:
            print("WARNING:", msg, file=sys.stderr)
        else:
            sys.exit("ERROR: " + msg + "\nRerun cluster_and_split.py, or pass --allow-missing.")

    totals = Counter()
    totals["n_index_rows"] = len(index_rows)
    split_dist: Counter = Counter()

    try:
        from tqdm import tqdm
        iterator = tqdm(index_rows, desc="apply-splits", mininterval=1.0)
    except ImportError:
        iterator = index_rows

    for row in iterator:
        sid = row["sample_id"]
        entry = samples_map.get(sid)
        if entry is None:
            totals["n_missing_in_splits"] += 1
            continue

        sample_path = samples_dir / (sid_to_filename(sid) + ".json")
        with sample_path.open("r", encoding="utf-8") as f:
            sample = json.load(f)

        split_info = sample.setdefault("split_info", {
            "protein_cluster_id": None,
            "rna_cluster_id": None,
            "split": None,
        })

        needs_write = args.force or (
            split_info.get("protein_cluster_id") != entry["protein_cluster_id"]
            or split_info.get("rna_cluster_id") != entry["rna_cluster_id"]
            or split_info.get("split") != entry["split"]
        )

        if not needs_write:
            totals["n_unchanged"] += 1
            split_dist[entry["split"] or "unclustered"] += 1
            continue

        split_info["protein_cluster_id"] = entry["protein_cluster_id"]
        split_info["rna_cluster_id"] = entry["rna_cluster_id"]
        split_info["split"] = entry["split"]

        if not args.dry_run:
            tmp_path = sample_path.with_suffix(".json.tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(sample, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, sample_path)

        totals["n_updated"] += 1
        split_dist[entry["split"] or "unclustered"] += 1

    print()
    print(f"Samples processed : {totals['n_index_rows']}")
    print(f"Updated           : {totals['n_updated']}")
    print(f"Unchanged         : {totals['n_unchanged']}")
    print(f"Missing in splits : {totals['n_missing_in_splits']}")
    print("Per-split counts (post-apply):")
    for split in ("train", "val", "test", "unclustered"):
        n = split_dist.get(split, 0)
        print(f"  {split:12s}: {n}")
    if args.dry_run:
        print("(dry-run — no sample files written)")


if __name__ == "__main__":
    main()
