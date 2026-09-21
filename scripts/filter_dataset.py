"""Filter ``data/processed`` to a smaller, RNA-length-capped subset.

Why this exists
---------------
The 500-sample train batch surfaced two cohorts that no current tool
handles well:

  - Boltz-2 / Chai-1 OOM or timeout on RNA > ~200 nt (the underlying
    diffusion + structure module memory cost is roughly quadratic in
    sequence length).
  - RNA > 200 nt samples are mostly ribosomal / spliceosomal fragments,
    not the typical RNA-protein binding pairs we care about.

Empirically the F1 on those samples sat at ≈ 0. Excluding them up-front
keeps the training and eval sets focused on samples where every
deployed tool can actually contribute.

Usage
-----
::

    python scripts/filter_dataset.py \\
        --processed-dir data/processed \\
        --max-rna-length 200 \\
        --output-dir data/processed_filtered

This copies every ``samples/<id>.json`` whose ``rna.length`` is at most
``--max-rna-length`` into ``<output-dir>/samples/`` and prints the
counts. ``splits.json`` and any other top-level files are NOT copied —
the output dir is meant to feed into ``recluster_split.py``, which
re-derives the splits from scratch on the filtered set.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional


def filter_dataset(
    processed_dir: Path,
    max_rna_length: int,
    output_dir: Path,
) -> dict:
    """Copy in-range ``samples/<id>.json`` to ``<output_dir>/samples/``.

    Returns a counts dict ``{total, kept, filtered, no_rna_length}``.

    Decisions / edge cases:
      - ``rna.length`` missing or non-integer → treat as ``no_rna_length``;
        skipped, logged separately so we don't silently lose them.
      - ``rna.length == 0`` → kept (the original step-1 pipeline already
        dropped truly-empty RNAs; a 0 here would be a step-1 bug worth
        spotting in the output, not a filter target).
      - The output dir is created fresh-or-extended; existing files are
        overwritten via ``shutil.copy2`` (idempotent re-runs).
    """
    samples_dir = processed_dir / "samples"
    if not samples_dir.is_dir():
        raise FileNotFoundError(
            f"samples dir not found: {samples_dir}. "
            f"Pass the parent of samples/ as --processed-dir."
        )

    out_samples = output_dir / "samples"
    out_samples.mkdir(parents=True, exist_ok=True)

    total = kept = filtered = no_rna_length = 0
    for f in sorted(samples_dir.glob("*.json")):
        total += 1
        try:
            sample = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"WARN: skipping malformed sample {f.name}: {e}",
                  file=sys.stderr)
            continue

        rna_len = (sample.get("rna") or {}).get("length")
        if not isinstance(rna_len, int):
            no_rna_length += 1
            continue

        if rna_len > max_rna_length:
            filtered += 1
            continue

        shutil.copy2(f, out_samples / f.name)
        kept += 1

    return {
        "total": total,
        "kept": kept,
        "filtered": filtered,
        "no_rna_length": no_rna_length,
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed-dir", type=Path, required=True,
                   help="step 1 output dir (parent of samples/).")
    p.add_argument("--max-rna-length", type=int, default=200,
                   help="drop samples whose rna.length exceeds this "
                        "(default 200; matches the empirical Boltz-2 / "
                        "Chai-1 memory ceiling).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="root for the filtered dataset; samples/ is "
                        "(re)created underneath.")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.max_rna_length <= 0:
        print(f"ERROR: --max-rna-length must be positive (got "
              f"{args.max_rna_length})", file=sys.stderr)
        return 1

    try:
        counts = filter_dataset(
            args.processed_dir, args.max_rna_length, args.output_dir,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Total samples scanned:               {counts['total']:>6}")
    print(f"Kept (RNA length ≤ {args.max_rna_length:>3}):"
          f"{'':<14}{counts['kept']:>6}")
    print(f"Filtered (RNA length > {args.max_rna_length:>3}):"
          f"{'':<10}{counts['filtered']:>6}")
    if counts["no_rna_length"]:
        print(f"Skipped (no integer rna.length):     "
              f"{counts['no_rna_length']:>6}")
    print()
    print(f"Wrote: {args.output_dir / 'samples'}")
    print()
    print("Next steps:")
    print(f"  python src/step1_server/cluster_and_split.py \\")
    print(f"      --processed-dir {args.output_dir} \\")
    print(f"      --stats-dir     {args.output_dir / 'stats'} \\")
    print(f"      --work-dir      {args.output_dir / '_cluster_work'}")
    print()
    print(f"  python scripts/split_samples.py \\")
    print(f"      --processed-dir {args.output_dir} \\")
    print(f"      --output-dir    {args.output_dir / 'splits'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
