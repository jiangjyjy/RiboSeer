"""Filter processed samples by RNA length, carry over the cluster split.

Why
---
``scripts/filter_dataset.py`` capped RNA at 200 nt. Boltz-2 / Chai-1
still OOM / time out on the 100–200 nt tail, so this keeps a tunable
band ``--min-rna-length`` ≤ L ≤ ``--max-rna-length`` (default 20–100;
the lower bound drops degenerate ultra-short RNAs). Unlike
filter_dataset.py this also:

  - reuses the ORIGINAL cluster-based split from
    ``data/processed/splits.json`` (never re-splits — leakage-safe),
  - writes ``splits/{train,val,test}.txt`` for the kept samples,
  - draws reproducible ``train_200.txt`` / ``test_200.txt`` subsets
    (fixed seed),
  - prints per-split counts + the kept RNA-length distribution.

Usage
-----
::

    python scripts/filter_samples.py \\
        --input-dir   data/processed/samples/ \\
        --output-dir  data/processed_filtered_100/ \\
        --min-rna-length 20 --max-rna-length 100

Layout produced under ``--output-dir``::

    samples/<id>.json            (copied, or symlinked with --symlink)
    splits/train.txt val.txt test.txt
    splits/train_200.txt test_200.txt
    filter_stats.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Optional


# ---- core (pure, unit-tested) --------------------------------------------


def rna_length_of(sample: dict) -> Optional[int]:
    """``rna.length`` if a positive int, else ``len(rna.sequence)``,
    else ``None`` (no usable RNA length → sample is dropped)."""
    rna = sample.get("rna") or {}
    n = rna.get("length")
    if isinstance(n, int) and n > 0:
        return n
    seq = rna.get("sequence")
    if isinstance(seq, str) and seq:
        return len(seq)
    return None


def load_splits_map(splits_json: Path) -> dict[str, str]:
    """``{sample_id: split}`` from splits.json's ``samples`` block.

    Keys are also stored lower-cased so lookups survive the
    case-mismatch the rest of the pipeline guards against. ``split``
    may be None (unclusterable) — those map to ``""``.
    """
    if not splits_json.is_file():
        return {}
    data = json.loads(splits_json.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for sid, meta in (data.get("samples") or {}).items():
        sp = meta.get("split") if isinstance(meta, dict) else meta
        sp = sp or ""
        out[sid] = sp
        out.setdefault(sid.lower(), sp)
    return out


def filter_samples(
    input_dir: Path,
    splits_map: dict[str, str],
    max_rna_length: int,
    min_rna_length: int = 0,
) -> dict:
    """Scan ``input_dir/*.json``; classify each by RNA length + split.

    Keep iff ``min_rna_length <= rna_len <= max_rna_length``. Returns a
    dict with: ``kept`` (list of (sample_id, path, rna_len, split)),
    counters, per-split id lists, kept RNA-length list. Read-only.
    """
    kept: list[tuple[str, Path, int, str]] = []
    by_split: dict[str, list[str]] = {"train": [], "val": [],
                                      "test": [], "unknown": []}
    rna_lens: list[int] = []
    n_total = n_long = n_short = n_no_rna = n_bad_json = 0

    for jf in sorted(input_dir.glob("*.json")):
        n_total += 1
        try:
            sample = json.loads(jf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            n_bad_json += 1
            continue
        sid = sample.get("sample_id") or jf.stem
        rlen = rna_length_of(sample)
        if rlen is None:
            n_no_rna += 1
            continue
        if rlen > max_rna_length:
            n_long += 1
            continue
        if rlen < min_rna_length:
            n_short += 1
            continue
        split = (splits_map.get(sid)
                 or splits_map.get(sid.lower()) or "")
        bucket = split if split in ("train", "val", "test") else "unknown"
        kept.append((sid, jf, rlen, bucket))
        by_split[bucket].append(sid)
        rna_lens.append(rlen)

    return {
        "kept": kept,
        "by_split": by_split,
        "rna_lens": rna_lens,
        "counts": {
            "total_scanned": n_total,
            "kept": len(kept),
            "filtered_too_long": n_long,
            "filtered_too_short": n_short,
            "no_rna_length": n_no_rna,
            "bad_json": n_bad_json,
        },
    }


def sample_subset(ids: list[str], n: int, seed: int) -> list[str]:
    """Reproducible subset: sort for determinism, then seeded sample.
    If ``len(ids) <= n`` return all (sorted)."""
    pool = sorted(ids)
    if len(pool) <= n:
        return pool
    return sorted(random.Random(seed).sample(pool, n))


def _length_stats(lens: list[int]) -> dict:
    if not lens:
        return {"n": 0, "min": None, "max": None,
                "mean": None, "median": None}
    return {
        "n": len(lens),
        "min": min(lens),
        "max": max(lens),
        "mean": round(statistics.fmean(lens), 2),
        "median": round(statistics.median(lens), 1),
    }


# ---- IO ------------------------------------------------------------------


def _materialise(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        try:
            os.symlink(src.resolve(), dst)
            return
        except (OSError, NotImplementedError):
            pass  # fall back to copy (Windows w/o privilege, etc.)
    dst.write_bytes(src.read_bytes())


def _write_lines(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""),
                    encoding="utf-8")


# ---- main ----------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path,
                   default=Path("data/processed/samples"),
                   help="dir of <sample_id>.json (default "
                        "data/processed/samples).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="filtered dataset root (samples/ + splits/).")
    p.add_argument("--max-rna-length", type=int, default=100,
                   help="keep samples with RNA length <= N (default 100).")
    p.add_argument("--min-rna-length", type=int, default=20,
                   help="keep samples with RNA length >= N (default 20; "
                        "0 = no lower bound).")
    p.add_argument("--splits-json", type=Path,
                   default=Path("data/processed/splits.json"),
                   help="source of the original cluster split.")
    p.add_argument("--train-sample-n", type=int, default=200)
    p.add_argument("--test-sample-n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--symlink", action="store_true",
                   help="symlink sample json instead of copying "
                        "(falls back to copy if unsupported).")
    p.add_argument("--dry-run", action="store_true",
                   help="compute + print stats, write nothing.")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.max_rna_length <= 0:
        print(f"ERROR: --max-rna-length must be > 0 "
              f"(got {args.max_rna_length})", file=sys.stderr)
        return 1
    if args.min_rna_length < 0:
        print(f"ERROR: --min-rna-length must be >= 0 "
              f"(got {args.min_rna_length})", file=sys.stderr)
        return 1
    if args.min_rna_length > args.max_rna_length:
        print(f"ERROR: --min-rna-length ({args.min_rna_length}) > "
              f"--max-rna-length ({args.max_rna_length})",
              file=sys.stderr)
        return 1
    if not args.input_dir.is_dir():
        print(f"ERROR: --input-dir not a directory: {args.input_dir}",
              file=sys.stderr)
        return 1

    splits_map = load_splits_map(args.splits_json)
    if not splits_map:
        print(f"WARN: no splits loaded from {args.splits_json}; every "
              f"kept sample will land in 'unknown'.", file=sys.stderr)

    res = filter_samples(args.input_dir, splits_map,
                         args.max_rna_length, args.min_rna_length)
    by_split = res["by_split"]
    counts = res["counts"]

    train_200 = sample_subset(by_split["train"],
                              args.train_sample_n, args.seed)
    test_200 = sample_subset(by_split["test"],
                             args.test_sample_n, args.seed)

    if not args.dry_run:
        samples_out = args.output_dir / "samples"
        splits_out = args.output_dir / "splits"
        for sid, src, _rlen, _b in res["kept"]:
            _materialise(src, samples_out / f"{sid}.json", args.symlink)
        for name in ("train", "val", "test"):
            _write_lines(splits_out / f"{name}.txt",
                         sorted(by_split[name]))
        _write_lines(splits_out / "train_200.txt", train_200)
        _write_lines(splits_out / "test_200.txt", test_200)
        (args.output_dir / "filter_stats.json").write_text(
            json.dumps({
                "max_rna_length": args.max_rna_length,
                "min_rna_length": args.min_rna_length,
                "seed": args.seed,
                "counts": counts,
                "split_counts": {k: len(v) for k, v in by_split.items()},
                "train_200_n": len(train_200),
                "test_200_n": len(test_200),
                "rna_length_stats": _length_stats(res["rna_lens"]),
            }, indent=2), encoding="utf-8")

    st = _length_stats(res["rna_lens"])
    tag = "DRY-RUN (no writes)" if args.dry_run else "written"
    print(f"filter_samples [{tag}]  RNA length "
          f"{args.min_rna_length} <= L <= {args.max_rna_length}")
    print(f"  input dir            : {args.input_dir}")
    print(f"  scanned              : {counts['total_scanned']}")
    print(f"  kept                 : {counts['kept']}")
    print(f"  filtered (too long)  : {counts['filtered_too_long']}")
    print(f"  filtered (too short) : {counts['filtered_too_short']}")
    print(f"  no RNA length        : {counts['no_rna_length']}")
    if counts["bad_json"]:
        print(f"  unreadable json      : {counts['bad_json']}")
    print(f"  split: train={len(by_split['train'])}  "
          f"val={len(by_split['val'])}  test={len(by_split['test'])}  "
          f"unknown={len(by_split['unknown'])}")
    print(f"  sampled: train_200={len(train_200)}  "
          f"test_200={len(test_200)}  (seed={args.seed})")
    print(f"  kept RNA length: n={st['n']}  min={st['min']}  "
          f"max={st['max']}  mean={st['mean']}  median={st['median']}")
    if not args.dry_run:
        print(f"  output               : {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
