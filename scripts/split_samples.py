"""Emit per-split sample-id lists from ``data/processed/splits.json``.

Step 1 already does the train / val / test cluster split (and writes
the result into ``splits.json``); this script just materialises that
information as plain-text files so downstream batch scripts
(``batch_predict.py``, ``evaluate.py``) can consume one sample id
per line.

Outputs (under ``--output-dir``):

  - ``train.txt``     all train samples
  - ``val.txt``       all val samples
  - ``test.txt``      all test samples
  - ``train_<N>.txt``  random subset of N train samples (seeded)

Samples whose ``split`` is ``None`` (122 unclusterable in the current
dataset) are skipped — they're never used for training or evaluation.

Why a separate file per split
-----------------------------
``batch_predict.py`` is path-driven (``--sample-list <txt>``). Plain
text keeps the filter trivial, makes ``--start N --end M`` slicing
deterministic, and is easy to ``head -100`` for debugging.

Reproducibility
---------------
``random.Random(args.seed)`` (default seed=42) is the ONLY source of
randomness. Same input + same seed → byte-identical
``train_<N>.txt``. Each split's sample ids are sorted ascending in the
output so re-running with the same seed but a different
``splits.json`` ordering still produces the same file.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional


# ---------- IO -------------------------------------------------------------


def load_splits(splits_path: Path) -> dict[str, list[str]]:
    """Read step-1 splits.json → ``{"train": [...], "val": [...], ...}``.

    Skips samples with ``split == None`` (the 122 unclusterable rows).
    Sample ids inside each list are sorted ascending so the output is
    independent of dict ordering in the source JSON.
    """
    if not splits_path.is_file():
        raise FileNotFoundError(
            f"splits.json not found at {splits_path}. "
            f"Run step 1 (mmseqs cluster split) first."
        )
    with splits_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples") or {}
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    n_skipped = 0
    for sid, entry in samples.items():
        split = (entry or {}).get("split")
        if split in out:
            out[split].append(sid)
        else:
            # split is None / unclusterable / unknown
            n_skipped += 1
    for k in out:
        out[k].sort()
    return out, n_skipped


def build_sample_index(samples_dir: Path) -> dict[str, str]:
    """Return ``{lowercase_stem: actual_stem}`` for every JSON in samples/.

    Used to align splits.json ids with the on-disk filenames: splits.json
    occasionally carries an id whose case differs from the actual file
    (e.g. ``3j46_y_1`` vs ``3j46_Y_1.json``). Returns an empty dict when
    ``samples_dir`` doesn't exist — the caller treats this as "no
    validation possible, leave ids unchanged".
    """
    if not samples_dir.is_dir():
        return {}
    index: dict[str, str] = {}
    for f in samples_dir.iterdir():
        if f.suffix == ".json":
            index[f.stem.lower()] = f.stem
    return index


def validate_sample_ids(
    sample_ids: list[str],
    sample_index: dict[str, str],
) -> tuple[list[str], int]:
    """Filter ``sample_ids`` against ``sample_index``.

    Returns ``(canonical_ids, skipped_count)`` where canonical_ids are
    the actual on-disk filenames (stems) for ids that exist, with the
    case as it appears in the file system. Ids with no matching file are
    dropped. When the index is empty (e.g. samples_dir absent), passes
    sample_ids through unchanged with ``skipped_count=0``.
    """
    if not sample_index:
        return list(sample_ids), 0
    canonical: list[str] = []
    skipped = 0
    for sid in sample_ids:
        actual = sample_index.get(sid.lower())
        if actual is None:
            skipped += 1
            continue
        canonical.append(actual)
    return canonical, skipped


def write_list(path: Path, ids: list[str]) -> None:
    """Atomic write of a per-split id list.

    One id per line, trailing newline. ``.tmp + replace`` so a SIGINT
    mid-write can't leave a half-written file that downstream scripts
    would read as truncated.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    body = "\n".join(ids) + ("\n" if ids else "")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


# ---------- subsampling ----------------------------------------------------


def subsample_train(
    train_ids: list[str], n: int, seed: int,
) -> list[str]:
    """Pick ``n`` ids at random (without replacement, seeded).

    When ``n >= len(train_ids)``, returns the full sorted train set.
    Output order is sorted so the file is diff-friendly across runs
    that use the same seed (random.sample's order depends on input
    order which we already canonicalised in ``load_splits``).
    """
    if n <= 0:
        return []
    rng = random.Random(seed)
    if n >= len(train_ids):
        return list(train_ids)
    chosen = rng.sample(train_ids, n)
    return sorted(chosen)


# ---------- main -----------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed-dir", type=Path,
                   default=Path("data/processed"),
                   help="step 1 output dir (containing splits.json)")
    p.add_argument("--splits-json", type=Path, default=None,
                   help="path to splits.json; default: "
                        "<processed-dir>/splits.json")
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/splits"),
                   help="where train.txt / val.txt / test.txt land")
    p.add_argument("--train-subset", type=int, default=500,
                   help="size of the seeded train subsample (default 500)")
    p.add_argument("--seed", type=int, default=42,
                   help="random seed for the subsample (default 42)")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    splits_path = args.splits_json or (args.processed_dir / "splits.json")
    try:
        splits, n_skipped = load_splits(splits_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Reconcile splits.json ids with the actual ``samples/<id>.json``
    # filenames. splits.json sometimes stores an id whose case differs
    # from the filename (e.g. ``3j46_y_1`` vs ``3j46_Y_1.json``); the
    # downstream batch_predict will fail on those. We canonicalise to the
    # on-disk casing and drop ids with no matching file.
    sample_index = build_sample_index(args.processed_dir / "samples")
    n_missing_total = 0
    for split_name in ("train", "val", "test"):
        canonical, n_missing = validate_sample_ids(
            splits[split_name], sample_index,
        )
        splits[split_name] = sorted(canonical)
        n_missing_total += n_missing

    write_list(args.output_dir / "train.txt", splits["train"])
    write_list(args.output_dir / "val.txt", splits["val"])
    write_list(args.output_dir / "test.txt", splits["test"])

    train_subset_ids = subsample_train(
        splits["train"], args.train_subset, args.seed,
    )
    train_subset_path = args.output_dir / f"train_{args.train_subset}.txt"
    write_list(train_subset_path, train_subset_ids)

    # ---- summary --------------------------------------------------------
    print(f"train: {len(splits['train']):>6} samples")
    print(f"val:   {len(splits['val']):>6} samples")
    print(f"test:  {len(splits['test']):>6} samples")
    print(f"train_{args.train_subset}: "
          f"{len(train_subset_ids):>6} samples (seed={args.seed})")
    if n_skipped:
        print(f"skipped: {n_skipped} unclusterable (split=None)")
    if n_missing_total:
        print(f"skipped: {n_missing_total} ids missing from "
              f"{args.processed_dir / 'samples'}")
    print()
    print(f"wrote: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
