#!/usr/bin/env python3
"""Build the deduplicated PDB-ID list for a BindUP batch-mode submission.

BindUP's batch mode takes a plain list of PDB IDs. From the 107-sample test
split this script extracts the unique PDB IDs (the token before the first ``_``
in each ``sample_id``) and also writes a ``pdb_id, protein_chain, sample_id``
map so the per-residue BindUP output can later be tied back to each RiboSeer
sample (one PDB+chain can back several samples — different RNA partners share
the same protein chain).

The protein chain is read from the sample JSON's ``protein.chain_id`` when
available (authoritative, handles multi-character chains), falling back to the
second ``_``-delimited token of the ``sample_id``.

Outputs
-------
* ``--out-ids``  : one PDB ID per line, sorted, deduplicated.
* ``--out-map``  : CSV ``pdb_id,protein_chain,sample_id`` sorted by all three.

Usage
-----
::

    python -m step4_tool_adapters.external.bindup_inputs.py \
        --sample-list  data/processed_quality/splits_tmscore_035/test.txt \
        --processed-dir data/processed_quality \
        --out-ids      data/batch_test_v7/bindup_pdb_ids.txt \
        --out-map      data/batch_test_v7/bindup_sample_map.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

_INVISIBLE = "﻿​‌‍‎‏⁠"


def clean_id(s: str) -> str:
    return s.strip().strip(_INVISIBLE).strip()


def read_sample_list(path: Path) -> list[str]:
    """Read sample ids from a ``.txt`` (one id/line) or ``.csv`` (``sample_id``
    column). Blanks / ``#`` comments skipped; ids cleaned of invisible marks."""
    if not path.is_file():
        raise FileNotFoundError(f"sample-list not found: {path}")
    if path.suffix.lower() == ".csv":
        out: list[str] = []
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "sample_id" not in reader.fieldnames:
                raise ValueError(f"{path} has no 'sample_id' column")
            for row in reader:
                sid = clean_id(row.get("sample_id") or "")
                if sid:
                    out.append(sid)
        return out
    out = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        s = clean_id(line)
        if s and not s.startswith("#"):
            out.append(clean_id(s.split()[0]))
    return out


def protein_chain(samples_dir: Optional[Path], sample_id: str) -> str:
    """Protein chain for a sample: prefer sample JSON ``protein.chain_id``,
    else the 2nd ``_``-token of the id, else ``""``."""
    if samples_dir is not None:
        for cand in (samples_dir / f"{sample_id}.json",):
            if cand.is_file():
                try:
                    d = json.loads(cand.read_text(encoding="utf-8"))
                    cid = (d.get("protein") or {}).get("chain_id")
                    if cid:
                        return str(cid)
                except Exception:  # noqa: BLE001
                    break
        # case-insensitive fallback scan
        target = f"{sample_id}.json".lower()
        if samples_dir.is_dir():
            for child in samples_dir.iterdir():
                if child.is_file() and child.name.lower() == target:
                    try:
                        d = json.loads(child.read_text(encoding="utf-8"))
                        cid = (d.get("protein") or {}).get("chain_id")
                        if cid:
                            return str(cid)
                    except Exception:  # noqa: BLE001
                        pass
                    break
    parts = sample_id.split("_")
    return parts[1] if len(parts) > 1 else ""


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sample-list", type=Path,
                   default=Path("data/processed_quality/splits_tmscore_035/test.txt"),
                   help="txt (one id/line) or csv (sample_id column)")
    p.add_argument("--processed-dir", type=Path,
                   default=Path("data/processed_quality"),
                   help="step-1 processed dir (its samples/ gives chain ids)")
    p.add_argument("--samples-dir", type=Path, default=None,
                   help="override sample-JSON dir (default <processed-dir>/samples)")
    p.add_argument("--out-ids", type=Path,
                   default=Path("data/batch_test_v7/bindup_pdb_ids.txt"))
    p.add_argument("--out-map", type=Path,
                   default=Path("data/batch_test_v7/bindup_sample_map.csv"))
    args = p.parse_args(argv)

    samples_dir = args.samples_dir or (args.processed_dir / "samples")
    if not samples_dir.is_dir():
        samples_dir = None  # chain falls back to sample_id parsing

    sample_ids = read_sample_list(args.sample_list)

    rows = []  # (pdb_id, protein_chain, sample_id)
    for sid in sample_ids:
        pdb_id = sid.split("_")[0]
        chain = protein_chain(samples_dir, sid)
        rows.append((pdb_id, chain, sid))

    pdb_ids = sorted({r[0] for r in rows})
    pdb_chain = sorted({(r[0], r[1]) for r in rows})
    rows.sort()

    args.out_ids.parent.mkdir(parents=True, exist_ok=True)
    args.out_ids.write_text("\n".join(pdb_ids) + "\n", encoding="utf-8")

    args.out_map.parent.mkdir(parents=True, exist_ok=True)
    with args.out_map.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pdb_id", "protein_chain", "sample_id"])
        w.writerows(rows)

    print(f"samples read            : {len(sample_ids)}")
    print(f"unique PDB IDs          : {len(pdb_ids)}")
    print(f"unique PDB+chain combos : {len(pdb_chain)}")
    print(f"pdb id list  -> {args.out_ids}")
    print(f"sample map   -> {args.out_map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
