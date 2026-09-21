"""Sanitize the AlphaFold-3 submission TSV.

Strips non-standard characters from the protein / RNA sequences in
``af3_inputs.tsv`` (gaps, dots, whitespace, selenocysteine, modified
RNA bases, ``N`` ambiguity codes — anything not in the 20 standard
amino acids or {A, U, G, C}) and emits three artefacts:

1. ``af3_inputs_clean.tsv`` — same 5-column TSV with cleaned sequences
   and re-computed lengths.
2. ``af3_jobs/<sample_id>.json`` — one AlphaFold Server-ready JSON per
   sample (``proteinChain`` + ``rnaChain`` + ``modelSeeds=[42]``).
3. ``af3_submission_order.csv`` — every sample sorted by
   ``prot_len + rna_len`` ascending, with a 1-based ``priority``
   column so the short, cheap, high-success-rate complexes go first.

Stats printed on stderr include how many samples were modified and how
many characters were stripped per side.

Usage
-----
::

    python -m step4_tool_adapters.external.af3_inputs.py \\
        --input data/af3_inputs.tsv \\
        --output-dir data/batch_test_v7/

Output filenames inside ``--output-dir`` are fixed: ``af3_inputs_clean.tsv``,
``af3_jobs/`` (subdir), ``af3_submission_order.csv``. The dir is created
if missing.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# What counts as "standard". Anything outside these sets gets dropped.
# Selenocysteine (U), pyrrolysine (O), and the ambiguity codes
# (B/J/X/Z) all fall out — AlphaFold Server doesn't accept them.
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
STANDARD_RNA = set("AUGC")

MODEL_SEEDS = [42]   # AF3 Server JSON wants a list

# AlphaFold Server batch input is a top-level *array* of job objects.
# Each object carries ``dialect="alphafoldserver"`` + ``version=1``.
# Source of truth: ``Input.from_alphafoldserver_fold_job`` in
# google-deepmind/alphafold3/src/alphafold3/common/folding_input.py.
#
# Field-name gotcha: protein chains use ``proteinChain`` but
# **RNA uses ``rnaSequence``** (not ``rnaChain`` as one might guess
# from the symmetry — that's the bug AF Server signalled with
# "No jobs found in file").
AF_SERVER_DIALECT = "alphafoldserver"
AF_SERVER_VERSION = 1


# ---- core cleaning ------------------------------------------------------


@dataclass
class CleanedRow:
    sample_id: str
    protein_seq: str
    rna_seq: str
    prot_len: int
    rna_len: int
    n_prot_removed: int
    n_rna_removed: int
    modified: bool


def clean_sequence(seq: str, alphabet: set[str]) -> tuple[str, int]:
    """Strip any character not in ``alphabet``. Uppercases on entry so
    lowercase residue letters (rare but legal in some FASTA exports)
    are kept. Returns ``(cleaned, n_removed)``."""
    if not seq:
        return "", 0
    up = seq.upper()
    kept = [c for c in up if c in alphabet]
    return "".join(kept), len(up) - len(kept)


def clean_row(sample_id: str, protein_seq: str,
              rna_seq: str) -> CleanedRow:
    """Apply ``clean_sequence`` to both sides; mark ``modified`` if
    either lost characters or the original input wasn't already
    uppercase + standard-only."""
    new_prot, n_prot = clean_sequence(protein_seq, STANDARD_AA)
    new_rna, n_rna = clean_sequence(rna_seq, STANDARD_RNA)
    modified = (n_prot > 0 or n_rna > 0
                or new_prot != protein_seq
                or new_rna != rna_seq)
    return CleanedRow(
        sample_id=sample_id,
        protein_seq=new_prot,
        rna_seq=new_rna,
        prot_len=len(new_prot),
        rna_len=len(new_rna),
        n_prot_removed=n_prot,
        n_rna_removed=n_rna,
        modified=modified,
    )


# ---- IO -----------------------------------------------------------------


_TSV_FIELDS = ["sample_id", "protein_seq", "rna_seq",
               "prot_len", "rna_len"]


def read_input_tsv(path: Path) -> list[dict]:
    """Reads the AF3 input TSV. Header is required and must contain
    sample_id / protein_seq / rna_seq (lengths are recomputed so the
    on-disk values are ignored).

    Raises ValueError if the header is missing required columns —
    fail loud rather than silently producing empty cleaned rows.
    """
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing = {"sample_id", "protein_seq", "rna_seq"} - set(
            reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path}: missing required columns {sorted(missing)} "
                f"(saw {reader.fieldnames})")
        for r in reader:
            rows.append({
                "sample_id": (r.get("sample_id") or "").strip(),
                "protein_seq": r.get("protein_seq") or "",
                "rna_seq": r.get("rna_seq") or "",
            })
    return rows


def write_clean_tsv(path: Path, rows: list[CleanedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_TSV_FIELDS, delimiter="\t",
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({
                "sample_id": r.sample_id,
                "protein_seq": r.protein_seq,
                "rna_seq": r.rna_seq,
                "prot_len": r.prot_len,
                "rna_len": r.rna_len,
            })


# ---- AF3 Server JSON ---------------------------------------------------


def build_job_object(row: CleanedRow,
                     model_seeds: Optional[list[int]] = None) -> dict:
    """One AlphaFold Server job object (the *inner* dict — the
    serialised file wraps a list of these). Schema:

    - ``name``: human-readable id; we use ``sample_id``.
    - ``modelSeeds``: non-empty list → deterministic; empty list →
      AF Server picks a random seed. Default ``[42]``.
    - ``sequences``: list of chain dicts. Protein uses
      ``proteinChain``; RNA uses ``rnaSequence`` (NOT ``rnaChain`` —
      that was the bug behind "No jobs found in file").
    - ``dialect`` / ``version``: optional but emitted explicitly so
      a future schema bump on AF Server's end fails loudly here
      rather than silently mis-parsing."""
    return {
        "name": row.sample_id,
        "modelSeeds": list(model_seeds or MODEL_SEEDS),
        "sequences": [
            {"proteinChain": {"sequence": row.protein_seq,
                              "count": 1}},
            {"rnaSequence": {"sequence": row.rna_seq, "count": 1}},
        ],
        "dialect": AF_SERVER_DIALECT,
        "version": AF_SERVER_VERSION,
    }


def build_job_json(row: CleanedRow,
                   model_seeds: Optional[list[int]] = None) -> list[dict]:
    """The on-disk JSON payload for one sample: a one-element list
    holding the job object. AlphaFold Server's batch parser keys off
    "top-level is a list" to detect the alphafoldserver dialect, so
    we always emit the list — even for single-job files."""
    return [build_job_object(row, model_seeds=model_seeds)]


def write_job_jsons(dir_path: Path,
                    rows: list[CleanedRow]) -> int:
    """Writes one ``<sample_id>.json`` per row. Empty-sequence rows
    (cleaned away to nothing) are skipped — AlphaFold Server would
    reject them anyway. Returns the count actually written."""
    dir_path.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in rows:
        if not r.protein_seq or not r.rna_seq:
            continue
        payload = build_job_json(r)
        (dir_path / f"{r.sample_id}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8")
        n += 1
    return n


# ---- submission order ---------------------------------------------------


_ORDER_COLUMNS = ["priority", "sample_id", "prot_len", "rna_len",
                  "total_len"]


def build_submission_order(rows: list[CleanedRow]) -> list[dict]:
    """Sort ascending by ``total_len`` (shortest complex first → fastest
    folding, lowest OOM risk, highest success rate). Tie-break by
    ``prot_len`` then ``sample_id`` so the ordering is deterministic
    across reruns. ``priority`` is 1-based."""
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda it: (
        it[1].prot_len + it[1].rna_len,
        it[1].prot_len,
        it[1].sample_id,
    ))
    out: list[dict] = []
    for prio, (_orig_idx, r) in enumerate(indexed, start=1):
        out.append({
            "priority": prio,
            "sample_id": r.sample_id,
            "prot_len": r.prot_len,
            "rna_len": r.rna_len,
            "total_len": r.prot_len + r.rna_len,
        })
    return out


def write_submission_order(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_ORDER_COLUMNS,
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---- stats --------------------------------------------------------------


def summarise(rows: list[CleanedRow]) -> dict:
    n_modified = sum(1 for r in rows if r.modified)
    n_prot_removed = sum(r.n_prot_removed for r in rows)
    n_rna_removed = sum(r.n_rna_removed for r in rows)
    empty_after = sum(1 for r in rows
                      if not r.protein_seq or not r.rna_seq)
    return {
        "n_total": len(rows),
        "n_modified": n_modified,
        "n_prot_chars_removed": n_prot_removed,
        "n_rna_chars_removed": n_rna_removed,
        "n_empty_after_clean": empty_after,
    }


# ---- main ---------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True,
                   help="path to af3_inputs.tsv")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="output directory (default: same as --input). "
                        "Three artefacts are written into this dir "
                        "with fixed names — see module docstring.")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not args.input.is_file():
        print(f"ERROR: --input not a file: {args.input}",
              file=sys.stderr)
        return 1

    out_dir = args.output_dir or args.input.parent
    try:
        raw = read_input_tsv(args.input)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    cleaned = [clean_row(r["sample_id"], r["protein_seq"], r["rna_seq"])
               for r in raw]

    clean_tsv = out_dir / "af3_inputs_clean.tsv"
    jobs_dir = out_dir / "af3_jobs"
    order_csv = out_dir / "af3_submission_order.csv"

    write_clean_tsv(clean_tsv, cleaned)
    n_jobs = write_job_jsons(jobs_dir, cleaned)
    order_rows = build_submission_order(cleaned)
    write_submission_order(order_csv, order_rows)

    stats = summarise(cleaned)
    print(f"input  : {args.input}")
    print(f"samples: {stats['n_total']}")
    print(f"modified: {stats['n_modified']}  "
          f"(prot chars removed: {stats['n_prot_chars_removed']}, "
          f"rna chars removed: {stats['n_rna_chars_removed']})")
    if stats["n_empty_after_clean"]:
        print(f"WARN: {stats['n_empty_after_clean']} samples became "
              f"empty after cleaning — skipped from job JSONs.",
              file=sys.stderr)
    print(f"wrote {clean_tsv}")
    print(f"wrote {jobs_dir}/   ({n_jobs} JSON jobs)")
    print(f"wrote {order_csv}   (sorted by total length asc)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
