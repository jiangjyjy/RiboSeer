#!/usr/bin/env python3
"""Build the FASTA input (and submission batches) for RNABindRPlus.

RNABindRPlus (http://ailab-projects2.ist.psu.edu/RNABindRPlus/) is a
sequence-based RNA-binding-residue predictor that takes one or more protein
sequences in FASTA and emails per-residue results.

Index alignment (important)
---------------------------
A sample's ``protein.sequence`` is the **full 1-based polymer sequence** with
unresolved residues written as ``-`` (so ``len(sequence) == protein.length`` and
a position in it *is* the ground-truth label_seq index). RNABindRPlus needs a
gap-free sequence of standard amino acids, so we drop ``-`` and any non-standard
residue — which **renumbers** the submitted sequence. To undo that at parse
time we record, per sample, the original 1-based position of every kept residue
in ``rnabindrplus_index_map.json``: submitted residue ``k`` (1-based) came from
original position ``kept_positions[k-1]``. (Verified: the kept positions equal
``protein.resolved_residues``.)

Outputs (under ``--out-dir``, default ``data/batch_test_v7``)
-------------------------------------------------------------
* ``rnabindrplus_input.fasta``          — all samples
* ``rnabindrplus_input_batch{N}.fasta`` — chunks of ``--batch-size`` (default 30;
  107 samples -> 30/30/30/17) so the ~10 min/seq runtime can be spread over
  several jobs
* ``rnabindrplus_index_map.json``       — submitted -> original position map

Usage
-----
::

    python -m step4_tool_adapters.external.rnabindrplus_inputs.py \
        --sample-list   data/processed_quality/splits_tmscore_035/test.txt \
        --processed-dir data/processed_quality \
        --out-dir       data/batch_test_v7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from step4_tool_adapters.tool_io import (  # noqa: E402
    clean_sample_id,
    load_sample_json,
    read_sample_list,
)

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq: str):
    # type: (str) -> tuple
    """Drop gaps / non-standard residues from a 1-based ``protein.sequence``.

    Returns ``(clean_seq, kept_positions)`` where ``kept_positions[k]`` is the
    original 1-based position of the ``k``-th kept residue."""
    clean = []
    kept = []
    for i, ch in enumerate(seq.upper(), start=1):
        if ch in STANDARD_AA:
            clean.append(ch)
            kept.append(i)
    return "".join(clean), kept


def wrap(seq: str, width: int = 60) -> str:
    if width and width > 0:
        return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))
    return seq


def write_fasta(path: Path, records) -> None:
    """``records`` = list of ``(sample_id, clean_seq)``."""
    lines = []
    for sid, seq in records:
        lines.append(">" + sid)
        lines.append(wrap(seq))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sample-list", type=Path,
                   default=Path("data/processed_quality/splits_tmscore_035/test.txt"),
                   help="txt (one id/line) or csv (sample_id column)")
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed_quality"))
    p.add_argument("--samples-dir", type=Path, default=None,
                   help="override sample-JSON dir (default <processed-dir>/samples)")
    p.add_argument("--out-dir", type=Path, default=Path("data/batch_test_v7"))
    p.add_argument("--batch-size", type=int, default=30,
                   help="sequences per submission batch (default 30)")
    p.add_argument("--wrap", type=int, default=60,
                   help="FASTA line width (0 = single line)")
    args = p.parse_args(argv)

    samples_dir = args.samples_dir or (args.processed_dir / "samples")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = [clean_sample_id(s) for s in read_sample_list(args.sample_list)]

    records = []                 # (sample_id, clean_seq)
    index_map = {}               # sample_id -> {length, kept_positions, n_removed}
    n_with_removed = 0
    n_empty = 0
    n_missing = 0
    missing_ids = []
    total_res = 0
    for sid in sample_ids:
        try:
            sample = load_sample_json(samples_dir, sid)
        except FileNotFoundError:
            # Local data is a subset; skip-and-warn instead of crashing so
            # this runs on whatever sample JSONs are present (the full set
            # lives on the server). Missing ids are reported for follow-up.
            n_missing += 1
            missing_ids.append(sid)
            print("WARNING: no sample JSON for {}; skipped".format(sid),
                  file=sys.stderr)
            continue
        seq = (sample.get("protein") or {}).get("sequence") or ""
        clean, kept = clean_sequence(seq)
        if not clean:
            n_empty += 1
            print("WARNING: {} has no standard residues after cleaning; skipped"
                  .format(sid), file=sys.stderr)
            continue
        n_removed = len(seq) - len(clean)
        if n_removed:
            n_with_removed += 1
        total_res += len(clean)
        records.append((sid, clean))
        index_map[sid] = {
            "length": len(seq),
            "submitted_len": len(clean),
            "n_removed": n_removed,
            "kept_positions": kept,   # submitted pos k (1-based) -> kept[k-1]
        }

    # full + batches
    full = out_dir / "rnabindrplus_input.fasta"
    write_fasta(full, records)

    bs = max(1, args.batch_size)
    n_batches = (len(records) + bs - 1) // bs
    batch_files = []
    for b in range(n_batches):
        chunk = records[b * bs:(b + 1) * bs]
        bf = out_dir / "rnabindrplus_input_batch{}.fasta".format(b + 1)
        write_fasta(bf, chunk)
        batch_files.append((bf, len(chunk)))

    map_path = out_dir / "rnabindrplus_index_map.json"
    map_path.write_text(json.dumps(index_map, indent=2), encoding="utf-8")

    print("samples submitted     : {}".format(len(records)))
    print("samples skipped empty : {}".format(n_empty))
    print("samples missing JSON  : {}".format(n_missing))
    if missing_ids:
        print("  missing ids: {}".format(" ".join(missing_ids)))
    print("samples with residues removed (gaps/non-std): {}".format(n_with_removed))
    print("total residues         : {}".format(total_res))
    print("full fasta -> {}".format(full))
    for bf, n in batch_files:
        print("  {} ({} seqs)".format(bf, n))
    print("index map  -> {}".format(map_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
