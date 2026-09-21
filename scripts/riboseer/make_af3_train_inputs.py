#!/usr/bin/env python3
"""Prepare AlphaFold 3 Server submission files for the TRAIN split.

AF3 Server is submitted by hand (paste one protein + one RNA sequence per
job on the web form) and caps at **30 jobs/day**, so 225 train samples
spread over **8 days** (30×7 + 15). This script extracts each sample's
protein / RNA sequence from its step-1 JSON, sanitises them with the same
rules as the test-split tool (``af3_inputs``: keep the 20 standard
amino acids / {A,U,G,C}, drop gaps / modified / ambiguous residues), and
lays everything out per day for copy-paste:

    <out-dir>/
      day1/
        01_<sample_id>.json  ← AlphaFold Server upload payload (proteinChain
                               + rnaSequence, dialect=alphafoldserver — same
                               schema as the test split's af3_jobs/)
        02_<sample_id>.json
        ...
        manifest.csv         ← sample_id, protein_name, protein_length, rna_length
      day2/ ...
      ...
      day8/

Drag a day's ``.json`` files into AlphaFold Server's file upload (≤30/day).

Day membership follows the **order in ``--sample-list``**: positions 1-30
→ day1, 31-60 → day2, … (the spec's fixed buckets), so the same id always
lands in the same day regardless of which samples were skipped. A sample
whose JSON is missing is recovered from the raw structure when ``--raw-dir``
is given; one whose protein/RNA cleans to empty (AF3 needs both) is left
out of the job files and flagged in the manifest with length 0, rather
than crashing the run.

Usage
-----
::

    python scripts/riboseer/make_af3_train_inputs.py \\
        --sample-list   data/processed_quality/splits_tmscore_035/train.txt \\
        --processed-dir data/processed_quality \\
        --out-dir       data/batch_train_v7/af3_inputs \\
        --per-day 30
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from step4_tool_adapters.external.af3_inputs import (  # noqa: E402
    STANDARD_AA, STANDARD_RNA, build_job_json, clean_sequence,
)
from step4_tool_adapters.tool_io import (  # noqa: E402
    clean_sample_id, load_sample_json, read_sample_list,
)
from scripts.compute_tmscore_matrix import (  # noqa: E402
    find_raw_cif, find_raw_pdb,
)


@dataclass
class SampleAF3:
    sample_id: str
    protein_name: str
    protein_seq: str
    rna_seq: str
    prot_len: int
    rna_len: int
    n_prot_removed: int
    n_rna_removed: int
    ok: bool          # both sequences non-empty after cleaning
    missing_json: bool = False


def _protein_name(sample: dict, sample_id: str) -> str:
    """Human-ish label: ``<source_pdb>_<protein.chain_id>`` when available,
    else the sample id."""
    pdb = sample.get("source_pdb") or sample_id.split("_")[0]
    chain = (sample.get("protein") or {}).get("chain_id") or ""
    return f"{pdb}_{chain}" if chain else str(pdb)


def _chain_one_letter(st, name: str) -> str:
    """One-letter polymer sequence for the chain whose ``name`` (mmCIF
    auth_asym_id) matches ``name``; '' if absent. gemmi handles protein
    and RNA polymers and multi-char chain ids (e.g. 'Ll')."""
    for model in st:
        for chain in model:
            if chain.name == name:
                return chain.get_polymer().make_one_letter_sequence() or ""
    return ""


def extract_seqs_from_raw(raw_dir: Path,
                          sample_id: str) -> Optional[tuple[str, str]]:
    """Recover ``(protein_seq, rna_seq)`` for a sample with no step-1 JSON
    by reading its raw structure. ``sample_id`` = ``<pdb>_<prot>_<rna>``;
    the two chain tokens are auth_asym_ids. Prefers mmCIF (multi-char
    chains live there), falls back to legacy PDB. Returns None if the
    structure can't be located / read."""
    parts = sample_id.split("_")
    if len(parts) < 3:
        return None
    pdb, prot_chain, rna_chain = parts[0], parts[1], parts[2]
    path = find_raw_cif(raw_dir, pdb)
    if path is None:
        try:
            path = find_raw_pdb(raw_dir, pdb)
        except FileNotFoundError:
            return None
    try:
        import gemmi  # lazy
        st = gemmi.read_structure(str(path))
        st.setup_entities()
    except Exception:  # noqa: BLE001 — unreadable structure → treat as missing
        return None
    return _chain_one_letter(st, prot_chain), _chain_one_letter(st, rna_chain)


def extract_sample(processed_dir: Path, sample_id: str,
                   raw_dir: Optional[Path] = None) -> SampleAF3:
    """Clean sequences for one sample. Prefers the step-1 JSON; when it's
    missing and ``raw_dir`` is given, falls back to reading the raw
    structure (full local PDB/mmCIF). Never raises."""
    samples_dir = processed_dir / "samples"
    prot = rna = ""
    name = sample_id
    from_raw = False
    try:
        sample = load_sample_json(samples_dir, sample_id)
        prot = (sample.get("protein") or {}).get("sequence") or ""
        rna = (sample.get("rna") or {}).get("sequence") or ""
        name = _protein_name(sample, sample_id)
    except FileNotFoundError:
        recovered = (extract_seqs_from_raw(raw_dir, sample_id)
                     if raw_dir is not None else None)
        if recovered is None:
            return SampleAF3(sample_id, sample_id, "", "", 0, 0, 0, 0,
                             ok=False, missing_json=True)
        prot, rna = recovered
        from_raw = True
        parts = sample_id.split("_")
        name = f"{parts[0]}_{parts[1]}" if len(parts) > 1 else sample_id

    cprot, n_prot = clean_sequence(prot, STANDARD_AA)
    crna, n_rna = clean_sequence(rna, STANDARD_RNA)
    return SampleAF3(
        sample_id=sample_id,
        protein_name=(name + " (from raw)") if from_raw else name,
        protein_seq=cprot, rna_seq=crna,
        prot_len=len(cprot), rna_len=len(crna),
        n_prot_removed=n_prot, n_rna_removed=n_rna,
        ok=bool(cprot and crna),
    )


_MANIFEST_FIELDS = ["sample_id", "protein_name", "protein_length", "rna_length"]


def write_manifest(path: Path, samples: list[SampleAF3]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_MANIFEST_FIELDS, lineterminator="\n")
        w.writeheader()
        for s in samples:
            w.writerow({
                "sample_id": s.sample_id,
                "protein_name": s.protein_name,
                "protein_length": s.prot_len,
                "rna_length": s.rna_len,
            })


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample-list", type=Path, required=True)
    p.add_argument("--processed-dir", type=Path,
                   default=Path("data/processed_quality"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("data/batch_train_v7/af3_inputs"))
    p.add_argument("--per-day", type=int, default=30,
                   help="jobs per day (AF3 Server cap; default 30)")
    p.add_argument("--raw-dir", type=Path, default=None,
                   help="raw PDB/mmCIF dir; used to recover sequences for "
                        "samples whose step-1 JSON is missing")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    per_day = max(1, args.per_day)
    sample_ids = [clean_sample_id(s) for s in read_sample_list(args.sample_list)]
    if not sample_ids:
        print("ERROR: empty sample list", file=sys.stderr)
        return 1

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    by_day: dict[int, list[SampleAF3]] = {}
    n_ok = n_missing = n_empty = 0
    for pos, sid in enumerate(sample_ids):          # 0-based position
        day = pos // per_day + 1
        s = extract_sample(args.processed_dir, sid, raw_dir=args.raw_dir)
        by_day.setdefault(day, []).append(s)
        if s.missing_json:
            n_missing += 1
            print(f"WARNING: no sample JSON for {sid} (day{day}); "
                  f"no job file written", file=sys.stderr)
        elif not s.ok:
            n_empty += 1
            print(f"WARNING: {sid} empty after cleaning "
                  f"(prot={s.prot_len}, rna={s.rna_len}); no job file written",
                  file=sys.stderr)
        else:
            n_ok += 1

    n_days = max(by_day) if by_day else 0
    total_jobs = 0
    for day in range(1, n_days + 1):
        members = by_day.get(day, [])
        day_dir = out_dir / f"day{day}"
        day_dir.mkdir(parents=True, exist_ok=True)
        # per-day AF3 Server JSON payloads (only submittable ones),
        # ordered by position. proteinChain + rnaSequence, dialect=
        # alphafoldserver — same schema as the test split's af3_jobs/.
        for i, s in enumerate(members, start=1):
            if not s.ok:
                continue
            (day_dir / f"{i:02d}_{s.sample_id}.json").write_text(
                json.dumps(build_job_json(s), indent=2, ensure_ascii=False),
                encoding="utf-8")
            total_jobs += 1
        # manifest lists the full day roster (incl. skipped, length 0)
        write_manifest(day_dir / "manifest.csv", members)
        n_submittable = sum(1 for s in members if s.ok)
        print(f"day{day}: {len(members)} samples, {n_submittable} submittable "
              f"-> {day_dir}")

    print()
    print(f"samples total : {len(sample_ids)}")
    print(f"submittable    : {n_ok}  (job files: {total_jobs})")
    print(f"missing JSON   : {n_missing}")
    print(f"empty-after-clean: {n_empty}")
    print(f"days           : {n_days}  (cap {per_day}/day)")
    print(f"out dir        : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
