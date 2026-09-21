#!/usr/bin/env python3
"""Classify the test split's RNA + protein sequences by family.

For each test sample:

* **RNA**     → Infernal ``cmsearch`` vs the Rfam covariance-model database
  → Rfam family (e.g. ``tRNA`` / ``5S_rRNA`` / ``SSU_rRNA_eukarya``).
* **protein** → HMMER ``hmmscan`` vs the Pfam-A HMM database → Pfam domain
  (e.g. ``RRM_1`` / ``KH_1`` / ``zf-CCCH``).

Pipeline (three modes)
----------------------
* ``--extract-only`` — only write the two FASTA files from the sample JSONs.
* default (full)     — extract → run ``cmsearch`` + ``hmmscan`` → parse →
  write the summary CSV. Needs the binaries + pressed databases.
* ``--parse-only``   — parse existing ``*_hits.tbl`` files already in
  ``--output-dir`` → write the summary CSV (no extraction, no tool run).

All artifacts live under ``--output-dir`` with fixed names::

    rna_sequences.fasta        protein_sequences.fasta
    rna_rfam_hits.tbl          protein_pfam_hits.tbl
    classification_summary.csv

Best hit per sequence = the lowest E-value; a sample whose best E-value is
above the threshold (``--rna-evalue`` / ``--protein-evalue``, default 0.01)
or has no hit at all is reported ``unclassified``.

Environment (server)
--------------------
::

    conda install -c bioconda infernal hmmer    # cmsearch + hmmscan
    # Rfam CM:
    wget https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz
    gunzip Rfam.cm.gz && cmpress Rfam.cm
    # Pfam-A HMM:
    wget https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz
    gunzip Pfam-A.hmm.gz && hmmpress Pfam-A.hmm

Usage
-----
::

    python scripts/riboseer/run_cmsearch_classification.py \\
        --processed-dir data/processed_quality \\
        --split-file    data/processed_quality/splits_tmscore_035/test.txt \\
        --rfam-cm       /path/to/Rfam.cm \\
        --pfam-hmm      /path/to/Pfam-A.hmm \\
        --output-dir    data/batch_test_v7/classification \\
        --cpu 4
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json  # noqa: E402

# Fixed artifact names under --output-dir.
RNA_FASTA = "rna_sequences.fasta"
PROTEIN_FASTA = "protein_sequences.fasta"
RNA_TBL = "rna_rfam_hits.tbl"
PROTEIN_TBL = "protein_pfam_hits.tbl"
SUMMARY_CSV = "classification_summary.csv"

UNCLASSIFIED = "unclassified"


# ---------------------------------------------------------------------------
# Sample IO
# ---------------------------------------------------------------------------


def load_sample_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"split file not found: {path}")
    return [ln.split()[0] for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def load_sample(processed_dir: Path, sid: str) -> Optional[dict]:
    for cand in (processed_dir / "samples" / f"{sid}.json",
                 processed_dir / f"{sid}.json"):
        if cand.is_file():
            try:
                return json.loads(cand.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
    return None


def _sanitize(seq: str) -> str:
    """Uppercase and keep only letters — drops gap/placeholder characters
    (``-`` / ``.`` / ``*``), digits and whitespace that the JSON sequence may
    carry for unresolved residues, which cmsearch / hmmscan would choke on."""
    return "".join(c for c in (seq or "").upper() if c.isalpha())


def _wrap(seq: str, width: int = 60) -> str:
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def extract_fastas(processed_dir: Path, sample_ids: list[str], out_dir: Path
                   ) -> tuple[int, int]:
    """Write RNA + protein FASTA files (header = sample_id). Returns the
    number of RNA and protein records written. Empty sequences are skipped
    with a warning."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rna_lines: list[str] = []
    prot_lines: list[str] = []
    n_rna = n_prot = 0
    for sid in sample_ids:
        sample = load_sample(processed_dir, sid)
        if sample is None:
            print(f"  WARN: no sample JSON for {sid}", file=sys.stderr)
            continue
        rna = sample.get("rna") or sample.get("ligand_rna") or {}
        prot = sample.get("protein") or {}
        rseq = _sanitize(rna.get("sequence") or "")
        pseq = _sanitize(prot.get("sequence") or "")
        if rseq:
            rna_lines.append(f">{sid}\n{_wrap(rseq)}")
            n_rna += 1
        else:
            print(f"  WARN: empty RNA sequence for {sid}", file=sys.stderr)
        if pseq:
            prot_lines.append(f">{sid}\n{_wrap(pseq)}")
            n_prot += 1
        else:
            print(f"  WARN: empty protein sequence for {sid}", file=sys.stderr)
    (out_dir / RNA_FASTA).write_text("\n".join(rna_lines) + "\n",
                                     encoding="utf-8")
    (out_dir / PROTEIN_FASTA).write_text("\n".join(prot_lines) + "\n",
                                         encoding="utf-8")
    return n_rna, n_prot


# ---------------------------------------------------------------------------
# Tool invocation
# ---------------------------------------------------------------------------


def _pressed_ok(db: Path, suffixes: tuple[str, ...]) -> bool:
    """Has the DB been cmpress/hmmpress'd? (index sidecar present)."""
    return any((db.parent / (db.name + s)).is_file() for s in suffixes)


def run_cmsearch(rfam_cm: Path, rna_fasta: Path, tbl_out: Path, cpu: int
                 ) -> bool:
    """``cmsearch --tblout <tbl> --noali --cpu N <Rfam.cm> <rna.fasta>``."""
    if shutil.which("cmsearch") is None:
        print("ERROR: cmsearch not found (conda install -c bioconda infernal)",
              file=sys.stderr)
        return False
    if not rfam_cm.is_file():
        print(f"ERROR: Rfam CM not found: {rfam_cm}", file=sys.stderr)
        return False
    if not _pressed_ok(rfam_cm, (".i1m", ".i1i", ".i1f", ".i1p")):
        print(f"ERROR: {rfam_cm} not cmpress'd (run: cmpress {rfam_cm})",
              file=sys.stderr)
        return False
    cmd = ["cmsearch", "--tblout", str(tbl_out), "--noali", "--cpu", str(cpu),
           str(rfam_cm), str(rna_fasta)]
    return _run(cmd, "cmsearch")


def run_hmmscan(pfam_hmm: Path, prot_fasta: Path, tbl_out: Path, cpu: int
                ) -> bool:
    """``hmmscan --tblout <tbl> --noali --cpu N <Pfam-A.hmm> <protein.fasta>``."""
    if shutil.which("hmmscan") is None:
        print("ERROR: hmmscan not found (conda install -c bioconda hmmer)",
              file=sys.stderr)
        return False
    if not pfam_hmm.is_file():
        print(f"ERROR: Pfam HMM not found: {pfam_hmm}", file=sys.stderr)
        return False
    if not _pressed_ok(pfam_hmm, (".h3m", ".h3i", ".h3f", ".h3p")):
        print(f"ERROR: {pfam_hmm} not hmmpress'd (run: hmmpress {pfam_hmm})",
              file=sys.stderr)
        return False
    cmd = ["hmmscan", "--tblout", str(tbl_out), "--noali", "--cpu", str(cpu),
           str(pfam_hmm), str(prot_fasta)]
    return _run(cmd, "hmmscan")


def _run(cmd: list[str], name: str) -> bool:
    print("  $ " + " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as e:
        print(f"ERROR: failed to launch {name}: {e}", file=sys.stderr)
        return False
    if proc.returncode != 0:
        print(f"ERROR: {name} exited {proc.returncode}\n{proc.stderr[-2000:]}",
              file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# tblout parsing
# ---------------------------------------------------------------------------


def _strip_version(acc: str) -> str:
    """``PF00076.20`` → ``PF00076`` (Rfam accessions have no version)."""
    return acc.split(".")[0] if acc and acc != "-" else acc


def parse_infernal_tbl(path: Path) -> dict[str, dict]:
    """cmsearch ``--tblout`` → ``{seq_id: {family, accession, evalue}}`` (best
    = lowest E-value per sequence).

    Columns (0-based): 0 target(seq) 2 query(family) 3 query-acc 15 E-value.
    """
    best: dict[str, dict] = {}
    if not path.is_file():
        return best
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 16:
            continue
        seq_id, family, acc = f[0], f[2], f[3]
        try:
            ev = float(f[15])
        except ValueError:
            continue
        cur = best.get(seq_id)
        if cur is None or ev < cur["evalue"]:
            best[seq_id] = {"family": family,
                            "accession": _strip_version(acc), "evalue": ev}
    return best


def parse_hmmer_tbl(path: Path) -> dict[str, dict]:
    """hmmscan ``--tblout`` → ``{seq_id: {domain, accession, evalue}}`` (best
    = lowest full-sequence E-value per query).

    Columns (0-based): 0 target(domain) 1 target-acc 2 query(seq) 4 E-value.
    """
    best: dict[str, dict] = {}
    if not path.is_file():
        return best
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 5:
            continue
        domain, acc, seq_id = f[0], f[1], f[2]
        try:
            ev = float(f[4])
        except ValueError:
            continue
        cur = best.get(seq_id)
        if cur is None or ev < cur["evalue"]:
            best[seq_id] = {"domain": domain,
                            "accession": _strip_version(acc), "evalue": ev}
    return best


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

_COLUMNS = ["sample_id", "rna_rfam_family", "rna_rfam_accession",
            "rna_evalue", "protein_pfam_domain", "protein_pfam_accession",
            "protein_evalue"]


def build_summary(sample_ids: list[str], rna_hits: dict[str, dict],
                  prot_hits: dict[str, dict], *, rna_evalue: float,
                  protein_evalue: float) -> list[dict]:
    rows: list[dict] = []
    for sid in sample_ids:
        r = rna_hits.get(sid)
        p = prot_hits.get(sid)
        r_ok = r is not None and r["evalue"] <= rna_evalue
        p_ok = p is not None and p["evalue"] <= protein_evalue
        rows.append({
            "sample_id": sid,
            "rna_rfam_family": r["family"] if r_ok else UNCLASSIFIED,
            "rna_rfam_accession": r["accession"] if r_ok else "",
            "rna_evalue": f"{r['evalue']:.2e}" if r_ok else "",
            "protein_pfam_domain": p["domain"] if p_ok else UNCLASSIFIED,
            "protein_pfam_accession": p["accession"] if p_ok else "",
            "protein_evalue": f"{p['evalue']:.2e}" if p_ok else "",
        })
    return rows


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in _COLUMNS})


def print_distribution(rows: list[dict]) -> None:
    from collections import Counter
    rna = Counter(r["rna_rfam_family"] for r in rows)
    prot = Counter(r["protein_pfam_domain"] for r in rows)
    print(f"\nclassified {len(rows)} samples")
    print("  RNA Rfam families (top 10):")
    for fam, n in rna.most_common(10):
        print(f"    {fam:24s} {n}")
    print("  Protein Pfam domains (top 10):")
    for dom, n in prot.most_common(10):
        print(f"    {dom:24s} {n}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=None,
                   help="dir with samples/<sid>.json (needed unless --parse-only)")
    p.add_argument("--split-file", type=Path, default=None,
                   help="sample-id list (one per line); needed for extraction "
                        "and to order the summary.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--rfam-cm", type=Path, default=None,
                   help="cmpress'd Rfam.cm (full mode RNA classification)")
    p.add_argument("--pfam-hmm", type=Path, default=None,
                   help="hmmpress'd Pfam-A.hmm (full mode protein classification)")
    p.add_argument("--cpu", type=int, default=4)
    p.add_argument("--rna-evalue", type=float, default=0.01,
                   help="max RNA E-value to count as classified (default 0.01)")
    p.add_argument("--protein-evalue", type=float, default=0.01,
                   help="max protein E-value to count as classified (0.01)")
    p.add_argument("--extract-only", action="store_true",
                   help="only write the FASTA files, then stop.")
    p.add_argument("--parse-only", action="store_true",
                   help="only parse existing *_hits.tbl in --output-dir.")
    p.add_argument("--skip-rna", action="store_true")
    p.add_argument("--skip-protein", action="store_true")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.extract_only and args.parse_only:
        print("ERROR: --extract-only and --parse-only are mutually exclusive",
              file=sys.stderr)
        return 1

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Sample ids (from split file when available; else recover from the
    # FASTA headers already written, so --parse-only works standalone).
    sample_ids: list[str] = []
    if args.split_file is not None:
        try:
            sample_ids = load_sample_ids(args.split_file)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

    # ---- extraction ----
    if not args.parse_only:
        if args.processed_dir is None or not args.processed_dir.is_dir():
            print("ERROR: --processed-dir required for extraction",
                  file=sys.stderr)
            return 1
        if not sample_ids:
            print("ERROR: --split-file required for extraction", file=sys.stderr)
            return 1
        n_rna, n_prot = extract_fastas(args.processed_dir, sample_ids, out)
        print(f"wrote {n_rna} RNA → {out / RNA_FASTA}")
        print(f"wrote {n_prot} protein → {out / PROTEIN_FASTA}")
        if args.extract_only:
            return 0

    # ---- run tools (full mode only) ----
    if not args.parse_only:
        if not args.skip_rna:
            if args.rfam_cm is None:
                print("ERROR: --rfam-cm required (or --skip-rna / --parse-only)",
                      file=sys.stderr)
                return 1
            if not run_cmsearch(args.rfam_cm, out / RNA_FASTA, out / RNA_TBL,
                                args.cpu):
                return 1
        if not args.skip_protein:
            if args.pfam_hmm is None:
                print("ERROR: --pfam-hmm required (or --skip-protein)",
                      file=sys.stderr)
                return 1
            if not run_hmmscan(args.pfam_hmm, out / PROTEIN_FASTA,
                               out / PROTEIN_TBL, args.cpu):
                return 1

    # ---- parse + summarise ----
    rna_hits = parse_infernal_tbl(out / RNA_TBL)
    prot_hits = parse_hmmer_tbl(out / PROTEIN_TBL)
    print(f"parsed RNA hits: {len(rna_hits)}  protein hits: {len(prot_hits)}")

    if not sample_ids:
        # parse-only without a split file: order by union of hit ids.
        sample_ids = sorted(set(rna_hits) | set(prot_hits))

    rows = build_summary(sample_ids, rna_hits, prot_hits,
                         rna_evalue=args.rna_evalue,
                         protein_evalue=args.protein_evalue)
    write_summary(out / SUMMARY_CSV, rows)
    print(f"wrote {out / SUMMARY_CSV}  ({len(rows)} samples)")
    print_distribution(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
