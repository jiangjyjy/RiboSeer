"""Re-derive train / val / test splits on a filtered processed dir.

Why a separate script (vs ``src/step1_server/cluster_and_split.py``)
-------------------------------------------------------------------
The original step-1 cluster-and-split script reads ``index.csv`` to
discover samples — that CSV is generated once during step 1 and tracks
the full unfiltered dataset. After ``filter_dataset.py`` produces a
trimmed ``samples/`` dir we want to recluster on the filtered set, but
we don't have a refreshed ``index.csv``. The wrapper below walks
``samples/*.json`` directly and then delegates to the existing
``cluster_and_split.py`` helpers (``dedup_chains``, ``write_*_fasta``,
``run_mmseqs_easy_cluster``, ``group_samples_by_double_cluster``,
``split_groups``, ``audit_leakage``, ``render_report``) so the
clustering / splitting logic is shared verbatim.

Server-only — MMseqs2 not on Windows
------------------------------------
``mmseqs easy-cluster`` is invoked via ``subprocess.run``, so the
script needs the ``mmseqs`` binary on PATH. That binary is only
packaged for Linux / macOS via bioconda. On Windows the call will
exit with a clear "mmseqs not found" error from
``check_mmseqs``. Run this on the Linux server.

Usage
-----
::

    python scripts/recluster_split.py \\
        --processed-dir data/processed_filtered \\
        --stats-dir     data/processed_filtered/stats \\
        --work-dir      data/processed_filtered/_cluster_work \\
        --protein-min-seq-id 0.30 \\
        --rna-min-seq-id     0.80 \\
        --split-ratio        0.8,0.1,0.1 \\
        --seed 42

Reads every ``data/processed_filtered/samples/*.json`` and writes:

  - ``data/processed_filtered/splits.json``     canonical split map
  - ``data/processed_filtered/stats/cluster_stats.md``  human report

After this finishes, regenerate the per-split text files with::

    python scripts/split_samples.py \\
        --processed-dir data/processed_filtered \\
        --output-dir    data/processed_filtered/splits \\
        --train-subset  500
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

# step1_server lives under src/, not on the default Python path.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from step1_server.cluster_and_split import (  # noqa: E402
    assign_cluster_ids, audit_leakage, check_mmseqs, dedup_chains,
    group_samples_by_double_cluster, parse_cluster_tsv, render_report,
    run_mmseqs_easy_cluster, split_groups, write_protein_fasta,
    write_rna_fasta,
)


# ---------- sample loading from samples/<id>.json directly ----------------


def load_samples_from_dir(processed_dir: Path) -> list[dict]:
    """Walk ``processed_dir/samples/*.json`` and produce the dict shape
    cluster_and_split's helpers expect.

    The original ``cluster_and_split.load_samples`` reads ``index.csv``
    + the per-sample JSON; we skip the CSV (filtered datasets don't
    have a refreshed one) and pull the same fields straight from the
    sample dict — every field the downstream pipeline needs is already
    inside ``data/processed/samples/<id>.json``:

      - ``sample_id`` — top-level key
      - ``pdb_id``    — ``source_pdb`` field
      - ``protein_chain`` — ``protein.chain_id``
      - ``rna_chain``     — ``rna.chain_id``
      - ``protein_sequence`` / ``rna_sequence`` — under their objects
      - ``quality_tier`` — under ``data_availability`` (sometimes None;
        downstream cluster_and_split helpers don't actually use it,
        but we surface it so the dict shape stays compatible)
    """
    samples_dir = processed_dir / "samples"
    if not samples_dir.is_dir():
        raise FileNotFoundError(
            f"samples dir not found: {samples_dir}. Did you run "
            f"filter_dataset.py with --output-dir matching this path?"
        )
    out: list[dict] = []
    for path in sorted(samples_dir.glob("*.json")):
        try:
            sample = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"WARN: skipping malformed {path.name}: {e}",
                  file=sys.stderr)
            continue
        protein = sample.get("protein") or {}
        rna = sample.get("rna") or {}
        avail = sample.get("data_availability") or {}
        out.append({
            "sample_id": sample.get("sample_id") or path.stem,
            "pdb_id": sample.get("source_pdb") or "",
            "protein_chain": protein.get("chain_id") or "",
            "rna_chain": rna.get("chain_id") or "",
            "protein_sequence": protein.get("sequence") or "",
            "rna_sequence": rna.get("sequence") or "",
            "quality_tier": avail.get("quality_tier") or "",
        })
    return out


# ---------- main -----------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, required=True,
                        help="filtered dataset root (parent of samples/).")
    parser.add_argument("--stats-dir", type=Path, default=None,
                        help="dir for cluster_stats.md "
                             "(default: <processed-dir>/stats)")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="MMseqs2 work dir "
                             "(default: <processed-dir>/_cluster_work)")
    parser.add_argument("--protein-min-seq-id", type=float, default=0.30)
    parser.add_argument("--rna-min-seq-id", type=float, default=0.80)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--split-ratio", type=str, default="0.8,0.1,0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rna-len", type=int, default=0,
                        help="cap RNA chains at this length for clustering "
                             "(default 0 = no cap; filter_dataset.py already "
                             "trims by length, so this is a belt-and-braces "
                             "guard).")
    parser.add_argument("--mmseqs-bin", type=str, default="mmseqs")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--keep-work", action="store_true",
                        help="don't delete the FASTA + MMseqs2 intermediates")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ratio = tuple(float(x) for x in args.split_ratio.split(","))
    if len(ratio) != 3 or not abs(sum(ratio) - 1.0) < 1e-6:
        print(f"ERROR: --split-ratio must be 3 floats summing to 1; "
              f"got {ratio}", file=sys.stderr)
        return 1

    work_dir = args.work_dir or (args.processed_dir / "_cluster_work")
    stats_dir = args.stats_dir or (args.processed_dir / "stats")

    # mmseqs probe — fails fast on Windows / on a server without bioconda.
    mmseqs_bin = check_mmseqs(args.mmseqs_bin)
    print(f"Using mmseqs: {mmseqs_bin}")

    work_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading samples from {args.processed_dir}/samples/ …")
    samples = load_samples_from_dir(args.processed_dir)
    print(f"  {len(samples)} samples")

    if not samples:
        print(f"ERROR: no samples under {args.processed_dir}/samples — "
              f"did you run filter_dataset.py first?", file=sys.stderr)
        return 1

    prot_chains, rna_chains = dedup_chains(samples)
    print(f"  unique protein chains: {len(prot_chains)}")
    print(f"  unique RNA chains:     {len(rna_chains)}")

    prot_fasta = work_dir / "protein.fasta"
    rna_fasta = work_dir / "rna.fasta"
    prot_written, prot_skipped_X = write_protein_fasta(prot_fasta, prot_chains)
    rna_written, rna_skipped_empty, rna_skipped_long, n_rna_inosine = \
        write_rna_fasta(rna_fasta, rna_chains, args.max_rna_len)
    print(f"  protein FASTA: {len(prot_written)} kept / "
          f"{len(prot_skipped_X)} all-X skipped")
    print(f"  RNA FASTA:     {len(rna_written)} kept / "
          f"{len(rna_skipped_empty)} empty / "
          f"{len(rna_skipped_long)} too long / "
          f"{n_rna_inosine} inosine-sanitized")

    prot_out = work_dir / "protein_cluster"
    rna_out = work_dir / "rna_cluster"
    prot_tmp = work_dir / "tmp_protein"
    rna_tmp = work_dir / "tmp_rna"

    print("\nClustering proteins …")
    prot_tsv = run_mmseqs_easy_cluster(
        mmseqs_bin, prot_fasta, prot_out, prot_tmp,
        min_seq_id=args.protein_min_seq_id,
        coverage=args.coverage,
        threads=args.threads,
        nucleotide=False,
    )
    prot_m2r = parse_cluster_tsv(prot_tsv)
    prot_chain_to_cid, prot_cid_to_rep = assign_cluster_ids(prot_m2r, "prot")
    print(f"  protein clusters: {len(prot_cid_to_rep)}")

    print("\nClustering RNAs …")
    rna_tsv = run_mmseqs_easy_cluster(
        mmseqs_bin, rna_fasta, rna_out, rna_tmp,
        min_seq_id=args.rna_min_seq_id,
        coverage=args.coverage,
        threads=args.threads,
        nucleotide=True,
    )
    rna_m2r = parse_cluster_tsv(rna_tsv)
    rna_chain_to_cid, rna_cid_to_rep = assign_cluster_ids(rna_m2r, "rna")
    print(f"  RNA clusters: {len(rna_cid_to_rep)}")

    groups, unclusterable = group_samples_by_double_cluster(
        samples, prot_chain_to_cid, rna_chain_to_cid,
        prot_skipped_X, rna_skipped_empty, rna_skipped_long,
    )
    print(f"\nClusterable samples:   {sum(len(v) for v in groups.values())}")
    print(f"Unclusterable samples: {len(unclusterable)}")
    print(f"Double-cluster groups: {len(groups)}")

    assignment = split_groups(groups, ratio, args.seed)

    samples_out: dict[str, dict] = {}
    for (prot_cid, rna_cid), sids in groups.items():
        sp = assignment[(prot_cid, rna_cid)]
        for sid in sids:
            samples_out[sid] = {
                "protein_cluster_id": prot_cid,
                "rna_cluster_id": rna_cid,
                "split": sp,
            }
    for sid, info in unclusterable.items():
        samples_out[sid] = {
            "protein_cluster_id": info["protein_cluster_id"],
            "rna_cluster_id": info["rna_cluster_id"],
            "split": None,
        }

    audit = audit_leakage(samples_out)

    group_sizes = sorted(len(v) for v in groups.values())
    stats_dict = {
        "n_samples": len(samples),
        "n_unique_prot_chains": len(prot_chains),
        "n_unique_rna_chains": len(rna_chains),
        "n_prot_fasta": len(prot_written),
        "n_prot_skipped_X": len(prot_skipped_X),
        "n_rna_fasta": len(rna_written),
        "n_rna_skipped_empty": len(rna_skipped_empty),
        "n_rna_skipped_long": len(rna_skipped_long),
        "n_rna_inosine": n_rna_inosine,
        "n_prot_clusters": len(prot_cid_to_rep),
        "n_rna_clusters": len(rna_cid_to_rep),
        "n_clusterable_samples": sum(len(v) for v in groups.values()),
        "n_unclusterable_samples": len(unclusterable),
        "unclusterable_reasons": dict(Counter(
            info["reason"] for info in unclusterable.values()
        )),
        "n_groups": len(groups),
        "max_group_size": group_sizes[-1] if group_sizes else 0,
        "min_group_size": group_sizes[0] if group_sizes else 0,
        "median_group_size":
            group_sizes[len(group_sizes) // 2] if group_sizes else 0,
        "audit": audit,
    }

    cfg = {
        "protein_min_seq_id": args.protein_min_seq_id,
        "rna_min_seq_id": args.rna_min_seq_id,
        "coverage": args.coverage,
        "split_ratio": list(ratio),
        "seed": args.seed,
        "max_rna_len": args.max_rna_len,
        "mmseqs_bin": mmseqs_bin,
        "rna_sanitize_rule": "I->G, -dropped, else non-ACGU -> N",
        "source": "scripts/recluster_split.py (filtered dataset)",
    }

    splits_json = {
        "config": cfg,
        "stats": stats_dict,
        "cluster_index": {
            "protein": prot_cid_to_rep,
            "rna": rna_cid_to_rep,
        },
        "samples": samples_out,
    }
    splits_path = args.processed_dir / "splits.json"
    tmp_path = splits_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(splits_json, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, splits_path)
    print(f"\nWrote {splits_path}")

    report = render_report(cfg, stats_dict)
    report_path = stats_dir / "cluster_stats.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")

    print()
    print(f"Split counts: {audit['per_split_counts']}")
    print(f"val  single-factor leak: {audit['val_single_factor_leak']}")
    print(f"test single-factor leak: {audit['test_single_factor_leak']}")

    if not args.keep_work:
        print(f"\nCleaning up work dir {work_dir} (use --keep-work to retain)")
        shutil.rmtree(work_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
