#!/usr/bin/env python
"""Stage 1.5 — MMseqs2 clustering + cluster-aware train/val/test split (server).

⚠️  LINUX SERVER TASK — DO NOT RUN ON WINDOWS.
    MMseqs2 is only packaged for Linux/macOS through bioconda. This script
    shells out to the `mmseqs` binary, so it cannot run on the local
    Windows env.

Why this script exists
----------------------
The whole point of this step is **preventing data leakage** into val/test.
Two samples are "homologous" in our definition iff:
  - their protein chains share a 30 % identity cluster  AND
  - their RNA chains share an 80 % identity cluster
Homologous samples MUST go to the same split. That way, at eval time the
model never sees a train-like (protein, RNA) pair.

Pipeline
--------
  1. Load every sample JSON referenced by `data/processed/index.csv`.
  2. Deduplicate protein chains by `(pdb_id, chain_id)` and RNA chains the
     same way — many samples from the same ribosome share identical chains.
  3. Filter out unclusterable sequences:
       * all-X protein sequences  (122 samples, known edge case from 1.4)
       * empty RNA after sanitization
       * RNA longer than --max-rna-len (opt-in; default 0 = no cap)
  4. Sanitize RNA before writing FASTA:
       * drop gap '-'
       * 'I' → 'G'                (poly-inosine samples from 7wv3/7wv4)
       * anything non-ACGU → 'N'
  5. Write two FASTAs (one per biomolecule) into a work directory.
  6. Call `mmseqs easy-cluster`:
       * protein:  --min-seq-id 0.3  -c 0.8  --cov-mode 0
       * RNA:      --min-seq-id 0.8  -c 0.8  --cov-mode 0  --search-type 3
     The cov-mode 0 + -c 0.8 combination requires both representative and
     member to cover each other ≥80 %; this avoids the classic "short
     fragment hides inside a long reference" leak.
  7. Parse `*_cluster.tsv` → `{member_header: rep_header}` mapping.
  8. Assign each sample a `(protein_cluster_id, rna_cluster_id)` pair by
     looking up its two chains.
  9. Group samples by that pair, shuffle groups with a fixed seed,
     greedy bin-pack into train/val/test targeting --split-ratio while
     keeping group-atomicity (entire groups never straddle splits).
 10. Run a leakage audit: count how many val / test samples share a
     *single-factor* cluster (protein OR RNA) with any train sample, and
     report it. The double-cluster definition makes single-factor overlap
     legal, but it's worth flagging so the user can choose a stricter
     split policy later.
 11. Write:
       * `data/processed/splits.json`           — canonical split map
       * `data/stats/cluster_stats.md`          — human-readable report

Output: `splits.json`
---------------------
```
{
  "config": {...},
  "cluster_index": {
    "protein": { "prot_000001": "1un6_B", "prot_000002": "2xxa_A", ... },
    "rna":     { "rna_000001":  "1un6_F", "rna_000002":  "2xxa_F", ... }
  },
  "samples": {
    "1un6_B_F": {
       "protein_cluster_id": "prot_000001",
       "rna_cluster_id":     "rna_000001",
       "split":              "train"
    },
    "3j92_x_5": {
       "protein_cluster_id": "unclusterable_X_only",
       "rna_cluster_id":     "rna_000123",
       "split":              null
    },
    ...
  }
}
```

The stage 1.6 `dataset_loader` reads `splits.json` to build per-split sample
iterators. `apply_splits.py` (the sibling script) merges these values into
every sample JSON's `split_info` block so individual samples carry their
split without needing to re-consult `splits.json`.

Setup
-----
    conda create -n riboseer python=3.11 -y
    conda activate riboseer
    conda config --add channels defaults
    conda config --add channels bioconda
    conda config --add channels conda-forge
    conda config --set channel_priority strict
    conda install -c conda-forge -c bioconda \
        pydantic tqdm mmseqs2 -y
    # (this script doesn't need ViennaRNA — it shells out to `mmseqs`.)

Usage
-----
    python src/step1_server/cluster_and_split.py \\
        --processed-dir /srv/pocket/data/processed \\
        --stats-dir     /srv/pocket/data/stats \\
        --work-dir      /srv/pocket/data/_cluster_work

    # custom thresholds
    python src/step1_server/cluster_and_split.py \\
        --processed-dir /srv/pocket/data/processed \\
        --stats-dir     /srv/pocket/data/stats \\
        --work-dir      /srv/pocket/data/_cluster_work \\
        --protein-min-seq-id 0.25 \\
        --rna-min-seq-id 0.9 \\
        --split-ratio 0.85,0.05,0.10 \\
        --seed 7 \\
        --threads 16

    # cap RNA length for clustering (doesn't affect the sample JSONs)
    python src/step1_server/cluster_and_split.py ... --max-rna-len 4000
"""

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
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


# ---------- RNA sanitization (must match step1_server/rna_secondary_structure.py) ----


def sanitize_rna(seq: str) -> tuple[str, bool]:
    inosine = False
    out = []
    for ch in seq:
        if ch == "-":
            continue
        if ch == "I":
            inosine = True
            out.append("G")
        elif ch in "ACGU":
            out.append(ch)
        else:
            out.append("N")
    return "".join(out), inosine


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def is_protein_all_X(seq: str) -> bool:
    return not any(c in STANDARD_AA for c in seq)


# ---------- load samples + build unique chain tables ------------------------


def load_samples(processed_dir: Path) -> list[dict]:
    """Return a list of sample dicts — we need sequences, not just metadata."""
    index_csv = processed_dir / "index.csv"
    samples_dir = processed_dir / "samples"
    with index_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        sid = row["sample_id"]
        path = samples_dir / (sid_to_filename(sid) + ".json")
        with path.open("r", encoding="utf-8") as f:
            sample = json.load(f)
        out.append({
            "sample_id": sid,
            "pdb_id": row["pdb_id"],
            "protein_chain": row["protein_chain"],
            "rna_chain": row["rna_chain"],
            "protein_sequence": sample["protein"]["sequence"],
            "rna_sequence": sample["rna"]["sequence"],
            "quality_tier": row["quality_tier"],
        })
    return out


def dedup_chains(samples: list[dict]) -> tuple[dict, dict]:
    """Return (prot_chain_seqs, rna_chain_seqs) keyed by '{pdb}_{chain}'."""
    prot: dict[str, str] = {}
    rna: dict[str, str] = {}
    for s in samples:
        pkey = f"{s['pdb_id']}_{s['protein_chain']}"
        rkey = f"{s['pdb_id']}_{s['rna_chain']}"
        if pkey not in prot:
            prot[pkey] = s["protein_sequence"]
        if rkey not in rna:
            rna[rkey] = s["rna_sequence"]
    return prot, rna


# ---------- FASTA writers ---------------------------------------------------


def write_protein_fasta(path: Path, chains: dict[str, str]) -> tuple[set, set]:
    """Return (written_keys, skipped_all_X_keys)."""
    written = set()
    skipped = set()
    with path.open("w", encoding="utf-8") as f:
        for key, seq in chains.items():
            if is_protein_all_X(seq):
                skipped.add(key)
                continue
            # Remove gaps — MMseqs2 dislikes '-' in protein input.
            cleaned = seq.replace("-", "")
            if not cleaned:
                skipped.add(key)
                continue
            f.write(f">{key}\n{cleaned}\n")
            written.add(key)
    return written, skipped


def write_rna_fasta(path: Path, chains: dict[str, str],
                    max_len: int) -> tuple[set, set, set, int]:
    """Return (written_keys, skipped_empty_keys, skipped_too_long_keys, n_inosine_sanitized)."""
    written = set()
    skipped_empty = set()
    skipped_too_long = set()
    n_inosine = 0
    with path.open("w", encoding="utf-8") as f:
        for key, seq in chains.items():
            cleaned, inosine = sanitize_rna(seq)
            if inosine:
                n_inosine += 1
            if not cleaned:
                skipped_empty.add(key)
                continue
            if max_len and len(cleaned) > max_len:
                skipped_too_long.add(key)
                continue
            f.write(f">{key}\n{cleaned}\n")
            written.add(key)
    return written, skipped_empty, skipped_too_long, n_inosine


# ---------- MMseqs2 wrapper -------------------------------------------------


def check_mmseqs(mmseqs_bin: str) -> str:
    path = shutil.which(mmseqs_bin)
    if path is None:
        sys.exit(
            f"ERROR: '{mmseqs_bin}' not found on PATH.\n"
            "Install via `conda install -c bioconda mmseqs2` on the server."
        )
    return path


def run_mmseqs_easy_cluster(
    mmseqs_bin: str,
    fasta: Path,
    out_prefix: Path,
    tmp_dir: Path,
    min_seq_id: float,
    coverage: float,
    threads: int,
    nucleotide: bool,
) -> Path:
    """Run `mmseqs easy-cluster` and return the path to *_cluster.tsv."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        mmseqs_bin, "easy-cluster",
        str(fasta),
        str(out_prefix),
        str(tmp_dir),
        "--min-seq-id", f"{min_seq_id}",
        "-c", f"{coverage}",
        "--cov-mode", "0",
    ]
    if threads:
        cmd += ["--threads", str(threads)]
    # Nucleotide vs protein is auto-detected from FASTA content by MMseqs2.
    # We used to pass `--search-type 3` for RNA, but some MMseqs2 builds
    # (e.g. the one in this conda env) don't expose that flag on easy-cluster
    # and error out with "Unrecognized parameter". Auto-detect is reliable for
    # our inputs (RNA FASTA is ACGUN-only, protein is 20-letter).
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    tsv = Path(f"{out_prefix}_cluster.tsv")
    if not tsv.exists():
        sys.exit(f"MMseqs2 ran but {tsv} is missing")
    return tsv


def parse_cluster_tsv(tsv_path: Path) -> dict[str, str]:
    """Return `{member_id: representative_id}` keyed on FASTA headers."""
    member_to_rep: dict[str, str] = {}
    with tsv_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            rep, mem = line.split("\t", 1)
            member_to_rep[mem] = rep
    return member_to_rep


def assign_cluster_ids(member_to_rep: dict[str, str], prefix: str) -> tuple[dict, dict]:
    """Return (chain_key -> cluster_id, cluster_id -> rep_chain_key).

    Deterministic: cluster numbering follows the lexicographic order of
    representative keys, so splits are reproducible across runs.
    """
    reps_sorted = sorted(set(member_to_rep.values()))
    rep_to_cid = {
        rep: f"{prefix}_{i + 1:06d}" for i, rep in enumerate(reps_sorted)
    }
    chain_to_cid = {mem: rep_to_cid[rep] for mem, rep in member_to_rep.items()}
    cid_to_rep = {cid: rep for rep, cid in rep_to_cid.items()}
    return chain_to_cid, cid_to_rep


# ---------- split by double-cluster group -----------------------------------


def group_samples_by_double_cluster(
    samples: list[dict],
    prot_chain_to_cid: dict[str, str],
    rna_chain_to_cid: dict[str, str],
    prot_skipped: set[str],
    rna_skipped_empty: set[str],
    rna_skipped_long: set[str],
) -> tuple[dict[tuple[str, str], list[str]], dict[str, dict]]:
    """Return (clusterable_groups, unclusterable_samples).

    clusterable_groups: {(prot_cid, rna_cid): [sample_ids]}
    unclusterable_samples: {sample_id: {"protein_cluster_id": ..., "rna_cluster_id": ..., "reason": ...}}
    """
    clusterable: dict[tuple[str, str], list[str]] = defaultdict(list)
    unclusterable: dict[str, dict] = {}

    for s in samples:
        sid = s["sample_id"]
        pkey = f"{s['pdb_id']}_{s['protein_chain']}"
        rkey = f"{s['pdb_id']}_{s['rna_chain']}"

        prot_cid = prot_chain_to_cid.get(pkey)
        rna_cid = rna_chain_to_cid.get(rkey)

        reasons = []
        if prot_cid is None:
            if pkey in prot_skipped:
                prot_cid = "unclusterable_X_only"
                reasons.append("protein_all_X")
            else:
                prot_cid = "unclusterable_protein_missing"
                reasons.append("protein_missing_from_cluster_tsv")
        if rna_cid is None:
            if rkey in rna_skipped_empty:
                rna_cid = "unclusterable_rna_empty"
                reasons.append("rna_empty_after_sanitize")
            elif rkey in rna_skipped_long:
                rna_cid = "unclusterable_rna_too_long"
                reasons.append("rna_too_long_for_clustering")
            else:
                rna_cid = "unclusterable_rna_missing"
                reasons.append("rna_missing_from_cluster_tsv")

        if reasons:
            unclusterable[sid] = {
                "protein_cluster_id": prot_cid,
                "rna_cluster_id": rna_cid,
                "reason": ",".join(reasons),
            }
        else:
            clusterable[(prot_cid, rna_cid)].append(sid)

    return clusterable, unclusterable


def split_groups(
    groups: dict[tuple[str, str], list[str]],
    ratio: tuple[float, float, float],
    seed: int,
) -> dict[tuple[str, str], str]:
    """Greedy bin-packing: every group is atomic; splits target `ratio`."""
    rng = random.Random(seed)
    group_list = list(groups.items())
    rng.shuffle(group_list)
    # Sort largest first so the bin-packer can correct imbalances early.
    group_list.sort(key=lambda kv: -len(kv[1]))

    total = sum(len(samples) for _, samples in group_list)
    splits = ["train", "val", "test"]
    targets = {name: total * r for name, r in zip(splits, ratio)}
    filled = {name: 0 for name in splits}
    assignment: dict[tuple[str, str], str] = {}

    for key, samples in group_list:
        # Pick the split with the largest *remaining* target capacity.
        best = max(splits, key=lambda s: targets[s] - filled[s])
        assignment[key] = best
        filled[best] += len(samples)

    return assignment


def audit_leakage(
    samples_meta: dict[str, dict],
) -> dict:
    """Count per-split and single-factor cross-split overlaps.

    `samples_meta[sid] = {"protein_cluster_id": ..., "rna_cluster_id": ..., "split": ...}`
    """
    per_split_counts: Counter = Counter()
    train_prot_cids = set()
    train_rna_cids = set()
    for sid, meta in samples_meta.items():
        sp = meta["split"]
        if sp is None:
            per_split_counts["unclustered"] += 1
            continue
        per_split_counts[sp] += 1
        if sp == "train":
            train_prot_cids.add(meta["protein_cluster_id"])
            train_rna_cids.add(meta["rna_cluster_id"])

    val_prot_leak = val_rna_leak = 0
    test_prot_leak = test_rna_leak = 0
    for sid, meta in samples_meta.items():
        sp = meta["split"]
        if sp not in ("val", "test"):
            continue
        prot_leak = meta["protein_cluster_id"] in train_prot_cids
        rna_leak = meta["rna_cluster_id"] in train_rna_cids
        if sp == "val":
            val_prot_leak += int(prot_leak)
            val_rna_leak += int(rna_leak)
        else:
            test_prot_leak += int(prot_leak)
            test_rna_leak += int(rna_leak)

    return {
        "per_split_counts": dict(per_split_counts),
        "val_single_factor_leak": {
            "protein_cluster_shared_with_train": val_prot_leak,
            "rna_cluster_shared_with_train": val_rna_leak,
        },
        "test_single_factor_leak": {
            "protein_cluster_shared_with_train": test_prot_leak,
            "rna_cluster_shared_with_train": test_rna_leak,
        },
    }


# ---------- report ----------------------------------------------------------


def render_report(cfg: dict, stats: dict) -> str:
    lines = [
        "# Stage 1.5 — Cluster & Split Report",
        "",
        "## Parameters",
        "",
        f"- protein min seq id: **{cfg['protein_min_seq_id']}**",
        f"- RNA min seq id: **{cfg['rna_min_seq_id']}**",
        f"- coverage: **{cfg['coverage']}**",
        f"- split ratio: **{cfg['split_ratio']}**",
        f"- random seed: **{cfg['seed']}**",
        f"- RNA max length (for clustering): "
        f"**{cfg['max_rna_len'] if cfg['max_rna_len'] else 'unlimited'}**",
        f"- RNA sanitize rule: drop `-`, `I → G`, any other non-ACGU → `N`",
        "",
        "## Input scale",
        "",
        f"- Total samples: **{stats['n_samples']}**",
        f"- Unique protein chains after dedup: **{stats['n_unique_prot_chains']}**",
        f"- Unique RNA chains after dedup: **{stats['n_unique_rna_chains']}**",
        f"- Chains written to the protein FASTA: **{stats['n_prot_fasta']}**",
        f"- All-X protein chains skipped: **{stats['n_prot_skipped_X']}**",
        f"- Chains written to the RNA FASTA: **{stats['n_rna_fasta']}**",
        f"- Empty RNA chains skipped (after sanitize): **{stats['n_rna_skipped_empty']}**",
        f"- Over-long RNA chains skipped (>max-rna-len): **{stats['n_rna_skipped_long']}**",
        f"- RNA chains sanitized poly-inosine → G: **{stats['n_rna_inosine']}**",
        "",
        "## Clustering results",
        "",
        f"- Protein clusters: **{stats['n_prot_clusters']}** "
        f"(mean {stats['n_prot_fasta'] / max(stats['n_prot_clusters'], 1):.2f} chains/cluster)",
        f"- RNA clusters: **{stats['n_rna_clusters']}** "
        f"(mean {stats['n_rna_fasta'] / max(stats['n_rna_clusters'], 1):.2f} chains/cluster)",
        "",
        "### Sample level",
        "",
        f"- Clusterable samples: **{stats['n_clusterable_samples']}**",
        f"- Unclusterable samples: **{stats['n_unclusterable_samples']}**",
        f"- Unclusterable reason distribution:",
    ]
    for reason, n in stats["unclusterable_reasons"].items():
        lines.append(f"  - {reason}: {n}")
    lines += [
        "",
        f"- Distinct (protein_cid, rna_cid) groups: **{stats['n_groups']}**",
        f"- Largest group size: **{stats['max_group_size']}** samples",
        f"- Smallest group size: **{stats['min_group_size']}** samples",
        f"- Median group size: **{stats['median_group_size']}**",
    ]

    lines += ["", "## Split results", ""]
    for split in ("train", "val", "test", "unclustered"):
        n = stats["audit"]["per_split_counts"].get(split, 0)
        lines.append(f"- {split}: **{n}**")

    lines += [
        "",
        "## Leakage audit",
        "",
        "Under the double-cluster definition, val / test share no (protein_cid, rna_cid) pair with train. "
        "What is reported below is the **single-factor** cross-split overlap (overlap of the protein cluster only, "
        "or of the RNA cluster only), which the double-cluster definition allows; but when the numbers get large, "
        "generalization pressure comes mostly from unfamiliar pair combinations ",
        "rather than from entirely unfamiliar proteins/RNAs.",
        "",
        f"- val samples whose protein cluster overlaps train: **{stats['audit']['val_single_factor_leak']['protein_cluster_shared_with_train']}**",
        f"- val samples whose RNA cluster overlaps train: **{stats['audit']['val_single_factor_leak']['rna_cluster_shared_with_train']}**",
        f"- test samples whose protein cluster overlaps train: **{stats['audit']['test_single_factor_leak']['protein_cluster_shared_with_train']}**",
        f"- test samples whose RNA cluster overlaps train: **{stats['audit']['test_single_factor_leak']['rna_cluster_shared_with_train']}**",
        "",
        "## Notes",
        "",
        "- **Greedy bin-packing with shuffle + largest-group-first**: groups are atomic and large groups go in first, ",
        "  so even with wildly uneven group sizes (the huge ribosome groups), the per-split sample ratios stay close to target.",
        "- Actual versus target ratios appear under Split results; the deviation comes mainly from large groups that cannot be split.",
        "- **Unclusterable samples have `split = null`**: they do not enter train/val/test, and the downstream dataloader decides explicitly whether to load them.",
        "- **`unclusterable_X_only`**: from the 122 all-X protein sequences found in stage 1.4 (chains of unidentified identity in large RNPs).",
    ]
    return "\n".join(lines) + "\n"


# ---------- main ------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1.5: MMseqs2 cluster + cluster-aware split (server)."
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="where FASTAs and MMseqs2 intermediates go")
    parser.add_argument("--protein-min-seq-id", type=float, default=0.30)
    parser.add_argument("--rna-min-seq-id", type=float, default=0.80)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--split-ratio", type=str, default="0.8,0.1,0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rna-len", type=int, default=0,
                        help="skip RNA chains longer than this from clustering "
                             "(default 0 = no cap). Skipped chains are marked "
                             "unclusterable_rna_too_long.")
    parser.add_argument("--mmseqs-bin", type=str, default="mmseqs")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--keep-work", action="store_true",
                        help="don't delete the FASTA + MMseqs2 intermediates")
    args = parser.parse_args()

    ratio = tuple(float(x) for x in args.split_ratio.split(","))
    if len(ratio) != 3 or not abs(sum(ratio) - 1.0) < 1e-6:
        sys.exit(f"--split-ratio must be 3 floats summing to 1; got {ratio}")

    mmseqs_bin = check_mmseqs(args.mmseqs_bin)
    print(f"Using mmseqs: {mmseqs_bin}")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.stats_dir.mkdir(parents=True, exist_ok=True)

    print("Loading samples from index.csv + sample JSONs …")
    samples = load_samples(args.processed_dir)
    print(f"  {len(samples)} samples")

    prot_chains, rna_chains = dedup_chains(samples)
    print(f"  unique protein chains: {len(prot_chains)}")
    print(f"  unique RNA chains:     {len(rna_chains)}")

    prot_fasta = args.work_dir / "protein.fasta"
    rna_fasta = args.work_dir / "rna.fasta"
    prot_written, prot_skipped_X = write_protein_fasta(prot_fasta, prot_chains)
    rna_written, rna_skipped_empty, rna_skipped_long, n_rna_inosine = \
        write_rna_fasta(rna_fasta, rna_chains, args.max_rna_len)
    print(f"  protein FASTA: {len(prot_written)} kept / {len(prot_skipped_X)} all-X skipped")
    print(f"  RNA FASTA:     {len(rna_written)} kept / {len(rna_skipped_empty)} empty / "
          f"{len(rna_skipped_long)} too long / {n_rna_inosine} inosine-sanitized")

    # Run MMseqs2 twice
    prot_out = args.work_dir / "protein_cluster"
    rna_out = args.work_dir / "rna_cluster"
    prot_tmp = args.work_dir / "tmp_protein"
    rna_tmp = args.work_dir / "tmp_rna"

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

    # Group by double cluster
    groups, unclusterable = group_samples_by_double_cluster(
        samples,
        prot_chain_to_cid,
        rna_chain_to_cid,
        prot_skipped_X,
        rna_skipped_empty,
        rna_skipped_long,
    )
    print(f"\nClusterable samples: {sum(len(v) for v in groups.values())}")
    print(f"Unclusterable samples: {len(unclusterable)}")
    print(f"Double-cluster groups: {len(groups)}")

    # Split
    assignment = split_groups(groups, ratio, args.seed)

    # Build final sample → metadata dict
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
        "median_group_size": group_sizes[len(group_sizes) // 2] if group_sizes else 0,
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
    }

    # Write splits.json
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

    # Write cluster_stats.md
    report = render_report(cfg, stats_dict)
    report_path = args.stats_dir / "cluster_stats.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")

    print()
    print(f"Split counts: {audit['per_split_counts']}")
    print(f"val  single-factor leak: {audit['val_single_factor_leak']}")
    print(f"test single-factor leak: {audit['test_single_factor_leak']}")

    if not args.keep_work:
        print(f"\nCleaning up work dir {args.work_dir} (use --keep-work to retain)")
        shutil.rmtree(args.work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
