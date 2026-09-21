"""Stage 1.4: protein basic features (local).

For every sample in `data/processed/index.csv`, writes back three features
under `sample["protein"]["features"]`:

  - pI             — isoelectric point via Biopython's `IsoelectricPoint`,
                     computed on the sample's protein sequence after
                     filtering to the 20 standard amino acids
  - mean_bfactor   — mean B-factor over heavy (non-hydrogen) protein atoms
                     of the chain, *excluding atoms with b_iso == 0*
                     (cryo-EM commonly leaves the B column as a placeholder).
                     `null` when every heavy atom has `b_iso == 0` — the
                     structure simply doesn't carry a meaningful B field.
  - aa_composition — {aa: fraction} dict over the 20 standard AAs. Non-
                     standard / unknown / gap characters are excluded from
                     both numerator and denominator.

`sample["protein"]["domain"]` and `sample["protein"]["plddt_mean"]` stay
`null` — they need external tools (InterPro, AlphaFold) and are deferred.

Performance notes
-----------------
* Samples are grouped by `pdb_id` so each raw file is parsed at most once
  per run. Within one PDB, mean_bfactor is cached per chain so multiple
  samples that reuse the same protein chain (same protein bound to
  different RNA partners) only iterate atoms once.
* Per-PDB resume: if **every** sample from a PDB already has all three
  feature keys, the raw structure is never opened. Combined with per-
  sample resume, re-invoking this script after a Ctrl-C is nearly free.
* Idempotent + atomic write (tmp + `os.replace`), same style as 1.3a.

Usage
-----
    python src/step1_local/protein_features.py \\
        --raw-dir       /path/to/riboseer/data/raw \\
        --processed-dir /path/to/riboseer/data/processed \\
        --stats-dir     /path/to/riboseer/data/stats
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from _common import (  # noqa: E402 — fix_windows_dll_path() runs on import
    classify_residue,
    sid_to_filename,
)
import gemmi  # noqa: E402
from tqdm import tqdm  # noqa: E402
from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint  # noqa: E402


STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"
FEATURE_KEYS = ("pI", "mean_bfactor", "aa_composition")


def compute_pi(sequence: str) -> float | None:
    """Isoelectric point over standard AAs only. None if no standard AAs."""
    clean = "".join(c for c in sequence if c in STANDARD_AAS)
    if not clean:
        return None
    try:
        return round(IsoelectricPoint(clean).pi(), 3)
    except Exception:
        return None


def compute_aa_composition(sequence: str) -> dict | None:
    """20-dim vector of AA fractions. None if sequence has no standard AAs."""
    counter = Counter(c for c in sequence if c in STANDARD_AAS)
    total = sum(counter.values())
    if total == 0:
        return None
    return {aa: round(counter.get(aa, 0) / total, 4) for aa in STANDARD_AAS}


def compute_mean_bfactor(chain) -> tuple[float | None, int, int]:
    """Return (mean_b, n_nonzero, n_total).

    mean_b is None if `n_nonzero == 0` (everything is placeholder 0.0).
    Skips hydrogens and non-protein residues (ligands / waters merged into
    the chain by gemmi's dominant-polymer view).
    """
    total = 0.0
    n_nonzero = 0
    n_total = 0
    for res in chain:
        cat, _ = classify_residue(res.name)
        if cat != "protein":
            continue
        for atom in res:
            if atom.element.is_hydrogen:
                continue
            n_total += 1
            b = atom.b_iso
            if b > 0.0:
                total += b
                n_nonzero += 1
    if n_nonzero == 0:
        return None, 0, n_total
    return round(total / n_nonzero, 3), n_nonzero, n_total


def find_raw_file(raw_dir: Path, pdb_id: str) -> Path | None:
    for ext in (".pdb", ".cif"):
        p = raw_dir / f"{pdb_id}{ext}"
        if p.exists():
            return p
    return None


def sample_is_complete(sample: dict) -> bool:
    feats = sample.get("protein", {}).get("features", {})
    return all(k in feats for k in FEATURE_KEYS)


def _quartiles(xs: list[float]) -> tuple:
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return (None,) * 5
    return xs[0], xs[n // 4], xs[n // 2], xs[min(3 * n // 4, n - 1)], xs[-1]


def render_report(totals, per_tier_pi, per_tier_bfactor,
                  aa_avg, n_bfactor_null, n_pi_null,
                  samples_per_method_bfactor_null):
    lines = [
        "# Stage 1.4 — Protein Basic Features Report",
        "",
        f"- Total samples scanned: **{totals['n_samples']}**",
        f"- Samples updated this run: **{totals['n_updated']}**",
        f"- Resume-skipped (all 3 features already present): **{totals['n_skipped_resume']}**",
        f"- Whole PDB skipped (every sample already complete): **{totals['n_pdbs_skipped_all_done']}** PDBs",
        f"- PDBs with no raw file found: **{totals['n_failed_raw_lookup']}**",
        f"- PDBs that failed gemmi parsing: **{totals['n_failed_parse']}**",
        f"- Samples whose chain is missing from the structure: **{totals['n_chain_missing']}**",
        f"- Samples with `mean_bfactor` null (B-factor column all zeros): **{n_bfactor_null}**",
        f"- Samples with `pI` null (sequence has no standard AA): **{n_pi_null}**",
        "",
        "## pI distribution (by tier)",
        "",
        "| tier | n | min | p25 | median | p75 | max |",
        "|------|---|-----|-----|--------|-----|-----|",
    ]
    for tier in ("strict", "standard", "low", "discard"):
        xs = per_tier_pi.get(tier, [])
        n = len(xs)
        if not n:
            lines.append(f"| {tier} | 0 | - | - | - | - | - |")
            continue
        lo, p25, med, p75, hi = _quartiles(xs)
        lines.append(f"| {tier} | {n} | {lo:.2f} | {p25:.2f} | {med:.2f} | {p75:.2f} | {hi:.2f} |")

    lines += [
        "",
        "## mean_bfactor distribution (by tier, nulls excluded)",
        "",
        "| tier | n | min | p25 | median | p75 | max |",
        "|------|---|-----|-----|--------|-----|-----|",
    ]
    for tier in ("strict", "standard", "low", "discard"):
        xs = per_tier_bfactor.get(tier, [])
        n = len(xs)
        if not n:
            lines.append(f"| {tier} | 0 | - | - | - | - | - |")
            continue
        lo, p25, med, p75, hi = _quartiles(xs)
        lines.append(f"| {tier} | {n} | {lo:.1f} | {p25:.1f} | {med:.1f} | {p75:.1f} | {hi:.1f} |")

    if samples_per_method_bfactor_null:
        lines += [
            "",
            "## Samples with `mean_bfactor is null`, by experimental method",
            "",
        ]
        for method, n in samples_per_method_bfactor_null.most_common():
            lines.append(f"- {method}: {n}")

    if aa_avg:
        lines += [
            "",
            "## Mean amino-acid composition (averaged over all samples, sorted by frequency)",
            "",
            "| aa | mean fraction |",
            "|----|---------------|",
        ]
        for aa, frac in sorted(aa_avg.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {aa}  | {frac:.4f} |")

    lines += [
        "",
        "## Notes",
        "",
        "- `pI` uses Biopython's `IsoelectricPoint`, computed after filtering characters that are not one of the 20 standard AAs out of the sequence.",
        "- `mean_bfactor` only averages non-zero heavy-atom b_iso values: cryo-EM PDBs often fill the B-factor column with 0 as a placeholder, "
        "and a plain average would be dragged down. If every heavy atom of a chain has b_iso == 0, `mean_bfactor` is set to null.",
        "- The `aa_composition` denominator is the number of standard AAs in the sequence (X / non-standard / gaps removed), "
        "so the 20-dim vector sums to 1.",
        "- The `domain` / `plddt_mean` fields stay null: the former awaits the InterPro query (a later stage), "
        "the latter awaits AlphaFold structures (step 3+).",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1.4: compute protein basic features and write back into sample JSONs."
    )
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="root containing the original .pdb / .cif files")
    parser.add_argument("--processed-dir", type=Path, required=True,
                        help="root containing samples/*.json and index.csv")
    parser.add_argument("--stats-dir", type=Path, required=True,
                        help="where protein_features_report.md goes")
    parser.add_argument("--force", action="store_true",
                        help="recompute features even for samples that already have them")
    parser.add_argument("--limit", type=int, default=0,
                        help="only process first N PDBs (debug)")
    args = parser.parse_args()

    samples_dir = args.processed_dir / "samples"
    index_csv = args.processed_dir / "index.csv"
    args.stats_dir.mkdir(parents=True, exist_ok=True)

    if not index_csv.exists():
        sys.exit(f"index.csv not found at {index_csv} — run stage 1.2 first")

    with index_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} sample rows from {index_csv}")

    # Group samples by pdb_id
    by_pdb: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_pdb[row["pdb_id"]].append(row)
    pdbs = sorted(by_pdb.keys())
    if args.limit:
        pdbs = pdbs[: args.limit]
    print(f"Processing {len(pdbs)} PDB files")

    totals = Counter()
    totals["n_samples"] = len(rows) if not args.limit else sum(len(by_pdb[p]) for p in pdbs)

    per_tier_pi: dict[str, list[float]] = defaultdict(list)
    per_tier_bfactor: dict[str, list[float]] = defaultdict(list)
    aa_composition_sums: Counter = Counter()
    aa_n = 0
    n_bfactor_null = 0
    n_pi_null = 0
    samples_per_method_bfactor_null: Counter = Counter()

    for pdb_id in tqdm(pdbs, desc="pdb", mininterval=1.0):
        sample_rows = by_pdb[pdb_id]

        # Per-PDB resume: if every sample for this PDB is complete, don't
        # even open the raw structure.
        pdb_rows_loaded: list[tuple[dict, Path]] = []
        if not args.force:
            all_done = True
            for row in sample_rows:
                fn = sid_to_filename(row["sample_id"]) + ".json"
                path = samples_dir / fn
                with path.open("r", encoding="utf-8") as f:
                    sample = json.load(f)
                if not sample_is_complete(sample):
                    all_done = False
                pdb_rows_loaded.append((sample, path))
            if all_done:
                totals["n_pdbs_skipped_all_done"] += 1
                totals["n_skipped_resume"] += len(sample_rows)
                # Still accumulate stats from already-populated features so the
                # report reflects the full dataset, not just this run's delta.
                for sample, _ in pdb_rows_loaded:
                    feats = sample["protein"]["features"]
                    tier = sample["data_availability"]["quality_tier"]
                    if feats.get("pI") is not None:
                        per_tier_pi[tier].append(feats["pI"])
                    else:
                        n_pi_null += 1
                    if feats.get("mean_bfactor") is not None:
                        per_tier_bfactor[tier].append(feats["mean_bfactor"])
                    else:
                        n_bfactor_null += 1
                        samples_per_method_bfactor_null[
                            sample["data_availability"].get("experimental_method", "unknown")
                        ] += 1
                    if feats.get("aa_composition"):
                        for aa, frac in feats["aa_composition"].items():
                            aa_composition_sums[aa] += frac
                        aa_n += 1
                continue

        # Parse the PDB (expensive)
        raw_file = find_raw_file(args.raw_dir, pdb_id)
        if raw_file is None:
            totals["n_failed_raw_lookup"] += 1
            continue
        try:
            structure = gemmi.read_structure(str(raw_file), merge_chain_parts=True)
            structure.setup_entities()
            structure.assign_label_seq_id(True)
            model = structure[0]
        except Exception:
            totals["n_failed_parse"] += 1
            continue

        # Cache mean_bfactor per chain within this PDB
        needed_chains = {r["protein_chain"] for r in sample_rows}
        chain_bfactor: dict[str, tuple[float | None, int, int]] = {}
        for chain in model:
            if chain.name in needed_chains:
                chain_bfactor[chain.name] = compute_mean_bfactor(chain)

        # Update each sample JSON for this PDB
        for row, (sample, path) in zip(sample_rows, pdb_rows_loaded or [(None, None)] * len(sample_rows)):
            if sample is None:
                # --force path: we never preloaded; load now.
                fn = sid_to_filename(row["sample_id"]) + ".json"
                path = samples_dir / fn
                with path.open("r", encoding="utf-8") as f:
                    sample = json.load(f)
            if not args.force and sample_is_complete(sample):
                totals["n_skipped_resume"] += 1
                # Accumulate existing-feature stats
                feats = sample["protein"]["features"]
                tier = sample["data_availability"]["quality_tier"]
                if feats.get("pI") is not None:
                    per_tier_pi[tier].append(feats["pI"])
                else:
                    n_pi_null += 1
                if feats.get("mean_bfactor") is not None:
                    per_tier_bfactor[tier].append(feats["mean_bfactor"])
                else:
                    n_bfactor_null += 1
                    samples_per_method_bfactor_null[
                        sample["data_availability"].get("experimental_method", "unknown")
                    ] += 1
                if feats.get("aa_composition"):
                    for aa, frac in feats["aa_composition"].items():
                        aa_composition_sums[aa] += frac
                    aa_n += 1
                continue

            prot_chain_name = row["protein_chain"]
            seq = sample["protein"]["sequence"]
            tier = sample["data_availability"]["quality_tier"]

            pI = compute_pi(seq)
            aa_comp = compute_aa_composition(seq)
            if prot_chain_name not in chain_bfactor:
                totals["n_chain_missing"] += 1
                mean_b = None
            else:
                mean_b, _, _ = chain_bfactor[prot_chain_name]

            features = sample["protein"].setdefault("features", {})
            features["pI"] = pI
            features["mean_bfactor"] = mean_b
            features["aa_composition"] = aa_comp

            tmp = path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(sample, f, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
            totals["n_updated"] += 1

            if pI is not None:
                per_tier_pi[tier].append(pI)
            else:
                n_pi_null += 1
            if mean_b is not None:
                per_tier_bfactor[tier].append(mean_b)
            else:
                n_bfactor_null += 1
                samples_per_method_bfactor_null[
                    sample["data_availability"].get("experimental_method", "unknown")
                ] += 1
            if aa_comp:
                for aa, frac in aa_comp.items():
                    aa_composition_sums[aa] += frac
                aa_n += 1

    aa_avg = {aa: aa_composition_sums[aa] / aa_n for aa in STANDARD_AAS} if aa_n else {}

    report = render_report(totals, per_tier_pi, per_tier_bfactor, aa_avg,
                           n_bfactor_null, n_pi_null,
                           samples_per_method_bfactor_null)
    report_path = args.stats_dir / "protein_features_report.md"
    report_path.write_text(report, encoding="utf-8")

    print()
    print(f"Samples scanned   : {totals['n_samples']}")
    print(f"Updated           : {totals['n_updated']}")
    print(f"Resume-skipped    : {totals['n_skipped_resume']}")
    print(f"PDBs fully skipped: {totals['n_pdbs_skipped_all_done']}")
    print(f"Parse failures    : {totals['n_failed_parse']}")
    print(f"Raw lookup misses : {totals['n_failed_raw_lookup']}")
    print(f"Chain missing     : {totals['n_chain_missing']}")
    print(f"pI null           : {n_pi_null}")
    print(f"mean_bfactor null : {n_bfactor_null}")
    print(f"Report            : {report_path}")


if __name__ == "__main__":
    main()
