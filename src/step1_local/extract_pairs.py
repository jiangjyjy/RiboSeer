"""Stage 1.2: pair enumeration + interface computation + sample JSON export.

For every raw structure file:
  1. Parse (gemmi handles both .pdb and .cif uniformly)
  2. Assign label_seq_id so residue indices are consistent
  3. Classify chains (dominant-polymer rule) into protein / rna / dna / other
  4. Build a NeighborSearch atom index over the whole model
  5. Enumerate every (protein_chain, rna_chain) combination
  6. For each pair, collect heavy-atom contacts < --contact-cutoff (default 4.5 Å)
  7. Pairs with ≥1 contact become a sample; pairs with 0 contacts are skipped
  8. Each sample is saved as `samples/{pdb}_{prot}_{rna}.json` with the schema
     from the RiboSeer spec; `index.csv` aggregates per-sample headline info;
     `failed.log` records parse failures.

The sample JSON's `data_availability.quality_tier` is set using the soft tier
decision from stage 1.1 (see _common.quality_tier).

Usage:
    python extract_pairs.py \
        --raw-dir /path/to/riboseer/data/raw \
        --processed-dir /path/to/riboseer/data/processed \
        --stats-dir /path/to/riboseer/data/stats
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from _common import (  # noqa: E402  — fix_windows_dll_path runs on import
    classify_chain_dominant,
    classify_residue,
    normalize_exp_method,
    quality_tier,
    sid_to_filename,
    STANDARD_RNA,
)
import gemmi  # noqa: E402
from tqdm import tqdm  # noqa: E402


INDEX_FIELDS = [
    "sample_id", "pdb_id", "protein_chain", "rna_chain",
    "protein_length", "rna_length",
    "n_binding_protein_residues", "n_binding_rna_nucleotides",
    "n_contact_pairs",
    "resolution", "exp_method", "quality_tier", "rna_has_modification",
]


# Fallback 1-letter code for modified RNA bases that gemmi doesn't map cleanly.
# Keys come from scan_dataset.py top-20 modifications in this dataset.
RNA_MOD_FALLBACK = {
    "PSU": "U", "OMG": "G", "OMC": "C", "OMU": "U", "5MU": "U",
    "2MG": "G", "MA6": "A", "1MA": "A", "7MG": "G", "M2G": "G",
    "6MZ": "A", "UR3": "U", "H2U": "U", "4SU": "U", "2MA": "A",
    "4OC": "C", "1MG": "G", "MIA": "A", "5MC": "C",
    "I": "A",  # inosine — pairs like G but originates from A
    "N": "N",
}

PROT_UNKNOWN = "X"
RNA_UNKNOWN = "N"


# ---------- sequence / residue extraction -----------------------------------


def _one_letter(res_name: str, is_rna: bool) -> str:
    """Single-letter code for a residue. Uses gemmi first, then a fallback map."""
    name = res_name.strip().upper()
    info = gemmi.find_tabulated_residue(name)
    if info is not None and info.one_letter_code:
        code = info.one_letter_code.upper()
        if code and code != " ":
            return code
    if is_rna and name in RNA_MOD_FALLBACK:
        return RNA_MOD_FALLBACK[name]
    return RNA_UNKNOWN if is_rna else PROT_UNKNOWN


def extract_chain_sequence(chain, is_rna: bool) -> tuple[str, list[int], list[str]]:
    """Build sequence string (with gaps) and sorted resolved-residue list.

    Only residues whose category matches the chain type contribute to the
    sequence — stray ligands/waters that gemmi happens to merge into a polymer
    chain are ignored. `label_seq` is the canonical 1-based position.

    Gaps (positions in [1, max_label_seq] with no residue of the expected type)
    are written as '-' so sequence length == max_label_seq and resolved indices
    stay meaningful.

    Returns (sequence_str, resolved_label_seqs_sorted, modifications_seen).
    """
    positions: dict[int, str] = {}
    modifications: list[str] = []
    expected = "rna" if is_rna else "protein"
    for res in chain:
        if res.label_seq is None:
            continue
        cat, is_mod = classify_residue(res.name)
        if cat != expected:
            continue
        letter = _one_letter(res.name, is_rna)
        positions[res.label_seq] = letter
        if is_rna and is_mod:
            modifications.append(res.name.strip().upper())

    if not positions:
        return "", [], modifications

    max_pos = max(positions)
    seq = ["-"] * max_pos
    for pos, letter in positions.items():
        seq[pos - 1] = letter
    resolved = sorted(positions)
    return "".join(seq), resolved, modifications


def gc_content(rna_sequence: str) -> float | None:
    """Fraction of G+C among the resolved A/U/G/C positions (gaps and N excluded)."""
    counted = [c for c in rna_sequence if c in "AUGC"]
    if not counted:
        return None
    gc = sum(1 for c in counted if c in "GC")
    return round(gc / len(counted), 4)


# ---------- metadata --------------------------------------------------------


def read_exp_meta(structure) -> tuple[float | None, str]:
    """Return (resolution_or_None, normalized_exp_method)."""
    try:
        resolution = float(structure.resolution) if structure.resolution else None
        if resolution is not None and resolution <= 0:
            resolution = None
    except Exception:
        resolution = None

    exp_method = "unknown"
    try:
        if structure.info and "_exptl.method" in structure.info:
            exp_method = structure.info["_exptl.method"]
    except Exception:
        pass
    return resolution, normalize_exp_method(exp_method)


# ---------- main processing -------------------------------------------------


def process_file(path: Path, contact_cutoff: float) -> dict:
    """Return {pdb_id, tier, samples: [sample_dict, ...]} or {error: str}."""
    try:
        structure = gemmi.read_structure(str(path), merge_chain_parts=True)
    except Exception as e:
        return {"error": f"read_structure: {type(e).__name__}: {e}"}

    if len(structure) == 0:
        return {"error": "no_models"}

    try:
        structure.setup_entities()
        structure.assign_label_seq_id(True)
    except Exception as e:
        return {"error": f"label_seq: {type(e).__name__}: {e}"}

    model = structure[0]
    resolution, exp_method = read_exp_meta(structure)
    tier = quality_tier(resolution, exp_method)
    pdb_id = path.stem.lower()

    # Split chains by dominant type
    protein_chains: list = []
    rna_chains: list = []
    for chain in model:
        ctype = classify_chain_dominant(chain)
        if ctype == "protein":
            protein_chains.append(chain)
        elif ctype == "rna":
            rna_chains.append(chain)

    if not protein_chains or not rna_chains:
        return {
            "pdb_id": pdb_id,
            "tier": tier,
            "resolution": resolution,
            "exp_method": exp_method,
            "samples": [],
            "skip_reason": "no_protein_or_no_rna_chain",
        }

    # Build atom-level spatial index over the whole model.
    # NeighborSearch radius sets the max cell size; we'll still distance-filter
    # hits with contact_cutoff.
    ns = gemmi.NeighborSearch(model, structure.cell, max(5.0, contact_cutoff + 0.5)).populate()

    protein_chain_names = {ch.name for ch in protein_chains}

    # contacts[(prot_name, rna_name)][(prot_label_seq, rna_label_seq)] = min_dist
    contacts: dict[tuple[str, str], dict[tuple[int, int], float]] = defaultdict(dict)

    for rna_chain in rna_chains:
        for res in rna_chain:
            if res.label_seq is None:
                continue
            rna_cat, _ = classify_residue(res.name)
            if rna_cat != "rna":
                continue
            for atom in res:
                if atom.element.is_hydrogen:
                    continue
                try:
                    marks = ns.find_atoms(atom.pos, "\0", radius=contact_cutoff)
                except Exception:
                    continue
                for mark in marks:
                    cra = mark.to_cra(model)
                    if cra.chain.name not in protein_chain_names:
                        continue
                    if cra.atom.element.is_hydrogen:
                        continue
                    if cra.residue.label_seq is None:
                        continue
                    prot_cat, _ = classify_residue(cra.residue.name)
                    if prot_cat != "protein":
                        continue  # ignore ligand/ion atoms mixed into the chain
                    dist = atom.pos.dist(cra.atom.pos)
                    if dist >= contact_cutoff:
                        continue
                    key = (cra.chain.name, rna_chain.name)
                    pair_key = (cra.residue.label_seq, res.label_seq)
                    cur = contacts[key].get(pair_key)
                    if cur is None or dist < cur:
                        contacts[key][pair_key] = dist

    # Build sample objects only for pairs that actually have contacts
    samples = []
    for (prot_name, rna_name), pair_contacts in contacts.items():
        if not pair_contacts:
            continue

        prot_chain = next(ch for ch in protein_chains if ch.name == prot_name)
        rna_chain = next(ch for ch in rna_chains if ch.name == rna_name)

        prot_seq, prot_resolved, _ = extract_chain_sequence(prot_chain, is_rna=False)
        rna_seq, rna_resolved, rna_mods = extract_chain_sequence(rna_chain, is_rna=True)

        if not prot_seq or not rna_seq:
            continue  # degenerate chain, skip

        sorted_pairs = sorted(pair_contacts.items())  # ((p,r), d) sorted by (p, r)
        contact_pair_list = [[p, r] for (p, r), _ in sorted_pairs]
        contact_dist_list = [round(d, 3) for _, d in sorted_pairs]
        binding_prot = sorted({p for (p, _), _ in sorted_pairs})
        binding_rna = sorted({r for (_, r), _ in sorted_pairs})

        mods_unique = sorted(set(rna_mods))
        sample = {
            "sample_id": f"{pdb_id}_{prot_name}_{rna_name}",
            "source_pdb": pdb_id,
            "protein": {
                "chain_id": prot_name,
                "sequence": prot_seq,
                "length": len(prot_seq),
                "resolved_residues": prot_resolved,
                "features": {},
                "domain": None,
                "plddt_mean": None,
            },
            "rna": {
                "chain_id": rna_name,
                "sequence": rna_seq,
                "length": len(rna_seq),
                "resolved_residues": rna_resolved,
                "has_modification": bool(mods_unique),
                "modifications": mods_unique,
                "features": {
                    "gc_content": gc_content(rna_seq),
                    "secondary_structure_pred": None,
                    "structure_composition": None,
                },
                "rfam_family": None,
            },
            "interaction": {
                "binding_protein_residues": binding_prot,
                "binding_rna_nucleotides": binding_rna,
                "contact_pairs": contact_pair_list,
                "contact_distances": contact_dist_list,
                "distance_cutoff": contact_cutoff,
                "has_homolog": None,
                "eclip_count": None,
            },
            "data_availability": {
                "has_experimental_structure": True,
                "resolution": resolution,
                "experimental_method": exp_method,
                "quality_tier": tier,
                "msa_depth": None,
            },
            "split_info": {
                "protein_cluster_id": None,
                "rna_cluster_id": None,
                "split": None,
            },
        }
        samples.append(sample)

    return {
        "pdb_id": pdb_id,
        "tier": tier,
        "resolution": resolution,
        "exp_method": exp_method,
        "samples": samples,
        "n_protein_chains": len(protein_chains),
        "n_rna_chains": len(rna_chains),
        "n_candidate_pairs": len(protein_chains) * len(rna_chains),
    }


def _sample_to_index_row(sample: dict) -> dict:
    return {
        "sample_id": sample["sample_id"],
        "pdb_id": sample["source_pdb"],
        "protein_chain": sample["protein"]["chain_id"],
        "rna_chain": sample["rna"]["chain_id"],
        "protein_length": sample["protein"]["length"],
        "rna_length": sample["rna"]["length"],
        "n_binding_protein_residues": len(sample["interaction"]["binding_protein_residues"]),
        "n_binding_rna_nucleotides": len(sample["interaction"]["binding_rna_nucleotides"]),
        "n_contact_pairs": len(sample["interaction"]["contact_pairs"]),
        "resolution": sample["data_availability"]["resolution"],
        "exp_method": sample["data_availability"]["experimental_method"],
        "quality_tier": sample["data_availability"]["quality_tier"],
        "rna_has_modification": sample["rna"]["has_modification"],
    }


def _load_processed(jsonl_path: Path) -> tuple[dict, set]:
    """Return (pdb_id -> summary dict, set of pdb_ids already processed)."""
    summaries: dict[str, dict] = {}
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                summaries[rec["pdb_id"]] = rec
    return summaries, set(summaries.keys())


def _load_index_jsonl(path: Path) -> dict:
    """Return sample_id -> row dict, de-duplicated by last-seen."""
    rows: dict[str, dict] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows[row["sample_id"]] = row
    return rows


def main():
    parser = argparse.ArgumentParser(description="Stage 1.2: extract (protein, RNA) pair samples.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True,
                        help="root for processed/samples/*.json + index.csv + failed.log")
    parser.add_argument("--stats-dir", type=Path, required=True,
                        help="where pair_extraction_report.md goes")
    parser.add_argument("--contact-cutoff", type=float, default=4.5,
                        help="heavy atom distance threshold in Å (default 4.5)")
    parser.add_argument("--limit", type=int, default=0,
                        help="if >0, only process first N files (debug)")
    parser.add_argument("--pdb-ids", type=str, default="",
                        help="comma-separated PDB stems; if given, restrict processing to these")
    parser.add_argument("--force", action="store_true",
                        help="ignore processed_files.jsonl resume state and reprocess")
    args = parser.parse_args()

    samples_dir = args.processed_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    args.stats_dir.mkdir(parents=True, exist_ok=True)

    processed_path = args.processed_dir / "processed_files.jsonl"
    index_jsonl_path = args.processed_dir / "index.jsonl"
    index_csv_path = args.processed_dir / "index.csv"
    failed_log_path = args.processed_dir / "failed.log"

    # --- Load resume state ---------------------------------------------------
    if args.force:
        processed_summaries, processed_ids = {}, set()
        index_rows_by_sid = {}
    else:
        processed_summaries, processed_ids = _load_processed(processed_path)
        index_rows_by_sid = _load_index_jsonl(index_jsonl_path)

    all_files = sorted(
        p for p in args.raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".pdb", ".cif")
    )
    if args.pdb_ids:
        wanted = {s.strip().lower() for s in args.pdb_ids.split(",") if s.strip()}
        all_files = [p for p in all_files if p.stem.lower() in wanted]
    if args.limit:
        all_files = all_files[: args.limit]

    files_to_process = [p for p in all_files if p.stem.lower() not in processed_ids]
    skipped_count = len(all_files) - len(files_to_process)
    print(f"Found {len(all_files)} structure files; skipping {skipped_count} already in "
          f"processed_files.jsonl; processing {len(files_to_process)}")

    # --- Open append-only sinks ---------------------------------------------
    processed_fp = processed_path.open("a", encoding="utf-8")
    index_jsonl_fp = index_jsonl_path.open("a", encoding="utf-8")
    failed_fp = failed_log_path.open("a", encoding="utf-8")

    try:
        for p in tqdm(files_to_process, desc="extracting"):
            rec = process_file(p, args.contact_cutoff)
            pdb_id = p.stem.lower()

            if "error" in rec:
                summary = {
                    "pdb_id": pdb_id,
                    "status": "failed",
                    "error": rec["error"],
                    "tier": None,
                    "n_candidate_pairs": 0,
                    "n_samples": 0,
                }
                processed_summaries[pdb_id] = summary
                processed_fp.write(json.dumps(summary, ensure_ascii=False) + "\n")
                processed_fp.flush()
                failed_fp.write(f"{p.name}\t{rec['error']}\n")
                failed_fp.flush()
                continue

            n_candidate = rec.get("n_candidate_pairs", 0)
            samples = rec.get("samples", [])

            for sample in samples:
                filename = sid_to_filename(sample["sample_id"]) + ".json"
                out_path = samples_dir / filename
                with out_path.open("w", encoding="utf-8") as sf:
                    json.dump(sample, sf, indent=2, ensure_ascii=False)
                row = _sample_to_index_row(sample)
                index_rows_by_sid[row["sample_id"]] = row
                index_jsonl_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            index_jsonl_fp.flush()

            status = "ok" if samples else (
                "skipped_no_polymer"
                if rec.get("skip_reason") == "no_protein_or_no_rna_chain"
                else "ok_no_contacts"
            )
            summary = {
                "pdb_id": pdb_id,
                "status": status,
                "error": None,
                "tier": rec["tier"],
                "resolution": rec.get("resolution"),
                "exp_method": rec.get("exp_method"),
                "n_protein_chains": rec.get("n_protein_chains", 0),
                "n_rna_chains": rec.get("n_rna_chains", 0),
                "n_candidate_pairs": n_candidate,
                "n_samples": len(samples),
            }
            processed_summaries[pdb_id] = summary
            processed_fp.write(json.dumps(summary, ensure_ascii=False) + "\n")
            processed_fp.flush()
    finally:
        processed_fp.close()
        index_jsonl_fp.close()
        failed_fp.close()

    # --- Re-materialize index.csv from de-duplicated in-memory state --------
    sorted_rows = [index_rows_by_sid[k] for k in sorted(index_rows_by_sid)]
    with index_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(sorted_rows)

    # --- Aggregate counters for the report ----------------------------------
    totals = {
        "files": len(all_files),
        "parsed": 0,
        "failed": 0,
        "files_skipped_no_polymer": 0,
        "candidate_pairs": 0,
        "contact_pairs": 0,
    }
    tier_file_counts: Counter = Counter()
    failed_entries: list[dict] = []
    per_file_summary: list[dict] = []
    for pdb_id, summary in processed_summaries.items():
        if summary["status"] == "failed":
            totals["failed"] += 1
            failed_entries.append({"file": pdb_id, "error": summary.get("error", "")})
            continue
        totals["parsed"] += 1
        tier_file_counts[summary.get("tier") or "unknown"] += 1
        totals["candidate_pairs"] += summary.get("n_candidate_pairs", 0)
        totals["contact_pairs"] += summary.get("n_samples", 0)
        if summary["status"] == "skipped_no_polymer":
            totals["files_skipped_no_polymer"] += 1
        per_file_summary.append({
            "pdb_id": pdb_id,
            "tier": summary.get("tier") or "unknown",
            "candidate_pairs": summary.get("n_candidate_pairs", 0),
            "contact_pairs": summary.get("n_samples", 0),
        })

    tier_sample_counts: Counter = Counter(
        row["quality_tier"] for row in index_rows_by_sid.values()
    )

    report = render_report(totals, tier_file_counts, tier_sample_counts,
                           failed_entries, per_file_summary)
    report_path = args.stats_dir / "pair_extraction_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nWrote {len(index_rows_by_sid)} samples-index rows to {index_csv_path}")
    print(f"Samples dir: {samples_dir}")
    print(f"Resume log: {processed_path}  ({len(processed_summaries)} files recorded)")
    print(f"Failed log: {failed_log_path}  ({totals['failed']} entries)")
    print(f"Report:     {report_path}")
    print(
        f"Totals: parsed={totals['parsed']} candidate_pairs={totals['candidate_pairs']} "
        f"contact_pairs={totals['contact_pairs']} failed={totals['failed']}"
    )


def render_report(totals, tier_files, tier_samples, failed, per_file):
    failed_count = totals.get("failed", len(failed))
    lines = [
        "# Pair Extraction Report (Stage 1.2)",
        "",
        f"- Total input files: **{totals['files']}**",
        f"- Parsed successfully: **{totals['parsed']}**",
        f"- Parse failures: **{failed_count}**",
        f"- Files skipped for having no protein or no RNA chain: **{totals['files_skipped_no_polymer']}**",
        f"- Total candidate pairs (n_prot × n_rna enumeration): **{totals['candidate_pairs']}**",
        f"- Total pairs with contacts (= final sample count): **{totals['contact_pairs']}**",
    ]
    if totals["candidate_pairs"]:
        rate = totals["contact_pairs"] / totals["candidate_pairs"]
        lines.append(f"- Contact-filter pass rate: **{rate:.1%}**")

    lines += ["", "## Quality tier distribution (by file)"]
    for tier in ("strict", "standard", "low", "discard"):
        lines.append(f"- {tier}: {tier_files.get(tier, 0)}")
    lines += ["", "## Quality tier distribution (by final sample)"]
    for tier in ("strict", "standard", "low", "discard"):
        lines.append(f"- {tier}: {tier_samples.get(tier, 0)}")

    # Pair-per-file distribution
    if per_file:
        pair_hist = Counter()
        for row in per_file:
            pair_hist[row["contact_pairs"]] += 1
        lines += ["", "## Distribution of samples produced per file (top 15)"]
        for k in sorted(pair_hist)[:15]:
            lines.append(f"- {k} samples: {pair_hist[k]} files")
        if len(pair_hist) > 15:
            lines.append("- (tail omitted)")
        heaviest = sorted(per_file, key=lambda r: r["contact_pairs"], reverse=True)[:5]
        lines += ["", "## Top 5 files by number of samples produced"]
        for row in heaviest:
            lines.append(
                f"- {row['pdb_id']} ({row['tier']}): "
                f"{row['contact_pairs']} samples / {row['candidate_pairs']} candidate"
            )

    if failed:
        lines += ["", "## Files that failed to parse"]
        for fail in failed[:30]:
            lines.append(f"- {fail['file']}: {fail['error']}")
        if len(failed) > 30:
            lines.append(f"- ... ({len(failed) - 30} more)")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
