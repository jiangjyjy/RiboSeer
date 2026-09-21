"""Stage 1.1: scan raw structure files and compute statistics only.

Reads every .pdb / .cif under --raw-dir, classifies chains, and writes
machine-readable JSON plus a human-readable Markdown report. Does NOT do any
processing — no pair extraction, no contact computation, no sample export.

Usage:
    python scan_dataset.py \
        --raw-dir /path/to/riboseer/data/raw \
        --out-dir /path/to/riboseer/data/stats \
        --plots
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from _common import (  # noqa: E402  (import after path fix inside _common)
    classify_residue,
    normalize_exp_method,
    quality_tier,
)
import gemmi  # noqa: E402
from tqdm import tqdm  # noqa: E402


STANDARD_RNA = {"A", "U", "G", "C"}


def classify_chain(chain) -> dict:
    """Classify one gemmi chain.

    Returns dict with keys: type, length, rna_mods (Counter as dict).
    Dominant polymer residue count drives the type; non-polymer-only chains
    become 'water' / 'ligand' / 'other'.
    """
    counts = Counter()
    rna_mods = Counter()
    for res in chain:
        cat, is_mod = classify_residue(res.name)
        counts[cat] += 1
        if cat == "rna" and is_mod:
            rna_mods[res.name.strip().upper()] += 1

    protein_n = counts["protein"]
    rna_n = counts["rna"]
    dna_n = counts["dna"]

    if protein_n == 0 and rna_n == 0 and dna_n == 0:
        if counts["water"]:
            ctype = "water"
        elif counts["ligand"]:
            ctype = "ligand"
        else:
            ctype = "other"
        return {"type": ctype, "length": 0, "rna_mods": {}}

    if protein_n >= rna_n and protein_n >= dna_n:
        ctype, length = "protein", protein_n
    elif rna_n >= dna_n:
        ctype, length = "rna", rna_n
    else:
        ctype, length = "dna", dna_n

    return {
        "type": ctype,
        "length": length,
        "rna_mods": dict(rna_mods) if ctype == "rna" else {},
    }


def scan_file(path: Path) -> dict:
    """Return stats dict for one structure file, or {error: ...} on failure."""
    try:
        structure = gemmi.read_structure(str(path), merge_chain_parts=True)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    if len(structure) == 0:
        return {"error": "no_models"}

    try:
        resolution = float(structure.resolution) if structure.resolution else None
        if resolution == 0:
            resolution = None
    except Exception:
        resolution = None

    exp_method = "unknown"
    try:
        if structure.info and "_exptl.method" in structure.info:
            exp_method = structure.info["_exptl.method"].upper()
    except Exception:
        pass

    model = structure[0]
    chains_info = []
    for chain in model:
        info = classify_chain(chain)
        info["chain_id"] = chain.name
        chains_info.append(info)

    type_counts = Counter(c["type"] for c in chains_info)

    return {
        "pdb_id": path.stem.lower(),
        "format": path.suffix.lower().lstrip("."),
        "resolution": resolution,
        "exp_method": exp_method,
        "n_chains": len(chains_info),
        "n_protein": type_counts["protein"],
        "n_rna": type_counts["rna"],
        "n_dna": type_counts["dna"],
        "n_ligand": type_counts["ligand"],
        "n_water": type_counts["water"],
        "n_other": type_counts["other"],
        "chains": chains_info,
    }


def _stats(arr):
    if not arr:
        return None
    return {
        "n": len(arr),
        "min": min(arr),
        "max": max(arr),
        "mean": round(statistics.mean(arr), 2),
        "median": statistics.median(arr),
        "p25": round(statistics.quantiles(arr, n=4)[0], 2) if len(arr) >= 4 else None,
        "p75": round(statistics.quantiles(arr, n=4)[2], 2) if len(arr) >= 4 else None,
    }


def aggregate(per_file: list[dict]) -> dict:
    protein_lens = [
        c["length"] for r in per_file for c in r["chains"] if c["type"] == "protein"
    ]
    rna_lens = [c["length"] for r in per_file for c in r["chains"] if c["type"] == "rna"]

    fmt_dist = Counter(r["format"] for r in per_file)
    exp_dist = Counter(normalize_exp_method(r["exp_method"]) for r in per_file)
    tier_dist = Counter(
        quality_tier(r["resolution"], r["exp_method"]) for r in per_file
    )

    resolutions = [r["resolution"] for r in per_file if r["resolution"] is not None]

    n_prot_dist = Counter(r["n_protein"] for r in per_file)
    n_rna_dist = Counter(r["n_rna"] for r in per_file)

    clean_11 = sum(1 for r in per_file if r["n_protein"] == 1 and r["n_rna"] == 1)
    has_dna = sum(1 for r in per_file if r["n_dna"] > 0)
    has_ligand = sum(1 for r in per_file if r["n_ligand"] > 0)
    zero_rna = sum(1 for r in per_file if r["n_rna"] == 0)
    zero_protein = sum(1 for r in per_file if r["n_protein"] == 0)

    rna_mod_global = Counter()
    n_files_with_mod = 0
    for r in per_file:
        has_any = False
        for c in r["chains"]:
            if c["type"] == "rna":
                for mod, cnt in c["rna_mods"].items():
                    rna_mod_global[mod] += cnt
                    has_any = True
        if has_any:
            n_files_with_mod += 1

    return {
        "format_distribution": dict(fmt_dist),
        "experimental_method_distribution": dict(exp_dist),
        "quality_tier_distribution": dict(tier_dist),
        "resolution_stats": _stats(resolutions),
        "protein_chain_count_distribution": dict(sorted(n_prot_dist.items())),
        "rna_chain_count_distribution": dict(sorted(n_rna_dist.items())),
        "protein_length_stats": _stats(protein_lens),
        "rna_length_stats": _stats(rna_lens),
        "clean_1to1_complexes": clean_11,
        "has_dna_complexes": has_dna,
        "has_ligand_complexes": has_ligand,
        "files_with_zero_rna_chains": zero_rna,
        "files_with_zero_protein_chains": zero_protein,
        "rna_modifications_total_unique": len(rna_mod_global),
        "rna_modifications_total_occurrences": sum(rna_mod_global.values()),
        "files_with_rna_modification": n_files_with_mod,
        "rna_modifications_top20": dict(rna_mod_global.most_common(20)),
    }


def render_markdown(overview: dict, n_files: int, n_parsed: int, failed: list) -> str:
    lines = [
        "# Dataset Overview (Stage 1.1)",
        "",
        f"- Total files: **{n_files}**",
        f"- Parsed successfully: **{n_parsed}**",
        f"- Parse failures: **{len(failed)}**",
        "",
        "## Format distribution",
    ]
    for fmt, n in overview["format_distribution"].items():
        lines.append(f"- `.{fmt}`: {n}")
    lines += ["", "## Experimental method distribution"]
    for m, n in overview["experimental_method_distribution"].items():
        lines.append(f"- {m}: {n}")
    lines += ["", "## Quality tier distribution (soft tiering by resolution + experimental method; used by stage 1.2)"]
    tier_labels = {
        "strict": "strict   (res ≤ 3.5 Å)",
        "standard": "standard (3.5 < res ≤ 4.0 Å)",
        "low": "low      (4.0 < res ≤ 6.0 Å)",
        "discard": "discard  (res > 6.0 Å / NMR / no res)",
    }
    for tier in ("strict", "standard", "low", "discard"):
        n = overview["quality_tier_distribution"].get(tier, 0)
        lines.append(f"- {tier_labels[tier]}: {n}")
    lines += ["", "## Resolution statistics (files with a resolution only)"]
    r = overview["resolution_stats"]
    if r:
        lines.append(
            f"- n={r['n']}, min={r['min']}, max={r['max']}, "
            f"mean={r['mean']}, median={r['median']}, p25={r['p25']}, p75={r['p75']}"
        )
    else:
        lines.append("- (no data)")
    lines += ["", "## Protein chains per file distribution"]
    for k, n in overview["protein_chain_count_distribution"].items():
        lines.append(f"- {k} protein chains: {n} files")
    lines += ["", "## RNA chains per file distribution"]
    for k, n in overview["rna_chain_count_distribution"].items():
        lines.append(f"- {k} RNA chains: {n} files")
    lines += ["", "## Chain length statistics"]
    p = overview["protein_length_stats"]
    if p:
        lines.append(
            f"- Protein chains: n={p['n']}, min={p['min']}, max={p['max']}, "
            f"mean={p['mean']}, median={p['median']}, p25={p['p25']}, p75={p['p75']}"
        )
    rn = overview["rna_length_stats"]
    if rn:
        lines.append(
            f"- RNA chains: n={rn['n']}, min={rn['min']}, max={rn['max']}, "
            f"mean={rn['mean']}, median={rn['median']}, p25={rn['p25']}, p75={rn['p75']}"
        )
    lines += ["", "## Complex composition"]
    lines.append(f"- Clean 1:1 (exactly 1 protein + 1 RNA): {overview['clean_1to1_complexes']} files")
    lines.append(f"- Contains DNA chains: {overview['has_dna_complexes']}")
    lines.append(f"- Contains ligand chains: {overview['has_ligand_complexes']}")
    lines.append(f"- 0 RNA chains (likely to be discarded): {overview['files_with_zero_rna_chains']}")
    lines.append(f"- 0 protein chains (likely to be discarded): {overview['files_with_zero_protein_chains']}")
    lines += ["", "## Modified RNA bases"]
    lines.append(f"- Files with modified bases: {overview['files_with_rna_modification']}")
    lines.append(f"- Number of unique modified base types: {overview['rna_modifications_total_unique']}")
    lines.append(f"- Total occurrences of modified bases: {overview['rna_modifications_total_occurrences']}")
    lines.append("- Top 20 most common modified bases:")
    for name, cnt in overview["rna_modifications_top20"].items():
        lines.append(f"  - {name}: {cnt}")
    if failed:
        lines += ["", "## Files that failed to parse"]
        for f in failed[:30]:
            lines.append(f"- {f['file']}: {f['error']}")
        if len(failed) > 30:
            lines.append(f"- ... ({len(failed) - 30} more)")
    lines += ["", "## Key conclusions (auto-flagged)"]
    conclusions = []
    if overview["files_with_zero_rna_chains"] > 0:
        conclusions.append(
            f"⚠️  {overview['files_with_zero_rna_chains']} files have no RNA chain; they will be filtered out in stage 1.2"
        )
    if overview["files_with_zero_protein_chains"] > 0:
        conclusions.append(
            f"⚠️  {overview['files_with_zero_protein_chains']} files have no protein chain; they will be filtered out in stage 1.2"
        )
    if p and p["max"] > 2000:
        conclusions.append(f"ℹ️  Longest protein chain is {p['max']} residues; consider whether such huge chains should be trimmed")
    if rn and rn["max"] > 2000:
        conclusions.append(
            f"ℹ️  Longest RNA chain is {rn['max']} nucleotides, very likely ribosomal 23S/28S rRNA"
        )
    if not conclusions:
        conclusions.append("(no automatic flags; inspect the histograms manually)")
    lines += [f"- {c}" for c in conclusions]
    return "\n".join(lines) + "\n"


def save_plots(per_file, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable: {e}")
        return

    protein_lens = [c["length"] for r in per_file for c in r["chains"] if c["type"] == "protein"]
    rna_lens = [c["length"] for r in per_file for c in r["chains"] if c["type"] == "rna"]
    resolutions = [r["resolution"] for r in per_file if r["resolution"] is not None]

    pairs = [
        (protein_lens, "protein_length", "residues"),
        (rna_lens, "rna_length", "nucleotides"),
        (resolutions, "resolution", "Å"),
    ]
    for arr, name, unit in pairs:
        if not arr:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(arr, bins=60)
        ax.set_title(f"{name} distribution")
        ax.set_xlabel(f"{name} ({unit})")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{name}.png", dpi=120)
        plt.close(fig)
    print("[plots] saved histograms to", out_dir)


def main():
    parser = argparse.ArgumentParser(description="Stage 1.1: scan raw structure dataset.")
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="directory containing .pdb / .cif files")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="output directory for dataset_overview.{json,md}")
    parser.add_argument("--plots", action="store_true", help="also save histogram PNGs")
    parser.add_argument("--limit", type=int, default=0,
                        help="if >0, only scan first N files (for debugging)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in args.raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".pdb", ".cif")
    )
    if args.limit:
        files = files[: args.limit]
    print(f"Found {len(files)} structure files under {args.raw_dir}")

    per_file = []
    failed = []
    for p in tqdm(files, desc="scanning"):
        rec = scan_file(p)
        if "error" in rec:
            failed.append({"file": p.name, "error": rec["error"]})
        else:
            per_file.append(rec)

    overview = aggregate(per_file)
    payload = {
        "n_files": len(files),
        "n_parsed": len(per_file),
        "n_failed": len(failed),
        "overview": overview,
        "per_file": per_file,
        "failed": failed,
    }

    json_path = args.out_dir / "dataset_overview.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    md_path = args.out_dir / "dataset_overview.md"
    md = render_markdown(overview, len(files), len(per_file), failed)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(md)

    if args.plots:
        save_plots(per_file, args.out_dir)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Parsed {len(per_file)} / Failed {len(failed)}")


if __name__ == "__main__":
    main()
