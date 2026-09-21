"""Stage 1.3a: recompute and normalize RNA basic features in every sample JSON.

This is the canonical producer for the locally-computable RNA feature set:

  - `rna.length`                       — `len(rna.sequence)` (incl. gap positions)
  - `rna.features.gc_content`          — G+C / (A+U+G+C), excluding gaps and 'N'
  - `rna.has_modification`             — preserved from stage 1.2 (can't be
                                         re-derived from the canonicalized
                                         sequence, since modifications are
                                         already mapped to standard bases).
  - `rna.modifications`                — preserved from stage 1.2

Server-side fields are left as `null`:

  - `rna.features.secondary_structure_pred`
  - `rna.features.structure_composition`

Stage 1.2's `extract_pairs.py` already writes these fields. This script exists
to be the canonical stage-1.3a producer, to run idempotently (safe to re-invoke
whenever upstream parsing changes), and to emit a feature-distribution report.

Usage:
    python rna_features_basic.py \
        --processed-dir /path/to/riboseer/data/processed \
        --stats-dir /path/to/riboseer/data/stats
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from _common import sid_to_filename  # noqa: E402 — runs fix_windows_dll_path()
from tqdm import tqdm  # noqa: E402


# Fields we guarantee after this stage. Used both for writing and for validation.
RNA_FEATURE_KEYS = ("gc_content", "secondary_structure_pred", "structure_composition")


def compute_gc_content(sequence: str) -> float | None:
    """G+C fraction among resolved standard bases (A/U/G/C).

    Gaps ('-') and unknowns ('N') are excluded from both numerator and
    denominator — standard treatment, same as stage 1.2.
    """
    counted = [c for c in sequence if c in "AUGC"]
    if not counted:
        return None
    gc = sum(1 for c in counted if c in "GC")
    return round(gc / len(counted), 4)


def normalize_rna_block(rna: dict) -> tuple[dict, list[str]]:
    """Return (updated_rna, list_of_changed_fields).

    Idempotent: only fields that actually differ from the computed canonical
    value are mutated. `modifications` / `has_modification` are preserved as-is
    because they come from the original residue names in stage 1.2 and can't be
    recovered from the mapped-to-standard sequence alone.
    """
    changes: list[str] = []
    seq = rna.get("sequence", "")
    canonical_length = len(seq)
    if rna.get("length") != canonical_length:
        rna["length"] = canonical_length
        changes.append("length")

    canonical_gc = compute_gc_content(seq)
    features = rna.get("features")
    if not isinstance(features, dict):
        features = {}
        rna["features"] = features
        changes.append("features.{}")

    if features.get("gc_content") != canonical_gc:
        features["gc_content"] = canonical_gc
        changes.append("features.gc_content")

    # Server-side placeholders — set-if-missing so schema stays complete.
    for key in ("secondary_structure_pred", "structure_composition"):
        if key not in features:
            features[key] = None
            changes.append(f"features.{key}")

    # Preserve (but normalize types for) modification fields.
    if "has_modification" not in rna:
        rna["has_modification"] = bool(rna.get("modifications"))
        changes.append("has_modification")
    if "modifications" not in rna:
        rna["modifications"] = []
        changes.append("modifications")

    return rna, changes


def _quartiles(xs: list[float]) -> tuple:
    xs = sorted(x for x in xs if x is not None)
    n = len(xs)
    if not n:
        return (None,) * 5
    return xs[0], xs[n // 4], xs[n // 2], xs[3 * n // 4], xs[-1]


def render_report(totals: dict, per_tier_gc: dict, per_tier_len: dict,
                  mod_counts: Counter, modified_pct: float,
                  mods_hist: Counter) -> str:
    lines = [
        "# Stage 1.3a — RNA Basic Features Report",
        "",
        f"- Total samples scanned: **{totals['n_samples']}**",
        f"- Samples whose fields needed updating: **{totals['n_changed']}**",
        f"- Samples already fully consistent: **{totals['n_unchanged']}**",
        f"- Samples with no sequence, so GC was not computable (gc_content=null): **{totals['n_gc_null']}**",
        f"- `secondary_structure_pred` still null: **{totals['n_ss_null']}** (to be filled by server stage 1.3b)",
        f"- `structure_composition` still null: **{totals['n_sc_null']}** (to be filled by server stage 1.3b)",
        "",
        "## Length distribution (by tier)",
        "",
        "| tier | n | min | p25 | median | p75 | max |",
        "|------|---|-----|-----|--------|-----|-----|",
    ]
    for tier in ("strict", "standard", "low", "discard"):
        xs = per_tier_len.get(tier, [])
        n = len(xs)
        if not n:
            lines.append(f"| {tier} | 0 | - | - | - | - | - |")
            continue
        lo, p25, med, p75, hi = _quartiles(xs)
        lines.append(f"| {tier} | {n} | {lo} | {p25} | {med} | {p75} | {hi} |")

    lines += [
        "",
        "## GC content distribution (by tier)",
        "",
        "| tier | n | min | p25 | median | p75 | max |",
        "|------|---|-----|-----|--------|-----|-----|",
    ]
    for tier in ("strict", "standard", "low", "discard"):
        xs = per_tier_gc.get(tier, [])
        n = len(xs)
        if not n:
            lines.append(f"| {tier} | 0 | - | - | - | - | - |")
            continue
        lo, p25, med, p75, hi = _quartiles(xs)
        lines.append(
            f"| {tier} | {n} | {lo:.3f} | {p25:.3f} | {med:.3f} | {p75:.3f} | {hi:.3f} |"
        )

    lines += [
        "",
        "## Modified base distribution",
        "",
        f"- Samples containing modified bases: **{mod_counts['with_mod']}** "
        f"({modified_pct:.1%})",
        f"- Without modifications: **{mod_counts['without_mod']}**",
        "",
        "### Top 15 modified bases (by number of samples they appear in)",
        "",
    ]
    if mods_hist:
        for name, n in mods_hist.most_common(15):
            lines.append(f"- {name}: {n}")
    else:
        lines.append("- (none)")

    lines += [
        "",
        "## Notes",
        "",
        "- This stage's script is an **idempotent canonicalizer**: each invocation resets "
        "`rna.length` and `rna.features.gc_content` from `len(rna.sequence)` and a recomputed GC, "
        "but preserves `has_modification` / `modifications` (they come from the original residue names "
        "seen in stage 1.2 and cannot be recovered from the sequence after mapping to standard bases).",
        "- Server stage 1.3b (ViennaRNA) fills in `secondary_structure_pred` and "
        "`structure_composition`.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1.3a: normalize RNA basic features in sample JSONs."
    )
    parser.add_argument("--processed-dir", type=Path, required=True,
                        help="root containing samples/*.json and index.csv")
    parser.add_argument("--stats-dir", type=Path, required=True,
                        help="where rna_features_basic_report.md goes")
    parser.add_argument("--dry-run", action="store_true",
                        help="compute but do not write any sample files")
    args = parser.parse_args()

    samples_dir = args.processed_dir / "samples"
    index_csv = args.processed_dir / "index.csv"
    args.stats_dir.mkdir(parents=True, exist_ok=True)

    if not index_csv.exists():
        sys.exit(f"index.csv not found at {index_csv} — run stage 1.2 first")

    with index_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} sample rows from {index_csv}")

    totals = {
        "n_samples": len(rows),
        "n_changed": 0,
        "n_unchanged": 0,
        "n_gc_null": 0,
        "n_ss_null": 0,
        "n_sc_null": 0,
    }
    change_counter: Counter = Counter()

    per_tier_gc: dict[str, list[float]] = defaultdict(list)
    per_tier_len: dict[str, list[int]] = defaultdict(list)
    mod_counts = Counter()
    mods_hist: Counter = Counter()

    for row in tqdm(rows, desc="rna-feat", mininterval=1.0):
        sid = row["sample_id"]
        tier = row["quality_tier"]
        filename = sid_to_filename(sid) + ".json"
        sample_path = samples_dir / filename

        with sample_path.open("r", encoding="utf-8") as f:
            sample = json.load(f)

        _, changes = normalize_rna_block(sample["rna"])
        if changes:
            totals["n_changed"] += 1
            for c in changes:
                change_counter[c] += 1
            if not args.dry_run:
                tmp_path = sample_path.with_suffix(".json.tmp")
                with tmp_path.open("w", encoding="utf-8") as f:
                    json.dump(sample, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, sample_path)
        else:
            totals["n_unchanged"] += 1

        rna = sample["rna"]
        feats = rna["features"]
        gc = feats.get("gc_content")
        per_tier_len[tier].append(rna["length"])
        if gc is None:
            totals["n_gc_null"] += 1
        else:
            per_tier_gc[tier].append(gc)
        if feats.get("secondary_structure_pred") is None:
            totals["n_ss_null"] += 1
        if feats.get("structure_composition") is None:
            totals["n_sc_null"] += 1
        if rna.get("has_modification"):
            mod_counts["with_mod"] += 1
            for m in rna.get("modifications", []):
                mods_hist[m] += 1
        else:
            mod_counts["without_mod"] += 1

    modified_pct = (
        mod_counts["with_mod"] / totals["n_samples"] if totals["n_samples"] else 0.0
    )
    report = render_report(totals, per_tier_gc, per_tier_len,
                           mod_counts, modified_pct, mods_hist)
    report_path = args.stats_dir / "rna_features_basic_report.md"
    report_path.write_text(report, encoding="utf-8")

    print()
    print(f"Samples scanned  : {totals['n_samples']}")
    print(f"Changed          : {totals['n_changed']}")
    print(f"Unchanged        : {totals['n_unchanged']}")
    if change_counter:
        print("Change breakdown :")
        for field, n in change_counter.most_common():
            print(f"  {field}: {n}")
    print(f"GC null          : {totals['n_gc_null']}")
    print(f"SS pred null     : {totals['n_ss_null']} (server TODO)")
    print(f"SC null          : {totals['n_sc_null']} (server TODO)")
    print(f"Report           : {report_path}")
    if args.dry_run:
        print("(dry-run — no files were modified)")


if __name__ == "__main__":
    main()
