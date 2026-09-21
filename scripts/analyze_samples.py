"""Difficulty / quality distribution report for the pocket dataset.

Reads every ``<sample_id>.json`` under ``--input-dir`` and reports, for
each difficulty dimension, ``min / max / mean / median``, a percentile
ladder (p5..p99) and an ASCII histogram, plus categorical breakdowns
(domain, experimental method, quality tier) and per-split counts.

Dimensions
----------
  1. protein.length
  2. rna.length
  3. |interaction.binding_protein_residues|            (GT binding count)
  4. binding ratio = |GT| / protein.length
  5. data_availability.resolution                       (angstrom)
  6. protein.domain                                     (categorical)
  7. split (train/val/test) - from splits.json or splits/*.txt

It then prints a *recommended threshold* block (data-driven: aimed at
keeping >= ~2000 samples with train >= 1000 / test >= 400 while shaving
the hard tail) and the same combo ladder as ``filter_quality.py`` so
the two scripts agree by construction.

Usage
-----
::

    python scripts/analyze_samples.py \\
        --input-dir data/processed_filtered_100/samples/ \\
        --splits-json data/processed/splits.json

Read-only: writes nothing unless ``--json OUT.json`` is given.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Optional

# Reuse the exact extractors / predicate the real filter uses, so the
# report and filter_quality.py can never drift.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.filter_quality import (  # noqa: E402
    DEFAULT_COMBOS, combo_table, domain_of, gt_binding_count,
    load_splits_map, method_of, protein_length_of, resolution_of,
    rna_length_of, scan_dir, tier_of,
)

PCTS = (5, 10, 25, 50, 75, 90, 95, 99)


def _num_summary(name: str, vals: list[float], unit: str = "") -> dict:
    """min/max/mean/median + percentile ladder for a numeric series."""
    vals = [v for v in vals if v is not None]
    if not vals:
        print(f"\n## {name}: no data")
        return {"n": 0}
    sv = sorted(vals)
    n = len(sv)

    def pct(q: int) -> float:
        # nearest-rank percentile (robust, no interpolation surprises)
        idx = min(n - 1, max(0, round(q / 100 * n) - 1))
        return sv[idx]

    summ = {
        "n": n,
        "min": round(min(sv), 4),
        "max": round(max(sv), 4),
        "mean": round(statistics.fmean(sv), 4),
        "median": round(statistics.median(sv), 4),
        "stdev": round(statistics.pstdev(sv), 4) if n > 1 else 0.0,
        "pct": {f"p{q}": round(pct(q), 4) for q in PCTS},
    }
    u = f" {unit}" if unit else ""
    print(f"\n## {name}")
    print(f"  n={n}  min={summ['min']}{u}  max={summ['max']}{u}  "
          f"mean={summ['mean']}{u}  median={summ['median']}{u}  "
          f"stdev={summ['stdev']}")
    print("  percentiles: " + "  ".join(
        f"{k}={v}" for k, v in summ["pct"].items()))
    _histogram(sv)
    return summ


def _histogram(sv: list[float], bins: int = 12, width: int = 50) -> None:
    lo, hi = sv[0], sv[-1]
    if hi <= lo:
        print(f"  [all = {lo}]  ({len(sv)})")
        return
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in sv:
        b = min(bins - 1, int((v - lo) / step))
        counts[b] += 1
    mx = max(counts) or 1
    for i, c in enumerate(counts):
        a = lo + i * step
        b = a + step
        bar = "#" * round(c / mx * width)
        print(f"  [{a:8.1f},{b:8.1f}) {c:6d} | {bar}")


def _cat_summary(name: str, vals: list[str], top: int = 15) -> dict:
    counts: dict[str, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    print(f"\n## {name}  ({len(counts)} distinct, n={len(vals)})")
    for k, c in ordered[:top]:
        print(f"  {k:<24} {c:6d}  ({100.0 * c / len(vals):5.1f}%)")
    if len(ordered) > top:
        rest = sum(c for _, c in ordered[top:])
        print(f"  ... {len(ordered) - top} more            {rest:6d}")
    return dict(ordered)


def _recommend(p_sum, gt_sum, res_sum, n_total) -> None:
    print("\n" + "=" * 64)
    print("RECOMMENDED THRESHOLDS (data-driven)")
    print("=" * 64)
    print(f"  pool size = {n_total}.  Target: keep >= ~2000 "
          f"(train >= 1000, test >= 400),")
    print("  drop the hard tail (long protein / few GT / poor "
          "resolution).")

    def near(summ, key, fallback):
        return summ.get("pct", {}).get(key, fallback) if summ else fallback

    print("\n  protein.length:")
    print(f"    median={p_sum.get('median')}  "
          f"p75={near(p_sum, 'p75', '?')}  "
          f"p90={near(p_sum, 'p90', '?')}")
    print("    -> --max-protein-length 400  (keeps the bulk; long "
          "chains hurt Boltz-2/Chai-1)")
    print("\n  |GT| binding residues:")
    print(f"    median={gt_sum.get('median')}  "
          f"p10={near(gt_sum, 'p10', '?')}  "
          f"p25={near(gt_sum, 'p25', '?')}")
    print("    -> --min-gt-binding 5  (below ~5 the pocket is a noisy "
          "needle)")
    print("\n  resolution (angstrom, X-ray only):")
    print(f"    median={res_sum.get('median')}  "
          f"p75={near(res_sum, 'p75', '?')}  "
          f"p90={near(res_sum, 'p90', '?')}")
    print("    -> --max-resolution 3.5  (keep NMR/cryo-EM via default "
          "missing=keep)")
    print("\n  RNA length: keep the existing 20-100 nt band.")
    print("\n  Suggested first cut:")
    print("    --max-protein-length 400 --min-gt-binding 5 "
          "--max-resolution 3.5 \\")
    print("    --min-rna-length 20 --max-rna-length 100")
    print("  Inspect the combo ladder below, then commit one with "
          "filter_quality.py.")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path,
                   default=Path("data/processed_filtered_100/samples"))
    p.add_argument("--splits-json", type=Path,
                   default=Path("data/processed/splits.json"))
    p.add_argument("--json", type=Path, default=None,
                   help="also dump the machine-readable summary here.")
    p.add_argument("--no-combo-table", action="store_true")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not args.input_dir.is_dir():
        print(f"ERROR: --input-dir not a directory: {args.input_dir}",
              file=sys.stderr)
        return 1

    scanned = scan_dir(args.input_dir)
    splits_map = load_splits_map(args.splits_json)
    n = len(scanned)
    print(f"analyze_samples: {n} readable samples in {args.input_dir}")

    plens = [protein_length_of(s) for _, _, s in scanned]
    rlens = [rna_length_of(s) for _, _, s in scanned]
    gts = [float(gt_binding_count(s)) for _, _, s in scanned]
    ratios = []
    for _, _, s in scanned:
        pl = protein_length_of(s)
        if pl:
            ratios.append(gt_binding_count(s) / pl)
    ress = [resolution_of(s) for _, _, s in scanned]
    n_res_missing = sum(1 for r in ress if r is None)

    p_sum = _num_summary("protein.length", plens, "aa")
    _num_summary("rna.length", rlens, "nt")
    gt_sum = _num_summary("|GT| binding_protein_residues", gts)
    _num_summary("binding ratio (|GT|/protein.length)", ratios)
    res_sum = _num_summary(
        f"resolution (X-ray; {n_res_missing}/{n} null -> "
        f"NMR/cryo-EM/pred)", ress, "A")

    dom = _cat_summary("protein.domain", [domain_of(s)
                                          for _, _, s in scanned])
    meth = _cat_summary("experimental_method",
                        [method_of(s) for _, _, s in scanned])
    tier = _cat_summary("quality_tier",
                        [tier_of(s) for _, _, s in scanned])

    split_counts: dict[str, int] = {"train": 0, "val": 0,
                                    "test": 0, "unknown": 0}
    for sid, _, _ in scanned:
        sp = (splits_map.get(sid)
              or splits_map.get(sid.lower()) or "")
        split_counts[sp if sp in ("train", "val", "test")
                     else "unknown"] += 1
    print("\n## split (from splits.json)")
    for k in ("train", "val", "test", "unknown"):
        print(f"  {k:<8} {split_counts[k]:6d}")

    _recommend(p_sum, gt_sum, res_sum, n)

    if not args.no_combo_table:
        rows = combo_table(scanned, splits_map, DEFAULT_COMBOS,
                           dict(min_rna=20, max_rna=100))
        w = max(len(r["label"]) for r in rows)
        print("\n" + "=" * 64)
        print("COMBO LADDER (all keep RNA 20-100 nt)")
        print("=" * 64)
        print(f"  {'combo'.ljust(w)}  {'kept':>6} {'train':>6} "
              f"{'val':>5} {'test':>5}")
        for r in rows:
            print(f"  {r['label'].ljust(w)}  {r['kept']:>6} "
                  f"{r['train']:>6} {r['val']:>5} {r['test']:>5}")

    if args.json:
        args.json.write_text(json.dumps({
            "n": n,
            "protein_length": p_sum,
            "gt_binding": gt_sum,
            "resolution": res_sum,
            "resolution_missing": n_res_missing,
            "domain": dom,
            "experimental_method": meth,
            "quality_tier": tier,
            "split_counts": split_counts,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote machine summary -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
