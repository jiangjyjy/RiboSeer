"""mmseqs2 cluster / split inspector for a sample subset.

Reads the leakage-safe cluster split produced upstream
(``data/processed/splits.json``) and reports, for any sample subset:

  - global cluster numbers from ``splits.json.stats``
  - subset size + number of distinct ``protein_cluster_id`` and
    ``rna_cluster_id`` it spans
  - cluster-size distribution within the subset (samples per cluster)
  - per-split (train/val/test/unknown) counts + clusters
  - leakage check at two levels:
      * (protein_cluster, rna_cluster) GROUP overlap between train and
        test - this is the splitting unit (see ``stats.n_groups``);
        must be empty for a leakage-safe split.
      * single-axis protein / rna cluster overlap - reported for
        context only; under a group-based split these CAN overlap
        legitimately (same protein cluster paired with different RNAs).
  - the largest clusters in the subset (handy for spotting dominators)

Subset selection (any one of these; ``input_dir`` and ``sample_list``
are unioned)::

    --input-dir   data/processed_quality/samples/      # filenames -> ids
    --sample-list path/to/ids.txt                       # one id per line
    --splits-json-only                                  # use the whole
                                                        # splits.json
                                                        # universe

If ``--apply-quality-filter`` is also given, ``--input-dir`` is filtered
in-memory using the same ``QualityThresholds`` as ``filter_quality.py``
(handy to inspect a quality cut before materialising it on disk).

Usage
-----
::

    # 1. inspect what filter_quality WILL produce, no disk writes
    python scripts/cluster_stats.py \\
        --input-dir data/processed_filtered_100/samples/ \\
        --apply-quality-filter \\
        --max-protein-length 400 --min-gt-binding 5 \\
        --max-resolution 3.5 \\
        --splits-json data/processed/splits.json

    # 2. after materialising:
    python scripts/cluster_stats.py \\
        --input-dir data/processed_quality/samples/ \\
        --splits-json data/processed/splits.json

    # 3. whole universe:
    python scripts/cluster_stats.py --splits-json-only

Read-only: writes nothing unless ``--json OUT.json`` is given.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.filter_quality import (  # noqa: E402
    QualityThresholds, scan_dir,
)


# ---- splits.json loader --------------------------------------------------


def load_splits_meta(splits_json: Path) -> dict:
    """Returns the parsed splits.json, or raises FileNotFoundError."""
    if not splits_json.is_file():
        raise FileNotFoundError(splits_json)
    return json.loads(splits_json.read_text(encoding="utf-8"))


def sample_meta(splits: dict, sid: str) -> dict:
    """Case-insensitive lookup of ``splits['samples'][sid]``;
    returns ``{}`` if missing (sample not in splits.json universe)."""
    s = splits.get("samples") or {}
    if sid in s:
        return s[sid] or {}
    low = sid.lower()
    for k, v in s.items():
        if k.lower() == low:
            return v or {}
    return {}


# ---- subset summary -------------------------------------------------------


def _stats(vals: list[int]) -> dict:
    if not vals:
        return {"n": 0, "min": None, "max": None,
                "mean": None, "median": None}
    return {
        "n": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": round(statistics.fmean(vals), 2),
        "median": round(statistics.median(vals), 1),
    }


def summarise_subset(splits: dict, ids: Iterable[str]) -> dict:
    """Crunch the cluster picture for the given sample ids.

    Returns a dict ready to print and to dump as JSON. Pure: only reads
    ``splits``. Samples missing from splits.json land in ``unknown``."""
    by_split: dict[str, list[str]] = {"train": [], "val": [],
                                      "test": [], "unknown": []}
    # cluster_id -> list of sample_ids (within subset), keyed by split
    pc_by_split: dict[str, dict[str, list[str]]] = {
        k: defaultdict(list) for k in by_split}
    rc_by_split: dict[str, dict[str, list[str]]] = {
        k: defaultdict(list) for k in by_split}
    # (prot_cluster, rna_cluster) GROUP -> samples, keyed by split.
    # The split was made at the group level, so this is what must be
    # disjoint between train and test.
    grp_by_split: dict[str, dict[tuple[str, str], list[str]]] = {
        k: defaultdict(list) for k in by_split}
    # subset-wide cluster -> count
    pc_all: dict[str, list[str]] = defaultdict(list)
    rc_all: dict[str, list[str]] = defaultdict(list)
    grp_all: dict[tuple[str, str], list[str]] = defaultdict(list)
    no_prot_cluster = no_rna_cluster = not_in_splits = 0

    for sid in ids:
        meta = sample_meta(splits, sid)
        if not meta:
            not_in_splits += 1
            by_split["unknown"].append(sid)
            continue
        sp = meta.get("split") or ""
        bucket = sp if sp in ("train", "val", "test") else "unknown"
        by_split[bucket].append(sid)
        pc = meta.get("protein_cluster_id")
        rc = meta.get("rna_cluster_id")
        if pc:
            pc_all[pc].append(sid)
            pc_by_split[bucket][pc].append(sid)
        else:
            no_prot_cluster += 1
        if rc:
            rc_all[rc].append(sid)
            rc_by_split[bucket][rc].append(sid)
        else:
            no_rna_cluster += 1
        if pc and rc:
            grp = (pc, rc)
            grp_all[grp].append(sid)
            grp_by_split[bucket][grp].append(sid)

    def _per_split(d):
        return {k: {"samples": len(by_split[k]),
                    "protein_clusters": len(pc_by_split[k]),
                    "rna_clusters": len(rc_by_split[k]),
                    "groups": len(grp_by_split[k])}
                for k in ("train", "val", "test", "unknown")}

    # Group-level leakage: the splitting unit, must be empty.
    grp_train = set(grp_by_split["train"])
    grp_test = set(grp_by_split["test"])
    grp_leak = sorted(grp_train & grp_test)
    # Single-axis "overlap" (context only - expected under a group split).
    pc_train = set(pc_by_split["train"])
    pc_test = set(pc_by_split["test"])
    rc_train = set(rc_by_split["train"])
    rc_test = set(rc_by_split["test"])
    pc_leak = sorted(pc_train & pc_test)
    rc_leak = sorted(rc_train & rc_test)

    # Cluster-size distribution within the subset
    pc_sizes = sorted((len(v) for v in pc_all.values()), reverse=True)
    rc_sizes = sorted((len(v) for v in rc_all.values()), reverse=True)

    return {
        "n_samples": sum(len(v) for v in by_split.values()),
        "not_in_splits_json": not_in_splits,
        "no_protein_cluster_id": no_prot_cluster,
        "no_rna_cluster_id": no_rna_cluster,
        "n_protein_clusters": len(pc_all),
        "n_rna_clusters": len(rc_all),
        "n_groups": len(grp_all),
        "protein_cluster_size_stats": _stats(pc_sizes),
        "rna_cluster_size_stats": _stats(rc_sizes),
        "per_split": _per_split(None),
        "leakage": {
            "group_overlap_count": len(grp_leak),
            "group_overlap_examples":
                [f"{p}/{r}" for p, r in grp_leak[:20]],
            "protein_cluster_overlap_count": len(pc_leak),
            "protein_cluster_overlap_ids": pc_leak[:20],
            "rna_cluster_overlap_count": len(rc_leak),
            "rna_cluster_overlap_ids": rc_leak[:20],
        },
        # keep raw maps for top-N + JSON dump
        "_pc_sizes": pc_sizes, "_rc_sizes": rc_sizes,
        "_pc_all": {k: len(v) for k, v in pc_all.items()},
        "_rc_all": {k: len(v) for k, v in rc_all.items()},
        "_by_split": by_split,
    }


# ---- pretty print --------------------------------------------------------


def _print_global(splits: dict) -> None:
    st = splits.get("stats") or {}
    print("\n## global (whole dataset, from splits.json.stats)")
    print(f"  n_samples                : {st.get('n_samples')}")
    print(f"  n_unique_prot_chains     : "
          f"{st.get('n_unique_prot_chains')}")
    print(f"  n_unique_rna_chains      : "
          f"{st.get('n_unique_rna_chains')}")
    print(f"  n_protein_clusters       : {st.get('n_prot_clusters')}")
    print(f"  n_rna_clusters           : {st.get('n_rna_clusters')}")
    print(f"  n_clusterable_samples    : "
          f"{st.get('n_clusterable_samples')}")
    print(f"  n_unclusterable_samples  : "
          f"{st.get('n_unclusterable_samples')}")
    print(f"  cluster groups (prot x rna) : "
          f"n={st.get('n_groups')}  min/median/max group size = "
          f"{st.get('min_group_size')}/"
          f"{st.get('median_group_size')}/"
          f"{st.get('max_group_size')}")
    cfg = splits.get("config") or {}
    print(f"  config: prot_min_seq_id={cfg.get('protein_min_seq_id')}  "
          f"rna_min_seq_id={cfg.get('rna_min_seq_id')}  "
          f"coverage={cfg.get('coverage')}  "
          f"split_ratio={cfg.get('split_ratio')}  "
          f"seed={cfg.get('seed')}")


def _print_subset(label: str, summ: dict, top_n: int = 10) -> None:
    print(f"\n## subset: {label}")
    print(f"  n_samples                : {summ['n_samples']}")
    if summ["not_in_splits_json"]:
        print(f"  not in splits.json       : "
              f"{summ['not_in_splits_json']}  (-> 'unknown')")
    if summ["no_protein_cluster_id"]:
        print(f"  no protein_cluster_id    : "
              f"{summ['no_protein_cluster_id']}")
    if summ["no_rna_cluster_id"]:
        print(f"  no rna_cluster_id        : "
              f"{summ['no_rna_cluster_id']}")
    print(f"  distinct protein clusters: {summ['n_protein_clusters']}")
    print(f"  distinct rna clusters    : {summ['n_rna_clusters']}")
    print(f"  distinct (prot,rna) groups: {summ['n_groups']}")

    ps = summ["protein_cluster_size_stats"]
    rs = summ["rna_cluster_size_stats"]
    print(f"  protein cluster sizes    : "
          f"min={ps['min']} median={ps['median']} mean={ps['mean']} "
          f"max={ps['max']}")
    print(f"  rna cluster sizes        : "
          f"min={rs['min']} median={rs['median']} mean={rs['mean']} "
          f"max={rs['max']}")

    print("\n  per-split breakdown")
    print(f"    {'split':<8} {'samples':>8} {'prot_clu':>9} "
          f"{'rna_clu':>8} {'groups':>7}")
    for k in ("train", "val", "test", "unknown"):
        s = summ["per_split"][k]
        print(f"    {k:<8} {s['samples']:>8} "
              f"{s['protein_clusters']:>9} {s['rna_clusters']:>8} "
              f"{s['groups']:>7}")

    lk = summ["leakage"]
    grp_ok = lk["group_overlap_count"] == 0
    print("\n  leakage check (train vs test)")
    print(f"    (prot,rna) GROUP overlap : "
          f"{lk['group_overlap_count']:>4}  "
          f"{'OK (split is group-disjoint)' if grp_ok else 'LEAK!'}"
          f"   <- the splitting unit; must be 0")
    if not grp_ok:
        print(f"      examples: {lk['group_overlap_examples']}")
    print(f"    protein-cluster overlap  : "
          f"{lk['protein_cluster_overlap_count']:>4}  "
          f"(expected under group split: same prot cluster can pair "
          f"with different RNAs)")
    print(f"    rna-cluster overlap      : "
          f"{lk['rna_cluster_overlap_count']:>4}  "
          f"(expected under group split: same RNA cluster can pair "
          f"with different proteins)")

    # Top dominator clusters in the subset
    pc_top = sorted(summ["_pc_all"].items(),
                    key=lambda kv: (-kv[1], kv[0]))[:top_n]
    rc_top = sorted(summ["_rc_all"].items(),
                    key=lambda kv: (-kv[1], kv[0]))[:top_n]
    print(f"\n  top-{top_n} protein clusters in subset")
    for k, c in pc_top:
        print(f"    {k:<14} {c:>5}")
    print(f"\n  top-{top_n} rna clusters in subset")
    for k, c in rc_top:
        print(f"    {k:<14} {c:>5}")


# ---- id collection -------------------------------------------------------


def _ids_from_dir(d: Path) -> list[str]:
    return sorted(p.stem for p in d.glob("*.json"))


def _ids_from_list(p: Path) -> list[str]:
    return [ln.strip() for ln in
            p.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]


def _apply_quality(input_dir: Path, thr: QualityThresholds) -> list[str]:
    """Run the QualityThresholds predicate over input_dir/*.json
    in memory and return kept sample_ids."""
    kept = []
    for sid, _jf, sample in scan_dir(input_dir):
        if thr.passes(sample)[0]:
            kept.append(sid)
    return kept


# ---- main ----------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits-json", type=Path,
                   default=Path("data/processed/splits.json"))
    p.add_argument("--input-dir", type=Path, default=None,
                   help="dir of <sample_id>.json; filename stems = ids.")
    p.add_argument("--sample-list", type=Path, default=None,
                   help="one sample_id per line (# comments ok).")
    p.add_argument("--splits-json-only", action="store_true",
                   help="use the entire splits.json universe as the "
                        "subset (sanity check / global view).")
    p.add_argument("--label", default=None,
                   help="header tag for the subset (defaults to the "
                        "source path).")
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--json", type=Path, default=None,
                   help="also dump machine-readable summary here.")

    # Optional quality filter (mirrors filter_quality.py flags)
    p.add_argument("--apply-quality-filter", action="store_true",
                   help="run filter_quality predicate over --input-dir "
                        "before computing cluster stats.")
    p.add_argument("--min-rna-length", type=int, default=20)
    p.add_argument("--max-rna-length", type=int, default=100)
    p.add_argument("--max-protein-length", type=int, default=400)
    p.add_argument("--min-protein-length", type=int, default=0)
    p.add_argument("--min-gt-binding", type=int, default=5)
    p.add_argument("--max-resolution", type=float, default=None)
    p.add_argument("--min-binding-ratio", type=float, default=0.0)
    p.add_argument("--drop-missing-resolution", action="store_true")
    p.add_argument("--quality-tier", default=None)
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    try:
        splits = load_splits_meta(args.splits_json)
    except FileNotFoundError:
        print(f"ERROR: splits.json not found: {args.splits_json}",
              file=sys.stderr)
        return 1

    _print_global(splits)

    # Build subset id list
    ids: list[str] = []
    label = args.label
    if args.splits_json_only:
        ids = list((splits.get("samples") or {}).keys())
        label = label or "splits.json universe"
    else:
        if args.input_dir:
            if not args.input_dir.is_dir():
                print(f"ERROR: --input-dir not a directory: "
                      f"{args.input_dir}", file=sys.stderr)
                return 1
            if args.apply_quality_filter:
                thr = QualityThresholds(
                    min_rna=args.min_rna_length,
                    max_rna=args.max_rna_length,
                    min_prot=args.min_protein_length,
                    max_prot=args.max_protein_length,
                    min_gt=args.min_gt_binding,
                    max_res=args.max_resolution,
                    min_ratio=args.min_binding_ratio,
                    drop_missing_res=args.drop_missing_resolution,
                    allowed_tiers=([t.strip() for t in
                                    args.quality_tier.split(",")
                                    if t.strip()]
                                   if args.quality_tier else None),
                )
                ids = _apply_quality(args.input_dir, thr)
                label = (label or
                         f"{args.input_dir} + quality filter")
            else:
                ids = _ids_from_dir(args.input_dir)
                label = label or str(args.input_dir)
        if args.sample_list:
            extra = _ids_from_list(args.sample_list)
            ids = sorted(set(ids) | set(extra))
            label = label or str(args.sample_list)

    if not ids:
        print("\nERROR: no subset selected; pass --input-dir, "
              "--sample-list, or --splits-json-only.", file=sys.stderr)
        return 1

    summ = summarise_subset(splits, ids)
    _print_subset(label, summ, top_n=args.top_n)

    if args.json:
        # Strip private "_" keys for the dump
        dump = {k: v for k, v in summ.items() if not k.startswith("_")}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(dump, indent=2),
                             encoding="utf-8")
        print(f"\nwrote machine summary -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
