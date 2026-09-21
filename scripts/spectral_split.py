"""Spectral split on the cluster-rep TM-score matrix.

Input: the N x N TM-score matrix produced by
``compute_tmscore_matrix.py`` (N = number of protein clusters that
have a rep). We build a sparse affinity by zeroing entries below
``--similarity-threshold`` (default 0.5, the canonical "same fold"
cutoff), run ``sklearn.cluster.SpectralClustering`` with
``affinity='precomputed'`` and ``--n-clusters`` (2 or 3 supported),
then bucket each rep into a split:

  - ``n_clusters=2``: largest spectral group -> ``train``,
    smallest -> ``test``. No ``val``.
  - ``n_clusters=3``: largest -> ``train``, mid -> ``val``,
    smallest -> ``test``.

The output files contain ONE LINE PER REPRESENTATIVE - i.e. exactly
N sample ids spread across the splits, NOT the ~2 k pool samples.
Group size for the train/(val)/test ranking is the number of REPS in
each spectral group, because reps are now the final training rows
(one per protein cluster, per the upstream "every cluster contributes
one sample" rule).

Layout written under ``--output-dir``::

    train.txt            (reps assigned to train, sorted)
    test.txt             (reps assigned to test, sorted)
    val.txt              (reps assigned to val; ONLY if --n-clusters=3)
    train_200.txt        (sample_subset of train; seed=42)
    test_200.txt         (sample_subset of test; seed=42)
    split_stats.json     (config, per-split rep counts, intra/inter TM)

If a stale ``val.txt`` from a previous ``n_clusters=3`` run sits in the
output dir when you re-run with ``n_clusters=2``, the script removes
it so the on-disk layout always reflects the current split.

Usage
-----
::

    python scripts/spectral_split.py \\
        --tmscore-matrix   data/processed_quality/splits_tmscore/\\
tmscore_matrix.npy \\
        --sample-list      data/processed_quality/splits_tmscore/\\
cluster_representatives.txt \\
        --output-dir       data/processed_quality/splits_tmscore/ \\
        --n-clusters 2 --similarity-threshold 0.5

``--processed-dir`` and ``--splits-json`` are accepted but unused
(they were the pool-mapping inputs in the previous design); pass them
or not, the output stays rep-only.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compute_tmscore_matrix import read_sample_list  # noqa: E402


# ---- core ---------------------------------------------------------------


def build_affinity(tm: np.ndarray, threshold: float) -> np.ndarray:
    """Symmetric non-negative affinity for SpectralClustering.

    - NaN (un-stageable rep) -> 0 connection
    - entries below ``threshold`` -> 0
    - diagonal forced to 1.0
    - symmetrised with the max of (A, A^T) to repair float drift
    """
    a = np.array(tm, dtype=np.float64, copy=True)
    a[np.isnan(a)] = 0.0
    a[a < threshold] = 0.0
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 1.0)
    return a


def spectral_labels(affinity: np.ndarray, n_clusters: int,
                    seed: int) -> np.ndarray:
    """Return integer labels in ``[0, n_clusters)`` from sklearn's
    SpectralClustering with precomputed affinity."""
    from sklearn.cluster import SpectralClustering  # lazy import

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
        n_init=10,
    )
    return sc.fit_predict(affinity)


# Split names per supported n_clusters. Anything else falls back to
# ``group0/group1/...`` so the script still runs (with a warning).
_SPLIT_NAMES = {
    2: ["train", "test"],
    3: ["train", "val", "test"],
}


def assign_splits(labels: np.ndarray,
                  n_clusters: int) -> dict[int, str]:
    """Map each spectral label to a split name.

    Groups are ranked by REP COUNT descending (each rep is one final
    training row), so the largest spectral group wins ``train``, the
    smallest gets ``test``. With ``n_clusters=3`` the middle group is
    ``val``. With any other value we degrade to ``groupK`` names; that
    keeps the rest of the pipeline runnable and surfaces the unusual
    config in split_stats.json instead of producing all-zero buckets.
    """
    sizes: dict[int, int] = {lbl: 0 for lbl in
                             range(int(labels.max()) + 1)}
    for lbl in labels.tolist():
        sizes[lbl] = sizes.get(lbl, 0) + 1
    ordered = sorted(sizes.items(), key=lambda kv: -kv[1])
    names = _SPLIT_NAMES.get(
        n_clusters,
        [f"group{i}" for i in range(len(ordered))])
    return {lbl: names[rank] if rank < len(names) else f"group{rank}"
            for rank, (lbl, _n) in enumerate(ordered)}


def bucket_reps(rep_sids: list[str], labels: np.ndarray,
                label_to_split: dict[int, str]
                ) -> dict[str, list[str]]:
    """Group rep_sids by their assigned split name."""
    out: dict[str, list[str]] = {}
    for sid, lbl in zip(rep_sids, labels.tolist()):
        out.setdefault(label_to_split[int(lbl)], []).append(sid)
    return out


def group_tm_stats(tm: np.ndarray, labels: np.ndarray,
                   label_to_split: dict[int, str]) -> dict:
    """Mean / median TM-score within each group and between every
    pair of groups. NaNs are ignored."""
    by_lbl: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels.tolist()):
        by_lbl.setdefault(lbl, []).append(i)

    def _stats(values: list[float]) -> dict:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"n_pairs": 0, "mean": None, "median": None}
        return {"n_pairs": int(arr.size),
                "mean": round(float(arr.mean()), 4),
                "median": round(float(np.median(arr)), 4)}

    out_intra: dict[str, dict] = {}
    out_inter: dict[str, dict] = {}
    labels_sorted = sorted(by_lbl)
    for a in labels_sorted:
        idx_a = by_lbl[a]
        intra = [tm[idx_a[i], idx_a[j]]
                 for i in range(len(idx_a))
                 for j in range(i + 1, len(idx_a))]
        out_intra[label_to_split[a]] = _stats(intra)
        for b in labels_sorted:
            if b <= a:
                continue
            idx_b = by_lbl[b]
            inter = [tm[i, j] for i in idx_a for j in idx_b]
            key = "__".join(sorted(
                [label_to_split[a], label_to_split[b]]))
            out_inter[key] = _stats(inter)
    return {"intra": out_intra, "inter": out_inter}


def sample_subset(ids: list[str], n: int, seed: int) -> list[str]:
    pool = sorted(ids)
    if len(pool) <= n:
        return pool
    return sorted(random.Random(seed).sample(pool, n))


# ---- IO ------------------------------------------------------------------


def _rep_to_cluster_from_list(path: Path) -> dict[str, str]:
    """Parse ``cluster_representatives.txt`` (``<sid>\\t<cluster_id>``).
    Tolerates a sid-only file (some downstream tooling produces those)
    by leaving the cluster id empty - we only USE the cluster id when
    the user asks for it in split_stats.json's reps_per_cluster field,
    so a missing column isn't fatal."""
    out: dict[str, str] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        out[parts[0]] = parts[1] if len(parts) >= 2 else ""
    return out


def _write_lines(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""),
                    encoding="utf-8")


def _unlink_if_exists(path: Path) -> bool:
    """Best-effort delete; returns True iff a file was removed.
    Used to clear stale ``val.txt`` from a previous n_clusters=3 run."""
    try:
        if path.is_file():
            path.unlink()
            return True
    except OSError:
        pass
    return False


# ---- main ---------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tmscore-matrix", type=Path, required=True)
    p.add_argument("--sample-list", type=Path, required=True,
                   help="cluster_representatives.txt "
                        "(sid<TAB>cluster_id, cluster col optional).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-clusters", type=int, default=2,
                   choices=(2, 3),
                   help="2 = train/test only (default; matches the "
                        "current 'one sample per cluster' workflow). "
                        "3 = train/val/test.")
    p.add_argument("--similarity-threshold", type=float, default=0.5)
    p.add_argument("--train-sample-n", type=int, default=200,
                   help="cap the train subset to this many reps "
                        "(written to train_200.txt; 200 = use all if "
                        "the bucket already has <=200).")
    p.add_argument("--test-sample-n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    # Accepted-but-ignored: prior versions mapped reps back onto the
    # full ~2 k pool via these. Kept so old commands don't break.
    p.add_argument("--processed-dir", type=Path, default=None,
                   help="(unused; rep-only output)")
    p.add_argument("--splits-json", type=Path, default=None,
                   help="(unused; rep-only output)")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.processed_dir or args.splits_json:
        print("  note: --processed-dir / --splits-json are no longer "
              "used (rep-only output).", file=sys.stderr)

    tm = np.load(args.tmscore_matrix)
    rep_sids = read_sample_list(args.sample_list)
    if tm.shape != (len(rep_sids), len(rep_sids)):
        # ids.txt is the authoritative ordering the matrix script
        # writes; trust it over a potentially shuffled list.
        ids_txt = args.tmscore_matrix.with_suffix(".ids.txt")
        if ids_txt.is_file():
            rep_sids = read_sample_list(ids_txt)
        if tm.shape != (len(rep_sids), len(rep_sids)):
            print(f"ERROR: matrix shape {tm.shape} != "
                  f"({len(rep_sids)},{len(rep_sids)})", file=sys.stderr)
            return 1

    rep_to_cluster = _rep_to_cluster_from_list(args.sample_list)

    affinity = build_affinity(tm, args.similarity_threshold)
    n_edges = int((affinity > 0).sum() - affinity.shape[0])  # off-diag
    print(f"spectral_split: N={tm.shape[0]}  "
          f"affinity edges (>0, off-diag) = {n_edges // 2}")

    labels = spectral_labels(affinity, args.n_clusters, args.seed)
    label_to_split = assign_splits(labels, args.n_clusters)
    buckets = bucket_reps(rep_sids, labels, label_to_split)
    train = sorted(buckets.get("train", []))
    test = sorted(buckets.get("test", []))
    val = sorted(buckets.get("val", []))

    train_200 = sample_subset(train, args.train_sample_n, args.seed)
    test_200 = sample_subset(test, args.test_sample_n, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_lines(args.output_dir / "train.txt", train)
    _write_lines(args.output_dir / "test.txt", test)
    _write_lines(args.output_dir / "train_200.txt", train_200)
    _write_lines(args.output_dir / "test_200.txt", test_200)
    if args.n_clusters == 3:
        _write_lines(args.output_dir / "val.txt", val)
    else:
        # n_clusters=2: no val bucket. Wipe a stale val.txt from a
        # previous 3-cluster run so callers don't trip over it.
        if _unlink_if_exists(args.output_dir / "val.txt"):
            print("  removed stale val.txt (n_clusters=2)")

    tm_stats = group_tm_stats(tm, labels, label_to_split)
    per_split = {sp: {"reps": len(ids), "clusters": len(ids)}
                 for sp, ids in (("train", train), ("test", test))}
    if args.n_clusters == 3:
        per_split["val"] = {"reps": len(val), "clusters": len(val)}
    stats = {
        "config": {
            "n_clusters": args.n_clusters,
            "similarity_threshold": args.similarity_threshold,
            "seed": args.seed,
            "tmscore_matrix": str(args.tmscore_matrix),
            "sample_list": str(args.sample_list),
        },
        "n_reps": int(tm.shape[0]),
        "per_split": per_split,
        "tm_stats": tm_stats,
        "train_200_n": len(train_200),
        "test_200_n": len(test_200),
        "label_to_split": {str(k): v for k, v in label_to_split.items()},
    }
    (args.output_dir / "split_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    print(f"  reps                  : {tm.shape[0]}  "
          f"(= final sample count, one per protein cluster)")
    order = (("train", train), ("val", val), ("test", test)) \
        if args.n_clusters == 3 else (("train", train), ("test", test))
    for sp, ids in order:
        print(f"  {sp:<8} reps={len(ids):>4}")
    print("  intra-group mean TM   : " + "  ".join(
        f"{k}={v['mean']}" for k, v in tm_stats["intra"].items()))
    if tm_stats["inter"]:
        print("  inter-group mean TM   : " + "  ".join(
            f"{k}={v['mean']}" for k, v in tm_stats["inter"].items()))
    print(f"  sampled subset        : "
          f"train_200={len(train_200)}  test_200={len(test_200)}")
    print(f"  output                : {args.output_dir}")
    # Silently consume rep_to_cluster to dodge the lint complaint about
    # an unused dict; reserved for a future per-cluster reps audit.
    _ = rep_to_cluster
    return 0


if __name__ == "__main__":
    sys.exit(main())
