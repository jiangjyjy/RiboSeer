"""Pick one representative sample per protein_cluster_id.

Tie-break (deterministic): lowest resolution wins; if a tie, the larger
|GT| (more pocket signal) wins; if still tied, the lexicographically
smallest sample_id wins. Samples missing a numeric resolution are
ranked AFTER those that have one (the structure is fuzzier on the
unknown side), so an X-ray entry will beat an NMR/predicted sibling
in the same cluster.

Reads ``protein_cluster_id`` from ``splits.json`` (NOT from the sample
json, which stores ``split_info.protein_cluster_id = null`` in this
dataset).

Usage
-----
::

    python scripts/extract_cluster_representatives.py \\
        --input-dir   data/processed_quality/samples/ \\
        --splits-json data/processed/splits.json \\
        --output      data/processed_quality/splits_tmscore/\\
cluster_representatives.txt

Each output line: ``<sample_id>\\t<protein_cluster_id>``. The companion
``cluster_representatives.json`` records the full ranking key per rep
(resolution, gt, pdb, chain) for downstream debugging.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cluster_stats import load_splits_meta, sample_meta  # noqa: E402
from scripts.filter_quality import (  # noqa: E402
    gt_binding_count, resolution_of, scan_dir,
)


def _rank_key(sample: dict) -> tuple:
    """Smaller tuple wins. Missing resolution is treated as +inf so
    samples with a real resolution beat NMR/predicted siblings."""
    res = resolution_of(sample)
    if res is None:
        res_key = (1, float("inf"))  # missing -> ranked after
    else:
        res_key = (0, res)
    # negative GT so larger |GT| sorts first
    return (res_key, -gt_binding_count(sample),
            sample.get("sample_id") or "")


def pick_representatives(scanned, splits_map) -> dict[str, dict]:
    """``{protein_cluster_id: rep_record}``. Samples whose cluster id
    cannot be resolved (not in splits.json, or unclusterable) are
    grouped under ``__unclustered__`` so they don't silently vanish."""
    by_cluster: dict[str, list[tuple[tuple, str, dict, Path]]] = {}
    for sid, path, sample in scanned:
        meta = splits_map.get(sid) or splits_map.get(sid.lower()) or {}
        pc = meta.get("protein_cluster_id") or "__unclustered__"
        by_cluster.setdefault(pc, []).append(
            (_rank_key(sample), sid, sample, path))

    reps: dict[str, dict] = {}
    for pc, candidates in by_cluster.items():
        candidates.sort(key=lambda t: t[0])
        _, sid, sample, _path = candidates[0]
        reps[pc] = {
            "sample_id": sid,
            "protein_cluster_id": pc,
            "source_pdb": sample.get("source_pdb"),
            "protein_chain_id": (sample.get("protein") or {})
                .get("chain_id"),
            "resolution": resolution_of(sample),
            "gt_binding": gt_binding_count(sample),
            "cluster_size_in_pool": len(candidates),
        }
    return reps


def _splits_map_full(splits: dict) -> dict[str, dict]:
    """Like ``cluster_stats.sample_meta`` but bulk: ``{sid: meta_dict}``
    including a lowercase-key alias for case-insensitive lookup."""
    out: dict[str, dict] = {}
    for sid, meta in (splits.get("samples") or {}).items():
        if not isinstance(meta, dict):
            continue
        out[sid] = meta
        out.setdefault(sid.lower(), meta)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path,
                   default=Path("data/processed_quality/samples"))
    p.add_argument("--splits-json", type=Path,
                   default=Path("data/processed/splits.json"))
    p.add_argument("--output", type=Path, required=True,
                   help="<output>.txt: one sample_id per line "
                        "(tab + cluster_id). Sidecar .json holds the "
                        "full per-rep record.")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not args.input_dir.is_dir():
        print(f"ERROR: --input-dir not a directory: {args.input_dir}",
              file=sys.stderr)
        return 1

    splits = load_splits_meta(args.splits_json)
    splits_map = _splits_map_full(splits)
    scanned = scan_dir(args.input_dir)
    reps = pick_representatives(scanned, splits_map)

    real = {k: v for k, v in reps.items() if k != "__unclustered__"}
    unclustered = reps.get("__unclustered__")

    # Stable ordering: by cluster id (so a re-run is deterministic).
    ordered = sorted(real.items(), key=lambda kv: kv[0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(f"{rec['sample_id']}\t{cid}"
                  for cid, rec in ordered) + "\n",
        encoding="utf-8")
    # Sidecar is always <stem>.json next to --output (and *is* --output
    # if the user already passed a .json path).
    sidecar = (args.output if args.output.suffix == ".json"
               else args.output.with_name(args.output.stem + ".json"))
    sidecar.write_text(
        json.dumps({
            "n_clusters": len(real),
            "n_samples_unclustered": (unclustered or {})
                .get("cluster_size_in_pool", 0),
            "input_dir": str(args.input_dir),
            "splits_json": str(args.splits_json),
            "representatives": [rec for _cid, rec in ordered],
        }, indent=2), encoding="utf-8")

    print(f"extract_cluster_representatives")
    print(f"  input dir            : {args.input_dir}")
    print(f"  samples scanned      : {len(scanned)}")
    print(f"  protein clusters     : {len(real)}")
    if unclustered:
        print(f"  unclustered samples  : "
              f"{unclustered['cluster_size_in_pool']}  "
              f"(missing protein_cluster_id)")
    print(f"  output               : {args.output}")
    print(f"  sidecar              : {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
