"""Table 23 — split robustness.

Retrains the deterministic off/off/off LightGBM fusion (5 default tools, no
LLM) under four train/test splits and reports RiboSeer vs Boltz-2 mean
per-sample Pearson R (the LLM modules are frozen — only the fusion model is
retrained, matching the paper's "retrained from scratch, LLM 5.1 prompts
unchanged").

Splits
------
1. **Spectral cut (default)** — the shipped ``train.txt`` / ``test.txt``.
2. **Random 70/30** — all 332 ids shuffled with seed 42, first 70% train.
3. **Temporal (<=2021 / >=2022)** — by PDB release date. The sample JSONs
   carry NO date field, so this needs ``--release-dates FILE`` (a JSON map
   ``{sample_id_or_pdb: "YYYY-MM-DD" | year}``); without it the row is
   skipped ("–") with a note.
4. **Tight identity (25%)** — spectral bisection of the TM-score affinity
   graph at a stricter threshold (links pairs with ``TM >= threshold``), the
   larger half → train. Needs ``--tmscore-matrix matrix.npy`` (N×N) +
   ``--tight-threshold`` (default 0.30, the TM-score proxy for the
   25%-identity tightening). The N row/col ids are read from
   ``--tmscore-ids`` if given, else auto-derived from the matrix path
   (``matrix.npy`` → ``matrix.ids.txt``); without a usable matrix+ids the
   row is skipped ("–").

Both Pearson R (headline), Spearman R and R² are computed; only Pearson is
shown in the table (the others are printed underneath).

Run
---
::

    python scripts/tables/table22_split_robustness.py \\
        --data-dir      data/processed_quality \\
        --step4-dirs    data/batch_train_v7/step4 data/batch_test_v7/step4 \\
        --train-list    data/processed_quality/splits_tmscore_035/train.txt \\
        --test-list     data/processed_quality/splits_tmscore_035/test.txt \\
        --default-tools boltz2 chai1 rosettafold2na equipnas p2rank
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scripts.riboseer.retrain_eval_common import (  # noqa: E402
    DEFAULT_TOOLS, LGBM_OK, Sample, all_ids, boltz_corrs, collect_samples,
    load_ids, mean_metric, riboseer_corrs, train_predict,
)

RANDOM_SEED = 42
TRAIN_FRAC = 0.70
TIGHT_DEFAULT_THRESHOLD = 0.30


# ---------------------------------------------------------------------------
# Split builders → (train_ids, test_ids) or None when unavailable
# ---------------------------------------------------------------------------


def split_random(ids: list[str]) -> tuple[list[str], list[str]]:
    shuffled = sorted(ids)
    random.Random(RANDOM_SEED).shuffle(shuffled)
    n_train = int(round(TRAIN_FRAC * len(shuffled)))
    return shuffled[:n_train], shuffled[n_train:]


def _load_date_map(path: Optional[Path]) -> dict[str, int]:
    """``{key: year}`` from a JSON date map (value may be a year or an
    ISO/`YYYY-...` date string). Keys can be sample_id or pdb id."""
    if path is None or not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, int] = {}
    for k, v in (raw or {}).items():
        year = None
        if isinstance(v, int):
            year = v
        elif isinstance(v, str) and len(v) >= 4 and v[:4].isdigit():
            year = int(v[:4])
        if year is not None:
            out[str(k)] = year
    return out


def split_temporal(ids: list[str], date_map: dict[str, int]
                   ) -> Optional[tuple[list[str], list[str]]]:
    """train = release year <= 2021, test = >= 2022. ``date_map`` keyed by
    sample_id or by source pdb (``sid.split('_')[0]``)."""
    if not date_map:
        return None
    train, test, missing = [], [], 0
    for sid in ids:
        year = date_map.get(sid) or date_map.get(sid.split("_")[0])
        if year is None:
            missing += 1
            continue
        (train if year <= 2021 else test).append(sid)
    if missing:
        print(f"  [temporal] {missing}/{len(ids)} ids had no date "
              f"(dropped from this split)")
    if not train or not test:
        return None
    return train, test


def load_tmscore_matrix(npy_path: Optional[Path], ids_path: Optional[Path]
                        ) -> tuple[Optional["np.ndarray"], Optional[list[str]]]:
    """Load the (N×N) TM-score matrix (.npy) + its N row/col ids (.txt).

    ``ids_path`` is optional: when not given it is derived from the matrix
    path by swapping the ``.npy`` suffix for ``.ids.txt`` (so
    ``tmscore_matrix.npy`` → ``tmscore_matrix.ids.txt``). Returns (None,
    None) if the matrix/ids file is missing or shapes disagree."""
    import numpy as np
    if not (npy_path and npy_path.is_file()):
        return None, None
    # Auto-derive the ids file from the matrix path when not passed explicitly.
    if ids_path is None:
        ids_path = npy_path.with_name(npy_path.stem + ".ids.txt")
        print(f"  [tight] --tmscore-ids not given; derived {ids_path}")
    if not ids_path.is_file():
        print(f"  [tight] ids file not found: {ids_path}")
        return None, None
    try:
        m = np.load(npy_path)
    except (OSError, ValueError) as e:
        print(f"  [tight] failed to load matrix {npy_path}: {e}")
        return None, None
    mat_ids = [ln.strip() for ln in
               ids_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] != len(mat_ids):
        print(f"  [tight] matrix shape {m.shape} != #ids {len(mat_ids)}")
        return None, None
    return m, mat_ids


def split_tight_identity(ids: list[str], matrix, mat_ids: Optional[list[str]],
                         threshold: float
                         ) -> Optional[tuple[list[str], list[str]]]:
    """Spectral-bisect the samples at a stricter TM-score threshold.

    Affinity graph: ``A[i,j] = 1`` iff ``TM(i,j) >= threshold`` (similar →
    linked), symmetrised (TM-score is asymmetric). SpectralClustering
    (n_clusters=2, precomputed affinity, seed 42) cuts the graph so the two
    groups share few high-TM pairs; the larger group is train, the smaller
    test — same convention as the shipped spectral split, just tighter."""
    if matrix is None or not mat_ids:
        return None
    import numpy as np
    from sklearn.cluster import SpectralClustering

    idx_of = {sid: i for i, sid in enumerate(mat_ids)}
    present = [sid for sid in ids if sid in idx_of]
    missing = len(ids) - len(present)
    if missing:
        print(f"  [tight] {missing}/{len(ids)} ids absent from the TM matrix "
              f"(dropped from this split)")
    if len(present) < 4:
        return None

    rows = [idx_of[sid] for sid in present]
    sub = np.asarray(matrix, dtype=np.float64)[np.ix_(rows, rows)]
    affinity = (sub >= threshold).astype(np.float64)
    affinity = np.maximum(affinity, affinity.T)   # symmetrise
    np.fill_diagonal(affinity, 0.0)

    sc = SpectralClustering(n_clusters=2, affinity="precomputed",
                            assign_labels="discretize", random_state=RANDOM_SEED)
    labels = sc.fit_predict(affinity)
    g0 = [present[i] for i in range(len(present)) if labels[i] == 0]
    g1 = [present[i] for i in range(len(present)) if labels[i] == 1]
    train, test = (g0, g1) if len(g0) >= len(g1) else (g1, g0)
    if not train or not test:
        return None
    if len(test) < 50 or len(test) > 150:
        print(f"  [tight] WARNING: unbalanced split (train={len(train)}, "
              f"test={len(test)}) at threshold {threshold}")
    return train, test


# ---------------------------------------------------------------------------
# Run one split
# ---------------------------------------------------------------------------


def run_split(name: str, split: Optional[tuple[list[str], list[str]]],
              samples: dict[str, Sample]) -> dict:
    if split is None:
        return {"name": name, "skipped": True}
    train_ids, test_ids = split
    train = [samples[i] for i in train_ids if i in samples]
    test = [samples[i] for i in test_ids if i in samples]
    if not train or not test:
        return {"name": name, "skipped": True}
    preds = train_predict(train, test, RANDOM_SEED)
    rib = riboseer_corrs(test, preds)
    bol = boltz_corrs(test)
    rib_pr = mean_metric(rib, "pearson_r")
    bol_pr = mean_metric(bol, "pearson_r")
    delta = (round(rib_pr - bol_pr, 4)
             if rib_pr is not None and bol_pr is not None else None)
    return {
        "name": name, "skipped": False,
        "n_train": len(train), "n_test": len(test),
        "riboseer_pr": rib_pr, "boltz_pr": bol_pr, "delta": delta,
        "riboseer_sr": mean_metric(rib, "spearman_r"),
        "riboseer_r2": mean_metric(rib, "r_squared"),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_table(rows: list[dict]) -> None:
    print("=== Table 23: Split Robustness ===")
    hdr = (f"{'Split scheme':<30s} {'n_train':>7s} {'n_test':>6s} "
           f"{'RiboSeer_PR':>11s} {'Boltz2_PR':>9s} {'Delta':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("skipped"):
            print(f"{r['name']:<30s} {'–':>7s} {'–':>6s} "
                  f"{'–':>11s} {'–':>9s} {'–':>7s}")
            continue

        def f(v):
            return f"{v:.3f}" if v is not None else "–"
        d = r["delta"]
        ds = (f"+{d:.3f}" if d is not None and d >= 0
              else (f"{d:.3f}" if d is not None else "–"))
        print(f"{r['name']:<30s} {r['n_train']:>7d} {r['n_test']:>6d} "
              f"{f(r['riboseer_pr']):>11s} {f(r['boltz_pr']):>9s} {ds:>7s}")
    print()
    print("RiboSeer Spearman / R² per split:")
    for r in rows:
        if r.get("skipped"):
            continue
        print(f"  {r['name']:<30s} Spearman={r['riboseer_sr']}  "
              f"R2={r['riboseer_r2']}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--step4-dirs", type=Path, nargs="+", required=True,
                   help="one or more step4 JSONL dirs (probed in order)")
    p.add_argument("--train-list", type=Path, required=True)
    p.add_argument("--test-list", type=Path, required=True)
    p.add_argument("--default-tools", type=str, nargs="+",
                   default=list(DEFAULT_TOOLS))
    p.add_argument("--release-dates", type=Path, default=None,
                   help="JSON {sample_id|pdb: 'YYYY-MM-DD'|year} for the "
                        "Temporal split (sample JSONs carry no date field)")
    p.add_argument("--tmscore-matrix", type=Path, default=None,
                   help="N×N TM-score matrix (.npy) for the Tight-identity "
                        "split")
    p.add_argument("--tmscore-ids", type=Path, default=None,
                   help="text file of N ids (one per line) matching the "
                        "matrix rows/cols. Optional — if omitted it is "
                        "derived from --tmscore-matrix (.npy → .ids.txt).")
    p.add_argument("--tight-threshold", type=float,
                   default=TIGHT_DEFAULT_THRESHOLD,
                   help="TM-score affinity threshold for Tight identity "
                        "(default 0.30; stricter than the shipped 0.35)")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not LGBM_OK:
        print("ERROR: lightgbm not installed", file=sys.stderr)
        return 1
    for d in [args.data_dir, *args.step4_dirs]:
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 1
    try:
        spec_train = load_ids(args.train_list)
        spec_test = load_ids(args.test_list)
        ids = all_ids(args.train_list, args.test_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"collecting {len(ids)} samples (off/off/off, "
          f"tools={args.default_tools}) ...")
    samples = collect_samples(args.step4_dirs, args.data_dir, ids,
                              args.default_tools)
    print(f"usable samples: {len(samples)}/{len(ids)}")
    if not samples:
        print("ERROR: no usable samples", file=sys.stderr)
        return 1

    date_map = _load_date_map(args.release_dates)
    if not date_map:
        print("NOTE: no usable --release-dates; Temporal split skipped. "
              "(Sample JSONs have no release_date/deposition_date field — "
              "checked source_pdb/protein/rna/interaction/data_availability/"
              "split_info.)")
    matrix, mat_ids = load_tmscore_matrix(args.tmscore_matrix, args.tmscore_ids)
    if matrix is None:
        print("NOTE: no usable --tmscore-matrix/--tmscore-ids; Tight-identity "
              "split skipped.")

    rows = [
        run_split("Spectral cut (default)", (spec_train, spec_test), samples),
        run_split("Random 70/30 (cluster level)", split_random(ids), samples),
        run_split("Temporal (<=2021 / >=2022)",
                  split_temporal(ids, date_map), samples),
        run_split("Tight identity (25%)",
                  split_tight_identity(ids, matrix, mat_ids,
                                       args.tight_threshold),
                  samples),
    ]
    print()
    print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
