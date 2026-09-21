#!/usr/bin/env python3
"""Table 8 (v2) — feature-group ablation on the new headline space.

The original Table 8 partitioned the **old** 79-D enriched features (fixed
5/6-tool ``TOOL_ORDER``) into PR / XT / NB and found the groups barely
helped (G1 alone 0.557 ≳ G1+G2+G3 0.555). The headline is now the 154-D
``table9`` feature space — MANDATORY optimal-7 ∪ MAESTRO + SCOPE profile —
so the partition is re-run there, adding the **new G4 (SCOPE) block** that
didn't exist before.

Four groups, routed by column name on ``table9_feature_names`` (the fixed
154-col contract; no hard-coded indices):

* **G1 / PR — per-residue multi-metric (90)** — every tool's per-row
  signals ``{tool}_{main,gate,main_rank,main_zscore,secondary,global_q}``.
* **G2 / XT — cross-tool interaction (33)** — ``cross_*`` score products +
  ``gate_*`` gate products + the summary counters ``vote_count`` /
  ``catA_agree`` / ``n_tools_active``.
* **G3 / NB — neighborhood context (15)** — ``*_win5`` / ``*_gate_density5``
  window means + ``vote_count_win5`` / ``max_score_win5`` /
  ``binding_streak``.
* **G4 / SC — SCOPE profile (16)** — the ``scope_*`` Group-4 block.

Each row builds the full 154-D matrix, slices to the active groups'
columns, and retrains LightGBM (same recipe as Table 7 / 9, so the Full
row reproduces the headline SCOPE+MAESTRO, **POLISH-off** number ≈ 0.593).
No POLISH post-edit is applied — this compares feature groups only.

Six combinations (output order)::

    PR
    PR XT
    PR    NB
       XT NB
    PR XT NB
    PR XT NB SC   ← Full (= headline, ≈ 0.593)

Usage
-----
::

    python scripts/tables/table08_feature_groups.py \\
        --processed-dir      data/processed_quality \\
        --train-step4-dir    data/batch_train_v7/step4 \\
        --test-step4-dir     data/batch_test_v7/step4 \\
        --train-list         data/processed_quality/splits_tmscore_035/train.txt \\
        --test-list          data/processed_quality/splits_tmscore_035/test.txt \\
        --scope-dir-train    data/batch_train_v7/scope_profiles_llm \\
        --scope-dir-test     data/batch_test_v7/scope_profiles_llm \\
        --maestro-dir-train  data/batch_train_v7/maestro_selections_llm_v4 \\
        --maestro-dir-test   data/batch_test_v7/maestro_selections_llm_v4 \\
        --output             data/batch_test_v7/table8_v2.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from step5_fusion.lightgbm_fusion import (  # noqa: E402
    LGBM_OK,
    SampleT9,
    _corr_from_probdict,
    _load_json_dir,
    _train_model,
    build_sample_matrix,
    collect_samples,
)
from scripts.tables.table09_llm_modules import (  # noqa: E402
    _load_sample_ids,
)
from scripts.riboseer.config import MANDATORY_TOOLS  # noqa: E402
from step5_fusion.features_15tool import table9_feature_names  # noqa: E402

# Summary counters that belong to XT (cross-tool aggregation), per the v2
# group definition (the old Table 8 put these in PR).
_SUMMARY_XT = {"vote_count", "catA_agree", "n_tools_active"}

# Group id → display column header.
GROUP_DISPLAY = {"G1": "PR", "G2": "XT", "G3": "NB", "G4": "SC"}
GROUP_ORDER = ["G1", "G2", "G3", "G4"]

# Six ablation rows (active group sets), in output order.
CONFIGS: list[tuple[str, ...]] = [
    ("G1",),
    ("G1", "G2"),
    ("G1", "G3"),
    ("G2", "G3"),
    ("G1", "G2", "G3"),
    ("G1", "G2", "G3", "G4"),
]


# ---------------------------------------------------------------------------
# Column → group routing
# ---------------------------------------------------------------------------


def feature_group_indices(names: list[str]) -> dict[str, list[int]]:
    """Partition the 154 column names into G1/G2/G3/G4 index lists. Routing
    (checked in this order so blocks don't collide):

    * G4 ← ``scope_*``
    * G3 ← ``*_win5`` / ``*_gate_density5`` / ``binding_streak``
    * G2 ← ``cross_*`` / ``gate_*`` / summary counters
    * G1 ← everything else (per-tool base columns)
    """
    groups: dict[str, list[int]] = {g: [] for g in GROUP_ORDER}
    for i, name in enumerate(names):
        if name.startswith("scope_"):
            groups["G4"].append(i)
        elif (name.endswith("_win5") or name.endswith("_gate_density5")
              or name == "binding_streak"):
            groups["G3"].append(i)
        elif (name.startswith("cross_") or name.startswith("gate_")
              or name in _SUMMARY_XT):
            groups["G2"].append(i)
        else:
            groups["G1"].append(i)
    return groups


def config_columns(active: tuple[str, ...],
                   group_idx: dict[str, list[int]]) -> list[int]:
    """Sorted column indices for the union of the active groups."""
    cols: list[int] = []
    for g in active:
        cols.extend(group_idx[g])
    return sorted(cols)


# ---------------------------------------------------------------------------
# Train / eval (full matrix built once; sliced per config)
# ---------------------------------------------------------------------------


def _mean(vs): return round(statistics.fmean(vs), 4) if vs else None


def train_eval(train: list[SampleT9], test: list[SampleT9],
               X_train_full: np.ndarray, X_test_full: list[np.ndarray],
               y_train: np.ndarray, cols: list[int]) -> dict:
    """Train LightGBM on the sliced columns and aggregate the per-sample
    test correlation. No POLISH."""
    model = _train_model(X_train_full[:, cols], y_train)
    prs, srs, r2s = [], [], []
    for s, X in zip(test, X_test_full):
        vec = np.asarray(model.predict(X[:, cols]), dtype=np.float64)
        prob = {rid: float(vec[i]) for i, rid in enumerate(s.residue_ids)}
        corr = _corr_from_probdict(prob, s)
        if corr is None:
            continue
        if corr["pearson_r"] is not None:
            prs.append(corr["pearson_r"])
        if corr["spearman_r"] is not None:
            srs.append(corr["spearman_r"])
        if corr["r_squared"] is not None:
            r2s.append(corr["r_squared"])
    return {"n_dims": len(cols), "n_samples": len(prs),
            "pearson_r": _mean(prs), "spearman_r": _mean(srs),
            "r_squared": _mean(r2s)}


def run_ablation(
    train: list[SampleT9], test: list[SampleT9],
    profiles_train, profiles_test, selections_train, selections_test,
) -> list[dict]:
    names = table9_feature_names(use_context=True, use_scope=True)
    group_idx = feature_group_indices(names)

    X_train_full = np.vstack([
        build_sample_matrix(s, True, True, profiles_train, selections_train,
                            mandatory=MANDATORY_TOOLS)
        for s in train])
    y_train = np.concatenate([s.y for s in train])
    X_test_full = [
        build_sample_matrix(s, True, True, profiles_test, selections_test,
                            mandatory=MANDATORY_TOOLS)
        for s in test]

    rows: list[dict] = []
    for active in CONFIGS:
        cols = config_columns(active, group_idx)
        res = train_eval(train, test, X_train_full, X_test_full,
                         y_train, cols)
        rows.append({
            **{g: (g in active) for g in GROUP_ORDER},
            **res})
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_COLUMNS = ["pr", "xt", "nb", "sc", "n_dims", "n_samples",
            "pearson_r", "spearman_r", "r_squared"]
_GROUP_TO_COL = {"G1": "pr", "G2": "xt", "G3": "nb", "G4": "sc"}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            out = {_GROUP_TO_COL[g]: int(bool(r[g])) for g in GROUP_ORDER}
            for c in ("n_dims", "n_samples", "pearson_r", "spearman_r",
                      "r_squared"):
                out[c] = r.get(c)
            w.writerow(out)


def _fnum(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def print_table(rows: list[dict]) -> None:
    print("=== Table 8: Feature Group Ablation ===")
    hdr = (f"{'PR':>2s} {'XT':>3s} {'NB':>3s} {'SC':>3s}  "
           f"{'PearsonR':>8s} {'SpearmanR':>10s} {'R2':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def chk(g): return "✓" if r[g] else " "
        print(f"{chk('G1'):>2s} {chk('G2'):>3s} {chk('G3'):>3s} "
              f"{chk('G4'):>3s}  {_fnum(r['pearson_r']):>8s} "
              f"{_fnum(r['spearman_r']):>10s} {_fnum(r['r_squared']):>8s}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--train-step4-dir", type=Path, required=True)
    p.add_argument("--test-step4-dir", type=Path, required=True)
    p.add_argument("--train-list", type=Path, required=True)
    p.add_argument("--test-list", type=Path, required=True)
    p.add_argument("--scope-dir-train", type=Path, default=None)
    p.add_argument("--scope-dir-test", type=Path, default=None)
    p.add_argument("--maestro-dir-train", type=Path, default=None)
    p.add_argument("--maestro-dir-test", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not LGBM_OK:
        print("ERROR: lightgbm not installed (pip install lightgbm)",
              file=sys.stderr)
        return 1
    for d in (args.processed_dir, args.train_step4_dir, args.test_step4_dir):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 1
    try:
        train_ids = _load_sample_ids(args.train_list)
        test_ids = _load_sample_ids(args.test_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    train = collect_samples(args.train_step4_dir, args.processed_dir,
                            train_ids)
    test = collect_samples(args.test_step4_dir, args.processed_dir, test_ids)
    print(f"train usable: {len(train)}, test usable: {len(test)}")
    if not train or not test:
        print("ERROR: no usable train/test samples", file=sys.stderr)
        return 1

    profiles_train = _load_json_dir(args.scope_dir_train)
    profiles_test = _load_json_dir(args.scope_dir_test)
    selections_train = _load_json_dir(args.maestro_dir_train)
    selections_test = _load_json_dir(args.maestro_dir_test)
    print(f"loaded SCOPE: train={len(profiles_train)} test={len(profiles_test)}"
          f"  MAESTRO: train={len(selections_train)} "
          f"test={len(selections_test)}  (POLISH not applied)")

    rows = run_ablation(train, test, profiles_train, profiles_test,
                        selections_train, selections_test)
    write_csv(args.output, rows)
    print(f"\nwrote {args.output}  ({len(rows)} configurations)\n")
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
