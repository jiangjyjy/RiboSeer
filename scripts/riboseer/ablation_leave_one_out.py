#!/usr/bin/env python3
"""Table 10 — leave-one-out tool ablation.

RiboSeer fuses up to 15 tools (``features_15tool.ALL_KNOWN_TOOLS``). This
table retrains the fusion LightGBM once per configuration, dropping a
single tool (or restricting to a category) each time, and reports the
per-residue Pearson / Spearman / R² on the 107-sample test split — so
each tool's marginal contribution (ΔPearson vs the full system) is read
straight off the table.

This is a **pure tool ablation**: no SCOPE block (``use_scope=False``),
no MAESTRO selection, no POLISH edit. The only thing that varies between
rows is the selected tool subset; an un-selected tool's feature columns
become NaN, which LightGBM splits on natively.

Training vs test tool coverage
------------------------------
The test split carries all 15 tools; the training split carries 13
(``alphafold3`` and ``bindup`` were never run on train). Those two tools'
columns are therefore all-NaN at train time for every configuration —
the model simply never learns a split on them, while they still feed real
signal at test time. Dropping either at test (rows ``w/o AlphaFold 3`` /
``w/o BindUP``) still measures their test-time contribution.

Configurations (18 rows, in output order)
------------------------------------------
1.  Full system (all 15 tools)
2-16. ``w/o <tool>`` for each of the 15 tools (library order)
17. ``Cat. A only`` (boltz2 + chai1 + rosettafold2na + rfaa + alphafold3)
18. ``Best single (Boltz-2)`` — a fixed reference row copied from Table 4
    (Boltz-2 alone, Pearson R = 0.439), not retrained here.

The LightGBM recipe is the project standard (``table09_llm_modules.
_make_lightgbm``: n_estimators=100, max_depth=4, learning_rate=0.1, +
standard regularization, objective=regression), so rows stay directly
comparable to Table 4 / Table 9.

Usage
-----
::

    python scripts/riboseer/ablation_leave_one_out.py \\
        --train-step4-dir data/batch_train_v7/step4/ \\
        --test-step4-dir  data/batch_test_v7/step4/ \\
        --processed-dir   data/processed_quality \\
        --train-list      data/processed_quality/splits_tmscore_035/train.txt \\
        --test-list       data/processed_quality/splits_tmscore_035/test.txt \\
        --output          data/batch_test_v7/ablation_leave_one_out.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.riboseer.ablation_fusion_method import per_sample_corr  # noqa: E402
from step5_fusion.lightgbm_fusion import (  # noqa: E402
    LGBM_OK,
    SampleT9,
    _make_lightgbm,
    collect_samples,
)
from scripts.tables.table09_llm_modules import (  # noqa: E402
    _load_sample_ids,
)
from step5_fusion.features_15tool import (  # noqa: E402
    ALL_KNOWN_TOOLS, build_15tool_features,
)

# Category A subset for the "Cat. A only" row.
CAT_A: tuple[str, ...] = (
    "boltz2", "chai1", "rosettafold2na", "rfaa", "alphafold3")

# Pretty labels for the report (library id → paper name).
DISPLAY: dict[str, str] = {
    "boltz2": "Boltz-2", "chai1": "Chai-1",
    "rosettafold2na": "RoseTTAFold2NA", "rfaa": "RFAA",
    "alphafold3": "AlphaFold 3", "p2rank": "P2Rank", "fpocket": "Fpocket",
    "deeppocket": "DeepPocket", "equipnas": "EquiPNAS",
    "nucleicnet": "NucleicNet", "graphbind": "GraphBind",
    "rnabindrplus": "RNABindRPlus", "bindup": "BindUP",
    "hdock": "HDOCK", "haddock3": "HADDOCK 3",
}

# Best-single reference row, copied verbatim from Table 4 (Boltz-2 alone).
BEST_SINGLE = {"n": 104, "pearson_r": 0.439, "spearman_r": 0.387,
               "r_squared": 0.260}


@dataclass
class Config:
    label: str
    tools: Optional[tuple[str, ...]]   # None → fixed reference row
    fixed: Optional[dict] = None       # populated for reference rows


def build_configs() -> list[Config]:
    cfgs: list[Config] = [Config("Full system (15 tools)", ALL_KNOWN_TOOLS)]
    for t in ALL_KNOWN_TOOLS:
        kept = tuple(x for x in ALL_KNOWN_TOOLS if x != t)
        cfgs.append(Config(f"w/o {DISPLAY.get(t, t)}", kept))
    cfgs.append(Config("Cat. A only (5 tools)", CAT_A))
    cfgs.append(Config("Best single (Boltz-2)", None, fixed=BEST_SINGLE))
    return cfgs


# ---------------------------------------------------------------------------
# Features + LightGBM (pure tool ablation: no scope block)
# ---------------------------------------------------------------------------


def build_matrix(sample: SampleT9, selected: tuple[str, ...]) -> np.ndarray:
    """Per-residue feature matrix for a tool subset, SCOPE block off."""
    return build_15tool_features(
        sample.step4_data, sample.residue_ids,
        selected_tools=list(selected), scope_vector=None,
        use_context=True, use_scope=False)


# Exposed so tests can inject a deterministic fake without LightGBM.
def _train_model(X: np.ndarray, y: np.ndarray):
    model = _make_lightgbm()
    model.fit(X, y)
    return model


def train_and_eval(train: list[SampleT9], test: list[SampleT9],
                   selected: tuple[str, ...]) -> list[dict]:
    """Train one LightGBM on ``selected`` and return per-sample corr dicts
    over the test set (skipping samples with an undefined correlation)."""
    X_train = np.vstack([build_matrix(s, selected) for s in train])
    y_train = np.concatenate([s.y for s in train])
    model = _train_model(X_train, y_train)

    corrs: list[dict] = []
    for s in test:
        X = build_matrix(s, selected)
        vec = np.asarray(model.predict(X), dtype=np.float64)
        pred = vec[s.eval_mask]
        gt = s.y[s.eval_mask]
        c = per_sample_corr(pred, gt)
        if c is not None:
            corrs.append(c)
    return corrs


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean(vs: list[float]) -> Optional[float]:
    return round(statistics.fmean(vs), 4) if vs else None


def _aggregate(label: str, corrs: list[dict]) -> dict:
    prs = [c["pearson_r"] for c in corrs if c["pearson_r"] is not None]
    srs = [c["spearman_r"] for c in corrs if c["spearman_r"] is not None]
    r2s = [c["r_squared"] for c in corrs if c["r_squared"] is not None]
    return {"configuration": label, "n_samples": len(prs),
            "pearson_r": _mean(prs), "spearman_r": _mean(srs),
            "r_squared": _mean(r2s)}


def run_table10(train: list[SampleT9], test: list[SampleT9]) -> list[dict]:
    """Build every Table-10 row and fill the ΔPearson column (vs the full
    system, which is always row 0)."""
    rows: list[dict] = []
    for cfg in build_configs():
        if cfg.tools is None:                       # fixed reference row
            rows.append({"configuration": cfg.label,
                         "n_samples": cfg.fixed["n"],
                         "pearson_r": cfg.fixed["pearson_r"],
                         "spearman_r": cfg.fixed["spearman_r"],
                         "r_squared": cfg.fixed["r_squared"]})
        else:
            rows.append(_aggregate(cfg.label,
                                   train_and_eval(train, test, cfg.tools)))

    full_pr = rows[0]["pearson_r"]
    for i, row in enumerate(rows):
        if i == 0 or full_pr is None or row["pearson_r"] is None:
            row["delta_pearson_r"] = None
        else:
            row["delta_pearson_r"] = round(row["pearson_r"] - full_pr, 4)
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_COLUMNS = ["configuration", "n_samples", "pearson_r", "spearman_r",
            "r_squared", "delta_pearson_r"]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in _COLUMNS})


def _fnum(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _fdelta(v: Optional[float]) -> str:
    return "–" if v is None else f"{v:+.3f}"


def print_table(rows: list[dict]) -> None:
    hdr = (f"{'Configuration':<26s} {'n':>4s} {'PearsonR':>9s} "
           f"{'SpearmanR':>10s} {'R2':>7s} {'dPearsonR':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['configuration']:<26s} {r['n_samples']:>4d} "
              f"{_fnum(r['pearson_r']):>9s} {_fnum(r['spearman_r']):>10s} "
              f"{_fnum(r['r_squared']):>7s} "
              f"{_fdelta(r.get('delta_pearson_r')):>10s}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-step4-dir", type=Path, required=True)
    p.add_argument("--test-step4-dir", type=Path, required=True)
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--train-list", type=Path, required=True)
    p.add_argument("--test-list", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not LGBM_OK:
        print("ERROR: lightgbm not installed (pip install lightgbm)",
              file=sys.stderr)
        return 1
    for d in (args.train_step4_dir, args.test_step4_dir, args.processed_dir):
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

    rows = run_table10(train, test)
    write_csv(args.output, rows)
    print(f"\nwrote {args.output}  ({len(rows)} configurations)\n")
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
