#!/usr/bin/env python3
"""Table 10 (v2) — leave-one-out tool ablation on the MAESTRO baseline.

Why a v2
--------
The original ``ablation_leave_one_out`` baseline was the *all-15-tool UCB*
system (Pearson R ≈ 0.581 — no SCOPE / no MAESTRO selection / no POLISH).
That baseline sits **above** the headline number (0.588), so the table
read backwards: dropping some tools *improved* the score, and the "full
system" row didn't match the paper's headline.

v2 fixes the framing by making the baseline the **headline system itself**:

    SCOPE profile → G4 block
    MAESTRO LLM selection ∪ config.MANDATORY_TOOLS (optimal-7 guardrail)
        → per-sample tool subset (un-selected → NaN)
    LightGBM fusion (retrained per configuration)
    POLISH soft edit (saved LLM actions)

i.e. exactly the Table 9 on/on/on arm (RiboSeer 0.588). Leave-one-out
then means *removing a single tool from the MAESTRO candidate pool* — for
every test/train sample whose MAESTRO selection contains ``tool_k``, that
tool's feature columns are forced to NaN (as if the tool didn't exist),
the fusion LightGBM is **retrained** on the same exclusion, and the test
prediction is re-POLISHed. Samples where MAESTRO never picked ``tool_k``
are unaffected (its columns were already NaN there).

No LLM call is made: the saved MAESTRO selections / SCOPE profiles /
POLISH actions are reused verbatim — we only *subtract* a tool from the
already-made selection.

Configurations (18 rows, in output order)
------------------------------------------
1.  ``Full system (15 tools)`` — the headline on/on/on baseline (≈ 0.588).
2-16. ``w/o <tool>`` for each of the 15 library tools (library order):
      drop that tool from every sample's MAESTRO selection, retrain.
17. ``Cat. A only (3 tools)`` — force exactly the 3 mandatory Category-A
    structure predictors (Boltz-2 / Chai-1 / RoseTTAFold2NA) active, all
    other tools NaN (SCOPE + POLISH still on).
18. ``Best single (Boltz-2)`` — fixed reference row from Table 4 / the
    per-tool step4 eval (Boltz-2 alone, Pearson R = 0.439), not retrained.

The Δ column is ``row Pearson − Full-system Pearson``.

Usage
-----
::

    python scripts/tables/table10_leave_one_out.py \\
        --processed-dir      data/processed_quality \\
        --train-step4-dir    data/batch_train_v7/step4 \\
        --test-step4-dir     data/batch_test_v7/step4 \\
        --train-list         data/processed_quality/splits_tmscore_035/train.txt \\
        --test-list          data/processed_quality/splits_tmscore_035/test.txt \\
        --scope-dir-train    data/batch_train_v7/scope_profiles_llm \\
        --scope-dir-test     data/batch_test_v7/scope_profiles_llm \\
        --maestro-dir-train  data/batch_train_v7/maestro_selections_llm_v4 \\
        --maestro-dir-test   data/batch_test_v7/maestro_selections_llm_v4 \\
        --polish-dir         data/batch_test_v7/polish_actions_llm_v2 \\
        --output             data/batch_test_v7/table10_v2.csv
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Reuse the headline (on/on/on) machinery verbatim so this table stays
# byte-for-byte comparable to the Table 9 full-system row.
from step5_fusion.lightgbm_fusion import (  # noqa: E402
    LGBM_OK,
    SampleT9,
    _corr_from_probdict,
    _load_json_dir,
    _train_model,
    apply_polish,
    collect_samples,
    resolve_scope_vector,
    resolve_selected_tools,
)
from scripts.tables.table09_llm_modules import (  # noqa: E402
    _load_sample_ids,
)
from step5_fusion.features_15tool import (  # noqa: E402
    ALL_KNOWN_TOOLS, build_15tool_features,
)
from scripts.riboseer.config import MANDATORY_TOOLS  # noqa: E402

# The 3 mandatory Category-A structure predictors for the "Cat. A only" row.
CAT_A3: tuple[str, ...] = ("boltz2", "chai1", "rosettafold2na")

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

# Best-single reference row, copied from Table 4 / the per-tool step4 eval
# (Boltz-2 alone). Not retrained here — it carries no MAESTRO/SCOPE/POLISH.
BEST_SINGLE = {"n": 104, "pearson_r": 0.439, "spearman_r": 0.387,
               "r_squared": 0.260}


@dataclass
class Config:
    label: str
    exclude_tool: Optional[str] = None        # leave-one-out drop
    restrict_to: Optional[tuple[str, ...]] = None  # force exactly these tools
    fixed: Optional[dict] = None              # fixed reference row (no train)


def build_configs() -> list[Config]:
    cfgs: list[Config] = [Config("Full system (15 tools)")]
    for t in ALL_KNOWN_TOOLS:
        cfgs.append(Config(f"w/o {DISPLAY.get(t, t)}", exclude_tool=t))
    cfgs.append(Config("Cat. A only (3 tools)", restrict_to=CAT_A3))
    cfgs.append(Config("Best single (Boltz-2)", fixed=BEST_SINGLE))
    return cfgs


# ---------------------------------------------------------------------------
# Feature matrix (headline on/on/on, with an optional tool exclusion)
# ---------------------------------------------------------------------------


def resolve_tools_for_config(
    sample: SampleT9, selections: dict[str, dict],
    *, exclude_tool: Optional[str], restrict_to: Optional[tuple[str, ...]],
) -> list[str]:
    """The per-sample active tool subset for one configuration.

    * ``restrict_to`` set → force exactly that subset (the "Cat. A only"
      row; step4 availability still gates each tool inside the builder).
    * otherwise → the headline MAESTRO selection ∪ ``MANDATORY_TOOLS``
      (the new optimal-7 guardrail, same as the full-system baseline),
      minus ``exclude_tool`` when present. Dropping a mandatory tool here
      genuinely removes it — that is the point of the leave-one-out.
    """
    if restrict_to is not None:
        return list(restrict_to)
    selected = resolve_selected_tools(sample, True, selections,  # MAESTRO on
                                      mandatory=MANDATORY_TOOLS)
    if exclude_tool is not None:
        selected = [t for t in selected if t != exclude_tool]
    return selected


def build_matrix(
    sample: SampleT9, profiles: dict[str, dict], selections: dict[str, dict],
    *, exclude_tool: Optional[str] = None,
    restrict_to: Optional[tuple[str, ...]] = None,
) -> np.ndarray:
    """Headline feature matrix (SCOPE on, MAESTRO selection) for one sample,
    with the requested tool exclusion / restriction applied. Column layout
    is the constant 154-D ``table9`` contract regardless of the subset."""
    selected = resolve_tools_for_config(
        sample, selections, exclude_tool=exclude_tool, restrict_to=restrict_to)
    scope_vec = resolve_scope_vector(sample, True, profiles)  # SCOPE on
    return build_15tool_features(
        sample.step4_data, sample.residue_ids,
        selected_tools=selected, scope_vector=scope_vec,
        use_context=True, use_scope=True)


def train_and_eval(
    train: list[SampleT9], test: list[SampleT9],
    profiles_train: dict[str, dict], profiles_test: dict[str, dict],
    selections_train: dict[str, dict], selections_test: dict[str, dict],
    polish_actions: dict[str, dict],
    *, exclude_tool: Optional[str] = None,
    restrict_to: Optional[tuple[str, ...]] = None,
    polish_on: bool = True,
) -> list[dict]:
    """Retrain the fusion LightGBM with the configuration's tool exclusion
    applied to **both** splits, predict on test, optionally apply the saved
    POLISH actions (``polish_on``), and return the per-sample correlation
    dicts. ``polish_on=False`` leaves the raw fusion prediction untouched
    (POLISH removed from the headline)."""
    X_train = np.vstack([
        build_matrix(s, profiles_train, selections_train,
                     exclude_tool=exclude_tool, restrict_to=restrict_to)
        for s in train])
    y_train = np.concatenate([s.y for s in train])
    model = _train_model(X_train, y_train)

    corrs: list[dict] = []
    for s in test:
        X = build_matrix(s, profiles_test, selections_test,
                         exclude_tool=exclude_tool, restrict_to=restrict_to)
        vec = np.asarray(model.predict(X), dtype=np.float64)
        prob = {rid: float(vec[i]) for i, rid in enumerate(s.residue_ids)}
        prob = apply_polish(prob, s, polish_on, polish_actions)
        corr = _corr_from_probdict(prob, s)
        if corr is not None:
            corrs.append(corr)
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


def run_table10_v2(
    train: list[SampleT9], test: list[SampleT9],
    profiles_train: dict[str, dict], profiles_test: dict[str, dict],
    selections_train: dict[str, dict], selections_test: dict[str, dict],
    polish_actions: dict[str, dict],
    polish_on: bool = True,
) -> list[dict]:
    """Build every Table-10 v2 row and fill the ΔPearson column (vs the
    full system, which is always row 0). ``polish_on=False`` runs every
    config with POLISH removed."""
    rows: list[dict] = []
    for cfg in build_configs():
        if cfg.fixed is not None:                       # fixed reference row
            rows.append({"configuration": cfg.label,
                         "n_samples": cfg.fixed["n"],
                         "pearson_r": cfg.fixed["pearson_r"],
                         "spearman_r": cfg.fixed["spearman_r"],
                         "r_squared": cfg.fixed["r_squared"]})
            continue
        corrs = train_and_eval(
            train, test, profiles_train, profiles_test,
            selections_train, selections_test, polish_actions,
            exclude_tool=cfg.exclude_tool, restrict_to=cfg.restrict_to,
            polish_on=polish_on)
        rows.append(_aggregate(cfg.label, corrs))

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
    print("=== Table 10 (v2): Leave-one-out tool ablation "
          "(MAESTRO baseline) ===")
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
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--train-step4-dir", type=Path, required=True)
    p.add_argument("--test-step4-dir", type=Path, required=True)
    p.add_argument("--train-list", type=Path, required=True)
    p.add_argument("--test-list", type=Path, required=True)
    p.add_argument("--scope-dir-train", type=Path, default=None)
    p.add_argument("--scope-dir-test", type=Path, default=None)
    p.add_argument("--maestro-dir-train", type=Path, default=None)
    p.add_argument("--maestro-dir-test", type=Path, default=None)
    p.add_argument("--polish-dir", type=Path, default=None,
                   help="saved POLISH actions; omit (or pass --no-polish) to "
                        "run with POLISH off (the new headline).")
    p.add_argument("--no-polish", action="store_true",
                   help="force POLISH off even if --polish-dir is given.")
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

    profiles_train = _load_json_dir(args.scope_dir_train)
    profiles_test = _load_json_dir(args.scope_dir_test)
    selections_train = _load_json_dir(args.maestro_dir_train)
    selections_test = _load_json_dir(args.maestro_dir_test)
    polish_actions = _load_json_dir(args.polish_dir)
    polish_on = (args.polish_dir is not None) and not args.no_polish
    print(f"loaded SCOPE: train={len(profiles_train)} test={len(profiles_test)}"
          f"  MAESTRO: train={len(selections_train)} "
          f"test={len(selections_test)}  POLISH: "
          f"{'on (' + str(len(polish_actions)) + ' actions)' if polish_on else 'off'}")

    rows = run_table10_v2(
        train, test, profiles_train, profiles_test,
        selections_train, selections_test, polish_actions,
        polish_on=polish_on)
    write_csv(args.output, rows)
    print(f"\nwrote {args.output}  ({len(rows)} configurations)\n")
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
