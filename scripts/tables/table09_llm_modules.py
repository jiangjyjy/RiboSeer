"""Table 9 — SCOPE × MAESTRO × POLISH LLM-module ablation.

Runs the 8 on/off combinations of the three LLM modules and reports
per-residue Pearson / Spearman / R² for each (same metric convention as
``table04_main_results.py`` — correlate each sample's per-residue scores
against its binary GT vector on the resolved-residue subset, then mean
across samples).

How each module is operationalised:

* **SCOPE** — a 16-D profile block (Group 4) appended to the fusion
  features. ON → the sample's SCOPE profile vector (LLM profile JSON when
  ``--scope-profiles-*`` is given, else the heuristic ``cauto`` profile);
  OFF → a zero block (module removed).
* **MAESTRO** — the per-sample tool subset fed to fusion (un-selected /
  unavailable tools become NaN columns). ON → the LLM selection
  (``--maestro-selections-*`` JSON) or, lacking that, *all* tools
  available in step4; OFF → the fixed 5-tool baseline.
* **POLISH** — a post-fusion edit of the per-residue probability vector.
  ON → saved LLM POLISH actions (``--polish-actions``) or, lacking that,
  the deterministic ``auto_polish_action`` control; OFF → no edit.

Phase-1 (no LLM API): omit the JSON dirs and every "ON" arm falls back to
its deterministic control (cauto / all-tools / auto-polish), so the 8
rows differ without any API call. Phase-2: drop in the saved LLM JSONs
and re-run — same script, same columns.

The full 15-tool layout (``features_15tool.ALL_KNOWN_TOOLS``) is constant
across all 8 combos, so train/test feature columns always align; tools
absent from the training split simply contribute NaN columns (LightGBM
splits on missing natively).

Usage
-----
::

    python scripts/tables/table09_llm_modules.py \\
        --train-step4-dir data/batch_train_v7/step4/ \\
        --test-step4-dir  data/batch_test_v7/step4/ \\
        --processed-dir   data/processed_quality \\
        --train-list      data/processed_quality/splits_tmscore_035/train.txt \\
        --test-list       data/processed_quality/splits_tmscore_035/test.txt \\
        --output          data/batch_test_v7/table09_llm_modules.csv
        # optional Phase-2 inputs:
        # --scope-profiles-train ... --scope-profiles-test ...
        # --maestro-selections-train ... --maestro-selections-test ...
        # --polish-actions ...
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from step5_fusion.data_collector import (  # noqa: E402
    load_sample_json, read_jsonl_record,
)
from step5_fusion.features_15tool import (  # noqa: E402
    ALL_KNOWN_TOOLS, auto_scope_profile, build_15tool_features,
    encode_scope_profile, _canonical, _index_predictions,
    table9_feature_names,
)
from step5_fusion.metrics import per_sample_corr  # noqa: E402
from step5_fusion.prediction_io import (  # noqa: E402
    save_prob_dict, save_lightgbm_model,
)
from step5_fusion import lightgbm_fusion as lgf  # noqa: E402
from step5_fusion.lightgbm_fusion import (  # noqa: E402
    COMBOS, FIXED5, SampleT9, apply_polish, collect_samples,
    compute_fullsystem_predictions, fullsystem_feature_names,
    inject_mandatory, predict_setting, resolve_scope_vector,
    resolve_selected_tools, save_fullsystem_predictions, _corr_from_probdict,
    _load_json_dir, _rna_len,
)
from step7_iteration.polish_ops import (  # noqa: E402
    apply_actions, auto_polish_action,
)



# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_ablation(
    *,
    train: list[SampleT9], test: list[SampleT9],
    profiles_train: dict[str, dict], profiles_test: dict[str, dict],
    selections_train: dict[str, dict], selections_test: dict[str, dict],
    polish_actions: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """Returns (aggregate rows, per-sample rows). Trains one model per
    distinct (scope, maestro) and reuses it across the POLISH toggle."""
    pred_cache: dict[tuple[bool, bool], dict[str, dict[int, float]]] = {}
    agg_rows: list[dict] = []
    per_sample: list[dict] = []

    for scope_on, maestro_on, polish_on in COMBOS:
        key = (scope_on, maestro_on)
        if key not in pred_cache:
            pred_cache[key] = predict_setting(
                scope_on, maestro_on, train, test,
                profiles_train, profiles_test,
                selections_train, selections_test)
        base_preds = pred_cache[key]

        corrs: list[dict] = []
        for s in test:
            prob = dict(base_preds.get(s.sid, {}))
            prob = apply_polish(prob, s, polish_on, polish_actions)
            corr = _corr_from_probdict(prob, s)
            if corr is None:
                continue
            corrs.append(corr)
            per_sample.append({
                "scope": int(scope_on), "maestro": int(maestro_on),
                "polish": int(polish_on), "sample_id": s.sid, **corr})

        agg_rows.append(_aggregate_row(scope_on, maestro_on, polish_on,
                                       corrs))
    return agg_rows, per_sample


def _mean(vs): return round(statistics.fmean(vs), 4) if vs else None
def _median(vs): return round(statistics.median(vs), 4) if vs else None
def _std(vs):
    if len(vs) < 2:
        return 0.0 if vs else None
    return round(statistics.pstdev(vs), 4)


def _aggregate_row(scope_on: bool, maestro_on: bool, polish_on: bool,
                   corrs: list[dict]) -> dict:
    prs = [c["pearson_r"] for c in corrs if c["pearson_r"] is not None]
    srs = [c["spearman_r"] for c in corrs if c["spearman_r"] is not None]
    r2s = [c["r_squared"] for c in corrs if c["r_squared"] is not None]
    return {
        "scope": "on" if scope_on else "off",
        "maestro": "on" if maestro_on else "off",
        "polish": "on" if polish_on else "off",
        "n_samples": len(corrs),
        "pearson_r_mean": _mean(prs),
        "pearson_r_std": _std(prs),
        "pearson_r_median": _median(prs),
        "spearman_r_mean": _mean(srs),
        "r2_mean": _mean(r2s),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_COLUMNS = ["scope", "maestro", "polish", "n_samples",
            "pearson_r_mean", "pearson_r_std", "pearson_r_median",
            "spearman_r_mean", "r2_mean"]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in _COLUMNS})


def write_per_sample_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["scope", "maestro", "polish", "sample_id",
            "pearson_r", "spearman_r", "r_squared"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def print_table(rows: list[dict]) -> None:
    hdr = (f"{'SCOPE':>5s} {'MAESTRO':>7s} {'POLISH':>6s} {'n':>4s} "
           f"{'PearsonR':>9s} {'SpearmanR':>10s} {'R2':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['scope']:>5s} {r['maestro']:>7s} {r['polish']:>6s} "
              f"{r['n_samples']:>4d} {str(r['pearson_r_mean']):>9s} "
              f"{str(r['spearman_r_mean']):>10s} "
              f"{str(r['r2_mean']):>8s}")


def _load_sample_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"sample list not found: {path}")
    return [ln.split()[0] for ln in
            path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


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
    p.add_argument("--scope-profiles-train", type=Path, default=None)
    p.add_argument("--scope-profiles-test", type=Path, default=None)
    p.add_argument("--maestro-selections-train", type=Path, default=None)
    p.add_argument("--maestro-selections-test", type=Path, default=None)
    p.add_argument("--polish-actions", type=Path, default=None)
    p.add_argument("--save-predictions-dir", type=Path, default=None,
                   help="also dump the on/on/on (full-system) per-sample "
                        "post-POLISH predictions to <dir>/<sid>.json for "
                        "downstream tables (--predictions-dir).")
    p.add_argument("--save-model-dir", type=Path, default=None,
                   help="also dump the trained on/on/on LightGBM "
                        "(<dir>/model.txt + feature names) for Table 27; "
                        "defaults to --save-predictions-dir when that is set.")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not lgf.LGBM_OK:
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

    profiles_train = _load_json_dir(args.scope_profiles_train)
    profiles_test = _load_json_dir(args.scope_profiles_test)
    selections_train = _load_json_dir(args.maestro_selections_train)
    selections_test = _load_json_dir(args.maestro_selections_test)
    polish_actions = _load_json_dir(args.polish_actions)

    rows, per_sample = run_ablation(
        train=train, test=test,
        profiles_train=profiles_train, profiles_test=profiles_test,
        selections_train=selections_train, selections_test=selections_test,
        polish_actions=polish_actions,
    )

    save_model_dir = args.save_model_dir or args.save_predictions_dir
    if args.save_predictions_dir is not None or args.save_model_dir is not None:
        fs_preds, fs_model = compute_fullsystem_predictions(
            train=train, test=test,
            profiles_train=profiles_train, profiles_test=profiles_test,
            selections_train=selections_train, selections_test=selections_test,
            polish_actions=polish_actions, return_model=True)
        if args.save_predictions_dir is not None:
            n = save_fullsystem_predictions(
                args.save_predictions_dir, test, fs_preds)
            print(f"wrote {n} full-system predictions to "
                  f"{args.save_predictions_dir}")
        if save_model_dir is not None:
            save_lightgbm_model(save_model_dir, fs_model,
                                fullsystem_feature_names())
            print(f"wrote on/on/on LightGBM model to {save_model_dir}/model.txt")

    write_csv(args.output, rows)
    ps_path = args.output.with_name(args.output.stem + "_per_sample.csv")
    write_per_sample_csv(ps_path, per_sample)
    print(f"wrote {args.output}  ({len(rows)} combos)")
    print(f"wrote {ps_path}  ({len(per_sample)} sample-combo rows)")
    print()
    print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
