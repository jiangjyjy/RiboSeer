"""Generate RiboSeer **full-system (on/on/on)** per-sample predictions.

This is the prediction behind the headline 0.565 row of Table 9 — SCOPE on
(LLM G4 profile) × MAESTRO on (per-sample 15-tool variable selection) ×
POLISH on (soft probability edits). Every downstream table that used to
load the off/off/off model bundle now reads these JSONs via
``--predictions-dir``.

Why it needs the *train* split too
----------------------------------
The on/on/on prediction is not a frozen bundle on disk — it is produced by
a LightGBM trained on the on/on/on (15-tool + SCOPE) feature space. The
shipped ``data/enriched_v7_lgbm`` bundle is the OLD 6-tool off-pipeline
model, so we must retrain on the train split with the same SCOPE/MAESTRO
inputs to reproduce 0.565 exactly. This mirrors ``table09_llm_modules``'s
on/on/on arm (it is literally the same code path); running that script with
``--save-predictions-dir`` produces byte-identical output.

Output — ``<output-dir>/<sample_id>.json``::

    {"residue_ids": [...], "probabilities": [...]}

Usage
-----
::

    python scripts/riboseer/generate_fullsystem_predictions.py \\
        --data-dir          data/processed_quality \\
        --train-step4-dir   data/batch_train_v7/step4 \\
        --test-step4-dir    data/batch_test_v7/step4 \\
        --train-list        data/processed_quality/splits_tmscore_035/train.txt \\
        --test-list         data/processed_quality/splits_tmscore_035/test.txt \\
        --scope-dir-train   data/batch_train_v7/scope_profiles_llm \\
        --scope-dir-test    data/batch_test_v7/scope_profiles_llm \\
        --maestro-dir-train data/batch_train_v7/maestro_selections_llm_v4 \\
        --maestro-dir-test  data/batch_test_v7/maestro_selections_llm_v4 \\
        --polish-dir        data/batch_test_v7/polish_actions_llm_v2 \\
        --output-dir        data/batch_test_v7/fullsystem_predictions
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step5_fusion.lightgbm_fusion import (  # noqa: E402
    LGBM_OK,
    collect_samples,
    compute_fullsystem_predictions,
    fullsystem_feature_names,
    save_fullsystem_predictions,
    _load_json_dir,
)
from scripts.tables.table09_llm_modules import (  # noqa: E402
    _load_sample_ids,
)
from step5_fusion.prediction_io import save_lightgbm_model  # noqa: E402


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="processed dir (has samples/<sid>.json)")
    p.add_argument("--train-step4-dir", type=Path, required=True)
    p.add_argument("--test-step4-dir", type=Path, required=True)
    p.add_argument("--train-list", type=Path, required=True)
    p.add_argument("--test-list", type=Path, required=True)
    p.add_argument("--scope-dir-train", type=Path, default=None,
                   help="LLM SCOPE profiles for the train split; missing → "
                        "heuristic cauto profile (SCOPE still 'on').")
    p.add_argument("--scope-dir-test", type=Path, default=None)
    p.add_argument("--maestro-dir-train", type=Path, default=None,
                   help="LLM MAESTRO selections for the train split; missing "
                        "→ all available tools.")
    p.add_argument("--maestro-dir-test", type=Path, default=None)
    p.add_argument("--polish-dir", type=Path, default=None,
                   help="LLM POLISH actions (test); missing → deterministic "
                        "auto-polish control.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--save-model-dir", type=Path, default=None,
                   help="where to dump the trained on/on/on LightGBM "
                        "(<dir>/model.txt + feature names) for Table 27; "
                        "defaults to --output-dir.")
    p.add_argument("--no-save-model", action="store_true",
                   help="skip saving the LightGBM model (predictions only).")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not LGBM_OK:
        print("ERROR: lightgbm not installed (pip install lightgbm)",
              file=sys.stderr)
        return 1
    for d in (args.data_dir, args.train_step4_dir, args.test_step4_dir):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 1
    try:
        train_ids = _load_sample_ids(args.train_list)
        test_ids = _load_sample_ids(args.test_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    train = collect_samples(args.train_step4_dir, args.data_dir, train_ids)
    test = collect_samples(args.test_step4_dir, args.data_dir, test_ids)
    print(f"train usable: {len(train)}, test usable: {len(test)}")
    if not train or not test:
        print("ERROR: no usable train/test samples", file=sys.stderr)
        return 1

    preds, model = compute_fullsystem_predictions(
        train=train, test=test,
        profiles_train=_load_json_dir(args.scope_dir_train),
        profiles_test=_load_json_dir(args.scope_dir_test),
        selections_train=_load_json_dir(args.maestro_dir_train),
        selections_test=_load_json_dir(args.maestro_dir_test),
        polish_actions=_load_json_dir(args.polish_dir),
        return_model=True,
    )
    n = save_fullsystem_predictions(args.output_dir, test, preds)
    print(f"wrote {n} full-system predictions to {args.output_dir}")

    if not args.no_save_model:
        model_dir = args.save_model_dir or args.output_dir
        save_lightgbm_model(model_dir, model, fullsystem_feature_names())
        print(f"wrote on/on/on LightGBM model to {model_dir}/model.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
