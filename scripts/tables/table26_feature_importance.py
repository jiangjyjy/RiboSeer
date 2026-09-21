"""Table 27 — HARMONY feature importance (by gain).

Extracts the top-10 most important features from the trained RiboSeer
fusion model (``EnrichedFusion`` bundle), ranked by **gain**, normalises
the importances so the #1 feature = 1.000, and tags each feature with its
design group (G1–G4).

Importance source (model-type aware, all *gain*):

* **LightGBM** (``enriched_v7_lgbm``) → the underlying
  ``booster_.feature_importance(importance_type='gain')`` (NOT the sklearn
  wrapper's default ``feature_importances_``, which is split-count).
* **XGBoost** (``enriched_v7_model``) →
  ``get_booster().get_score(importance_type='gain')`` (features the model
  never split on get gain 0).
* **Ridge** → ``|weight|`` as a cheap gain proxy.

Feature names come from the bundle's own ``feature_names`` (the pinned
``enriched_fusion.FEATURE_NAMES`` column contract, + the 15-col context
block when ``use_context``). The group is inferred from the name:

* **G1 Per-Residue Multi-Metric** — single tool + metric
  (``boltz2_dist_score``, ``chai1_plddt``, ``equipnas_conf_rank`` …).
* **G2 Cross-Tool Interaction** — pairwise products (``cross_*`` /
  ``gate_*``) and the cross-tool agreement summaries
  (``vote_count`` / ``catA_agree`` / ``n_tools_available``).
* **G3 Neighborhood Context** — sliding-window / streak features
  (``*_win5``, ``*_density5``, ``binding_streak``).
* **G4 SCOPE Profile** — the 16-D SCOPE block (``scope_*``); only present
  if SCOPE features were baked into the model's feature space.

Run
---
::

    python scripts/tables/table26_feature_importance.py \\
        --model-dir data/enriched_v7_lgbm

Options
-------
``--top-k N``  number of features to list (default 10).
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from step5_fusion.enriched_fusion import EnrichedFusion  # noqa: E402
from step5_fusion.prediction_io import (  # noqa: E402
    has_lightgbm_model, load_lightgbm_gain,
)


# ---------------------------------------------------------------------------
# Feature → design group
# ---------------------------------------------------------------------------

GROUP_NAMES = {
    "G1": "Per-Residue Multi-Metric",
    "G2": "Cross-Tool Interaction",
    "G3": "Neighborhood Context",
    "G4": "SCOPE Profile",
}

# Cross-tool summary columns that aren't pairwise products but still
# describe agreement *across* tools → G2. (``n_tools_available`` is the
# 79-D bundle's name; ``n_tools_active`` is the on/on/on table9 name.)
_G2_SUMMARY = {"vote_count", "cata_agree",
               "n_tools_available", "n_tools_active"}


def feature_group(name: str) -> str:
    """Infer the design group (G1–G4) from a feature name.

    Order matters: context (G3) before cross-tool (G2) so a
    ``vote_count_win5`` lands in G3, not G2; SCOPE (G4) is checked first
    since its names are unambiguous.
    """
    n = name.lower()
    if "scope" in n or "profile" in n:
        return "G4"
    if "win" in n or "density" in n or "streak" in n or "neighbor" in n:
        return "G3"
    if (n.startswith("cross_") or n.startswith("gate_")
            or n in _G2_SUMMARY):
        return "G2"
    return "G1"


# ---------------------------------------------------------------------------
# Gain extraction (model-type aware)
# ---------------------------------------------------------------------------


def extract_gain(model: EnrichedFusion) -> tuple[list[str], list[float]]:
    """``(feature_names, gain_importances)`` aligned by index.

    Falls back gracefully: a LightGBM wrapper missing ``booster_`` uses
    ``feature_importances_``; an XGBoost feature the model never split on
    gets gain 0.
    """
    names = list(model.feature_names)
    mt = model.model_type

    if mt == "lightgbm":
        est = model.model
        if est is None:
            raise RuntimeError("model not loaded")
        booster = getattr(est, "booster_", None)
        if booster is not None:
            gains = list(booster.feature_importance(importance_type="gain"))
        else:  # pragma: no cover — wrapper without a fitted booster
            gains = list(est.feature_importances_)
        return names, [float(g) for g in gains]

    if mt == "xgboost":
        est = model.model
        if est is None:
            raise RuntimeError("model not loaded")
        booster = est.get_booster()
        score = booster.get_score(importance_type="gain")
        gains = [0.0] * len(names)
        for key, val in score.items():
            if key.startswith("f") and key[1:].isdigit():
                idx = int(key[1:])
                if 0 <= idx < len(gains):
                    gains[idx] = float(val)
            elif key in names:
                gains[names.index(key)] = float(val)
        return names, gains

    # ridge — |weight| proxy
    w = model._ridge_w
    if w is None:
        raise RuntimeError("ridge model not loaded")
    return names, [abs(float(x)) for x in w]


def rank_normalized(names: list[str], gains: list[float], k: int
                    ) -> list[dict]:
    """Top-``k`` features sorted by gain, normalised so rank-1 = 1.000."""
    pairs = sorted(zip(names, gains), key=lambda kv: -kv[1])[:k]
    top1 = pairs[0][1] if pairs and pairs[0][1] > 0 else 1.0
    return [
        {"rank": i + 1, "feature": nm, "group": feature_group(nm),
         "norm_importance": g / top1}
        for i, (nm, g) in enumerate(pairs)
    ]


# ---------------------------------------------------------------------------
# Output (stdout only — no LaTeX)
# ---------------------------------------------------------------------------


def print_table(rows: list[dict], model_type: str, n_features: int) -> None:
    print("=== Table 27: HARMONY Feature Importance (by gain) ===")
    print(f"model: {model_type}   features: {n_features}")
    print()
    hdr = f"{'Rank':<5s} {'Feature':<32s} {'Group':<7s} {'Norm. Importance':>16s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['rank']:<5d} {r['feature']:<32s} {r['group']:<7s} "
              f"{r['norm_importance']:>16.3f}")
    # Group legend for the groups that actually appear.
    seen = []
    for r in rows:
        if r["group"] not in seen:
            seen.append(r["group"])
    print()
    print("groups: " + "  ".join(
        f"{g}={GROUP_NAMES[g]}" for g in sorted(seen)))


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True,
                   help="trained RiboSeer model dir. Either an EnrichedFusion "
                        "bundle (e.g. data/enriched_v7_lgbm) OR a dir holding "
                        "the on/on/on export 'model.txt' "
                        "(data/batch_test_v7/fullsystem_predictions).")
    p.add_argument("--top-k", type=int, default=10,
                   help="number of top features to list (default 10)")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Prefer the on/on/on export (raw LightGBM booster + feature-name
    # sidecar) when it's present; otherwise load the EnrichedFusion bundle.
    if has_lightgbm_model(args.model_dir):
        try:
            names, gains = load_lightgbm_gain(args.model_dir)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: failed to load on/on/on model from "
                  f"{args.model_dir}: {e}", file=sys.stderr)
            return 1
        model_type = "lightgbm (on/on/on)"
    else:
        try:
            model = EnrichedFusion.load(args.model_dir)
        except Exception as e:  # noqa: BLE001 — surface load failure clearly
            print(f"ERROR: failed to load model from {args.model_dir}: {e}",
                  file=sys.stderr)
            return 1
        try:
            names, gains = extract_gain(model)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: could not extract feature importance: {e}",
                  file=sys.stderr)
            return 1
        model_type = model.model_type

    rows = rank_normalized(names, gains, args.top_k)
    print_table(rows, model_type, len(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
