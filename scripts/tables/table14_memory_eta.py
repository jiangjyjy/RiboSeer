"""Table 15 — MEMORY EMA learning-rate (η) ablation.

The MEMORY module updates the weight tensor W after every processed
sample with an exponential moving average:

    W_new = (1 - η) · W_old + η · r          (r = this sample's per-tool
                                              quality)

η controls how fast tool-reliability weights adapt:

    η = 0.00  frozen (W never updates)
    η = 0.05  slow
    η = 0.10  default
    η = 0.20  fast
    η = 0.50  very fast
    η = 1.00  memoryless (only the most recent sample)

This script simulates the online test-time loop for each η and reports
the resulting overall per-sample Pearson / Spearman / R².

How η bites
-----------
W feeds MAESTRO's UCB tool ranking. Per test sample (processed in
``--test-list`` order):

1. category ← SCOPE profile (or step2 record) — same resolution as
   ``generate_maestro_selections``.
2. MAESTRO picks the top-k available tools by the CURRENT W's UCB score
   (greedy, ≥1 Cat-A guaranteed — ``greedy_ucb_select``). No LLM call, so
   the ablation is API-free.
3. Build the Table-9 feature matrix for that tool subset (un-selected
   tools → NaN columns) + the SCOPE block, predict with a single
   LightGBM fusion model trained once on the train split (all available
   tools, SCOPE on — the MAESTRO-on / SCOPE-on design from
   ``table09_llm_modules``).
4. Reward r_k = max(0, tool k's per-sample Pearson R vs GT on the
   resolved-residue subset) for each available tool; EMA-update every
   metric cell W[k, ·, category]. Frozen (η=0) leaves W untouched, so its
   selections are constant across the whole pass.

Only the W *weights* are EMA-updated; UCB exploration counts are left at
the initial tensor's values so η=0 is exactly frozen and the formula
above is the sole knob. With a small enough ``--top-k`` (fewer than the
tools available per sample) the evolving ranking changes the selected
subset and the rows diverge; when nearly all tools are always selected
the rows converge — the expected "MAESTRO selection has limited effect"
regime noted in the plan.

Usage
-----
::

    python scripts/tables/table14_memory_eta.py \\
        --train-step4-dir   data/batch_train_v7/step4/ \\
        --test-step4-dir    data/batch_test_v7/step4/ \\
        --processed-dir     data/processed_quality \\
        --train-list        data/processed_quality/splits_tmscore_035/train.txt \\
        --test-list         data/processed_quality/splits_tmscore_035/test.txt \\
        --weight-tensor     data/batch_train_v7/W_v2.json \\
        --scope-profiles-test data/batch_test_v7/scope_profiles/ \\
        --output            data/batch_test_v7/table15_memory_eta.csv
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

from step3_tool_selection.weight_tensor import METRICS, WeightTensor  # noqa: E402
from step5_fusion.data_collector import per_residue_to_int_dict  # noqa: E402
from scripts.riboseer.ablation_fusion_method import per_sample_corr  # noqa: E402
from step5_fusion.lightgbm_fusion import (  # noqa: E402
    LGBM_OK,
    SampleT9,
    _load_json_dir,
    _train_model,
    build_sample_matrix,
    collect_samples,
    resolve_scope_vector,
)
from scripts.tables.table09_llm_modules import (  # noqa: E402
    _load_sample_ids,
)
from step5_fusion.features_15tool import main_score_field  # noqa: E402
from scripts.riboseer.generate_maestro_selections import (  # noqa: E402
    category_for, greedy_ucb_select, utility_scores,
)

# η grid (Table 15). Order = display order.
ETAS: tuple[float, ...] = (0.00, 0.05, 0.10, 0.20, 0.50, 1.00)


# ---------------------------------------------------------------------------
# Per-tool reward (this sample's quality signal r_k)
# ---------------------------------------------------------------------------


def tool_rewards(sample: SampleT9) -> dict[str, float]:
    """``{tool_id: r_k}`` for every available tool on this sample.

    r_k = max(0, the tool's per-sample Pearson R between its raw
    per-residue score and the binary GT, on the resolved-residue subset).
    Clamped to [0, 1] so it lands on the same scale as W's [0, 1] quality
    cells. Tools whose correlation is undefined (constant score) are
    omitted → no W update for them this round.
    """
    from step5_fusion.features_15tool import _index_predictions
    preds = _index_predictions(sample.step4_data)
    mask = sample.eval_mask
    rewards: dict[str, float] = {}
    for tool_id in sample.available_tools:
        pred = preds.get(tool_id)
        if pred is None:
            continue
        scores = per_residue_to_int_dict(pred.get(main_score_field(tool_id)))
        if not scores:
            scores = per_residue_to_int_dict(
                pred.get("per_residue_confidence"))
        if not scores:
            continue
        vec = np.fromiter((float(scores.get(r, 0.0))
                           for r in sample.residue_ids),
                          dtype=np.float64, count=len(sample.residue_ids))
        if mask.size != vec.size:
            continue
        corr = per_sample_corr(vec[mask], sample.y[mask])
        if corr is None or corr["pearson_r"] is None:
            continue
        rewards[tool_id] = min(max(corr["pearson_r"], 0.0), 1.0)
    return rewards


def ema_update(W: WeightTensor, tool_id: str, category: str,
               r: float, eta: float) -> None:
    """In-place EMA on every metric cell W[tool, ·, category]."""
    if eta <= 0.0:
        return  # frozen — leave W untouched
    for m in METRICS:
        old = W.get_weight(tool_id, m, category)
        W.update(tool_id, category, m, (1.0 - eta) * old + eta * r)


# ---------------------------------------------------------------------------
# One η pass (sequential online simulation)
# ---------------------------------------------------------------------------


def run_eta(eta: float, *, model, test: list[SampleT9],
            profiles_test: dict[str, dict], step2_dir: Optional[Path],
            tensor_path: Optional[Path], top_k: int) -> dict:
    """Process the test split in order under learning-rate ``eta`` and
    return the aggregate metric row."""
    W = (WeightTensor.load(tensor_path) if tensor_path and tensor_path.is_file()
         else WeightTensor())
    rewards_cache: dict[str, dict[str, float]] = {}

    corrs: list[dict] = []
    for s in test:
        profile = profiles_test.get(s.sid) or s.cauto_profile
        category = category_for(s.sid, step2_dir, profile)

        # MAESTRO selection under the current W.
        utility = utility_scores(W, s.available_tools, category)
        selected = greedy_ucb_select(s.available_tools, utility, top_k)

        # Fuse with the trained model over this sample's selected subset.
        X = build_sample_matrix(s, scope_on=True, maestro_on=True,
                                profiles={s.sid: profile}, selections={
                                    s.sid: {"selected_tools": selected}})
        vec = np.asarray(model.predict(X), dtype=np.float64)
        prob = {rid: float(vec[i]) for i, rid in enumerate(s.residue_ids)}

        corr = _corr_for_sample(prob, s)
        if corr is not None:
            corrs.append(corr)

        # MEMORY: EMA-update W from this sample's per-tool quality.
        if eta > 0.0:
            if s.sid not in rewards_cache:
                rewards_cache[s.sid] = tool_rewards(s)
            for tool_id, r in rewards_cache[s.sid].items():
                ema_update(W, tool_id, category, r, eta)

    return _aggregate(eta, corrs)


def _corr_for_sample(prob: dict[int, float], s: SampleT9) -> Optional[dict]:
    vec = np.fromiter((prob.get(r, 0.0) for r in s.residue_ids),
                      dtype=np.float64, count=len(s.residue_ids))
    mask = s.eval_mask
    if mask.size != vec.size:
        return None
    return per_sample_corr(vec[mask], s.y[mask])


def _mean(vs):
    return round(statistics.fmean(vs), 4) if vs else None


def _aggregate(eta: float, corrs: list[dict]) -> dict:
    prs = [c["pearson_r"] for c in corrs if c["pearson_r"] is not None]
    srs = [c["spearman_r"] for c in corrs if c["spearman_r"] is not None]
    r2s = [c["r_squared"] for c in corrs if c["r_squared"] is not None]
    return {
        "eta": eta,
        "n_samples": len(corrs),
        "pearson_r_mean": _mean(prs),
        "spearman_r_mean": _mean(srs),
        "r2_mean": _mean(r2s),
    }


# ---------------------------------------------------------------------------
# Train the shared fusion model (once, MAESTRO-on / SCOPE-on design)
# ---------------------------------------------------------------------------


def train_fusion(train: list[SampleT9],
                 profiles_train: dict[str, dict]):
    """One LightGBM over the train split with every available tool active
    and SCOPE on — the MAESTRO-on/SCOPE-on feature design that the
    per-η test passes predict against."""
    X = np.vstack([
        build_sample_matrix(s, scope_on=True, maestro_on=True,
                            profiles=profiles_train, selections={})
        for s in train])
    y = np.concatenate([s.y for s in train])
    return _train_model(X, y)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_COLUMNS = ["eta", "n_samples", "pearson_r_mean", "spearman_r_mean", "r2_mean"]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in _COLUMNS})


def print_table(rows: list[dict]) -> None:
    hdr = (f"{'eta':>5s} {'n':>4s} {'PearsonR':>9s} "
           f"{'SpearmanR':>10s} {'R2':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['eta']:>5.2f} {r['n_samples']:>4d} "
              f"{str(r['pearson_r_mean']):>9s} "
              f"{str(r['spearman_r_mean']):>10s} "
              f"{str(r['r2_mean']):>8s}")


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
    p.add_argument("--weight-tensor", type=Path, required=True,
                   help="initial W tensor JSON (e.g. W_v2.json)")
    p.add_argument("--scope-profiles-test", type=Path, default=None,
                   help="dir of test SCOPE profile JSONs (category + Group-4 "
                        "block); falls back to the cauto profile per sample")
    p.add_argument("--scope-profiles-train", type=Path, default=None,
                   help="dir of train SCOPE profile JSONs (training features)")
    p.add_argument("--step2-dir", type=Path, default=None,
                   help="dir of step2 JSONL records (authoritative category)")
    p.add_argument("--top-k", type=int, default=4,
                   help="MAESTRO greedy-UCB pick size (smaller → η bites "
                        "more; default 4)")
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
    if not args.weight_tensor.is_file():
        print(f"ERROR: weight tensor not found: {args.weight_tensor}",
              file=sys.stderr)
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

    print("training shared fusion model (MAESTRO-on / SCOPE-on) ...")
    model = train_fusion(train, profiles_train)

    rows: list[dict] = []
    for eta in ETAS:
        row = run_eta(eta, model=model, test=test,
                      profiles_test=profiles_test, step2_dir=args.step2_dir,
                      tensor_path=args.weight_tensor, top_k=args.top_k)
        rows.append(row)
        print(f"  eta={eta:.2f}  PearsonR={row['pearson_r_mean']}  "
              f"n={row['n_samples']}")

    write_csv(args.output, rows)
    print(f"wrote {args.output}  ({len(rows)} eta values, top_k={args.top_k})")
    print()
    print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
