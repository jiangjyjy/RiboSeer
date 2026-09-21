"""Table 24 — 5-seed variance.

Under the default spectral-cut split, retrains the deterministic off/off/off
LightGBM fusion (5 default tools, no LLM) with 5 random seeds and reports
mean ± std of the per-sample metrics. The LLM outputs are frozen (never
re-called); only the fusion model's ``random_state`` varies.

Metrics: Pearson R, Spearman R, R² (per-sample mean), and Top-k F1 (reusing
Table 21's ``compute_topk_metrics`` with k = |Bp| on the resolved subset).

With ``--compute-dcc --raw-dir <dir>`` it also reports **DCC (Å)** — the
distance between the predicted-pocket and GT-pocket Cα centroids (predicted
pocket = the top-|Bp| residues by RiboSeer score), reusing the Table-5
pocket-geometry reader (``table05_pocket_geometry``). Needs gemmi + raw
structures. **DockQ** stays "–": it needs a predicted Cat-A complex (Table-6
``table06_complex_quality``) and is not a function of the LightGBM seed
(the fusion reranks residues, not structures) — left as TODO.

Run
---
::

    python scripts/tables/table23_seed_variance.py \\
        --data-dir      data/processed_quality \\
        --step4-dirs    data/batch_train_v7/step4 data/batch_test_v7/step4 \\
        --train-list    data/processed_quality/splits_tmscore_035/train.txt \\
        --test-list     data/processed_quality/splits_tmscore_035/test.txt \\
        --model-dir     data/enriched_v7_lgbm \\
        --default-tools boltz2 chai1 rosettafold2na equipnas p2rank \\
        --seeds         0 42 123 456 789
"""
from __future__ import annotations

import argparse
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

from step5_fusion.data_collector import load_sample_json  # noqa: E402
from scripts.riboseer.retrain_eval_common import (  # noqa: E402
    DEFAULT_TOOLS, LGBM_OK, Sample, collect_samples, load_ids, make_lgbm,
    mean_metric, riboseer_corrs, train_predict,
)
from scripts.tables.table20_topk import compute_topk_metrics  # noqa: E402
from scripts.tables.table05_pocket_geometry import (  # noqa: E402
    compute_dcc, extract_chain_coords, _resolve_structure_path,
)
from scripts.tables import table09_llm_modules as alm  # noqa: E402
from step5_fusion.prediction_io import predictions_to_vector  # noqa: E402

SEEDS = [0, 42, 123, 456, 789]
_BASE_KEYS = ["pearson_r", "spearman_r", "r2", "topk_f1"]
_METRIC_LABELS = {"pearson_r": "PearsonR", "spearman_r": "SpearmanR",
                  "r2": "R2", "topk_f1": "TopkF1", "dcc": "DCC(A)"}
# DockQ needs a predicted 3D complex (Cat-A consensus) vs experimental + the
# DockQ program (see table06_complex_quality.py / Table 6); it is NOT a
# function of the LightGBM seed (the fusion reranks residues, not structures),
# so it is reported as "–" here. TODO: wire table06_complex_quality if wanted.
_SHOW_DOCKQ = True


def mean_topk_f1(test: list[Sample], preds: dict[str, np.ndarray]
                 ) -> Optional[float]:
    """Mean Top-k F1 (k=|Bp|) over the resolved subset — reuses Table 21."""
    f1s: list[float] = []
    for s in test:
        m = s.eval_mask
        y = s.y[m].astype(int)
        k = int(y.sum())
        if k <= 0 or k >= y.size:
            continue
        res = compute_topk_metrics(y, np.asarray(preds[s.sid])[m], k)
        if res is not None:
            f1s.append(res[2])
    return round(statistics.fmean(f1s), 4) if f1s else None


def build_dcc_context(test: list[Sample], data_dir: Path, raw_dir: Path
                      ) -> tuple[dict[str, tuple[list[int], dict]], int]:
    """``{sid: (gt_binding_residues, ca_map)}`` (+ skipped count) for DCC.

    Parses each test sample's raw structure ONCE for Cα coords (keyed by
    label_seq, matching residue_ids), via the Table-5 pocket-geometry
    reader. Seed-independent, so it's built once and reused across seeds."""
    ctx: dict[str, tuple[list[int], dict]] = {}
    skipped = 0
    for s in test:
        sample = load_sample_json(data_dir, s.sid)
        if sample is None:
            skipped += 1
            continue
        path = _resolve_structure_path(raw_dir, sample.get("source_pdb") or "")
        if path is None:
            skipped += 1
            continue
        prot_chain = (sample.get("protein") or {}).get("chain_id")
        rna_chain = (sample.get("rna") or {}).get("chain_id")
        try:
            ca, _rna = extract_chain_coords(path, prot_chain, rna_chain)
        except (FileNotFoundError, ValueError):
            skipped += 1
            continue
        gt = sorted({int(r) for r in (sample.get("interaction") or {})
                     .get("binding_protein_residues") or []})
        if not ca or not gt:
            skipped += 1
            continue
        ctx[s.sid] = (gt, ca)
    return ctx, skipped


def mean_dcc(test: list[Sample], preds: dict[str, np.ndarray],
             ctx: dict[str, tuple[list[int], dict]]) -> Optional[float]:
    """Mean DCC (Å) over samples: predicted pocket = top-|Bp| residues by
    RiboSeer score; DCC = ‖centroid(P̂) − centroid(P*)‖ (lower = better)."""
    vals: list[float] = []
    for s in test:
        c = ctx.get(s.sid)
        if c is None:
            continue
        gt, ca = c
        k = len(gt)
        pred = np.asarray(preds[s.sid], dtype=np.float64)
        order = np.argsort(pred)[::-1][:k]
        topk = [s.residue_ids[i] for i in order]
        d = compute_dcc(topk, gt, ca)
        if d is not None:
            vals.append(d)
    return round(statistics.fmean(vals), 4) if vals else None


# ---------------------------------------------------------------------------
# Full-system (on/on/on) per-seed retrain — SCOPE + MAESTRO + POLISH
# ---------------------------------------------------------------------------


class _FSContext:
    """Frozen LLM inputs for the full-system seed retrain (all seed-
    independent): SCOPE profiles + MAESTRO selections (train/test) and the
    POLISH actions. Only the LightGBM ``random_state`` changes per seed."""

    def __init__(self, profiles_train, profiles_test,
                 selections_train, selections_test, polish_actions):
        self.profiles_train = profiles_train
        self.profiles_test = profiles_test
        self.selections_train = selections_train
        self.selections_test = selections_test
        self.polish_actions = polish_actions


def fs_train_predict(train, test, seed: int, ctx: _FSContext
                     ) -> dict[str, np.ndarray]:
    """Train the on/on/on (SCOPE+MAESTRO, 15-tool) LightGBM for one seed,
    predict the test set, apply POLISH, and return ``{sid: pred_vector}``
    over each sample's residue range (post-POLISH). Mirrors the headline
    full-system pipeline (``table09_llm_modules`` on/on/on), seedable."""
    X = np.vstack([
        alm.build_sample_matrix(s, True, True,
                                ctx.profiles_train, ctx.selections_train,
                                mandatory=alm.MANDATORY_TOOLS)
        for s in train])
    y = np.concatenate([s.y for s in train])
    model = make_lgbm(seed)
    model.fit(X, y)
    out: dict[str, np.ndarray] = {}
    for s in test:
        Xs = alm.build_sample_matrix(s, True, True,
                                     ctx.profiles_test, ctx.selections_test,
                                     mandatory=alm.MANDATORY_TOOLS)
        vec = np.asarray(model.predict(Xs), dtype=np.float64)
        prob = {rid: float(vec[i]) for i, rid in enumerate(s.residue_ids)}
        prob = alm.apply_polish(prob, s, True, ctx.polish_actions)
        out[s.sid] = predictions_to_vector(prob, s.residue_ids)
    return out


def run_seed(train: list[Sample], test: list[Sample], seed: int,
             dcc_ctx: Optional[dict] = None,
             fs_ctx: Optional[_FSContext] = None) -> dict:
    if fs_ctx is not None:
        preds = fs_train_predict(train, test, seed, fs_ctx)
    else:
        preds = train_predict(train, test, seed)
    corrs = riboseer_corrs(test, preds)
    row = {
        "seed": seed,
        "pearson_r": mean_metric(corrs, "pearson_r"),
        "spearman_r": mean_metric(corrs, "spearman_r"),
        "r2": mean_metric(corrs, "r_squared"),
        "topk_f1": mean_topk_f1(test, preds),
    }
    if dcc_ctx is not None:
        row["dcc"] = mean_dcc(test, preds, dcc_ctx)
    return row


def _agg(rows: list[dict], key: str, fn) -> Optional[float]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return round(fn(vals), 4) if len(vals) > 1 else round(vals[0], 4)


def _cell(v: Optional[float]) -> str:
    return f"{v:.4f}" if v is not None else "–"


def print_table(rows: list[dict], metric_keys: list[str],
                show_dockq: bool) -> None:
    print("=== Table 24: 5-Seed Variance ===")
    cols = [_METRIC_LABELS[k] for k in metric_keys] + (
        ["DockQ"] if show_dockq else [])
    hdr = f"{'Seed':>5s} " + " ".join(f"{c:>10s}" for c in cols)
    print(hdr)
    for r in rows:
        cells = [_cell(r.get(k)) for k in metric_keys] + (
            ["–"] if show_dockq else [])
        print(f"{r['seed']:>5d} " + " ".join(f"{c:>10s}" for c in cells))
    print("-" * len(hdr))

    def stat(label, fn):
        cells = [_cell(_agg(rows, k, fn)) for k in metric_keys] + (
            ["–"] if show_dockq else [])
        print(f"{label:>5s} " + " ".join(f"{c:>10s}" for c in cells))
    stat("Mean", statistics.fmean)
    stat("Std", lambda vs: statistics.pstdev(vs) if len(vs) > 1 else 0.0)
    print()
    if "dcc" in metric_keys:
        print("NOTE: DCC (Å) lower = better (predicted vs GT Cα-centroid "
              "distance, predicted pocket = top-|Bp| residues).")
    else:
        print("NOTE: DCC not computed (pass --compute-dcc --raw-dir ...).")
    if show_dockq:
        print("NOTE: DockQ ('–') needs a predicted Cat-A complex (Table-6 "
              "table06_complex_quality) and is seed-independent — TODO.")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--step4-dirs", type=Path, nargs="+", required=True,
                   help="one or more step4 JSONL dirs (probed in order)")
    p.add_argument("--train-list", type=Path, required=True)
    p.add_argument("--test-list", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=None,
                   help="accepted for parity; this table retrains from "
                        "scratch and does not load the bundle")
    p.add_argument("--default-tools", type=str, nargs="+",
                   default=list(DEFAULT_TOOLS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--compute-dcc", action="store_true",
                   help="also compute DCC (Å); requires --raw-dir + gemmi")
    p.add_argument("--raw-dir", type=Path, default=None,
                   help="raw structure dir for DCC (.cif/.pdb)")
    # ---- full-system (on/on/on) variance ----
    p.add_argument("--full-system", action="store_true",
                   help="retrain the headline on/on/on pipeline per seed "
                        "(SCOPE+MAESTRO 15-tool features + POLISH) instead of "
                        "the off/off/off 5-tool baseline. Requires exactly 2 "
                        "--step4-dirs (train then test).")
    p.add_argument("--scope-dir-train", type=Path, default=None)
    p.add_argument("--scope-dir-test", type=Path, default=None)
    p.add_argument("--maestro-dir-train", type=Path, default=None)
    p.add_argument("--maestro-dir-test", type=Path, default=None)
    p.add_argument("--polish-dir", type=Path, default=None,
                   help="LLM POLISH actions (test); full-system only")
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
        train_ids = load_ids(args.train_list)
        test_ids = load_ids(args.test_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    fs_ctx = None
    if args.full_system:
        if len(args.step4_dirs) != 2:
            print("ERROR: --full-system needs exactly 2 --step4-dirs "
                  "(train then test)", file=sys.stderr)
            return 1
        train_step4, test_step4 = args.step4_dirs
        print(f"collecting full-system (on/on/on) samples: "
              f"train={train_step4}, test={test_step4} ...")
        train = alm.collect_samples(train_step4, args.data_dir, train_ids)
        test = alm.collect_samples(test_step4, args.data_dir, test_ids)
        fs_ctx = _FSContext(
            profiles_train=alm._load_json_dir(args.scope_dir_train),
            profiles_test=alm._load_json_dir(args.scope_dir_test),
            selections_train=alm._load_json_dir(args.maestro_dir_train),
            selections_test=alm._load_json_dir(args.maestro_dir_test),
            polish_actions=alm._load_json_dir(args.polish_dir),
        )
    else:
        ids = train_ids + test_ids
        print(f"collecting {len(ids)} samples (off/off/off, "
              f"tools={args.default_tools}) ...")
        samples = collect_samples(args.step4_dirs, args.data_dir, ids,
                                  args.default_tools)
        train = [samples[i] for i in train_ids if i in samples]
        test = [samples[i] for i in test_ids if i in samples]
    print(f"usable: train={len(train)}, test={len(test)}")
    if not train or not test:
        print("ERROR: no usable train/test samples", file=sys.stderr)
        return 1

    metric_keys = list(_BASE_KEYS)
    dcc_ctx = None
    if args.compute_dcc:
        if args.raw_dir is None or not args.raw_dir.is_dir():
            print("ERROR: --compute-dcc requires a valid --raw-dir",
                  file=sys.stderr)
            return 1
        dcc_ctx, skipped = build_dcc_context(test, args.data_dir, args.raw_dir)
        print(f"DCC: Cα coords for {len(dcc_ctx)}/{len(test)} test samples "
              f"(skipped {skipped})")
        metric_keys.append("dcc")

    rows = [run_seed(train, test, s, dcc_ctx, fs_ctx) for s in args.seeds]
    print()
    print_table(rows, metric_keys, _SHOW_DOCKQ)
    return 0


if __name__ == "__main__":
    sys.exit(main())
