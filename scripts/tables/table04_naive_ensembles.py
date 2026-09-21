"""Table 4 Category-E naive ensembles over the **full 15-tool library**.

Table 4's Cat-E rows (Simple Mean / Weighted Mean / Stacked LR) were
originally computed over the 5-6 tool core (see
``ablation_fusion_method``'s ``weighted_mean`` / ``stacked_lr``, which key
off ``TOOL_ORDER``). The headline system now selects from the full 15-tool
library, so the naive baselines are recomputed over the same 15 tools for
an apples-to-apples comparison.

The three ensembles operate on the **raw per-residue tool score**
(``per_residue_pae_score`` → ``per_residue_confidence`` fallback — the
identical signal ``table04_main_results.py`` and the other naive baselines
use). A tool that produced no score for a sample contributes a 0 vector.

* **Simple Mean** — per-residue mean across the tools *present* on that
  sample (uniform-denominator over present tools, matching ``mean_raw``).
* **Weighted Mean** — ``Σ_k w_k·s_k / Σ_k w_k`` over the present tools,
  with ``w_k = max(train per-sample Pearson R, 0)`` (the public-benchmark
  weighting: each tool's own Table-4 per-tool R on the TRAIN split,
  negatives clamped to 0).
* **Stacked LR** — sklearn ``LogisticRegression`` (``class_weight=
  'balanced'``) on the 15-D raw-score vector (one column per library tool,
  fixed order), trained on the concatenated train residues.

Metric: per-sample Pearson R on the resolved-residue subset (same
convention as Tables 4 / 16-25), then mean across the valid test samples.

Usage
-----
::

    python scripts/tables/table04_naive_ensembles.py \\
        --data-dir        data/processed_quality \\
        --train-step4-dir data/batch_train_v7/step4 \\
        --test-step4-dir  data/batch_test_v7/step4 \\
        --train-list      data/processed_quality/splits_tmscore_035/train.txt \\
        --test-list       data/processed_quality/splits_tmscore_035/test.txt \\
        --tools           boltz2 chai1 rosettafold2na rfaa alphafold3 \\
                          p2rank fpocket deeppocket equipnas nucleicnet \\
                          graphbind rnabindrplus bindup hdock haddock3
"""
from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
from scripts.riboseer.ablation_fusion_method import (  # noqa: E402
    per_sample_corr, _tool_raw_scores,
)
from step5_fusion.features_15tool import (  # noqa: E402
    ALL_KNOWN_TOOLS, _canonical, _index_predictions,
)

METHOD_ORDER = ["simple_mean", "weighted_mean", "stacked_lr"]


@dataclass
class ENSample:
    sid: str
    residue_ids: list[int]
    y: np.ndarray
    eval_mask: np.ndarray
    # {tool_id: {res_id: raw_score}} for every present (successful) tool.
    raw_by_tool: dict[str, dict[int, float]] = field(default_factory=dict)

    def tool_vector(self, tool_id: str) -> np.ndarray:
        scores = self.raw_by_tool.get(tool_id, {})
        return np.fromiter(
            (float(scores.get(r, 0.0)) for r in self.residue_ids),
            dtype=np.float64, count=len(self.residue_ids))


def collect(step4_dir: Path, data_dir: Path, ids: list[str],
            tools: list[str]) -> list[ENSample]:
    """One ENSample per usable sample (GT + length present + ≥1 of
    ``tools`` produced a score)."""
    canon = [_canonical(t) for t in tools]
    out: list[ENSample] = []
    for sid in ids:
        s4 = read_jsonl_record(step4_dir / f"{sid}.jsonl")
        if s4 is None:
            continue
        sample = load_sample_json(data_dir, sid)
        if sample is None:
            continue
        prot = sample.get("protein") or {}
        length = int(prot.get("length") or len(prot.get("sequence") or "") or 0)
        gt = {int(r) for r in (sample.get("interaction") or {})
              .get("binding_protein_residues") or []}
        if not length or not gt:
            continue
        preds = _index_predictions(s4)
        raw: dict[str, dict[int, float]] = {}
        for t in canon:
            p = preds.get(t)
            if p is None:
                continue
            scores = _tool_raw_scores(p)
            if scores:
                raw[t] = scores
        if not raw:
            continue

        residue_ids = list(range(1, length + 1))
        y = np.fromiter((1.0 if r in gt else 0.0 for r in residue_ids),
                        dtype=np.float64, count=length)
        resolved = prot.get("resolved_residues") or []
        if resolved:
            rs = {int(r) for r in resolved}
            mask = np.fromiter((r in rs for r in residue_ids),
                               dtype=bool, count=length)
        else:
            mask = np.ones(length, dtype=bool)
        out.append(ENSample(sid=sid, residue_ids=residue_ids, y=y,
                            eval_mask=mask, raw_by_tool=raw))
    return out


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------


def predict_simple_mean(s: ENSample) -> np.ndarray:
    present = list(s.raw_by_tool)
    if not present:
        return np.zeros(len(s.residue_ids))
    stack = np.vstack([s.tool_vector(t) for t in present])
    return stack.mean(axis=0)


def compute_tool_weights(train: list[ENSample], tools: list[str]
                         ) -> dict[str, float]:
    """``w_t = mean train per-sample Pearson R`` (resolved subset). Tools
    with no defined train correlation → 0."""
    weights: dict[str, float] = {}
    for t in tools:
        prs: list[float] = []
        for s in train:
            if t not in s.raw_by_tool:
                continue
            m = s.eval_mask
            corr = per_sample_corr(s.tool_vector(t)[m], s.y[m])
            if corr is not None and corr["pearson_r"] is not None:
                prs.append(corr["pearson_r"])
        weights[t] = statistics.fmean(prs) if prs else 0.0
    return weights


def predict_weighted_mean(s: ENSample, weights: dict[str, float]
                          ) -> np.ndarray:
    n = len(s.residue_ids)
    num = np.zeros(n, dtype=np.float64)
    den = 0.0
    for t in s.raw_by_tool:
        w = max(weights.get(t, 0.0), 0.0)
        if w <= 0.0:
            continue
        num += w * s.tool_vector(t)
        den += w
    return num / den if den > 0.0 else num


def _stack_features(s: ENSample, tools: list[str]) -> np.ndarray:
    """``(n_residues × n_tools)`` raw-score matrix in fixed ``tools`` order
    (0 where a tool is absent) — the stacked-LR feature space."""
    if not tools:
        return np.zeros((len(s.residue_ids), 0))
    return np.column_stack([s.tool_vector(t) for t in tools])


def fit_stacked_lr(train: list[ENSample], tools: list[str]):
    from sklearn.linear_model import LogisticRegression
    if not tools or not train:
        return None
    X = np.vstack([_stack_features(s, tools) for s in train])
    y = np.concatenate([s.y for s in train]).astype(int)
    clf = LogisticRegression(class_weight="balanced", solver="lbfgs",
                             max_iter=1000)
    clf.fit(X, y)
    return clf


def predict_stacked_lr(clf, s: ENSample, tools: list[str]) -> np.ndarray:
    proba = clf.predict_proba(_stack_features(s, tools))
    return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _eval(pred: np.ndarray, s: ENSample) -> Optional[dict]:
    """Per-sample Pearson / Spearman / R² on the resolved subset (same
    convention as the other tables); None when undefined."""
    m = s.eval_mask
    if m.size != pred.size:
        return None
    return per_sample_corr(pred[m], s.y[m])


def evaluate(train: list[ENSample], test: list[ENSample],
             tools: list[str]
             ) -> tuple[dict[str, list[dict]], dict[str, float]]:
    weights = compute_tool_weights(train, tools)
    clf = fit_stacked_lr(train, tools)
    out: dict[str, list[dict]] = {m: [] for m in METHOD_ORDER}
    for s in test:
        for name, vec in (
            ("simple_mean", predict_simple_mean(s)),
            ("weighted_mean", predict_weighted_mean(s, weights)),
            ("stacked_lr", (predict_stacked_lr(clf, s, tools)
                            if clf is not None else None)),
        ):
            if vec is None:
                continue
            corr = _eval(vec, s)
            if corr is not None:
                out[name].append(corr)
    return out, weights


def _mean_of(corrs: list[dict], key: str) -> Optional[float]:
    vs = [c[key] for c in corrs if c.get(key) is not None]
    return round(statistics.fmean(vs), 4) if vs else None


def print_table(dist: dict[str, list[dict]], weights: dict[str, float],
                tools: list[str]) -> None:
    print("=== Table 4: Category-E naive ensembles (15-tool) ===")
    print("weighted_mean weights (train per-tool Pearson R, clamped ≥0):")
    norm = sum(max(w, 0.0) for w in weights.values()) or 1.0
    for t in tools:
        w = weights.get(t, 0.0)
        print(f"    {t:16s} R={w:+.4f}  w={max(w, 0.0) / norm:.4f}")
    print()
    hdr = (f"{'method':16s} {'n':>4s} {'PearsonR':>9s} "
           f"{'SpearmanR':>10s} {'R2':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for m in METHOD_ORDER:
        corrs = dist[m]
        print(f"{m:16s} {len(corrs):>4d} "
              f"{str(_mean_of(corrs, 'pearson_r')):>9s} "
              f"{str(_mean_of(corrs, 'spearman_r')):>10s} "
              f"{str(_mean_of(corrs, 'r_squared')):>8s}")


def _load_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"sample list not found: {path}")
    return [ln.split()[0] for ln in
            path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--train-step4-dir", type=Path, required=True)
    p.add_argument("--test-step4-dir", type=Path, required=True)
    p.add_argument("--train-list", type=Path, required=True)
    p.add_argument("--test-list", type=Path, required=True)
    p.add_argument("--tools", type=str, nargs="+",
                   default=list(ALL_KNOWN_TOOLS),
                   help="library tool ids (default = all 15).")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for d in (args.data_dir, args.train_step4_dir, args.test_step4_dir):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 1
    try:
        train_ids = _load_ids(args.train_list)
        test_ids = _load_ids(args.test_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    tools = [_canonical(t) for t in args.tools]
    train = collect(args.train_step4_dir, args.data_dir, train_ids, tools)
    test = collect(args.test_step4_dir, args.data_dir, test_ids, tools)
    print(f"usable: train={len(train)}, test={len(test)}  "
          f"({len(tools)} tools)")
    if not train or not test:
        print("ERROR: no usable train/test samples", file=sys.stderr)
        return 1

    dist, weights = evaluate(train, test, tools)
    print()
    print_table(dist, weights, tools)
    return 0


if __name__ == "__main__":
    sys.exit(main())
