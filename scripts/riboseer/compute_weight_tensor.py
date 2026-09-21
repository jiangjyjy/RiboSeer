#!/usr/bin/env python3
"""Compute the real UCB weight tensor W.json for all 15 library tools.

The shipped W tensor only carries the original 5 tools' history, so at
MAESTRO selection time the UCB utilities of the 8 newer tools fall back to
flat per-category priors and the LLM rarely picks them. Now that the
training split has step4 predictions for 13 tools, we can estimate each
tool's real reliability μ̂(k) = mean per-sample Pearson R on the training
set, and N(k) = number of training samples on which the tool produced a
usable prediction.

Two tools (``alphafold3``, ``bindup``) were never run on the training
split, so their μ̂ is fixed in code from their **test-set** Pearson R, with
a deliberately small N to flag the limited evidence:

    alphafold3 : μ̂ = 0.246, N = 100
    bindup     : μ̂ = 0.250, N = 100

Encoding (WeightTensor-compatible, see step3_tool_selection.weight_tensor)
--------------------------------------------------------------------------
The per-residue Pearson R is a single scalar per tool, while the tensor is
W[tool, metric, category]. μ̂(k) is global (not category-specific), so we
write it into **all 5 metrics** under the **category-agnostic global cell**
(``weight_tensor.GLOBAL_CATEGORY``) and set ``counts[tool][GLOBAL] = N(k)``.
``WeightTensor.get_weight`` falls back to this cell for *any* category that
lacks a specific value, so a sample resolving to e.g. ``novel_x_stem-loop``
still gets the real μ̂ — instead of the UCB=0.000 / default-prior result of
writing under one concrete category. The per-category-letter ``defaults``
are also set to the **data-driven** mean μ̂ of each letter's tools (a sane
fallback for any tool that somehow lacks a global cell). The result loads
cleanly via ``WeightTensor.load`` and ranks high-μ̂ tools first.

Usage
-----
::

    python scripts/riboseer/compute_weight_tensor.py \\
        --step4-dir     data/batch_train_v7/step4/ \\
        --processed-dir data/processed_quality \\
        --sample-list   data/processed_quality/splits_tmscore_035/train.txt \\
        --output        data/batch_train_v7/W_v2.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step5_fusion.data_collector import per_residue_to_int_dict  # noqa: E402
from step3_tool_selection.weight_tensor import (  # noqa: E402
    GLOBAL_CATEGORY, METRICS, WeightTensor, _DEFAULT_CATEGORY_WEIGHTS,
)
from scripts.riboseer.ablation_fusion_method import per_sample_corr  # noqa: E402
from step5_fusion.lightgbm_fusion import (  # noqa: E402
    SampleT9,
    collect_samples,
)
from scripts.tables.table09_llm_modules import (  # noqa: E402
    _load_sample_ids,
)
from step5_fusion.features_15tool import (  # noqa: E402
    ALL_KNOWN_TOOLS, TOOL_CATEGORY, _index_predictions,
)

# μ̂ is a single global scalar per tool (not category-specific), so it is
# written into the WeightTensor's category-agnostic GLOBAL_CATEGORY cell.
# Any MAESTRO query — whatever pocket category a sample resolves to —
# then falls back to this cell and gets the real μ̂ (see weight_tensor.
# get_weight). Writing it under a single concrete category instead would
# only match samples with that exact category and return 0/defaults for
# the rest, which is the UCB=0.000 bug this fixes.
STORE_CATEGORY = GLOBAL_CATEGORY

# Tools absent from the training split: μ̂ fixed from their test-set
# Pearson R, with a small N to mark the limited evidence. Hard-coded by
# design — not exposed as a CLI flag.
TEST_ESTIMATE_TOOLS: dict[str, tuple[float, int]] = {
    "alphafold3": (0.246, 100),
    "bindup": (0.250, 100),
}


# ---------------------------------------------------------------------------
# Per-tool μ̂ on the training split
# ---------------------------------------------------------------------------


def _tool_pred(pred: dict) -> dict[int, float]:
    """Per-residue score for one tool prediction (pae → confidence)."""
    pae = per_residue_to_int_dict(pred.get("per_residue_pae_score"))
    return pae if pae else per_residue_to_int_dict(
        pred.get("per_residue_confidence"))


def per_tool_pearson(samples: list[SampleT9]
                     ) -> tuple[dict[str, float], dict[str, int]]:
    """μ̂(k) = mean per-sample Pearson R, N(k) = #samples with a defined
    correlation, over every tool that appears in ``samples``' step4."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for s in samples:
        for tool, pred in _index_predictions(s.step4_data).items():
            scores = _tool_pred(pred)
            if not scores:
                continue
            vec = np.fromiter((scores.get(r, 0.0) for r in s.residue_ids),
                              dtype=np.float64, count=len(s.residue_ids))
            c = per_sample_corr(vec[s.eval_mask], s.y[s.eval_mask])
            if c is None or c["pearson_r"] is None:
                continue
            sums[tool] = sums.get(tool, 0.0) + float(c["pearson_r"])
            counts[tool] = counts.get(tool, 0) + 1
    mu = {t: sums[t] / counts[t] for t in counts}
    return mu, counts


# ---------------------------------------------------------------------------
# Assemble per-tool records (15 tools)
# ---------------------------------------------------------------------------


def assemble_tool_stats(mu_train: dict[str, float], n_train: dict[str, int]
                        ) -> list[dict]:
    """One record per library tool: μ̂, N, and provenance.

    Priority: fixed test-estimate (af3/bindup) > training estimate >
    category prior (tool with no data anywhere)."""
    rows: list[dict] = []
    for tool in ALL_KNOWN_TOOLS:
        cat = TOOL_CATEGORY.get(tool, "C")
        if tool in TEST_ESTIMATE_TOOLS:
            mu, n = TEST_ESTIMATE_TOOLS[tool]
            source = "test_estimate"
        elif n_train.get(tool, 0) > 0:
            mu, n, source = round(mu_train[tool], 6), n_train[tool], "train"
        else:
            mu = _DEFAULT_CATEGORY_WEIGHTS.get(cat, 0.5)
            n, source = 0, "prior"
        rows.append({"tool_id": tool, "category": cat, "mu_hat": float(mu),
                     "n": int(n), "source": source})
    return rows


def _letter_defaults(rows: list[dict]) -> dict[str, float]:
    """Data-driven per-category-letter priors: mean μ̂ of the letter's
    tools that carry real evidence (N>0); static prior when none do."""
    out: dict[str, float] = {}
    for letter, static in _DEFAULT_CATEGORY_WEIGHTS.items():
        vals = [r["mu_hat"] for r in rows
                if r["category"] == letter and r["n"] > 0]
        out[letter] = round(sum(vals) / len(vals), 6) if vals else static
    return out


def build_weight_tensor(rows: list[dict], *, beta: float = 1.0
                        ) -> WeightTensor:
    """Encode μ̂ into all 5 metrics under the global cell, N into counts,
    and data-driven letter priors into defaults."""
    data: dict[str, dict[str, dict[str, float]]] = {}
    counts: dict[str, dict[str, int]] = {}
    for r in rows:
        for m in METRICS:
            data.setdefault(r["tool_id"], {}).setdefault(
                m, {})[STORE_CATEGORY] = r["mu_hat"]
        counts.setdefault(r["tool_id"], {})[STORE_CATEGORY] = r["n"]
    return WeightTensor(data=data, counts=counts,
                        default_weights=_letter_defaults(rows), beta=beta)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_summary(rows: list[dict], wt: WeightTensor) -> None:
    print(f"\n{'tool_id':<16s}{'μ̂':>8s}{'N':>6s}{'UCB_score':>11s}  source")
    print("-" * 52)
    for r in rows:
        ucb = wt.compute_utility(r["tool_id"], STORE_CATEGORY)
        print(f"{r['tool_id']:<16s}{r['mu_hat']:>8.3f}{r['n']:>6d}"
              f"{ucb:>11.3f}  {r['source']}")


def write_summary_csv(path: Path, rows: list[dict], wt: WeightTensor) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tool_id", "category", "mu_hat", "n", "ucb_score",
                    "source"])
        for r in rows:
            ucb = round(wt.compute_utility(r["tool_id"], STORE_CATEGORY), 6)
            w.writerow([r["tool_id"], r["category"], r["mu_hat"], r["n"],
                        ucb, r["source"]])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step4-dir", type=Path, required=True)
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--sample-list", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True,
                   help="W tensor JSON (WeightTensor.save format)")
    p.add_argument("--beta", type=float, default=1.0,
                   help="UCB exploration coefficient (default 1.0)")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not args.step4_dir.is_dir() or not args.processed_dir.is_dir():
        print("ERROR: --step4-dir / --processed-dir must be dirs",
              file=sys.stderr)
        return 1
    try:
        ids = _load_sample_ids(args.sample_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    samples = collect_samples(args.step4_dir, args.processed_dir, ids)
    print(f"usable training samples: {len(samples)}")
    if not samples:
        print("ERROR: no usable samples", file=sys.stderr)
        return 1

    mu_train, n_train = per_tool_pearson(samples)
    rows = assemble_tool_stats(mu_train, n_train)
    wt = build_weight_tensor(rows, beta=args.beta)

    wt.save(args.output)
    write_summary_csv(args.output.with_suffix(".summary.csv"), rows, wt)
    print_summary(rows, wt)
    print(f"\nwrote {args.output}  (15 tools; "
          f"{sum(1 for r in rows if r['source'] == 'train')} from train, "
          f"{sum(1 for r in rows if r['source'] == 'test_estimate')} "
          f"test-estimate, "
          f"{sum(1 for r in rows if r['source'] == 'prior')} prior)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
