"""Aggregate batch_predict results into paper-ready evaluation tables.

Reads ``summary/results.jsonl`` produced by ``batch_predict.py`` and
emits these files:

  - ``aggregate.json``   Overall metrics + by-category / by-quality
                         breakdowns + tool success rates +
                         PocketQA→F1 correlation. The number you'd
                         quote in a paper.
  - ``by_category.csv``  One row per category with mean P/R/F1 and N.
  - ``per_sample.csv``   One row per sample (results.jsonl flattened
                         to CSV — Excel-friendly view of the same data).
  - ``tool_analysis.csv``  Per-tool success rate + count of samples
                           that listed it in ``tools_used``.
  - ``pocketqa_vs_f1.csv``  Two columns (qa_total, f1) for scatter
                            plots. Useful for sanity-checking that
                            QA actually predicts F1.

When ``--step4-dir`` and ``--step5-dir`` are also given, plus
``--processed-dir``, two extra paper-ready files are emitted:

  - ``per_residue_correlation.json``  Per-method aggregate (mean / std /
                                      median) of Pearson R / Spearman R /
                                      R² between each tool's per-residue
                                      probability vector and the binary
                                      ground-truth vector.
  - ``per_residue_correlation.csv``   Flat CSV of the same numbers — the
                                      shape paper tables consume.

Stdlib only (no pandas/numpy) so this can run on any host with the
project Python env installed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional


# ---------- IO helpers ----------------------------------------------------


def load_results(path: Path) -> list[dict]:
    """Read a JSONL file → list of dicts. Skips blank / # lines."""
    if not path.is_file():
        raise FileNotFoundError(f"results file not found: {path}")
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"WARN: skipping malformed JSONL line: {e}",
                  file=sys.stderr)
    return out


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in columns})


# ---------- aggregate math ------------------------------------------------


def _coerce_float(v) -> Optional[float]:
    """Float-or-None coercion. NaN/inf treated as None too."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _mean(values: Iterable[float]) -> Optional[float]:
    """Return the mean of ``values`` or ``None`` if the iterable is empty.

    Round to 4 decimals to keep the JSON readable; downstream consumers
    who need full precision should re-aggregate from per_sample.csv.
    """
    vs = list(values)
    if not vs:
        return None
    return round(statistics.fmean(vs), 4)


def _median(values: Iterable[float]) -> Optional[float]:
    vs = list(values)
    if not vs:
        return None
    return round(statistics.median(vs), 4)


def _std(values: Iterable[float]) -> Optional[float]:
    vs = list(values)
    if len(vs) < 2:
        return 0.0 if vs else None
    return round(statistics.pstdev(vs), 4)


def _pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    """Pearson r of two equal-length lists. Returns None on degenerate
    input (n < 2 or zero variance in either dimension)."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx2 = sum((x - mx) ** 2 for x in xs)
    sy2 = sum((y - my) ** 2 for y in ys)
    if sx2 == 0 or sy2 == 0:
        return None
    return round(cov / math.sqrt(sx2 * sy2), 4)


def _spearman(xs: list[float], ys: list[float]) -> Optional[float]:
    """Spearman ρ via rank-then-Pearson. Ties get average ranks
    (the standard convention in scipy.stats.spearmanr)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    def _ranks(vs: list[float]) -> list[float]:
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        ranks = [0.0] * len(vs)
        # Average ties.
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # 1-based avg rank
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    return _pearson(_ranks(xs), _ranks(ys))


# ---------- summarisers ---------------------------------------------------


def aggregate(results: list[dict]) -> dict:
    """Compute the overall + breakdowns block for ``aggregate.json``."""
    n_total = len(results)
    # "success" = a row landed in results.jsonl at all (the failures
    # list lives in failures.jsonl, which this script doesn't read).
    # We further partition by whether F1 is computable.
    f1s = [_coerce_float(r.get("f1")) for r in results]
    f1s_present = [v for v in f1s if v is not None]
    p_present = [v for v in (_coerce_float(r.get("precision"))
                             for r in results) if v is not None]
    r_present = [v for v in (_coerce_float(r.get("recall"))
                             for r in results) if v is not None]

    # By-category breakdown.
    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in results:
        f1 = _coerce_float(r.get("f1"))
        if f1 is None:
            continue
        by_cat[r.get("category") or "unknown"].append(f1)
    by_cat_block = {
        cat: {
            "n": len(vs),
            "mean_f1": _mean(vs),
            "median_f1": _median(vs),
        }
        for cat, vs in sorted(by_cat.items())
    }

    # By-quality-tier breakdown (high/medium/low/...).
    by_q: dict[str, list[float]] = defaultdict(list)
    for r in results:
        f1 = _coerce_float(r.get("f1"))
        if f1 is None:
            continue
        by_q[r.get("quality_tier") or "unknown"].append(f1)
    by_q_block = {
        q: {
            "n": len(vs),
            "mean_f1": _mean(vs),
            "median_f1": _median(vs),
        }
        for q, vs in sorted(by_q.items())
    }

    # Tool success: how often each tool was *attempted* vs *succeeded*.
    # results.jsonl only carries ``tools_used`` (the SUCCEEDED set),
    # so we infer attempts from ``tools_attempted`` / ``tools_succeeded``
    # counts (tool-level breakdown needs per-step JSONLs to be exact).
    tool_used_counts: Counter[str] = Counter()
    for r in results:
        for t in r.get("tools_used") or []:
            tool_used_counts[t] += 1
    n_with_tools = sum(1 for r in results if (r.get("tools_attempted") or 0))
    tool_block = {
        t: {
            "n_succeeded": c,
            "share_of_samples": round(c / n_with_tools, 4)
            if n_with_tools else None,
        }
        for t, c in tool_used_counts.most_common()
    }

    # PocketQA → F1 correlation. Only consider rows where both are present.
    paired = [
        (_coerce_float(r.get("qa_total")), _coerce_float(r.get("f1")))
        for r in results
    ]
    paired = [(q, f) for q, f in paired if q is not None and f is not None]
    qa_xs = [q for q, _ in paired]
    f1_ys = [f for _, f in paired]
    pearson = _pearson(qa_xs, f1_ys)
    spearman = _spearman(qa_xs, f1_ys)

    return {
        "n_samples": n_total,
        "n_with_f1": len(f1s_present),
        "aggregate": {
            "mean_precision": _mean(p_present),
            "mean_recall": _mean(r_present),
            "mean_f1": _mean(f1s_present),
            "median_f1": _median(f1s_present),
            "std_f1": _std(f1s_present),
        },
        "by_category": by_cat_block,
        "by_quality_tier": by_q_block,
        "tool_success_rate": tool_block,
        "pocketqa_correlation": {
            "n": len(paired),
            "pearson_r": pearson,
            "spearman_rho": spearman,
        },
    }


def by_category_rows(results: list[dict]) -> list[dict]:
    """``by_category.csv``: one row per category."""
    by_cat: dict[str, list[float]] = defaultdict(list)
    by_cat_p: dict[str, list[float]] = defaultdict(list)
    by_cat_r: dict[str, list[float]] = defaultdict(list)
    for r in results:
        cat = r.get("category") or "unknown"
        f1 = _coerce_float(r.get("f1"))
        p = _coerce_float(r.get("precision"))
        rec = _coerce_float(r.get("recall"))
        if f1 is not None:
            by_cat[cat].append(f1)
        if p is not None:
            by_cat_p[cat].append(p)
        if rec is not None:
            by_cat_r[cat].append(rec)
    rows = []
    for cat in sorted(by_cat.keys() | by_cat_p.keys() | by_cat_r.keys()):
        rows.append({
            "category": cat,
            "n": len(by_cat[cat]),
            "mean_precision": _mean(by_cat_p[cat]),
            "mean_recall": _mean(by_cat_r[cat]),
            "mean_f1": _mean(by_cat[cat]),
            "median_f1": _median(by_cat[cat]),
        })
    return rows


def per_sample_rows(results: list[dict]) -> list[dict]:
    """``per_sample.csv``: results.jsonl flattened to CSV.

    ``tools_used`` is a list → joined with ``;`` so a single CSV cell
    can hold it. Other list/dict fields aren't currently emitted by
    batch_predict so we don't worry about them here.
    """
    rows = []
    for r in results:
        rows.append({
            "sample_id": r.get("sample_id"),
            "category": r.get("category"),
            "quality_tier": r.get("quality_tier"),
            "protein_length": r.get("protein_length"),
            "rna_length": r.get("rna_length"),
            "tools_attempted": r.get("tools_attempted"),
            "tools_succeeded": r.get("tools_succeeded"),
            "tools_used": ";".join(r.get("tools_used") or []),
            "fusion_status": r.get("fusion_status"),
            "binding_protein_predicted": r.get("binding_protein_predicted"),
            "binding_protein_gt": r.get("binding_protein_gt"),
            "precision": r.get("precision"),
            "recall": r.get("recall"),
            "f1": r.get("f1"),
            "qa_total": r.get("qa_total"),
            "qa_final": r.get("qa_final"),
            "iteration_action": r.get("iteration_action"),
            "total_iterations": r.get("total_iterations"),
            "termination_reason": r.get("termination_reason"),
            "runtime_seconds": r.get("runtime_seconds"),
            "total_tokens": r.get("total_tokens"),
        })
    return rows


def tool_analysis_rows(results: list[dict]) -> list[dict]:
    """``tool_analysis.csv``: per-tool success rate + mean F1 of samples
    where it succeeded."""
    n_total = len(results)
    by_tool_success: Counter[str] = Counter()
    by_tool_f1s: dict[str, list[float]] = defaultdict(list)
    for r in results:
        f1 = _coerce_float(r.get("f1"))
        for t in r.get("tools_used") or []:
            by_tool_success[t] += 1
            if f1 is not None:
                by_tool_f1s[t].append(f1)
    rows = []
    for tool in sorted(by_tool_success.keys()):
        n_ok = by_tool_success[tool]
        rows.append({
            "tool_id": tool,
            "n_samples_total": n_total,
            "n_samples_succeeded": n_ok,
            "success_rate": round(n_ok / n_total, 4) if n_total else None,
            "mean_f1_when_succeeded": _mean(by_tool_f1s[tool]),
        })
    return rows


def pocketqa_vs_f1_rows(results: list[dict]) -> list[dict]:
    """``pocketqa_vs_f1.csv``: scatter-plot input (qa_total, f1)."""
    rows = []
    for r in results:
        q = _coerce_float(r.get("qa_total"))
        f = _coerce_float(r.get("f1"))
        if q is None or f is None:
            continue
        rows.append({
            "sample_id": r.get("sample_id"),
            "category": r.get("category"),
            "qa_total": q,
            "f1": f,
        })
    return rows


# ---------- per-residue correlation (paper §4) -----------------------------

# The fusion method gets bucketed under this synthetic tool_id. Distinct
# from any real tool_id so the per-method breakdown can list it
# alongside the per-tool numbers without collision.
FUSION_METHOD_ID = "pocketagent_fusion"


def _read_jsonl_record(path: Path) -> Optional[dict]:
    """Read a single-record JSONL written by batch_predict.write_step_record.

    batch_predict writes one JSON object per file (one line). Returns
    ``None`` on missing / malformed file so the caller can skip the
    sample silently.
    """
    if not path.is_file():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            return json.loads(line)
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _load_sample_json(processed_dir: Path, sample_id: str) -> Optional[dict]:
    """Load step-1 sample JSON with the same case-insensitive fallback
    batch_predict.load_sample uses (splits.json sometimes carries a
    different case than the on-disk filename)."""
    samples_dir = processed_dir / "samples"
    for p in (samples_dir / f"{sample_id}.json",
              processed_dir / f"{sample_id}.json"):
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
    if samples_dir.is_dir():
        target = sample_id.lower()
        for f in samples_dir.iterdir():
            if f.suffix == ".json" and f.stem.lower() == target:
                try:
                    return json.loads(f.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return None
    return None


def _per_residue_to_int_dict(
    per_res: Optional[dict],
) -> dict[int, float]:
    """Coerce a per_residue_confidence map to ``{int: float}``.

    JSON serialisation forces dict keys to strings (step5 explicitly
    stringifies them in build_output_record), so we accept either str
    or int keys. Non-coercible entries are silently dropped — we'd
    rather lose a few residues than crash the whole evaluation.
    """
    if not per_res:
        return {}
    out: dict[int, float] = {}
    for k, v in per_res.items():
        try:
            ki = int(k)
            vf = float(v)
        except (TypeError, ValueError):
            continue
        out[ki] = vf
    return out


def build_pred_gt_vectors(
    per_residue: dict[int, float],
    binding_residues: Optional[list[int]],
    binding_gt: list[int],
    protein_length: int,
    *,
    is_cat_a: bool,
) -> tuple[list[float], list[float]]:
    """Build the prediction + ground-truth vectors for one (sample, tool).

    Both vectors are length ``protein_length``, indexed 0..len-1 for
    1-based residues 1..len.

    Cat A tools store pLDDT in ``per_residue_confidence`` on a 0-100
    scale; we divide by 100 so every method shares the same [0, 1]
    range. Values still outside [0, 1] after that are clamped.

    Falls back to a binary indicator from ``binding_residues`` when
    ``per_residue`` is empty — important for tools that emitted only a
    binding list (e.g. fusion before per-residue probabilities were
    added).
    """
    pred = [0.0] * protein_length
    gt = [0.0] * protein_length
    gt_set = set(binding_gt or [])
    binding_set = set(binding_residues or [])

    for i in range(1, protein_length + 1):
        if i in gt_set:
            gt[i - 1] = 1.0
        if per_residue:
            v = per_residue.get(i)
            if v is None:
                continue
            if is_cat_a and v > 1.0:
                v = v / 100.0
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            pred[i - 1] = v
        elif binding_set and i in binding_set:
            # Binary fallback when the tool didn't emit per-residue scores.
            pred[i - 1] = 1.0
    return pred, gt


def _has_variance(xs: list[float], eps: float = 1e-12) -> bool:
    """True if ``xs`` has non-trivial variance (Pearson / Spearman defined)."""
    if not xs:
        return False
    mn = min(xs)
    mx = max(xs)
    return (mx - mn) > eps


def compute_per_residue_correlation(
    per_residue: dict[int, float],
    binding_residues: Optional[list[int]],
    binding_gt: list[int],
    protein_length: int,
    *,
    is_cat_a: bool,
) -> Optional[dict]:
    """Pearson / Spearman / R² for one (sample, method).

    Returns ``None`` when the correlation is undefined: GT all-0
    (sample has no binding residues at all — shouldn't happen for
    real positive samples but we guard), GT all-1 (every residue
    binds — pathological), or the prediction has zero variance
    (the tool gave the same score everywhere).
    """
    pred, gt = build_pred_gt_vectors(
        per_residue, binding_residues, binding_gt, protein_length,
        is_cat_a=is_cat_a,
    )
    if not _has_variance(gt) or not _has_variance(pred):
        return None
    pr = _pearson(pred, gt)
    if pr is None:
        return None
    sr = _spearman(pred, gt)
    return {
        "pearson_r": pr,
        "spearman_r": sr,
        "r_squared": round(pr * pr, 4),
    }


def collect_per_residue_correlations(
    results: list[dict],
    *,
    step4_dir: Path,
    step5_dir: Path,
    processed_dir: Path,
) -> dict[str, list[dict]]:
    """For every result row, compute per-residue correlations for each
    tool (from step-4 JSONL) and the fused method (from step-5 JSONL).

    Returns ``{method_id: [{sample_id, pearson_r, spearman_r,
    r_squared}, ...]}`` — one entry per (method, sample) pair where
    the correlation was definable.
    """
    bucket: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        sid = r.get("sample_id")
        if not sid:
            continue
        sample = _load_sample_json(processed_dir, sid)
        if sample is None:
            continue
        protein_length = ((sample.get("protein") or {}).get("length") or 0)
        if not protein_length:
            continue
        binding_gt = (
            (sample.get("interaction") or {}).get("binding_protein_residues")
            or []
        )
        if not binding_gt:
            # No GT — every metric below would be undefined.
            continue

        # ---- per-tool correlations (step-4) -----------------------------
        s4 = _read_jsonl_record(step4_dir / f"{sid}.jsonl")
        for pred in (s4 or {}).get("predictions") or []:
            if not pred.get("success"):
                continue
            tool_id = pred.get("tool_id")
            category = pred.get("category") or ""
            binding = pred.get("binding_protein_residues") or []

            # Score-source selection (paper §4 fix):
            #   - Cat A tools (boltz2, chai1) prefer per_residue_pae_score
            #     when present. PAE-derived score is already in [0, 1]
            #     and covers ALL protein residues, not just the contact
            #     list — that's what makes Pearson meaningful instead
            #     of near-zero with the pLDDT-via-binding-list path.
            #   - Fall back to per_residue_confidence (pLDDT for Cat A,
            #     ligandability / probability for Cat B/C) so older
            #     batch_predict outputs without the new field still
            #     produce numbers.
            pae_scores = _per_residue_to_int_dict(
                pred.get("per_residue_pae_score"),
            )
            if category == "A" and pae_scores:
                per_res = pae_scores
                # PAE-derived scores are already in [0, 1] — no /100
                # scaling. ``is_cat_a=False`` here is the right flag,
                # despite the tool itself being Cat A.
                is_cat_a_path = False
            else:
                per_res = _per_residue_to_int_dict(
                    pred.get("per_residue_confidence"),
                )
                is_cat_a_path = (category == "A")

            corr = compute_per_residue_correlation(
                per_res, binding, binding_gt, protein_length,
                is_cat_a=is_cat_a_path,
            )
            if corr is None:
                continue
            bucket[tool_id].append({"sample_id": sid, **corr})

        # ---- fused method (step-5) --------------------------------------
        s5 = _read_jsonl_record(step5_dir / f"{sid}.jsonl")
        if s5 is not None:
            per_res = _per_residue_to_int_dict(s5.get("per_residue_probability"))
            binding = s5.get("binding_protein_residues") or []
            corr = compute_per_residue_correlation(
                per_res, binding, binding_gt, protein_length,
                # Fusion outputs already in [0, 1] — never the pLDDT path.
                is_cat_a=False,
            )
            if corr is not None:
                bucket[FUSION_METHOD_ID].append({"sample_id": sid, **corr})
    return bucket


def aggregate_per_residue(
    bucket: dict[str, list[dict]],
) -> dict[str, dict]:
    """Reduce per-(method, sample) rows into mean / std / median per method.

    Ordered by method_id so the output JSON / CSV is reproducible.
    """
    out: dict[str, dict] = {}
    for method in sorted(bucket.keys()):
        rows = bucket[method]
        prs = [row["pearson_r"] for row in rows if row["pearson_r"] is not None]
        srs = [row["spearman_r"] for row in rows if row["spearman_r"] is not None]
        r2s = [row["r_squared"] for row in rows if row["r_squared"] is not None]
        out[method] = {
            "n_samples": len(rows),
            "pearson_r": {
                "mean": _mean(prs), "std": _std(prs), "median": _median(prs),
            },
            "spearman_r": {
                "mean": _mean(srs), "std": _std(srs), "median": _median(srs),
            },
            "r_squared": {
                "mean": _mean(r2s), "std": _std(r2s), "median": _median(r2s),
            },
        }
    return out


def per_residue_csv_rows(
    aggregate: dict[str, dict],
) -> list[dict]:
    """Flatten the aggregate dict into one row per method (paper-table shape)."""
    rows: list[dict] = []
    for method, agg in aggregate.items():
        rows.append({
            "method": method,
            "n_samples": agg["n_samples"],
            "pearson_r_mean": agg["pearson_r"]["mean"],
            "pearson_r_std": agg["pearson_r"]["std"],
            "pearson_r_median": agg["pearson_r"]["median"],
            "spearman_r_mean": agg["spearman_r"]["mean"],
            "spearman_r_std": agg["spearman_r"]["std"],
            "spearman_r_median": agg["spearman_r"]["median"],
            "r2_mean": agg["r_squared"]["mean"],
            "r2_std": agg["r_squared"]["std"],
            "r2_median": agg["r_squared"]["median"],
        })
    return rows


# ---------- main -----------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True,
                   help="batch_predict's summary/results.jsonl")
    p.add_argument("--processed-dir", type=Path, default=None,
                   help="step 1 output dir (containing samples/<id>.json). "
                        "Required for the per-residue correlation outputs; "
                        "ignored otherwise.")
    p.add_argument("--step4-dir", type=Path, default=None,
                   help="batch_predict's step4/ dir (one JSONL per sample). "
                        "Required for per-tool per-residue correlations.")
    p.add_argument("--step5-dir", type=Path, default=None,
                   help="batch_predict's step5/ dir (one JSONL per sample). "
                        "Required for the fused per-residue correlation.")
    p.add_argument("--output", type=Path, required=True,
                   help="output dir; emits the 5 base files + 2 extra per-"
                        "residue files when --step4-dir/--step5-dir/"
                        "--processed-dir are all set.")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    try:
        results = load_results(args.results)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if not results:
        print("ERROR: results.jsonl is empty — nothing to evaluate",
              file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    # 1) aggregate.json
    write_json(args.output / "aggregate.json", aggregate(results))

    # 2) by_category.csv
    cat_rows = by_category_rows(results)
    write_csv(args.output / "by_category.csv", cat_rows,
              columns=["category", "n", "mean_precision", "mean_recall",
                       "mean_f1", "median_f1"])

    # 3) per_sample.csv
    sample_rows = per_sample_rows(results)
    write_csv(args.output / "per_sample.csv", sample_rows,
              columns=[
                  "sample_id", "category", "quality_tier",
                  "protein_length", "rna_length",
                  "tools_attempted", "tools_succeeded", "tools_used",
                  "fusion_status",
                  "binding_protein_predicted", "binding_protein_gt",
                  "precision", "recall", "f1",
                  "qa_total", "qa_final",
                  "iteration_action", "total_iterations", "termination_reason",
                  "runtime_seconds", "total_tokens",
              ])

    # 4) tool_analysis.csv
    tool_rows = tool_analysis_rows(results)
    write_csv(args.output / "tool_analysis.csv", tool_rows,
              columns=["tool_id", "n_samples_total",
                       "n_samples_succeeded", "success_rate",
                       "mean_f1_when_succeeded"])

    # 5) pocketqa_vs_f1.csv
    qa_rows = pocketqa_vs_f1_rows(results)
    write_csv(args.output / "pocketqa_vs_f1.csv", qa_rows,
              columns=["sample_id", "category", "qa_total", "f1"])

    # 6) per-residue correlation (optional — needs step4/step5 JSONLs)
    n_per_residue_methods = 0
    n_per_residue_pairs = 0
    if (args.step4_dir is not None and args.step5_dir is not None
            and args.processed_dir is not None):
        bucket = collect_per_residue_correlations(
            results,
            step4_dir=args.step4_dir,
            step5_dir=args.step5_dir,
            processed_dir=args.processed_dir,
        )
        agg = aggregate_per_residue(bucket)
        write_json(args.output / "per_residue_correlation.json",
                   {"methods": agg})
        pr_rows = per_residue_csv_rows(agg)
        write_csv(args.output / "per_residue_correlation.csv", pr_rows,
                  columns=[
                      "method", "n_samples",
                      "pearson_r_mean", "pearson_r_std", "pearson_r_median",
                      "spearman_r_mean", "spearman_r_std", "spearman_r_median",
                      "r2_mean", "r2_std", "r2_median",
                  ])
        n_per_residue_methods = len(agg)
        n_per_residue_pairs = sum(v["n_samples"] for v in agg.values())

    n_files = 5 + (2 if n_per_residue_methods else 0)
    print(f"wrote {n_files} files under {args.output}")
    print(f"  aggregate.json       (n_samples={len(results)})")
    print(f"  by_category.csv      ({len(cat_rows)} categories)")
    print(f"  per_sample.csv       ({len(sample_rows)} rows)")
    print(f"  tool_analysis.csv    ({len(tool_rows)} tools)")
    print(f"  pocketqa_vs_f1.csv   ({len(qa_rows)} rows)")
    if n_per_residue_methods:
        print(f"  per_residue_correlation.json ({n_per_residue_methods} methods, "
              f"{n_per_residue_pairs} sample-method pairs)")
        print(f"  per_residue_correlation.csv  (same)")
    elif (args.step4_dir or args.step5_dir or args.processed_dir):
        print("  (per-residue correlation skipped: pass --step4-dir, "
              "--step5-dir AND --processed-dir to compute it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
