"""Evaluate the enriched fusion model — per-residue correlation tables.

Emits, under ``--output``:

  - ``per_residue_correlation.csv`` / ``.json``
      One row per *method*. With both ablation dirs passed the 4-row
      table is:
        enriched_fusion             (--model-dir, e.g. 61-d +context)
        enriched_fusion_no_context  (--no-context-model-dir, 48-d)
        enriched_fusion_base        (--base-model-dir, 28-d)
        noisy_or_baseline           ("old fusion")
      Same column shape as ``scripts/evaluate.py`` so paper tables
      consume it unchanged.
  - ``feature_importance.csv``
      The trained model's full ranked feature importances (which of the
      48 columns — especially the cross-tool terms — actually mattered).
  - ``per_sample_correlation.csv``
      Per-(sample, method) Pearson/Spearman/R² (scatter / debugging).

The noisy-OR baseline mirrors ``src/step5_fusion/noisy_or.py``'s gating
+ per-category normalisation, computed directly off the step4 dicts so
no step5 rerun is needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from step5_fusion.data_collector import (  # noqa: E402
    load_sample_json, per_residue_to_int_dict, read_jsonl_record,
)
from step5_fusion.enriched_fusion import (  # noqa: E402
    TOOL_ORDER, EnrichedFusion, _pearson, _spearman,
)


# ---- aggregation helpers -------------------------------------------------


def _round(v: Optional[float], n: int = 4) -> Optional[float]:
    return None if v is None else round(v, n)


def _mean(vs): return round(statistics.fmean(vs), 4) if vs else None
def _median(vs): return round(statistics.median(vs), 4) if vs else None
def _std(vs):
    if len(vs) < 2:
        return 0.0 if vs else None
    return round(statistics.pstdev(vs), 4)


def _corr_row(pred: list[float], gt: list[float]) -> Optional[dict]:
    """Pearson/Spearman/R² for one (sample, method); None if undefined."""
    if len(pred) < 2:
        return None
    if (max(pred) - min(pred)) <= 1e-12 or (max(gt) - min(gt)) <= 1e-12:
        return None
    pr = _pearson(pred, gt)
    if pr is None:
        return None
    return {"pearson_r": round(pr, 4),
            "spearman_r": _round(_spearman(pred, gt)),
            "r_squared": round(pr * pr, 4)}


# ---- noisy-OR baseline (mirrors src/step5_fusion/noisy_or.py) ------------

_CAT = {"boltz2": "A", "chai1": "A", "rosettafold2na": "A",
        "equipnas": "C", "p2rank": "B", "fpocket": "B"}


def _noisy_or_baseline(step4: dict, length: int) -> dict[int, float]:
    """Equal-weight (c_k=1) noisy-OR over the 5-tool gated per-residue
    confidence — the "old fusion" reference."""
    pairs: list[dict[int, float]] = []
    for p in (step4.get("predictions") or []):
        tid = p.get("tool_id") or ""
        if tid not in TOOL_ORDER or not p.get("success"):
            continue
        cat = _CAT.get(tid, "")
        binding = p.get("binding_protein_residues") or []
        if not binding:
            continue
        prc = per_residue_to_int_dict(p.get("per_residue_confidence"))
        probs: dict[int, float] = {}
        for raw in binding:
            try:
                i = int(raw)
            except (TypeError, ValueError):
                continue
            if i in prc:
                v = prc[i] / 100.0 if cat == "A" else prc[i]
                probs[i] = min(1.0, max(0.0, v))
            else:
                probs[i] = 1.0
        if probs:
            pairs.append(probs)
    fused: dict[int, float] = {}
    for i in range(1, length + 1):
        prod = 1.0
        for probs in pairs:
            prod *= (1.0 - probs.get(i, 0.0))
        fused[i] = 1.0 - prod
    return fused


# ---- core ---------------------------------------------------------------


def _load_sample_ids(path: Optional[Path]) -> Optional[list[str]]:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"sample-list not found: {path}")
    return [ln.strip() for ln in path.read_text(encoding="utf-8")
            .splitlines() if ln.strip() and not ln.startswith("#")]


def evaluate(
    *,
    step4_dir: Path,
    processed_dir: Path,
    model: EnrichedFusion,
    no_context_model: Optional[EnrichedFusion],
    base_model: Optional[EnrichedFusion],
    sample_ids: Optional[list[str]],
) -> tuple[dict[str, list[dict]], list[dict]]:
    if sample_ids is None:
        sample_ids = sorted(p.stem for p in step4_dir.glob("*.jsonl"))
    bucket: dict[str, list[dict]] = defaultdict(list)
    per_sample: list[dict] = []

    for sid in sample_ids:
        s4 = read_jsonl_record(step4_dir / f"{sid}.jsonl")
        if s4 is None:
            continue
        sample = load_sample_json(processed_dir, sid)
        if sample is None:
            continue
        prot = sample.get("protein") or {}
        length = prot.get("length") or len(prot.get("sequence") or "")
        gt_set = set((sample.get("interaction") or {})
                     .get("binding_protein_residues") or [])
        if not length or not gt_set:
            continue
        residue_ids = list(range(1, int(length) + 1))
        gt = [1.0 if r in gt_set else 0.0 for r in residue_ids]
        preds = s4.get("predictions") or []

        methods: dict[str, list[float]] = {}
        m = model.predict_sample(preds, prot.get("sequence") or "", length)
        methods["enriched_fusion"] = [m.get(r, 0.0) for r in residue_ids]
        if no_context_model is not None:
            mn = no_context_model.predict_sample(
                preds, prot.get("sequence") or "", length)
            methods["enriched_fusion_no_context"] = [
                mn.get(r, 0.0) for r in residue_ids]
        if base_model is not None:
            mb = base_model.predict_sample(
                preds, prot.get("sequence") or "", length)
            methods["enriched_fusion_base"] = [
                mb.get(r, 0.0) for r in residue_ids]
        no = _noisy_or_baseline(s4, int(length))
        methods["noisy_or_baseline"] = [no.get(r, 0.0) for r in residue_ids]

        for name, pred in methods.items():
            corr = _corr_row(pred, gt)
            if corr is None:
                continue
            bucket[name].append({"sample_id": sid, **corr})
            per_sample.append({"sample_id": sid, "method": name, **corr})
    return bucket, per_sample


def _aggregate(bucket: dict[str, list[dict]]) -> list[dict]:
    rows: list[dict] = []
    order = ["enriched_fusion", "enriched_fusion_no_context",
             "enriched_fusion_base", "noisy_or_baseline"]
    for method in sorted(bucket, key=lambda k: (order.index(k)
                         if k in order else 99, k)):
        rs = bucket[method]
        prs = [r["pearson_r"] for r in rs if r["pearson_r"] is not None]
        srs = [r["spearman_r"] for r in rs if r["spearman_r"] is not None]
        r2s = [r["r_squared"] for r in rs if r["r_squared"] is not None]
        rows.append({
            "method": method, "n_samples": len(rs),
            "pearson_r_mean": _mean(prs), "pearson_r_std": _std(prs),
            "pearson_r_median": _median(prs),
            "spearman_r_mean": _mean(srs), "spearman_r_std": _std(srs),
            "spearman_r_median": _median(srs),
            "r2_mean": _mean(r2s), "r2_std": _std(r2s),
            "r2_median": _median(r2s),
        })
    return rows


def _write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step4-dir", type=Path, required=True)
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True,
                   help="enriched fusion bundle (train_enriched_fusion).")
    p.add_argument("--no-context-model-dir", type=Path, default=None,
                   help="optional 48-d full-no-context bundle for the "
                        "context-ablation row (enriched_fusion_no_"
                        "context). Typically the prior "
                        "data/enriched_fusion_model/.")
    p.add_argument("--base-model-dir", type=Path, default=None,
                   help="optional 28-d base-feature bundle for the "
                        "cross-term ablation row.")
    p.add_argument("--sample-list", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not args.step4_dir.is_dir():
        print(f"ERROR: --step4-dir not a directory: {args.step4_dir}",
              file=sys.stderr)
        return 1
    try:
        sample_ids = _load_sample_ids(args.sample_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    try:
        model = EnrichedFusion.load(args.model_dir)
    except (OSError, ValueError, ImportError, json.JSONDecodeError) as e:
        print(f"ERROR: could not load model bundle: {e}", file=sys.stderr)
        return 1
    no_context_model = None
    if args.no_context_model_dir is not None:
        try:
            no_context_model = EnrichedFusion.load(
                args.no_context_model_dir)
        except (OSError, ValueError, ImportError,
                json.JSONDecodeError) as e:
            print(f"ERROR: could not load --no-context-model-dir: {e}",
                  file=sys.stderr)
            return 1
    base_model = None
    if args.base_model_dir is not None:
        try:
            base_model = EnrichedFusion.load(args.base_model_dir)
        except (OSError, ValueError, ImportError,
                json.JSONDecodeError) as e:
            print(f"ERROR: could not load --base-model-dir: {e}",
                  file=sys.stderr)
            return 1

    bucket, per_sample = evaluate(
        step4_dir=args.step4_dir, processed_dir=args.processed_dir,
        model=model, no_context_model=no_context_model,
        base_model=base_model, sample_ids=sample_ids)

    agg = _aggregate(bucket)
    args.output.mkdir(parents=True, exist_ok=True)

    (args.output / "per_residue_correlation.json").write_text(
        json.dumps({"methods": agg}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    _write_csv(args.output / "per_residue_correlation.csv", agg, [
        "method", "n_samples",
        "pearson_r_mean", "pearson_r_std", "pearson_r_median",
        "spearman_r_mean", "spearman_r_std", "spearman_r_median",
        "r2_mean", "r2_std", "r2_median"])

    fi = (model.training_report or {}).get("feature_importances") or []
    _write_csv(args.output / "feature_importance.csv", [
        {"rank": i + 1, "feature": e["feature"],
         "importance": round(e["importance"], 6)}
        for i, e in enumerate(fi)
    ], ["rank", "feature", "importance"])

    _write_csv(args.output / "per_sample_correlation.csv", per_sample,
               ["sample_id", "method", "pearson_r", "spearman_r",
                "r_squared"])

    print(f"wrote evaluation to {args.output}")
    print(f"  per_residue_correlation.csv ({len(agg)} methods)")
    print(f"  feature_importance.csv      ({len(fi)} features)")
    print(f"  per_sample_correlation.csv  ({len(per_sample)} rows)")
    print()
    hdr = (f"{'method':24s} {'n':>4s} {'PearsonR':>9s} "
           f"{'SpearmanR':>10s} {'R2':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in agg:
        print(f"{r['method']:24s} {r['n_samples']:>4d} "
              f"{str(r['pearson_r_mean']):>9s} "
              f"{str(r['spearman_r_mean']):>10s} "
              f"{str(r['r2_mean']):>7s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
