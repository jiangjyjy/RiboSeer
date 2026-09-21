"""Per-tool per-residue correlation, straight from step4 JSONL.

For every tool that appears in any step4 record, build its per-residue
prediction vector (over ``protein.resolved_residues``) and the binary
GT vector (residue ∈ ``interaction.binding_protein_residues``), then
report mean / std / median of Pearson R / Spearman R / R² across
samples — the same table shape ``scripts/evaluate.py`` and
``scripts/evaluate_enriched_fusion.py`` emit, so paper tables consume
it unchanged.

Tool ids are discovered from the data (not hard-coded), so a newly
deployed tool — e.g. ``rosettafold2na`` — shows up automatically.

Usage
-----
::

    python scripts/tables/table04_main_results.py \\
        --step4-dir     data/batch_train_v4/step4/ \\
        --processed-dir data/processed_filtered \\
        --sample-list   data/processed_filtered/splits/train_200.txt \\
        --output        data/batch_train_v4/evaluation_v5/per_tool_correlation.csv

Optional
--------
``--enriched-model-dir DIR``  also score the trained enriched-fusion
bundle and append it as the last CSV row (method ``enriched_fusion``).
``--min-gt N``  skip samples whose GT binding-residue count is ``<= N``
(default 0 → no extra filtering; all-0 GT is always skipped anyway).

Per-residue score source (per tool, per residue)
------------------------------------------------
``per_residue_pae_score`` (the Cat-A distance/PAE interface score) when
present and non-empty, else ``per_residue_confidence``. Pearson /
Spearman are scale-invariant so no per-category /100 rescaling is
needed (R² is reported as Pearson²). Score dict keys may be ``str`` or
``int``; residues a tool didn't score contribute 0.

A (sample, method) pair is skipped when the correlation is undefined:
GT has no variance (all-0 / all-1), the prediction is constant, or the
tool's record is ``success=False``.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from step5_fusion.data_collector import (  # noqa: E402
    load_sample_json, per_residue_to_int_dict,
)
from step5_fusion.enriched_fusion import (  # noqa: E402
    EnrichedFusion, _pearson, _spearman,
)


# ---- aggregation helpers (byte-identical to evaluate_enriched_fusion) ----


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
    # GT all-0/all-1 or constant prediction ⇒ correlation undefined.
    if (max(pred) - min(pred)) <= 1e-8 or (max(gt) - min(gt)) <= 1e-8:
        return None
    pr = _pearson(pred, gt)
    if pr is None:
        return None
    return {"pearson_r": round(pr, 4),
            "spearman_r": _round(_spearman(pred, gt)),
            "r_squared": round(pr * pr, 4)}


# ---- IO ------------------------------------------------------------------


def _read_last_jsonl_record(path: Path) -> Optional[dict]:
    """Last parseable JSON object in a JSONL file (spec: take the last
    line). batch_predict writes one record per file, but a re-run can
    append; the last line is the freshest."""
    if not path.is_file():
        return None
    rec = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
    except OSError:
        return None
    return rec


def _load_sample_ids(path: Optional[Path]) -> Optional[list[str]]:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"sample-list not found: {path}")
    return [ln.strip() for ln in path.read_text(encoding="utf-8")
            .splitlines() if ln.strip() and not ln.startswith("#")]


# ---- per-tool vector building --------------------------------------------


def _tool_scores(pred: dict) -> dict[int, float]:
    """Per-residue score map for one tool prediction: prefer
    per_residue_pae_score, fall back to per_residue_confidence.
    Keys coerced to int (JSON stringifies them)."""
    pae = per_residue_to_int_dict(pred.get("per_residue_pae_score"))
    if pae:
        return pae
    return per_residue_to_int_dict(pred.get("per_residue_confidence"))


def evaluate(
    *,
    step4_dir: Path,
    processed_dir: Path,
    sample_ids: Optional[list[str]],
    min_gt: int,
    enriched_model: Optional[EnrichedFusion],
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Returns ({method: [per-sample corr dicts]}, per_sample_rows)."""
    if sample_ids is None:
        sample_ids = sorted(p.stem for p in step4_dir.glob("*.jsonl"))

    bucket: dict[str, list[dict]] = defaultdict(list)
    per_sample: list[dict] = []

    for sid in sample_ids:
        s4 = _read_last_jsonl_record(step4_dir / f"{sid}.jsonl")
        if s4 is None:
            continue
        sample = load_sample_json(processed_dir, sid)
        if sample is None:
            continue
        prot = sample.get("protein") or {}
        # Residue order = resolved_residues; fall back to 1..length.
        residues = list(prot.get("resolved_residues") or [])
        if not residues:
            length = prot.get("length") or len(prot.get("sequence") or "")
            residues = list(range(1, int(length) + 1))
        if len(residues) < 2:
            continue
        gt_set = set((sample.get("interaction") or {})
                     .get("binding_protein_residues") or [])
        if len(gt_set) <= min_gt:
            # default min_gt=0 → only drops empty-GT samples (which
            # would be skipped by the all-0 variance guard anyway).
            continue
        gt = [1.0 if r in gt_set else 0.0 for r in residues]

        preds = s4.get("predictions") or []
        for p in preds:
            tid = p.get("tool_id")
            if not tid or not p.get("success"):
                continue
            scores = _tool_scores(p)
            if not scores:
                continue
            vec = [float(scores.get(r, 0.0)) for r in residues]
            corr = _corr_row(vec, gt)
            if corr is None:
                continue
            bucket[tid].append({"sample_id": sid, **corr})
            per_sample.append({"sample_id": sid, "method": tid, **corr})

        if enriched_model is not None:
            length = prot.get("length") or len(prot.get("sequence") or "")
            probs = enriched_model.predict_sample(
                preds, prot.get("sequence") or "", int(length or 0))
            vec = [float(probs.get(r, 0.0)) for r in residues]
            corr = _corr_row(vec, gt)
            if corr is not None:
                bucket["enriched_fusion"].append(
                    {"sample_id": sid, **corr})
                per_sample.append(
                    {"sample_id": sid, "method": "enriched_fusion",
                     **corr})
    return bucket, per_sample


def _aggregate(bucket: dict[str, list[dict]]) -> list[dict]:
    """One row per method. Tools sorted by n_samples desc;
    ``enriched_fusion`` (if present) always appended last."""
    rows: list[dict] = []
    for method, rs in bucket.items():
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
    enriched = [r for r in rows if r["method"] == "enriched_fusion"]
    tools = [r for r in rows if r["method"] != "enriched_fusion"]
    tools.sort(key=lambda r: (-r["n_samples"], r["method"]))
    return tools + enriched


_COLUMNS = [
    "method", "n_samples",
    "pearson_r_mean", "pearson_r_std", "pearson_r_median",
    "spearman_r_mean", "spearman_r_std", "spearman_r_median",
    "r2_mean", "r2_std", "r2_median",
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in _COLUMNS})


def _print_table(rows: list[dict]) -> None:
    hdr = (f"{'method':22s} {'n':>4s} {'PearsonR':>9s} "
           f"{'SpearmanR':>10s} {'R2':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['method']:22s} {r['n_samples']:>4d} "
              f"{str(r['pearson_r_mean']):>9s} "
              f"{str(r['spearman_r_mean']):>10s} "
              f"{str(r['r2_mean']):>8s}")


# ---- main ----------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step4-dir", type=Path, required=True,
                   help="step4 JSONL dir (one .jsonl per sample; the "
                        "LAST record line is used).")
    p.add_argument("--processed-dir", type=Path, required=True,
                   help="GT dir (samples/<id>.json with "
                        "interaction.binding_protein_residues + "
                        "protein.resolved_residues).")
    p.add_argument("--sample-list", type=Path, default=None,
                   help="one sample_id per line; default = every "
                        "*.jsonl under --step4-dir.")
    p.add_argument("--output", type=Path, required=True,
                   help="output CSV path.")
    p.add_argument("--enriched-model-dir", type=Path, default=None,
                   help="optional EnrichedFusion bundle — appends an "
                        "'enriched_fusion' row last.")
    p.add_argument("--min-gt", type=int, default=0,
                   help="skip samples with GT binding-residue count "
                        "<= N (default 0).")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not args.step4_dir.is_dir():
        print(f"ERROR: --step4-dir not a directory: {args.step4_dir}",
              file=sys.stderr)
        return 1
    if not (args.processed_dir / "samples").is_dir() \
            and not args.processed_dir.is_dir():
        print(f"ERROR: --processed-dir not found: {args.processed_dir}",
              file=sys.stderr)
        return 1
    try:
        sample_ids = _load_sample_ids(args.sample_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    enriched_model = None
    if args.enriched_model_dir is not None:
        try:
            enriched_model = EnrichedFusion.load(args.enriched_model_dir)
        except (OSError, ValueError, ImportError,
                json.JSONDecodeError) as e:
            print(f"ERROR: could not load --enriched-model-dir: {e}",
                  file=sys.stderr)
            return 1

    bucket, per_sample = evaluate(
        step4_dir=args.step4_dir,
        processed_dir=args.processed_dir,
        sample_ids=sample_ids,
        min_gt=args.min_gt,
        enriched_model=enriched_model,
    )
    rows = _aggregate(bucket)

    _write_csv(args.output, rows)
    # Companion per-sample CSV next to the aggregate (handy for scatter
    # / debugging; mirrors evaluate_enriched_fusion's extra output).
    ps_path = args.output.with_name(
        args.output.stem + "_per_sample.csv")
    ps_path.parent.mkdir(parents=True, exist_ok=True)
    with ps_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sample_id", "method", "pearson_r", "spearman_r",
            "r_squared"])
        w.writeheader()
        for r in per_sample:
            w.writerow(r)

    print(f"wrote {args.output}  ({len(rows)} methods)")
    print(f"wrote {ps_path}  ({len(per_sample)} sample-method rows)")
    print()
    _print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
