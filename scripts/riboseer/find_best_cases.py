#!/usr/bin/env python3
"""Pick 6-8 standout RiboSeer cases and export everything a visualiser
needs (structure file + full metric summary + per-residue table).

Step 1 — rank
    For every test sample compute the per-residue Pearson R
    (``per_sample_corr`` over the resolved-residue subset, the paper's
    Table-4 evaluation domain) of RiboSeer (full-system prediction) and
    of every baseline tool found in the step4 record (all 15 — the 6
    core TOOL_ORDER tools plus the merged external ones), then
    ``delta = RiboSeer R - best baseline R``.

Step 2 — select (diverse top-k)
    Sort by delta desc, keep only the highest-delta chain-pair per PDB
    (no two cases from the same structure), take the top ``--top-k``
    (default 8).

Step 3 — export (per selected sample, under ``--export-dir/<sid>/``)
    (a) the raw ``.pdb``/``.cif`` from ``--raw-dir`` (actual format
        reported);
    (b) ``<sid>_metrics.json`` — RiboSeer's full metric pack
        (Pearson/Spearman/R², AUROC, AUPRC, top-k P/R/F1, MCC, DCC) +
        best baseline + all-15-baseline Pearson + delta;
    (c) ``<sid>_residues.csv`` — one row per residue: id, resname,
        gt_binding, riboseer_prob, and every tool's per-residue score
        (missing → NaN).

Metric reuse (so numbers match the paper tables):
    * Pearson/Spearman/R² — ``ablation_fusion_method.per_sample_corr``
    * top-k P/R/F1 + optimal-F1 MCC —
      ``table20_topk.compute_topk_metrics`` /
      ``compute_optimal_mcc``
    * DCC — ``table05_pocket_geometry.evaluate_sample_method`` (top-k
      cluster centroid distance)
    * AUROC/AUPRC — sklearn.
    Per-residue tool scores + Pearson vectors reuse
    ``extract_case_studies`` (``_tool_scores`` / ``_read_last_jsonl_record``).

Usage
-----
::

    python scripts/riboseer/find_best_cases.py \\
        --data-dir        data/processed_quality \\
        --step4-dir       data/batch_test_v7/step4 \\
        --predictions-dir data/batch_test_v7/fullsystem_predictions \\
        --split-file      data/processed_quality/splits_tmscore_035/test.txt \\
        --raw-dir         data/raw \\
        --export-dir      exports/case_studies \\
        --top-k           8

``--raw-dir`` / ``--export-dir`` are optional: without them the ranking
table is still printed (no files written).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step5_fusion.data_collector import load_sample_json  # noqa: E402
from step5_fusion.prediction_io import load_predictions_dir  # noqa: E402
from scripts.riboseer.extract_case_studies import (  # noqa: E402
    _read_last_jsonl_record, _tool_scores, _load_sample_ids,
)
from scripts.tables.table05_pocket_geometry import (  # noqa: E402
    _resolve_structure_path, extract_chain_coords, evaluate_sample_method,
)
from scripts.riboseer.ablation_fusion_method import per_sample_corr  # noqa: E402
# NB: table20_topk + sklearn are imported lazily inside
# riboseer_metric_pack — they pull in sklearn, which is server-only, so
# keeping them out of module scope lets the ranking/export path import
# and run on the local dev box without sklearn installed.

# All 15 deployed baselines in display order (matches the case-study
# brief). Best-baseline + the all_baselines block iterate this list; any
# tool absent from a sample's step4 record is simply skipped / NaN.
ALL_TOOLS: list[str] = [
    "boltz2", "equipnas", "chai1", "p2rank", "rosettafold2na",
    "hdock", "nucleicnet", "graphbind", "rnabindrplus", "fpocket",
    "deeppocket", "haddock3", "rfaa", "alphafold3", "bindup",
]

# One-letter -> three-letter for the residues CSV resname column.
ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "U": "SEC", "O": "PYL", "B": "ASX", "Z": "GLX", "X": "UNK",
}


def _resname(seq: str, res_id: int) -> str:
    """Three-letter name for a 1-based residue id from the protein
    sequence; UNK when out of range / unknown one-letter code."""
    i = res_id - 1
    if 0 <= i < len(seq):
        return ONE_TO_THREE.get(seq[i].upper(), "UNK")
    return "UNK"


# ---- per-sample collection (ranking pass; no structure read) ------------


def collect_sample(sid: str, *, data_dir: Path, step4_dir: Path,
                   fs_preds: dict[str, dict[int, float]]) -> Optional[dict]:
    """Build a sample record: RiboSeer + per-tool Pearson R, best
    baseline, delta, plus everything pass 2 needs (vectors, GT, meta).
    None if the sample can't be scored."""
    s4 = _read_last_jsonl_record(step4_dir / f"{sid}.jsonl")
    if s4 is None:
        return None
    sample = load_sample_json(data_dir, sid)
    if sample is None:
        return None
    prot = sample.get("protein") or {}
    rna = sample.get("rna") or sample.get("ligand_rna") or {}
    inter = sample.get("interaction") or {}

    gt_residues = [int(r) for r in
                   (inter.get("binding_protein_residues") or [])]
    gt_set = set(gt_residues)
    if not gt_set:
        return None

    seq = prot.get("sequence") or ""
    length = int(prot.get("length") or len(seq) or 0)
    residue_ids = list(range(1, length + 1))
    if len(residue_ids) < 2:
        return None

    resolved = prot.get("resolved_residues") or []
    resolved_set = {int(r) for r in resolved} if resolved else set(residue_ids)
    mask = np.fromiter((r in resolved_set for r in residue_ids),
                       dtype=bool, count=len(residue_ids))
    y = np.fromiter((1.0 if r in gt_set else 0.0 for r in residue_ids),
                    dtype=np.float64, count=len(residue_ids))

    # RiboSeer per-residue prediction (1-based dict)
    fs = fs_preds.get(sid)
    if not fs:
        return None
    ribo_vec = np.fromiter((float(fs.get(r, 0.0)) for r in residue_ids),
                           dtype=np.float64, count=len(residue_ids))
    ribo_corr = per_sample_corr(ribo_vec[mask], y[mask])
    if ribo_corr is None:
        return None

    # every tool present in step4 (all 15, unfiltered)
    tool_dicts: dict[str, dict[int, float]] = {}
    for p in (s4.get("predictions") or []):
        tid = p.get("tool_id")
        if not tid or not p.get("success"):
            continue
        scores = _tool_scores(p)
        if scores:
            tool_dicts[tid] = scores

    tool_pearson: dict[str, Optional[float]] = {}
    for tid, scores in tool_dicts.items():
        vec = np.fromiter((float(scores.get(r, 0.0)) for r in residue_ids),
                          dtype=np.float64, count=len(residue_ids))
        c = per_sample_corr(vec[mask], y[mask])
        tool_pearson[tid] = None if c is None else c["pearson_r"]

    scored = {t: r for t, r in tool_pearson.items() if r is not None}
    if not scored:
        return None
    best_tool = max(scored, key=lambda t: scored[t])
    best_r = scored[best_tool]
    ribo_r = ribo_corr["pearson_r"]

    return {
        "sample_id": sid,
        "pdb_code": sid[:4],
        "protein_chain": prot.get("chain_id"),
        "rna_chain": rna.get("chain_id"),
        "source_pdb": sample.get("source_pdb") or sid[:4],
        "protein_length": length,
        "rna_length": int(rna.get("length") or len(rna.get("sequence") or "")
                          or 0),
        "gt_binding_residues": len(gt_set),
        "riboseer_pearson": ribo_r,
        "riboseer_corr": ribo_corr,
        "best_baseline_tool": best_tool,
        "best_baseline_r": best_r,
        "delta": round(ribo_r - best_r, 4),
        "tool_pearson": tool_pearson,
        # carried for pass 2 (export)
        "_residue_ids": residue_ids,
        "_seq": seq,
        "_mask": mask,
        "_y": y,
        "_ribo_vec": ribo_vec,
        "_ribo_dict": {r: float(fs.get(r, 0.0)) for r in residue_ids},
        "_tool_dicts": tool_dicts,
        "_gt_residues": gt_residues,
    }


# ---- selection -----------------------------------------------------------


def select_diverse(records: list[dict], top_k: int) -> list[dict]:
    """Highest-delta chain-pair per PDB, then top-k by delta. Keeps two
    cases from ever sharing a structure."""
    best_per_pdb: dict[str, dict] = {}
    for r in records:
        cur = best_per_pdb.get(r["pdb_code"])
        if cur is None or r["delta"] > cur["delta"]:
            best_per_pdb[r["pdb_code"]] = r
    deduped = sorted(best_per_pdb.values(), key=lambda r: -r["delta"])
    return deduped[:top_k]


# ---- pass 2: full RiboSeer metric pack + export -------------------------


def riboseer_metric_pack(rec: dict, *, raw_dir: Optional[Path],
                         hit_threshold: float) -> dict:
    """RiboSeer's full per-sample metrics. AUROC/AUPRC/top-k/MCC over the
    resolved subset; DCC from the structure (None when --raw-dir missing
    or the structure can't be read)."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scripts.tables.table20_topk import (
        compute_topk_metrics, compute_optimal_mcc,
    )

    mask = rec["_mask"]
    y = rec["_y"][mask].astype(int)
    pred = rec["_ribo_vec"][mask]
    corr = rec["riboseer_corr"]
    pack: dict = {
        "pearson_r": corr["pearson_r"],
        "spearman_r": corr["spearman_r"],
        "r_squared": corr["r_squared"],
        "auroc": None, "auprc": None,
        "topk_precision": None, "topk_recall": None, "topk_f1": None,
        "mcc": None, "dcc": None,
    }
    k = int(y.sum())
    if 0 < k < y.size:
        try:
            pack["auroc"] = round(float(roc_auc_score(y, pred)), 4)
            pack["auprc"] = round(float(average_precision_score(y, pred)), 4)
        except ValueError:
            pass
        topk = compute_topk_metrics(y, pred, k)
        if topk is not None:
            pack["topk_precision"] = round(topk[0], 4)
            pack["topk_recall"] = round(topk[1], 4)
            pack["topk_f1"] = round(topk[2], 4)
        pack["mcc"] = round(compute_optimal_mcc(y, pred), 4)

    # DCC: needs the raw structure for Cα / RNA coords.
    if raw_dir is not None and rec.get("source_pdb") and \
            rec.get("protein_chain") and rec.get("rna_chain"):
        struct = _resolve_structure_path(raw_dir, str(rec["source_pdb"]))
        if struct is not None:
            try:
                ca, rna_atoms = extract_chain_coords(
                    struct, str(rec["protein_chain"]),
                    str(rec["rna_chain"]))
                if ca:
                    geo = evaluate_sample_method(
                        scores=rec["_ribo_dict"],
                        gt_residues=rec["_gt_residues"],
                        ca=ca, rna_atoms=rna_atoms,
                        hit_threshold=hit_threshold)
                    if geo is not None:
                        pack["dcc"] = geo.get("dcc")
            except (FileNotFoundError, ValueError):
                pass
    return pack


def _copy_structure(rec: dict, dst_dir: Path,
                    raw_dir: Path) -> Optional[str]:
    """Copy the raw structure into dst_dir. Returns the copied filename
    (e.g. '4n0t.cif') or None if no raw file was found."""
    struct = _resolve_structure_path(raw_dir, str(rec["source_pdb"]))
    if struct is None:
        return None
    dst = dst_dir / f"{rec['pdb_code']}{struct.suffix}"
    try:
        shutil.copy2(struct, dst)
    except OSError:
        return None
    return dst.name


def export_sample(rec: dict, metrics: dict, *, export_dir: Path,
                  raw_dir: Optional[Path]) -> dict:
    """Write the three artefacts for one sample. Returns a small summary
    of what was written (for the stdout tree)."""
    sdir = export_dir / rec["sample_id"]
    sdir.mkdir(parents=True, exist_ok=True)
    written: dict = {"dir": sdir, "structure": None}

    # (a) structure
    if raw_dir is not None:
        written["structure"] = _copy_structure(rec, sdir, raw_dir)

    # (b) metrics json
    all_baselines = {t: rec["tool_pearson"].get(t) for t in ALL_TOOLS}
    doc = {
        "sample_id": rec["sample_id"],
        "pdb_code": rec["pdb_code"],
        "protein_chain": rec["protein_chain"],
        "rna_chain": rec["rna_chain"],
        "protein_length": rec["protein_length"],
        "rna_length": rec["rna_length"],
        "gt_binding_residues": rec["gt_binding_residues"],
        "riboseer": metrics,
        "best_baseline": {
            "tool": rec["best_baseline_tool"],
            "pearson_r": rec["best_baseline_r"],
        },
        "all_baselines": all_baselines,
        "delta": rec["delta"],
    }
    mpath = sdir / f"{rec['sample_id']}_metrics.json"
    mpath.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    written["metrics"] = mpath.name

    # (c) per-residue csv
    cpath = sdir / f"{rec['sample_id']}_residues.csv"
    gt_set = set(rec["_gt_residues"])
    seq = rec["_seq"]
    ribo = rec["_ribo_dict"]
    tdicts = rec["_tool_dicts"]
    cols = (["residue_id", "resname", "gt_binding", "riboseer_prob"]
            + [f"{t}_score" for t in ALL_TOOLS])
    with cpath.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rec["_residue_ids"]:
            row = [r, _resname(seq, r), 1 if r in gt_set else 0,
                   f"{ribo.get(r, 0.0):.6f}"]
            for t in ALL_TOOLS:
                sc = tdicts.get(t, {}).get(r)
                row.append("NaN" if sc is None else f"{float(sc):.6f}")
            w.writerow(row)
    written["residues"] = cpath.name
    return written


# ---- stdout summary ------------------------------------------------------


def _f(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def print_summary(selected: list[dict], export_dir: Optional[Path],
                  exported: list[dict]) -> None:
    print("=" * 96)
    print(f"=== Selected {len(selected)} Best Cases for Visualization ===")
    print("=" * 96)
    print(f"{'Rank':<5s}{'sample_id':<18s}{'PDB':<6s}{'RiboSeer_R':>11s}  "
          f"{'BestBaseline(tool)':<24s}{'Delta':>8s}  "
          f"{'RNA_len':>8s}{'Prot_len':>9s}{'|Bp|':>6s}")
    print("-" * 96)
    for i, r in enumerate(selected, 1):
        bb = f"{r['best_baseline_r']:.3f} ({r['best_baseline_tool']})"
        print(f"{i:<5d}{r['sample_id']:<18s}{r['pdb_code']:<6s}"
              f"{r['riboseer_pearson']:>11.3f}  {bb:<24s}"
              f"{r['delta']:>+8.3f}  {r['rna_length']:>8d}"
              f"{r['protein_length']:>9d}{r['gt_binding_residues']:>6d}")
    print()
    if export_dir is not None and exported:
        print(f"Exported to: {export_dir}/")
        for r, w in zip(selected, exported):
            print(f"  {r['sample_id']}/")
            if w.get("structure"):
                print(f"    {w['structure']}")
            else:
                print(f"    (no raw structure found for {r['source_pdb']})")
            print(f"    {w['metrics']}")
            print(f"    {w['residues']}")
        print()


# ---- driver --------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="processed_quality root (with samples/).")
    p.add_argument("--step4-dir", type=Path, required=True,
                   help="step4 JSONL dir (one .jsonl per sample).")
    p.add_argument("--predictions-dir", type=Path, required=True,
                   help="full-system per-sample predictions <sid>.json.")
    p.add_argument("--split-file", type=Path, required=True,
                   help="test split (one sample_id per line).")
    p.add_argument("--raw-dir", type=Path, default=None,
                   help="raw PDB/CIF dir; needed for DCC + structure copy.")
    p.add_argument("--export-dir", type=Path, default=None,
                   help="write per-sample structure/metrics/residues here.")
    p.add_argument("--top-k", type=int, default=8,
                   help="how many diverse cases to select (default 8).")
    p.add_argument("--hit-threshold", type=float, default=4.0,
                   help="DCC pocket-cluster cutoff in A (default 4.0).")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for d in (args.data_dir, args.step4_dir):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 1
    if args.raw_dir is not None and not args.raw_dir.is_dir():
        print(f"WARNING: --raw-dir not a directory: {args.raw_dir} "
              f"(DCC + structure copy disabled)", file=sys.stderr)
        args.raw_dir = None
    try:
        sample_ids = _load_sample_ids(args.split_file)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    fs_preds = load_predictions_dir(args.predictions_dir)
    if not fs_preds:
        print(f"ERROR: no full-system predictions under "
              f"{args.predictions_dir}", file=sys.stderr)
        return 1
    print(f"loaded {len(fs_preds)} full-system predictions; "
          f"{len(sample_ids)} samples in split")

    records: list[dict] = []
    skipped = 0
    for sid in sample_ids:
        rec = collect_sample(sid, data_dir=args.data_dir,
                             step4_dir=args.step4_dir, fs_preds=fs_preds)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)
    print(f"scored {len(records)} samples ({skipped} skipped: missing "
          f"step4 / sample / GT / RiboSeer / baseline R)")
    if not records:
        print("ERROR: no scorable samples", file=sys.stderr)
        return 1

    selected = select_diverse(records, args.top_k)
    n_pdb = len({r["pdb_code"] for r in records})
    print(f"selected {len(selected)} diverse cases "
          f"(from {n_pdb} distinct PDBs) by delta desc\n")

    # pass 2: full metrics (+ export if requested)
    exported: list[dict] = []
    for rec in selected:
        metrics = riboseer_metric_pack(
            rec, raw_dir=args.raw_dir, hit_threshold=args.hit_threshold)
        rec["_metrics"] = metrics
        if args.export_dir is not None:
            exported.append(export_sample(
                rec, metrics, export_dir=args.export_dir,
                raw_dir=args.raw_dir))

    print_summary(selected, args.export_dir, exported)

    if args.export_dir is None:
        print("(no --export-dir given: ranking only, no files written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
