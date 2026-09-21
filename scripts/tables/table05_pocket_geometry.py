"""Pocket-level geometric evaluation for paper Table 5.

Per-residue Pearson R (table04_main_results.py) only measures whether a
tool's score *ranks* the binding residues high — it doesn't tell us
whether the predicted pocket lands in the right spot in 3-D, or how
well its residue set overlaps the GT. This script fills that gap.

For each method (every tool that appears in step4 + optionally the
trained enriched-fusion bundle) and each sample, we:

  1. Read the per-residue score map (prefer
     ``per_residue_pae_score`` → fall back to ``per_residue_confidence``,
     same convention as ``table04_main_results.py``).
  2. Pick the top-k residues by score, with k = |GT pocket|.
  3. Cluster them on the sequence (residues with index gap ≤ 2 stay
     together) and order clusters by mean score descending.
  4. The top-1 cluster is the "predicted pocket"; the top-3 are used
     for the Top-3 success metric.
  5. Pull Cα coordinates for the protein chain + heavy-atom coordinates
     for the RNA chain straight from the source PDB / mmCIF (one read
     per sample, shared across methods).
  6. Compute DCC (predicted-vs-GT Cα-centroid distance), DCA (predicted
     Cα-centroid to nearest RNA heavy atom), IoU (residue-set overlap),
     and Top-1 / Top-3 success rates at the configurable hit threshold
     (default 4.0 Å).

Outputs:

* ``--output`` aggregated CSV — one row per method.
* ``<output>_per_sample.csv`` — one row per (sample, method) for
  debugging / scatter plots.

Usage
-----
::

    python scripts/tables/table05_pocket_geometry.py \\
        --step4-dir     data/batch_test_v7/step4/ \\
        --processed-dir data/processed_quality \\
        --sample-list   data/processed_quality/splits/test.txt \\
        --raw-dir       data/raw/ \\
        --output        data/batch_test_v7/pocket_geometry.csv \\
        --enriched-model-dir data/enriched_v7_model/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from step5_fusion.data_collector import (  # noqa: E402
    load_sample_json, per_residue_to_int_dict,
)
from step5_fusion.enriched_fusion import EnrichedFusion  # noqa: E402
from step5_fusion.prediction_io import load_predictions_dir  # noqa: E402

# Compute_tmscore_matrix already encodes the raw-dir layout (top-level
# vs rna2p_balanced/ fallback, case-insensitive stem). Reuse it so the
# probe stays in lockstep with the splits we already ran.
from scripts.compute_tmscore_matrix import (  # noqa: E402
    find_raw_pdb, find_raw_cif,
)


# ---- aggregation helpers ------------------------------------------------


def _mean(vs): return round(statistics.fmean(vs), 4) if vs else None
def _median(vs): return round(statistics.median(vs), 4) if vs else None
def _std(vs):
    if len(vs) < 2:
        return 0.0 if vs else None
    return round(statistics.pstdev(vs), 4)


# ---- IO -----------------------------------------------------------------


def _read_last_jsonl_record(path: Path) -> Optional[dict]:
    """Last parseable JSON object in a JSONL file — matches the
    last-line-wins semantics table04_main_results.py uses (a re-run can
    append, the freshest record is at the bottom)."""
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
    return [ln.split()[0] for ln in path.read_text(encoding="utf-8")
            .splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


# ---- per-tool score extraction ------------------------------------------


def _tool_scores(pred: dict) -> dict[int, float]:
    """Per-residue score map: prefer per_residue_pae_score (the Cat-A
    interface-affinity score), fall back to per_residue_confidence. Same
    rule table04_main_results.py uses, so per-residue R and pocket-level
    DCC/IoU run off the same vectors."""
    pae = per_residue_to_int_dict(pred.get("per_residue_pae_score"))
    if pae:
        return pae
    return per_residue_to_int_dict(pred.get("per_residue_confidence"))


# ---- pocket extraction --------------------------------------------------


def extract_pocket_clusters(
    scores: dict[int, float],
    k: int,
    max_gap: int = 2,
) -> list[list[int]]:
    """Return clusters of residues by sequence adjacency, sorted by
    mean score descending.

    1. Keep the top-k residues by score (ties broken by residue id —
       deterministic across runs).
    2. Group residues whose sorted index gap is ≤ ``max_gap``.
    3. Sort the resulting clusters by their mean score (desc).

    Returns ``[[res_id, ...], ...]``. Empty input → empty list.
    """
    if k <= 0 or not scores:
        return []
    # Deterministic tie-break: score desc, residue id asc.
    items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top = items[:k]
    if not top:
        return []
    score_map = dict(top)
    # Cluster by sequence proximity.
    sorted_ids = sorted(score_map)
    clusters: list[list[int]] = [[sorted_ids[0]]]
    for r in sorted_ids[1:]:
        if r - clusters[-1][-1] <= max_gap:
            clusters[-1].append(r)
        else:
            clusters.append([r])
    # Order by mean score desc; tie-break by cluster size desc, then
    # by first residue id asc — keeps ordering deterministic.
    clusters.sort(key=lambda c: (
        -statistics.fmean(score_map[r] for r in c),
        -len(c), c[0],
    ))
    return clusters


# ---- metric helpers -----------------------------------------------------


def _centroid(residues: Iterable[int],
              ca: dict[int, tuple[float, float, float]],
              ) -> Optional[tuple[float, float, float]]:
    """Mean (x, y, z) over the Cα atoms we actually have. Residues
    without coords are dropped silently — common when a sample's
    structure has gaps. Returns None if zero coords survive."""
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for r in residues:
        c = ca.get(r)
        if c is None:
            continue
        xs.append(c[0])
        ys.append(c[1])
        zs.append(c[2])
    if not xs:
        return None
    n = len(xs)
    return (sum(xs) / n, sum(ys) / n, sum(zs) / n)


def _euclid(a: tuple[float, float, float],
            b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                     + (a[2] - b[2]) ** 2)


def compute_dcc(pred: list[int], gt: Iterable[int],
                ca: dict[int, tuple[float, float, float]],
                ) -> Optional[float]:
    """Distance between predicted-pocket Cα centroid and GT Cα centroid
    (Å). None if either set has no usable Cα coords."""
    cp = _centroid(pred, ca)
    cg = _centroid(gt, ca)
    if cp is None or cg is None:
        return None
    return _euclid(cp, cg)


def compute_dca(
    pred: list[int],
    ca: dict[int, tuple[float, float, float]],
    rna_atoms: list[tuple[float, float, float]],
) -> Optional[float]:
    """Predicted Cα centroid → nearest RNA heavy atom (Å). None when
    the centroid is undefined or there are no RNA atoms."""
    cp = _centroid(pred, ca)
    if cp is None or not rna_atoms:
        return None
    return min(_euclid(cp, a) for a in rna_atoms)


def compute_iou(pred: Iterable[int], gt: Iterable[int]) -> float:
    """Residue-set IoU: |P ∩ G| / |P ∪ G|. Empty union → 0.0 (so the
    aggregate behaves the same as "no overlap")."""
    p = set(pred)
    g = set(gt)
    u = p | g
    if not u:
        return 0.0
    return len(p & g) / len(u)


# ---- gemmi structure reader ---------------------------------------------


def extract_chain_coords(
    structure_path: Path,
    protein_chain_id: str,
    rna_chain_id: str,
) -> tuple[dict[int, tuple[float, float, float]],
           list[tuple[float, float, float]]]:
    """Read one structure file once → ``(protein_ca, rna_heavy_atoms)``.

    Uses gemmi's ``assign_label_seq_id(True)`` so residue ids match
    step1's ``label_seq``-based ``binding_protein_residues`` /
    ``resolved_residues``. Protein side: one Cα per residue, keyed by
    ``label_seq``. RNA side: every non-hydrogen atom of the named chain,
    no keying needed (DCA only ever asks for the nearest one).

    Raises FileNotFoundError if ``structure_path`` doesn't exist;
    ValueError if gemmi can't parse the file or assign_label_seq_id
    fails. Missing chains are NOT errors — they return empty maps so
    the caller can skip the sample cleanly.
    """
    import gemmi  # lazy: ~100ms

    if not structure_path.is_file():
        raise FileNotFoundError(structure_path)
    try:
        st = gemmi.read_structure(str(structure_path))
    except Exception as e:  # gemmi raises a variety of RuntimeError types
        raise ValueError(f"gemmi.read_structure failed: {e}") from e
    if len(st) == 0:
        return {}, []
    try:
        st.setup_entities()
        st.assign_label_seq_id(True)
    except Exception as e:
        raise ValueError(f"assign_label_seq_id failed: {e}") from e

    model = st[0]
    ca_map: dict[int, tuple[float, float, float]] = {}
    rna_atoms: list[tuple[float, float, float]] = []

    # Standard protein amino-acid 3-letter codes (subset; gemmi's
    # is_amino_acid would be authoritative but importing
    # find_tabulated_residue here adds another dependency surface — the
    # subset below covers every residue step1 ever produced).
    AA3 = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
        "TYR", "VAL", "SEC", "PYL", "MSE",  # selenocysteine / pyrrolysine / SeMet
    }
    RNA1 = {"A", "U", "G", "C", "I"}

    for chain in model:
        if chain.name == protein_chain_id:
            for res in chain:
                if res.label_seq is None:
                    continue
                if res.name.strip().upper() not in AA3:
                    continue
                ca = None
                for atom in res:
                    if atom.name.strip() == "CA":
                        ca = atom
                        break
                if ca is None:
                    continue
                ca_map[int(res.label_seq)] = (
                    float(ca.pos.x), float(ca.pos.y), float(ca.pos.z))
        if chain.name == rna_chain_id:
            for res in chain:
                rname = res.name.strip().upper()
                # Accept standard RNA + any modified base (≥ 2 chars
                # that ends in a standard letter, common 1-letter modis).
                if rname not in RNA1 and len(rname) > 1:
                    # Heuristic: many modified RNA residues live in
                    # gemmi.find_tabulated_residue. Use it as a tiebreak
                    # without importing it eagerly.
                    try:
                        info = gemmi.find_tabulated_residue(rname)
                    except Exception:
                        info = None
                    if info is None or not info.is_nucleic_acid():
                        continue
                elif rname not in RNA1:
                    continue
                for atom in res:
                    if atom.element.is_hydrogen:
                        continue
                    rna_atoms.append((float(atom.pos.x),
                                      float(atom.pos.y),
                                      float(atom.pos.z)))
    return ca_map, rna_atoms


def _resolve_structure_path(raw_dir: Path,
                            source_pdb: str) -> Optional[Path]:
    """Locate raw structure for ``source_pdb``: PDB first (fast),
    mmCIF fallback. None if neither exists — caller marks the sample
    skipped."""
    try:
        return find_raw_pdb(raw_dir, source_pdb)
    except FileNotFoundError:
        pass
    cif = find_raw_cif(raw_dir, source_pdb)
    return cif


# ---- per-sample metric pack --------------------------------------------


def evaluate_sample_method(
    *,
    scores: dict[int, float],
    gt_residues: list[int],
    ca: dict[int, tuple[float, float, float]],
    rna_atoms: list[tuple[float, float, float]],
    hit_threshold: float,
    top_n: int = 3,
    max_gap: int = 2,
) -> Optional[dict]:
    """Compute the metric pack for one (sample, method).

    Returns dict with ``dcc``, ``dca``, ``iou``, ``top1_hit``,
    ``top3_hit``. None if the prediction is empty or the metrics are
    fundamentally undefined (no scores at all, no GT, missing CA for
    the full GT set).

    ``hit_threshold`` is the DCC cutoff in Å used for Top-1 / Top-3
    success (paper Table 5 uses 4.0 Å)."""
    if not scores or not gt_residues:
        return None
    k = len(gt_residues)
    clusters = extract_pocket_clusters(scores, k, max_gap=max_gap)
    if not clusters:
        return None
    top1 = clusters[0]
    top_n_clusters = clusters[:top_n]
    dcc_top1 = compute_dcc(top1, gt_residues, ca)
    dca = compute_dca(top1, ca, rna_atoms)
    iou = compute_iou(top1, gt_residues)
    # Top-N hit: any of the first N clusters within the threshold.
    top1_hit = top3_hit = False
    if dcc_top1 is not None and dcc_top1 <= hit_threshold:
        top1_hit = True
        top3_hit = True  # top1 ⊂ top3
    if not top3_hit:
        for cl in top_n_clusters[1:]:
            d = compute_dcc(cl, gt_residues, ca)
            if d is not None and d <= hit_threshold:
                top3_hit = True
                break
    return {
        "dcc": None if dcc_top1 is None else round(dcc_top1, 4),
        "dca": None if dca is None else round(dca, 4),
        "iou": round(iou, 4),
        "top1_hit": 1 if top1_hit else 0,
        "top3_hit": 1 if top3_hit else 0,
    }


# ---- driver -------------------------------------------------------------


def evaluate(
    *,
    step4_dir: Path,
    processed_dir: Path,
    raw_dir: Path,
    sample_ids: Optional[list[str]],
    hit_threshold: float,
    enriched_model: Optional[EnrichedFusion],
    fs_preds: Optional[dict[str, dict[int, float]]] = None,
    max_gap: int = 2,
    top_n: int = 3,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Returns ``({method: [per-sample dicts]}, per_sample_rows)``.

    The RiboSeer pocket (``enriched_fusion`` row) is scored from
    ``fs_preds`` (pre-computed full-system per-residue predictions) when
    supplied, else from ``enriched_model.predict_sample``."""
    if sample_ids is None:
        sample_ids = sorted(p.stem for p in step4_dir.glob("*.jsonl"))

    bucket: dict[str, list[dict]] = defaultdict(list)
    per_sample: list[dict] = []
    skipped: dict[str, int] = defaultdict(int)

    for sid in sample_ids:
        s4 = _read_last_jsonl_record(step4_dir / f"{sid}.jsonl")
        if s4 is None:
            skipped["no_step4"] += 1
            continue
        sample = load_sample_json(processed_dir, sid)
        if sample is None:
            skipped["no_sample"] += 1
            continue
        prot = sample.get("protein") or {}
        rna_node = sample.get("rna") or {}
        gt_residues = list(
            (sample.get("interaction") or {})
            .get("binding_protein_residues") or [])
        if not gt_residues:
            skipped["no_gt"] += 1
            continue
        src_pdb = sample.get("source_pdb")
        prot_chain = prot.get("chain_id")
        rna_chain = rna_node.get("chain_id")
        if not src_pdb or not prot_chain or not rna_chain:
            skipped["missing_chain_meta"] += 1
            continue
        struct_path = _resolve_structure_path(raw_dir, str(src_pdb))
        if struct_path is None:
            skipped["no_structure"] += 1
            continue
        try:
            ca, rna_atoms = extract_chain_coords(
                struct_path, str(prot_chain), str(rna_chain))
        except (FileNotFoundError, ValueError):
            skipped["structure_read_fail"] += 1
            continue
        if not ca:
            skipped["no_protein_ca"] += 1
            continue

        preds = s4.get("predictions") or []
        # Each tool gets its own row.
        for p in preds:
            tid = p.get("tool_id")
            if not tid or not p.get("success"):
                continue
            scores = _tool_scores(p)
            if not scores:
                continue
            pack = evaluate_sample_method(
                scores=scores, gt_residues=gt_residues,
                ca=ca, rna_atoms=rna_atoms,
                hit_threshold=hit_threshold,
                top_n=top_n, max_gap=max_gap)
            if pack is None:
                continue
            bucket[tid].append({"sample_id": sid, **pack})
            per_sample.append({"sample_id": sid, "method": tid, **pack})

        if fs_preds is not None or enriched_model is not None:
            if fs_preds is not None:
                probs = fs_preds.get(sid) or {}
            else:
                length = prot.get("length") or len(prot.get("sequence") or "")
                probs = enriched_model.predict_sample(
                    preds, prot.get("sequence") or "", int(length or 0))
            if probs:
                pack = evaluate_sample_method(
                    scores=probs, gt_residues=gt_residues,
                    ca=ca, rna_atoms=rna_atoms,
                    hit_threshold=hit_threshold,
                    top_n=top_n, max_gap=max_gap)
                if pack is not None:
                    bucket["enriched_fusion"].append(
                        {"sample_id": sid, **pack})
                    per_sample.append(
                        {"sample_id": sid, "method": "enriched_fusion",
                         **pack})

    if skipped:
        print("skipped counts:")
        for k, v in sorted(skipped.items()):
            print(f"  {k:24s} {v}")
    return bucket, per_sample


def _aggregate(bucket: dict[str, list[dict]]) -> list[dict]:
    """One row per method. Tools sorted by n_samples desc;
    ``enriched_fusion`` (if present) always appended last."""
    rows: list[dict] = []
    for method, rs in bucket.items():
        dccs = [r["dcc"] for r in rs if r["dcc"] is not None]
        dcas = [r["dca"] for r in rs if r["dca"] is not None]
        ious = [r["iou"] for r in rs if r["iou"] is not None]
        top1 = [r["top1_hit"] for r in rs]
        top3 = [r["top3_hit"] for r in rs]
        rows.append({
            "method": method,
            "n_samples": len(rs),
            "dcc_mean": _mean(dccs), "dcc_median": _median(dccs),
            "dca_mean": _mean(dcas), "dca_median": _median(dcas),
            "iou_mean": _mean(ious), "iou_median": _median(ious),
            "top1_success": (round(100 * statistics.fmean(top1), 2)
                             if top1 else None),
            "top3_success": (round(100 * statistics.fmean(top3), 2)
                             if top3 else None),
        })
    enriched = [r for r in rows if r["method"] == "enriched_fusion"]
    tools = [r for r in rows if r["method"] != "enriched_fusion"]
    tools.sort(key=lambda r: (-r["n_samples"], r["method"]))
    return tools + enriched


_COLUMNS = [
    "method", "n_samples",
    "dcc_mean", "dcc_median",
    "dca_mean", "dca_median",
    "iou_mean", "iou_median",
    "top1_success", "top3_success",
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in _COLUMNS})


def _write_per_sample_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["sample_id", "method", "dcc", "dca", "iou",
            "top1_hit", "top3_hit"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def _print_table(rows: list[dict]) -> None:
    hdr = (f"{'method':22s} {'n':>4s} "
           f"{'DCC':>8s} {'DCA':>8s} {'IoU':>8s} "
           f"{'Top1%':>7s} {'Top3%':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['method']:22s} {r['n_samples']:>4d} "
              f"{str(r['dcc_mean']):>8s} "
              f"{str(r['dca_mean']):>8s} "
              f"{str(r['iou_mean']):>8s} "
              f"{str(r['top1_success']):>7s} "
              f"{str(r['top3_success']):>7s}")


# ---- main ---------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step4-dir", type=Path, required=True,
                   help="step4 JSONL dir (one .jsonl per sample).")
    p.add_argument("--processed-dir", type=Path, required=True,
                   help="processed_quality root (with samples/).")
    p.add_argument("--sample-list", type=Path, default=None,
                   help="one sample_id per line; default = every "
                        "*.jsonl under --step4-dir.")
    p.add_argument("--raw-dir", type=Path, required=True,
                   help="raw PDB / mmCIF directory (top-level + "
                        "rna2p_balanced/ fallback supported).")
    p.add_argument("--output", type=Path, required=True,
                   help="aggregate CSV path; per-sample CSV is written "
                        "alongside as <stem>_per_sample.csv.")
    p.add_argument("--enriched-model-dir", type=Path, default=None,
                   help="optional EnrichedFusion bundle — appends an "
                        "'enriched_fusion' row last.")
    p.add_argument("--predictions-dir", type=Path, default=None,
                   help="pre-computed full-system (on/on/on) predictions "
                        "<sid>.json; when set the RiboSeer pocket is scored "
                        "from these instead of the model bundle.")
    p.add_argument("--hit-threshold", type=float, default=4.0,
                   help="DCC cutoff in Å for Top-1 / Top-3 success "
                        "(paper Table 5: 4.0).")
    p.add_argument("--max-gap", type=int, default=2,
                   help="residues with sequence gap ≤ N merge into one "
                        "cluster (default 2).")
    p.add_argument("--top-n", type=int, default=3,
                   help="how many clusters count toward 'Top-N' hit "
                        "(default 3).")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not args.step4_dir.is_dir():
        print(f"ERROR: --step4-dir not a directory: {args.step4_dir}",
              file=sys.stderr)
        return 1
    if not args.raw_dir.is_dir():
        print(f"ERROR: --raw-dir not a directory: {args.raw_dir}",
              file=sys.stderr)
        return 1
    try:
        sample_ids = _load_sample_ids(args.sample_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    fs_preds = None
    if args.predictions_dir is not None:
        fs_preds = load_predictions_dir(args.predictions_dir)
        if not fs_preds:
            print(f"ERROR: no predictions under {args.predictions_dir}",
                  file=sys.stderr)
            return 1
        print(f"loaded {len(fs_preds)} full-system predictions from "
              f"{args.predictions_dir}")

    enriched_model = None
    if args.enriched_model_dir is not None and fs_preds is None:
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
        raw_dir=args.raw_dir,
        sample_ids=sample_ids,
        hit_threshold=args.hit_threshold,
        enriched_model=enriched_model,
        fs_preds=fs_preds,
        max_gap=args.max_gap,
        top_n=args.top_n,
    )
    rows = _aggregate(bucket)
    _write_csv(args.output, rows)
    ps_path = args.output.with_name(
        args.output.stem + "_per_sample.csv")
    _write_per_sample_csv(ps_path, per_sample)

    print(f"wrote {args.output}  ({len(rows)} methods)")
    print(f"wrote {ps_path}  ({len(per_sample)} sample-method rows)")
    print()
    _print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
