"""Table 20 — distance-cutoff sensitivity of the binding-residue GT.

A protein residue is labelled "binding" when its minimum heavy-atom
distance to any RNA atom is below a cutoff. The shipped data uses
**4.5 Å** (``interaction.distance_cutoff`` in every sample JSON — produced
by step1 ``extract_pairs`` and mirrored by step4
``contact_extractor.DEFAULT_CONTACT_CUTOFF``). Table 20 re-labels the GT
at a range of cutoffs and reports how the per-residue Pearson R moves —
**only the GT changes, never the tool / fusion predictions.**

Method
------
For each test sample, parse the raw structure **once** at the largest
cutoff (8.0 Å) via ``contact_extractor.extract_contacts`` and keep its
``per_residue_protein_min_distance`` map. For a cutoff ``d`` the GT is then
exactly ``{residue : min_dist < d}`` (a residue absent from the map is
≥ 8.0 Å away → non-binding at every cutoff ≤ 8.0), so one parse covers the
whole grid with no re-reading.

Predictions are computed once per sample and correlated against each
cutoff's GT on the resolved-residue subset (same convention as Table 4 /
16-19):

* **Boltz-2 / EquiPNAS / Chai-1 / P2Rank / RoseTTAFold2NA** — each tool's
  raw per-residue score from step4 (``_tool_raw_vector``).
* **RiboSeer** — the pre-trained ``EnrichedFusion`` bundle.

A tool that produced no score for a sample (constant 0 vector) is
undefined there and drops out of that tool's per-cutoff count, so the
per-method ``n`` can differ.

Cutoffs: 3.0, 3.5, 4.0, 4.5(default), 5.0, 6.0, 8.0 Å.

A sanity check prints, at the default cutoff, the mean Jaccard overlap
between the recomputed binding set and each sample's stored
``binding_protein_residues`` — high overlap confirms the chain selection /
residue numbering line up with how the dataset was built.

Usage
-----
::

    python scripts/tables/table19_distance_cutoff.py \\
        --step4-dir          data/batch_test_v7/step4/ \\
        --processed-dir      data/processed_quality \\
        --sample-list        data/processed_quality/splits_tmscore_035/test.txt \\
        --enriched-model-dir data/enriched_v7_lgbm \\
        --raw-dir            data/raw \\
        --output             data/batch_test_v7/table20_distance.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
from step5_fusion.enriched_fusion import EnrichedFusion  # noqa: E402
from step4_tool_adapters.contact_extractor import extract_contacts  # noqa: E402
from scripts.riboseer.ablation_fusion_method import (  # noqa: E402
    SampleData, collect_sample_data, per_sample_corr, _tool_raw_vector,
)
from scripts.tables.table15_rna_length import (  # noqa: E402
    riboseer_predict, _load_sample_ids,
)
from step5_fusion.prediction_io import (  # noqa: E402
    load_predictions_dir, predictions_to_vector,
)

CUTOFFS: tuple[float, ...] = (3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0)
MAX_CUTOFF = max(CUTOFFS)
DEFAULT_CUTOFF = 4.5  # interaction.distance_cutoff in the shipped samples

# Individual tools: (display name, step4 canonical tool_id). RiboSeer (the
# fusion model) is appended after these so it prints last.
TOOL_METHODS: tuple[tuple[str, str], ...] = (
    ("Boltz-2", "boltz2"),
    ("EquiPNAS", "equipnas"),
    ("Chai-1", "chai1"),
    ("P2Rank", "p2rank"),
    ("RoseTTAFold2NA", "rosettafold2na"),
)
RIBOSEER = "RiboSeer"
METHODS: tuple[str, ...] = tuple(n for n, _ in TOOL_METHODS) + (RIBOSEER,)

# Accept display names, step4 tool ids, and a few aliases as --methods tokens.
_NAME_BY_KEY: dict[str, str] = {RIBOSEER.lower(): RIBOSEER, "rf2na":
                                "RoseTTAFold2NA"}
for _disp, _tid in TOOL_METHODS:
    _NAME_BY_KEY[_disp.lower()] = _disp
    _NAME_BY_KEY[_tid.lower()] = _disp


def resolve_methods(tokens: Optional[list[str]]) -> list[str]:
    """Map ``--methods`` tokens (display names / tool ids / aliases) to the
    canonical method list, preserving ``METHODS`` order. None → all."""
    if not tokens:
        return list(METHODS)
    chosen: set[str] = set()
    for t in tokens:
        name = _NAME_BY_KEY.get(t.strip().lower())
        if name is None:
            raise ValueError(
                f"unknown method {t!r}; choose from "
                f"{[d for d, _ in TOOL_METHODS] + [RIBOSEER]}")
        chosen.add(name)
    return [m for m in METHODS if m in chosen]


# ---------------------------------------------------------------------------
# Raw structure lookup (self-contained; mirrors compute_tmscore_matrix._probe)
# ---------------------------------------------------------------------------


def find_structure(raw_dir: Path, pdb: str) -> Optional[Path]:
    """Locate ``<raw>/<pdb>.{cif,pdb}`` (cif preferred), also probing the
    ``rna2p_balanced/`` subdir, case-insensitively. None if absent."""
    pdb_lc = (pdb or "").lower()
    for ext in ("cif", "pdb"):
        for c in (raw_dir / f"{pdb_lc}.{ext}",
                  raw_dir / f"{pdb}.{ext}",
                  raw_dir / "rna2p_balanced" / f"{pdb_lc}.{ext}",
                  raw_dir / "rna2p_balanced" / f"{pdb}.{ext}"):
            if c.is_file():
                return c
    return None


# ---------------------------------------------------------------------------
# GT at a cutoff + per-cutoff correlations (pure, testable)
# ---------------------------------------------------------------------------


def gt_vector(residue_ids: list[int], min_dist: dict[int, float],
              cutoff: float) -> np.ndarray:
    """Binary GT over ``residue_ids``: 1 where the residue's min RNA
    distance is < ``cutoff``."""
    return np.fromiter(
        (1.0 if min_dist.get(r, math.inf) < cutoff else 0.0
         for r in residue_ids),
        dtype=np.float64, count=len(residue_ids))


def per_cutoff_corr(residue_ids: list[int], eval_mask: np.ndarray,
                    pred: np.ndarray, min_dist: dict[int, float],
                    cutoff: float) -> Optional[dict]:
    """One sample's Pearson/Spearman/R² of ``pred`` vs the cutoff-``d`` GT,
    on the resolved-residue subset. None when undefined (constant pred or
    no positive/negative GT in the subset)."""
    y = gt_vector(residue_ids, min_dist, cutoff)
    if eval_mask.size != pred.size or eval_mask.size != y.size:
        return None
    return per_sample_corr(pred[eval_mask], y[eval_mask])


def _jaccard(a: set[int], b: set[int]) -> Optional[float]:
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Per-sample min-distance extraction
# ---------------------------------------------------------------------------


def sample_min_distances(sample: dict, raw_dir: Path,
                         max_cutoff: float = MAX_CUTOFF
                         ) -> tuple[Optional[dict[int, float]], str]:
    """``({residue: min_RNA_distance}, note)`` for one sample by parsing its
    raw structure once at ``max_cutoff``. ``min_dist`` is None on any
    failure (missing file / unreadable / no contacts), with ``note`` the
    reason."""
    pdb = sample.get("source_pdb") or ""
    path = find_structure(raw_dir, pdb)
    if path is None:
        return None, f"no structure for {pdb!r}"
    prot_chain = (sample.get("protein") or {}).get("chain_id")
    rna_chain = (sample.get("rna") or {}).get("chain_id")
    res = extract_contacts(path, cutoff=max_cutoff,
                           protein_chain_id=prot_chain, rna_chain_id=rna_chain)
    if res.note:
        return None, res.note
    md = res.per_residue_protein_min_distance
    if not md:
        return None, f"no contacts within {max_cutoff}A"
    return md, ""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _mean(vs):
    return round(statistics.fmean(vs), 4) if vs else None


def run(samples: list[SampleData], model: Optional[EnrichedFusion],
        processed_dir: Path, raw_dir: Path, methods: list[str],
        cutoffs: list[float],
        fs_preds: Optional[dict[str, dict[int, float]]] = None,
        ) -> tuple[dict[tuple[float, str], list[dict]], dict]:
    """Returns ({(cutoff, method): [per-sample corr]}, diagnostics).

    ``methods`` is the subset to evaluate; ``cutoffs`` the GT thresholds.
    Structures are parsed once at ``max(cutoffs)`` so the whole grid is
    covered by a single read per sample. RiboSeer's per-residue vector
    comes from ``fs_preds`` (pre-computed full-system predictions) when
    supplied, else from the loaded ``model``."""
    tool_id_by_name = dict(TOOL_METHODS)
    max_cutoff = max(cutoffs)
    bucket: dict[tuple[float, str], list[dict]] = {
        (d, m): [] for d in cutoffs for m in methods}
    jaccard: list[float] = []
    skipped = 0
    used = 0

    for s in samples:
        sample = load_sample_json(processed_dir, s.sid)
        if sample is None:
            skipped += 1
            continue
        min_dist, note = sample_min_distances(sample, raw_dir, max_cutoff)
        if min_dist is None:
            print(f"  skip {s.sid}: {note}", file=sys.stderr)
            skipped += 1
            continue
        used += 1

        # Predictions computed once; only the GT varies across cutoffs.
        preds: dict[str, np.ndarray] = {}
        for m in methods:
            if m == RIBOSEER:
                preds[m] = (predictions_to_vector(
                    fs_preds.get(s.sid), s.residue_ids)
                    if fs_preds is not None
                    else riboseer_predict(model, s.X))
            else:
                preds[m] = _tool_raw_vector(s, tool_id_by_name[m])
        for d in cutoffs:
            for m in methods:
                corr = per_cutoff_corr(s.residue_ids, s.eval_mask,
                                       preds[m], min_dist, d)
                if corr is not None:
                    bucket[(d, m)].append(corr)

        # Sanity: recomputed default-cutoff binding vs stored GT.
        recomputed = {r for r, dist in min_dist.items() if dist < DEFAULT_CUTOFF}
        stored = {int(r) for r in
                  (sample.get("interaction") or {})
                  .get("binding_protein_residues") or []}
        j = _jaccard(recomputed, stored)
        if j is not None:
            jaccard.append(j)

    diagnostics = {
        "used": used, "skipped": skipped,
        "mean_jaccard_at_default": _mean(jaccard),
        "n_jaccard": len(jaccard),
    }
    return bucket, diagnostics


def aggregate(bucket: dict[tuple[float, str], list[dict]],
              methods: list[str], cutoffs: list[float]) -> list[dict]:
    rows: list[dict] = []
    for d in cutoffs:
        for m in methods:
            corrs = bucket[(d, m)]
            prs = [c["pearson_r"] for c in corrs if c["pearson_r"] is not None]
            srs = [c["spearman_r"] for c in corrs
                   if c["spearman_r"] is not None]
            r2s = [c["r_squared"] for c in corrs if c["r_squared"] is not None]
            rows.append({
                "cutoff": d, "method": m, "n": len(corrs),
                "pearson_r_mean": _mean(prs), "spearman_r_mean": _mean(srs),
                "r2_mean": _mean(r2s),
                "is_default": (d == DEFAULT_CUTOFF),
            })
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_COLUMNS = ["cutoff", "method", "n", "pearson_r_mean", "spearman_r_mean",
            "r2_mean", "is_default"]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in _COLUMNS})


def print_table(rows: list[dict]) -> None:
    """Combined long table: one row per (cutoff, method), grouped by cutoff
    with all methods under each cutoff. ``rows`` is already in (cutoff,
    method) order from ``aggregate``."""
    hdr = (f"{'Cutoff(A)':>9s} {'Method':<15s} {'n':>4s} {'PearsonR':>9s} "
           f"{'SpearmanR':>10s} {'R2':>8s}")
    print(hdr)
    print("-" * len(hdr))
    last_cutoff = None
    for r in rows:
        if last_cutoff is not None and r["cutoff"] != last_cutoff:
            print()  # blank line between cutoff blocks
        last_cutoff = r["cutoff"]
        tag = "  <- default" if r["is_default"] else ""
        print(f"{r['cutoff']:>9.1f} {r['method']:<15s} {r['n']:>4d} "
              f"{str(r['pearson_r_mean']):>9s} "
              f"{str(r['spearman_r_mean']):>10s} "
              f"{str(r['r2_mean']):>8s}{tag}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step4-dir", type=Path, required=True,
                   help="test-split step4 JSONL dir")
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--sample-list", type=Path, required=True)
    p.add_argument("--enriched-model-dir", type=Path, default=None,
                   help="pre-trained RiboSeer EnrichedFusion bundle "
                        "(required only when RiboSeer is in --methods and "
                        "--predictions-dir is not given)")
    p.add_argument("--predictions-dir", type=Path, default=None,
                   help="pre-computed full-system (on/on/on) predictions "
                        "<sid>.json; when set RiboSeer reads these instead "
                        "of loading the model")
    p.add_argument("--raw-dir", type=Path, required=True,
                   help="raw structure dir (.cif/.pdb, + rna2p_balanced/)")
    p.add_argument("--methods", type=str, nargs="+", default=None,
                   help="subset to evaluate (display names / tool ids); "
                        "default = all 6 (Boltz-2 EquiPNAS Chai-1 P2Rank "
                        "RoseTTAFold2NA RiboSeer)")
    p.add_argument("--cutoffs", type=float, nargs="+", default=list(CUTOFFS),
                   help="GT distance cutoffs in Å (default 3.0 3.5 4.0 4.5 "
                        "5.0 6.0 8.0)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for d in (args.step4_dir, args.processed_dir, args.raw_dir):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 1
    try:
        methods = resolve_methods(args.methods)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    cutoffs = sorted(set(args.cutoffs))
    try:
        sample_ids = _load_sample_ids(args.sample_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # RiboSeer needs either pre-computed predictions or the fusion model;
    # single-tool-only runs need neither.
    model = None
    fs_preds = None
    if RIBOSEER in methods:
        if args.predictions_dir is not None:
            fs_preds = load_predictions_dir(args.predictions_dir)
            if not fs_preds:
                print(f"ERROR: no predictions under {args.predictions_dir}",
                      file=sys.stderr)
                return 1
            print(f"loaded {len(fs_preds)} full-system predictions from "
                  f"{args.predictions_dir}")
        elif args.enriched_model_dir is None:
            print("ERROR: pass --predictions-dir or --enriched-model-dir "
                  "when RiboSeer is in --methods", file=sys.stderr)
            return 1
        else:
            try:
                model = EnrichedFusion.load(args.enriched_model_dir)
            except (OSError, ValueError, ImportError,
                    json.JSONDecodeError) as e:
                print(f"ERROR: --enriched-model-dir load failed: {e}",
                      file=sys.stderr)
                return 1

    samples = collect_sample_data(
        args.step4_dir, args.processed_dir, sample_ids,
        feature_set=(model.feature_set if model else "full"),
        use_context=(model.use_context if model else True))
    if not samples:
        print("ERROR: no usable test samples", file=sys.stderr)
        return 1
    print(f"usable test samples: {len(samples)}  "
          f"methods={methods}  cutoffs={cutoffs}")

    bucket, diag = run(samples, model, args.processed_dir, args.raw_dir,
                       methods, cutoffs, fs_preds)
    rows = aggregate(bucket, methods, cutoffs)

    write_csv(args.output, rows)
    print(f"wrote {args.output}  ({len(cutoffs)} cutoffs × {len(methods)} "
          f"methods); structures used={diag['used']}, "
          f"skipped={diag['skipped']}")
    mj = diag["mean_jaccard_at_default"]
    print(f"sanity: mean Jaccard(recomputed@{DEFAULT_CUTOFF}A vs stored GT) = "
          f"{mj} over {diag['n_jaccard']} samples "
          f"(≈1.0 confirms chain/numbering alignment)")
    print()
    print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
