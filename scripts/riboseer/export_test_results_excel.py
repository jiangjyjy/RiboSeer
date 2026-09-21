#!/usr/bin/env python3
"""Export the 107-sample test set — meta + per-sample per-method Pearson
R — to a single .xlsx (one row per sample, sorted by sample_id).

Columns
-------
Meta (9): sample_id, pdb_code, protein_chain, rna_chain, protein_length,
rna_length, protein_sequence, rna_sequence, n_binding_residues.

Pearson R (16): RiboSeer + the 15 baselines, one column each. A method
that produced no result for a sample gets ``NaN``.

RiboSeer-only extra metrics (6): SpearmanR, R2, AUROC, AUPRC, TopkF1,
MCC. (Baselines get Pearson R only, per the brief.) The four sklearn
metrics fall back to ``NaN`` if sklearn isn't installed (a one-line
warning is printed); SpearmanR / R2 come from ``per_sample_corr`` and are
always filled.

All metric math reuses ``find_best_cases`` (which itself reuses
``per_sample_corr`` / ``table20_topk`` / sklearn), so the numbers
match the paper tables and the case-study export.

The first row is a frozen header (freeze_panes='A2').

Usage
-----
::

    python scripts/riboseer/export_test_results_excel.py \\
        --data-dir        data/processed_quality \\
        --step4-dir       data/batch_test_v7/step4 \\
        --predictions-dir data/batch_test_v7/fullsystem_predictions \\
        --split-file      data/processed_quality/splits_tmscore_035/test.txt \\
        --output          exports/test_results_107.xlsx

If openpyxl is missing:
``pip install openpyxl --break-system-packages``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step5_fusion.data_collector import load_sample_json  # noqa: E402
from step5_fusion.prediction_io import load_predictions_dir  # noqa: E402
from scripts.riboseer.extract_case_studies import _load_sample_ids  # noqa: E402
from scripts.riboseer.find_best_cases import (  # noqa: E402
    collect_sample, riboseer_metric_pack,
)

# Baseline Pearson-R columns, in the brief's display order.
METRIC_TOOLS: list[tuple[str, str]] = [
    ("Boltz2", "boltz2"),
    ("Chai1", "chai1"),
    ("EquiPNAS", "equipnas"),
    ("P2Rank", "p2rank"),
    ("RoseTTAFold2NA", "rosettafold2na"),
    ("HDOCK", "hdock"),
    ("NucleicNet", "nucleicnet"),
    ("GraphBind", "graphbind"),
    ("RNABindRPlus", "rnabindrplus"),
    ("Fpocket", "fpocket"),
    ("DeepPocket", "deeppocket"),
    ("HADDOCK3", "haddock3"),
    ("RFAA", "rfaa"),
    ("AlphaFold3", "alphafold3"),
    ("BindUP", "bindup"),
]

META_COLUMNS = [
    "sample_id", "pdb_code", "protein_chain", "rna_chain",
    "protein_length", "rna_length", "protein_sequence", "rna_sequence",
    "n_binding_residues",
]
# RiboSeer-only extra metric columns -> riboseer_metric_pack keys.
RIBOSEER_EXTRA: list[tuple[str, str]] = [
    ("RiboSeer_SpearmanR", "spearman_r"),
    ("RiboSeer_R2", "r_squared"),
    ("RiboSeer_AUROC", "auroc"),
    ("RiboSeer_AUPRC", "auprc"),
    ("RiboSeer_TopkF1", "topk_f1"),
    ("RiboSeer_MCC", "mcc"),
]

COLUMNS: list[str] = (
    META_COLUMNS
    + ["RiboSeer_PearsonR"]
    + [f"{disp}_PearsonR" for disp, _ in METRIC_TOOLS]
    + [c for c, _ in RIBOSEER_EXTRA]
)

NAN = "NaN"   # placeholder written for any missing metric cell


def _metric(v: Optional[float]) -> object:
    return NAN if v is None else round(float(v), 4)


def riboseer_extra(rec: dict, *, sklearn_ok: bool) -> dict:
    """RiboSeer's 6 extra metrics. SpearmanR / R2 always come from the
    already-computed correlation; AUROC/AUPRC/TopkF1/MCC need sklearn —
    when it's absent they're NaN."""
    corr = rec["riboseer_corr"]
    out = {
        "spearman_r": corr.get("spearman_r"),
        "r_squared": corr.get("r_squared"),
        "auroc": None, "auprc": None, "topk_f1": None, "mcc": None,
    }
    if sklearn_ok:
        # raw_dir=None → skip DCC (not part of the spreadsheet).
        pack = riboseer_metric_pack(rec, raw_dir=None, hit_threshold=4.0)
        for k in ("spearman_r", "r_squared", "auroc", "auprc",
                  "topk_f1", "mcc"):
            out[k] = pack.get(k)
    return out


def build_row(rec: dict, sample: dict, *, sklearn_ok: bool) -> dict:
    """One spreadsheet row (column -> value) for a sample."""
    prot = sample.get("protein") or {}
    rna = sample.get("rna") or sample.get("ligand_rna") or {}
    row: dict[str, object] = {
        "sample_id": rec["sample_id"],
        "pdb_code": rec["pdb_code"],
        "protein_chain": rec["protein_chain"],
        "rna_chain": rec["rna_chain"],
        "protein_length": rec["protein_length"],
        "rna_length": rec["rna_length"],
        "protein_sequence": prot.get("sequence") or "",
        "rna_sequence": rna.get("sequence") or "",
        "n_binding_residues": rec["gt_binding_residues"],
        "RiboSeer_PearsonR": _metric(rec["riboseer_pearson"]),
    }
    for disp, tid in METRIC_TOOLS:
        row[f"{disp}_PearsonR"] = _metric(rec["tool_pearson"].get(tid))
    extra = riboseer_extra(rec, sklearn_ok=sklearn_ok)
    for col, key in RIBOSEER_EXTRA:
        row[col] = _metric(extra.get(key))
    return row


def build_rows(sample_ids: list[str], *, data_dir: Path, step4_dir: Path,
               fs_preds: dict, sklearn_ok: bool
               ) -> tuple[list[dict], int]:
    """Build every row (sorted by sample_id). Returns (rows, n_skipped)."""
    rows: list[dict] = []
    skipped = 0
    for sid in sorted(sample_ids):
        rec = collect_sample(sid, data_dir=data_dir, step4_dir=step4_dir,
                             fs_preds=fs_preds)
        if rec is None:
            skipped += 1
            continue
        sample = load_sample_json(data_dir, sid) or {}
        rows.append(build_row(rec, sample, sklearn_ok=sklearn_ok))
    return rows, skipped


def write_xlsx(rows: list[dict], path: Path) -> None:
    """Write rows to an .xlsx with a frozen header row. Raises
    ImportError (with a pip hint) if openpyxl isn't installed."""
    try:
        from openpyxl import Workbook
    except ImportError as e:
        raise ImportError(
            "openpyxl is required to write .xlsx. Install it with:\n"
            "  pip install openpyxl --break-system-packages") from e

    wb = Workbook()
    ws = wb.active
    ws.title = "test_results"
    ws.append(COLUMNS)
    for r in rows:
        ws.append([r.get(c, NAN) for c in COLUMNS])
    ws.freeze_panes = "A2"          # freeze header row
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(path))


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
    p.add_argument("--output", type=Path,
                   default=Path("exports/test_results_107.xlsx"),
                   help="output .xlsx path.")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for d in (args.data_dir, args.step4_dir):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 1
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

    try:
        import sklearn  # noqa: F401
        sklearn_ok = True
    except ImportError:
        sklearn_ok = False
        print("WARNING: sklearn not installed — RiboSeer AUROC/AUPRC/"
              "TopkF1/MCC columns will be NaN "
              "(pip install scikit-learn).", file=sys.stderr)

    print(f"loaded {len(fs_preds)} full-system predictions; "
          f"{len(sample_ids)} samples in split")
    rows, skipped = build_rows(
        sample_ids, data_dir=args.data_dir, step4_dir=args.step4_dir,
        fs_preds=fs_preds, sklearn_ok=sklearn_ok)
    print(f"built {len(rows)} rows ({skipped} skipped: missing "
          f"step4 / sample / GT / RiboSeer / baseline R)")
    if not rows:
        print("ERROR: no rows to write", file=sys.stderr)
        return 1

    try:
        write_xlsx(rows, args.output)
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"wrote {args.output}  ({len(rows)} rows x {len(COLUMNS)} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
