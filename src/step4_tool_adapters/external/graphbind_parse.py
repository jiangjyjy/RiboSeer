#!/usr/bin/env python3
"""Parse GraphBind output into step-4 ``<sample_id>.jsonl`` (Category C).

GraphBind is a Category-C RNA-binding-residue predictor (same family as
EquiPNAS): it emits one row per protein residue with a binding probability and
a 0/1 call. Per sample the result is
``<results>/<sample_id>/RNA-binding_result.csv``::

    ,Residue_ID,Residue,Probability,Binary
    0,9,E,0.001,0
    1,10,P,0.000,0
    ...

  * ``Residue_ID`` — residue number in the single-chain PDB GraphBind was given
  * ``Probability`` — RNA-binding probability in [0, 1]
  * ``Binary`` — 0/1 thresholded call

Residue numbering / GT alignment
--------------------------------
The GraphBind input PDB (``graphbind_inputs/<sample_id>.pdb``) is the single
protein chain extracted with gemmi and **renumbered to 1-based label_seq**, the
same convention P2Rank / EquiPNAS / NucleicNet use — so ``Residue_ID`` is the
1-based polymer index and lines up with step-1 ``binding_protein_residues``
directly. We therefore emit ``per_residue_confidence`` keyed by ``Residue_ID``
with no remap. When ``--processed-dir`` is supplied we *warn* (but don't drop)
if any ``Residue_ID`` falls outside ``[1, protein.length]`` — the tell-tale sign
the input was numbered in some other (e.g. author) scheme and the alignment
assumption broke.

Output
------
A ``ToolPrediction(tool_id="graphbind", category="C")`` carrying
``per_residue_confidence`` (every CSV row) and ``binding_protein_residues``
(``Binary == 1``, or ``Probability > --threshold`` when that flag is given),
merged into ``<step4-dir>/<sample_id>.jsonl`` — replacing any prior
``graphbind`` entry and leaving other tools untouched (same convention as
nucleicnet_parse / deeppocket_parse / af3_parse).

Usage
-----
::

    python -m step4_tool_adapters.external.graphbind_parse.py \
        --results-dir   graphbind_outputs/ \
        --step4-dir     data/batch_test_v7/step4/ \
        --processed-dir data/processed_quality \
        --sample-list   data/processed_quality/splits_tmscore_035/test.txt
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Optional

# Make repo root + src importable for the schemas and shared helpers.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from step4_tool_adapters.schemas import (  # noqa: E402
    ToolPrediction,
    ToolPredictionSet,
)
from step4_tool_adapters.tool_io import (  # noqa: E402
    clean_sample_id,
    read_sample_list,
)

logger = logging.getLogger("graphbind_parse")

TOOL_ID = "graphbind"
CATEGORY = "C"

_RESULT_CSV_NAME = "RNA-binding_result.csv"
_TRUE_TOKENS = {"1", "1.0", "true", "yes", "y", "t"}


# --------------------------------------------------------------------------
# locating + parsing the CSV
# --------------------------------------------------------------------------
def find_result_csv(sample_results_dir: Path) -> Optional[Path]:
    """Locate the GraphBind result CSV for one sample.

    Prefers the canonical ``RNA-binding_result.csv``; falls back to any
    ``*result*.csv`` to tolerate naming differences across GraphBind builds.
    """
    if not sample_results_dir.is_dir():
        return None
    direct = sample_results_dir / _RESULT_CSV_NAME
    if direct.is_file():
        return direct
    hits = sorted(sample_results_dir.glob("*result*.csv"))
    return hits[0] if hits else None


def _pick_column(fieldnames: list[str], *candidates: str) -> Optional[str]:
    """Case-insensitive lookup of the first matching column name."""
    lower = {(f or "").strip().lower(): f for f in fieldnames}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def _coerce_int(value: str) -> Optional[int]:
    try:
        return int(float(str(value).strip()))  # tolerate "9" or "9.0"
    except (TypeError, ValueError):
        return None


def _coerce_prob(value: str) -> Optional[float]:
    try:
        f = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return min(1.0, max(0.0, f))


def parse_graphbind_csv(
    path: Path, *, prob_threshold: Optional[float] = None,
) -> tuple[dict[int, float], list[int]]:
    """Parse a GraphBind result CSV.

    Returns ``(per_residue {Residue_ID: Probability}, binding_residue_ids)``.
    ``binding`` is ``Binary == 1`` unless ``prob_threshold`` is given, in which
    case it is ``Probability > prob_threshold``. Rows with an unparseable /
    non-positive ``Residue_ID`` are skipped (logged as a count)."""
    text = path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    fieldnames = [(f or "").strip() for f in (reader.fieldnames or [])]
    res_col = _pick_column(fieldnames, "residue_id", "residue id", "resid",
                           "residue_index", "residue")
    prob_col = _pick_column(fieldnames, "probability", "prob", "score")
    bin_col = _pick_column(fieldnames, "binary", "binary_label", "label",
                           "prediction", "pred")
    # ``Residue`` (the one-letter code) and ``Residue_ID`` both fuzzy-match
    # "residue"; make sure we didn't grab the wrong one.
    if res_col is not None and res_col.strip().lower() == "residue":
        better = _pick_column(fieldnames, "residue_id", "residue id", "resid")
        if better is not None:
            res_col = better
    if res_col is None or prob_col is None:
        raise ValueError(
            f"{path} missing Residue_ID / Probability columns "
            f"(found: {fieldnames})"
        )

    per_res: dict[int, float] = {}
    binding: list[int] = []
    n_bad = 0
    for row in reader:
        rid = _coerce_int(row.get(res_col, ""))
        prob = _coerce_prob(row.get(prob_col, ""))
        if rid is None or rid < 1 or prob is None:
            n_bad += 1
            continue
        per_res[rid] = prob
        if prob_threshold is not None:
            is_bind = prob > prob_threshold
        elif bin_col is not None:
            is_bind = str(row.get(bin_col, "")).strip().lower() in _TRUE_TOKENS
        else:  # no Binary column and no threshold -> fall back to >0.5
            is_bind = prob > 0.5
        if is_bind:
            binding.append(rid)
    if n_bad:
        logger.warning("%s: skipped %d row(s) with bad Residue_ID/Probability",
                       path.name, n_bad)
    return per_res, sorted(set(binding))


# --------------------------------------------------------------------------
# record assembly + merge
# --------------------------------------------------------------------------
def build_prediction(
    sample_id: str,
    per_res: dict[int, float],
    binding: list[int],
    *,
    raw_output_dir: Optional[str] = None,
) -> ToolPrediction:
    """Build the ``graphbind`` ToolPrediction from the parsed CSV."""
    per_res_round = {int(k): round(float(v), 4) for k, v in per_res.items()}
    binding = sorted({int(b) for b in binding if int(b) >= 1})
    return ToolPrediction(
        tool_id=TOOL_ID,
        category=CATEGORY,
        sample_id=sample_id,
        success=True,
        binding_protein_residues=binding or None,
        per_residue_confidence=per_res_round or None,
        raw_output_dir=raw_output_dir,
    )


def merge_into_set(
    step4_dir: Path, sample_id: str, prediction: ToolPrediction,
) -> Path:
    """Insert ``prediction`` into ``<step4_dir>/<sample_id>.jsonl``, dropping any
    prior ``graphbind`` entry and preserving other tools."""
    step4_dir.mkdir(parents=True, exist_ok=True)
    out_path = step4_dir / f"{sample_id}.jsonl"

    pset = ToolPredictionSet(sample_id=sample_id)
    if out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    pset = ToolPredictionSet.model_validate_json(line)
                except Exception:  # noqa: BLE001
                    pass
    pset.sample_id = sample_id
    pset.predictions = [p for p in pset.predictions if p.tool_id != TOOL_ID]
    pset.predictions.append(prediction)
    pset.tools_run = sorted({p.tool_id for p in pset.predictions})

    tmp = out_path.with_suffix(".jsonl.tmp")
    tmp.write_text(pset.model_dump_json() + "\n", encoding="utf-8")
    tmp.replace(out_path)
    return out_path


def _protein_length(samples_dir: Optional[Path], sample_id: str) -> Optional[int]:
    """Best-effort protein length, used only to sanity-check residue indices."""
    if samples_dir is None:
        return None
    try:
        from step4_tool_adapters.tool_io import load_sample_json
        sample = load_sample_json(samples_dir, sample_id)
    except Exception:  # noqa: BLE001
        return None
    length = (sample.get("protein") or {}).get("length")
    return int(length) if isinstance(length, int) else None


def parse_one_sample(
    sample_id: str,
    *,
    results_dir: Path,
    samples_dir: Optional[Path],
    prob_threshold: Optional[float],
) -> ToolPrediction:
    csv_path = find_result_csv(results_dir / sample_id)
    if csv_path is None:
        raise FileNotFoundError(
            f"no GraphBind result CSV for {sample_id} under "
            f"{results_dir / sample_id}"
        )
    per_res, binding = parse_graphbind_csv(csv_path, prob_threshold=prob_threshold)
    if not per_res:
        raise ValueError(f"{csv_path} parsed to 0 per-residue rows")

    # Alignment sanity check (warn only — do not remap/drop).
    length = _protein_length(samples_dir, sample_id)
    if length is not None:
        oob = [r for r in per_res if r < 1 or r > length]
        if oob:
            logger.warning(
                "%s: %d Residue_ID(s) outside [1, %d] (e.g. %s); GraphBind input "
                "numbering may not match the ground-truth index space",
                sample_id, len(oob), length, sorted(oob)[:5])

    return build_prediction(
        sample_id, per_res, binding, raw_output_dir=str(csv_path.parent))


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results-dir", required=True, type=Path,
                   help="GraphBind output dir (holds <sample_id>/ folders)")
    p.add_argument("--step4-dir", required=True, type=Path,
                   help="step-4 JSONL dir to merge into")
    p.add_argument("--processed-dir", type=Path, default=None,
                   help="step-1 processed dir; its samples/ is used only to "
                        "sanity-check residue indices (optional)")
    p.add_argument("--samples-dir", type=Path, default=None,
                   help="override sample-JSON dir "
                        "(default <processed-dir>/samples)")
    p.add_argument("--sample-list", type=Path, default=None,
                   help="txt/csv of sample ids; default: discover from "
                        "results-dir subfolders")
    p.add_argument("--threshold", type=float, default=None,
                   help="if set, derive binding from Probability > THRESHOLD "
                        "instead of the GraphBind Binary column")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level)

    results_dir = args.results_dir.expanduser().resolve()
    step4_dir = args.step4_dir.expanduser().resolve()
    samples_dir = None
    if args.samples_dir is not None:
        samples_dir = args.samples_dir.expanduser().resolve()
    elif args.processed_dir is not None:
        samples_dir = args.processed_dir.expanduser().resolve() / "samples"

    if not results_dir.is_dir():
        logger.error("results dir not found: %s", results_dir)
        return 1

    if args.sample_list is not None:
        samples = read_sample_list(args.sample_list.expanduser().resolve())
    else:
        samples = sorted(
            c.name for c in results_dir.iterdir()
            if c.is_dir() and find_result_csv(c) is not None
        )
    if not samples:
        logger.error("no samples to parse under %s", results_dir)
        return 1

    n_ok = n_missing = n_fail = 0
    total = len(samples)
    for i, sid in enumerate(samples, 1):
        sid = clean_sample_id(sid)
        try:
            pred = parse_one_sample(
                sid, results_dir=results_dir, samples_dir=samples_dir,
                prob_threshold=args.threshold)
        except FileNotFoundError as e:
            n_missing += 1
            logger.warning("[%d/%d] %s missing: %s", i, total, sid, e)
            continue
        except Exception:  # noqa: BLE001
            n_fail += 1
            logger.exception("[%d/%d] %s failed", i, total, sid)
            continue
        out = merge_into_set(step4_dir, sid, pred)
        n_res = len(pred.per_residue_confidence or {})
        n_bind = len(pred.binding_protein_residues or [])
        n_ok += 1
        logger.info("[%d/%d] %s -> %s (%d residues, %d binding)",
                    i, total, sid, out, n_res, n_bind)

    logger.info("done: %d ok, %d missing, %d failed (of %d)",
                n_ok, n_missing, n_fail, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
