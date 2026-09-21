#!/usr/bin/env python3
"""Parse RNABindRPlus output into step-4 ``<sample_id>.jsonl`` (Category C).

RNABindRPlus is a sequence-based RNA-binding-residue predictor; results are
emailed per submitted job. This is a **framework**: the format-independent parts
(submitted->original residue remap, step4 emit, sample lookup) are complete; the
``parse_result_table`` function is a *tolerant best-effort* parser that should be
tightened once the real emailed format is in hand (see the TODO there). Run with
``--inspect`` on one real result file to see what it received.

Residue index remap (critical)
-------------------------------
The FASTA submitted to RNABindRPlus had gaps / non-standard residues stripped,
so its residue ``k`` (1-based) is **not** the ground-truth index. The companion
``rnabindrplus_index_map.json`` (written by ``rnabindrplus_inputs.py``) maps
``k -> kept_positions[k-1]`` (the original 1-based / label_seq position). This
parser applies that remap so the emitted ``per_residue_confidence`` /
``binding_protein_residues`` line up with step-1 ground truth. If the map is
absent it re-derives the mapping from the sample JSON (its ``protein.sequence``
non-gap positions == ``protein.resolved_residues``).

Output: ``ToolPrediction(tool_id="rnabindrplus", category="C")`` merged into
``<step4-dir>/<sample_id>.jsonl`` (replaces any prior ``rnabindrplus`` entry,
preserves other tools).

Usage
-----
::

    # once results are downloaded into a dir (one file per sample, or adjust)
    python -m step4_tool_adapters.external.rnabindrplus_parse.py \
        --results-dir   rnabindrplus_results/ \
        --step4-dir     data/batch_test_v7/step4/ \
        --processed-dir data/processed_quality \
        --index-map     data/batch_test_v7/rnabindrplus_index_map.json \
        --sample-list   data/processed_quality/splits_tmscore_035/test.txt

    # peek at one downloaded result file to confirm its columns
    python -m step4_tool_adapters.external.rnabindrplus_parse.py --inspect path/to/result.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

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
    load_sample_json,
    read_sample_list,
)

logger = logging.getLogger("rnabindrplus_parse")

TOOL_ID = "rnabindrplus"
CATEGORY = "C"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Column-name candidates (case-insensitive) for the tolerant table parser.
_POS_COLS = ("position", "residue_number", "resid", "res_id", "index", "idx",
             "pos", "num", "i", "residue_id")
_PROB_COLS = ("probability", "prob", "rnabindrplus", "score", "binding_probability",
              "rvp", "p_binding")
_BIN_COLS = ("binary", "prediction", "pred", "label", "binding", "class", "call")
_TRUE = {"1", "1.0", "+", "b", "y", "yes", "true", "binding"}


# --------------------------------------------------------------------------
# index remap
# --------------------------------------------------------------------------
def load_index_map(path: Optional[Path]) -> dict:
    """Load ``rnabindrplus_index_map.json`` -> {sample_id: {kept_positions: [...]}}.
    Returns {} when absent (callers fall back to the sample JSON)."""
    if path is None or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def kept_positions_for(
    sample_id: str, index_map: dict, samples_dir: Optional[Path],
) -> Optional[list]:
    """Original 1-based positions of the submitted residues for a sample.

    Prefer the index map; else re-derive from the sample JSON (non-gap positions
    of ``protein.sequence``). Returns None if neither is available."""
    entry = index_map.get(sample_id)
    if entry and entry.get("kept_positions"):
        return list(entry["kept_positions"])
    if samples_dir is not None:
        try:
            sample = load_sample_json(samples_dir, sample_id)
        except Exception:  # noqa: BLE001
            return None
        seq = (sample.get("protein") or {}).get("sequence") or ""
        kept = [i for i, c in enumerate(seq.upper(), start=1) if c in STANDARD_AA]
        if kept:
            return kept
    return None


# --------------------------------------------------------------------------
# result parsing (real RNABindRPlus "finalpredictions" format)
# --------------------------------------------------------------------------
# Each batch is ONE combined file holding many sequence blocks::
#
#     #Input sequence length: 86
#     #Number of binding residues predicted by ...
#     >3wbm_A_X
#     sequence:            T,P,T,...
#     Prediction from HomPRIP:        ?,?,...
#     Predicted score from HomPRIP:   ?,?,...
#     Prediction from SVM:            0,0,1,...
#     Predicted score from SVM:       0.00,...
#     Prediction from RNABindRPlus:   0,0,1,...     <- binary call
#     Predicted score from RNABindRPlus: 0.01,...   <- per-residue probability
#
# We take the **RNABindRPlus** score + binary lines (the method's own combined
# output); values are comma-separated and 1:1 with the submitted (gap-free)
# residues, so position k -> kept_positions[k-1] for GT alignment.
_SCORE_PREFIX = "predicted score from rnabindrplus"
_BINARY_PREFIX = "prediction from rnabindrplus"


def _as_float(s):
    try:
        f = float(str(s).strip())
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _as_binary(s):
    t = str(s).strip().lower()
    if t in _TRUE:
        return 1
    if t in ("0", "0.0", "-", "n", "no", "false", "nonbinding", "non-binding"):
        return 0
    f = _as_float(s)
    if f is not None:
        return 1 if f >= 0.5 else 0
    return None


def _values_after_colon(line: str) -> list:
    """``label:<tabs>v1,v2,...`` -> ``['v1','v2',...]`` (empty tokens dropped)."""
    rhs = line.split(":", 1)[1] if ":" in line else ""
    return [t.strip() for t in rhs.split(",") if t.strip() != ""]


def parse_block(block_lines: list):
    """Parse one sequence block's lines into ``[(pos, prob, binary), ...]``.

    Uses the ``RNABindRPlus`` score (probability) and prediction (0/1) rows;
    positions are 1-based in submitted-sequence order. Returns ``[]`` if neither
    RNABindRPlus row is present."""
    score = None
    binary = None
    for ln in block_lines:
        low = ln.strip().lower()
        if low.startswith(_SCORE_PREFIX):
            score = [_as_float(v) for v in _values_after_colon(ln)]
        elif low.startswith(_BINARY_PREFIX):
            binary = [_as_binary(v) for v in _values_after_colon(ln)]
    if not score and not binary:
        return []
    n = max(len(score or []), len(binary or []))
    rows = []
    for i in range(n):
        prob = score[i] if score and i < len(score) else None
        b = binary[i] if binary and i < len(binary) else None
        rows.append((i + 1, prob, b))
    return rows


def parse_combined_file(text: str) -> dict:
    """Split a combined ``finalpredictions`` file into ``{sample_id: rows}``.

    Blocks are delimited by ``>sample_id`` header lines."""
    out = {}
    sid = None
    buf = []

    def _flush():
        if sid is not None:
            rows = parse_block(buf)
            if rows:
                out[sid] = rows

    for ln in text.splitlines():
        if ln.lstrip().startswith(">"):
            _flush()
            sid = clean_sample_id(ln.strip()[1:])
            buf = []
        elif sid is not None:
            buf.append(ln)
    _flush()
    return out


def load_all_results(results_dir: Path, glob: str) -> dict:
    """Parse every matching ``finalpredictions`` file under ``results_dir`` and
    merge into one ``{sample_id: rows}`` map."""
    combined = {}
    files = sorted(results_dir.glob(glob)) or sorted(results_dir.rglob(glob))
    for f in files:
        try:
            part = parse_combined_file(f.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        for sid, rows in part.items():
            if sid in combined:
                logger.warning("duplicate result for %s (in %s); keeping first",
                               sid, f.name)
                continue
            combined[sid] = rows
    logger.info("parsed %d sequence result(s) from %d file(s)",
                len(combined), len(files))
    return combined


# --------------------------------------------------------------------------
# record assembly + merge
# --------------------------------------------------------------------------
def build_prediction(
    sample_id: str,
    rows,
    kept_positions: Optional[list],
    *,
    threshold: float,
    raw_output_dir: Optional[str] = None,
):
    # type: (...) -> ToolPrediction
    """Remap submitted positions -> original and build the ToolPrediction.

    ``rows`` = ``[(submitted_pos, prob, binary), ...]``. submitted_pos ``k``
    (1-based) -> original ``kept_positions[k-1]`` when the map is available; if
    not, submitted_pos is used as-is (assumes no residues were stripped)."""
    per_res = {}
    binding = []
    n_unmapped = 0
    for sub_pos, prob, binary in rows:
        if sub_pos is None or sub_pos < 1:
            continue
        if kept_positions is not None:
            if sub_pos > len(kept_positions):
                n_unmapped += 1
                continue
            orig = int(kept_positions[sub_pos - 1])
        else:
            orig = int(sub_pos)
        if orig < 1:
            continue
        if prob is not None:
            per_res[orig] = round(float(min(1.0, max(0.0, prob))), 4)
        is_bind = (binary == 1) if binary is not None else (
            prob is not None and prob > threshold)
        if is_bind:
            binding.append(orig)
    if n_unmapped:
        logger.warning("%s: %d result rows beyond submitted length (ignored)",
                       sample_id, n_unmapped)

    return ToolPrediction(
        tool_id=TOOL_ID,
        category=CATEGORY,
        sample_id=sample_id,
        success=True,
        binding_protein_residues=sorted(set(binding)),
        per_residue_confidence=per_res or None,
        raw_output_dir=raw_output_dir,
    )


def merge_into_set(step4_dir: Path, sample_id: str, prediction) -> Path:
    """Insert ``prediction`` into ``<step4_dir>/<sample_id>.jsonl``, dropping any
    prior ``rnabindrplus`` entry and preserving other tools."""
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


def run_inspect(path: Path) -> int:
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8", errors="replace")
    print(f"file   : {path} ({len(text)} chars)")
    print("--- first 18 lines ---")
    for ln in text.splitlines()[:18]:
        print(ln)
    combined = parse_combined_file(text)
    print(f"--- parsed {len(combined)} sequence block(s)")
    for sid in list(combined)[:3]:
        rows = combined[sid]
        print(f"    {sid}: {len(rows)} residues; first 5 (pos,prob,bin): {rows[:5]}")
    return 0


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inspect", nargs="?", const=None, type=Path, default=False,
                   metavar="RESULT_FILE",
                   help="dump a combined result file + the blocks parsed, then exit")
    p.add_argument("--results-dir", type=Path, default=None,
                   help="dir with downloaded RNABindRPlus 'finalpredictions' files")
    p.add_argument("--results-glob", default="*finalpredictions*.txt",
                   help="glob for the combined result files under --results-dir")
    p.add_argument("--step4-dir", type=Path, default=None,
                   help="step-4 JSONL dir to merge into")
    p.add_argument("--processed-dir", type=Path, default=None,
                   help="step-1 processed dir (samples/ used for index fallback)")
    p.add_argument("--samples-dir", type=Path, default=None)
    p.add_argument("--index-map", type=Path,
                   default=Path("data/batch_test_v7/rnabindrplus_index_map.json"),
                   help="submitted->original position map from rnabindrplus_inputs.py")
    p.add_argument("--sample-list", type=Path, default=None,
                   help="txt/csv of sample ids; default: every parsed sequence")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="prob cutoff for binding when no binary value is present")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level)

    # --inspect FILE  (argparse: False=not given, None=given w/o value, Path=given)
    if args.inspect is not False:
        if args.inspect is None:
            print("--inspect needs a result file path", file=sys.stderr)
            return 1
        return run_inspect(args.inspect)

    if args.results_dir is None or args.step4_dir is None:
        print("ERROR: --results-dir and --step4-dir are required", file=sys.stderr)
        return 2
    results_dir = args.results_dir.expanduser().resolve()
    step4_dir = args.step4_dir.expanduser().resolve()
    samples_dir = None
    if args.samples_dir is not None:
        samples_dir = args.samples_dir.expanduser().resolve()
    elif args.processed_dir is not None:
        samples_dir = (args.processed_dir.expanduser().resolve() / "samples")

    if not results_dir.is_dir():
        logger.error("results dir not found: %s", results_dir)
        return 1

    index_map = load_index_map(args.index_map)
    combined = load_all_results(results_dir, args.results_glob)
    if not combined:
        logger.error("no RNABindRPlus results parsed under %s (glob %s)",
                     results_dir, args.results_glob)
        return 1

    if args.sample_list is not None:
        samples = [clean_sample_id(s) for s in
                   read_sample_list(args.sample_list.expanduser().resolve())]
    else:
        samples = sorted(combined.keys())

    n_ok = n_missing = n_fail = 0
    total = len(samples)
    for i, sid in enumerate(samples, 1):
        rows = combined.get(sid)
        if not rows:
            n_missing += 1
            logger.warning("[%d/%d] %s missing from results", i, total, sid)
            continue
        try:
            kept = kept_positions_for(sid, index_map, samples_dir)
            if kept is None:
                logger.warning("%s: no index map / sample JSON; using submitted "
                               "positions as-is (only correct if nothing stripped)", sid)
            pred = build_prediction(sid, rows, kept, threshold=args.threshold,
                                    raw_output_dir=str(results_dir))
        except Exception:  # noqa: BLE001
            n_fail += 1
            logger.exception("[%d/%d] %s failed", i, total, sid)
            continue
        out = merge_into_set(step4_dir, sid, pred)
        n_ok += 1
        logger.info("[%d/%d] %s -> %s (%d residues, %d binding)", i, total, sid,
                    out, len(pred.per_residue_confidence or {}),
                    len(pred.binding_protein_residues or []))

    logger.info("done: %d ok, %d missing, %d failed (of %d)",
                n_ok, n_missing, n_fail, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
