#!/usr/bin/env python3
"""Parse DeepPocket output into step-4 ``<sample_id>.jsonl`` (Category B).

DeepPocket emits pocket-level results; we convert them to a per-residue binding
score the same way the P2Rank adapter does (Cat B): each residue gets the
confidence of the best pocket it belongs to.

Per sample, DeepPocket writes (under ``<results>/<sample_id>/``)::

    <sid>_nowat_out/pockets/
        pocket{K}_atm.pdb            fpocket's atoms for candidate pocket K
        bary_centers.txt             K  x y z          (fpocket order)
        bary_centers_ranked.types    K  x y z  <gpath> (CONFIDENCE order)
        bary_centers_confidence.txt  "[c0, c1, ...]"   (CONFIDENCE order)

The crucial detail (verified against ``devalab/DeepPocket@master``): the
confidence file is the *Python-list repr* of the CNN probabilities sorted
**descending**, and ``bary_centers_ranked.types`` is the matching reorder of the
candidate lines — so line ``i`` of ranked.types carries, as its first token, the
**fpocket pocket id** of the ``i``-th most confident pocket, whose score is
``confidence[i]``. We therefore pair ``confidence[i]`` with
``pocket{first_token_of_ranked_line_i}_atm.pdb`` rather than naively assuming
``pocket{i+1}_atm.pdb`` (fpocket's numbering is *not* the confidence order).

``clean_pdb`` (DeepPocket) only drops het/non-standard residues without
renumbering, so the residue numbers in ``pocket{K}_atm.pdb`` equal the numbering
of the single-chain PDB the batch script wrote (1-based polymer / label_seq),
which lines up with step-1 ``binding_protein_residues`` directly — no remap.

Per residue: ``score = max(confidence over pockets containing it)``; residues in
no pocket are simply absent (treated as 0 downstream, like P2Rank). The result
is merged into ``<step4-dir>/<sample_id>.jsonl`` as a
``ToolPrediction(tool_id="deeppocket", category="B")`` carrying ``pockets``,
``per_residue_confidence`` and ``binding_protein_residues``, replacing any prior
``deeppocket`` entry and leaving other tools untouched.

Usage
-----
::

    python -m step4_tool_adapters.external.deeppocket_parse.py \
        --results-dir   data/batch_test_v7/deeppocket_outputs \
        --step4-dir     data/batch_test_v7/step4 \
        --processed-dir data/processed_quality \
        --sample-list   data/processed_quality/splits_tmscore_035/test.txt
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

# Make repo root + src importable for the schemas and shared helpers.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from step4_tool_adapters.schemas import (  # noqa: E402
    Pocket,
    ToolPrediction,
    ToolPredictionSet,
)
from step4_tool_adapters.tool_io import (  # noqa: E402
    clean_sample_id,
    read_sample_list,
)

logger = logging.getLogger("deeppocket_parse")

TOOL_ID = "deeppocket"
CATEGORY = "B"

# A float (incl. scientific notation and bare-integer mantissa like "1e-05").
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


# --------------------------------------------------------------------------
# locating + parsing DeepPocket output
# --------------------------------------------------------------------------
def find_pockets_dir(sample_results_dir: Path) -> Optional[Path]:
    """Locate the ``*_out/pockets`` dir holding ``bary_centers_confidence.txt``.

    Primary name is ``<sid>_nowat_out/pockets``; we glob to stay robust to
    DeepPocket version differences in the ``_out`` infix.
    """
    if not sample_results_dir.is_dir():
        return None
    for cand in sorted(sample_results_dir.glob("*_out/pockets")):
        if (cand / "bary_centers_confidence.txt").is_file():
            return cand
    return None


def parse_confidences(path: Path) -> list[float]:
    """Parse ``bary_centers_confidence.txt`` (a ``str([...])`` list repr) into a
    list of floats in confidence-descending order. Values are clamped to [0,1]."""
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    out: list[float] = []
    for tok in _FLOAT_RE.findall(text):
        try:
            v = float(tok)
        except ValueError:
            continue
        out.append(min(1.0, max(0.0, v)))
    return out


def parse_ranked_pocket_ids(path: Path) -> list[int]:
    """Parse ``bary_centers_ranked.types`` -> fpocket pocket ids in confidence
    order (the first whitespace token of each non-empty line)."""
    ids: list[int] = []
    if not path.is_file():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        first = s.split()[0]
        try:
            ids.append(int(float(first)))  # tolerate "3" or "3.0"
        except ValueError:
            continue
    return ids


def read_pocket_residues(pdb_path: Path) -> list[int]:
    """Residue numbers (sorted, >=1, unique) from an fpocket ``pocket*_atm.pdb``.

    Reads the resSeq field (PDB cols 23-26) of ATOM/HETATM records directly —
    fpocket's pocket atom files are plain ATOM records, and a manual parse keeps
    this script free of a structure-library dependency.
    """
    res: set[int] = set()
    if not pdb_path.is_file():
        return []
    for line in pdb_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                num = int(line[22:26])
            except ValueError:
                continue
            if num >= 1:
                res.add(num)
    return sorted(res)


# --------------------------------------------------------------------------
# pocket -> per residue
# --------------------------------------------------------------------------
def build_prediction(
    sample_id: str,
    pockets_dir: Path,
    *,
    threshold: float,
    max_residue: Optional[int] = None,
) -> ToolPrediction:
    """Assemble the ``deeppocket`` ToolPrediction for one sample.

    Pairs ``confidence[i]`` with the fpocket pocket id from ranked.types line
    ``i`` and reads that pocket's residues from ``pocket{id}_atm.pdb``. Builds
    the Cat-B ``pockets`` list (rank 1 = most confident), the per-residue map
    (max confidence over containing pockets) and the thresholded binding set.
    """
    conf_path = pockets_dir / "bary_centers_confidence.txt"
    ranked_path = pockets_dir / "bary_centers_ranked.types"

    confidences = parse_confidences(conf_path)
    pocket_ids = parse_ranked_pocket_ids(ranked_path)

    n = min(len(confidences), len(pocket_ids))
    if len(confidences) != len(pocket_ids):
        logger.warning(
            "%s: confidence count (%d) != ranked.types count (%d); "
            "pairing first %d", sample_id, len(confidences), len(pocket_ids), n)
    if n == 0:
        raise ValueError(
            f"{sample_id}: no usable confidences/ranked pockets in {pockets_dir}"
        )

    pockets: list[Pocket] = []
    per_res: dict[int, float] = {}
    for rank, (pid, conf) in enumerate(zip(pocket_ids[:n], confidences[:n]), 1):
        residues = read_pocket_residues(pockets_dir / f"pocket{pid}_atm.pdb")
        if max_residue is not None:
            residues = [r for r in residues if r <= max_residue]
        pockets.append(Pocket(rank=rank, score=round(conf, 4), residues=residues))
        for r in residues:
            if conf > per_res.get(r, 0.0):
                per_res[r] = conf

    per_res = {int(k): round(float(v), 4) for k, v in per_res.items()}
    binding = sorted(k for k, v in per_res.items() if v > threshold)

    return ToolPrediction(
        tool_id=TOOL_ID,
        category=CATEGORY,
        sample_id=sample_id,
        success=True,
        binding_protein_residues=binding or None,
        per_residue_confidence=per_res or None,
        pockets=pockets or None,
        raw_output_dir=str(pockets_dir),
    )


# --------------------------------------------------------------------------
# step4 merge (same convention as nucleicnet_parse / parse_af3)
# --------------------------------------------------------------------------
def merge_into_set(
    step4_dir: Path, sample_id: str, prediction: ToolPrediction,
) -> Path:
    """Insert ``prediction`` into ``<step4_dir>/<sample_id>.jsonl``, dropping any
    prior ``deeppocket`` entry and preserving other tools."""
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


def _protein_length(samples_dir: Path, sample_id: str) -> Optional[int]:
    """Best-effort protein length for clipping out-of-range pocket residues."""
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
    threshold: float,
) -> ToolPrediction:
    pockets_dir = find_pockets_dir(results_dir / sample_id)
    if pockets_dir is None:
        raise FileNotFoundError(
            f"no DeepPocket output for {sample_id} under {results_dir / sample_id}"
        )
    max_res = _protein_length(samples_dir, sample_id) if samples_dir else None
    return build_prediction(
        sample_id, pockets_dir, threshold=threshold, max_residue=max_res)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results-dir", required=True, type=Path,
                   help="batch output dir (holds <sample_id>/ folders)")
    p.add_argument("--step4-dir", required=True, type=Path,
                   help="step-4 JSONL dir to merge into")
    p.add_argument("--processed-dir", type=Path, default=None,
                   help="step-1 processed dir; its samples/ is used only to "
                        "clip out-of-range pocket residue indices (optional)")
    p.add_argument("--samples-dir", type=Path, default=None,
                   help="override sample-JSON dir "
                        "(default <processed-dir>/samples)")
    p.add_argument("--sample-list", type=Path, default=None,
                   help="txt/csv of sample ids; default: discover from "
                        "results-dir subfolders")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="confidence cutoff for binding_protein_residues")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level)

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

    if args.sample_list is not None:
        samples = read_sample_list(args.sample_list.expanduser().resolve())
    else:
        samples = sorted(
            c.name for c in results_dir.iterdir()
            if c.is_dir() and find_pockets_dir(c) is not None
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
                threshold=args.threshold)
        except FileNotFoundError as e:
            n_missing += 1
            logger.warning("[%d/%d] %s missing: %s", i, total, sid, e)
            continue
        except Exception:  # noqa: BLE001
            n_fail += 1
            logger.exception("[%d/%d] %s failed", i, total, sid)
            continue
        out = merge_into_set(step4_dir, sid, pred)
        n_ok += 1
        n_pock = len(pred.pockets or [])
        n_bind = len(pred.binding_protein_residues or [])
        logger.info("[%d/%d] %s -> %s (%d pockets, %d binding res)",
                    i, total, sid, out, n_pock, n_bind)

    logger.info("done: %d ok, %d missing, %d failed (of %d)",
                n_ok, n_missing, n_fail, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
