#!/usr/bin/env python3
"""Parse HDOCK docked complexes into step-4 ``<sample_id>.jsonl`` (Category D).

HDOCK is a Category-D rigid-body docking tool, structurally identical to
HADDOCK 3 from Step 4's point of view: ``run_hdock_batch.py`` leaves one
top-ranked complex per sample at ``<results-dir>/<sample_id>/model_1.pdb``
(receptor chain ``A`` = protein, ligand chain ``B`` = RNA). This parser reuses
the exact helpers the live ``Haddock3Adapter`` uses
(``extract_contacts``, ``compute_distance_binding_scores``) and emits the same
``ToolPrediction`` schema.

For each requested sample it:
  1. locates ``model_1.pdb`` (the best docked pose);
  2. computes heavy-atom contacts -> ``binding_protein_residues`` /
     ``binding_rna_nucleotides`` (``extract_contacts``);
  3. computes the per-residue CA->RNA distance binding probability
     (``compute_distance_binding_scores``) -> ``per_residue_pae_score``;
  4. builds a ``ToolPrediction(tool_id="hdock", category="D")`` and merges it
     into ``<step4-dir>/<sample_id>.jsonl`` (a ``ToolPredictionSet``), replacing
     any pre-existing ``hdock`` entry while leaving other tools untouched.

Residue numbering / GT alignment
--------------------------------
Both docking inputs were renumbered to 1-based label_seq before docking
(receptor reused from ``graphbind_inputs`` / freshly extracted; RNA ligand
extracted to chain ``B``), and HDOCK preserves the input numbering in the
docked pose. So the emitted indices line up with step-1
``binding_protein_residues`` / ``binding_rna_nucleotides`` directly — **no
remap**, exactly like the HADDOCK 3 adapter. Like ``graphbind_parse``
we *warn* (but never drop/remap) if a protein residue index falls outside
``[1, protein.length]`` — the tell-tale sign the input numbering broke.

``per_residue_confidence`` stays ``None``: HDOCK reports no pLDDT-equivalent, so
the distance-derived signal is carried only on ``per_residue_pae_score`` (the
same field Boltz-2 / Chai-1 / HADDOCK 3 use), and the noisy-OR fusion path does
not double-count a confidence the tool never produced.

Usage
-----
::

    python -m step4_tool_adapters.external.hdock_parse.py \
        --results-dir   data/batch_test_v7/hdock_outputs \
        --step4-dir     data/batch_test_v7/step4 \
        --processed-dir data/processed_quality \
        --sample-list   data/processed_quality/splits_tmscore_035/test.txt

See also
--------
scripts/riboseer/run_hdock_batch.py : produces the model_1.pdb this parser reads.
src/step4_tool_adapters/adapters/haddock3_adapter.py : the Cat-D template.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Make repo root + src importable for the schemas and shared helpers.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from step4_tool_adapters.contact_extractor import extract_contacts  # noqa: E402
from step4_tool_adapters.schemas import (  # noqa: E402
    ToolPrediction,
    ToolPredictionSet,
)
from step4_tool_adapters.adapters.structure_utils import (  # noqa: E402
    compute_distance_binding_scores,
)
from step4_tool_adapters.tool_io import (  # noqa: E402
    clean_sample_id,
    read_sample_list,
)

logger = logging.getLogger("hdock_parse")

TOOL_ID = "hdock"
CATEGORY = "D"
PROTEIN_CHAIN = "A"  # receptor chain written by run_hdock_batch
RNA_CHAIN = "B"      # ligand chain written by run_hdock_batch

_DEFAULT_MODEL = "model_1.pdb"


# --------------------------------------------------------------------------
# locating the best model
# --------------------------------------------------------------------------
def find_best_model(sample_results_dir: Path, model_name: str) -> Optional[Path]:
    """Locate the top-ranked docked complex for one sample.

    Prefers the canonical ``model_1.pdb``; falls back to the
    lowest-numbered ``model_*.pdb`` to tolerate naming differences.
    """
    if not sample_results_dir.is_dir():
        return None
    direct = sample_results_dir / model_name
    if direct.is_file():
        return direct
    hits = sorted(
        sample_results_dir.glob("model_*.pdb"),
        key=lambda p: (len(p.stem), p.stem),  # model_1 before model_10
    )
    return hits[0] if hits else None


# --------------------------------------------------------------------------
# record assembly
# --------------------------------------------------------------------------
def build_prediction(
    model: Path,
    sample_id: str,
    *,
    contact_cutoff: float,
    distance_scale: float,
) -> ToolPrediction:
    """Build the ``hdock`` ToolPrediction from one docked complex.

    Mirrors ``Haddock3Adapter.parse_output`` (HDOCK is the same Cat-D shape):
    contacts + CA->RNA distance probability, no remap. Chain hints ``A``/``B``
    are passed but both helpers fall back to content-based chain selection, so
    a relabelled pose is still handled correctly.
    """
    contacts = extract_contacts(
        model, cutoff=contact_cutoff,
        protein_chain_id=PROTEIN_CHAIN, rna_chain_id=RNA_CHAIN,
    )
    per_res_bind = compute_distance_binding_scores(
        model, protein_chain_id=PROTEIN_CHAIN, rna_chain_id=RNA_CHAIN,
        distance_scale=distance_scale,
    )

    binding_prot = sorted(contacts.binding_protein_residues)
    binding_rna = sorted(contacts.binding_rna_nucleotides)

    return ToolPrediction(
        tool_id=TOOL_ID,
        category=CATEGORY,
        sample_id=sample_id,
        success=True,
        binding_protein_residues=binding_prot or None,
        binding_rna_nucleotides=binding_rna or None,
        per_residue_confidence=None,                 # HDOCK has no pLDDT
        per_residue_pae_score=per_res_bind or None,  # CA->RNA distance prob
        predicted_structure_path=str(Path(model).resolve()),
        raw_output_dir=str(Path(model).parent),
    )


def merge_into_set(
    step4_dir: Path, sample_id: str, prediction: ToolPrediction,
) -> Path:
    """Insert ``prediction`` into ``<step4_dir>/<sample_id>.jsonl``, dropping any
    prior ``hdock`` entry and preserving other tools (same convention as the
    sibling parse_* scripts)."""
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
                    pass  # keep a fresh set on unparsable content
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
    model_name: str,
    contact_cutoff: float,
    distance_scale: float,
) -> ToolPrediction:
    model = find_best_model(results_dir / sample_id, model_name)
    if model is None:
        raise FileNotFoundError(
            f"no docked model ({model_name}) for {sample_id} under "
            f"{results_dir / sample_id}"
        )
    pred = build_prediction(
        model, sample_id,
        contact_cutoff=contact_cutoff, distance_scale=distance_scale,
    )

    # Alignment sanity check (warn only — do not remap/drop).
    length = _protein_length(samples_dir, sample_id)
    if length is not None and pred.binding_protein_residues:
        oob = [r for r in pred.binding_protein_residues if r < 1 or r > length]
        if oob:
            logger.warning(
                "%s: %d binding residue(s) outside [1, %d] (e.g. %s); HDOCK "
                "input numbering may not match the ground-truth index space",
                sample_id, len(oob), length, sorted(oob)[:5])
    return pred


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results-dir", required=True, type=Path,
                   help="HDOCK output dir (holds <sample_id>/model_1.pdb)")
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
    p.add_argument("--model-name", default=_DEFAULT_MODEL,
                   help=f"docked-model filename to parse (default {_DEFAULT_MODEL})")
    p.add_argument("--contact-cutoff", type=float, default=4.5,
                   help="heavy-atom contact distance (A) for binding residues")
    p.add_argument("--distance-scale", type=float, default=8.0,
                   help="scale for the CA->RNA distance binding probability")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

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
            if c.is_dir() and find_best_model(c, args.model_name) is not None
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
                model_name=args.model_name,
                contact_cutoff=args.contact_cutoff,
                distance_scale=args.distance_scale)
        except FileNotFoundError as e:
            n_missing += 1
            logger.warning("[%d/%d] %s missing: %s", i, total, sid, e)
            continue
        except Exception:  # noqa: BLE001
            n_fail += 1
            logger.exception("[%d/%d] %s failed", i, total, sid)
            continue
        out = merge_into_set(step4_dir, sid, pred)
        n_bind = len(pred.binding_protein_residues or [])
        n_res = len(pred.per_residue_pae_score or {})
        n_ok += 1
        logger.info("[%d/%d] %s -> %s (%d binding, %d per-residue)",
                    i, total, sid, out, n_bind, n_res)

    logger.info("done: %d ok, %d missing, %d failed (of %d)",
                n_ok, n_missing, n_fail, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
