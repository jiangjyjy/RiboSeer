#!/usr/bin/env python3
"""Parse AlphaFold 3 Server result bundles into Step-4 ``<sample_id>.jsonl``.

AlphaFold 3 is a Category-A (co-folding) tool, structurally identical to
Boltz-2 / Chai-1 from Step 4's point of view: it emits a predicted mmCIF
complex with per-residue pLDDT in the B-factor column plus a confidence JSON.
This parser therefore reuses the exact helpers the live ``Boltz2Adapter`` uses
(``extract_contacts``, ``read_protein_plddt_from_cif``,
``compute_distance_binding_scores``, ``remap_indices`` / ``remap_per_residue``)
and emits the same ``ToolPrediction`` schema — only the on-disk file naming and
the confidence-JSON keys differ.

AF3-Server layout (one folder per job; folder name == ``sample_id``)::

    <sample_id>/
        fold_<sample_id>_model_0.cif                <- ranked-0 complex
        fold_<sample_id>_summary_confidences_0.json <- ptm / iptm / ranking_score
        fold_<sample_id>_full_data_0.json
        msas/  templates/  ...

For each requested sample this script:
  1. maps the (possibly mixed-case) ``sample_id`` to its lower-cased folder,
  2. computes heavy-atom contacts -> ``binding_protein_residues`` /
     ``binding_rna_nucleotides`` (``extract_contacts``, chains A=protein,
     B=RNA — the AF3-Server convention),
  3. computes the per-residue CA->RNA distance binding probability
     (``compute_distance_binding_scores``) -> ``per_residue_pae_score``,
  4. reads per-residue pLDDT from the CIF B-factors
     (``read_protein_plddt_from_cif``) -> ``per_residue_confidence``,
  5. optionally remaps cleaned 1-based indices back to the dataset's original
     numbering via the sample's ``protein_index_map`` (same as Boltz-2),
  6. pulls ``iptm`` / ``ptm`` from ``summary_confidences_0.json``
     (``iptm`` -> ``iptm_score``; ``plddt_mean`` from the per-residue mean),
  7. builds a ``ToolPrediction(tool_id="alphafold3", category="A")`` and merges
     it into ``<step4-dir>/<sample_id>.jsonl`` (a ``ToolPredictionSet``),
     replacing any pre-existing ``alphafold3`` entry while leaving other tools'
     predictions untouched.

Usage
-----
::

    python -m step4_tool_adapters.external.af3_parse.py \
        --af3-dir data/batch_test_v7/af3_structures/ \
        --step4-dir data/batch_test_v7/step4/ \
        --processed-dir data/processed_quality \
        --sample-list data/processed_quality/splits_tmscore_035/test.txt

See also
--------
src/step4_tool_adapters/adapters/boltz2_adapter.py : the Category-A template
reparse_haddock3.py : sibling backfill script (same step4 JSONL convention)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Make both the repo root and ``src`` importable when run as a script.
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
from step4_tool_adapters.adapters.boltz2_adapter import (  # noqa: E402
    read_protein_plddt_from_cif,
    remap_indices,
    remap_per_residue,
)

try:  # optional: supplies protein_index_map for cleaned->original remap
    from step5_fusion.data_collector import load_sample_json
except Exception:  # noqa: BLE001
    load_sample_json = None  # type: ignore

logger = logging.getLogger("af3_parse")

TOOL_ID = "alphafold3"
CATEGORY = "A"
PROTEIN_CHAIN = "A"  # AF3-Server assigns chains in job-request order:
RNA_CHAIN = "B"      # protein first (A), RNA second (B).


# --------------------------------------------------------------------------
# sample / folder discovery
# --------------------------------------------------------------------------
def _read_sample_list(path: Optional[Path]) -> Optional[list[str]]:
    """Read newline-delimited sample ids (first whitespace token per line),
    skipping blanks and ``#`` comments. ``None`` -> caller discovers folders."""
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"sample-list not found: {path}")
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s.split()[0])
    return out


def _is_job_folder(path: Path) -> bool:
    """A job folder is a directory holding a ranked-0 model CIF."""
    return path.is_dir() and any(path.glob("*_model_0.cif"))


def _discover_sample_ids(af3_dir: Path) -> list[str]:
    """Discover sample ids as the names of AF3 job folders under ``af3_dir``."""
    return sorted(c.name for c in af3_dir.iterdir() if _is_job_folder(c))


def _resolve_sample_dir(af3_dir: Path, sample_id: str) -> Optional[Path]:
    """Map a (possibly mixed-case) ``sample_id`` to its on-disk job folder.

    AF3 Server lower-cases folder names while RiboSeer sample ids keep the
    original PDB chain casing (``8z4l_C_M`` -> folder ``8z4l_c_m``). Try the
    exact name, the lower-cased name, then a case-insensitive scan.
    """
    for cand in (af3_dir / sample_id, af3_dir / sample_id.lower()):
        if cand.is_dir():
            return cand
    target = sample_id.lower()
    for child in af3_dir.iterdir():
        if child.is_dir() and child.name.lower() == target:
            return child
    return None


def _find_one(sample_dir: Path, patterns: tuple[str, ...]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(sample_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def _find_structure(sample_dir: Path, sample_id: str) -> Optional[Path]:
    """Locate the ranked-0 predicted CIF inside an AF3 job folder."""
    return _find_one(
        sample_dir,
        (
            f"fold_{sample_id}_model_0.cif",
            f"fold_{sample_id.lower()}_model_0.cif",
            "*_model_0.cif",
            "*.cif",
        ),
    )


def _read_summary(sample_dir: Path, sample_id: str) -> dict:
    """Read the ranked-0 summary-confidences JSON (ptm / iptm / ranking_score).
    Returns ``{}`` when absent or unparsable."""
    path = _find_one(
        sample_dir,
        (
            f"fold_{sample_id}_summary_confidences_0.json",
            f"fold_{sample_id.lower()}_summary_confidences_0.json",
            "*summary_confidences_0.json",
            "*summary_confidences*.json",
        ),
    )
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # drop NaN


# --------------------------------------------------------------------------
# record assembly
# --------------------------------------------------------------------------
def build_prediction(
    cif: Path,
    sample: dict,
    *,
    contact_cutoff: float,
    distance_scale: float,
) -> ToolPrediction:
    """Build the ``alphafold3`` ToolPrediction from one predicted complex.

    Mirrors ``Boltz2Adapter.parse_output`` (AF3 is the same Cat-A shape):
    reuses the shared distance/contact/pLDDT helpers and the cleaned->original
    index remap. When the sample carries no ``protein_index_map`` the residue
    indices stay in the predicted CIF's 1-based numbering.
    """
    sample_id = sample["sample_id"]

    contacts = extract_contacts(
        cif, cutoff=contact_cutoff,
        protein_chain_id=PROTEIN_CHAIN, rna_chain_id=RNA_CHAIN,
    )
    per_res_plddt = read_protein_plddt_from_cif(cif, chain_id=PROTEIN_CHAIN)
    per_res_bind = compute_distance_binding_scores(
        cif, protein_chain_id=PROTEIN_CHAIN, rna_chain_id=RNA_CHAIN,
        distance_scale=distance_scale,
    )

    # Cleaned -> original numbering, if the processed sample provides the maps.
    pmap = sample.get("protein_index_map") or {}
    rmap = sample.get("rna_index_map") or {}
    if pmap:
        pmap = {int(k): int(v) for k, v in pmap.items()}
        binding_prot = remap_indices(contacts.binding_protein_residues, pmap)
        per_res_plddt = remap_per_residue(per_res_plddt, pmap)
        per_res_bind = remap_per_residue(per_res_bind, pmap)
    else:
        binding_prot = sorted(contacts.binding_protein_residues)
    if rmap:
        rmap = {int(k): int(v) for k, v in rmap.items()}
        binding_rna = remap_indices(contacts.binding_rna_nucleotides, rmap)
    else:
        binding_rna = sorted(contacts.binding_rna_nucleotides)

    summary = _read_summary(cif.parent, sample_id)
    iptm_score = _safe_float(summary.get("iptm"))
    if iptm_score is not None:
        iptm_score = max(0.0, min(1.0, iptm_score))

    plddt_mean = None
    if per_res_plddt:
        plddt_mean = max(
            0.0, min(100.0, round(sum(per_res_plddt.values()) / len(per_res_plddt), 3))
        )

    return ToolPrediction(
        tool_id=TOOL_ID,
        category=CATEGORY,
        sample_id=sample_id,
        success=True,
        binding_protein_residues=binding_prot or None,
        binding_rna_nucleotides=binding_rna or None,
        per_residue_confidence=per_res_plddt or None,   # pLDDT
        per_residue_pae_score=per_res_bind or None,     # CA->RNA distance prob
        predicted_structure_path=str(Path(cif).resolve()),
        plddt_mean=plddt_mean,
        iptm_score=iptm_score,
        raw_output_dir=str(cif.parent),
    )


def merge_into_set(
    step4_dir: Path, sample_id: str, prediction: ToolPrediction
) -> Path:
    """Insert ``prediction`` into ``<step4_dir>/<sample_id>.jsonl``.

    Loads an existing ``ToolPredictionSet`` if present, drops any prior
    ``alphafold3`` entry, appends the new one, refreshes ``tools_run``, and
    rewrites the file atomically. Other tools' predictions are preserved.
    """
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


def _load_sample(processed_dir: Path, sample_id: str) -> dict:
    """Load the Step-3 sample dict (for ``protein_index_map``); fall back to a
    minimal dict so the parser still runs without processed metadata."""
    if load_sample_json is not None:
        try:
            sample = load_sample_json(processed_dir, sample_id)
        except Exception:  # noqa: BLE001
            sample = None
        if sample:
            sample.setdefault("sample_id", sample_id)
            return sample
    return {"sample_id": sample_id}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--af3-dir", required=True, type=Path)
    p.add_argument("--step4-dir", required=True, type=Path)
    p.add_argument("--processed-dir", required=True, type=Path)
    p.add_argument("--sample-list", type=Path, default=None)
    p.add_argument("--contact-cutoff", type=float, default=4.5,
                   help="heavy-atom contact distance (A) for binding residues")
    p.add_argument("--distance-scale", type=float, default=8.0,
                   help="scale for the CA->RNA distance binding probability")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level)

    if not args.af3_dir.is_dir():
        logger.error("--af3-dir not a directory: %s", args.af3_dir)
        return 1

    samples = _read_sample_list(args.sample_list)
    if samples is None:
        samples = _discover_sample_ids(args.af3_dir)

    n_ok = n_missing = n_fail = 0
    for sample_id in samples:
        sample_dir = _resolve_sample_dir(args.af3_dir, sample_id)
        if sample_dir is None:
            n_missing += 1
            logger.warning("no AF3 folder for sample: %s", sample_id)
            continue
        cif = _find_structure(sample_dir, sample_id)
        if cif is None:
            n_missing += 1
            logger.warning("no model CIF in %s", sample_dir)
            continue
        sample = _load_sample(args.processed_dir, sample_id)
        try:
            prediction = build_prediction(
                cif, sample,
                contact_cutoff=args.contact_cutoff,
                distance_scale=args.distance_scale,
            )
            out = merge_into_set(args.step4_dir, sample_id, prediction)
        except Exception:  # noqa: BLE001
            n_fail += 1
            logger.exception("failed to parse %s", sample_id)
            continue
        n_ok += 1
        logger.info("[ok] %s -> %s", sample_id, out)

    logger.info(
        "done: %d ok, %d missing, %d failed (of %d requested)",
        n_ok, n_missing, n_fail, len(samples),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
