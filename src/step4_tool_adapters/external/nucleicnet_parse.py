#!/usr/bin/env python3
"""Parse NucleicNet SXPR per-voxel output into step-4 ``<sample_id>.jsonl``.

NucleicNet (Cat C) does not score residues directly — it classifies a halo of
3D grid points (voxels) around the protein surface into
``{Base / Phosphate / Ribose / Nonsite}``. The model-averaged result lives in
``<out>/<sample_id>/sxpr/Result_EnsembleAvDf.pkl`` (a pandas DataFrame, one row
per voxel) with these relevant columns (verified against
``jhmlam/NucleicNet@master`` ``commandServer.py``):

    x, y, z                      voxel coordinates (Angstrom)
    Raw_0 .. Raw_3               raw per-class probabilities (sum to 1)
    Smoothened_0 .. Smoothened_3 spatially-smoothed per-class probabilities
    Pdbid, HaloIdx               provenance keys

The integer class index ``0..3`` maps to a name via the sibling
``sxpr/ClassName_ClassIndex_Dict.pkl`` (a ``{name: index}`` dict, e.g.
``{'Base': 0, 'P': 1, 'R': 2, 'Nonsite': 3}`` — order is not fixed, which is
exactly why we read the dict rather than hard-coding it).

Per-voxel binding signal
------------------------
A voxel's "is this a binding site" probability is ``1 - P(Nonsite)`` (the union
of Base + Phosphate + Ribose), taken from the ``Smoothened`` columns by default
(the column family NucleicNet's own visualisation pipeline uses).

Voxel -> residue
----------------
Each voxel is assigned to its nearest protein heavy atom (within ``--radius``,
default 6.0 A — wide enough to capture the whole 2.5-5.0 A halo shell). A
residue's score is the **max** (``--agg``, default) site probability over the
voxels assigned to it; residues with no nearby voxel (fully buried) score 0.0.
This mirrors NucleicNet notebook 07's "count site voxels per atom, take max per
residue" recipe, expressed as a probability rather than a raw count.

The protein coordinates / residue numbering come from the single-chain PDB this
sample's batch run wrote to ``--inputs-dir`` (renumbered to 1-based polymer
index, chain ``A``). NucleicNet's sanitiser never moves atoms, so the voxels sit
in that same frame and the residue indices we emit line up with step-1
``binding_protein_residues`` directly.

The result is merged into ``<step4-dir>/<sample_id>.jsonl`` as a
``ToolPrediction(tool_id="nucleicnet", category="C")``, replacing any prior
``nucleicnet`` entry and leaving other tools untouched (same convention as
``af3_parse.py`` / ``reparse_haddock3.py``).

Usage
-----
::

    # one-off schema check on a single produced pkl (no writes)
    python -m step4_tool_adapters.external.nucleicnet_parse.py --inspect \
        --output-dir data/batch_test_v7/nucleicnet --sample-id 3wbm_A_X

    # parse the whole split
    python -m step4_tool_adapters.external.nucleicnet_parse.py \
        --output-dir data/batch_test_v7/nucleicnet \
        --inputs-dir data/batch_test_v7/nucleicnet/nucleicnet_inputs \
        --step4-dir  data/batch_test_v7/step4 \
        --sample-list data/batch_test_v7/af3_submission_order.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Make repo root + src importable when run as a script (for the schemas).
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from step4_tool_adapters.schemas import (  # noqa: E402
    ToolPrediction,
    ToolPredictionSet,
)

logger = logging.getLogger("nucleicnet_parse")

TOOL_ID = "nucleicnet"
CATEGORY = "C"


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
# Zero-width / directional marks that survive str.strip() and corrupt ids
# copied from a list saved with a BOM or edited cross-OS.
_INVISIBLE = "﻿​‌‍‎‏⁠"


def clean_sample_id(sample_id: str) -> str:
    """Strip surrounding whitespace and invisible/zero-width marks from an id."""
    return sample_id.strip().strip(_INVISIBLE).strip()


def read_sample_list(path: Path) -> list[str]:
    """Read sample ids from a ``.txt`` (one id/line) or ``.csv`` (``sample_id``
    column). Shared convention with run_nucleicnet_batch.py."""
    if not path.is_file():
        raise FileNotFoundError(f"sample-list not found: {path}")
    if path.suffix.lower() == ".csv":
        out: list[str] = []
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "sample_id" not in reader.fieldnames:
                raise ValueError(f"{path} has no 'sample_id' column")
            for row in reader:
                sid = clean_sample_id(row.get("sample_id") or "")
                if sid:
                    out.append(sid)
        return out
    out = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        s = clean_sample_id(line)
        if s and not s.startswith("#"):
            out.append(clean_sample_id(s.split()[0]))
    return out


def sxpr_dir(output_dir: Path, sample_id: str) -> Path:
    return output_dir / "nucleicnet_outputs" / sample_id / "sxpr"


def load_class_index_dict(sxpr_path: Path) -> dict:
    """Load ``ClassName_ClassIndex_Dict.pkl`` -> ``{class_name: int_index}``."""
    path = sxpr_path / "ClassName_ClassIndex_Dict.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"class-index dict not found: {path}")
    with path.open("rb") as f:
        d = pickle.load(f)
    if not isinstance(d, dict):
        raise ValueError(f"unexpected class-index dict type in {path}: {type(d)}")
    return d


def resolve_nonsite_index(
    class_dict: dict,
    *,
    nonsite_name: str = "Nonsite",
    override: Optional[int] = None,
) -> int:
    """Find the integer column index of the non-site class.

    ``class_dict`` is ``{name: index}``. Match ``nonsite_name`` case-insensitively
    (also tolerating ``non-site`` / ``non_site``). ``override`` short-circuits."""
    if override is not None:
        return int(override)
    norm = {str(k).strip().lower().replace("-", "").replace("_", ""): v
            for k, v in class_dict.items()}
    key = nonsite_name.strip().lower().replace("-", "").replace("_", "")
    if key in norm:
        return int(norm[key])
    raise KeyError(
        f"could not find non-site class {nonsite_name!r} in class dict "
        f"{class_dict}; pass --nonsite-index to override"
    )


def compute_voxel_site_probs(
    df,
    *,
    nonsite_index: int,
    prediction_type: str = "Smoothened",
) -> tuple[np.ndarray, np.ndarray]:
    """From the SXPR DataFrame return ``(coords[N,3], site_prob[N])``.

    ``site_prob = 1 - P(Nonsite)`` over the four ``<prediction_type>_{0..3}``
    columns. ``coords`` are the ``x,y,z`` voxel positions.
    """
    for col in ("x", "y", "z"):
        if col not in df.columns:
            raise KeyError(
                f"SXPR DataFrame missing coordinate column {col!r}; "
                f"columns = {list(df.columns)}"
            )
    nonsite_col = f"{prediction_type}_{nonsite_index}"
    if nonsite_col not in df.columns:
        raise KeyError(
            f"SXPR DataFrame missing {nonsite_col!r}; "
            f"columns = {list(df.columns)}"
        )
    coords = df[["x", "y", "z"]].to_numpy(dtype=float)
    nonsite = df[nonsite_col].to_numpy(dtype=float)
    site = 1.0 - nonsite
    site = np.clip(site, 0.0, 1.0)
    return coords, site


# --------------------------------------------------------------------------
# protein structure
# --------------------------------------------------------------------------
def load_protein_atoms(pdb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read protein heavy-atom coords + their 1-based residue index.

    Returns ``(coords[M,3], resids[M])``. Uses gemmi; hydrogens are skipped.
    The residue index is the PDB ``seqid`` (the batch script wrote label_seq
    1-based numbering), so it matches step-1 ground-truth residue indices.
    """
    import gemmi  # local import: parser may run where the batch deps differ

    if not pdb_path.is_file():
        raise FileNotFoundError(f"input PDB not found: {pdb_path}")
    structure = gemmi.read_structure(str(pdb_path))
    if len(structure) == 0:
        raise ValueError(f"no models in {pdb_path}")
    coords: list[tuple[float, float, float]] = []
    resids: list[int] = []
    for chain in structure[0]:
        for res in chain:
            seqid = int(res.seqid.num)
            if seqid < 1:
                continue
            for atom in res:
                if atom.is_hydrogen():
                    continue
                pos = atom.pos
                coords.append((pos.x, pos.y, pos.z))
                resids.append(seqid)
    if not coords:
        raise ValueError(f"no protein heavy atoms parsed from {pdb_path}")
    return np.asarray(coords, dtype=float), np.asarray(resids, dtype=int)


# --------------------------------------------------------------------------
# voxel -> residue aggregation
# --------------------------------------------------------------------------
def _nearest_atom(voxels: np.ndarray, atoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each voxel return (index of nearest atom, distance to it).

    Uses scipy.cKDTree when available (fast); otherwise a chunked numpy
    brute-force so the parser also runs in envs without scipy (e.g. ``pocket``).
    """
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(atoms)
        dist, idx = tree.query(voxels, k=1)
        return idx.astype(int), dist.astype(float)
    except Exception:  # noqa: BLE001 - scipy missing or query failure -> numpy
        n = voxels.shape[0]
        idx = np.empty(n, dtype=int)
        dist = np.empty(n, dtype=float)
        chunk = 2048
        atoms_sq = np.einsum("ij,ij->i", atoms, atoms)  # |a|^2
        for s in range(0, n, chunk):
            block = voxels[s:s + chunk]
            # squared euclidean dist: |v|^2 + |a|^2 - 2 v.a
            d2 = (np.einsum("ij,ij->i", block, block)[:, None]
                  + atoms_sq[None, :]
                  - 2.0 * block @ atoms.T)
            np.maximum(d2, 0.0, out=d2)
            j = np.argmin(d2, axis=1)
            idx[s:s + chunk] = j
            dist[s:s + chunk] = np.sqrt(d2[np.arange(block.shape[0]), j])
        return idx, dist


def aggregate_per_residue(
    voxel_coords: np.ndarray,
    site_probs: np.ndarray,
    atom_coords: np.ndarray,
    atom_resids: np.ndarray,
    *,
    radius: float = 6.0,
    agg: str = "max",
) -> dict[int, float]:
    """Assign each voxel to its nearest protein residue and aggregate.

    A voxel contributes to a residue only if its nearest atom is within
    ``radius``. Per-residue score = max (``agg='max'``) or mean (``agg='mean'``)
    of the contributing voxels' ``site_probs``. Every residue present in
    ``atom_resids`` appears in the output (score 0.0 if no voxel reached it).
    """
    out: dict[int, float] = {int(r): 0.0 for r in np.unique(atom_resids)}
    if voxel_coords.shape[0] == 0:
        return out

    nearest_idx, nearest_dist = _nearest_atom(voxel_coords, atom_coords)
    within = nearest_dist <= radius
    if not np.any(within):
        return out

    vox_resid = atom_resids[nearest_idx[within]]
    vox_score = site_probs[within]

    # Accumulate per residue.
    buckets: dict[int, list[float]] = {}
    for r, s in zip(vox_resid.tolist(), vox_score.tolist()):
        buckets.setdefault(int(r), []).append(float(s))

    for r, scores in buckets.items():
        if agg == "mean":
            out[r] = float(np.mean(scores))
        else:  # max (default)
            out[r] = float(np.max(scores))
    return out


# --------------------------------------------------------------------------
# record assembly + merge
# --------------------------------------------------------------------------
def build_prediction(
    sample_id: str,
    per_residue: dict[int, float],
    *,
    threshold: float,
    raw_output_dir: Optional[str] = None,
) -> ToolPrediction:
    """Build the ``nucleicnet`` ToolPrediction from per-residue site scores."""
    per_res_round = {int(k): round(float(v), 4) for k, v in per_residue.items()}
    binding = sorted(k for k, v in per_res_round.items() if v > threshold)
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
    """Insert ``prediction`` into ``<step4_dir>/<sample_id>.jsonl``.

    Drops any prior ``nucleicnet`` entry, preserves other tools, refreshes
    ``tools_run`` and rewrites atomically. Same shape as af3_parse."""
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


def parse_one_sample(
    sample_id: str,
    *,
    output_dir: Path,
    inputs_dir: Path,
    prediction_type: str,
    nonsite_name: str,
    nonsite_index_override: Optional[int],
    radius: float,
    agg: str,
    threshold: float,
) -> ToolPrediction:
    """Full per-sample pipeline: pkl + class dict + PDB -> ToolPrediction.

    Raises on missing inputs; the CLI loop catches and tallies."""
    import pandas as pd

    sx = sxpr_dir(output_dir, sample_id)
    pkl = sx / "Result_EnsembleAvDf.pkl"
    if not pkl.is_file():
        raise FileNotFoundError(f"no SXPR result for {sample_id}: {pkl}")
    df = pd.read_pickle(pkl)
    class_dict = load_class_index_dict(sx)
    nonsite_index = resolve_nonsite_index(
        class_dict, nonsite_name=nonsite_name, override=nonsite_index_override)

    voxel_coords, site_probs = compute_voxel_site_probs(
        df, nonsite_index=nonsite_index, prediction_type=prediction_type)

    pdb_path = inputs_dir / f"{sample_id}.pdb"
    atom_coords, atom_resids = load_protein_atoms(pdb_path)

    per_res = aggregate_per_residue(
        voxel_coords, site_probs, atom_coords, atom_resids,
        radius=radius, agg=agg)

    return build_prediction(
        sample_id, per_res, threshold=threshold, raw_output_dir=str(sx))


def run_inspect(output_dir: Path, sample_id: str) -> int:
    """Print one sample's SXPR DataFrame schema + class dict (no writes)."""
    import pandas as pd

    sx = sxpr_dir(output_dir, sample_id)
    pkl = sx / "Result_EnsembleAvDf.pkl"
    print(f"pkl : {pkl}  exists={pkl.is_file()}")
    if not pkl.is_file():
        return 1
    df = pd.read_pickle(pkl)
    print(f"shape   : {df.shape}")
    print(f"columns : {list(df.columns)}")
    print("dtypes  :")
    print(df.dtypes.to_string())
    print("head    :")
    print(df.head(3).to_string())
    try:
        print(f"class dict: {load_class_index_dict(sx)}")
    except Exception as e:  # noqa: BLE001
        print(f"class dict: <error: {e}>")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output-dir", required=True, type=Path,
                   help="base dir holding nucleicnet_outputs/ (and inputs)")
    p.add_argument("--inputs-dir", type=Path, default=None,
                   help="single-chain input PDBs "
                        "(default <output-dir>/nucleicnet_inputs)")
    p.add_argument("--step4-dir", type=Path, default=None,
                   help="step-4 JSONL dir to merge into (required unless --inspect)")
    p.add_argument("--sample-list", type=Path, default=None,
                   help="txt/csv of sample ids; default: discover from "
                        "nucleicnet_outputs/ subdirs")
    p.add_argument("--prediction-type", default="Smoothened",
                   choices=["Smoothened", "Raw"],
                   help="which SXPR probability family to use")
    p.add_argument("--nonsite-name", default="Nonsite",
                   help="class name treated as non-site (site = 1 - this)")
    p.add_argument("--nonsite-index", type=int, default=None,
                   help="override the non-site class column index 0..3")
    p.add_argument("--radius", type=float, default=6.0,
                   help="max voxel->nearest-atom distance (A) to count a voxel")
    p.add_argument("--agg", default="max", choices=["max", "mean"],
                   help="per-residue aggregation over its voxels")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="site-prob cutoff for binding_protein_residues")
    p.add_argument("--inspect", action="store_true",
                   help="dump one sample's DataFrame schema and exit "
                        "(needs --sample-id)")
    p.add_argument("--sample-id", default=None, help="sample id for --inspect")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level)

    output_dir = args.output_dir.resolve()
    if args.inspect:
        if not args.sample_id:
            logger.error("--inspect needs --sample-id")
            return 1
        return run_inspect(output_dir, args.sample_id)

    if args.step4_dir is None:
        logger.error("--step4-dir is required (unless --inspect)")
        return 1
    inputs_dir = (args.inputs_dir or output_dir / "nucleicnet_inputs").resolve()
    step4_dir = args.step4_dir.resolve()

    if args.sample_list is not None:
        samples = read_sample_list(args.sample_list)
    else:
        outs = output_dir / "nucleicnet_outputs"
        samples = sorted(
            c.name for c in outs.iterdir()
            if c.is_dir() and (c / "sxpr" / "Result_EnsembleAvDf.pkl").is_file()
        ) if outs.is_dir() else []
    if not samples:
        logger.error("no samples to parse")
        return 1

    n_ok = n_missing = n_fail = 0
    total = len(samples)
    for i, sample_id in enumerate(samples, 1):
        try:
            pred = parse_one_sample(
                sample_id,
                output_dir=output_dir, inputs_dir=inputs_dir,
                prediction_type=args.prediction_type,
                nonsite_name=args.nonsite_name,
                nonsite_index_override=args.nonsite_index,
                radius=args.radius, agg=args.agg, threshold=args.threshold,
            )
        except FileNotFoundError as e:
            n_missing += 1
            logger.warning("[%d/%d] %s missing: %s", i, total, sample_id, e)
            continue
        except Exception:  # noqa: BLE001
            n_fail += 1
            logger.exception("[%d/%d] %s failed", i, total, sample_id)
            continue
        out = merge_into_set(step4_dir, sample_id, pred)
        n_ok += 1
        n_bind = len(pred.binding_protein_residues or [])
        logger.info("[%d/%d] %s -> %s (%d binding res)",
                    i, total, sample_id, out, n_bind)

    logger.info("done: %d ok, %d missing, %d failed (of %d)",
                n_ok, n_missing, n_fail, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
