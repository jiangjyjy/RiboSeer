"""Shared IO helpers for the tool adapters and the batch drivers.

Every prediction tool in the library consumes the same two things: a
step-1 sample JSON and a protein (or protein + RNA) structure. This module
owns the plumbing that resolves them — including the two quirks that the
deployed hosts kept surfacing:

  * sample ids carrying a BOM or zero-width mark that makes a path that
    exists look like a path that does not;
  * filenames whose case differs from the id (PDB chain casing is
    preserved by RiboSeer but some filesystems lower-case on write).

``extract_protein_chain_pdb`` is re-exported from the P2Rank adapter: the
chain extraction has to produce exactly the same 1-based polymer index
the P2Rank / EquiPNAS / DeepPocket / NucleicNet parsers report, so there is
one implementation rather than one per tool.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Optional

try:
    import gemmi  # type: ignore
except ImportError:  # pragma: no cover - gemmi ships in requirements.txt
    gemmi = None  # type: ignore

__all__ = [
    "clean_sample_id",
    "read_sample_list",
    "load_sample_json",
    "find_raw_structure",
    "extract_protein_chain_pdb",
    "write_failures",
    "_is_protein_residue",
]


# The structure helpers live with the adapters, but this module must stay
# importable *without* importing the adapter package: ``external/*`` imports
# ``tool_io`` while ``adapters/__init__`` is still executing, so a module-level
# import from ``.adapters`` would be circular.
def _is_protein_residue(resname: str) -> bool:
    """True if ``resname`` is a (standard or modified) amino acid."""
    name = (resname or "").strip().upper()
    if not name or gemmi is None:
        return False
    try:
        info = gemmi.find_tabulated_residue(name)
    except Exception:  # noqa: BLE001
        return False
    return bool(info) and info.is_amino_acid()


# BOM + zero-width / directional marks that ``str.strip`` alone keeps.
_INVISIBLE = "﻿​‌‍‎‏⁠"


def clean_sample_id(sample_id: str) -> str:
    """Strip surrounding whitespace and invisible/zero-width marks from an id."""
    return sample_id.strip().strip(_INVISIBLE).strip()


def read_sample_list(path: Path) -> list[str]:
    """Read sample ids from a ``.txt`` (one id per line) or ``.csv`` (a
    ``sample_id`` column) file. ``#`` comments and blanks are skipped; every id
    is run through :func:`clean_sample_id`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"sample-list not found: {path}")
    if path.suffix.lower() == ".csv":
        out: list[str] = []
        # utf-8-sig transparently eats a leading BOM if present.
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "sample_id" not in reader.fieldnames:
                raise ValueError(
                    f"{path} has no 'sample_id' column "
                    f"(found: {reader.fieldnames})"
                )
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


def load_sample_json(samples_dir: Path, sample_id: str) -> dict:
    """Load ``<samples_dir>/<sample_id>.json``.

    Robust to the two failure modes seen on the server (a file that ``ls``
    shows but a script "can't find"):
      1. invisible/zero-width chars in ``sample_id`` -> :func:`clean_sample_id`;
      2. filename case differing from the id -> a stem-wise case-insensitive
         scan.

    On a genuine miss it raises a *diagnostic* ``FileNotFoundError`` that
    reports whether the directory exists, how many JSONs it holds, and the
    nearest same-PDB filenames — enough to pinpoint the cause without another
    round-trip.
    """
    samples_dir = Path(samples_dir)
    sid = clean_sample_id(sample_id)
    # 1. direct hit (the normal path). Also try samples_dir's parent so a user
    #    who points --samples-dir at processed/ (instead of its samples/ child)
    #    still resolves.
    for p in (samples_dir / f"{sid}.json",
              samples_dir / "samples" / f"{sid}.json"):
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    # 2. case-insensitive stem scan.
    if samples_dir.is_dir():
        target = sid.lower()
        for child in samples_dir.iterdir():
            if child.suffix == ".json" and child.stem.lower() == target:
                return json.loads(child.read_text(encoding="utf-8"))

    # 3. diagnostic failure.
    direct = samples_dir / f"{sid}.json"
    dir_ok = samples_dir.is_dir()
    n_json = sum(1 for _ in samples_dir.glob("*.json")) if dir_ok else -1
    near = sorted(p.name for p in samples_dir.glob(f"{sid[:4]}*"))[:8] if dir_ok else []
    raise FileNotFoundError(
        f"no sample JSON for {sid!r} under {str(samples_dir)!r} "
        f"(dir_exists={dir_ok}, json_files={n_json}, "
        f"os.path.exists(direct)={os.path.exists(direct)}, "
        f"near_matches={near})"
    )


def find_raw_structure(raw_dir: Path, source_pdb: str) -> Optional[Path]:
    """Locate ``<source_pdb>.{pdb,cif}[.gz]`` under ``raw_dir``
    (case-insensitive fallback)."""
    from .adapters.p2rank_adapter import _find_raw_structure  # noqa: PLC0415

    return _find_raw_structure(Path(raw_dir), source_pdb)


def extract_protein_chain_pdb(
    raw_path: Path, chain_id: str, out_path: Path,
) -> Path:
    """Write a clean single-chain protein PDB renumbered to 1-based polymer
    index (chain renamed to ``A``); returns the written path.

    Delegates to the P2Rank adapter, which owns the index convention every
    structure-consuming parser relies on, so there is exactly one
    implementation.
    """
    from .adapters.p2rank_adapter import (  # noqa: PLC0415
        extract_protein_chain_pdb as _extract,
    )

    return _extract(Path(raw_path), chain_id, Path(out_path))


def write_failures(failures: list[dict], path: Path) -> None:
    """Append per-sample failure records to a JSONL log."""
    if not failures:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in failures:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
