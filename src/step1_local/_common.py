"""Shared helpers for Step 1 local scripts.

Kept deliberately small: path fix for running without `conda activate`, the
resolution-based quality tier (stage 1.1 decision), and a residue classifier
used by both scan_dataset.py and extract_pairs.py.
"""

import os
import sys
from pathlib import Path


def fix_windows_dll_path() -> None:
    """Add <env>/Library/bin to PATH so numpy/matplotlib MKL DLLs load.

    On Windows, invoking env python via an absolute path (without `conda activate`)
    leaves the env's native DLL directory off the search path, so any C extension
    that depends on MKL (numpy, matplotlib, pandas) crashes the process with
    exit code 127 before Python can raise an exception. This restores PATH.
    """
    env_bin = Path(sys.executable).parent / "Library" / "bin"
    if env_bin.is_dir() and str(env_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(env_bin) + os.pathsep + os.environ.get("PATH", "")


fix_windows_dll_path()

# gemmi is only needed by classify_residue() and the chain-polymer helpers
# below. We defer the import so lightweight consumers (e.g. validate.py on
# the server, which only needs sid_to_filename) don't require gemmi to be
# installed. Importers that call classify_residue / classify_chain_dominant
# will hit the import inside those functions.


STANDARD_RNA = {"A", "U", "G", "C"}
STANDARD_DNA = {"DA", "DT", "DG", "DC"}
STANDARD_AA_THREE = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def quality_tier(resolution, exp_method: str) -> str:
    """Classify a sample by experimental quality into a soft tier.

    Returns one of: 'strict', 'standard', 'low', 'discard'.

    Thresholds (see PROJECT_LOG.md key decisions for rationale):
      - strict:   resolution ≤ 3.5 Å   (side-chain rotamers trustworthy)
      - standard: 3.5 < res ≤ 4.0 Å    (backbone/phosphate reliable)
      - low:      4.0 < res ≤ 6.0 Å    (architectural; contact detection only)
      - discard:  res > 6.0 Å, NMR, or no resolution at all

    NMR is always discarded — ensemble-based distances aren't comparable to
    crystal/EM contact definitions.
    """
    method = (exp_method or "").upper()
    if "NMR" in method:
        return "discard"
    if resolution is None:
        return "discard"
    try:
        r = float(resolution)
    except (TypeError, ValueError):
        return "discard"
    if r <= 0:
        return "discard"
    if r <= 3.5:
        return "strict"
    if r <= 4.0:
        return "standard"
    if r <= 6.0:
        return "low"
    return "discard"


def classify_residue(name: str) -> tuple[str, bool]:
    """Return (category, is_modified_rna).

    category ∈ {protein, rna, dna, water, ligand, unknown}
    is_modified_rna is True when category == 'rna' but name is not one of A/U/G/C.
    """
    import gemmi  # deferred — see module docstring
    name_upper = name.strip().upper()
    info = gemmi.find_tabulated_residue(name_upper)
    if info is None:
        return "unknown", False
    if info.is_amino_acid():
        return "protein", False
    if info.is_nucleic_acid():
        kind_str = str(info.kind)
        if "RNA" in kind_str:
            return "rna", name_upper not in STANDARD_RNA
        if "DNA" in kind_str:
            return "dna", False
        return "rna", name_upper not in STANDARD_RNA
    if info.is_water():
        return "water", False
    return "ligand", False


def classify_chain_dominant(chain) -> str:
    """Return the dominant polymer type of a chain: protein / rna / dna / other.

    Walks residues, counts categories, picks the biggest polymer bucket. Used
    for coarse filtering before pair enumeration.
    """
    counts = {"protein": 0, "rna": 0, "dna": 0}
    for res in chain:
        cat, _ = classify_residue(res.name)
        if cat in counts:
            counts[cat] += 1
    total = counts["protein"] + counts["rna"] + counts["dna"]
    if total == 0:
        return "other"
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[0]


def _encode_case_marked(s: str) -> str:
    """Prepend '-' before every lowercase ASCII letter."""
    out = []
    for ch in s:
        if ch.isalpha() and ch.islower():
            out.append("-")
        out.append(ch)
    return "".join(out)


def _decode_case_marked(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        if s[i] == "-" and i + 1 < len(s):
            out.append(s[i + 1])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def sid_to_filename(sample_id: str) -> str:
    """Encode a sample_id so the on-disk filename is safe on case-insensitive FS.

    PDB chain names are case-sensitive (`A` ≠ `a`), but NTFS / HFS+ default to
    case-insensitive, so two legitimately different sample_ids like
    `3j46_Y_1` and `3j46_y_1` would silently collide and overwrite each other.

    Strategy: keep the PDB-id prefix as-is (it's always lowercased via
    path.stem.lower() upstream and never collides with itself), then prepend
    `-` before every lowercase letter in the chain-name tail. `-` never appears
    in PDB chain names, so the encoding is injective and folds to distinct
    strings.

    Reversible via filename_to_sid().
    """
    pdb, _, tail = sample_id.partition("_")
    if not tail:
        return _encode_case_marked(sample_id)
    return f"{pdb}_{_encode_case_marked(tail)}"


def filename_to_sid(stem: str) -> str:
    """Inverse of sid_to_filename (operates on the filename stem, no extension)."""
    pdb, _, tail = stem.partition("_")
    if not tail:
        return _decode_case_marked(stem)
    return f"{pdb}_{_decode_case_marked(tail)}"


def normalize_exp_method(m: str) -> str:
    """Collapse various method strings into {X-RAY, ELECTRON MICROSCOPY, NMR, NEUTRON, unknown}."""
    if not m or m == "unknown":
        return "unknown"
    m_up = m.upper()
    if "X-RAY" in m_up:
        return "X-RAY"
    if "ELECTRON MICROSCOPY" in m_up or "CRYO" in m_up or m_up == "EM":
        return "ELECTRON MICROSCOPY"
    if "NMR" in m_up:
        return "NMR"
    if "NEUTRON" in m_up:
        return "NEUTRON"
    return m_up
