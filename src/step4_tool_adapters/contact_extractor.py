"""Heavy-atom contact extraction for predicted RNA-protein complex structures.

Used by Cat A adapters (Boltz-2, RoseTTAFold2NA): given a predicted
mmCIF or PDB file, return the protein residues and RNA nucleotides that
fall within ``cutoff`` Å of each other (heavy atoms only).

Mirrors the logic of step 1's ``extract_pairs.process_file`` so the
ground truth and predicted contacts use identical semantics:
  - heavy atoms only (skip hydrogens)
  - residues classified by gemmi's tabulated residue dictionary
  - distance is the minimum heavy-atom pair distance
  - residue index = ``label_seq`` (1-based polymer index, gap-free)

Returns dataclass with sorted unique residue lists and the full pair list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import gemmi  # type: ignore
except ImportError as _e:  # pragma: no cover — gemmi is in environment.yml
    gemmi = None  # type: ignore
    _GEMMI_IMPORT_ERROR = _e
else:
    _GEMMI_IMPORT_ERROR = None


DEFAULT_CONTACT_CUTOFF = 4.5  # Å, matches step 1


# ---------- residue classification ----------------------------------------

# Minimal, kept in sync with step1_local._common.classify_residue but
# inlined to avoid a cross-step import (step 4 should be runnable by
# itself).
_AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}
_RNA1 = {"A", "C", "G", "U", "I"}


def _classify(res_name: str) -> str:
    """Return ``"protein"``, ``"rna"`` or ``"other"``."""
    name = res_name.strip().upper()
    if name in _AA3:
        return "protein"
    if name in _RNA1:
        return "rna"
    if gemmi is not None:
        try:
            info = gemmi.find_tabulated_residue(name)
        except Exception:
            info = None
        if info is not None:
            kind = str(info.kind) if hasattr(info, "kind") else ""
            # gemmi residue_kind enum has values like "AA", "RNA", "DNA" etc.
            if "AA" in kind:
                return "protein"
            if "RNA" in kind:
                return "rna"
    return "other"


# ---------- result dataclasses --------------------------------------------


@dataclass
class ContactPair:
    protein_residue: int
    rna_nucleotide: int
    distance: float


@dataclass
class ContactResult:
    """Output of :func:`extract_contacts`."""
    binding_protein_residues: list[int] = field(default_factory=list)
    binding_rna_nucleotides: list[int] = field(default_factory=list)
    contact_pairs: list[ContactPair] = field(default_factory=list)
    cutoff: float = DEFAULT_CONTACT_CUTOFF
    protein_chain_id: Optional[str] = None
    rna_chain_id: Optional[str] = None
    n_protein_residues: int = 0  # total protein residues in chosen chain
    n_rna_nucleotides: int = 0   # total RNA residues in chosen chain
    note: Optional[str] = None   # populated when extraction is incomplete

    @property
    def per_residue_protein_min_distance(self) -> dict[int, float]:
        """{protein_residue: min distance to any RNA atom} (sorted ascending)."""
        out: dict[int, float] = {}
        for cp in self.contact_pairs:
            cur = out.get(cp.protein_residue)
            if cur is None or cp.distance < cur:
                out[cp.protein_residue] = cp.distance
        return dict(sorted(out.items()))


# ---------- chain helpers --------------------------------------------------


def _classify_chain(chain) -> str:
    """Return dominant polymer type for a gemmi chain."""
    counts = {"protein": 0, "rna": 0, "other": 0}
    for res in chain:
        counts[_classify(res.name)] += 1
    if counts["protein"] == 0 and counts["rna"] == 0:
        return "other"
    return "protein" if counts["protein"] >= counts["rna"] else "rna"


def _pick_chains(
    model,
    protein_chain_id: Optional[str],
    rna_chain_id: Optional[str],
) -> tuple[Optional[object], Optional[object], Optional[str]]:
    """Pick a (protein_chain, rna_chain) pair from a gemmi model.

    If chain IDs are passed and present, use them directly.
    Otherwise fall back to the first protein and first RNA chain found.
    """
    by_id = {ch.name: ch for ch in model}

    prot_chain = None
    rna_chain = None

    if protein_chain_id:
        prot_chain = by_id.get(protein_chain_id)
    if rna_chain_id:
        rna_chain = by_id.get(rna_chain_id)

    if prot_chain is None or rna_chain is None:
        for ch in model:
            kind = _classify_chain(ch)
            if prot_chain is None and kind == "protein":
                prot_chain = ch
            elif rna_chain is None and kind == "rna":
                rna_chain = ch
            if prot_chain is not None and rna_chain is not None:
                break

    note = None
    if prot_chain is None:
        note = "no protein chain found"
    elif rna_chain is None:
        note = "no RNA chain found"
    return prot_chain, rna_chain, note


# ---------- main entry point ----------------------------------------------


def extract_contacts(
    structure_path: str | Path,
    *,
    cutoff: float = DEFAULT_CONTACT_CUTOFF,
    protein_chain_id: Optional[str] = None,
    rna_chain_id: Optional[str] = None,
) -> ContactResult:
    """Extract heavy-atom RNA-protein contacts from a structure file.

    Parameters
    ----------
    structure_path : str | Path
        Path to a .pdb or .cif file (anything gemmi can read).
    cutoff : float
        Distance threshold in Å (default 4.5, matches step 1).
    protein_chain_id, rna_chain_id : str, optional
        If provided, restrict extraction to the named chains. Otherwise
        the first protein and first RNA chain in the model are used.

    Returns
    -------
    ContactResult
        Always returned; on parse failure ``binding_*`` are empty and
        ``note`` carries the reason.
    """
    if gemmi is None:
        return ContactResult(
            cutoff=cutoff,
            note=f"gemmi not available: {_GEMMI_IMPORT_ERROR}",
        )

    path = Path(structure_path)
    if not path.is_file():
        return ContactResult(cutoff=cutoff, note=f"file not found: {path}")

    try:
        structure = gemmi.read_structure(str(path), merge_chain_parts=True)
        structure.setup_entities()
        structure.assign_label_seq_id(True)
    except Exception as e:
        return ContactResult(cutoff=cutoff, note=f"read_structure failed: {e}")

    if len(structure) == 0:
        return ContactResult(cutoff=cutoff, note="no models in file")

    model = structure[0]
    prot_chain, rna_chain, note = _pick_chains(
        model, protein_chain_id, rna_chain_id,
    )
    if prot_chain is None or rna_chain is None:
        return ContactResult(cutoff=cutoff, note=note)

    n_prot = sum(1 for r in prot_chain if _classify(r.name) == "protein")
    n_rna = sum(1 for r in rna_chain if _classify(r.name) == "rna")

    try:
        ns = gemmi.NeighborSearch(
            model, structure.cell, max(5.0, cutoff + 0.5),
        ).populate()
    except Exception as e:
        return ContactResult(
            cutoff=cutoff,
            protein_chain_id=prot_chain.name,
            rna_chain_id=rna_chain.name,
            n_protein_residues=n_prot,
            n_rna_nucleotides=n_rna,
            note=f"NeighborSearch failed: {e}",
        )

    # pair_key (protein_label_seq, rna_label_seq) -> min distance
    pair_min: dict[tuple[int, int], float] = {}
    prot_chain_name = prot_chain.name

    for rna_res in rna_chain:
        if rna_res.label_seq is None:
            continue
        if _classify(rna_res.name) != "rna":
            continue
        for atom in rna_res:
            if atom.element.is_hydrogen:
                continue
            try:
                marks = ns.find_atoms(atom.pos, "\0", radius=cutoff)
            except Exception:
                continue
            for mark in marks:
                cra = mark.to_cra(model)
                if cra.chain.name != prot_chain_name:
                    continue
                if cra.atom.element.is_hydrogen:
                    continue
                if cra.residue.label_seq is None:
                    continue
                if _classify(cra.residue.name) != "protein":
                    continue
                d = atom.pos.dist(cra.atom.pos)
                if d >= cutoff:
                    continue
                pk = (cra.residue.label_seq, rna_res.label_seq)
                cur = pair_min.get(pk)
                if cur is None or d < cur:
                    pair_min[pk] = d

    pairs = sorted(
        (ContactPair(p, r, round(d, 3)) for (p, r), d in pair_min.items()),
        key=lambda cp: (cp.protein_residue, cp.rna_nucleotide),
    )
    binding_prot = sorted({cp.protein_residue for cp in pairs})
    binding_rna = sorted({cp.rna_nucleotide for cp in pairs})

    return ContactResult(
        binding_protein_residues=binding_prot,
        binding_rna_nucleotides=binding_rna,
        contact_pairs=pairs,
        cutoff=cutoff,
        protein_chain_id=prot_chain_name,
        rna_chain_id=rna_chain.name,
        n_protein_residues=n_prot,
        n_rna_nucleotides=n_rna,
    )
