"""Shared structure-derived per-residue helpers for the adapter layer.

Originally a one-function module for the distance-based per-residue
binding score (used by Boltz-2 / Chai-1). HADDOCK 3 needs a separate
RNA-chain extractor that mirrors :func:`p2rank_adapter.extract_protein_chain_pdb`
so the docking input gets the same renumbered-to-1-based-polymer-index
treatment as the rest of the pipeline — that extractor lives here too.

Module placement: ``adapters/`` (next to the per-tool adapters) rather
than ``step5_fusion/`` — these helpers operate on raw predicted CIF /
PDB and are part of the adapter layer's pre / post-processing, same
level as ``contact_extractor`` and ``read_protein_plddt_from_cif``.
"""
from __future__ import annotations

from pathlib import Path

try:
    import gemmi  # type: ignore
except ImportError as _e:  # pragma: no cover — environment.yml dep
    gemmi = None  # type: ignore
    _GEMMI_IMPORT_ERROR: object = _e
else:
    _GEMMI_IMPORT_ERROR = None


# Standard RNA + common modified bases that should survive the chain
# filter. Step 1's RNA sanitizer maps modifications down to ACGUN
# already, but the raw PDB can still carry e.g. PSU (pseudouridine,
# very common in rRNA / tRNA) which gemmi reports under that 3-letter
# code. Keeping the set permissive here avoids dropping residues that
# legitimately belong to the chain just because they're non-canonical.
_RNA_RESIDUE_NAMES: set[str] = {
    "A", "U", "G", "C", "I",
    # Common modified bases that appear in rRNA / tRNA / spliceosome.
    "PSU", "OMG", "OMC", "OMU", "OMA",
    "1MA", "2MG", "5MC", "5MU", "7MG",
    "M2G", "M22G", "MA6", "H2U", "YYG",
    "UMS", "CCC",  # extra variants seen in the dataset
}


def _is_rna_residue(res_name: str) -> bool:
    """True if ``res_name`` is an RNA nucleotide (canonical or common
    modification).

    First tries the inline set above; falls back to gemmi's tabulated
    residue dictionary for anything we haven't enumerated. The fallback
    accepts any residue whose ``kind`` string contains ``"RNA"`` (gemmi
    distinguishes ``RNA``/``DNA``/``AA``/``HOH``/...).
    """
    name = (res_name or "").strip().upper()
    if not name:
        return False
    if name in _RNA_RESIDUE_NAMES:
        return True
    if gemmi is None:
        return False
    try:
        info = gemmi.find_tabulated_residue(name)
    except Exception:
        return False
    if info is None:
        return False
    kind = str(info.kind) if hasattr(info, "kind") else ""
    return "RNA" in kind


def _is_amino_acid_residue(res_name: str) -> bool:
    """True if ``res_name`` is a (standard or modified) amino acid.

    Used only by the chain-type classifier below. Delegates to gemmi's
    tabulated dictionary; ``MSE`` (selenomethionine) etc. are correctly
    flagged as amino acids by ``is_amino_acid()``.
    """
    name = (res_name or "").strip().upper()
    if not name or gemmi is None:
        return False
    try:
        info = gemmi.find_tabulated_residue(name)
    except Exception:
        return False
    return bool(info) and info.is_amino_acid()


def _chain_kind(chain) -> str:
    """Dominant polymer type of a gemmi chain: ``"protein"`` /
    ``"rna"`` / ``"other"`` (majority vote over residues)."""
    n_prot = n_rna = 0
    for res in chain:
        if _is_rna_residue(res.name):
            n_rna += 1
        elif _is_amino_acid_residue(res.name):
            n_prot += 1
    if n_prot == 0 and n_rna == 0:
        return "other"
    return "protein" if n_prot >= n_rna else "rna"


def _pick_chain(model, chain_id: str, want_kind: str):
    """Select the chain to treat as ``want_kind`` (``"protein"`` /
    ``"rna"``).

    Resolution order:

      1. A chain literally named ``chain_id`` **whose content matches**
         ``want_kind``. The content check is what dodges the HADDOCK 3
         trap where the protein chain is literally named ``"B"`` — a
         naive name lookup for the RNA chain (``"B"``) would otherwise
         grab the protein.
      2. Otherwise, the first chain whose content is ``want_kind``.

    Returns the gemmi chain or ``None`` when no chain of that kind
    exists at all. When the structure IS named ``A``/``B`` with the
    expected content (the Boltz-2 / Chai-1 case), step 1 matches and
    the output is byte-identical to the old strict-name behaviour.
    """
    if chain_id:
        for ch in model:
            if ch.name == chain_id and _chain_kind(ch) == want_kind:
                return ch
    for ch in model:
        if _chain_kind(ch) == want_kind:
            return ch
    return None


def compute_distance_binding_scores(
    structure_path,
    *,
    protein_chain_id: str = "A",
    rna_chain_id: str = "B",
    distance_scale: float = 8.0,
) -> dict[int, float]:
    """Per-protein-residue binding probability from CA → RNA distance.

    For each protein residue:

      d_i = min distance (Å) from CA to ANY RNA heavy atom
      prob_i = 1 / (1 + d_i / distance_scale)

    Why this is preferred over the PAE-matrix score (per offline ablation)
    --------------------------------------------------------------------
    The PAE-based score (``compute_pae_binding_scores``) uses *pairwise*
    inter-chain PAE values. Boltz-2's PAE for RNA-protein interfaces
    is calibrated against the model's own confidence — high PAE doesn't
    necessarily mean the interface is wrong, only that the model is
    less sure about the relative geometry. Direct CA → nearest-RNA
    distance bypasses that calibration: residues actually placed near
    the RNA in the predicted complex are flagged regardless of whether
    the model "knew" they were correct. On the 200-sample test set
    this took Boltz-2's per-residue Pearson R from 0.14 → ~0.35 and
    closed the gap to Chai-1's distance-derived score (which has been
    on this path from day one because Chai-1's scores.npz never
    carried a per-token PAE matrix anyway).

    ``distance_scale`` is intentionally on the same Å axis as the
    PAE scale: 8 Å is the "halfway" distance below which a residue
    is plausibly contacting RNA (heavy-atom contact cutoffs in step 1
    are 4.5 Å; Cα → atom distance for an actual contact runs 6-9 Å).

    Hydrogens are excluded from the RNA side — predicted complexes
    rarely include them and they'd just shrink ``d_i`` by ~1 Å versus
    the parent heavy atom, which the scale makes negligible.

    Returns
    -------
    ``{protein_residue_idx (1-based, in the predicted CIF's
    seqid.num — i.e. CLEANED-sequence numbering for our adapters):
    probability}``. Empty dict on any of:
      * gemmi unavailable / structure unreadable
      * no model in the structure
      * RNA chain absent or has no heavy atoms
      * no protein residues with a CA atom
      * non-positive ``distance_scale``

    The caller is responsible for remapping these clean indices back
    to the dataset's original numbering via ``remap_per_residue``
    and the protein index map (same as the per_residue_confidence
    pLDDT path does).
    """
    if gemmi is None:
        return {}
    if structure_path is None or not Path(structure_path).is_file():
        return {}
    if not (distance_scale > 0):
        return {}

    try:
        structure = gemmi.read_structure(str(structure_path))
    except Exception:  # noqa: BLE001
        return {}
    if len(structure) == 0:
        return {}
    model = structure[0]

    # Resolve the RNA / protein chains by name-then-content (see
    # _pick_chain). This is what fixes the HADDOCK 3 per-residue gap:
    # ``extract_protein_chain_pdb`` / ``extract_rna_chain_pdb`` only
    # rename MULTI-char source chains to A/B — a single-char source
    # chain (protein "A", RNA "E") is written under its ORIGINAL name,
    # which HADDOCK 3 then preserves in the docked pose. The old code
    # filtered strictly on the caller's "A"/"B" and so found no RNA
    # atoms (→ empty dict → null per_residue_pae_score) for every
    # sample whose RNA chain wasn't literally "B". Content-based
    # selection finds the right chains regardless of name; for the
    # Boltz-2 / Chai-1 complexes (genuinely A/B) it resolves to exactly
    # the same chains as before.
    rna_chain = _pick_chain(model, rna_chain_id, "rna")
    if rna_chain is None:
        return {}
    prot_chain = _pick_chain(model, protein_chain_id, "protein")
    if prot_chain is None:
        return {}

    # Collect RNA heavy atoms (skip hydrogens — see docstring).
    rna_positions = []
    for residue in rna_chain:
        for atom in residue:
            # gemmi.Element exposes .name (string) and .is_hydrogen.
            # Use .is_hydrogen so deuterium ('D') is skipped too.
            if atom.element.is_hydrogen:
                continue
            rna_positions.append(atom.pos)
    if not rna_positions:
        return {}

    scores: dict[int, float] = {}
    for residue in prot_chain:
        ca = residue.find_atom("CA", "*")
        if ca is None:
            continue
        idx = residue.seqid.num
        if idx is None or idx < 1:
            continue
        min_dist = min(ca.pos.dist(p) for p in rna_positions)
        prob = 1.0 / (1.0 + min_dist / distance_scale)
        # Defensive clamp; ca.pos.dist is non-negative so prob ≤ 1
        # by construction, but float drift on huge distances at
        # tiny scales can still push slightly out.
        if prob < 0.0:
            prob = 0.0
        elif prob > 1.0:
            prob = 1.0
        scores[idx] = round(prob, 6)
    return scores


# ---- RNA chain extractor (for HADDOCK 3 input prep) ---------------------


def effective_rna_chain_id(chain_id: str) -> str:
    """Single-char fallback for the RNA-chain name in PDB output.

    Mirrors ``effective_pdb_chain_id`` in p2rank_adapter but defaults
    to ``"B"`` instead of ``"A"`` — HADDOCK 3 reads two PDBs (protein
    and RNA) and pairs them with chain IDs ``A`` / ``B`` respectively
    in the docked complex, so naming the RNA chain ``B`` keeps the
    downstream parse step's chain-filter trivial.
    """
    return "B" if len(chain_id or "") > 1 else chain_id


def extract_rna_chain_pdb(
    raw_path: Path,
    chain_id: str,
    out_path: Path,
) -> Path:
    """Write a single-chain RNA PDB renumbered to 1-based polymer index.

    Mirror of :func:`p2rank_adapter.extract_protein_chain_pdb` for
    the RNA side. Drops non-RNA residues (waters, ions, ligands, the
    protein chain if accidentally selected). Output chain name is
    single-character (``"B"`` for multi-char inputs, else passthrough)
    so PDB-format writers don't silently truncate.

    HADDOCK 3 wants a PDB per molecule. The output of this function is
    suitable as the ``[molecules]`` entry alongside the protein PDB.

    Raises
    ------
    FileNotFoundError
        if ``raw_path`` doesn't exist.
    ValueError
        if the requested chain is not present, or the chain has no RNA
        residues with a polymer index.
    """
    if gemmi is None:
        raise RuntimeError(f"gemmi not available: {_GEMMI_IMPORT_ERROR}")
    if not Path(raw_path).is_file():
        raise FileNotFoundError(f"raw structure not found: {raw_path}")

    structure = gemmi.read_structure(str(raw_path), merge_chain_parts=True)
    structure.setup_entities()
    structure.assign_label_seq_id(True)

    if len(structure) == 0:
        raise ValueError(f"no models in {raw_path}")
    src_model = structure[0]

    src_chain = None
    for ch in src_model:
        if ch.name == chain_id:
            src_chain = ch
            break
    if src_chain is None:
        raise ValueError(
            f"chain {chain_id!r} not found in {raw_path} "
            f"(available: {[c.name for c in src_model]})"
        )

    new_structure = gemmi.Structure()
    new_structure.cell = structure.cell
    try:
        new_structure.spacegroup_hm = structure.spacegroup_hm
    except Exception:
        pass

    new_model = gemmi.Model("1")
    new_chain = gemmi.Chain(effective_rna_chain_id(chain_id))

    kept = 0
    for res in src_chain:
        if not _is_rna_residue(res.name):
            continue
        if res.label_seq is None:
            continue
        new_res = res.clone()
        new_res.seqid = gemmi.SeqId(int(res.label_seq), " ")
        new_chain.add_residue(new_res)
        kept += 1

    if kept == 0:
        raise ValueError(
            f"chain {chain_id!r} in {raw_path} has 0 RNA residues with "
            f"a polymer index"
        )

    new_model.add_chain(new_chain)
    new_structure.add_model(new_model)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_structure.write_pdb(str(out_path))
    return out_path
