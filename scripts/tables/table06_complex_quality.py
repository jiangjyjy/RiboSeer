"""Complex-structure quality metrics for paper Table 6.

Every tool that emits a full 3-D RNA-protein complex is scored: the Cat A
co-folders (Boltz-2, Chai-1, RoseTTAFold2NA, AlphaFold3, RFAA) and the
Cat D docking tools (HDOCK, HADDOCK3) — see ``COMPLEX_TOOLS``. A tool is
evaluated iff its step4 record carries a ``predicted_structure_path``;
the record's ``category`` field is NOT used to gate inclusion (docking
tools report 'D' yet still produce a complex). Table 6 reports how close
those predicted complexes are to the experimental reference: a global TM-score
(US-align, normalised by the *reference* chain length), an interface
RMSD (iRMSD), a ligand RMSD (LRMSD, RNA position after the protein is
superposed), the fraction of native heavy-atom contacts recovered
(Fnat), and the composite DockQ score.

DockQ source of truth
---------------------
By default we delegate iRMSD / LRMSD / Fnat / DockQ to the canonical
``DockQ`` Python package (``pip install DockQ``,
``from DockQ.DockQ import load_PDB, run_on_all_native_interfaces``).
If a complex has multiple interfaces, we keep the one with the highest
DockQ — typical Cat A outputs have one (protein, RNA) pair so this is
a no-op. When the package is unavailable (e.g. local dev box) the
script falls back to a manual CAPRI-formula path
(``compute_irmsd`` / ``compute_lrmsd`` / ``compute_fnat`` +
``compute_dockq``) so the run still produces a (slightly approximate)
table. A one-time stderr warning marks the fallback.

Pipeline per (sample, tool)
---------------------------
1. Read the step4 JSONL last record → the tool's
   ``predicted_structure_path`` (only Cat A, only success=True).
2. Build a *reference complex PDB* by extracting the sample's protein
   chain + RNA chain from ``data/raw/<source_pdb>.{pdb,cif}`` and
   renaming them to ``A`` / ``B`` (multi-char chain ids are handled
   via gemmi, mirroring ``compute_tmscore_matrix.extract_chain_from_cif``).
3. Run ``USalign pred ref -ter 0`` and parse TM-score normalised by
   reference (the ``Structure_2`` / ``Chain_2`` form the
   ``compute_tmscore_matrix._TM_RE`` regex already accepts).
4. Pass (pred, ref) to ``DockQ.DockQ.run_on_all_native_interfaces``;
   pick best interface by DockQ. On failure / no package, fall back to
   the formula path:

   - Extract heavy atoms + backbone (Cα for protein, P for RNA) from
     both structures with gemmi, keyed by ``label_seq``.
   - iRMSD: Cα RMSD on the union of pred + ref 10 Å interface
     residues after Kabsch superposition.
   - LRMSD: superpose proteins, apply transform to predicted RNA,
     measure RNA backbone RMSD.
   - Fnat: ≤ 5 Å heavy-atom contacts shared with reference /
     |reference contacts|.
   - DockQ = ``(Fnat + 1/(1+(iRMSD/1.5)²) + 1/(1+(LRMSD/8.5)²)) / 3``.

5. ``DockQ ≥ 0.23`` acceptable, ``≥ 0.49`` medium (CAPRI standard).

RiboSeer (Cat. A consensus)
---------------------------
With ``--enriched-model-dir``, picks one Cat A tool per sample whose
predicted binding residues have the highest *mean enriched-fusion
probability* — i.e. the Cat A model whose predicted interface most
agrees with RiboSeer's fused per-residue scores. The chosen tool's
metrics are aggregated into a ``riboseer_consensus`` row.

Usage
-----
::

    python scripts/tables/table06_complex_quality.py \\
        --step4-dir          data/batch_test_v7/step4/ \\
        --processed-dir      data/processed_quality \\
        --sample-list        data/processed_quality/splits/test.txt \\
        --raw-dir            data/raw/ \\
        --usalign-bin        /opt/biotools/USalign/USalign \\
        --output             data/batch_test_v7/complex_quality.csv \\
        --enriched-model-dir data/enriched_v7_model/
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from step5_fusion.data_collector import (  # noqa: E402
    load_sample_json, per_residue_to_int_dict,
)
from step5_fusion.enriched_fusion import EnrichedFusion  # noqa: E402
from scripts.compute_tmscore_matrix import (  # noqa: E402
    find_raw_pdb, find_raw_cif, parse_usalign_output,
)
from step5_fusion.prediction_io import load_predictions_dir  # noqa: E402

# ---- DockQ package (optional, primary path) -----------------------------
# Lazy import: not installed on the local dev box but is on the server.
# The fallback formula path below kicks in (with a one-time stderr
# warning) when the package isn't importable.
try:
    from DockQ.DockQ import (  # type: ignore # noqa: E402
        load_PDB as _dockq_load_PDB,
        run_on_all_native_interfaces as _dockq_run,
    )
    DOCKQ_OK = True
except ImportError:
    _dockq_load_PDB = None  # type: ignore[assignment]
    _dockq_run = None       # type: ignore[assignment]
    DOCKQ_OK = False

# Tracks whether the "DockQ missing → using formula" stderr line has
# been printed in this process. Tests reset it via the module symbol.
_DOCKQ_WARNING_EMITTED = False

# Same idea for US-align: if the binary can't be exec'd, every TM-score
# would silently come back None. Warn loudly, exactly once.
_USALIGN_WARNING_EMITTED = False


def _warn_once_no_dockq() -> None:
    """Emit the no-DockQ stderr warning at most once per process so a
    400-sample batch doesn't get 400 copies of it."""
    global _DOCKQ_WARNING_EMITTED
    if _DOCKQ_WARNING_EMITTED:
        return
    _DOCKQ_WARNING_EMITTED = True
    print(
        "WARNING: DockQ package not importable — falling back to the "
        "manual CAPRI formula for iRMSD / LRMSD / Fnat / DockQ. "
        "Results may differ slightly from the canonical DockQ tool. "
        "Install with: pip install DockQ",
        file=sys.stderr,
    )


# ---- constants ----------------------------------------------------------

# step4 ids for every tool that emits a full 3-D RNA-protein complex —
# these are the rows of Table 6. Membership here (NOT the record's
# ``category`` field) is the column contract: a tool is evaluated iff its
# id is in this set AND it carries a ``predicted_structure_path``.
#   - Cat A co-folders: boltz2 / chai1 / rosettafold2na (live adapters),
#     alphafold3 (af3_parse.py), rfaa (rfaa adapter / parse_rfaa_results.py)
#   - Cat D docking:    hdock (hdock_parse.py), haddock3 (haddock3 adapter)
COMPLEX_TOOLS: tuple[str, ...] = (
    "boltz2", "chai1", "rosettafold2na",
    "alphafold3", "rfaa",
    "hdock", "haddock3",
)

# Co-folding (Cat A) subset eligible for the RiboSeer consensus pick.
# The consensus row means "the co-folding model whose predicted interface
# best agrees with RiboSeer's fused per-residue probs", so the docking
# tools (hdock / haddock3) are deliberately NOT candidates. Defaults to the
# three originally-published co-folders for reproducibility; widen via
# ``--consensus-tools`` to also let AlphaFold3 / RFAA compete.
CAT_A_TOOLS: tuple[str, ...] = ("boltz2", "chai1", "rosettafold2na")

# DockQ scale constants (CAPRI standard, Basu & Wallner 2016).
DOCKQ_D1 = 1.5
DOCKQ_D2 = 8.5
DOCKQ_ACCEPTABLE = 0.23
DOCKQ_MEDIUM = 0.49

# Interface / contact cutoffs (Å).
IFACE_CUTOFF = 10.0
CONTACT_CUTOFF = 5.0

# Minimum points for a meaningful Kabsch superposition.
KABSCH_MIN_POINTS = 3

# Standard 3-letter aa / 1-letter RNA, plus common modifications gemmi
# also flags as aa/nucleic. Mirrors step1's classifier — kept inline so
# this script can be lifted out without dragging step1's import graph.
_AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}
_RNA1 = {"A", "C", "G", "U", "I"}


# ---- aggregation helpers ------------------------------------------------


def _mean(vs): return round(statistics.fmean(vs), 4) if vs else None
def _median(vs): return round(statistics.median(vs), 4) if vs else None


# ---- IO -----------------------------------------------------------------


def _read_last_jsonl_record(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    rec = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
    except OSError:
        return None
    return rec


def _load_sample_ids(path: Optional[Path]) -> Optional[list[str]]:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"sample-list not found: {path}")
    return [ln.split()[0] for ln in path.read_text(encoding="utf-8")
            .splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


# ---- atom-coordinate container ------------------------------------------


@dataclasses.dataclass
class ChainAtoms:
    """Heavy-atom + backbone coords for ONE chain side (all protein or
    all RNA residues from the structure), keyed by ``label_seq``.

    - ``backbone[label_seq]`` is the single 3-vector for the chain's
      backbone atom (Cα for protein, P for RNA). Missing residues
      simply aren't keys.
    - ``heavy[label_seq]`` is an ``(N, 3)`` array of every heavy atom
      in that residue (no hydrogens). Used for contact / interface
      tests.
    """
    backbone: dict[int, np.ndarray]
    heavy: dict[int, np.ndarray]


def _classify_residue(res_name: str) -> str:
    """Return ``"protein"`` / ``"rna"`` / ``"other"``. Mirrors
    ``step4_tool_adapters.contact_extractor._classify`` so step1's GT,
    step4's contact extraction and this script's atom triage all
    agree on which residues "count"."""
    import gemmi
    name = res_name.strip().upper()
    if name in _AA3:
        return "protein"
    if name in _RNA1:
        return "rna"
    try:
        info = gemmi.find_tabulated_residue(name)
    except Exception:
        info = None
    if info is not None:
        kind = str(info.kind) if hasattr(info, "kind") else ""
        if "AA" in kind:
            return "protein"
        if "RNA" in kind:
            return "rna"
    return "other"


def _is_amino_acid_chain(chain) -> bool:
    """Majority vote over residue kinds — handles chains with stray
    modified residues / ions that ``classify_residue`` would file under
    ``"other"``."""
    n_prot = n_rna = 0
    for res in chain:
        cat = _classify_residue(res.name)
        if cat == "protein":
            n_prot += 1
        elif cat == "rna":
            n_rna += 1
    return n_prot >= n_rna and n_prot > 0


def extract_complex_atoms(struct_path: Path
                          ) -> tuple[ChainAtoms, ChainAtoms]:
    """Read a complex structure → ``(protein_side, rna_side)``.

    Chains are classified by content (majority residue type), so the
    function works on predicted outputs from any of the Cat A tools
    regardless of chain naming convention. Multiple protein / RNA
    chains in one file are merged into the same side (later residues
    overwrite earlier ones on ``label_seq`` collision — a non-issue
    for typical 1+1 Cat A outputs).

    Raises FileNotFoundError if the path is missing, ValueError on
    gemmi parse failure / setup error.
    """
    import gemmi

    if not struct_path.is_file():
        raise FileNotFoundError(struct_path)
    try:
        st = gemmi.read_structure(str(struct_path))
    except Exception as e:
        raise ValueError(
            f"gemmi.read_structure failed: {e}") from e
    if len(st) == 0:
        raise ValueError(f"no models in {struct_path}")
    try:
        st.setup_entities()
        st.assign_label_seq_id(True)
    except Exception as e:
        raise ValueError(f"assign_label_seq_id failed: {e}") from e

    model = st[0]
    prot = ChainAtoms(backbone={}, heavy={})
    rna = ChainAtoms(backbone={}, heavy={})

    for chain in model:
        is_protein = _is_amino_acid_chain(chain)
        target = prot if is_protein else rna
        backbone_name = "CA" if is_protein else "P"
        for res in chain:
            if res.label_seq is None:
                continue
            cat = _classify_residue(res.name)
            if is_protein and cat != "protein":
                continue
            if not is_protein and cat != "rna":
                continue
            atoms: list[tuple[float, float, float]] = []
            backbone: Optional[tuple[float, float, float]] = None
            for atom in res:
                if atom.element.is_hydrogen:
                    continue
                coord = (float(atom.pos.x), float(atom.pos.y),
                         float(atom.pos.z))
                atoms.append(coord)
                if atom.name.strip() == backbone_name:
                    backbone = coord
            if not atoms:
                continue
            target.heavy[int(res.label_seq)] = np.asarray(
                atoms, dtype=np.float64)
            if backbone is not None:
                target.backbone[int(res.label_seq)] = np.asarray(
                    backbone, dtype=np.float64)
    return prot, rna


def build_ref_complex_pdb(raw_dir: Path, source_pdb: str,
                          prot_chain_id: str, rna_chain_id: str,
                          dst: Path) -> Path:
    """Extract one protein chain + one RNA chain from the raw PDB / CIF
    and write them as a single 2-chain PDB at ``dst`` (protein → 'A',
    RNA → 'B'). Returns ``dst``.

    PDB first (cheap line-grep would lose multi-char chain ids, so we
    always use gemmi); falls back to mmCIF if PDB absent. Renaming to
    single-character ids is required because PDB column 22 holds only
    one char — the predicted Cat A complexes already follow this
    convention.
    """
    import gemmi

    src: Optional[Path] = None
    try:
        src = find_raw_pdb(raw_dir, source_pdb)
    except FileNotFoundError:
        src = None
    if src is None:
        src = find_raw_cif(raw_dir, source_pdb)
    if src is None:
        raise FileNotFoundError(
            f"no raw structure for {source_pdb} under {raw_dir}")
    try:
        st = gemmi.read_structure(str(src))
    except Exception as e:
        raise ValueError(f"gemmi failed on {src}: {e}") from e
    try:
        st.setup_entities()
        st.assign_label_seq_id(True)
    except Exception as e:
        raise ValueError(f"label_seq assign failed on {src}: {e}") from e

    new_st = gemmi.Structure()
    new_model = gemmi.Model("1")
    found_prot = found_rna = False
    for model in st:
        for chain in model:
            if chain.name == prot_chain_id and not found_prot:
                cl = chain.clone()
                cl.name = "A"
                new_model.add_chain(cl)
                found_prot = True
            elif chain.name == rna_chain_id and not found_rna:
                cl = chain.clone()
                cl.name = "B"
                new_model.add_chain(cl)
                found_rna = True
        if found_prot and found_rna:
            break
    if not (found_prot and found_rna):
        missing = []
        if not found_prot:
            missing.append(f"protein chain {prot_chain_id!r}")
        if not found_rna:
            missing.append(f"RNA chain {rna_chain_id!r}")
        raise ValueError(
            f"{src.name}: missing " + " + ".join(missing))
    new_st.add_model(new_model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    new_st.write_pdb(str(dst))
    return dst


# ---- US-align wrapper ---------------------------------------------------


def run_usalign_tmscore(usalign_bin: str, pred_path: Path,
                        ref_path: Path,
                        timeout: float = 120.0) -> Optional[float]:
    """Return the TM-score normalised by the *reference* (second arg).

    Uses ``parse_usalign_output`` so the regex stays in lock-step with
    the build US-align ships — both the old ``Chain_1/2`` wording and
    the modern ``Structure_1/2`` wording are accepted. Returns ``None``
    on any failure (non-zero rc, parse miss, timeout) so the caller
    can record an empty TM cell instead of aborting the run.
    """
    cmd = [usalign_bin, str(pred_path), str(ref_path), "-ter", "0"]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=timeout)
    except FileNotFoundError:
        # The binary isn't on PATH / the --usalign-bin path is wrong.
        # Without this, all-None TM-score is indistinguishable from a
        # parse miss. Warn once, then keep returning None.
        global _USALIGN_WARNING_EMITTED
        if not _USALIGN_WARNING_EMITTED:
            _USALIGN_WARNING_EMITTED = True
            print(
                f"WARNING: US-align binary not found: {usalign_bin!r}. "
                "Every TM-score will be empty. Pass a valid path via "
                "--usalign-bin (e.g. /opt/.../USalign/USalign) or "
                "put 'USalign' on PATH.",
                file=sys.stderr,
            )
        return None
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        return None
    if cp.returncode != 0:
        return None
    try:
        tm1, tm2 = parse_usalign_output(cp.stdout)
    except ValueError:
        return None
    # parse_usalign_output guarantees both values are filled (each is
    # set to the other when only one was reported), so tm2 is always
    # the reference-normalised score for Table 6.
    return float(tm2)


def usalign_preflight(usalign_bin: str) -> bool:
    """Return True iff ``usalign_bin`` is an executable we can launch.

    Run once at startup so a wrong ``--usalign-bin`` fails the run with a
    clear message instead of quietly producing an all-empty TM-score
    column. A non-zero rc from a bare invocation is fine (US-align prints
    usage and exits non-zero with no args) — we only care that exec
    itself succeeds (i.e. no FileNotFoundError)."""
    try:
        subprocess.run([usalign_bin], capture_output=True, text=True,
                       timeout=30.0)
    except FileNotFoundError:
        return False
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        # It launched (and hung / misbehaved) — that's "found".
        return True
    return True


# ---- geometric primitives -----------------------------------------------


def kabsch_align(P: np.ndarray, Q: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Find ``(R, t)`` such that ``P @ R.T + t`` minimises RMSD to Q.

    ``P``, ``Q`` are matched ``(N, 3)`` coordinate arrays. Uses SVD on
    the cross-covariance and corrects the determinant sign so the
    result is a proper rotation (no improper reflection). Caller is
    responsible for picking matched points; this routine doesn't
    re-pair anything.
    """
    if P.shape != Q.shape or P.shape[0] < KABSCH_MIN_POINTS:
        raise ValueError(
            f"kabsch_align needs ≥ {KABSCH_MIN_POINTS} matched points "
            f"(got shapes {P.shape} vs {Q.shape})")
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cQ - cP @ R.T
    return R, t


def apply_transform(P: np.ndarray, R: np.ndarray,
                    t: np.ndarray) -> np.ndarray:
    return P @ R.T + t


def superposed_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    R, t = kabsch_align(P, Q)
    Pa = apply_transform(P, R, t)
    return float(np.sqrt(np.mean(np.sum((Pa - Q) ** 2, axis=1))))


# ---- contact / interface ------------------------------------------------


def _min_residue_distance(p_heavy: np.ndarray,
                          rna_atoms_concat: np.ndarray) -> float:
    """Min Euclidean distance between any heavy atom of one protein
    residue (``p_heavy``) and any RNA heavy atom (``rna_atoms_concat``).

    Both arrays must be ``(*, 3)``. Returns ``inf`` if either side is
    empty so the caller can compare against any cutoff safely."""
    if p_heavy.size == 0 or rna_atoms_concat.size == 0:
        return float("inf")
    diff = p_heavy[:, None, :] - rna_atoms_concat[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    return float(np.sqrt(d2.min()))


def interface_residues(prot: ChainAtoms, rna: ChainAtoms,
                       cutoff: float = IFACE_CUTOFF) -> set[int]:
    """Protein label_seqs with ANY heavy atom within ``cutoff`` of ANY
    RNA heavy atom. (Symmetric to step1's contact extraction but at a
    looser 10 Å cutoff — the standard CAPRI interface definition.)"""
    if not rna.heavy:
        return set()
    rna_all = np.concatenate(list(rna.heavy.values()))
    out: set[int] = set()
    for pr, p_atoms in prot.heavy.items():
        if _min_residue_distance(p_atoms, rna_all) < cutoff:
            out.add(pr)
    return out


def native_contacts(prot: ChainAtoms, rna: ChainAtoms,
                    cutoff: float = CONTACT_CUTOFF
                    ) -> set[tuple[int, int]]:
    """``(prot_resid, rna_resid)`` pairs with min heavy-atom distance
    below ``cutoff``. The CAPRI native-contact definition uses 5 Å —
    looser than step1's 4.5 Å contact list, which is by design (Fnat
    is meant to be forgiving)."""
    c2 = cutoff * cutoff
    out: set[tuple[int, int]] = set()
    for pr, p_atoms in prot.heavy.items():
        for nr, r_atoms in rna.heavy.items():
            diff = p_atoms[:, None, :] - r_atoms[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            if d2.min() < c2:
                out.add((pr, nr))
    return out


# ---- metric pack --------------------------------------------------------


def compute_irmsd(pred_prot: ChainAtoms, pred_rna: ChainAtoms,
                  ref_prot: ChainAtoms, ref_rna: ChainAtoms,
                  cutoff: float = IFACE_CUTOFF) -> Optional[float]:
    """Cα RMSD on (pred interface ∪ ref interface) after Kabsch
    superposition. None if < 3 common residues survive the
    intersection with both backbone maps (Kabsch needs ≥ 3)."""
    pred_iface = interface_residues(pred_prot, pred_rna, cutoff)
    ref_iface = interface_residues(ref_prot, ref_rna, cutoff)
    union = pred_iface | ref_iface
    common = sorted(union & set(pred_prot.backbone)
                    & set(ref_prot.backbone))
    if len(common) < KABSCH_MIN_POINTS:
        return None
    P = np.asarray([pred_prot.backbone[r] for r in common])
    Q = np.asarray([ref_prot.backbone[r] for r in common])
    return superposed_rmsd(P, Q)


def compute_lrmsd(pred_prot: ChainAtoms, pred_rna: ChainAtoms,
                  ref_prot: ChainAtoms, ref_rna: ChainAtoms,
                  ) -> Optional[float]:
    """Superpose proteins (all matched Cα), apply the same transform
    to predicted RNA backbone (P atoms), report RMSD to the reference
    RNA backbone. None if either side lacks the minimum points."""
    common_prot = sorted(set(pred_prot.backbone)
                         & set(ref_prot.backbone))
    if len(common_prot) < KABSCH_MIN_POINTS:
        return None
    P_prot = np.asarray([pred_prot.backbone[r] for r in common_prot])
    Q_prot = np.asarray([ref_prot.backbone[r] for r in common_prot])
    R, t = kabsch_align(P_prot, Q_prot)

    common_rna = sorted(set(pred_rna.backbone) & set(ref_rna.backbone))
    if not common_rna:
        return None
    P_rna = np.asarray([pred_rna.backbone[r] for r in common_rna])
    Q_rna = np.asarray([ref_rna.backbone[r] for r in common_rna])
    P_rna_aligned = apply_transform(P_rna, R, t)
    return float(np.sqrt(np.mean(np.sum(
        (P_rna_aligned - Q_rna) ** 2, axis=1))))


def compute_fnat(pred_prot: ChainAtoms, pred_rna: ChainAtoms,
                 ref_prot: ChainAtoms, ref_rna: ChainAtoms,
                 cutoff: float = CONTACT_CUTOFF) -> Optional[float]:
    """|pred_contacts ∩ ref_contacts| / |ref_contacts|. None when the
    reference has zero native contacts (degenerate)."""
    ref_c = native_contacts(ref_prot, ref_rna, cutoff)
    if not ref_c:
        return None
    pred_c = native_contacts(pred_prot, pred_rna, cutoff)
    return len(pred_c & ref_c) / len(ref_c)


def compute_dockq(fnat: Optional[float], irmsd: Optional[float],
                  lrmsd: Optional[float],
                  d1: float = DOCKQ_D1,
                  d2: float = DOCKQ_D2) -> Optional[float]:
    """``(Fnat + 1/(1+(iRMSD/d1)²) + 1/(1+(LRMSD/d2)²)) / 3``.

    None if any component is missing — DockQ has no defined value when
    one of its three terms isn't computable, and dropping silently to
    e.g. ``0`` would skew Table 6's `% acceptable` aggregates."""
    if fnat is None or irmsd is None or lrmsd is None:
        return None
    f1 = 1.0 / (1.0 + (irmsd / d1) ** 2)
    f2 = 1.0 / (1.0 + (lrmsd / d2) ** 2)
    return (fnat + f1 + f2) / 3.0


def _maybe_float(v) -> Optional[float]:
    """Coerce DockQ-package values to ``float`` defensively. The package
    sometimes hands back numpy scalars or strings depending on the
    code path; we want a plain ``float`` (or ``None`` if unreadable)."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _try_dockq_package(pred_path: Path, ref_path: Path
                       ) -> Optional[dict]:
    """Run the DockQ package on one (pred, ref) pair and return the
    best-interface metric dict.

    Return shape: ``{"interface": ..., "irmsd": ..., "lrmsd": ...,
    "fnat": ..., "dockq": ...}`` — irmsd / lrmsd / fnat may be ``None``
    if a given DockQ build omits them, but ``dockq`` is always present
    (entries without a DockQ are skipped during ranking).

    Returns ``None`` (caller falls back to formula) when:

    - The package isn't importable (emits the warn-once stderr line).
    - ``load_PDB`` / ``run_on_all_native_interfaces`` raise — the
      exception is swallowed because any per-pair failure (e.g.
      malformed CIF, unknown residue, atom-count mismatch) should
      degrade gracefully rather than abort the batch.
    - The package returns an empty chain map.

    Multi-interface handling: DockQ ranks each chain pairing it
    discovers. Typical Cat A outputs are single (protein, RNA) so the
    map has one entry; we still ``max(DockQ)`` to be safe.
    """
    if not DOCKQ_OK:
        _warn_once_no_dockq()
        return None
    try:
        model = _dockq_load_PDB(str(pred_path))     # type: ignore[misc]
        native = _dockq_load_PDB(str(ref_path))     # type: ignore[misc]
        result = _dockq_run(model, native)          # type: ignore[misc]
    except Exception:                               # noqa: BLE001
        # Any DockQ failure → fall back to formula path. We don't log
        # here (a 400-sample batch with many bad refs would spam); the
        # caller's skip counter is sufficient bookkeeping.
        return None

    # DockQ ≥ v2 returns ``(chain_map: dict, total: float)``; older
    # builds returned just the chain map. Accept either shape.
    if isinstance(result, tuple) and result:
        chain_map = result[0]
    else:
        chain_map = result
    if not isinstance(chain_map, dict) or not chain_map:
        return None

    best: Optional[dict] = None
    for iface_id, metrics in chain_map.items():
        if not isinstance(metrics, dict):
            continue
        dq = _maybe_float(metrics.get("DockQ"))
        if dq is None:
            continue
        if best is not None and dq <= best["dockq"]:
            continue
        # DockQ exposes these keys in mixed case across versions; try
        # both spellings before giving up on a value.
        irmsd = _maybe_float(
            metrics.get("iRMSD") or metrics.get("irmsd"))
        lrmsd = _maybe_float(
            metrics.get("LRMSD") or metrics.get("lrmsd"))
        fnat = _maybe_float(
            metrics.get("fnat") or metrics.get("Fnat"))
        best = {
            "interface": iface_id,
            "irmsd": irmsd, "lrmsd": lrmsd,
            "fnat": fnat, "dockq": dq,
        }
    return best


def _compute_metrics_formula(pred_path: Path, ref_path: Path) -> dict:
    """Fallback path: derive iRMSD / LRMSD / Fnat / DockQ from gemmi-
    extracted atoms + the CAPRI formula. Used when ``_try_dockq_package``
    returns ``None``. Never raises; missing components surface as
    ``None`` in the returned dict."""
    try:
        pred_prot, pred_rna = extract_complex_atoms(pred_path)
        ref_prot, ref_rna = extract_complex_atoms(ref_path)
    except (FileNotFoundError, ValueError):
        return {"irmsd": None, "lrmsd": None,
                "fnat": None, "dockq": None}
    irmsd = compute_irmsd(pred_prot, pred_rna, ref_prot, ref_rna)
    lrmsd = compute_lrmsd(pred_prot, pred_rna, ref_prot, ref_rna)
    fnat = compute_fnat(pred_prot, pred_rna, ref_prot, ref_rna)
    dockq = compute_dockq(fnat, irmsd, lrmsd)
    return {"irmsd": irmsd, "lrmsd": lrmsd,
            "fnat": fnat, "dockq": dockq}


def evaluate_one(*, pred_path: Path, ref_path: Path,
                 usalign_bin: str, timeout: float) -> dict:
    """Full metric pack for one (sample, tool).

    TM-score always goes through US-align (US-align gives a global
    structural superposition score that DockQ doesn't compute). The
    docking-quality block delegates to the DockQ Python package when
    available, falling back to the manual formula otherwise. All keys
    are present in the return; ``None`` is the "couldn't compute"
    sentinel that propagates to the CSV cell."""
    tm = run_usalign_tmscore(usalign_bin, pred_path, ref_path,
                             timeout=timeout)
    metrics = _try_dockq_package(pred_path, ref_path)
    if metrics is None:
        metrics = _compute_metrics_formula(pred_path, ref_path)

    irmsd = metrics.get("irmsd")
    lrmsd = metrics.get("lrmsd")
    fnat = metrics.get("fnat")
    dockq = metrics.get("dockq")

    return {
        "tmscore": None if tm is None else round(tm, 4),
        "irmsd": None if irmsd is None else round(irmsd, 4),
        "lrmsd": None if lrmsd is None else round(lrmsd, 4),
        "fnat": None if fnat is None else round(fnat, 4),
        "dockq": None if dockq is None else round(dockq, 4),
        "acceptable": (1 if dockq is not None
                       and dockq >= DOCKQ_ACCEPTABLE else 0),
        "medium": (1 if dockq is not None
                   and dockq >= DOCKQ_MEDIUM else 0),
    }


# ---- consensus selection ------------------------------------------------


def select_consensus_tool(
    predictions: list[dict],
    enriched_probs: dict[int, float],
    consensus_tools: tuple[str, ...] = CAT_A_TOOLS,
) -> Optional[str]:
    """Pick the Cat A tool whose predicted binding residues have the
    highest mean ``enriched_probs`` value.

    Returns the tool_id or None if no Cat A tool qualifies (no
    successful prediction, no binding list, or no enriched probs).
    Ties broken in ``consensus_tools`` declaration order (deterministic)."""
    if not enriched_probs or not predictions:
        return None
    best_tid: Optional[str] = None
    best_score = -1.0
    for tid in consensus_tools:
        p = next((pp for pp in predictions
                  if pp.get("tool_id") == tid
                  and pp.get("success")
                  and pp.get("predicted_structure_path")), None)
        if p is None:
            continue
        binding = p.get("binding_protein_residues") or []
        if not binding:
            continue
        scores = [enriched_probs.get(int(r), 0.0) for r in binding]
        if not scores:
            continue
        score = sum(scores) / len(scores)
        if score > best_score:
            best_score = score
            best_tid = tid
    return best_tid


# ---- driver -------------------------------------------------------------


def evaluate(
    *,
    step4_dir: Path,
    processed_dir: Path,
    raw_dir: Path,
    sample_ids: Optional[list[str]],
    usalign_bin: str,
    enriched_model: Optional[EnrichedFusion],
    work_dir: Path,
    timeout: float,
    fs_preds: Optional[dict[str, dict[int, float]]] = None,
    complex_tools: tuple[str, ...] = COMPLEX_TOOLS,
    consensus_tools: tuple[str, ...] = CAT_A_TOOLS,
) -> tuple[dict[str, list[dict]], list[dict]]:
    """Returns ``({method: [per-sample dicts]}, per_sample_rows)``.

    ``work_dir`` holds the auto-built reference complex PDBs; reused
    across re-runs so a stuck US-align job can be retried cheaply.

    The ``riboseer_consensus`` pick uses ``fs_preds`` (pre-computed
    full-system per-residue predictions) when supplied, else
    ``enriched_model.predict_sample``."""
    if sample_ids is None:
        sample_ids = sorted(p.stem for p in step4_dir.glob("*.jsonl"))

    bucket: dict[str, list[dict]] = defaultdict(list)
    per_sample: list[dict] = []
    skipped: dict[str, int] = defaultdict(int)
    work_dir.mkdir(parents=True, exist_ok=True)

    for sid in sample_ids:
        s4 = _read_last_jsonl_record(step4_dir / f"{sid}.jsonl")
        if s4 is None:
            skipped["no_step4"] += 1
            continue
        sample = load_sample_json(processed_dir, sid)
        if sample is None:
            skipped["no_sample"] += 1
            continue
        src_pdb = sample.get("source_pdb")
        prot_node = sample.get("protein") or {}
        rna_node = sample.get("rna") or {}
        prot_chain = prot_node.get("chain_id")
        rna_chain = rna_node.get("chain_id")
        if not (src_pdb and prot_chain and rna_chain):
            skipped["missing_meta"] += 1
            continue

        ref_path = work_dir / f"{sid}_ref.pdb"
        if not ref_path.is_file():
            try:
                build_ref_complex_pdb(raw_dir, str(src_pdb),
                                      str(prot_chain), str(rna_chain),
                                      ref_path)
            except (FileNotFoundError, ValueError):
                skipped["ref_build_fail"] += 1
                continue

        predictions = s4.get("predictions") or []
        tool_results: dict[str, dict] = {}
        for p in predictions:
            tid = p.get("tool_id")
            if tid not in complex_tools:
                continue
            # No category gate: membership in ``complex_tools`` is the
            # contract. (Docking tools report category 'D' yet still emit
            # a 3-D complex — gating on 'A' silently dropped them.)
            if not p.get("success"):
                continue
            pred_path_str = p.get("predicted_structure_path")
            if not pred_path_str:
                continue
            pred_path = Path(pred_path_str)
            if not pred_path.is_file():
                skipped[f"{tid}_no_pred_file"] += 1
                continue
            try:
                metrics = evaluate_one(
                    pred_path=pred_path, ref_path=ref_path,
                    usalign_bin=usalign_bin, timeout=timeout)
            except (FileNotFoundError, ValueError):
                skipped[f"{tid}_eval_fail"] += 1
                continue
            tool_results[tid] = metrics
            row = {"sample_id": sid, **metrics}
            bucket[tid].append(row)
            per_sample.append({"sample_id": sid, "method": tid, **metrics})

        if (fs_preds is not None or enriched_model is not None) and tool_results:
            if fs_preds is not None:
                probs = fs_preds.get(sid) or {}
            else:
                prot_seq = prot_node.get("sequence") or ""
                prot_len = prot_node.get("length") or len(prot_seq)
                probs = enriched_model.predict_sample(
                    predictions, prot_seq, int(prot_len or 0))
            chosen = select_consensus_tool(predictions, probs,
                                           consensus_tools)
            if chosen and chosen in tool_results:
                m = tool_results[chosen]
                bucket["riboseer_consensus"].append(
                    {"sample_id": sid, "chosen_tool": chosen, **m})
                per_sample.append(
                    {"sample_id": sid, "method": "riboseer_consensus",
                     "chosen_tool": chosen, **m})

    if skipped:
        print("skipped counts:")
        for k, v in sorted(skipped.items()):
            print(f"  {k:24s} {v}")
    return bucket, per_sample


# ---- aggregate / CSV / print --------------------------------------------


def _aggregate(bucket: dict[str, list[dict]]) -> list[dict]:
    """One row per method. Cat A tools first (by n desc), consensus
    row appended last regardless of n."""
    rows: list[dict] = []
    for method, rs in bucket.items():
        tms = [r["tmscore"] for r in rs if r["tmscore"] is not None]
        ir = [r["irmsd"] for r in rs if r["irmsd"] is not None]
        lr = [r["lrmsd"] for r in rs if r["lrmsd"] is not None]
        fn = [r["fnat"] for r in rs if r["fnat"] is not None]
        dq = [r["dockq"] for r in rs if r["dockq"] is not None]
        acc = [r["acceptable"] for r in rs]
        med = [r["medium"] for r in rs]
        rows.append({
            "method": method,
            "n_samples": len(rs),
            "tmscore_mean": _mean(tms), "tmscore_median": _median(tms),
            "irmsd_mean": _mean(ir), "irmsd_median": _median(ir),
            "lrmsd_mean": _mean(lr), "lrmsd_median": _median(lr),
            "fnat_mean": _mean(fn), "fnat_median": _median(fn),
            "dockq_mean": _mean(dq), "dockq_median": _median(dq),
            "pct_acceptable": (round(100 * statistics.fmean(acc), 2)
                               if acc else None),
            "pct_medium": (round(100 * statistics.fmean(med), 2)
                           if med else None),
        })
    consensus = [r for r in rows if r["method"] == "riboseer_consensus"]
    tools = [r for r in rows if r["method"] != "riboseer_consensus"]
    tools.sort(key=lambda r: (-r["n_samples"], r["method"]))
    return tools + consensus


_COLUMNS = [
    "method", "n_samples",
    "tmscore_mean", "tmscore_median",
    "irmsd_mean", "irmsd_median",
    "lrmsd_mean", "lrmsd_median",
    "fnat_mean", "fnat_median",
    "dockq_mean", "dockq_median",
    "pct_acceptable", "pct_medium",
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in _COLUMNS})


def _write_per_sample_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["sample_id", "method", "chosen_tool",
            "tmscore", "irmsd", "lrmsd", "fnat", "dockq",
            "acceptable", "medium"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _print_table(rows: list[dict]) -> None:
    hdr = (f"{'method':22s} {'n':>4s} "
           f"{'TM':>7s} {'iRMSD':>8s} {'LRMSD':>8s} "
           f"{'Fnat':>7s} {'DockQ':>7s} "
           f"{'Acc%':>6s} {'Med%':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['method']:22s} {r['n_samples']:>4d} "
              f"{str(r['tmscore_mean']):>7s} "
              f"{str(r['irmsd_mean']):>8s} "
              f"{str(r['lrmsd_mean']):>8s} "
              f"{str(r['fnat_mean']):>7s} "
              f"{str(r['dockq_mean']):>7s} "
              f"{str(r['pct_acceptable']):>6s} "
              f"{str(r['pct_medium']):>6s}")


# ---- main ---------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step4-dir", type=Path, required=True)
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--sample-list", type=Path, default=None,
                   help="one sample_id per line; default = every "
                        "*.jsonl under --step4-dir.")
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--usalign-bin", default="USalign",
                   help="USalign binary on PATH or absolute path.")
    p.add_argument("--tools", default=None,
                   help="comma-separated tool_ids to evaluate; default = "
                        f"{','.join(COMPLEX_TOOLS)}.")
    p.add_argument("--consensus-tools", default=None,
                   help="comma-separated co-folding tool_ids the RiboSeer "
                        "consensus may pick from; default = "
                        f"{','.join(CAT_A_TOOLS)}. Pass e.g. "
                        "'boltz2,chai1,rosettafold2na,alphafold3,rfaa' to "
                        "let AF3/RFAA compete for the consensus row.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--enriched-model-dir", type=Path, default=None,
                   help="optional EnrichedFusion bundle — adds a "
                        "'riboseer_consensus' row (per-sample picks).")
    p.add_argument("--predictions-dir", type=Path, default=None,
                   help="pre-computed full-system (on/on/on) predictions "
                        "<sid>.json; when set the RiboSeer consensus pick "
                        "uses these instead of the model bundle.")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="cache dir for built reference complex PDBs; "
                        "defaults to <output_dir>/_ref_complexes/.")
    p.add_argument("--timeout", type=float, default=180.0,
                   help="USalign per-call timeout in seconds.")
    p.add_argument("--n-workers", type=int, default=1,
                   help="reserved for future parallelism — currently "
                        "serial (US-align dominates wall time per "
                        "sample but is fast for typical complex sizes).")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not args.step4_dir.is_dir():
        print(f"ERROR: --step4-dir not a directory: {args.step4_dir}",
              file=sys.stderr)
        return 1
    if not args.raw_dir.is_dir():
        print(f"ERROR: --raw-dir not a directory: {args.raw_dir}",
              file=sys.stderr)
        return 1
    try:
        sample_ids = _load_sample_ids(args.sample_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    fs_preds = None
    if args.predictions_dir is not None:
        fs_preds = load_predictions_dir(args.predictions_dir)
        if not fs_preds:
            print(f"ERROR: no predictions under {args.predictions_dir}",
                  file=sys.stderr)
            return 1
        print(f"loaded {len(fs_preds)} full-system predictions from "
              f"{args.predictions_dir}")

    enriched_model = None
    if args.enriched_model_dir is not None and fs_preds is None:
        try:
            enriched_model = EnrichedFusion.load(args.enriched_model_dir)
        except (OSError, ValueError, ImportError,
                json.JSONDecodeError) as e:
            print(f"ERROR: could not load --enriched-model-dir: {e}",
                  file=sys.stderr)
            return 1

    work_dir = args.work_dir or args.output.parent / "_ref_complexes"

    complex_tools = (tuple(t.strip() for t in args.tools.split(",")
                           if t.strip())
                     if args.tools else COMPLEX_TOOLS)
    consensus_tools = (tuple(t.strip() for t in
                             args.consensus_tools.split(",") if t.strip())
                       if args.consensus_tools else CAT_A_TOOLS)

    if not usalign_preflight(args.usalign_bin):
        print(f"WARNING: US-align binary not runnable: {args.usalign_bin!r}."
              " TM-score column will be empty. Other metrics (DockQ etc.) "
              "still compute. Fix with --usalign-bin <path/to/USalign>.",
              file=sys.stderr)

    bucket, per_sample = evaluate(
        step4_dir=args.step4_dir,
        processed_dir=args.processed_dir,
        raw_dir=args.raw_dir,
        sample_ids=sample_ids,
        usalign_bin=args.usalign_bin,
        enriched_model=enriched_model,
        work_dir=work_dir,
        timeout=args.timeout,
        fs_preds=fs_preds,
        complex_tools=complex_tools,
        consensus_tools=consensus_tools,
    )
    rows = _aggregate(bucket)
    _write_csv(args.output, rows)
    ps_path = args.output.with_name(
        args.output.stem + "_per_sample.csv")
    _write_per_sample_csv(ps_path, per_sample)

    print(f"wrote {args.output}  ({len(rows)} methods)")
    print(f"wrote {ps_path}  ({len(per_sample)} sample-method rows)")
    print()
    _print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
