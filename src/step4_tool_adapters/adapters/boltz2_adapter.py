"""Boltz-2 adapter (Category A — end-to-end RNA-protein structure prediction).

Workflow
--------
1. ``prepare_input``: write a Boltz-2 YAML at
   ``<work_dir>/<job_name>.yaml``. Format is exactly the one Boltz-2
   expects:

       sequences:
         - protein:
             id: A
             sequence: "..."
         - rna:
             id: B
             sequence: "..."

   No ``version:`` line — Boltz-2 rejects it. We always use chain A
   for protein and B for RNA so ``parse_output`` can look those up
   without any guessing.

   ``job_name`` is ``sample_id`` with non-alphanumeric chars replaced
   by underscore (sample IDs like ``2wj8_A_-a`` would otherwise become
   awkward filenames / job names).

2. ``run_tool``: ``boltz predict <yaml> --out_dir <out>
   --use_msa_server --no_kernels`` inside the ``boltz`` conda env.
   ``--use_msa_server`` triggers the MSA HTTP server so we don't need
   precomputed alignments; ``--no_kernels`` skips the cuEquivariance
   kernels that aren't available on every GPU (and don't even apply
   in CPU-only test runs).

3. ``parse_output``: Boltz-2 emits something like

       <out>/boltz_results_<job>/predictions/<job>/
           <job>_model_0.cif
           confidence_<job>_model_0.json
           [pae_<job>_model_0.npz]
           [plddt_<job>_model_0.npz]

   We rglob for the artefacts so we don't depend on the exact layout
   (Boltz layout has shifted between minor versions). From the mmCIF
   we extract:

   - protein-RNA contacts (4.5 Å heavy atom) via ``contact_extractor``
     on chains A / B, populating ``binding_protein_residues`` and
     ``binding_rna_nucleotides``.
   - per-residue pLDDT from CA B-factors on chain A.
   - ``plddt_mean``: prefer ``complex_plddt`` from the confidence
     JSON; fall back to mean of the per-residue values.

   From the confidence JSON we pull ``iptm`` (interface predicted TM)
   and ``complex_pde`` (predicted distance error, used here as the
   ``pae_mean`` proxy).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from ..base_adapter import BaseAdapter
from ..contact_extractor import extract_contacts
from ..schemas import ToolPrediction
from ..tool_runner import run_in_conda_env
from .structure_utils import compute_distance_binding_scores

try:
    import gemmi  # type: ignore
except ImportError as _e:  # pragma: no cover — required by environment.yml
    gemmi = None  # type: ignore
    _GEMMI_IMPORT_ERROR = _e
else:
    _GEMMI_IMPORT_ERROR = None

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore


# Same minimal AA classifier used by p2rank_adapter / contact_extractor.
_AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}


def _is_protein_residue(res_name: str) -> bool:
    return res_name.strip().upper() in _AA3


# ---------- prepare_input helpers -----------------------------------------


_NAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9_]")


def _sanitize_job_name(sample_id: str) -> str:
    """Make a sample_id safe to use as a filename / Boltz job name.

    Boltz uses the YAML basename as the job name, which appears in the
    output mmCIF filename. Some sample_ids contain ``-`` or other chars
    that we'd rather not have in filenames or paths."""
    cleaned = _NAME_SAFE_RE.sub("_", sample_id).strip("_")
    return cleaned or "sample"


_CUDA_DEVICE_RE = re.compile(r"^cuda:(\d+)$", re.IGNORECASE)


def _gpu_env_from_device(device: Optional[str]) -> dict:
    """Translate a ``device`` config string into env-var overrides.

    Why CUDA_VISIBLE_DEVICES rather than a CLI flag
    -----------------------------------------------
    Boltz-2's ``boltz predict`` CLI exposes ``--devices N`` (count of
    GPUs to use) but no ``--device cuda:N`` flag for picking a
    specific physical GPU. The standard cross-tool approach is the
    ``CUDA_VISIBLE_DEVICES`` env var: setting it to ``"0"`` makes the
    subprocess see only physical GPU 0, and Boltz will use that one.

    With Boltz-2 on cuda:0 + Chai-1 on cuda:1 (per the dual-A6000
    server layout), this isolates VRAM per tool and gets rid of the
    OOM failures that hit Chai-1 in the 500-sample batch.

    Parameters
    ----------
    device :
        Config value from ``tools.boltz2.device``. Accepts
        ``"cuda:N"`` (any case) — anything else (``"cpu"``, None,
        empty string, malformed) returns ``{}`` and the subprocess
        inherits the parent's CUDA visibility unchanged.
    """
    if not device:
        return {}
    m = _CUDA_DEVICE_RE.match(str(device).strip())
    if not m:
        return {}
    return {"CUDA_VISIBLE_DEVICES": m.group(1)}


# ---------- sequence cleaning ---------------------------------------------

# Standard 20 amino acids. Boltz-2's CCD lookup rejects gaps ("-"),
# ambiguous codes (X / B / Z), and selenocysteine / pyrrolysine (U / O).
# We also drop any other oddity rather than guessing.
_STANDARD_AA: set[str] = set("ACDEFGHIKLMNPQRSTVWY")
_STANDARD_RNA: set[str] = set("AUGC")

# Minimum lengths after cleaning. Below these thresholds we refuse to
# call Boltz-2 — the structure prediction would be either nonsensical
# (too few residues to form a fold) or an outright crash. ``BaseAdapter``
# folds the raised ValueError into ``success=False`` so the rest of the
# pipeline keeps running.
MIN_PROTEIN_LEN: int = 10
MIN_RNA_LEN: int = 3


def clean_protein_for_boltz(raw_seq: str) -> tuple[str, dict[int, int]]:
    """Drop non-standard residues; return cleaned seq + index mapping.

    Returns
    -------
    clean_seq : str
        Concatenation of every standard AA (uppercase) in ``raw_seq``.
    mapping : dict[int, int]
        ``clean_idx (1-based) → original_idx (1-based)``. Boltz-2
        numbers residues 1..len(clean_seq) in the predicted complex,
        so ``parse_output`` uses this to translate predicted residue
        IDs back to the dataset's original indexing (which the ground
        truth and other tools use).

    Examples
    --------
    >>> clean_protein_for_boltz("---MAGIC--K")
    ('MAGICK', {1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 11})
    """
    clean: list[str] = []
    mapping: dict[int, int] = {}
    for orig_idx, c in enumerate(raw_seq.upper(), start=1):
        if c in _STANDARD_AA:
            clean.append(c)
            mapping[len(clean)] = orig_idx
    return "".join(clean), mapping


def clean_rna_for_boltz(raw_seq: str) -> tuple[str, dict[int, int]]:
    """Same idea as ``clean_protein_for_boltz`` but for RNA.

    ``T`` is mapped to ``U`` (some PDBs annotate RNA chains with
    DNA-style T) — that's an in-place substitution, not a drop, so the
    original index is preserved for that position.
    """
    clean: list[str] = []
    mapping: dict[int, int] = {}
    for orig_idx, c in enumerate(raw_seq.upper(), start=1):
        if c == "T":
            clean.append("U")
            mapping[len(clean)] = orig_idx
        elif c in _STANDARD_RNA:
            clean.append(c)
            mapping[len(clean)] = orig_idx
    return "".join(clean), mapping


def remap_indices(
    indices: list[int], mapping: dict[int, int],
) -> list[int]:
    """Translate clean-indexed residue numbers back to original numbering.

    Indices the mapping doesn't cover are dropped (defensive: should
    never happen in practice, since Boltz-2 only emits residues 1..N
    where N = len(clean_seq), all of which are mapping keys). The
    output list is sorted ascending and de-duplicated.
    """
    out = {mapping[i] for i in indices if i in mapping}
    return sorted(out)


def remap_per_residue(
    per_res: dict[int, float], mapping: dict[int, int],
) -> dict[int, float]:
    """Translate ``{clean_idx: confidence}`` to the original indexing."""
    out: dict[int, float] = {}
    for clean_idx, val in per_res.items():
        orig_idx = mapping.get(clean_idx)
        if orig_idx is not None:
            out[orig_idx] = val
    return dict(sorted(out.items()))


# ---------- sequence-map sidecar I/O --------------------------------------

# Filename for the {clean_idx: orig_idx} mapping written next to the
# YAML. parse_output reads this back to translate Boltz-2's residue
# numbering into the dataset's original numbering. Plain JSON keeps the
# format human-readable for debugging.
_SEQ_MAP_FILENAME = "boltz2_seq_map.json"


def write_seq_map(
    work_dir: Path,
    protein_mapping: dict[int, int],
    rna_mapping: dict[int, int],
) -> Path:
    """Persist the clean→original index maps next to the YAML.

    JSON keys are stringified ints (json's only option for dict keys);
    ``read_seq_map`` reverses that to ``int → int``.
    """
    path = Path(work_dir) / _SEQ_MAP_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protein": {str(k): v for k, v in protein_mapping.items()},
        "rna": {str(k): v for k, v in rna_mapping.items()},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def read_seq_map(
    work_dir: Path,
) -> tuple[dict[int, int], dict[int, int]]:
    """Inverse of ``write_seq_map``.

    Returns ``({}, {})`` when the sidecar is missing — that's the
    backwards-compatible "no remap needed" path so old work_dirs (or
    callers that ran prepare_input without sequence cleaning) still
    parse correctly.
    """
    path = Path(work_dir) / _SEQ_MAP_FILENAME
    if not path.is_file():
        return {}, {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, {}
    if not isinstance(data, dict):
        return {}, {}
    prot = {int(k): int(v) for k, v in (data.get("protein") or {}).items()}
    rna  = {int(k): int(v) for k, v in (data.get("rna") or {}).items()}
    return prot, rna


def write_boltz_yaml(
    yaml_path: Path,
    protein_seq: str,
    rna_seq: str,
    *,
    protein_chain: str = "A",
    rna_chain: str = "B",
) -> Path:
    """Write a Boltz-2 input YAML.

    Hand-written rather than going through pyyaml because Boltz-2 is
    picky about the exact layout (and we want to be sure no
    ``version:`` line sneaks in).
    """
    if not protein_seq:
        raise ValueError("protein_seq is empty")
    if not rna_seq:
        raise ValueError("rna_seq is empty")
    if any(c in protein_seq for c in ('"', "\n", "\r")):
        raise ValueError("protein_seq contains illegal characters")
    if any(c in rna_seq for c in ('"', "\n", "\r")):
        raise ValueError("rna_seq contains illegal characters")

    yaml_text = (
        "sequences:\n"
        f"  - protein:\n"
        f"      id: {protein_chain}\n"
        f'      sequence: "{protein_seq}"\n'
        f"  - rna:\n"
        f"      id: {rna_chain}\n"
        f'      sequence: "{rna_seq}"\n'
    )
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


# ---------- parse_output helpers ------------------------------------------


def _find_first(output_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(output_dir.rglob(pattern))
    return matches[0] if matches else None


def _safe_float(v) -> Optional[float]:
    """Coerce ``v`` to float or return ``None`` for missing / non-numeric."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def parse_confidence_json(path: Path) -> dict:
    """Best-effort extraction of summary metrics from Boltz confidence JSON.

    Returns a dict with keys ``plddt`` / ``iptm`` / ``ptm`` / ``pde`` /
    ``iplddt`` / ``ipde``. Unknown / missing entries map to ``None``.
    Tolerates a missing or malformed file.
    """
    out = {
        "plddt": None, "iptm": None, "ptm": None,
        "pde": None, "iplddt": None, "ipde": None,
    }
    if not path or not Path(path).is_file():
        return out
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    if not isinstance(data, dict):
        return out

    out["plddt"] = _safe_float(data.get("complex_plddt"))
    out["iptm"] = _safe_float(data.get("iptm"))
    out["ptm"] = _safe_float(data.get("ptm"))
    out["pde"] = _safe_float(data.get("complex_pde"))
    out["iplddt"] = _safe_float(data.get("complex_iplddt"))
    out["ipde"] = _safe_float(data.get("complex_ipde"))
    return out


def read_protein_plddt_from_cif(
    cif_path: Path,
    *,
    chain_id: str = "A",
) -> dict[int, float]:
    """Read per-residue pLDDT from CA B-factors of the protein chain.

    Boltz-2 stores pLDDT in the B-factor field of every atom. We use
    the CA atom as the per-residue representative. Residue indices are
    taken from ``seqid.num`` (which equals ``label_seq`` for predicted
    structures because the input sequence has no gaps).
    """
    if gemmi is None:
        raise RuntimeError(f"gemmi not available: {_GEMMI_IMPORT_ERROR}")
    cif_path = Path(cif_path)
    if not cif_path.is_file():
        return {}
    try:
        structure = gemmi.read_structure(str(cif_path))
    except Exception:
        return {}
    if len(structure) == 0:
        return {}

    out: dict[int, float] = {}
    model = structure[0]
    for chain in model:
        if chain.name != chain_id:
            continue
        for res in chain:
            if not _is_protein_residue(res.name):
                continue
            ca = res.find_atom("CA", "*")
            if ca is None:
                continue
            idx = res.seqid.num
            if idx is None or idx < 1:
                continue
            out[idx] = round(float(ca.b_iso), 3)
    return dict(sorted(out.items()))


def read_pae_mean_from_npz(npz_path: Path) -> Optional[float]:
    """Read the mean of the PAE matrix from a Boltz pae_*.npz file."""
    if np is None or npz_path is None or not Path(npz_path).is_file():
        return None
    try:
        data = np.load(str(npz_path))
    except Exception:
        return None
    for key in ("pae", "predicted_aligned_error", "PAE"):
        if key in data.files:
            try:
                return float(np.mean(data[key]))
            except Exception:
                return None
    if len(data.files) == 1:
        try:
            return float(np.mean(data[data.files[0]]))
        except Exception:
            return None
    return None


def read_pae_matrix_from_npz(npz_path: Path):
    """Read the full PAE matrix from a Boltz pae_*.npz file.

    Returns the underlying ``np.ndarray`` (typically ``[N_total, N_total]``
    where ``N_total = N_protein + N_rna`` after sequence cleaning), or
    ``None`` on missing / unreadable input. Same key precedence as
    ``read_pae_mean_from_npz`` so a future Boltz version that renames the
    array key keeps working.
    """
    if np is None or npz_path is None or not Path(npz_path).is_file():
        return None
    try:
        data = np.load(str(npz_path))
    except Exception:
        return None
    for key in ("pae", "predicted_aligned_error", "PAE"):
        if key in data.files:
            try:
                arr = np.asarray(data[key])
            except Exception:
                return None
            return arr
    if len(data.files) == 1:
        try:
            return np.asarray(data[data.files[0]])
        except Exception:
            return None
    return None


def compute_pae_binding_scores(
    pae_matrix,
    n_protein: int,
    n_rna: int,
    *,
    pae_scale: float = 8.0,
) -> dict[int, float]:
    """Per-protein-residue binding probability derived from the PAE matrix.

    For each protein residue ``i``, take the minimum PAE to ANY RNA
    residue and convert via ``prob = 1 / (1 + min_pae / pae_scale)``.
    PAE is in Å (lower = more confident relative position), so this
    yields a probability in (0, 1] that decays smoothly with PAE.

    Layout assumption (verify per-tool when adding new Cat A adapters)
    ------------------------------------------------------------------
    Both Boltz-2 and Chai-1 emit a square ``[N_total, N_total]`` PAE
    matrix where ``N_total = N_protein + N_rna`` (sequences post-
    cleaning, in the order ``prepare_input`` wrote them — protein
    first, RNA second). Rows/cols ``[0..n_protein)`` are protein,
    ``[n_protein..n_total)`` are RNA. If a future tool reverses the
    chain ordering or interleaves chains, callers must permute first.

    Tunable
    -------
    ``pae_scale=8`` — at PAE = 8 Å, ``prob = 0.5``; at PAE = 0,
    ``prob ≈ 1.0``; at PAE = 30 Å, ``prob ≈ 0.21``. Scale chosen to
    sit just below the 12-15 Å interface PAE cutoff used in pDockQ /
    ipSAE. Bump ``pae_scale`` higher to be more lenient with weak
    interfaces; lower to weight only confidently-modelled contacts.

    Returns
    -------
    ``{protein_residue_idx (1-based, in CLEANED-sequence numbering):
       probability}``. Returns an empty dict on degenerate input
    (None matrix, wrong rank, smaller than ``n_protein + n_rna``,
    or no RNA columns).

    Callers that need original-sequence numbering should remap the
    keys with ``remap_per_residue`` and the protein index map.
    """
    if np is None or pae_matrix is None:
        return {}
    arr = np.asarray(pae_matrix)
    if arr.ndim != 2:
        return {}
    if n_protein <= 0 or n_rna <= 0:
        return {}
    needed = n_protein + n_rna
    if arr.shape[0] < needed or arr.shape[1] < needed:
        return {}
    if not (pae_scale > 0):
        return {}

    inter = arr[:n_protein, n_protein:needed]
    if inter.size == 0:
        return {}
    min_pae = inter.min(axis=1)
    out: dict[int, float] = {}
    for i, mp in enumerate(min_pae, start=1):
        prob = 1.0 / (1.0 + float(mp) / pae_scale)
        # Clamp defensively — negative PAE never occurs in practice but
        # would push prob > 1; gemmi-style float drift around 0 is
        # plausible.
        if prob < 0.0:
            prob = 0.0
        elif prob > 1.0:
            prob = 1.0
        out[i] = round(prob, 6)
    return out


# ---------- adapter --------------------------------------------------------


class Boltz2Adapter(BaseAdapter):
    """Adapter for Boltz-2 (Cat A)."""

    tool_id = "boltz2"
    category = "A"

    PROTEIN_CHAIN = "A"
    RNA_CHAIN = "B"

    # ------------------------------------------------------------------ prepare

    def prepare_input(
        self,
        sample_json: dict,
        work_dir: Path,
        config: dict,
    ) -> dict:
        sample_id = sample_json["sample_id"]
        protein_seq = (sample_json.get("protein") or {}).get("sequence")
        rna_seq = (sample_json.get("rna") or {}).get("sequence")
        if not protein_seq:
            raise ValueError(f"sample {sample_id!r} has no protein.sequence")
        if not rna_seq:
            raise ValueError(f"sample {sample_id!r} has no rna.sequence")

        # Boltz-2's CCD lookup rejects gaps and modified residues
        # (causes ``ValueError: CCD component - not found!`` deep in
        # the model). Strip non-standard symbols from BOTH sequences;
        # protein sequences with alignment gaps ("-") are the common
        # offender (e.g. 2bgg_A_P has 32 gaps in 427 aa). Each helper
        # returns a clean→original index map so ``parse_output`` can
        # translate Boltz-2's 1..N residue numbering back to the
        # dataset's original indexing (used by ground truth and the
        # other tools).
        protein_clean, protein_map = clean_protein_for_boltz(protein_seq)
        protein_dropped = len(protein_seq) - len(protein_clean)
        if protein_dropped > 0:
            logger.info(
                "boltz2[%s] protein cleaned: %d → %d aa (dropped %d "
                "non-standard residues)",
                sample_id, len(protein_seq), len(protein_clean), protein_dropped,
            )
        if len(protein_clean) < MIN_PROTEIN_LEN:
            raise ValueError(
                f"sample {sample_id!r}: protein.sequence too short after "
                f"cleaning ({len(protein_clean)} aa < min {MIN_PROTEIN_LEN}; "
                f"original len {len(protein_seq)}, dropped "
                f"{protein_dropped} non-standard residues)"
            )

        rna_clean, rna_map = clean_rna_for_boltz(rna_seq)
        rna_dropped = len(rna_seq) - len(rna_clean)
        if rna_dropped > 0:
            logger.info(
                "boltz2[%s] rna cleaned: %d → %d nt (dropped %d "
                "non-standard bases)",
                sample_id, len(rna_seq), len(rna_clean), rna_dropped,
            )
        if len(rna_clean) < MIN_RNA_LEN:
            raise ValueError(
                f"sample {sample_id!r}: rna.sequence too short after "
                f"cleaning ({len(rna_clean)} nt < min {MIN_RNA_LEN}; "
                f"original len {len(rna_seq)}, dropped {rna_dropped} "
                f"non-standard bases)"
            )

        job_name = _sanitize_job_name(sample_id)
        yaml_path = work_dir / f"{job_name}.yaml"
        write_boltz_yaml(
            yaml_path, protein_clean, rna_clean,
            protein_chain=self.PROTEIN_CHAIN, rna_chain=self.RNA_CHAIN,
        )
        # Persist the index maps so ``parse_output`` can find them
        # (parse_output's contract only takes ``output_dir`` so we can't
        # pass them via the input_paths dict).
        write_seq_map(work_dir, protein_map, rna_map)
        return {
            "yaml": yaml_path,
            "job_name": job_name,
            "protein_index_map": protein_map,
            "rna_index_map": rna_map,
        }

    # ------------------------------------------------------------------ run

    def run_tool(
        self,
        input_paths: dict,
        work_dir: Path,
        config: dict,
    ) -> Path:
        tool_cfg = (config.get("tools") or {}).get("boltz2") or {}
        env_name = tool_cfg.get("conda_env", "boltz")
        flags = tool_cfg.get("flags", "--use_msa_server --no_kernels")
        timeout = int(tool_cfg.get("timeout", 1800))

        yaml_path = Path(input_paths["yaml"]).resolve()
        if not yaml_path.is_file():
            raise FileNotFoundError(f"prepared YAML missing: {yaml_path}")

        output_dir = (work_dir / "boltz2_output").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = (
            f"boltz predict {yaml_path} "
            f"--out_dir {output_dir} {flags}".strip()
        )
        log_dir_cfg = config.get("log_dir")
        log_dir = Path(log_dir_cfg) if log_dir_cfg else None
        log_tag = f"boltz2_{yaml_path.stem}"

        # Pin Boltz-2 to a specific GPU when ``device: cuda:N`` is set,
        # so it doesn't compete for VRAM with whichever GPU Chai-1 was
        # configured for. See _gpu_env_from_device for the rationale.
        gpu_env = _gpu_env_from_device(tool_cfg.get("device"))

        result = run_in_conda_env(
            env_name, cmd, timeout=timeout,
            log_dir=log_dir, log_tag=log_tag,
            extra_env=gpu_env or None,
        )
        if not result.success:
            raise RuntimeError(
                f"Boltz-2 failed (rc={result.returncode}, "
                f"timed_out={result.timed_out}). "
                f"See log: {result.log_path}. "
                f"stderr: {result.stderr[:500]}"
            )
        return output_dir

    # ------------------------------------------------------------------ parse

    def parse_output(
        self,
        output_dir: Path,
        sample_json: dict,
        config: dict,
    ) -> ToolPrediction:
        sample_id = sample_json["sample_id"]
        cutoff = float(config.get("contact_threshold", 4.5))

        cif_path = _find_first(output_dir, "*_model_0.cif")
        if cif_path is None:
            return self.fail(
                sample_id,
                f"no *_model_0.cif found under {output_dir}",
                raw_output_dir=str(output_dir),
            )

        # Heavy-atom contacts from the predicted complex.
        contacts = extract_contacts(
            cif_path,
            cutoff=cutoff,
            protein_chain_id=self.PROTEIN_CHAIN,
            rna_chain_id=self.RNA_CHAIN,
        )

        # Per-residue pLDDT (protein chain, CA B-factor).
        per_res_plddt = read_protein_plddt_from_cif(
            cif_path, chain_id=self.PROTEIN_CHAIN,
        )

        # Translate Boltz-2's 1..N residue numbering (which counts the
        # CLEANED sequence) back to the dataset's original numbering so
        # downstream consumers — ground truth, fusion, scoring, the
        # other tools — see consistent residue IDs. The sidecar is
        # written by ``prepare_input`` next to the YAML; its parent dir
        # is ``output_dir.parent``. Empty maps from ``read_seq_map`` =
        # no remap needed (backwards compat with old work_dirs).
        protein_map, rna_map = read_seq_map(output_dir.parent)
        if protein_map:
            contacts.binding_protein_residues = remap_indices(
                contacts.binding_protein_residues, protein_map,
            )
            per_res_plddt = remap_per_residue(per_res_plddt, protein_map)
        if rna_map:
            contacts.binding_rna_nucleotides = remap_indices(
                contacts.binding_rna_nucleotides, rna_map,
            )

        # Confidence summary.
        conf_json_path = _find_first(output_dir, "confidence_*_model_0.json")
        conf = parse_confidence_json(conf_json_path) if conf_json_path else {}

        # plddt_mean: prefer JSON's complex_plddt; fallback to per-residue mean.
        plddt_mean = conf.get("plddt") if isinstance(conf, dict) else None
        if plddt_mean is None and per_res_plddt:
            plddt_mean = round(
                sum(per_res_plddt.values()) / len(per_res_plddt), 3,
            )

        iptm_score = conf.get("iptm") if isinstance(conf, dict) else None

        # pae_mean: try complex_pde from JSON, then fall back to npz file.
        # Also keep the npz path around so the PAE-derived per-residue
        # binding score can read the full matrix below — the JSON only
        # carries the scalar mean.
        npz_path = _find_first(output_dir, "pae_*_model_0.npz")
        pae_mean = conf.get("pde") if isinstance(conf, dict) else None
        if pae_mean is None and npz_path is not None:
            pae_mean = read_pae_mean_from_npz(npz_path)

        # Per-residue binding-probability score. The field name on
        # ToolPrediction (``per_residue_pae_score``) is historical —
        # the actual score source is now selectable via the
        # ``per_residue_score_method`` config knob:
        #
        #   - ``"distance"`` (default): CA → nearest RNA heavy-atom
        #     distance through ``1/(1 + d/distance_scale)``. Same
        #     mechanism Chai-1 uses; offline runs showed Boltz-2's
        #     Pearson R climbing 0.14 → ~0.35 vs the PAE matrix path,
        #     so this is the new default.
        #   - ``"pae"``: original PAE-matrix-derived score (kept for
        #     A/B comparison and backward-compat with old configs /
        #     bundles).
        #
        # ``compute_pae_scores: false`` (legacy knob) still disables
        # the whole block.
        per_res_pae: dict[int, float] = {}
        tool_cfg = (config.get("tools") or {}).get("boltz2") or {}
        score_method = str(
            tool_cfg.get("per_residue_score_method", "distance")
        ).strip().lower()
        if (bool(tool_cfg.get("compute_pae_scores", True))
                and protein_map):
            if score_method == "pae":
                # Original path: needs PAE matrix on disk + RNA map.
                if npz_path is not None and rna_map:
                    pae_scale = float(tool_cfg.get("pae_scale", 8.0))
                    n_protein_clean = max(protein_map.keys())
                    n_rna_clean = max(rna_map.keys())
                    pae_matrix = read_pae_matrix_from_npz(npz_path)
                    per_res_pae_clean = compute_pae_binding_scores(
                        pae_matrix, n_protein_clean, n_rna_clean,
                        pae_scale=pae_scale,
                    )
                    per_res_pae = remap_per_residue(
                        per_res_pae_clean, protein_map,
                    )
            else:
                # ``score_method == "distance"`` (or anything else /
                # default) — geometry-based score from the predicted CIF.
                # No PAE matrix or rna_map needed; the helper walks
                # the chains directly.
                distance_scale = float(tool_cfg.get("distance_scale", 8.0))
                per_res_clean = compute_distance_binding_scores(
                    cif_path,
                    protein_chain_id=self.PROTEIN_CHAIN,
                    rna_chain_id=self.RNA_CHAIN,
                    distance_scale=distance_scale,
                )
                per_res_pae = remap_per_residue(per_res_clean, protein_map)

        # Clamp pLDDT to schema range; some predictors emit slightly >100
        # due to float quirks.
        if plddt_mean is not None:
            plddt_mean = max(0.0, min(100.0, plddt_mean))

        return ToolPrediction(
            tool_id=self.tool_id,
            category=self.category,
            sample_id=sample_id,
            success=True,
            binding_protein_residues=contacts.binding_protein_residues or None,
            binding_rna_nucleotides=contacts.binding_rna_nucleotides or None,
            per_residue_confidence=per_res_plddt or None,
            per_residue_pae_score=per_res_pae or None,
            predicted_structure_path=str(Path(cif_path).resolve()),
            plddt_mean=plddt_mean,
            iptm_score=iptm_score,
            pae_mean=pae_mean,
            raw_output_dir=str(output_dir),
        )
