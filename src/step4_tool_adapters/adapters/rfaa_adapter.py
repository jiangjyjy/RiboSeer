"""RoseTTAFold-All-Atom (RFAA) adapter (Category A — RNA-protein structure
prediction).

Why RFAA (extra baseline, not mandatory)
-----------------------------------------
RFAA is RoseTTAFold-All-Atom run in single-sequence mode (no MSA search).
It's a regular (non-mandatory) member of the library (paper Table 1), so
MAESTRO may select it and RELAY dispatches it like any other Cat A tool.
``rerun_single_tool.py --tool rfaa`` runs it on its own over an existing
batch.

Workflow
--------
1. ``prepare_input``: clean the protein / RNA sequences with the same
   helpers Boltz-2 / Chai-1 use (drop gaps + non-standard residues,
   remember a clean→original index map) and write one single-record
   FASTA each. RFAA's Hydra config consumes them via
   ``+protein_inputs.A.fasta_file`` / ``+na_inputs.B.fasta``. The
   index map is persisted as a sidecar so ``parse_output`` can translate
   RFAA's 1..N residue numbering back to the dataset's original numbering.

2. ``run_tool`` (single-sequence, no MSA / template DB)::

       cd <install_dir>
       conda run -n RFAA --no-capture-output python -m rf2aa.run_inference \\
           --config-path=<install_dir>/rf2aa/config/inference \\
           --config-name=base \\
           job_name=<job> output_path=<work>/rfaa_output \\
           checkpoint_path=<weights> \\
           +protein_inputs.A.fasta_file=<protein.fa> \\
           +na_inputs.B.fasta=<rna.fa> +na_inputs.B.input_type=rna \\
           database_params.command="" database_params.sequencedb="" \\
           database_params.hhdb="" loader_params.n_templ=1

   The empty ``database_params.*`` overrides + ``n_templ=1`` are what
   put RFAA in the MSA-free / template-free single-sequence mode that
   was validated on the server.

3. ``parse_output``: RFAA writes ``<output_dir>/<job>.pdb`` (predicted
   complex, protein chain A + RNA chain B, pLDDT in the B-factor column)
   and ``<output_dir>/<job>_aux.pt`` (torch dict with ``plddts`` /
   ``pae`` / ``mean_plddt`` / ``pae_inter``). We extract, in the same
   shape as Boltz-2 / Chai-1:

   - protein-RNA contacts (4.5 Å heavy atom) via ``contact_extractor``
   - per-residue pLDDT from CA B-factors (chain A)
   - per-residue distance binding score (CA → nearest RNA heavy atom)
     via the shared ``compute_distance_binding_scores``
   - ``plddt_mean`` (prefer ``mean_plddt`` from the aux file, else mean
     of the per-residue pLDDT) and ``pae_mean`` (mean of the aux PAE
     matrix). The aux reader degrades cleanly when torch isn't installed
     in the parsing environment — the B-factor pLDDT path needs only
     gemmi, so the core prediction still parses.
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
from .boltz2_adapter import (
    MIN_PROTEIN_LEN, MIN_RNA_LEN,
    _gpu_env_from_device,
    clean_protein_for_boltz, clean_rna_for_boltz,
    read_protein_plddt_from_cif,
    remap_indices, remap_per_residue,
)
from .structure_utils import compute_distance_binding_scores

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore


# ---------- prepare_input helpers -----------------------------------------


_NAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9_]")


def _sanitize_job_name(sample_id: str) -> str:
    """Make a sample_id safe to use as a filename / RFAA Hydra job name.

    RFAA names its output ``<job_name>.pdb`` / ``<job_name>_aux.pt``, so
    the job name has to be filesystem-safe; sample IDs with ``-`` or
    other oddities would otherwise produce awkward paths.
    """
    cleaned = _NAME_SAFE_RE.sub("_", sample_id).strip("_")
    return cleaned or "sample"


def write_fasta(
    path: Path, target: str, sequence: str, *, line_width: int = 60,
) -> Path:
    """Write a one-record FASTA. Self-contained copy (kept independent of
    the RF2NA adapter so the two unrelated baselines don't share code by
    accident)."""
    if not sequence:
        raise ValueError("sequence is empty")
    if any(c in sequence for c in ("\n", "\r", " ", "\t")):
        raise ValueError("sequence contains whitespace")
    lines = [f">{target}"]
    if line_width and line_width > 0:
        for i in range(0, len(sequence), line_width):
            lines.append(sequence[i : i + line_width])
    else:
        lines.append(sequence)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------- sequence-map sidecar I/O --------------------------------------

# Distinct filename so a work_dir shared with another Cat A adapter
# (doesn't happen today, but might) doesn't silently swap maps. Same
# on-disk JSON shape as boltz2 / chai1 for parity.
_SEQ_MAP_FILENAME = "rfaa_seq_map.json"


def write_seq_map(
    work_dir: Path,
    protein_mapping: dict[int, int],
    rna_mapping: dict[int, int],
) -> Path:
    """Persist the clean→original index maps next to the FASTAs."""
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
    """Inverse of ``write_seq_map``. Empty dicts on missing / malformed."""
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
    rna = {int(k): int(v) for k, v in (data.get("rna") or {}).items()}
    return prot, rna


# ---------- parse_output helpers ------------------------------------------


def _find_first(output_dir: Path, pattern: str) -> Optional[Path]:
    """First match for ``pattern`` under ``output_dir`` (rglob)."""
    matches = sorted(output_dir.rglob(pattern))
    return matches[0] if matches else None


def _aux_to_array(value):
    """Coerce a torch tensor / numpy array / nested list to a float
    ndarray, or ``None`` when that's not possible."""
    if value is None or np is None:
        return None
    try:
        if hasattr(value, "detach"):  # torch.Tensor
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=float)
    except Exception:  # noqa: BLE001
        return None


def _aux_to_float(value) -> Optional[float]:
    """Coerce an aux scalar (torch 0-d tensor / numpy scalar / python
    number) to a float, or ``None``."""
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):  # torch.Tensor
            value = value.detach().cpu()
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except Exception:  # noqa: BLE001
        arr = _aux_to_array(value)
        if arr is not None and arr.size:
            return float(arr.mean())
        return None


def read_rfaa_aux(aux_path: Optional[Path]) -> dict:
    """Read pLDDT / PAE from RFAA's ``<job>_aux.pt`` (a ``torch.save`` dict).

    Returns a dict with keys ``per_residue_plddt`` / ``plddt_mean`` /
    ``pae_mean`` / ``pae_inter`` (any of which may be ``None``).

    Best-effort and never raises: a missing file, a numpy-less env, a
    torch-less parsing env, or a malformed payload all yield all-``None``.
    torch is imported lazily because the parsing environment (the
    ``pocket`` conda env) may not have it — when it doesn't, the caller
    falls back to the B-factor pLDDT, which only needs gemmi.
    """
    out = {
        "per_residue_plddt": None,
        "plddt_mean": None,
        "pae_mean": None,
        "pae_inter": None,
    }
    if aux_path is None or not Path(aux_path).is_file():
        return out
    try:
        import torch  # type: ignore
    except ImportError:
        return out
    try:
        try:
            data = torch.load(
                str(aux_path), map_location="cpu", weights_only=False,
            )
        except TypeError:
            # Older torch has no weights_only kwarg.
            data = torch.load(str(aux_path), map_location="cpu")
    except Exception:  # noqa: BLE001
        return out
    if not isinstance(data, dict):
        return out

    # Per-residue pLDDT tensor (covers all tokens; index from 1). Kept
    # for completeness / debugging — parse_output prefers the B-factor
    # pLDDT since that's already restricted to the protein chain.
    plddts = _aux_to_array(data.get("plddts"))
    if plddts is not None and plddts.size:
        flat = plddts.reshape(-1)
        out["per_residue_plddt"] = {
            i: round(float(v), 3) for i, v in enumerate(flat, start=1)
        }

    out["plddt_mean"] = _aux_to_float(data.get("mean_plddt"))
    if out["plddt_mean"] is None and plddts is not None and plddts.size:
        out["plddt_mean"] = round(float(plddts.mean()), 3)

    pae = _aux_to_array(data.get("pae"))
    if pae is not None and pae.size:
        out["pae_mean"] = float(pae.mean())

    out["pae_inter"] = _aux_to_float(data.get("pae_inter"))
    return out


# ---------- adapter --------------------------------------------------------


class RFAAAdapter(BaseAdapter):
    """Adapter for RoseTTAFold-All-Atom (Cat A, single-sequence mode)."""

    tool_id = "rfaa"
    category = "A"

    # RFAA's command labels protein as chain A, RNA as chain B
    # (+protein_inputs.A / +na_inputs.B), so the output PDB carries those
    # chain names. Same convention as Boltz-2 / Chai-1 / RF2NA.
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

        # Drop gaps + non-standard residues (same helpers Boltz-2 / Chai-1
        # use) and keep a clean→original index map so parse_output can
        # translate RFAA's 1..N numbering back to the dataset's numbering.
        protein_clean, protein_map = clean_protein_for_boltz(protein_seq)
        protein_dropped = len(protein_seq) - len(protein_clean)
        if protein_dropped > 0:
            logger.info(
                "rfaa[%s] protein cleaned: %d → %d aa (dropped %d "
                "non-standard residues)",
                sample_id, len(protein_seq), len(protein_clean),
                protein_dropped,
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
                "rfaa[%s] rna cleaned: %d → %d nt (dropped %d "
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
        protein_fa = work_dir / f"{job_name}_protein.fa"
        rna_fa = work_dir / f"{job_name}_rna.fa"
        # line_width=0 → sequence on a SINGLE line, never wrapped. RFAA's
        # parse_multichain_fasta treats every line after the header as a
        # separate sequence record, so a wrapped 60-char line would be
        # mis-read as multiple chains and crash. Both FASTAs must stay
        # one-line-per-sequence.
        write_fasta(protein_fa, f"{job_name}_protein", protein_clean,
                    line_width=0)
        write_fasta(rna_fa, f"{job_name}_rna", rna_clean, line_width=0)
        write_seq_map(work_dir, protein_map, rna_map)

        return {
            "protein_fa": protein_fa,
            "rna_fa": rna_fa,
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
        tool_cfg = (config.get("tools") or {}).get("rfaa") or {}
        install_dir = tool_cfg.get("install_dir")
        if not install_dir:
            raise ValueError(
                "config['tools']['rfaa']['install_dir'] not set"
            )
        install_dir = Path(install_dir)
        env_name = tool_cfg.get("conda_env", "RFAA")
        timeout = int(tool_cfg.get("timeout", 1200))

        protein_fa = Path(input_paths["protein_fa"]).resolve()
        rna_fa = Path(input_paths["rna_fa"]).resolve()
        if not protein_fa.is_file():
            raise FileNotFoundError(f"protein FASTA missing: {protein_fa}")
        if not rna_fa.is_file():
            raise FileNotFoundError(f"RNA FASTA missing: {rna_fa}")

        job_name = input_paths.get("job_name") or "sample"

        output_dir = (work_dir / "rfaa_output").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        def _resolve(rel: str) -> Path:
            p = Path(rel)
            return p if p.is_absolute() else (install_dir / p)

        config_path = _resolve(
            tool_cfg.get("config_path", "rf2aa/config/inference")
        ).resolve()
        config_name = tool_cfg.get("config_name", "base")
        weights = _resolve(
            tool_cfg.get("model_weights", "RFAA_paper_weights.pt")
        ).resolve()

        # Single-sequence inference: empty database_params.* + n_templ=1
        # skip the MSA search and template lookup (the validated mode).
        parts = [
            "python", "-m", "rf2aa.run_inference",
            f"--config-path={config_path}",
            f"--config-name={config_name}",
            f"job_name={job_name}",
            f"output_path={output_dir}",
            f"checkpoint_path={weights}",
            f"+protein_inputs.A.fasta_file={protein_fa}",
            f"+na_inputs.B.fasta={rna_fa}",
            "+na_inputs.B.input_type=rna",
            'database_params.command=""',
            'database_params.sequencedb=""',
            'database_params.hhdb=""',
            "loader_params.n_templ=1",
        ]
        extra = tool_cfg.get("extra_args")
        if extra:
            parts += (
                list(extra) if isinstance(extra, (list, tuple))
                else str(extra).split()
            )
        cmd = " ".join(str(p) for p in parts)

        log_dir_cfg = config.get("log_dir")
        log_dir = Path(log_dir_cfg) if log_dir_cfg else None
        log_tag = f"rfaa_{job_name}"

        # Pin RFAA to a specific GPU when ``device: cuda:N`` is set
        # (CUDA_VISIBLE_DEVICES) — same mechanism Boltz-2 uses so the
        # baselines don't fight over VRAM.
        gpu_env = _gpu_env_from_device(tool_cfg.get("device"))

        result = run_in_conda_env(
            env_name, cmd,
            cwd=str(install_dir),
            timeout=timeout,
            log_dir=log_dir,
            log_tag=log_tag,
            extra_env=gpu_env or None,
        )
        if not result.success:
            raise RuntimeError(
                f"RFAA failed (rc={result.returncode}, "
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
        job_name = _sanitize_job_name(sample_id)

        # RFAA writes <output_dir>/<job>.pdb; probe that first, then any
        # *.pdb so a layout shift doesn't sink the parse.
        pdb_path = (_find_first(output_dir, f"{job_name}.pdb")
                    or _find_first(output_dir, "*.pdb"))
        if pdb_path is None:
            return self.fail(
                sample_id,
                f"no predicted .pdb ({job_name}.pdb / *.pdb) found under "
                f"{output_dir}",
                raw_output_dir=str(output_dir),
            )

        # Heavy-atom protein-RNA contacts.
        contacts = extract_contacts(
            pdb_path,
            cutoff=cutoff,
            protein_chain_id=self.PROTEIN_CHAIN,
            rna_chain_id=self.RNA_CHAIN,
        )

        # Per-residue pLDDT (CA B-factor on the protein chain). gemmi
        # reads PDB the same way it reads mmCIF, so the boltz2 helper is
        # reusable verbatim despite its name.
        per_res_plddt = read_protein_plddt_from_cif(
            pdb_path, chain_id=self.PROTEIN_CHAIN,
        )

        # Translate RFAA's 1..N (clean-sequence) numbering back to the
        # dataset's original numbering. Sidecar lives next to the FASTAs
        # (work_dir/), the parent of output_dir.
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

        # Aux file: <job>_aux.pt (plddts / pae / mean_plddt / pae_inter).
        aux_path = (_find_first(output_dir, f"{job_name}_aux.pt")
                    or _find_first(output_dir, "*_aux.pt"))
        aux = read_rfaa_aux(aux_path) if aux_path else {}

        # plddt_mean: prefer the aux mean_plddt; else mean of the
        # per-residue B-factor pLDDT.
        plddt_mean = aux.get("plddt_mean")
        if plddt_mean is None and per_res_plddt:
            plddt_mean = round(
                sum(per_res_plddt.values()) / len(per_res_plddt), 3,
            )

        pae_mean = aux.get("pae_mean")

        # Per-residue distance binding score — same mechanism as
        # Boltz-2 / Chai-1 (CA → nearest RNA heavy atom through
        # 1/(1 + d/distance_scale)). This is the field the fusion's
        # MAIN_SCORE_FIELD reads for Cat A. Best-effort: a malformed PDB
        # must not sink an otherwise-good prediction.
        per_res_binding: dict[int, float] = {}
        tool_cfg = (config.get("tools") or {}).get("rfaa") or {}
        if bool(tool_cfg.get("compute_distance_scores", True)):
            distance_scale = float(tool_cfg.get("distance_scale", 8.0))
            try:
                per_res_clean = compute_distance_binding_scores(
                    pdb_path,
                    protein_chain_id=self.PROTEIN_CHAIN,
                    rna_chain_id=self.RNA_CHAIN,
                    distance_scale=distance_scale,
                )
            except Exception:  # noqa: BLE001
                per_res_clean = {}
            per_res_binding = (
                remap_per_residue(per_res_clean, protein_map)
                if protein_map else per_res_clean
            )

        # RFAA does not emit ipTM. Clamp pLDDT to the schema range against
        # float drift.
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
            per_residue_pae_score=per_res_binding or None,
            predicted_structure_path=str(Path(pdb_path).resolve()),
            plddt_mean=plddt_mean,
            iptm_score=None,
            pae_mean=pae_mean,
            raw_output_dir=str(output_dir),
        )
