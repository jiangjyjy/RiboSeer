"""Chai-1 adapter (Category A — end-to-end RNA-protein structure prediction).

Why Chai-1 (replacing RoseTTAFold2NA)
-------------------------------------
RF2NA's MSA pipeline (mandatory; runs PSI-BLAST + HHblits over UniRef30
± BFD) costs 30-40 min per sample; with the 600 s batch budget the Cat A
slot was 100 % failures. Chai-1 ships a single-sequence inference path
(``use_msa_server=False``), so we can run Cat A in minutes instead of
half-hours and stay within the per-sample budget. The protein / RNA
sequences come from ``sample_json`` directly — no auxiliary MSA stage.

Workflow
--------
1. ``prepare_input``: write a Chai-1 chain-tagged FASTA at
   ``<work_dir>/<job_name>.fasta``::

       >protein|chain_A
       MKTVLAGICK
       >rna|chain_B
       GCCGGCCAU

   We always use chain ``A`` for protein, ``B`` for RNA so
   ``parse_output`` (and the contact extractor) can look them up
   without guessing. Sequences are cleaned with the same helpers
   ``boltz2_adapter`` uses (``clean_protein_for_boltz`` /
   ``clean_rna_for_boltz``) — Chai-1's tokenizer also rejects
   gaps / ambiguous codes, so dropping them makes both adapters
   share identical preprocessing semantics. The resulting
   clean→original index maps are persisted as a sidecar so
   ``parse_output`` can translate Chai-1's 1..N residue numbering
   back to the dataset's original numbering.

2. ``run_tool``: Chai-1 only ships a Python API (no CLI), so we
   materialise a tiny runner script at ``<work_dir>/run_chai1.py``
   that calls ``chai_lab.chai1.run_inference`` and exits 0 / 1
   based on success. The script is then executed inside the
   ``chai1`` conda env. Generating the runner per-sample (rather
   than shipping a static one in the repo) keeps the runtime config
   — recycles, diffusion timesteps, device, MSA toggle — visible
   in the per-sample work_dir for post-mortem debugging.

3. ``parse_output``: Chai-1 writes ``pred.model_idx_0.cif`` plus a
   companion ``scores.model_idx_0.npz`` (summary metrics: ptm,
   iptm, per_chain_pair_iptm). Notably absent: a per-token PAE
   matrix on disk — that's why the ``per_residue_pae_score`` field
   is computed from geometry here rather than PAE (see
   ``compute_distance_binding_scores``). From the predicted mmCIF
   we extract:

   - protein-RNA contacts (4.5 Å heavy atom) via
     ``contact_extractor`` on chains A / B → ``binding_protein_residues``
     and ``binding_rna_nucleotides``.
   - per-residue pLDDT from CA B-factors on chain A (Chai-1, like
     Boltz-2, stores pLDDT in the B-factor column).
   - ``plddt_mean`` from the per-residue dict; ``pae_mean`` from
     the PAE numpy file when present.
   - ``per_residue_pae_score`` (despite the field name) from
     ``compute_distance_binding_scores``: CA → nearest RNA heavy
     atom distance, mapped through ``1/(1 + d/distance_scale)``
     so the resulting [0, 1] scores are directly comparable to
     Boltz-2's PAE-derived scores in evaluate.py.

   Indices are remapped from Chai-1's clean-sequence numbering back
   to the original dataset numbering using the sidecar.
"""
from __future__ import annotations

import json
import logging
import re
import shlex
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from ..base_adapter import BaseAdapter
from ..contact_extractor import extract_contacts
from ..schemas import ToolPrediction
from ..tool_runner import run_in_conda_env
from .boltz2_adapter import (
    MIN_PROTEIN_LEN, MIN_RNA_LEN,
    clean_protein_for_boltz, clean_rna_for_boltz,
    read_protein_plddt_from_cif,
    remap_indices, remap_per_residue,
)

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    import gemmi  # type: ignore
except ImportError as _e:  # pragma: no cover — required by environment.yml
    gemmi = None  # type: ignore
    _GEMMI_IMPORT_ERROR = _e
else:
    _GEMMI_IMPORT_ERROR = None


# ---------- prepare_input helpers -----------------------------------------


_NAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9_]")


def _sanitize_job_name(sample_id: str) -> str:
    """Make a sample_id safe to use as a filename / Chai-1 job name."""
    cleaned = _NAME_SAFE_RE.sub("_", sample_id).strip("_")
    return cleaned or "sample"


def write_chai1_fasta(
    fasta_path: Path,
    protein_seq: str,
    rna_seq: str,
    *,
    protein_chain: str = "A",
    rna_chain: str = "B",
) -> Path:
    """Write a Chai-1 chain-tagged FASTA.

    Format (from Chai-1's documented input shape)::

        >protein|chain_A
        MKTVLAGICK
        >rna|chain_B
        GCCGGCCAU

    Header convention is ``<entity_type>|chain_<chain_id>``. We don't
    use pyyaml or a templating library — the format is short and
    Chai-1 is picky about the header structure, so a hand-written
    string keeps it explicit.
    """
    if not protein_seq:
        raise ValueError("protein_seq is empty")
    if not rna_seq:
        raise ValueError("rna_seq is empty")
    if any(c in protein_seq for c in ("\n", "\r", " ", "\t")):
        raise ValueError("protein_seq contains whitespace")
    if any(c in rna_seq for c in ("\n", "\r", " ", "\t")):
        raise ValueError("rna_seq contains whitespace")

    text = (
        f">protein|chain_{protein_chain}\n"
        f"{protein_seq}\n"
        f">rna|chain_{rna_chain}\n"
        f"{rna_seq}\n"
    )
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    fasta_path.write_text(text, encoding="utf-8")
    return fasta_path


# ---------- sequence-map sidecar I/O --------------------------------------

# Distinct from boltz2's sidecar so a work_dir shared between adapters
# (which currently doesn't happen, but might in the future) doesn't
# silently swap maps. Same on-disk JSON shape as boltz2 for parity.
_SEQ_MAP_FILENAME = "chai1_seq_map.json"


def write_seq_map(
    work_dir: Path,
    protein_mapping: dict[int, int],
    rna_mapping: dict[int, int],
) -> Path:
    """Persist the clean→original index maps next to the FASTA."""
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
    rna  = {int(k): int(v) for k, v in (data.get("rna") or {}).items()}
    return prot, rna


# ---------- runner script generation --------------------------------------

# Chai-1 has no CLI; we materialise a tiny entry-point script per sample
# so the conda env can ``python <script>``. Keeping it generated (rather
# than shipping a static file) means the runtime config — device,
# recycles, diffusion timesteps — is captured next to the work_dir for
# every run, useful for post-mortem reproducibility.
_RUNNER_TEMPLATE = '''"""Auto-generated Chai-1 runner for sample {sample_id}.

Invoked by chai1_adapter.run_tool. Do not edit — this file is regenerated
on every adapter call. Exits 0 on success, 1 on any failure (with a
traceback on stderr).
"""
import sys
import traceback
from pathlib import Path


def main() -> int:
    fasta = Path({fasta!r})
    output_dir = Path({output_dir!r})
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from chai_lab.chai1 import run_inference
    except ImportError as e:
        print(f"chai_lab not importable in this env: {{e}}", file=sys.stderr)
        return 1

    try:
        run_inference(
            fasta_file=fasta,
            output_dir=output_dir,
            num_trunk_recycles={num_trunk_recycles},
            num_diffn_timesteps={num_diffn_timesteps},
            seed={seed},
            device={device!r},
            use_esm_embeddings={use_esm_embeddings},
            use_msa_server={use_msa_server},
        )
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def write_runner_script(
    work_dir: Path,
    sample_id: str,
    fasta_path: Path,
    output_dir: Path,
    *,
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    seed: int = 42,
    device: str = "cuda:0",
    use_esm_embeddings: bool = True,
    use_msa_server: bool = False,
) -> Path:
    """Generate the per-sample Chai-1 runner under ``<work_dir>/run_chai1.py``."""
    runner = Path(work_dir) / "run_chai1.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    text = _RUNNER_TEMPLATE.format(
        sample_id=sample_id,
        fasta=str(fasta_path.resolve()),
        output_dir=str(output_dir.resolve()),
        num_trunk_recycles=int(num_trunk_recycles),
        num_diffn_timesteps=int(num_diffn_timesteps),
        seed=int(seed),
        device=device,
        use_esm_embeddings=bool(use_esm_embeddings),
        use_msa_server=bool(use_msa_server),
    )
    runner.write_text(text, encoding="utf-8")
    return runner


# ---------- parse_output helpers ------------------------------------------


def _find_first(output_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(output_dir.rglob(pattern))
    return matches[0] if matches else None


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def read_pae_mean_from_npy(path: Path) -> Optional[float]:
    """Read the mean of the PAE matrix from a Chai-1 ``pae.model_idx_*.npy``.

    Chai-1 stores the predicted aligned error as a bare numpy array
    (rather than a keyed npz the way Boltz-2 does), so we ``np.load``
    and average the whole tensor. Returns ``None`` on missing /
    unreadable file or a numpy import failure (defensive — the env
    where the adapter runs always has numpy, but the test env might
    skip it).
    """
    if np is None or path is None or not Path(path).is_file():
        return None
    try:
        arr = np.load(str(path))
    except Exception:  # noqa: BLE001
        return None
    try:
        return float(np.mean(arr))
    except Exception:  # noqa: BLE001
        return None


# compute_distance_binding_scores has moved to ``structure_utils`` so
# Boltz-2 can call the same helper. Re-exported here so existing imports
# (``from chai1_adapter import compute_distance_binding_scores``) keep
# working.
from .structure_utils import compute_distance_binding_scores  # noqa: F401


def parse_chai1_scores_json(path: Path) -> dict:
    """Best-effort extraction of summary metrics from Chai-1 scores JSON.

    Chai-1's exact scores JSON shape has shifted across versions; we
    pull the fields we can and leave the rest as None. Common keys we
    look for (case-insensitive): ``aggregate_score``, ``ptm``, ``iptm``,
    ``plddt`` (mean), ``pae`` (mean / aggregate).
    """
    out: dict[str, Optional[float]] = {
        "plddt": None, "iptm": None, "ptm": None,
        "pae": None, "aggregate_score": None,
    }
    if not path or not Path(path).is_file():
        return out
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    if not isinstance(data, dict):
        return out

    # Case-insensitive lookup so we accept ``pLDDT`` / ``plddt`` etc.
    lower = {k.lower(): v for k, v in data.items()}
    out["plddt"] = _safe_float(lower.get("plddt") or lower.get("complex_plddt"))
    out["iptm"] = _safe_float(lower.get("iptm") or lower.get("complex_iptm"))
    out["ptm"] = _safe_float(lower.get("ptm") or lower.get("complex_ptm"))
    out["pae"] = _safe_float(
        lower.get("pae") or lower.get("complex_pae") or lower.get("mean_pae")
    )
    out["aggregate_score"] = _safe_float(
        lower.get("aggregate_score") or lower.get("ranking_score")
    )
    return out


# ---------- adapter --------------------------------------------------------


class Chai1Adapter(BaseAdapter):
    """Adapter for Chai-1 (Cat A, single-sequence mode)."""

    tool_id = "chai1"
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

        # Same cleaning rules as Boltz-2: drop non-standard residues
        # (Chai-1's tokenizer also rejects them) and remember the
        # clean→original mapping so parse_output can translate residue
        # IDs back to the dataset's numbering.
        protein_clean, protein_map = clean_protein_for_boltz(protein_seq)
        protein_dropped = len(protein_seq) - len(protein_clean)
        if protein_dropped > 0:
            logger.info(
                "chai1[%s] protein cleaned: %d → %d aa (dropped %d "
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
                "chai1[%s] rna cleaned: %d → %d nt (dropped %d "
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
        fasta_path = work_dir / f"{job_name}.fasta"
        write_chai1_fasta(
            fasta_path, protein_clean, rna_clean,
            protein_chain=self.PROTEIN_CHAIN, rna_chain=self.RNA_CHAIN,
        )
        write_seq_map(work_dir, protein_map, rna_map)

        return {
            "fasta": fasta_path,
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
        tool_cfg = (config.get("tools") or {}).get("chai1") or {}
        env_name = tool_cfg.get("conda_env", "chai1")
        timeout = int(tool_cfg.get("timeout", 600))
        device = tool_cfg.get("device", "cuda:0")
        num_trunk_recycles = int(tool_cfg.get("num_trunk_recycles", 3))
        num_diffn_timesteps = int(tool_cfg.get("num_diffn_timesteps", 200))
        seed = int(tool_cfg.get("seed", 42))
        use_esm_embeddings = bool(tool_cfg.get("use_esm_embeddings", True))
        use_msa_server = bool(tool_cfg.get("use_msa_server", False))

        fasta_path = Path(input_paths["fasta"]).resolve()
        if not fasta_path.is_file():
            raise FileNotFoundError(f"prepared FASTA missing: {fasta_path}")

        output_dir = (work_dir / "chai1_output").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        runner = write_runner_script(
            work_dir,
            sample_id=input_paths.get("job_name", "sample"),
            fasta_path=fasta_path,
            output_dir=output_dir,
            num_trunk_recycles=num_trunk_recycles,
            num_diffn_timesteps=num_diffn_timesteps,
            seed=seed,
            device=device,
            use_esm_embeddings=use_esm_embeddings,
            use_msa_server=use_msa_server,
        )

        log_dir_cfg = config.get("log_dir")
        log_dir = Path(log_dir_cfg) if log_dir_cfg else None
        log_tag = f"chai1_{fasta_path.stem}"

        cmd = f"python {shlex.quote(str(runner))}"
        result = run_in_conda_env(
            env_name, cmd, timeout=timeout,
            log_dir=log_dir, log_tag=log_tag,
        )
        if not result.success:
            raise RuntimeError(
                f"Chai-1 failed (rc={result.returncode}, "
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

        # Chai-1 emits ``pred.model_idx_0.cif`` for the top-ranked model.
        # rglob is forgiving against minor layout shifts between versions
        # (some forks nest under ``predictions/``).
        cif_path = (
            _find_first(output_dir, "pred.model_idx_0.cif")
            or _find_first(output_dir, "pred.model_idx_*.cif")
            or _find_first(output_dir, "*.cif")
        )
        if cif_path is None:
            return self.fail(
                sample_id,
                f"no Chai-1 prediction CIF found under {output_dir}",
                raw_output_dir=str(output_dir),
            )

        # Heavy-atom contacts from the predicted complex.
        contacts = extract_contacts(
            cif_path,
            cutoff=cutoff,
            protein_chain_id=self.PROTEIN_CHAIN,
            rna_chain_id=self.RNA_CHAIN,
        )

        # Per-residue pLDDT (chain A, CA B-factor). Chai-1 stores
        # pLDDT in the same field Boltz-2 does, so the helper
        # imported from boltz2_adapter is reusable verbatim.
        per_res_plddt = read_protein_plddt_from_cif(
            cif_path, chain_id=self.PROTEIN_CHAIN,
        )

        # Translate Chai-1's 1..N (clean-sequence) numbering back to the
        # dataset's original numbering. Sidecar lives next to the FASTA
        # (work_dir/), which is the parent of output_dir. Empty maps =
        # nothing to remap (defensive: keeps backwards-compat for
        # existing work_dirs that pre-date the sidecar).
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

        # Confidence summary from scores JSON if Chai-1 wrote one;
        # otherwise fall back to per-residue mean / npy file.
        scores_path = (
            _find_first(output_dir, "scores.model_idx_0.json")
            or _find_first(output_dir, "scores.model_idx_*.json")
        )
        scores = parse_chai1_scores_json(scores_path) if scores_path else {}

        plddt_mean = scores.get("plddt") if isinstance(scores, dict) else None
        if plddt_mean is None and per_res_plddt:
            plddt_mean = round(
                sum(per_res_plddt.values()) / len(per_res_plddt), 3,
            )

        iptm_score = scores.get("iptm") if isinstance(scores, dict) else None

        # PAE: prefer scores.json's mean; else fall back to the npy file
        # if present. Note the per_residue_pae_score field is NOT
        # populated from PAE for Chai-1 — see the distance branch below.
        pae_npy = (
            _find_first(output_dir, "pae.model_idx_0.npy")
            or _find_first(output_dir, "pae.model_idx_*.npy")
        )
        pae_mean = scores.get("pae") if isinstance(scores, dict) else None
        if pae_mean is None and pae_npy is not None:
            pae_mean = read_pae_mean_from_npy(pae_npy)

        # Per-residue binding score — distance-derived for Chai-1.
        # Chai-1's scores.npz only carries summary confidences (no
        # per-token PAE matrix), so the PAE path that boltz2_adapter
        # uses isn't available here. We approximate the same signal
        # from the predicted complex geometry: CA → nearest RNA heavy
        # atom distance, mapped through the same 1/(1 + d/scale)
        # decay so the resulting [0,1] scores are directly comparable
        # to Boltz-2's PAE-derived scores in evaluate.py. The field is
        # still ``per_residue_pae_score`` so downstream code (eval,
        # paper tables) doesn't need to branch by tool.
        per_res_binding: dict[int, float] = {}
        tool_cfg = (config.get("tools") or {}).get("chai1") or {}
        if (bool(tool_cfg.get("compute_distance_scores", True))
                and protein_map):
            distance_scale = float(tool_cfg.get("distance_scale", 8.0))
            per_res_clean = compute_distance_binding_scores(
                cif_path,
                protein_chain_id=self.PROTEIN_CHAIN,
                rna_chain_id=self.RNA_CHAIN,
                distance_scale=distance_scale,
            )
            per_res_binding = remap_per_residue(per_res_clean, protein_map)

        # Clamp pLDDT into the schema range for safety against float
        # drift (Boltz-2 / Chai-1 occasionally emit 100.0001 etc.).
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
            predicted_structure_path=str(Path(cif_path).resolve()),
            plddt_mean=plddt_mean,
            iptm_score=iptm_score,
            pae_mean=pae_mean,
            raw_output_dir=str(output_dir),
        )
