"""HADDOCK 3 adapter (Category D — protein-RNA molecular docking).

Why this is the only Cat D tool active
--------------------------------------
HADDOCK 3 is data-driven docking: feed it a protein PDB + an RNA PDB
and it returns a set of ranked, refined docked complexes. The original
tool list had two Cat D tools (HADDOCK3 + HDOCK); HDOCK is currently
inactive. HADDOCK 3 contributes a signal the other categories don't:
the agent gets to see WHERE a tool placed the RNA on the protein
surface, not just per-residue probabilities derived from sequence
features (Cat C) or surface geometry (Cat B) or single-step structure
prediction (Cat A).

Workflow
--------
1. ``prepare_input``:
     - extract the protein chain from the raw PDB / CIF →
       ``protein.pdb`` (chain ``A``, renumbered to 1-based polymer
       index — reuses :func:`p2rank_adapter.extract_protein_chain_pdb`
       so the residue numbering matches the rest of the pipeline)
     - extract the RNA chain → ``rna.pdb`` (chain ``B``, same
       renumbering — :func:`structure_utils.extract_rna_chain_pdb`)
     - write a minimal HADDOCK 3 TOML at
       ``<work_dir>/haddock_config.toml`` listing the two molecules
       and a rigidbody → flexref → emref pipeline. Sampling counts
       come from config; defaults are conservative
       (50 / 5 / 5) to keep runtime under the 600 s per-sample budget.

2. ``run_tool``: ``haddock3 <config.toml>`` inside the ``haddock3``
   conda env. CNS is provided via the ``CNS_SOLVE`` env var (HADDOCK
   needs it for the energy / refinement modules). Returns the
   ``run_dir`` path so parse_output knows where to look.

3. ``parse_output``: HADDOCK 3 writes per-stage subdirectories
   (``0_topoaa/`` / ``1_rigidbody/`` / ``2_flexref/`` / ``3_emref/``,
   numbering depends on the module list). The best refined pose
   lives in the LAST numbered subdir — we find that defensively
   with rglob (the exact filename convention has shifted across
   HADDOCK 3 minor versions: ``rigidbody_1.pdb``,
   ``cluster_1_model_1.pdb``, ``emref_1.pdb``, etc.). From the best
   pose we extract:

     - protein-RNA heavy-atom contacts via
       :func:`contact_extractor.extract_contacts` at 4.5 Å — same
       semantics as Cat A's contact extraction so binding_protein_residues
       are directly comparable across tools.
     - per-residue binding probability via
       :func:`structure_utils.compute_distance_binding_scores` — same
       mechanism Boltz-2 / Chai-1 now use, so HADDOCK 3 contributes
       a comparable per_residue_pae_score for the learned-fusion
       feature row.

   The exact output layout needs server confirmation (HADDOCK 3
   2026.5.0 docs claim ``<step>_<module>/`` but practice may differ);
   the rglob-based search keeps the adapter alive across minor
   version shifts and is logged on miss for easy debugging.
"""
from __future__ import annotations

import csv
import gzip
import re
import shlex
import shutil
from pathlib import Path
from typing import Optional

from ..base_adapter import BaseAdapter
from ..contact_extractor import extract_contacts
from ..schemas import ToolPrediction
from ..tool_runner import run_in_conda_env
from .p2rank_adapter import (
    _find_raw_structure, effective_pdb_chain_id, extract_protein_chain_pdb,
)
from .structure_utils import (
    compute_distance_binding_scores, effective_rna_chain_id,
    extract_rna_chain_pdb,
)


# ---- TOML writer ----------------------------------------------------------


def write_haddock_config(
    toml_path: Path,
    *,
    run_dir: Path,
    protein_pdb: Path,
    rna_pdb: Path,
    sampling: int = 50,
    flexref_sampling_factor: int = 1,
    emref_sampling_factor: int = 1,
    cmrest: bool = True,
) -> Path:
    """Hand-write a minimal HADDOCK 3 TOML.

    Why not tomllib / a templating library: HADDOCK 3 is picky about
    the exact TOML layout (module ordering matters, header-only sections
    like ``[topoaa]`` must appear with no body), and the schema is
    small enough that a triple-quoted f-string is the lowest-friction
    representation. The hard-coded module pipeline matches the
    "rigidbody → flexref → emref" path the user listed in the spec.

    Per-module parameter names follow HADDOCK 3's own schema:

      * ``[rigidbody] sampling = N`` — the count of poses to generate.
      * ``[rigidbody] cmrest = true`` — turn on centre-of-mass
        restraints. We never have experimental restraints in this
        pipeline (the agent's whole job is to PREDICT the interface),
        so rigidbody runs ab-initio and needs the CM restraint to keep
        the two molecules near each other during the energy-minimised
        sampling. Without it, the sampler can drift the RNA arbitrarily
        far from the protein and produce useless poses.
      * ``[flexref] sampling_factor = K`` — keeps ``K * <input_count>``
        poses after flexible refinement. NOT ``sampling``: passing
        ``sampling`` to flexref / emref makes haddock3 abort with an
        ``unknown parameter`` error (verified on 2026.5.0).
      * ``[emref] sampling_factor = K`` — same multiplicative semantic.

    Paths are written as absolute strings — HADDOCK 3 resolves them
    relative to wherever the run_dir is created, and any ambiguity
    there cost me an hour the first time I touched HADDOCK 2.
    """
    if sampling <= 0:
        raise ValueError(f"sampling must be > 0, got {sampling}")
    if flexref_sampling_factor < 0 or emref_sampling_factor < 0:
        raise ValueError(
            f"flexref_sampling_factor / emref_sampling_factor must be ≥ 0 "
            f"(got {flexref_sampling_factor}, {emref_sampling_factor})"
        )

    cmrest_literal = "true" if cmrest else "false"
    toml = (
        f'run_dir = "{Path(run_dir).resolve()}"\n'
        f'molecules = ["{Path(protein_pdb).resolve()}", '
        f'"{Path(rna_pdb).resolve()}"]\n'
        f"\n"
        f"[topoaa]\n"
        f"\n"
        f"[rigidbody]\n"
        f"sampling = {int(sampling)}\n"
        f"cmrest = {cmrest_literal}\n"
        f"\n"
        f"[flexref]\n"
        f"sampling_factor = {int(flexref_sampling_factor)}\n"
        f"\n"
        f"[emref]\n"
        f"sampling_factor = {int(emref_sampling_factor)}\n"
    )
    toml_path = Path(toml_path)
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text(toml, encoding="utf-8")
    return toml_path


# ---- output discovery -----------------------------------------------------


# HADDOCK 3 stages are written as ``<index>_<module>/`` under the
# run_dir. We rank by the leading integer so the LAST stage (highest
# index) wins regardless of which modules were configured.
_STAGE_DIR_RE = re.compile(r"^(\d+)_(\w+)$")


def find_final_stage_dir(run_dir: Path) -> Optional[Path]:
    """Return the highest-numbered ``<N>_<module>/`` under ``run_dir``.

    Falls back to ``None`` when the run_dir doesn't exist or contains
    no recognisable stage subdirs (HADDOCK 3 crash mid-pipeline, or
    the layout changed enough that the regex no longer matches).
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None
    best_idx = -1
    best_dir: Optional[Path] = None
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        m = _STAGE_DIR_RE.match(child.name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx > best_idx:
            best_idx = idx
            best_dir = child
    return best_dir


# Filename patterns we'll accept for "the best pose" in the legacy
# fallback path (no traceback.tsv on disk). Both ``.pdb`` and ``.pdb.gz``
# variants are listed — HADDOCK 3 2026.5 gzip-compresses every emitted
# pose, so the uncompressed pattern only matches manually-staged files.
_BEST_POSE_PATTERNS: tuple[str, ...] = (
    "*_1.pdb",      "*_1.pdb.gz",
    "cluster_1_model_1.pdb", "cluster_1_model_1.pdb.gz",
    "ranked_0.pdb", "ranked_0.pdb.gz",
    "*.pdb",        "*.pdb.gz",
)


def _resolve_pose_file(stage_dir: Path, pose_name: str) -> Optional[Path]:
    """Locate ``pose_name`` under ``stage_dir``; decompress ``.pdb.gz``
    on the fly if only the gzipped variant is on disk.

    HADDOCK 3 2026.5 writes every pose as ``<name>.pdb.gz`` and the
    traceback table references the un-suffixed name (``emref_33.pdb``).
    We resolve both: bare ``.pdb`` wins if present; otherwise we
    materialise it from ``.pdb.gz`` next to the source. gemmi /
    contact_extractor need an actual ``.pdb`` on disk, so the
    decompressed file is left in place (cheap re-read for downstream
    correlation passes).

    Returns ``None`` when neither file is present.
    """
    direct = stage_dir / pose_name
    if direct.is_file():
        return direct
    gz = stage_dir / f"{pose_name}.gz"
    if gz.is_file():
        try:
            with gzip.open(gz, "rb") as src, open(direct, "wb") as dst:
                shutil.copyfileobj(src, dst)
        except OSError:
            return None
        return direct
    return None


def _find_best_pose_via_traceback(
    run_dir: Path, stage_dir: Path,
) -> Optional[Path]:
    """Read ``<run_dir>/traceback/traceback.tsv`` to find the
    ``rank == 1`` pose for the final stage.

    Layout (HADDOCK 3 2026.5)::

        00_topo1  00_topo2  1_rigidbody  1_rigidbody_rank  ...  3_emref  3_emref_rank
        ...                 rigidbody_33.pdb  27           ...  emref_33.pdb  1

    The right-most ``*_rank`` column is the final ranking — column
    immediately to its left is the pose filename for that stage.

    Returns ``None`` on:
      - traceback.tsv missing
      - header has no ``*_rank`` columns
      - no row has ``rank == 1``
      - referenced pose file isn't on disk (even as ``.pdb.gz``)
    """
    tsv = Path(run_dir) / "traceback" / "traceback.tsv"
    if not tsv.is_file():
        return None
    try:
        with tsv.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.reader(fh, delimiter="\t"))
    except OSError:
        return None
    if len(rows) < 2:
        return None
    header = rows[0]

    rank_col = -1
    for i, h in enumerate(header):
        if (h or "").strip().endswith("_rank"):
            rank_col = i  # keep walking — we want the LAST _rank column
    if rank_col < 1:
        # Need a column immediately to the left for the pose filename.
        return None
    pose_col = rank_col - 1

    for row in rows[1:]:
        if len(row) <= rank_col:
            continue
        try:
            rank_val = int((row[rank_col] or "").strip())
        except (TypeError, ValueError):
            continue
        if rank_val != 1:
            continue
        pose_name = (row[pose_col] or "").strip()
        if not pose_name:
            continue
        return _resolve_pose_file(stage_dir, pose_name)
    return None


def _find_best_pose_via_patterns(stage_dir: Path) -> Optional[Path]:
    """Legacy fallback: pattern-match a ranked filename under
    ``stage_dir``. Matches sorted alphabetically so the choice is
    deterministic across filesystems; ``.pdb.gz`` matches are
    decompressed before return."""
    for pattern in _BEST_POSE_PATTERNS:
        matches = sorted(stage_dir.rglob(pattern))
        if not matches:
            continue
        best = matches[0]
        if best.suffix == ".gz":
            target = best.with_suffix("")  # strips final .gz → keeps .pdb
            if not target.is_file():
                try:
                    with gzip.open(best, "rb") as src, \
                            open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                except OSError:
                    return None
            return target
        return best
    return None


def find_best_pose(run_dir_or_stage: Path) -> Optional[Path]:
    """Pick the top-ranked HADDOCK 3 docked pose.

    Resolution order:

      1. ``<run_dir>/traceback/traceback.tsv`` — authoritative ranking
         that HADDOCK 3 itself writes. The pose with ``rank == 1`` in
         the last ``*_rank`` column wins.
      2. Filename pattern match within the final stage dir — legacy
         fallback for old runs that don't have ``traceback/`` (or for
         the unit tests that pre-date this code path).

    Parameters
    ----------
    run_dir_or_stage :
        Either a run_dir (parent of the ``<N>_<module>/`` stage dirs
        and the ``traceback/`` dir) or a stage_dir directly. The latter
        is the historical signature — preserved so existing callers /
        tests don't break. When a stage_dir is passed, the traceback
        step is skipped (no place to look) and we go straight to the
        pattern fallback.

    Both ``.pdb`` and ``.pdb.gz`` variants are resolved; the gzipped
    form is decompressed to a sibling ``.pdb`` on first read so the
    downstream gemmi / contact_extractor see an actual PDB file.
    """
    p = Path(run_dir_or_stage)
    if not p.is_dir():
        return None

    # Auto-detect: a leaf ``<N>_<module>`` dir name signals stage_dir;
    # anything else is treated as a run_dir whose stage we discover.
    if _STAGE_DIR_RE.match(p.name):
        # Caller passed a stage_dir directly — no traceback path
        # available. Go straight to pattern fallback.
        return _find_best_pose_via_patterns(p)

    run_dir = p
    stage_dir = find_final_stage_dir(run_dir)
    if stage_dir is None:
        return None

    via_traceback = _find_best_pose_via_traceback(run_dir, stage_dir)
    if via_traceback is not None:
        return via_traceback
    return _find_best_pose_via_patterns(stage_dir)


# ---- adapter --------------------------------------------------------------


class Haddock3Adapter(BaseAdapter):
    """Adapter for HADDOCK 3 (Cat D, protein-RNA docking)."""

    tool_id = "haddock3"
    category = "D"

    # The chain names HADDOCK 3 sees in the protein / RNA PDBs we
    # produce; in the docked output they're preserved so parse_output
    # can use them as chain filters directly.
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
        source_pdb = sample_json["source_pdb"]
        protein_chain = (sample_json.get("protein") or {}).get("chain_id")
        rna_chain = (sample_json.get("rna") or {}).get("chain_id")
        if not protein_chain:
            raise ValueError(
                f"sample {sample_id!r}: missing protein.chain_id"
            )
        if not rna_chain:
            raise ValueError(
                f"sample {sample_id!r}: missing rna.chain_id"
            )

        ss_cfg = config.get("structure_source") or {}
        raw_dir = Path(ss_cfg.get("raw_dir") or "data/raw")
        raw_path = _find_raw_structure(raw_dir, source_pdb)
        if raw_path is None:
            raise FileNotFoundError(
                f"no raw structure for {source_pdb!r} under {raw_dir} "
                f"(tried .pdb, .cif)"
            )

        work_dir = Path(work_dir)
        protein_pdb = work_dir / "protein.pdb"
        rna_pdb = work_dir / "rna.pdb"
        extract_protein_chain_pdb(raw_path, protein_chain, protein_pdb)
        extract_rna_chain_pdb(raw_path, rna_chain, rna_pdb)

        tool_cfg = (config.get("tools") or {}).get("haddock3") or {}
        run_dir = (work_dir / "haddock_run").resolve()
        toml_path = work_dir / "haddock_config.toml"
        write_haddock_config(
            toml_path,
            run_dir=run_dir,
            protein_pdb=protein_pdb,
            rna_pdb=rna_pdb,
            sampling=int(tool_cfg.get("sampling", 50)),
            flexref_sampling_factor=int(
                tool_cfg.get("flexref_sampling_factor", 1)
            ),
            emref_sampling_factor=int(
                tool_cfg.get("emref_sampling_factor", 1)
            ),
            cmrest=bool(tool_cfg.get("cmrest", True)),
        )
        return {
            "config_path": toml_path,
            "run_dir": run_dir,
            "protein_pdb": protein_pdb,
            "rna_pdb": rna_pdb,
        }

    # ------------------------------------------------------------------ run

    def run_tool(
        self,
        input_paths: dict,
        work_dir: Path,
        config: dict,
    ) -> Path:
        tool_cfg = (config.get("tools") or {}).get("haddock3") or {}
        env_name = tool_cfg.get("conda_env", "haddock3")
        timeout = int(tool_cfg.get("timeout", 600))
        cns_solve = tool_cfg.get("cns_solve")

        toml_path = Path(input_paths["config_path"]).resolve()
        if not toml_path.is_file():
            raise FileNotFoundError(
                f"prepared HADDOCK 3 TOML missing: {toml_path}"
            )

        run_dir = Path(input_paths.get("run_dir") or
                       (work_dir / "haddock_run")).resolve()

        # CNS is the underlying refinement engine HADDOCK calls into.
        # The server install puts it at /opt/biotools/cns_v1.3_r9
        # and HADDOCK looks for it via the CNS_SOLVE env var. Allow the
        # config to override the path so non-default installs work
        # without code changes; an absent path just gets skipped (the
        # haddock3 conda env may have CNS on PATH already).
        extra_env: dict[str, str] = {}
        if cns_solve:
            extra_env["CNS_SOLVE"] = str(cns_solve)

        log_dir_cfg = config.get("log_dir")
        log_dir = Path(log_dir_cfg) if log_dir_cfg else None
        log_tag = f"haddock3_{toml_path.stem}"

        cmd = f"haddock3 {shlex.quote(str(toml_path))}"
        result = run_in_conda_env(
            env_name, cmd,
            timeout=timeout,
            log_dir=log_dir, log_tag=log_tag,
            extra_env=extra_env or None,
        )
        if not result.success:
            raise RuntimeError(
                f"HADDOCK 3 failed (rc={result.returncode}, "
                f"timed_out={result.timed_out}). "
                f"See log: {result.log_path}. "
                f"stderr: {result.stderr[:500]}"
            )
        if not run_dir.is_dir():
            raise RuntimeError(
                f"HADDOCK 3 returned rc=0 but expected run_dir "
                f"{run_dir} is missing. The TOML may have specified a "
                f"different run_dir; check {toml_path}."
            )
        return run_dir

    # ------------------------------------------------------------------ parse

    def parse_output(
        self,
        output_dir: Path,
        sample_json: dict,
        config: dict,
    ) -> ToolPrediction:
        sample_id = sample_json["sample_id"]
        tool_cfg = (config.get("tools") or {}).get("haddock3") or {}
        cutoff = float(config.get("contact_threshold", 4.5))
        distance_scale = float(tool_cfg.get("distance_scale", 8.0))

        run_dir = Path(output_dir)
        stage_dir = find_final_stage_dir(run_dir)
        if stage_dir is None:
            return self.fail(
                sample_id,
                f"HADDOCK 3 run_dir {run_dir} has no <N>_<module>/ "
                f"subdirs — pipeline may have crashed before any stage "
                f"completed. Check the run log.",
                raw_output_dir=str(run_dir),
            )
        # Pass run_dir (not stage_dir) so find_best_pose can read the
        # traceback table for authoritative ranking; the pattern
        # fallback still kicks in when traceback/ is missing.
        best_pose = find_best_pose(run_dir)
        if best_pose is None:
            return self.fail(
                sample_id,
                f"no PDB found under HADDOCK 3 run dir {run_dir} "
                f"(stage: {stage_dir.name}; traceback table missing "
                f"or empty AND patterns {_BEST_POSE_PATTERNS!r} returned "
                f"no matches)",
                raw_output_dir=str(run_dir),
            )

        # The chains in the docked PDB match what we wrote in
        # prepare_input — single-char "A" / "B" even when the source
        # chain id was multi-char (extract_*_chain_pdb rewrote it).
        protein_chain = effective_pdb_chain_id(self.PROTEIN_CHAIN)
        rna_chain = effective_rna_chain_id(self.RNA_CHAIN)

        contacts = extract_contacts(
            best_pose,
            cutoff=cutoff,
            protein_chain_id=protein_chain,
            rna_chain_id=rna_chain,
        )

        # Per-residue binding probability from CA → nearest-RNA-atom
        # distance. Same mechanism Boltz-2 / Chai-1 use; the
        # ``per_residue_pae_score`` field name is historical (see
        # schemas.py). This lets the learned-fusion path treat HADDOCK
        # 3's contribution exactly like a Cat A tool's.
        per_res_distance = compute_distance_binding_scores(
            best_pose,
            protein_chain_id=protein_chain,
            rna_chain_id=rna_chain,
            distance_scale=distance_scale,
        )

        # No remap step — HADDOCK 3 preserves the residue numbering
        # from the input PDBs, and ``extract_*_chain_pdb`` already
        # renumbered them to 1-based polymer indices that match the
        # sample_json convention. So contacts.binding_protein_residues
        # is directly comparable to sample.interaction.binding_protein_residues
        # without translation.

        return ToolPrediction(
            tool_id=self.tool_id,
            category=self.category,
            sample_id=sample_id,
            success=True,
            binding_protein_residues=contacts.binding_protein_residues or None,
            binding_rna_nucleotides=contacts.binding_rna_nucleotides or None,
            # Cat D output isn't a per-residue score in the same sense
            # as Cat C — the value here is the distance-derived
            # probability, populated only on per_residue_pae_score so
            # the learned-fusion path picks it up the same way it does
            # for Boltz-2 / Chai-1. per_residue_confidence stays None
            # so noisy-OR doesn't double-count distance + pLDDT for a
            # tool that has no pLDDT to report.
            per_residue_pae_score=per_res_distance or None,
            predicted_structure_path=str(Path(best_pose).resolve()),
            raw_output_dir=str(run_dir),
        )
