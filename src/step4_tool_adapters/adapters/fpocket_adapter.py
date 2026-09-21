"""Fpocket adapter (Category B — geometric pocket detection).

Why add Fpocket alongside P2Rank
--------------------------------
P2Rank is ML-based (gradient boosting on geometric / chemical features
per surface point); Fpocket is purely geometric (Voronoi tessellation +
alpha-sphere clustering). The two pick up different pocket shapes —
P2Rank tends to over-call on flat surfaces, Fpocket misses very flat
binding sites — so feeding both into step-5 fusion gives a more robust
Cat B signal than either alone.

Workflow
--------
1. ``prepare_input``: same as P2Rank — locate the raw PDB/CIF for
   ``sample.source_pdb`` under ``config['structure_source']['raw_dir']``,
   extract the ``sample.protein.chain_id`` chain to a clean,
   1-based-renumbered PDB. Reuses :func:`extract_protein_chain_pdb`
   from ``p2rank_adapter`` so the residue numbering matches step 1's
   ground truth identically.

2. ``run_tool``: ``fpocket -f <protein.pdb>``. Fpocket writes its
   results into ``<protein>_out/`` next to the input PDB by default;
   we don't need a separate ``-o`` flag. We DO copy the output dir
   into ``<work_dir>/fpocket_output/`` afterwards so everything the
   adapter touches stays inside the per-sample staging dir (makes
   cleanup and post-mortem trivial).

   Fpocket isn't a Java / Python package — it's a native binary
   typically installed via ``conda install -c conda-forge fpocket``.
   No conda env switch is needed if the host env has it; the config's
   ``conda_env`` flag is optional for sites that put it in a separate
   env.

3. ``parse_output``: read ``<protein>_out/<protein>_info.txt``
   (per-pocket summary) plus per-pocket ``pockets/pocketN_atm.pdb``
   files (the atoms / residues belonging to each pocket). For each
   pocket we keep:

   - ``rank``: 1-based, ordered as Fpocket emits (best-scored first).
   - ``score``: ``Druggability Score`` from info.txt (0-1 calibrated
     by the upstream model — falls back to ``Score`` if absent).
   - ``residues``: 1-based polymer indices belonging to the pocket.

   Per-residue confidence comes from the highest-scoring pocket each
   residue appears in (Fpocket doesn't emit a true per-residue
   probability; we use the pocket's druggability score as a proxy).

   Residues whose proxy score exceeds ``residue_score_threshold``
   land in ``binding_protein_residues`` — same convention as P2Rank.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Optional

from ..base_adapter import BaseAdapter
from ..schemas import Pocket, ToolPrediction
from ..tool_runner import run_command, run_in_conda_env
from .p2rank_adapter import (
    _find_raw_structure, effective_pdb_chain_id, extract_protein_chain_pdb,
)


# ---------- info.txt parsing ----------------------------------------------


# Lines in info.txt look like::
#
#   Pocket 1 :
#       Score :                 0.541
#       Druggability Score :    0.892
#       Number of Alpha Spheres :   45
#       ...
#
# The header line tells us when a new pocket starts; subsequent
# ``Key : Value`` lines populate the pocket. Pocket scores can be
# negative, so the ``-?`` in the value pattern is intentional.
_POCKET_HEADER_RE = re.compile(r"^\s*Pocket\s+(\d+)\s*:\s*$")
_KV_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _\-/]+?)\s*:\s*(\S.*)$")


def parse_info_txt(path: Path) -> list[dict]:
    """Parse Fpocket's ``*_info.txt`` into a list of pocket dicts.

    Returns one dict per pocket with all numeric fields coerced to
    ``float`` (or kept as raw strings when not parseable). The list
    is in the order Fpocket emitted them — pocket 1 is the highest-
    scored.
    """
    if not path or not Path(path).is_file():
        return []
    pockets: list[dict] = []
    current: Optional[dict] = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        m_hdr = _POCKET_HEADER_RE.match(raw)
        if m_hdr:
            if current is not None:
                pockets.append(current)
            current = {"rank": int(m_hdr.group(1))}
            continue
        if current is None:
            continue
        m_kv = _KV_RE.match(raw)
        if not m_kv:
            continue
        key = m_kv.group(1).strip().lower().replace(" ", "_").replace("-", "_")
        val_raw = m_kv.group(2).strip()
        # Numeric coercion when possible; preserve the raw string
        # otherwise (Fpocket has a few non-numeric stat fields).
        try:
            current[key] = float(val_raw)
        except ValueError:
            current[key] = val_raw
    if current is not None:
        pockets.append(current)
    return pockets


# ---------- pocket atm PDB parsing ----------------------------------------


# ATOM record residue sequence number lives in columns 23-26 (1-based);
# chain ID is in column 22.
_ATOM_PREFIX = "ATOM"


def parse_pocket_atm_pdb(path: Path, chain_filter: Optional[str] = None) -> list[int]:
    """Return the sorted unique 1-based residue indices in a pocket file.

    Fpocket writes one ``pocketN_atm.pdb`` per pocket containing only
    the atoms involved in that pocket. Indices come from the residue
    sequence number column, so they line up with our renumbered PDB
    (``label_seq``-renumbered by ``extract_protein_chain_pdb``).
    """
    if not path or not Path(path).is_file():
        return []
    seen: set[int] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith(_ATOM_PREFIX):
            continue
        if len(line) < 26:
            continue
        chain = line[21]
        if chain_filter is not None and chain != chain_filter:
            continue
        try:
            idx = int(line[22:26].strip())
        except ValueError:
            continue
        seen.add(idx)
    return sorted(seen)


# ---------- adapter --------------------------------------------------------


class FpocketAdapter(BaseAdapter):
    """Adapter for Fpocket (Cat B, geometric pocket detection)."""

    tool_id = "fpocket"
    category = "B"

    # ------------------------------------------------------------------ prepare

    def prepare_input(
        self,
        sample_json: dict,
        work_dir: Path,
        config: dict,
    ) -> dict:
        sample_id = sample_json["sample_id"]
        source_pdb = sample_json["source_pdb"]
        chain_id = sample_json["protein"]["chain_id"]

        ss_cfg = config.get("structure_source") or {}
        raw_dir_str = ss_cfg.get("raw_dir") or "data/raw"
        raw_dir = Path(raw_dir_str)
        raw_path = _find_raw_structure(raw_dir, source_pdb)
        if raw_path is None:
            raise FileNotFoundError(
                f"no raw structure for {source_pdb!r} under {raw_dir} "
                f"(tried .pdb, .cif)"
            )

        out_pdb = work_dir / f"{sample_id}_protein.pdb"
        extract_protein_chain_pdb(raw_path, chain_id, out_pdb)
        return {"protein_pdb": out_pdb, "raw_path": raw_path}

    # ------------------------------------------------------------------ run

    def run_tool(
        self,
        input_paths: dict,
        work_dir: Path,
        config: dict,
    ) -> Path:
        tool_cfg = (config.get("tools") or {}).get("fpocket") or {}
        timeout = int(tool_cfg.get("timeout", 60))
        # Optional conda env — fpocket is usually system-installed, but
        # some sites put it in a dedicated env. Empty/None → use the
        # ambient PATH directly via run_command.
        env_name = tool_cfg.get("conda_env") or ""

        pdb_path = Path(input_paths["protein_pdb"]).resolve()
        if not pdb_path.is_file():
            raise FileNotFoundError(f"prepared PDB missing: {pdb_path}")

        log_dir_cfg = config.get("log_dir")
        log_dir = Path(log_dir_cfg) if log_dir_cfg else None
        log_tag = f"fpocket_{pdb_path.stem}"

        # Fpocket writes into ``<pdb_stem>_out/`` next to the input PDB,
        # not into a CWD-relative dir, so cwd here doesn't matter much.
        # We still pin cwd to work_dir to keep stray files contained if
        # an upstream change adds CWD-relative behaviour.
        cmd = f"fpocket -f {pdb_path}"
        if env_name:
            result = run_in_conda_env(
                env_name, cmd,
                cwd=work_dir, timeout=timeout,
                log_dir=log_dir, log_tag=log_tag,
            )
        else:
            result = run_command(
                cmd,
                cwd=work_dir, timeout=timeout,
                log_dir=log_dir, log_tag=log_tag,
            )
        if not result.success:
            raise RuntimeError(
                f"Fpocket failed (rc={result.returncode}, "
                f"timed_out={result.timed_out}). "
                f"See log: {result.log_path}. "
                f"stderr: {result.stderr[:500]}"
            )

        # Fpocket's emit dir sits next to the input PDB. Move/copy it
        # into our staging dir so the rest of the pipeline (and
        # cleanup) sees a single ``fpocket_output/`` subdirectory under
        # work_dir, regardless of where the input PDB lived.
        emit_dir = pdb_path.parent / f"{pdb_path.stem}_out"
        if not emit_dir.is_dir():
            raise RuntimeError(
                f"Fpocket completed (rc=0) but expected output dir "
                f"{emit_dir} is missing — upstream layout may have changed."
            )
        out_dir = (work_dir / "fpocket_output").resolve()
        if out_dir.exists():
            shutil.rmtree(out_dir)
        shutil.move(str(emit_dir), str(out_dir))
        return out_dir

    # ------------------------------------------------------------------ parse

    def parse_output(
        self,
        output_dir: Path,
        sample_json: dict,
        config: dict,
    ) -> ToolPrediction:
        sample_id = sample_json["sample_id"]
        chain_id = sample_json["protein"]["chain_id"]
        # Multi-char chain IDs get rewritten to "A" by
        # extract_protein_chain_pdb (PDB format limit). Filter pocket
        # atoms on whichever name actually ended up in the staged PDB.
        effective_chain = effective_pdb_chain_id(chain_id)
        tool_cfg = (config.get("tools") or {}).get("fpocket") or {}
        score_thresh = float(tool_cfg.get("residue_score_threshold", 0.5))
        top_n = tool_cfg.get("top_pockets")
        top_n = int(top_n) if top_n is not None else None

        # info.txt name follows the input PDB stem; rglob to be robust
        # against minor layout changes (some forks nest results).
        info_matches = sorted(output_dir.rglob("*_info.txt"))
        if not info_matches:
            return self.fail(
                sample_id,
                f"missing fpocket *_info.txt under {output_dir}",
                raw_output_dir=str(output_dir),
            )
        info_path = info_matches[0]
        raw_pockets = parse_info_txt(info_path)
        if not raw_pockets:
            return self.fail(
                sample_id,
                f"fpocket info.txt {info_path} parsed to 0 pockets",
                raw_output_dir=str(output_dir),
            )

        # Per-pocket atom files live under ``pockets/pocketN_atm.pdb``.
        pockets_dir = info_path.parent / "pockets"

        pockets_out: list[Pocket] = []
        # Highest druggability score seen for each residue — used as the
        # per-residue confidence proxy (Fpocket doesn't emit per-residue
        # scores natively).
        per_res_score: dict[int, float] = {}

        for raw in raw_pockets:
            rank = int(raw.get("rank", 0))
            # Druggability is the calibrated 0-1 score; fall back to the
            # raw geometric score if druggability is missing in older
            # Fpocket versions.
            score = raw.get("druggability_score")
            if not isinstance(score, (int, float)):
                score = raw.get("score", 0.0)
            score = float(score) if isinstance(score, (int, float)) else 0.0

            atm_pdb = pockets_dir / f"pocket{rank}_atm.pdb"
            residues = parse_pocket_atm_pdb(
                atm_pdb, chain_filter=effective_chain,
            )

            pockets_out.append(Pocket(
                rank=rank, score=score, residues=residues,
            ))

            for r in residues:
                if score > per_res_score.get(r, -1.0):
                    per_res_score[r] = score

        pockets_out.sort(key=lambda p: p.rank)
        if top_n is not None:
            pockets_out = pockets_out[:top_n]

        binding = sorted(
            idx for idx, s in per_res_score.items() if s > score_thresh
        )

        return ToolPrediction(
            tool_id=self.tool_id,
            category=self.category,
            sample_id=sample_id,
            success=True,
            binding_protein_residues=binding,
            per_residue_confidence=per_res_score or None,
            pockets=pockets_out or None,
            raw_output_dir=str(output_dir),
        )
