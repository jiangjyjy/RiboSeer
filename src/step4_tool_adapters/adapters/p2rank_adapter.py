"""P2Rank adapter (Category B — protein-surface pocket detection).

Workflow
--------
1. ``prepare_input``: locate the raw PDB/CIF for ``sample.source_pdb``
   under ``config['structure_source']['raw_dir']``, extract the
   protein chain identified by ``sample.protein.chain_id`` to a clean
   single-chain PDB **renumbered using gemmi's ``label_seq``**. This
   matters: step 1's ``binding_protein_residues`` are 1-based polymer
   indices, but the original PDB stores author residue numbers (which
   in 1un6 chain B start at 104, for instance). If we feed P2Rank the
   raw PDB, its residue_label column won't line up with the sample
   JSON. Renumbering once in prepare_input keeps the whole pipeline on
   the same coordinate system.
2. ``run_tool``: ``<install_dir>/prank predict -f <abs_pdb> -o <out_dir>``.
   Absolute paths are mandatory — P2Rank fails on relative input paths.
3. ``parse_output``: read the two CSVs P2Rank emits
   (``*_predictions.csv`` and ``*_residues.csv``), build a
   ``ToolPrediction`` with pockets, per-residue ligandability scores,
   and the binding residue set above the configured threshold.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

from ..base_adapter import BaseAdapter
from ..schemas import Pocket, ToolPrediction
from ..tool_runner import run_command

try:
    import gemmi  # type: ignore
except ImportError as _e:  # pragma: no cover — required by environment.yml
    gemmi = None  # type: ignore
    _GEMMI_IMPORT_ERROR = _e
else:
    _GEMMI_IMPORT_ERROR = None


# Same minimal classifier as contact_extractor — kept inline so the
# adapter doesn't have a hidden coupling to that module.
_AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}


def _is_protein_residue(res_name: str) -> bool:
    name = res_name.strip().upper()
    if name in _AA3:
        return True
    if gemmi is not None:
        try:
            info = gemmi.find_tabulated_residue(name)
        except Exception:
            info = None
        if info is not None:
            kind = str(info.kind) if hasattr(info, "kind") else ""
            return "AA" in kind
    return False


# ---------- prepare_input helpers -----------------------------------------


def _find_raw_structure(raw_dir: Path, source_pdb: str) -> Optional[Path]:
    """Locate ``<source_pdb>.{pdb,cif}[.gz]`` under raw_dir.

    Tries each extension verbatim first, then falls back to a
    case-insensitive directory scan so callers with mixed-case PDB ids
    (``3j46`` on disk, ``3J46`` in the sample) still resolve. Large
    complexes that lack a PDB-format file (multi-char chain IDs are the
    canonical case) get matched via their ``.cif`` here.
    """
    extensions = (".pdb", ".cif", ".pdb.gz", ".cif.gz",
                  ".PDB", ".CIF", ".PDB.gz", ".CIF.gz")
    for ext in extensions:
        candidate = raw_dir / f"{source_pdb}{ext}"
        if candidate.is_file():
            return candidate

    if raw_dir.is_dir():
        target = source_pdb.lower()
        for f in raw_dir.iterdir():
            if not f.is_file():
                continue
            name_lower = f.name.lower()
            if not name_lower.startswith(target + "."):
                continue
            tail = name_lower[len(target) + 1:]
            if tail in ("pdb", "cif", "pdb.gz", "cif.gz"):
                return f
    return None


def effective_pdb_chain_id(chain_id: str) -> str:
    """Return the chain name actually used in the extracted PDB.

    Multi-character chain IDs (e.g. ``"10"``, ``"A5"``, ``"KK"`` from
    large ribosome / spliceosome complexes) don't fit the PDB format's
    one-column chainID field — gemmi's ``write_pdb`` truncates them and
    downstream tools then can't find the chain. We always rewrite the
    extracted single-chain PDB under the name ``"A"`` in that case so
    the on-disk PDB is well-formed. Single-character ids pass through
    unchanged for backwards compatibility with existing test fixtures.

    Adapters call this once in ``parse_output`` to know what chain name
    to filter the tool's output on (since the tool only saw the
    rewritten PDB).
    """
    return "A" if len(chain_id or "") > 1 else chain_id


def extract_protein_chain_pdb(
    raw_path: Path,
    chain_id: str,
    out_path: Path,
) -> Path:
    """Write a single-chain PDB renumbered to 1-based polymer index.

    Drops non-protein residues and waters. The output residue numbers
    match step 1's sample_json.protein.binding_residues, so downstream
    tools that read ``residue_label`` from the PDB will report indices
    in the same coordinate system.

    ``raw_path`` may be a ``.pdb`` or ``.cif`` (gzipped variants are
    fine — gemmi auto-detects from the extension). For multi-character
    chain IDs (only representable in mmCIF) the output PDB is written
    under chain name ``"A"`` because the PDB format has a one-column
    chainID field — see :func:`effective_pdb_chain_id`.

    Raises
    ------
    FileNotFoundError
        if ``raw_path`` doesn't exist.
    ValueError
        if the requested chain is not present, or the chain has no
        protein residues with a polymer index.
    """
    if gemmi is None:
        raise RuntimeError(f"gemmi not available: {_GEMMI_IMPORT_ERROR}")
    if not raw_path.is_file():
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

    # Build a fresh structure with one chain, renumbered residues. The
    # output chain name is single-character so ``write_pdb`` doesn't
    # truncate multi-char ids silently — see ``effective_pdb_chain_id``.
    new_structure = gemmi.Structure()
    new_structure.cell = structure.cell
    try:
        new_structure.spacegroup_hm = structure.spacegroup_hm
    except Exception:
        pass

    new_model = gemmi.Model("1")
    new_chain = gemmi.Chain(effective_pdb_chain_id(chain_id))

    kept = 0
    for res in src_chain:
        if not _is_protein_residue(res.name):
            continue
        if res.label_seq is None:
            continue
        new_res = res.clone()
        # Renumber author seqid to match label_seq, drop insertion code.
        new_res.seqid = gemmi.SeqId(int(res.label_seq), " ")
        new_chain.add_residue(new_res)
        kept += 1

    if kept == 0:
        raise ValueError(
            f"chain {chain_id!r} in {raw_path} has 0 protein residues "
            f"with a polymer index"
        )

    new_model.add_chain(new_chain)
    new_structure.add_model(new_model)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_structure.write_pdb(str(out_path))
    return out_path


# ---------- parse_output helpers ------------------------------------------


# residue_id token in P2Rank predictions.csv looks like "A_14" or
# "AB_125A" (chain underscore residue, optionally with insertion code).
_RES_ID_RE = re.compile(r"^(?P<chain>[A-Za-z0-9]+)_(?P<num>-?\d+)([A-Za-z])?$")


def _parse_residue_id_token(token: str) -> Optional[tuple[str, int]]:
    """Parse one ``A_14`` token into (chain, residue_index). Returns
    ``None`` for malformed tokens."""
    token = token.strip()
    if not token:
        return None
    m = _RES_ID_RE.match(token)
    if not m:
        return None
    try:
        return m.group("chain"), int(m.group("num"))
    except ValueError:
        return None


def _open_csv_dictreader(path: Path):
    """Open a P2Rank CSV; tolerate leading whitespace in headers/cells."""
    f = path.open("r", encoding="utf-8", newline="")
    reader = csv.DictReader(f, skipinitialspace=True)
    # Strip whitespace from fieldnames (P2Rank pads them with spaces).
    if reader.fieldnames:
        reader.fieldnames = [(h or "").strip() for h in reader.fieldnames]
    return f, reader


def parse_predictions_csv(
    path: Path,
    *,
    chain_filter: Optional[str] = None,
    top_n: Optional[int] = None,
) -> list[Pocket]:
    """Parse P2Rank ``*_predictions.csv`` into a list of ``Pocket``.

    Parameters
    ----------
    chain_filter : str, optional
        If provided, only residue tokens whose chain matches are kept.
    top_n : int, optional
        Cap the number of pockets returned (rank-ordered).
    """
    pockets: list[Pocket] = []
    f, reader = _open_csv_dictreader(path)
    try:
        for row in reader:
            if not row:
                continue
            try:
                rank = int(str(row.get("rank", "")).strip())
                score = float(str(row.get("score", "0")).strip())
            except (TypeError, ValueError):
                continue
            ids_raw = (row.get("residue_ids") or "").strip()
            residues: list[int] = []
            seen: set[int] = set()
            for tok in ids_raw.split():
                parsed = _parse_residue_id_token(tok)
                if parsed is None:
                    continue
                chain, idx = parsed
                if chain_filter is not None and chain != chain_filter:
                    continue
                if idx in seen:
                    continue
                seen.add(idx)
                residues.append(idx)
            pockets.append(Pocket(rank=rank, score=score, residues=sorted(residues)))
    finally:
        f.close()

    pockets.sort(key=lambda p: p.rank)
    if top_n is not None:
        pockets = pockets[:top_n]
    return pockets


def parse_residues_csv(
    path: Path,
    *,
    chain_filter: Optional[str] = None,
) -> dict[int, float]:
    """Parse P2Rank ``*_residues.csv`` into ``{residue_index: score}``.

    The score we keep is the ``score`` column (raw ligandability).
    P2Rank also emits ``probability`` (calibrated 0-1) and ``zscore``;
    callers that need those should re-parse this file themselves.
    """
    out: dict[int, float] = {}
    f, reader = _open_csv_dictreader(path)
    try:
        for row in reader:
            chain = (row.get("chain") or "").strip()
            if chain_filter is not None and chain != chain_filter:
                continue
            label_raw = (row.get("residue_label") or "").strip()
            # Strip insertion code (last char if alphabetic).
            if label_raw and label_raw[-1].isalpha():
                label_raw = label_raw[:-1]
            try:
                idx = int(label_raw)
            except ValueError:
                continue
            try:
                score = float((row.get("score") or "0").strip())
            except ValueError:
                continue
            out[idx] = score
    finally:
        f.close()
    return out


def _find_csv(output_dir: Path, suffix: str) -> Optional[Path]:
    """Find the first file ending with ``suffix`` under ``output_dir``."""
    matches = sorted(output_dir.rglob(f"*{suffix}"))
    return matches[0] if matches else None


# ---------- adapter --------------------------------------------------------


class P2RankAdapter(BaseAdapter):
    """Adapter for P2Rank (Cat B)."""

    tool_id = "p2rank"
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
        tool_cfg = (config.get("tools") or {}).get("p2rank") or {}
        install_dir = tool_cfg.get("install_dir")
        if not install_dir:
            raise ValueError("config['tools']['p2rank']['install_dir'] not set")
        timeout = int(tool_cfg.get("timeout", 60))

        pdb_path = Path(input_paths["protein_pdb"]).resolve()
        if not pdb_path.is_file():
            raise FileNotFoundError(f"prepared PDB missing: {pdb_path}")

        output_dir = (work_dir / "p2rank_output").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        log_dir_cfg = config.get("log_dir")
        log_dir = Path(log_dir_cfg) if log_dir_cfg else None
        log_tag = f"p2rank_{pdb_path.stem}"

        cmd = (
            f"{install_dir}/prank predict "
            f"-f {pdb_path} "
            f"-o {output_dir}"
        )
        result = run_command(
            cmd,
            timeout=timeout,
            log_dir=log_dir,
            log_tag=log_tag,
        )
        if not result.success:
            raise RuntimeError(
                f"P2Rank failed (rc={result.returncode}, "
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
        chain_id = sample_json["protein"]["chain_id"]
        # Multi-char chain ids (e.g. ribosome subunits "A5", "KK") get
        # rewritten to "A" by ``extract_protein_chain_pdb`` because the
        # PDB format only has a one-column chainID. P2Rank emits residue
        # tokens using whatever chain name it saw, so filter with the
        # rewritten name.
        effective_chain = effective_pdb_chain_id(chain_id)
        tool_cfg = (config.get("tools") or {}).get("p2rank") or {}
        score_thresh = float(tool_cfg.get("residue_score_threshold", 0.5))
        top_n = tool_cfg.get("top_pockets")
        top_n = int(top_n) if top_n is not None else None

        predictions_csv = _find_csv(output_dir, "_predictions.csv")
        residues_csv = _find_csv(output_dir, "_residues.csv")
        if predictions_csv is None:
            return self.fail(
                sample_id,
                f"missing P2Rank predictions CSV under {output_dir}",
                raw_output_dir=str(output_dir),
            )
        if residues_csv is None:
            return self.fail(
                sample_id,
                f"missing P2Rank residues CSV under {output_dir}",
                raw_output_dir=str(output_dir),
            )

        pockets = parse_predictions_csv(
            predictions_csv, chain_filter=effective_chain, top_n=top_n,
        )
        per_res = parse_residues_csv(residues_csv, chain_filter=effective_chain)
        binding = sorted(idx for idx, s in per_res.items() if s > score_thresh)

        return ToolPrediction(
            tool_id=self.tool_id,
            category=self.category,
            sample_id=sample_id,
            success=True,
            binding_protein_residues=binding,
            per_residue_confidence=per_res or None,
            pockets=pockets or None,
            raw_output_dir=str(output_dir),
        )
