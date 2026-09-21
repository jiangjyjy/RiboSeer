"""EquiPNAS adapter (Category C — RNA-binding residue prediction).

EquiPNAS predicts a per-residue probability of being an RNA-binding
residue. Unlike P2Rank, it needs a fairly heavy preprocessing pipeline
that turns a single chain PDB into a feature directory:

    <work_dir>/equipnas_in/
      input/
        <target>.pdb            (single protein chain, 1-based renumbered)
        <target>.fasta          (matching sequence, one-line)
      distmaps/                 (populated by the preprocessing scripts)
      input.list                (one line: <target>, NO trailing newline)

After staging, the preprocessing pipeline runs in order, all inside the
``EquiPNAS`` conda env:

    # 0. (this adapter) generate DSSP secondary-structure file —
    #    gen_aa_structural_features.py reads <target>.dssp alongside the PDB
    mkdssp <work_dir>/equipnas_in/input/<target>.pdb \\
           <work_dir>/equipnas_in/input/<target>.dssp

    # 1-3. upstream preprocessing scripts, cwd = <install_dir>/Preprocessing/
    python gen_aa_structural_features.py    -t <input.list>
    python genpssmto20feat.py               -i input -o tmp/  -t <input.list>
    python gen_preprocessed_node_5461features_new.py -t <input.list>

The third script materialises the 5461-dim node feature directory that
``EquiPNAS.py`` reads. PSSM and ESM-2 inputs are *assumed to already
exist on disk* (per EquiPNAS upstream design). Generating those is out of
scope for this adapter — when the user runs this on the server they must
either pre-stage them under ``input/`` or set
``tools.equipnas.skip_preprocess: true`` and pre-stage the entire
preprocessed directory.

External dependencies inside the EquiPNAS conda env
---------------------------------------------------
The preprocessing scripts shell out to several external tools that we do
NOT install via pip — they must be available inside the same conda env:

  * **DSSP** (``mkdssp``): secondary-structure annotation.
    Install: ``conda install -n EquiPNAS -c salilab dssp`` or
    ``conda install -n EquiPNAS -c bioconda dssp``.
  * **PSI-BLAST** (``psiblast``, used to build PSSMs): only needed when
    PSSMs are generated on the fly. If you cache PSSM files under
    ``input/`` ahead of time, this binary isn't required.
    Install: ``conda install -n EquiPNAS -c bioconda blast``.
  * **ESM-2 (``fair-esm``)**: python package that EquiPNAS imports to
    embed the protein sequence with the ESM-2 language model. Install:
    ``conda run -n EquiPNAS pip install fair-esm``. Model weights are
    downloaded automatically on first use, so the env needs network
    access the first time.

``_run_preprocessing`` runs three sanity checks inside the conda env
before launching the heavy scripts: ``which mkdssp``, ``import esm``,
and the actual ``mkdssp`` invocation that produces ``<target>.dssp``.
Each can be individually skipped via the matching
``skip_dssp_check`` / ``skip_esm_check`` / ``skip_mkdssp`` config flag
when the upstream constraint doesn't apply at the deployment site.

Then the main entrypoint runs:

    python EquiPNAS.py \
        --model_state_dict <model_path> \
        --indir  <preprocessed_dir>/ \
        --outdir <output_dir>/

The output is a per-target CSV with one row per residue and a binding
probability column. We parse it into ``per_residue_confidence`` and
threshold it for ``binding_protein_residues``.

Coordinate system note
----------------------
``extract_protein_chain_pdb`` (reused from p2rank_adapter) renumbers
residues to ``label_seq`` so the input PDB's residue numbers are 1-based
polymer indices, matching ``sample_json.protein.binding_residues``.
EquiPNAS reads those residue numbers verbatim into its output, so the
indices we emit in ``per_residue_confidence`` line up with step 1's
ground truth without further translation.
"""
from __future__ import annotations

import csv
import logging
import re
import shlex
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from ..base_adapter import BaseAdapter
from ..schemas import ToolPrediction
from ..tool_runner import run_in_conda_env
from .p2rank_adapter import (
    _find_raw_structure, effective_pdb_chain_id, extract_protein_chain_pdb,
)

_log = logging.getLogger(__name__)


# ---------- prepare_input helpers -----------------------------------------


_NAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9_]")


def _sanitize_target_name(sample_id: str) -> str:
    """EquiPNAS keys files by ``<target>.{pdb,fasta,...}`` so the name has
    to be filesystem-safe and free of separators that would confuse the
    preprocessing scripts (which split on ``_`` in some places).

    We replace any non-alphanumeric/underscore char with ``_`` and strip
    edge underscores. Empty result falls back to ``"sample"``.
    """
    cleaned = _NAME_SAFE_RE.sub("_", sample_id).strip("_")
    return cleaned or "sample"


def write_fasta(path: Path, target: str, sequence: str, *, line_width: int = 60) -> Path:
    """Write a one-record FASTA (target name + sequence wrapped at 60 cols).

    ``line_width=0`` writes the sequence on a single line.
    """
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


def stage_equipnas_input(
    raw_path: Path,
    chain_id: str,
    sequence: str,
    target: str,
    work_dir: Path,
) -> dict:
    """Materialise the input directory layout EquiPNAS expects.

    Returns a dict with ``preprocessed_dir`` (the parent dir),
    ``input_dir`` (where PDB / FASTA live), ``input_list`` (path to
    ``input.list``), ``pdb`` and ``fasta`` paths. Does not run any
    preprocessing scripts — that's the caller's job.
    """
    equipnas_in = work_dir / "equipnas_in"
    input_dir = equipnas_in / "input"
    distmaps_dir = equipnas_in / "distmaps"
    input_dir.mkdir(parents=True, exist_ok=True)
    distmaps_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = input_dir / f"{target}.pdb"
    extract_protein_chain_pdb(raw_path, chain_id, pdb_path)

    fasta_path = input_dir / f"{target}.fasta"
    write_fasta(fasta_path, target, sequence)

    input_list = equipnas_in / "input.list"
    # NO trailing newline: some upstream Preprocessing scripts read with
    # ``open(f).read().split("\n")`` and treat a trailing "" as an extra
    # target. That bug surfaced as `mkdssp .pdb .dssp` (empty target +
    # extension) in stage 4.x server tests. Writing the bare target avoids
    # the empty entry without requiring an upstream patch.
    input_list.write_text(target, encoding="utf-8")

    return {
        "preprocessed_dir": equipnas_in,
        "input_dir": input_dir,
        "input_list": input_list,
        "pdb": pdb_path,
        "fasta": fasta_path,
        "target": target,
    }


# ---------- parse_output helpers ------------------------------------------


# Possible column names for residue index and binding probability.
# EquiPNAS upstream isn't 100% consistent across versions; we accept any
# of these and pick the first that's present.
_RES_COL_CANDIDATES = (
    "residue_id", "residue_idx", "residue_index", "residue_label",
    "res_id", "resid", "residue", "position", "pos", "index", "idx",
)
_PROB_COL_CANDIDATES = (
    "binding_probability", "binding_prob", "rna_binding_prob",
    "probability", "prob", "score", "pred", "y_pred",
)
_CHAIN_COL_CANDIDATES = ("chain", "chain_id")


# Residue token like "A_14" or "A14"; we only keep the integer.
_RES_TOKEN_RE = re.compile(r"^([A-Za-z]+)?[_-]?(-?\d+)([A-Za-z])?$")


def _coerce_residue_index(token: str) -> Optional[int]:
    """Parse a residue token into a 1-based integer index. Returns None
    when the token can't be coerced."""
    s = (token or "").strip()
    if not s:
        return None
    m = _RES_TOKEN_RE.match(s)
    if not m:
        return None
    try:
        return int(m.group(2))
    except ValueError:
        return None


def _coerce_probability(value: str) -> Optional[float]:
    """Parse a float in [0, 1]. Returns None on bad input. We do not
    clamp silently — out-of-range values are dropped so a buggy upstream
    is visible rather than silently squashed to 1.0."""
    s = (value or "").strip()
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    if f != f:  # NaN
        return None
    if f < 0.0 or f > 1.0:
        return None
    return f


def _pick_column(fieldnames: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    """Case-insensitive lookup of the first matching column name."""
    if not fieldnames:
        return None
    lower_map = {(f or "").strip().lower(): f for f in fieldnames}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def _detect_dialect(text: str) -> csv.Dialect:
    """Sniff CSV dialect; fall back to whitespace-aware comma."""
    sample = text[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t ")
    except csv.Error:
        class _Comma(csv.excel):
            skipinitialspace = True
        return _Comma()


def _try_parse_single_column_floats(text: str) -> dict[int, float]:
    """Detect and parse the EquiPNAS ``.out`` format: one probability per
    line, no header, no separator. Line N (1-based) → residue N.

    Returns ``{}`` if the file doesn't conform — i.e. any non-blank /
    non-comment line either has internal whitespace/commas or fails to
    parse as a probability in [0, 1]. That falsy return lets the caller
    fall through to the CSV / whitespace-separated parsers without a
    second attempt.
    """
    probs: list[float] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if any(c in s for c in (",", "\t", " ")):
            return {}
        p = _coerce_probability(s)
        if p is None:
            return {}
        probs.append(p)
    if not probs:
        return {}
    return {i + 1: p for i, p in enumerate(probs)}


def parse_equipnas_output(
    path: Path,
    *,
    chain_filter: Optional[str] = None,
) -> dict[int, float]:
    """Parse an EquiPNAS per-residue prediction file into ``{idx: prob}``.

    Tolerates several layouts:
      * ``.out`` — one probability per line, no header, no residue index.
        Line N (1-based) is taken to mean residue N. This is the default
        format EquiPNAS upstream emits as of 2026-05; detected first
        because the headerless single-column shape is unambiguous.
      * CSV / TSV with header — comma or whitespace separated, with or
        without a chain column. Header column names from a few common
        sets (residue_id, score, probability, ...).
      * 2-column headerless — ``<res_idx> <prob>`` whitespace/comma
        separated.

    Rows with unparseable indices or out-of-range probabilities are
    dropped silently — keeping a partial result is more useful than
    failing the whole sample.

    ``chain_filter`` only applies to the CSV path with an explicit chain
    column; the ``.out`` and headerless layouts have no chain info.
    """
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}

    # ``.out`` format: every line is a bare float. Try this first because
    # the shape is unambiguous and the CSV sniffer would otherwise fail
    # on a single-column file in confusing ways.
    single_col = _try_parse_single_column_floats(text)
    if single_col:
        return single_col

    dialect = _detect_dialect(text)
    reader = csv.DictReader(text.splitlines(), dialect=dialect)
    fieldnames = [(f or "").strip() for f in (reader.fieldnames or [])]
    res_col = _pick_column(fieldnames, _RES_COL_CANDIDATES)
    prob_col = _pick_column(fieldnames, _PROB_COL_CANDIDATES)
    chain_col = _pick_column(fieldnames, _CHAIN_COL_CANDIDATES)

    out: dict[int, float] = {}

    if res_col is None or prob_col is None:
        # No usable header — try a 2-column headerless layout: <res> <prob>
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[,\s]+", line)
            if len(parts) < 2:
                continue
            idx = _coerce_residue_index(parts[0])
            prob = _coerce_probability(parts[1])
            if idx is None or prob is None:
                continue
            out[idx] = prob
        return dict(sorted(out.items()))

    for row in reader:
        if not row:
            continue
        if chain_filter is not None and chain_col is not None:
            row_chain = (row.get(chain_col) or "").strip()
            if row_chain and row_chain != chain_filter:
                continue
        idx = _coerce_residue_index(str(row.get(res_col, "")))
        prob = _coerce_probability(str(row.get(prob_col, "")))
        if idx is None or prob is None:
            continue
        out[idx] = prob
    return dict(sorted(out.items()))


# Backwards-compatible alias. Older callers (and a few tests) imported
# parse_equipnas_csv before the upstream .out format was supported.
parse_equipnas_csv = parse_equipnas_output


def find_equipnas_output(output_dir: Path, target: str) -> Optional[Path]:
    """Locate EquiPNAS's per-target output file under ``output_dir``.

    Search order, by extension:
      1. ``.out`` — current upstream default (one probability per line,
         line index = residue index).
      2. ``.csv`` / ``.pred`` / ``.tsv`` / ``.txt`` — older variants and
         hand-converted dumps; ``parse_equipnas_output`` autodetects.

    For each extension we first probe ``output_dir/<target><ext>`` exactly,
    then fall back to ``rglob`` in case EquiPNAS dropped the file in a
    nested subdir. The very last fallback returns the first matching file
    of *any* known extension under output_dir.
    """
    extensions = (".out", ".csv", ".pred", ".tsv", ".txt")
    for ext in extensions:
        direct = output_dir / f"{target}{ext}"
        if direct.is_file():
            return direct
    for ext in extensions:
        matches = sorted(output_dir.rglob(f"{target}{ext}"))
        if matches:
            return matches[0]
    # Final fallback — first matching file anywhere under output_dir,
    # ordered by extension preference.
    for ext in extensions:
        any_match = sorted(output_dir.rglob(f"*{ext}"))
        if any_match:
            return any_match[0]
    return None


# ---------- adapter --------------------------------------------------------


# The 3 preprocessing scripts EquiPNAS ships under Preprocessing/. Order
# matters: the third one consumes outputs of the first two.
PREPROCESS_SCRIPTS = (
    "gen_aa_structural_features.py",
    "genpssmto20feat.py",
    "gen_preprocessed_node_5461features_new.py",
)


# tmp/ feature files written by the 3 preprocessing scripts. Their first
# axis must equal the protein sequence length, but DSSP routinely drops
# residues with missing atoms or non-standard chemistry, so the DSSP-derived
# files come out shorter than PSSM/imputed/concount files. aligns
# them all to the FASTA length before gen_preprocessed_node_5461features_new
# stacks them.
_FEATURE_FILES_TO_ALIGN = (
    "{target}.feat.npy",
    "{target}.feat22.npy",
    "{target}.feat_angle6.npy",
    "{target}.feat_forw_rev_ca6.npy",
    "{target}.imputed3.npy",
    "{target}.concount.npy",
    "{target}.npy",  # PSSM
)


def _read_fasta_seq_len(fasta_path: Path) -> int:
    """Sum the length of every non-header line in a FASTA file.

    EquiPNAS only ever sees a single record per file (one chain → one
    target), so we don't need a real parser — just strip the ``>`` line and
    concatenate the rest. Used by the alignment helper to discover the
    canonical sequence length without re-deriving it from the PDB.
    """
    text = fasta_path.read_text(encoding="utf-8")
    return sum(
        len(line.strip())
        for line in text.splitlines()
        if line.strip() and not line.startswith(">")
    )


def _symlink_preprocessing_scripts(
    preprocess_dir: Path, install_dir: Path,
) -> None:
    """Mirror ``install_dir/Preprocessing/*.py`` into ``preprocess_dir``.

    Several of the upstream preprocessing scripts shell out to *sibling*
    scripts via ``os.system("python extract_dssp_feat.py ...")`` with a
    bare relative path. set cwd to the staged dir, but the
    sibling scripts only live under ``install_dir/Preprocessing/``, so the
    relative invocation can't find them. Linking every ``*.py`` from
    ``Preprocessing/`` into the staged dir lets the relative invocation
    resolve without patching upstream code.

    Falls back to ``shutil.copy2`` when symlink creation isn't permitted
    (Windows without developer mode / admin) — same end result for the
    upstream script and the test suite needs to pass on Windows too.
    """
    src_dir = install_dir / "Preprocessing"
    if not src_dir.is_dir():
        # No Preprocessing dir to mirror — let the downstream call fail
        # naturally with a clear "script not found" rather than crashing
        # here on a config issue.
        return
    for py_file in src_dir.glob("*.py"):
        link = preprocess_dir / py_file.name
        if link.exists() or link.is_symlink():
            continue
        target = py_file.resolve()
        try:
            link.symlink_to(target)
        except OSError:
            shutil.copy2(target, link)


def _align_feature_arrays(
    preprocess_dir: Path, target: str, seq_len: int,
) -> None:
    """Pad/trim every per-residue feature in ``tmp/`` to ``seq_len``.

    DSSP-derived files (``.feat`` / ``.feat22`` / ``.feat_angle6`` /
    ``.feat_forw_rev_ca6``) come out shorter than ``seq_len`` because
    mkdssp 3.x drops residues with missing backbone atoms; PSSM /
    imputed / concount files come out at the full PDB length. Without
    alignment ``gen_preprocessed_node_5461features_new`` indexes past
    the end of the short arrays and dies with ``IndexError``.

    Padding uses zeros (``mode='constant'``) — for one-hot DSSP
    secondary-structure features that maps cleanly onto "no annotation",
    which is the safest default for a residue we have no info about.

    The ESM-2 (``rep_5120.npy``) and MSA-first-row files live under
    ``input/`` and arrive at ``seq_len + 2`` (BOS/EOS tokens for ESM-2);
    upstream's ``gen_preprocessed`` already slices ``[1:-1]`` and
    ``[:maxlen]`` respectively, so we deliberately leave them alone.
    """
    tmp_dir = preprocess_dir / "tmp"
    if not tmp_dir.is_dir():
        return
    for template in _FEATURE_FILES_TO_ALIGN:
        fpath = tmp_dir / template.format(target=target)
        if not fpath.is_file():
            continue
        arr = np.load(fpath)
        if arr.ndim < 1:
            continue
        old_len = int(arr.shape[0])
        if old_len == seq_len:
            continue
        if arr.ndim == 1:
            if old_len < seq_len:
                arr = np.pad(arr, (0, seq_len - old_len), mode="constant")
            else:
                arr = arr[:seq_len]
        else:
            if old_len < seq_len:
                pad_width = [(0, seq_len - old_len)] + [(0, 0)] * (arr.ndim - 1)
                arr = np.pad(arr, pad_width, mode="constant")
            else:
                arr = arr[:seq_len]
        np.save(fpath, arr)
        _log.warning(
            "EquiPNAS: aligned %s from %d to %d", fpath.name, old_len, seq_len,
        )


class EquiPNASAdapter(BaseAdapter):
    """Adapter for EquiPNAS-RNA (Cat C)."""

    tool_id = "equipnas"
    category = "C"

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
        sequence = (sample_json.get("protein") or {}).get("sequence")
        if not sequence:
            raise ValueError(f"sample {sample_id!r} has no protein.sequence")

        ss_cfg = config.get("structure_source") or {}
        raw_dir = Path(ss_cfg.get("raw_dir") or "data/raw")
        raw_path = _find_raw_structure(raw_dir, source_pdb)
        if raw_path is None:
            raise FileNotFoundError(
                f"no raw structure for {source_pdb!r} under {raw_dir} "
                f"(tried .pdb, .cif)"
            )

        target = _sanitize_target_name(sample_id)
        staged = stage_equipnas_input(
            raw_path=raw_path,
            chain_id=chain_id,
            sequence=sequence,
            target=target,
            work_dir=work_dir,
        )

        # Run preprocessing scripts unless the caller has pre-staged the
        # preprocessed dir externally (e.g. PSSM / ESM-2 already cached).
        tool_cfg = (config.get("tools") or {}).get("equipnas") or {}
        if not bool(tool_cfg.get("skip_preprocess", False)):
            self._run_preprocessing(
                staged=staged,
                config=config,
            )

        return staged

    def _ensure_dssp_available(
        self, env_name: str, log_dir: Optional[Path],
    ) -> None:
        """Fail fast if DSSP is missing inside the EquiPNAS conda env.

        ``gen_aa_structural_features.py`` shells out to ``mkdssp`` (and on
        some EquiPNAS forks falls back to a binary literally named
        ``dssp``). When neither is on PATH the script silently emits an
        empty ``.dssp`` filename to its inner ``os.system`` call, which
        surfaces only as ``sh: 2: .dssp: not found`` minutes later — too
        opaque to debug. We probe ``which`` up-front and raise a
        readable error instead.

        Skip the check entirely with
        ``tools.equipnas.skip_dssp_check: true`` for sites that put DSSP
        somewhere unusual (e.g. a system-wide ``mkdssp`` not visible to
        ``which`` inside ``conda run``).
        """
        result = run_in_conda_env(
            env_name,
            "bash -c 'command -v mkdssp || command -v dssp'",
            timeout=30,
            log_dir=log_dir,
            log_tag=f"equipnas_dssp_check_{env_name}",
        )
        if result.success and (result.stdout or "").strip():
            return
        raise RuntimeError(
            "DSSP binary (mkdssp / dssp) not found inside conda env "
            f"'{env_name}'. EquiPNAS preprocessing requires DSSP for "
            "secondary-structure annotation. Install it with one of:\n"
            f"  conda install -n {env_name} -c salilab dssp\n"
            f"  conda install -n {env_name} -c bioconda dssp\n"
            "or set `tools.equipnas.skip_dssp_check: true` in the config "
            "if DSSP is provided through a non-standard path."
        )

    def _ensure_psiblast_available(
        self, env_name: str, log_dir: Optional[Path],
    ) -> None:
        """Fail fast if ``psiblast`` is missing inside the EquiPNAS conda env.

        generates the per-target PSSM on the fly with PSI-BLAST.
        When the binary is missing the failure surfaces several minutes
        into preprocessing as a cryptic ``CalledProcessError`` from inside
        ``_generate_pssm`` — same problem we already solved for DSSP/ESM.

        Skip with ``tools.equipnas.skip_psiblast_check: true`` (or
        ``skip_pssm: true`` if the user has pre-generated PSSM files).
        """
        result = run_in_conda_env(
            env_name,
            "bash -c 'command -v psiblast'",
            timeout=30,
            log_dir=log_dir,
            log_tag=f"equipnas_psiblast_check_{env_name}",
        )
        if result.success and (result.stdout or "").strip():
            return
        raise RuntimeError(
            "PSI-BLAST (psiblast) not found inside conda env "
            f"'{env_name}'. EquiPNAS preprocessing needs it to build "
            "per-target PSSMs. Install with:\n"
            f"  conda install -n {env_name} -c bioconda blast\n"
            "or set `tools.equipnas.skip_psiblast_check: true` (and "
            "`skip_pssm: true` if PSSMs are already cached on disk)."
        )

    def _ensure_esm_available(
        self, env_name: str, log_dir: Optional[Path],
    ) -> None:
        """Fail fast if the ``esm`` (fair-esm) package is missing.

        EquiPNAS calls ``import esm`` to embed the protein sequence with
        Meta's ESM-2 language model. When fair-esm isn't installed the
        upstream script aborts with a python ImportError several minutes
        into preprocessing. We probe ``python -c "import esm"`` inside
        the conda env up-front and raise with the install command.

        Skip with ``tools.equipnas.skip_esm_check: true`` if your
        EquiPNAS fork uses an alternative protein LM or you've vendored
        ESM under a non-standard name.
        """
        result = run_in_conda_env(
            env_name,
            'python -c "import esm"',
            timeout=60,
            log_dir=log_dir,
            log_tag=f"equipnas_esm_check_{env_name}",
        )
        if result.success:
            return
        raise RuntimeError(
            "fair-esm (the `esm` Python package) not found inside conda "
            f"env '{env_name}'. EquiPNAS requires it for ESM-2 sequence "
            f"embeddings. Install with:\n"
            f"  conda run -n {env_name} pip install fair-esm\n"
            "or set `tools.equipnas.skip_esm_check: true` in the config "
            "if your fork uses a different protein LM."
        )

    def _run_mkdssp_for_target(
        self,
        env_name: str,
        input_dir: Path,
        target: str,
        timeout: int,
        log_dir: Optional[Path],
    ) -> None:
        """Generate ``input/<target>.dssp`` from ``input/<target>.pdb``.

        ``gen_aa_structural_features.py`` reads ``<target>.dssp`` from
        the same directory as ``<target>.pdb``; if it's missing the
        script crashes with ``FileNotFoundError: 'input/<target>.dssp'``.
        We pre-generate the DSSP annotation here.

        Tries the modern positional invocation first
        (``mkdssp <pdb> <dssp>``, the form salilab/bioconda 4.x ship)
        and falls back to the older ``-i / -o`` flag form. Both attempts
        are absolute-path-quoted so the working directory doesn't
        matter. Failure raises ``RuntimeError`` — BaseAdapter catches it
        and surfaces a ``success=False`` ToolPrediction with the message
        baked in (the user requirement: "record the error but don't crash" = adapter
        records failure cleanly, doesn't take down the batch).
        """
        pdb = (input_dir / f"{target}.pdb").resolve()
        dssp = (input_dir / f"{target}.dssp").resolve()
        if not pdb.is_file():
            raise FileNotFoundError(
                f"mkdssp input PDB missing: {pdb} "
                f"(expected after stage_equipnas_input)"
            )

        pdb_q = shlex.quote(str(pdb))
        dssp_q = shlex.quote(str(dssp))

        # mkdssp 4.x (salilab / modern bioconda) accepts positional args;
        # mkdssp 3.x and the older "dssp" rename use -i/-o. Try both so
        # the adapter works across server installs without a config knob.
        attempts = [
            ("positional", f"mkdssp {pdb_q} {dssp_q}"),
            ("flag",       f"mkdssp -i {pdb_q} -o {dssp_q}"),
        ]
        last: Optional[tuple[str, object]] = None
        for label, cmd in attempts:
            # Remove any half-written output from a previous attempt so
            # ``dssp.is_file()`` only returns True if the current attempt
            # actually wrote something.
            if dssp.is_file():
                try:
                    dssp.unlink()
                except OSError:
                    pass
            result = run_in_conda_env(
                env_name, cmd,
                timeout=timeout,
                log_dir=log_dir,
                log_tag=f"equipnas_mkdssp_{label}_{target}",
            )
            if result.success and dssp.is_file() and dssp.stat().st_size > 0:
                return
            last = (label, result)

        label, result = last  # type: ignore[misc]
        rc = getattr(result, "returncode", "?")
        stderr = getattr(result, "stderr", "") or ""
        raise RuntimeError(
            f"mkdssp failed for target {target!r} (last attempt: "
            f"{label} form, rc={rc}). The PDB file may be malformed or "
            f"the mkdssp version may not support either invocation form. "
            f"stderr (truncated): {stderr[:500]}"
        )

    def _generate_pssm(
        self,
        env_name: str,
        preprocess_dir: Path,
        target: str,
        tool_cfg: dict,
        timeout: int,
        log_dir: Optional[Path],
    ) -> None:
        """Build ``input/<target>.pssm`` with PSI-BLAST.

        ``genpssmto20feat.py`` expects this file to exist alongside the
        FASTA. stops requiring users to pre-generate it: we run
        ``psiblast`` against the configured BLAST DB inside the conda env.

        DB selection: ``pssm_db`` (default UniRef50) is tried first; if
        that invocation fails *and* ``pssm_fallback_db`` is configured,
        we retry once with the fallback. The intended use is "UniRef50
        is still downloading on the server, fall back to SwissProt for
        smoke runs" — a pragmatic relief valve, not a permanent setting.

        Skip with ``tools.equipnas.skip_pssm: true`` when the PSSM is
        already cached. The cache check ``pssm_out.exists()`` runs first
        regardless, so re-runs of the same target don't re-blast.
        """
        fasta = (preprocess_dir / "input" / f"{target}.fasta").resolve()
        pssm_out = (preprocess_dir / "input" / f"{target}.pssm").resolve()
        if pssm_out.is_file():
            return
        if not fasta.is_file():
            raise FileNotFoundError(
                f"PSSM input FASTA missing: {fasta} "
                f"(expected after stage_equipnas_input)"
            )

        primary_db = tool_cfg.get(
            "pssm_db", "/opt/biotools/blast_db/uniref50",
        )
        fallback_db = tool_cfg.get("pssm_fallback_db")
        iterations = int(tool_cfg.get("pssm_num_iterations", 3))
        evalue = tool_cfg.get("pssm_evalue", 0.001)
        threads = int(tool_cfg.get("pssm_threads", 8))

        fasta_q = shlex.quote(str(fasta))
        pssm_q = shlex.quote(str(pssm_out))

        attempts: list[tuple[str, str]] = []
        for label, db in (("primary", primary_db), ("fallback", fallback_db)):
            if not db:
                continue
            db_q = shlex.quote(str(db))
            attempts.append((
                label,
                f"psiblast -query {fasta_q} -db {db_q} "
                f"-num_iterations {iterations} -evalue {evalue} "
                f"-num_threads {threads} -out_ascii_pssm {pssm_q}",
            ))

        last_err = ""
        for label, cmd in attempts:
            result = run_in_conda_env(
                env_name, cmd,
                cwd=preprocess_dir.resolve(),
                timeout=timeout,
                log_dir=log_dir,
                log_tag=f"equipnas_pssm_{label}_{target}",
            )
            if result.success and pssm_out.is_file() and pssm_out.stat().st_size > 0:
                if label == "fallback":
                    _log.warning(
                        "EquiPNAS: PSSM fell back from primary DB to "
                        "%s for target %s", fallback_db, target,
                    )
                return
            last_err = (result.stderr or "")[:500]
            # Wipe any half-written file so the next attempt starts clean.
            if pssm_out.is_file():
                try:
                    pssm_out.unlink()
                except OSError:
                    pass

        raise RuntimeError(
            f"PSI-BLAST failed to build PSSM for target {target!r}. "
            f"Tried {len(attempts)} DB(s); last stderr: {last_err}"
        )

    def _generate_esm2_embedding(
        self,
        env_name: str,
        preprocess_dir: Path,
        target: str,
        tool_cfg: dict,
        timeout: int,
        log_dir: Optional[Path],
    ) -> None:
        """Build ``input/<target>.rep_5120.npy`` with EquiPNAS's ESM-2 helper.

        EquiPNAS ships ``input_details/supporting_scripts/esm2_15B_rep_5120.py``
        which loads ESM-2 (15B) and dumps the per-residue 5120-dim
        embedding to disk. We invoke it inside the EquiPNAS conda env so
        the heavy fair-esm dep is satisfied.

        The script writes ``<rep_out>.npy`` (or just appends ``.npy`` to
        whatever ``-o`` we pass), so the existence check uses the
        ``.rep_5120.npy`` suffix while the ``-o`` arg uses the
        extension-less stem — that's the upstream convention.

        Override ``esm2_script`` to point at a different helper, or set
        ``skip_esm2: true`` when the rep file is already cached.
        """
        fasta = (preprocess_dir / "input" / f"{target}.fasta").resolve()
        rep_stem = (preprocess_dir / "input" / f"{target}.rep_5120").resolve()
        npy_out = (preprocess_dir / "input" / f"{target}.rep_5120.npy").resolve()
        if npy_out.is_file():
            return
        if not fasta.is_file():
            raise FileNotFoundError(
                f"ESM-2 input FASTA missing: {fasta}"
            )

        install_dir = tool_cfg.get("install_dir")
        default_script = (
            Path(install_dir) / "input_details" / "supporting_scripts"
            / "esm2_15B_rep_5120.py"
        ) if install_dir else None
        script_path = tool_cfg.get("esm2_script") or default_script
        if not script_path:
            raise ValueError(
                "no ESM-2 script path: set tools.equipnas.esm2_script "
                "or tools.equipnas.install_dir"
            )

        cmd = (
            f"python {shlex.quote(str(script_path))} "
            f"-i {shlex.quote(str(fasta))} "
            f"-o {shlex.quote(str(rep_stem))}"
        )
        result = run_in_conda_env(
            env_name, cmd,
            cwd=preprocess_dir.resolve(),
            timeout=timeout,
            log_dir=log_dir,
            log_tag=f"equipnas_esm2_{target}",
        )
        if not result.success or not npy_out.is_file():
            raise RuntimeError(
                f"ESM-2 embedding generation failed for target {target!r} "
                f"(rc={getattr(result, 'returncode', '?')}). "
                f"stderr: {(result.stderr or '')[:500]}"
            )

    def _generate_dummy_msa_first_row(
        self,
        preprocess_dir: Path,
        target: str,
        seq_len: int,
    ) -> None:
        """Materialise a zero ``input/<target>msa_first_row.npy``.

        ``gen_preprocessed_node_5461features_new`` expects an MSA-first-row
        embedding to exist. When no real MSA is available we emit a
        zero-filled placeholder of shape ``(seq_len + 2, 256)`` — the +2
        accounts for the BOS/EOS tokens upstream slices off via
        ``[1:-1]``. Note the filename has *no* separator between target
        and ``msa_first_row``: that's the upstream convention.

        Idempotent: skips if the file already exists. No subprocess —
        ``np.save`` only.
        """
        out = preprocess_dir / "input" / f"{target}msa_first_row.npy"
        if out.is_file():
            return
        if seq_len <= 0:
            raise ValueError(
                f"cannot synthesise MSA first row: seq_len={seq_len}"
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        dummy = np.zeros((seq_len + 2, 256), dtype=np.float32)
        np.save(out, dummy)

    def _generate_distance_map(
        self,
        env_name: str,
        preprocess_dir: Path,
        install_dir: Path,
        target: str,
        tool_cfg: dict,
        timeout: int,
        log_dir: Optional[Path],
    ) -> None:
        """Run ``Preprocessing/pdb2rr.py`` to produce the Cb-Cb distance map.

        The third preprocessing script reads ``distmaps/<target>.dist``;
        the ``pdb2rr`` helper that ships with EquiPNAS computes it from
        the staged PDB. The threshold flag (``-t 14``) matches upstream
        defaults.

        Idempotent on the output file. Skip with ``skip_distance_map: true``
        when the file is already cached. Override the ``-t`` cutoff via
        ``distance_map_threshold``.
        """
        pdb = (preprocess_dir / "input" / f"{target}.pdb").resolve()
        dist_out = (preprocess_dir / "distmaps" / f"{target}.dist").resolve()
        if dist_out.is_file():
            return
        if not pdb.is_file():
            raise FileNotFoundError(
                f"distance-map input PDB missing: {pdb}"
            )

        script_path = (install_dir / "Preprocessing" / "pdb2rr.py").resolve()
        threshold = int(tool_cfg.get("distance_map_threshold", 14))

        dist_out.parent.mkdir(parents=True, exist_ok=True)
        cmd = (
            f"python {shlex.quote(str(script_path))} "
            f"-p {shlex.quote(str(pdb))} "
            f"-o {shlex.quote(str(dist_out))} "
            f"-t {threshold}"
        )
        result = run_in_conda_env(
            env_name, cmd,
            cwd=preprocess_dir.resolve(),
            timeout=timeout,
            log_dir=log_dir,
            log_tag=f"equipnas_pdb2rr_{target}",
        )
        if not result.success or not dist_out.is_file():
            raise RuntimeError(
                f"pdb2rr.py failed to build distance map for target "
                f"{target!r} (rc={getattr(result, 'returncode', '?')}). "
                f"stderr: {(result.stderr or '')[:500]}"
            )

    def _run_preprocessing(self, *, staged: dict, config: dict) -> None:
        """Run DSSP + dependency generation + 3 EquiPNAS scripts in order.

        Each call lives inside the ``EquiPNAS`` conda env. **All steps
        run with cwd set to ``<work_dir>/equipnas_in/``** — the upstream
        scripts use relative paths like ``input/<target>.dssp`` and
        ``distmaps/...`` which only resolve correctly when cwd is the
        staged input root. Scripts themselves are referenced by absolute
        path so they can live anywhere on disk.

        cwd points at the staged dir, not install_dir.
        PSSM / ESM-2 / MSA-first-row / distance-map generation
        is now built in. Order:

            ensure_dssp / ensure_esm / ensure_psiblast
            → mkdssp                          (DSSP annotation)
            → symlink Preprocessing/*.py      (sibling-script resolution)
            → _generate_pssm                  (PSI-BLAST)
            → _generate_esm2_embedding        (ESM-2 .rep_5120.npy)
            → _generate_dummy_msa_first_row   (zero-filled placeholder)
            → _generate_distance_map          (pdb2rr.py)
            → gen_aa_structural_features.py
            → genpssmto20feat.py              (PSSM → 20-dim features)
            → _align_feature_arrays           (pad/trim to FASTA length)
            → gen_preprocessed_node_5461features_new.py
        """
        tool_cfg = (config.get("tools") or {}).get("equipnas") or {}
        install_dir = tool_cfg.get("install_dir")
        if not install_dir:
            raise ValueError("config['tools']['equipnas']['install_dir'] not set")
        env_name = tool_cfg.get("conda_env", "EquiPNAS")
        timeout = int(tool_cfg.get("preprocess_timeout", 1800))
        dssp_timeout = int(tool_cfg.get("dssp_timeout", 300))
        pssm_timeout = int(tool_cfg.get("pssm_timeout", 7200))
        esm2_timeout = int(tool_cfg.get("esm2_timeout", 1800))
        distmap_timeout = int(tool_cfg.get("distance_map_timeout", 600))

        install_path = Path(install_dir)
        preprocessing_dir = (install_path / "Preprocessing").resolve()
        # cwd for ALL preprocessing-related calls. Relative paths in the
        # upstream scripts resolve against this directory.
        preprocess_cwd = staged["preprocessed_dir"].resolve()
        input_dir_abs = staged["input_dir"].resolve()
        tmp_dir = (staged["preprocessed_dir"] / "tmp").resolve()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # gen_preprocessed_node_5461features_new.py writes to
        # processed_features/<target>.5461featnew.npy but does NOT
        # create the directory itself; missing dir → FileNotFoundError
        # at np.save time. Create it alongside tmp/ and distmaps/.
        (staged["preprocessed_dir"] / "processed_features").mkdir(
            parents=True, exist_ok=True,
        )

        log_dir_cfg = config.get("log_dir")
        log_dir = Path(log_dir_cfg) if log_dir_cfg else None

        # Sanity-check external dependencies before launching the scripts.
        if not bool(tool_cfg.get("skip_dssp_check", False)):
            self._ensure_dssp_available(env_name, log_dir)
        if not bool(tool_cfg.get("skip_esm_check", False)):
            self._ensure_esm_available(env_name, log_dir)
        # PSI-BLAST is only needed when we're about to build a PSSM —
        # if the user pre-staged it (skip_pssm), there's nothing to probe.
        if (not bool(tool_cfg.get("skip_psiblast_check", False))
                and not bool(tool_cfg.get("skip_pssm", False))):
            self._ensure_psiblast_available(env_name, log_dir)

        # Generate <target>.dssp from <target>.pdb (gen_aa_structural_features
        # expects to read this; missing-file → 'input/<target>.dssp' error).
        if not bool(tool_cfg.get("skip_mkdssp", False)):
            self._run_mkdssp_for_target(
                env_name=env_name,
                input_dir=input_dir_abs,
                target=staged["target"],
                timeout=dssp_timeout,
                log_dir=log_dir,
            )

        # Mirror Preprocessing/*.py into the staged dir. Required because
        # the upstream scripts call sibling scripts via os.system("python
        # extract_dssp_feat.py ...") with bare relative paths — those only
        # resolve when the staged dir contains the .py files.
        _symlink_preprocessing_scripts(preprocess_cwd, install_path)

        # Read FASTA seq_len once: needed for both the MSA placeholder
        # (must match seq_len + 2) and the post-genpssm alignment step.
        seq_len = _read_fasta_seq_len(staged["fasta"])

        # build the four input artifacts the 3 main scripts
        # consume. Each helper is idempotent (re-runs of the same target
        # skip generation if the output is already on disk) and each
        # skip_* flag lets users supply their own pre-staged file.
        if not bool(tool_cfg.get("skip_pssm", False)):
            self._generate_pssm(
                env_name=env_name,
                preprocess_dir=preprocess_cwd,
                target=staged["target"],
                tool_cfg=tool_cfg,
                timeout=pssm_timeout,
                log_dir=log_dir,
            )
        if not bool(tool_cfg.get("skip_esm2", False)):
            self._generate_esm2_embedding(
                env_name=env_name,
                preprocess_dir=preprocess_cwd,
                target=staged["target"],
                tool_cfg=tool_cfg,
                timeout=esm2_timeout,
                log_dir=log_dir,
            )
        if not bool(tool_cfg.get("skip_msa_dummy", False)):
            self._generate_dummy_msa_first_row(
                preprocess_dir=preprocess_cwd,
                target=staged["target"],
                seq_len=seq_len,
            )
        if not bool(tool_cfg.get("skip_distance_map", False)):
            self._generate_distance_map(
                env_name=env_name,
                preprocess_dir=preprocess_cwd,
                install_dir=install_path,
                target=staged["target"],
                tool_cfg=tool_cfg,
                timeout=distmap_timeout,
                log_dir=log_dir,
            )

        # Args use paths RELATIVE to preprocess_cwd (= work_dir/equipnas_in).
        # The directory layout is:
        #   preprocess_cwd/
        #     input/      → -i input
        #     tmp/        → -o tmp
        #     input.list  → -t input.list
        #     distmaps/
        # So relative paths just work and the scripts find their inputs.
        script_args = {
            "gen_aa_structural_features.py":
                "-t input.list",
            "genpssmto20feat.py":
                "-i input -o tmp -t input.list",
            "gen_preprocessed_node_5461features_new.py":
                "-t input.list",
        }

        # ``seq_len`` was already read above (FASTA written by
        # stage_equipnas_input) so the MSA placeholder generator could
        # use it. Re-used here for post-genpssm tmp/ array alignment.

        for script in PREPROCESS_SCRIPTS:
            # Script lives in install_dir/Preprocessing/, but we run from
            # preprocess_cwd. Use absolute script path so cwd doesn't
            # have to be Preprocessing/.
            script_abs = preprocessing_dir / script
            args = script_args[script]
            cmd = f"python {script_abs} {args}"
            log_tag = f"equipnas_pre_{Path(script).stem}_{staged['target']}"
            result = run_in_conda_env(
                env_name, cmd,
                cwd=preprocess_cwd,
                timeout=timeout,
                log_dir=log_dir,
                log_tag=log_tag,
            )
            if not result.success:
                raise RuntimeError(
                    f"EquiPNAS preprocessing failed at {script} "
                    f"(rc={result.returncode}, timed_out={result.timed_out}). "
                    f"See log: {result.log_path}. "
                    f"stderr: {result.stderr[:500]}"
                )
            # Align tmp/ feature arrays to seq_len after the *second*
            # script (genpssmto20feat) so the third script
            # (gen_preprocessed_node_5461features_new) sees uniformly
            # shaped inputs. Running it after the third script would be
            # too late — that script is the consumer that crashes on the
            # length mismatch.
            if script == "genpssmto20feat.py" and seq_len > 0:
                _align_feature_arrays(preprocess_cwd, staged["target"], seq_len)

    # ------------------------------------------------------------------ run

    def run_tool(
        self,
        input_paths: dict,
        work_dir: Path,
        config: dict,
    ) -> Path:
        tool_cfg = (config.get("tools") or {}).get("equipnas") or {}
        install_dir = tool_cfg.get("install_dir")
        if not install_dir:
            raise ValueError("config['tools']['equipnas']['install_dir'] not set")
        env_name = tool_cfg.get("conda_env", "EquiPNAS")
        model_path = tool_cfg.get("model_path", "models/EquiPNAS-RNA/E-l12-768.pt")
        timeout = int(tool_cfg.get("timeout", 1200))

        preprocessed_dir = Path(input_paths["preprocessed_dir"]).resolve()
        if not preprocessed_dir.is_dir():
            raise FileNotFoundError(
                f"prepared EquiPNAS input dir missing: {preprocessed_dir}"
            )

        output_dir = (work_dir / "equipnas_output").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = (
            f"python EquiPNAS.py "
            f"--model_state_dict {model_path} "
            f"--indir {preprocessed_dir} "
            f"--outdir {output_dir}"
        )

        log_dir_cfg = config.get("log_dir")
        log_dir = Path(log_dir_cfg) if log_dir_cfg else None
        log_tag = f"equipnas_{input_paths.get('target', 'run')}"

        result = run_in_conda_env(
            env_name, cmd,
            cwd=install_dir,
            timeout=timeout,
            log_dir=log_dir,
            log_tag=log_tag,
        )
        if not result.success:
            raise RuntimeError(
                f"EquiPNAS failed (rc={result.returncode}, "
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
        chain_id = (sample_json.get("protein") or {}).get("chain_id")
        # Multi-char chain ids get rewritten to "A" inside the staged PDB
        # (PDB format limit) — apply the same mapping when filtering on
        # the chain column of older CSV-formatted EquiPNAS outputs. The
        # default ``.out`` format has no chain column so this is a no-op
        # for the common case.
        effective_chain = effective_pdb_chain_id(chain_id) if chain_id else None
        target = _sanitize_target_name(sample_id)
        tool_cfg = (config.get("tools") or {}).get("equipnas") or {}
        thresh = float(tool_cfg.get("residue_prob_threshold", 0.5))

        out_path = find_equipnas_output(output_dir, target)
        if out_path is None:
            return self.fail(
                sample_id,
                f"no EquiPNAS output for target {target!r} under {output_dir}",
                raw_output_dir=str(output_dir),
            )

        per_res = parse_equipnas_output(out_path, chain_filter=effective_chain)
        if not per_res:
            return self.fail(
                sample_id,
                f"EquiPNAS output {out_path} parsed to 0 per-residue rows",
                raw_output_dir=str(output_dir),
            )

        binding = sorted(idx for idx, p in per_res.items() if p > thresh)

        return ToolPrediction(
            tool_id=self.tool_id,
            category=self.category,
            sample_id=sample_id,
            success=True,
            binding_protein_residues=binding,
            per_residue_confidence=per_res,
            raw_output_dir=str(output_dir),
        )
