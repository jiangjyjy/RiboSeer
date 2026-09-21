"""RoseTTAFold2NA adapter (Category A — RNA-protein structure prediction).

Workflow
--------
1. ``prepare_input``: write the protein and RNA sequences out as
   single-record FASTAs under ``<work_dir>``. ``run_RF2NA.sh`` consumes
   files prefixed ``P:`` (protein) and ``R:`` (RNA) on the command line.

2. ``run_tool``::

       cd <install_dir>
       bash run_RF2NA.sh <output_dir> P:<protein.fa> R:<rna.fa>

   Inside ``conda run -n RF2NA2`` so the run picks up the dedicated env's
   pyrosetta / DGL / SE(3)-Transformer build.

3. ``parse_output``: RF2NA emits ``<output_dir>/models/model_00.pdb``
   (predicted complex with pLDDT in the B-factor column) and, when
   available, ``model_00.npz`` with predicted aligned-error metrics.

   - protein-RNA contacts via ``contact_extractor`` at 4.5 Å heavy atom
   - per-residue pLDDT from CA B-factor (``read_protein_plddt_from_cif``
     reused from boltz2_adapter — gemmi reads PDB the same way it
     reads mmCIF, the function name is historical)
   - ``pae_mean`` from the npz when present (``read_pae_mean_from_npz``
     reused from boltz2_adapter)

Status
------
The RF2NA database (~230 GB) is still downloading on the server. The
adapter code is ready but ``run_tool`` will fail with a database-missing
error until the download completes. Mock tests cover everything the
adapter does locally; the server-side smoke (`test_rf2na_real.py`) is
written but not runnable yet.
"""
from __future__ import annotations

import re
import shutil
import stat
from pathlib import Path
from typing import Optional

from ..base_adapter import BaseAdapter
from ..contact_extractor import extract_contacts
from ..schemas import ToolPrediction
from ..tool_runner import run_in_conda_env
from .boltz2_adapter import (
    read_pae_mean_from_npz,
    read_protein_plddt_from_cif,
)
from .structure_utils import compute_distance_binding_scores


# ---------- prepare_input helpers -----------------------------------------


_NAME_SAFE_RE = re.compile(r"[^a-zA-Z0-9_]")


def _sanitize_target_name(sample_id: str) -> str:
    """Make a sample_id safe for FASTA filenames and shell args.

    RF2NA's launcher script splits args on whitespace and uses bare
    filenames in PDB headers, so non-alphanumerics get squashed to ``_``.
    """
    cleaned = _NAME_SAFE_RE.sub("_", sample_id).strip("_")
    return cleaned or "sample"


def write_fasta(path: Path, target: str, sequence: str, *, line_width: int = 60) -> Path:
    """Write a one-record FASTA. Same shape as equipnas_adapter's helper.

    Kept as a self-contained copy here so the RF2NA module doesn't take
    a dependency on EquiPNAS adapter code (the two tools are unrelated;
    accidental refactoring of one shouldn't ripple to the other).
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


def make_single_seq_a3m(
    path: Path, sequence: str, *, name: str = "query",
) -> Path:
    """Write a query-only MSA (a3m / afa) — the single-sequence mode.

    RF2NA's ``run_RF2NA.sh`` normally spends ~30-40 min running
    hhblits (protein) + nhmmer (RNA) to build deep MSAs before the
    network ever runs; that stage is what blew the per-sample budget
    and got the tool disabled. A query-only MSA (just ``>query`` +
    the raw sequence) is a valid a3m/afa: the network treats it as a
    depth-1 alignment and skips all search. Same trick Chai-1 uses
    (single-sequence inference via ESM embeddings).

    The a3m and afa formats are both ">name\\nSEQ" for a single
    record — RF2NA's parsers read them identically — so one helper
    covers protein (.a3m) and RNA (.afa).
    """
    if not sequence:
        raise ValueError("sequence is empty")
    if any(c in sequence for c in ("\n", "\r", " ", "\t")):
        raise ValueError("sequence contains whitespace")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f">{name}\n{sequence}\n", encoding="utf-8")
    return path


# ---------- run_tool helpers ----------------------------------------------


# Match `conda activate RF2NA` and `source activate RF2NA` (with the env
# name as a whole word, so RF2NA2 / RF2NA-cpu etc. are NOT touched).
_RF2NA_ACTIVATE_RE = re.compile(
    r"^(\s*)(?:conda|source)\s+activate\s+RF2NA(\s|$)",
    re.MULTILINE,
)

# Patterns used to comment out BFD references in
# ``input_prep/make_protein_msa.sh`` when the BFD database (~272 GB)
# isn't available on the server. Without these the upstream script's
# ``set -e`` aborts at the BFD hhblits stage. UniRef30-only MSAs are
# usable for proteins that have UniRef hits — coverage is just lower.
_BFD_DB_ASSIGN_RE = re.compile(r"^(DB_BFD=.*)$", re.MULTILINE)
_BFD_HHBLITS_ASSIGN_RE = re.compile(r"^(HHBLITS_BFD=.*)$", re.MULTILINE)
_BFD_HHBLITS_USE_RE = re.compile(r"^(.*\$HHBLITS_BFD.*)$", re.MULTILINE)
_BFD_PATCH_TAG = "  # patched by riboseer: BFD not available"


# Match `export PIPEDIR=`dirname $SCRIPT`` (and the `$()` variant).
# Upstream's launcher derives PIPEDIR from the script's own location:
#   ``export PIPEDIR=`dirname $SCRIPT` ``  where SCRIPT=`realpath -s $0`.
# Once we relocate the patched copy into work_dir, that derivation
# resolves to work_dir, and the script can't find its bundled helpers
# (input_prep/, network/, pdb100_*/ etc., which live in install_dir).
# We append a SECOND `export PIPEDIR=...` override line after the
# upstream assignment so the install_dir wins. Anchored on the
# ``$SCRIPT`` substring so we only match the buggy derivation —
# unrelated future PIPEDIR= lines (e.g. a hardcoded one) pass through.
_PIPEDIR_DERIVED_RE = re.compile(
    r"^(\s*(?:export\s+)?PIPEDIR=[^\n]*\$SCRIPT[^\n]*)$",
    re.MULTILINE,
)


def patch_run_script(
    install_dir: Path,
    work_dir: Path,
    target_env: str,
    *,
    script_name: str = "run_RF2NA.sh",
) -> Path:
    """Copy ``run_RF2NA.sh`` into ``work_dir`` and rewrite the hardcoded
    ``conda activate RF2NA`` line to ``conda activate <target_env>``.

    Why: upstream's launcher activates a fixed env name internally
    (regardless of which env we used for ``conda run -n ...``). On
    sites where the env was renamed (e.g. ``RF2NA2``), the inner
    ``conda activate RF2NA`` fails with EnvironmentNameNotFound. The
    cleanest fix that doesn't fork upstream is to copy + sed in place
    on each invocation.

    The patched script is written into ``work_dir`` (not next to the
    upstream source) so:
      - we never write to the install dir (read-only on shared servers)
      - re-runs always pick up a fresh patch from the current config
      - concurrent runs don't fight over the same patched file

    cwd at execution time is still ``install_dir``, but the script's
    OWN ``$0`` resolves to the patched copy under work_dir, which
    breaks upstream's ``PIPEDIR=`dirname $SCRIPT``` derivation. We
    append an explicit ``PIPEDIR=<install_dir>`` override line right
    after the upstream assignment so the install_dir always wins (see
    ``_PIPEDIR_DERIVED_RE`` for the regex / rationale).

    Returns the patched script path (executable bit set).
    """
    src = Path(install_dir) / script_name
    if not src.is_file():
        raise FileNotFoundError(
            f"RF2NA launcher not found: {src} "
            f"(check tools.rosettafold2na.install_dir)"
        )

    dst = Path(work_dir) / f"{script_name.removesuffix('.sh')}_patched.sh"
    content = src.read_text(encoding="utf-8")

    # Patch 1: `conda|source activate RF2NA` → `conda activate <target_env>`.
    # We always emit `conda activate` regardless of the source form to
    # normalise the script — `source activate` is deprecated in newer
    # conda and mixing styles complicates debugging.
    patched = _RF2NA_ACTIVATE_RE.sub(
        rf"\1conda activate {target_env}\2",
        content,
    )

    # Patch 2: append `export PIPEDIR=<install_dir>` after the upstream
    # `PIPEDIR=`dirname $SCRIPT`` line so $PIPEDIR points to install_dir
    # regardless of where the patched script itself lives. Without this
    # the upstream derivation resolves to work_dir (where our copy sits)
    # and the script can't find its bundled helpers.
    install_dir_str = str(Path(install_dir).resolve())

    def _append_pipedir(m: re.Match) -> str:
        return (
            f"{m.group(1)}\n"
            f'export PIPEDIR="{install_dir_str}"  '
            f"# patched by riboseer: pin to install_dir"
        )

    patched, n_pipedir = _PIPEDIR_DERIVED_RE.subn(
        _append_pipedir, patched, count=1,
    )
    if n_pipedir == 0:
        # Upstream layout changed; fall back to injecting the override
        # near the top so subsequent uses of $PIPEDIR see install_dir.
        # Less precise (the upstream's own assignment may overwrite us
        # later if its style differs from what we matched) but safer
        # than silently leaving $PIPEDIR pointing at work_dir.
        patched = (
            f'# riboseer patch: PIPEDIR override (upstream assignment'
            f' style not recognised)\n'
            f'export PIPEDIR="{install_dir_str}"\n'
            + patched
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(patched, encoding="utf-8")
    # +x for the user; copy mode bits aren't preserved by write_text.
    try:
        dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError:
        # Windows / restrictive FS — `bash <path>` works without +x anyway.
        pass
    return dst


# ---------- BFD patching --------------------------------------------------


# Path of make_protein_msa.sh relative to install_dir. Centralised so a
# fork that relocates the helper can override via config.
_DEFAULT_MSA_SCRIPT = "input_prep/make_protein_msa.sh"

# Suffix appended to the backup of the unmodified upstream file. We
# always re-read from this when patching so the operation is idempotent
# across re-runs and ``skip_bfd: false`` (re-enable BFD) restores the
# original verbatim.
_BACKUP_SUFFIX = ".riboseer_original"


def patch_msa_script(
    install_dir: Path,
    *,
    skip_bfd: bool = True,
    msa_subpath: str = _DEFAULT_MSA_SCRIPT,
) -> Optional[Path]:
    """Patch ``<install_dir>/<msa_subpath>`` in place to comment out
    BFD-related lines when BFD isn't available on the server.

    Why patch in-place (and not copy to work_dir like ``patch_run_script``):
    the parent ``run_RF2NA.sh`` invokes this script via ``$PIPEDIR``,
    which we've already pinned to ``install_dir``. Putting a patched
    copy in work_dir wouldn't be picked up. To stay reversible, we keep
    the unmodified upstream content at ``<file>.sh.riboseer_original``
    and re-read from that on every call.

    Behaviour
    ---------
    - First call seeds the backup from the current file content.
    - Each subsequent call rewrites the active file from the backup,
      then optionally applies the BFD comment-out.
    - ``skip_bfd=False`` restores the active file to backup verbatim
      (used when the user has downloaded BFD and wants the upstream
      flow back).

    Returns the patched file path on success, or ``None`` when the
    target file doesn't exist (e.g. an RF2NA fork without that helper).

    Raises ``PermissionError`` (via shutil/Path) if ``install_dir`` is
    not writable. Catch that at the caller and surface a clean error.
    """
    msa_path = Path(install_dir) / msa_subpath
    if not msa_path.is_file():
        return None
    backup = msa_path.with_name(msa_path.name + _BACKUP_SUFFIX)

    # Seed backup once (atomically: copy2 is best-effort; if interrupted
    # the worst case is we re-create from the as-found file next call).
    if not backup.is_file():
        shutil.copy2(msa_path, backup)

    # Always re-read from the backup so re-running with the same flags
    # produces the same patched output regardless of in-flight state.
    original = backup.read_text(encoding="utf-8")

    if not skip_bfd:
        # Restore upstream unmodified.
        msa_path.write_text(original, encoding="utf-8")
        return msa_path

    # Apply three line-based regexes. Order matters slightly: comment
    # out the assignments first (so the `$HHBLITS_BFD` use matcher
    # doesn't grab the assign line itself, which already starts with
    # `HHBLITS_BFD=` and won't contain a literal `$HHBLITS_BFD`). The
    # `^...$` anchors with MULTILINE keep matches per-line.
    patched = _BFD_DB_ASSIGN_RE.sub(rf"# \1{_BFD_PATCH_TAG}", original)
    patched = _BFD_HHBLITS_ASSIGN_RE.sub(rf"# \1{_BFD_PATCH_TAG}", patched)
    patched = _BFD_HHBLITS_USE_RE.sub(rf"# \1{_BFD_PATCH_TAG}", patched)

    msa_path.write_text(patched, encoding="utf-8")
    return msa_path


# ---------- parse_output helpers ------------------------------------------


def _find_first(output_dir: Path, pattern: str) -> Optional[Path]:
    """First match for ``pattern`` under ``output_dir`` (rglob)."""
    matches = sorted(output_dir.rglob(pattern))
    return matches[0] if matches else None


# ---------- adapter --------------------------------------------------------


class RF2NAAdapter(BaseAdapter):
    """Adapter for RoseTTAFold2NA (Cat A)."""

    tool_id = "rosettafold2na"
    category = "A"

    # RF2NA's run_RF2NA.sh consumes args in order:
    #   P:<file>  → labelled chain "A" in the output PDB
    #   R:<file>  → labelled chain "B"
    # We pass them in this order so the chain naming is deterministic.
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

        target = _sanitize_target_name(sample_id)
        protein_fa = work_dir / f"{target}_protein.fa"
        rna_fa = work_dir / f"{target}_rna.fa"
        write_fasta(protein_fa, f"{target}_protein", protein_seq)
        write_fasta(rna_fa, f"{target}_rna", rna_seq)

        out = {
            "protein_fa": protein_fa,
            "rna_fa": rna_fa,
            "target": target,
        }

        # Single-sequence mode (default): also emit query-only MSAs so
        # run_tool can call the network directly and skip the MSA
        # search entirely. ``single_seq: false`` in config falls back
        # to the legacy run_RF2NA.sh + MSA-search path.
        tool_cfg = (config.get("tools") or {}).get("rosettafold2na") or {}
        if bool(tool_cfg.get("single_seq", True)):
            protein_a3m = work_dir / f"{target}_protein.a3m"
            rna_afa = work_dir / f"{target}_rna.afa"
            make_single_seq_a3m(protein_a3m, protein_seq,
                                name=f"{target}_protein")
            make_single_seq_a3m(rna_afa, rna_seq, name=f"{target}_rna")
            out["protein_a3m"] = protein_a3m
            out["rna_afa"] = rna_afa

        return out

    # ------------------------------------------------------------------ run

    def run_tool(
        self,
        input_paths: dict,
        work_dir: Path,
        config: dict,
    ) -> Path:
        tool_cfg = (config.get("tools") or {}).get("rosettafold2na") or {}
        install_dir = tool_cfg.get("install_dir")
        if not install_dir:
            raise ValueError(
                "config['tools']['rosettafold2na']['install_dir'] not set"
            )
        env_name = tool_cfg.get("conda_env", "RF2NA2")
        timeout = int(tool_cfg.get("timeout", 14400))

        protein_fa = Path(input_paths["protein_fa"]).resolve()
        rna_fa = Path(input_paths["rna_fa"]).resolve()
        if not protein_fa.is_file():
            raise FileNotFoundError(f"protein FASTA missing: {protein_fa}")
        if not rna_fa.is_file():
            raise FileNotFoundError(f"RNA FASTA missing: {rna_fa}")

        output_dir = (work_dir / "rf2na_output").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        log_dir_cfg = config.get("log_dir")
        log_dir = Path(log_dir_cfg) if log_dir_cfg else None
        log_tag = f"rf2na_{input_paths.get('target', 'run')}"

        # --- single-sequence mode (default) ------------------------------
        # Skip run_RF2NA.sh (which always runs hhblits/nhmmer) and the
        # MSA-script BFD patching; call the network's predict.py
        # directly with the query-only a3m/afa built in prepare_input.
        if bool(tool_cfg.get("single_seq", True)):
            return self._run_single_seq(
                input_paths=input_paths,
                tool_cfg=tool_cfg,
                install_dir=Path(install_dir),
                env_name=env_name,
                timeout=timeout,
                output_dir=output_dir,
                log_dir=log_dir,
                log_tag=log_tag,
            )

        # Patch run_RF2NA.sh's hardcoded `conda activate RF2NA` and
        # PIPEDIR derivation. See patch_run_script docstring for the why.
        if bool(tool_cfg.get("skip_script_patch", False)):
            launcher = "run_RF2NA.sh"  # raw upstream — caller asserts it's correct
        else:
            patched = patch_run_script(
                install_dir=Path(install_dir),
                work_dir=Path(work_dir).resolve(),
                target_env=env_name,
            )
            launcher = str(patched)

        # Patch input_prep/make_protein_msa.sh in-place to comment out
        # BFD references when BFD database isn't on disk. Default ON
        # since BFD (~272 GB) is rarely deployed; user flips to false
        # explicitly after staging the database.
        if not bool(tool_cfg.get("skip_script_patch", False)):
            try:
                patch_msa_script(
                    install_dir=Path(install_dir),
                    skip_bfd=bool(tool_cfg.get("skip_bfd", True)),
                )
            except PermissionError as e:
                # install_dir read-only: surface a clear error rather
                # than crash mid-run with a confused stack trace.
                raise RuntimeError(
                    f"Cannot patch make_protein_msa.sh: install_dir "
                    f"{install_dir} is not writable ({e}). Either grant "
                    f"write access (the patch leaves a "
                    f"`.riboseer_original` backup so it's reversible), "
                    f"set `tools.rosettafold2na.skip_script_patch: true` "
                    f"to disable both run-script and MSA-script patches, "
                    f"or pre-patch make_protein_msa.sh manually and use "
                    f"`skip_script_patch: true`."
                ) from e

        # run_RF2NA.sh signature (from upstream README):
        #   bash run_RF2NA.sh <out_dir> P:<protein.fa> R:<rna.fa>
        cmd = (
            f"bash {launcher} {output_dir} "
            f"P:{protein_fa} R:{rna_fa}"
        )

        result = run_in_conda_env(
            env_name, cmd,
            cwd=install_dir,
            timeout=timeout,
            log_dir=log_dir,
            log_tag=log_tag,
        )
        if not result.success:
            raise RuntimeError(
                f"RoseTTAFold2NA failed (rc={result.returncode}, "
                f"timed_out={result.timed_out}). "
                f"See log: {result.log_path}. "
                f"stderr: {result.stderr[:500]}"
            )
        return output_dir

    # ------------------------------------------------------------------ run (single-seq)

    def _run_single_seq(
        self,
        *,
        input_paths: dict,
        tool_cfg: dict,
        install_dir: Path,
        env_name: str,
        timeout: int,
        output_dir: Path,
        log_dir: Optional[Path],
        log_tag: str,
    ) -> Path:
        """Call RF2NA's network ``predict.py`` directly on the
        query-only a3m/afa — no MSA search, no run_RF2NA.sh.

        Upstream ``run_RF2NA.sh`` is just: build MSAs → call
        ``network/predict.py -inputs P:<a3m> R:<afa> -prefix <p>
        -model <weights>``. We skip the first half and run the second
        half verbatim, feeding the depth-1 alignments. ``predict.py``
        writes ``<prefix>.pdb`` (+ ``.npz`` when it has the metrics);
        parse_output globs for either that or the legacy
        ``model_00.pdb`` so the two run paths share one parser.

        Paths (``predict_script`` / ``model_weights`` / ``templates_db``)
        are config-overridable and resolved relative to ``install_dir``
        when not absolute. ``templates_db`` is optional — omitted from
        the command when unset (single-seq runs typically skip
        template search too).
        """
        a3m = input_paths.get("protein_a3m")
        afa = input_paths.get("rna_afa")
        if not a3m or not afa:
            raise RuntimeError(
                "single_seq mode but prepare_input did not emit "
                "protein_a3m / rna_afa — check tools.rosettafold2na."
                "single_seq is true at prepare time too."
            )
        a3m = Path(a3m).resolve()
        afa = Path(afa).resolve()
        if not a3m.is_file():
            raise FileNotFoundError(f"protein a3m missing: {a3m}")
        if not afa.is_file():
            raise FileNotFoundError(f"RNA afa missing: {afa}")

        def _resolve(rel: str) -> Path:
            p = Path(rel)
            return p if p.is_absolute() else (install_dir / p)

        predict_script = _resolve(
            tool_cfg.get("predict_script", "network/predict.py"))
        model_weights = _resolve(
            tool_cfg.get("model_weights", "weights/RF2NA_apr23.pt"))
        prefix = output_dir / "model"

        parts = [
            "python", str(predict_script),
            "-inputs", f"P:{a3m}", f"R:{afa}",
            "-prefix", str(prefix),
            "-model", str(model_weights),
        ]
        templates_db = tool_cfg.get("templates_db")
        if templates_db:
            parts += ["-db", str(_resolve(templates_db))]
        extra = tool_cfg.get("predict_extra_args")
        if extra:
            parts += list(extra) if isinstance(extra, (list, tuple)) \
                else str(extra).split()
        cmd = " ".join(parts)

        result = run_in_conda_env(
            env_name, cmd,
            cwd=str(install_dir),
            timeout=timeout,
            log_dir=log_dir,
            log_tag=log_tag,
        )
        if not result.success:
            raise RuntimeError(
                f"RoseTTAFold2NA (single-seq) failed "
                f"(rc={result.returncode}, "
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

        # Two run paths emit different names:
        #   - legacy run_RF2NA.sh → <output_dir>/models/model_00.pdb
        #   - single-seq predict.py -prefix <output_dir>/model
        #                             → <output_dir>/model.pdb
        # Probe both, then any *.pdb as a last resort, so the parser is
        # agnostic to which path produced the structure.
        pdb_path = (_find_first(output_dir, "model_00.pdb")
                    or _find_first(output_dir, "model.pdb")
                    or _find_first(output_dir, "*.pdb"))
        if pdb_path is None:
            return self.fail(
                sample_id,
                f"no predicted .pdb (model_00.pdb / model.pdb / *.pdb) "
                f"found under {output_dir}",
                raw_output_dir=str(output_dir),
            )

        # Heavy-atom protein-RNA contacts.
        contacts = extract_contacts(
            pdb_path,
            cutoff=cutoff,
            protein_chain_id=self.PROTEIN_CHAIN,
            rna_chain_id=self.RNA_CHAIN,
        )

        # Per-residue pLDDT (CA B-factor on the protein chain).
        # gemmi.read_structure handles both PDB and mmCIF — the helper
        # in boltz2_adapter is generic, despite its name.
        per_res_plddt = read_protein_plddt_from_cif(
            pdb_path, chain_id=self.PROTEIN_CHAIN,
        )

        plddt_mean: Optional[float]
        if per_res_plddt:
            plddt_mean = round(
                sum(per_res_plddt.values()) / len(per_res_plddt), 3,
            )
        else:
            plddt_mean = None

        # PAE from the npz, if RF2NA wrote one (legacy: model_00.npz;
        # single-seq predict.py: model.npz).
        npz_path = (_find_first(output_dir, "model_00.npz")
                    or _find_first(output_dir, "model.npz"))
        pae_mean = read_pae_mean_from_npz(npz_path) if npz_path else None

        # Distance-based per-residue binding score — same mechanism as
        # Boltz-2 / Chai-1 (CA → nearest RNA heavy atom, mapped through
        # 1/(1 + d/distance_scale)) so the [0,1] scores land on the
        # same axis in evaluate.py / the enriched fusion. RF2NA is a
        # Cat A structure predictor, so this is the field the fusion's
        # MAIN_SCORE_FIELD reads. Best-effort: a malformed PDB or a
        # gemmi hiccup must not sink an otherwise-good prediction.
        per_res_binding: dict[int, float] = {}
        tool_cfg = (config.get("tools") or {}).get("rosettafold2na") or {}
        if bool(tool_cfg.get("compute_distance_scores", True)):
            distance_scale = float(tool_cfg.get("distance_scale", 8.0))
            try:
                per_res_binding = compute_distance_binding_scores(
                    pdb_path,
                    protein_chain_id=self.PROTEIN_CHAIN,
                    rna_chain_id=self.RNA_CHAIN,
                    distance_scale=distance_scale,
                )
            except Exception:
                per_res_binding = {}

        # RF2NA does not emit ipTM in its standard output — leave None.
        # Schema clamps pLDDT to [0, 100] for safety against float quirks.
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
