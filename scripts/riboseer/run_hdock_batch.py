#!/usr/bin/env python3
"""Batch-run HDOCK (HDOCKlite) RNA-protein docking over the RiboSeer test split.

HDOCK is a Category-D rigid-body docking tool: given a receptor PDB (protein)
and a ligand PDB (RNA) it produces a sampled set of docked complexes ranked by
its shape+statistical score. From Step 4's point of view it is structurally
identical to HADDOCK 3 — the parser later reads the top-ranked complex and
derives ``binding_protein_residues`` (heavy-atom contacts) plus a per-residue
CA→RNA distance binding probability. See ``hdock_parse.py``.

The verified server invocation (HDOCKlite, CPU-only, ~5 min/sample) is::

    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    cd <workdir>
    <hdock-dir>/hdock receptor.pdb ligand.pdb        # -> Hdock.out  (cwd only)
    <hdock-dir>/createpl Hdock.out model.pdb \
        -nmax 10 -complex -models                    # -> model_1.pdb .. model_10.pdb

Two HDOCKlite quirks drive the design:
  * ``hdock`` / ``createpl`` ALWAYS write to the current working directory and
    ignore any output-path argument, so each sample is run inside its own
    ``<output-dir>/<sample_id>/`` directory that doubles as the result dir.
  * the input-PDB path is read into a fixed-length Fortran buffer, so the file
    name passed on the command line must be SHORT. We therefore write the
    receptor / ligand into the sample work dir as ``receptor.pdb`` / ``ligand.pdb``
    and invoke the binaries with those bare relative names (``cwd`` is the work
    dir) — the long absolute path never reaches the Fortran string.

Input preparation per sample
----------------------------
  receptor (protein) : reused from ``--protein-inputs-dir/<sample_id>.pdb``
      (the single protein chain, renamed to ``A`` and renumbered to 1-based
      label_seq — the same PDB GraphBind / NucleicNet consume). If that file is
      absent we fall back to extracting it fresh from the raw structure via the
      shared ``extract_protein_chain_pdb`` so the batch is self-sufficient.
  ligand (RNA)       : extracted from the raw complex
      (``sample.source_pdb`` under ``--raw-dir``, chain ``sample.rna.chain_id``)
      and written as chain ``B``, renumbered to 1-based label_seq. Forcing the
      ligand to ``B`` keeps it distinct from the receptor's ``A`` in the docked
      complex regardless of the source chain id, so the parser's A/B chain
      filter is unambiguous.

Both inputs are 1-based label_seq, so the indices the parser emits line up with
step-1 ``binding_protein_residues`` / ``binding_rna_nucleotides`` directly — no
remap, exactly like the HADDOCK 3 adapter.

Usage (server, inside the ``pocket`` conda env so libfftw3 is on the lib path)
-----------------------------------------------------------------------------
::

    python scripts/riboseer/run_hdock_batch.py \
        --processed-dir       data/processed_quality \
        --sample-list         data/processed_quality/splits_tmscore_035/test.txt \
        --raw-dir             data/raw \
        --protein-inputs-dir  data/batch_test_v7/graphbind_inputs \
        --hdock-dir           /opt/biotools/HDOCKlite \
        --output-dir          data/batch_test_v7/hdock_outputs \
        --timeout             600

Shard across runs with ``--start/--end`` (and ``--limit``); ``--resume`` skips
samples whose ``model_1.pdb`` already exists. ``--conda-prefix`` overrides the
``$CONDA_PREFIX`` used to locate ``libfftw3`` (``<prefix>/lib``).

See also
--------
src/step4_tool_adapters/external/hdock_parse.py : turns ``model_1.pdb`` into the step-4
    ``<sample_id>.jsonl`` (tool_id="hdock", category="D").
src/step4_tool_adapters/adapters/haddock3_adapter.py : the Category-D template
    this batch mirrors (same 1-based-label_seq input convention, no remap).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import gemmi  # type: ignore
except ImportError as _e:  # pragma: no cover - environment dependent
    gemmi = None  # type: ignore
    _GEMMI_IMPORT_ERROR = _e
else:
    _GEMMI_IMPORT_ERROR = None

# Make repo root + src importable for the shared helpers (mirrors the other
# riboseer scripts).
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Reuse the exact sample-list / sample-json / raw-structure / protein-chain
# helpers the NucleicNet batch already defines — keeping a single source of
# truth for id cleaning, case-insensitive lookup and the label_seq renumbering.
from step4_tool_adapters.tool_io import (  # noqa: E402
    clean_sample_id,
    extract_protein_chain_pdb,
    find_raw_structure,
    load_sample_json,
    read_sample_list,
)
from step4_tool_adapters.adapters.structure_utils import (  # noqa: E402
    _is_rna_residue,
)

logger = logging.getLogger("run_hdock_batch")

DEFAULT_TIMEOUT = 600       # seconds for the hdock docking step
DEFAULT_CREATEPL_TIMEOUT = 300  # seconds for the (fast) createpl model build
LIGAND_CHAIN = "B"          # RNA ligand chain in the docked complex


# --------------------------------------------------------------------------
# RNA ligand preparation
# --------------------------------------------------------------------------
def extract_rna_ligand_pdb(raw_path: Path, chain_id: str, out_path: Path) -> int:
    """Write the RNA chain as a single-chain PDB named ``B``, renumbered to
    1-based polymer index (gemmi label_seq). Returns the nucleotide count.

    Mirrors ``structure_utils.extract_rna_chain_pdb`` but always forces the
    output chain to ``B`` (rather than passing single-char source ids through):
    the HDOCK receptor is chain ``A``, so a fixed ``B`` guarantees the ligand
    never collides with the receptor chain in the docked complex regardless of
    the source RNA chain id. Reuses ``_is_rna_residue`` for the canonical /
    modified-base classification so waters, ions and ligands are dropped exactly
    as the rest of the pipeline drops them.

    Raises ``ValueError`` if the chain is absent or has no RNA residue with a
    polymer index.
    """
    if gemmi is None:
        raise RuntimeError(
            f"gemmi is required for RNA chain extraction but is not importable: "
            f"{_GEMMI_IMPORT_ERROR}"
        )
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

    new_structure = gemmi.Structure()
    new_structure.cell = structure.cell
    try:
        new_structure.spacegroup_hm = structure.spacegroup_hm
    except Exception:  # noqa: BLE001
        pass
    new_model = gemmi.Model("1")
    new_chain = gemmi.Chain(LIGAND_CHAIN)

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
            f"chain {chain_id!r} in {raw_path} has 0 RNA residues with a "
            f"polymer index"
        )

    new_model.add_chain(new_chain)
    new_structure.add_model(new_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_structure.write_pdb(str(out_path))
    return kept


def prepare_receptor_pdb(
    sample_id: str,
    sample: dict,
    *,
    protein_inputs_dir: Optional[Path],
    raw_path: Path,
    dest: Path,
) -> int:
    """Write the receptor (protein) PDB into ``dest`` as chain ``A``.

    Prefers the pre-built single-chain input at
    ``<protein_inputs_dir>/<sample_id>.pdb`` (the GraphBind input: chain ``A``,
    1-based label_seq). Falls back to extracting the chain fresh from the raw
    structure via the shared ``extract_protein_chain_pdb`` (identical
    convention) so the batch still runs when a pre-built input is missing.
    Returns the residue count written (best-effort: 0 when reused from disk).
    """
    if protein_inputs_dir is not None:
        src = protein_inputs_dir / f"{sample_id}.pdb"
        if not src.is_file():  # case-insensitive fallback (host may lower-case)
            target = f"{sample_id}.pdb".lower()
            if protein_inputs_dir.is_dir():
                for child in protein_inputs_dir.iterdir():
                    if child.is_file() and child.name.lower() == target:
                        src = child
                        break
        if src.is_file():
            dest.write_bytes(src.read_bytes())
            return 0

    chain_id = (sample.get("protein") or {}).get("chain_id")
    if not chain_id:
        raise ValueError(f"sample {sample_id!r}: missing protein.chain_id")
    return extract_protein_chain_pdb(raw_path, chain_id, dest)


# --------------------------------------------------------------------------
# HDOCK driving
# --------------------------------------------------------------------------
def build_env(conda_prefix: Optional[str]) -> dict:
    """Return a subprocess env with ``<conda_prefix>/lib`` (libfftw3) prepended
    to ``LD_LIBRARY_PATH``. Falls back to ``$CONDA_PREFIX`` when not given."""
    env = os.environ.copy()
    prefix = conda_prefix or env.get("CONDA_PREFIX")
    if prefix:
        lib = str(Path(prefix) / "lib")
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib}:{existing}" if existing else lib
    return env


def _run(cmd: list[str], *, cwd: Path, env: dict, timeout: Optional[int],
         log_path: Path) -> None:
    """Run one HDOCK binary in ``cwd``, tee-ing stdout+stderr to ``log_path``.
    Raises ``subprocess.TimeoutExpired`` on timeout (the child is killed) or
    ``RuntimeError`` on a non-zero exit."""
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ (cwd={cwd}) {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(
            cmd, cwd=str(cwd), env=env,
            stdout=log, stderr=subprocess.STDOUT,
            timeout=timeout if timeout and timeout > 0 else None,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{Path(cmd[0]).name} exited rc={proc.returncode} "
            f"(see {log_path})"
        )


def run_hdock(
    sample_dir: Path,
    *,
    hdock_bin: Path,
    createpl_bin: Path,
    env: dict,
    nmodels: int,
    dock_timeout: Optional[int],
    createpl_timeout: Optional[int],
) -> Path:
    """Run ``hdock`` then ``createpl`` inside ``sample_dir`` (which already holds
    ``receptor.pdb`` / ``ligand.pdb``). Returns the best model ``model_1.pdb``.

    Both binaries are invoked with bare relative filenames so the Fortran-bound
    input-path buffer never sees the long absolute work-dir path.
    """
    log_path = sample_dir / "hdock.log"
    # 1. docking -> Hdock.out (in cwd)
    _run([str(hdock_bin), "receptor.pdb", "ligand.pdb"],
         cwd=sample_dir, env=env, timeout=dock_timeout, log_path=log_path)
    hdock_out = sample_dir / "Hdock.out"
    if not hdock_out.is_file():
        raise RuntimeError(
            f"hdock returned rc=0 but {hdock_out} is missing (see {log_path})"
        )
    # 2. build top-N complex models -> model_1.pdb .. model_<n>.pdb (in cwd)
    _run([str(createpl_bin), "Hdock.out", "model.pdb",
          "-nmax", str(nmodels), "-complex", "-models"],
         cwd=sample_dir, env=env, timeout=createpl_timeout, log_path=log_path)
    best = sample_dir / "model_1.pdb"
    if not best.is_file():
        raise RuntimeError(
            f"createpl returned rc=0 but {best} is missing (see {log_path})"
        )
    return best


def run_one_sample(
    sample_id: str,
    *,
    samples_dir: Path,
    raw_dir: Path,
    protein_inputs_dir: Optional[Path],
    output_dir: Path,
    hdock_bin: Path,
    createpl_bin: Path,
    env: dict,
    nmodels: int,
    dock_timeout: Optional[int],
    createpl_timeout: Optional[int],
    resume: bool,
) -> dict:
    """Prepare inputs + dock one sample. Never raises — returns a status dict so
    the batch loop keeps going and tallies failures."""
    sample_dir = output_dir / sample_id
    result = {"sample_id": sample_id, "status": "ok", "error": None,
              "timed_out": False}

    if resume and (sample_dir / "model_1.pdb").is_file():
        result["status"] = "skipped"
        return result

    t0 = time.time()
    try:
        sample = load_sample_json(samples_dir, sample_id)
        source_pdb = sample["source_pdb"]
        rna_chain = (sample.get("rna") or {}).get("chain_id")
        if not rna_chain:
            raise ValueError(f"sample {sample_id!r}: missing rna.chain_id")

        raw_path = find_raw_structure(raw_dir, source_pdb)
        if raw_path is None:
            raise FileNotFoundError(
                f"no raw structure for {source_pdb!r} under {raw_dir}"
            )

        sample_dir.mkdir(parents=True, exist_ok=True)
        receptor = sample_dir / "receptor.pdb"
        ligand = sample_dir / "ligand.pdb"
        n_prot = prepare_receptor_pdb(
            sample_id, sample,
            protein_inputs_dir=protein_inputs_dir,
            raw_path=raw_path, dest=receptor,
        )
        n_rna = extract_rna_ligand_pdb(raw_path, rna_chain, ligand)
        result["n_protein_residues"] = n_prot
        result["n_rna_nucleotides"] = n_rna

        best = run_hdock(
            sample_dir,
            hdock_bin=hdock_bin, createpl_bin=createpl_bin, env=env,
            nmodels=nmodels,
            dock_timeout=dock_timeout, createpl_timeout=createpl_timeout,
        )
        result["best_model"] = str(best)
    except subprocess.TimeoutExpired as e:
        result["status"] = "failed"
        result["timed_out"] = True
        result["error"] = f"TimeoutExpired: {e}"
        logger.warning("sample %s TIMED OUT; skipping", sample_id)
    except Exception as e:  # noqa: BLE001 - one bad sample must not kill batch
        result["status"] = "failed"
        result["error"] = f"{type(e).__name__}: {e}"
        logger.exception("sample %s failed", sample_id)
    result["runtime_seconds"] = round(time.time() - t0, 1)
    return result


def write_failures(failures: list[dict], path: Path) -> None:
    """Append failed-sample records to a JSONL log."""
    if not failures:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in failures:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--processed-dir", required=True, type=Path,
                   help="step-1 processed dir (its samples/ holds the JSONs)")
    p.add_argument("--sample-list", required=True, type=Path,
                   help="txt (one id/line) or csv (sample_id column) of ids")
    p.add_argument("--raw-dir", required=True, type=Path,
                   help="dir with raw <source_pdb>.{pdb,cif}[.gz] structures")
    p.add_argument("--hdock-dir", required=True, type=Path,
                   help="HDOCKlite dir holding the hdock + createpl binaries")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="output base; one <sample_id>/ work+result dir per sample")
    p.add_argument("--protein-inputs-dir", type=Path, default=None,
                   help="dir with pre-built single-chain protein PDBs "
                        "(<sample_id>.pdb, chain A, 1-based label_seq, e.g. "
                        "graphbind_inputs/). Missing files fall back to fresh "
                        "extraction from --raw-dir.")
    p.add_argument("--samples-dir", type=Path, default=None,
                   help="override sample-JSON dir "
                        "(default <processed-dir>/samples)")
    p.add_argument("--conda-prefix", type=str, default=None,
                   help="conda prefix whose lib/ holds libfftw3 "
                        "(default: $CONDA_PREFIX)")
    p.add_argument("--nmodels", type=int, default=10,
                   help="number of complex models createpl builds (default 10)")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help="per-sample wall-clock budget (s) for the hdock docking "
                        f"step (0 disables; default {DEFAULT_TIMEOUT})")
    p.add_argument("--createpl-timeout", type=int, default=DEFAULT_CREATEPL_TIMEOUT,
                   help="wall-clock budget (s) for the createpl model build "
                        f"(0 disables; default {DEFAULT_CREATEPL_TIMEOUT})")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--limit", type=int, default=0,
                   help="cap number processed after start/end (0 = no cap)")
    p.add_argument("--resume", action="store_true",
                   help="skip samples whose model_1.pdb already exists")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    def _abs(p: Path) -> Path:
        return p.expanduser().resolve()

    processed_dir = _abs(args.processed_dir)
    samples_dir = _abs(args.samples_dir or processed_dir / "samples")
    raw_dir = _abs(args.raw_dir)
    hdock_dir = _abs(args.hdock_dir)
    output_dir = _abs(args.output_dir)
    protein_inputs_dir = _abs(args.protein_inputs_dir) \
        if args.protein_inputs_dir is not None else None

    if not samples_dir.is_dir():
        logger.error("samples dir not found: %s", samples_dir)
        return 1
    if not raw_dir.is_dir():
        logger.error("raw dir not found: %s", raw_dir)
        return 1
    hdock_bin = hdock_dir / "hdock"
    createpl_bin = hdock_dir / "createpl"
    for b in (hdock_bin, createpl_bin):
        if not b.is_file():
            logger.error("HDOCK binary not found: %s", b)
            return 1
    if gemmi is None:
        logger.error("gemmi not importable (needed for RNA extraction): %s",
                     _GEMMI_IMPORT_ERROR)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(args.conda_prefix)
    logger.info("LD_LIBRARY_PATH=%s", env.get("LD_LIBRARY_PATH", "<unset>"))

    samples = read_sample_list(_abs(args.sample_list))
    end = args.end if args.end is not None else len(samples)
    samples = samples[args.start:end]
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        logger.error("no samples selected (check --sample-list/--start/--end)")
        return 1

    dock_timeout = args.timeout if args.timeout and args.timeout > 0 else None
    createpl_timeout = args.createpl_timeout \
        if args.createpl_timeout and args.createpl_timeout > 0 else None
    if dock_timeout:
        logger.info("per-sample dock timeout: %ss (createpl: %ss)",
                    dock_timeout, createpl_timeout)
    else:
        logger.warning("per-sample dock timeout DISABLED")

    n_ok = n_skip = n_fail = n_timeout = 0
    failures: list[dict] = []
    total = len(samples)
    for i, raw_id in enumerate(samples, 1):
        sample_id = clean_sample_id(raw_id)
        res = run_one_sample(
            sample_id,
            samples_dir=samples_dir, raw_dir=raw_dir,
            protein_inputs_dir=protein_inputs_dir, output_dir=output_dir,
            hdock_bin=hdock_bin, createpl_bin=createpl_bin, env=env,
            nmodels=args.nmodels,
            dock_timeout=dock_timeout, createpl_timeout=createpl_timeout,
            resume=args.resume,
        )
        status = res["status"]
        if status == "ok":
            n_ok += 1
            logger.info("[%d/%d] %s ok (%ss, prot=%s rna=%s)", i, total,
                        sample_id, res.get("runtime_seconds"),
                        res.get("n_protein_residues"),
                        res.get("n_rna_nucleotides"))
        elif status == "skipped":
            n_skip += 1
            logger.info("[%d/%d] %s skipped (already done)", i, total, sample_id)
        else:
            n_fail += 1
            if res.get("timed_out"):
                n_timeout += 1
            failures.append(res)
            logger.warning("[%d/%d] %s FAILED: %s", i, total, sample_id,
                           res.get("error"))

    write_failures(failures, output_dir / "_failures.jsonl")
    logger.info("done: %d ok, %d skipped, %d failed (%d timed out) of %d",
                n_ok, n_skip, n_fail, n_timeout, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
