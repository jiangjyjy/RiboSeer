#!/usr/bin/env python3
"""Batch-run NucleicNet (SXPR task) over the RiboSeer test split.

NucleicNet is a Category-C RNA-binding-site predictor: it voxelises a halo
shell (2.5-5.0 A) around the protein surface and classifies every grid point
into ``{Base / Phosphate / Ribose / Nonsite}`` (the "SXPR" head). We only run
SXPR here — the finer AUCG base-identity head is not needed for a per-residue
binding score, and skipping it roughly halves the GPU time.

This script drives the server-style API the user verified on the backup host
(``~/pocket_baseline/NucleicNet/``)::

    ServerC = Server(SaveCleansed=True, SaveDf=True,
                     Select_HeavyAtoms=True, DIR_ServerFolder=<out>)
    ServerC.SimpleSanitise(DIR_InputPdbFile=<single-chain protein pdb>)
    ServerC.MakeHalo()
    ServerC.MakeDssp()
    ServerC.MakeFeature()
    ServerC.MakeDummyTypi()
    ServerC.MakeSXPR()        # -> <out>/sxpr/Result_EnsembleAvDf.pkl

For each requested sample it:
  1. locates the raw complex for ``sample.source_pdb`` under ``--raw-dir``;
  2. extracts the protein chain (``sample.protein.chain_id``), drops waters /
     ligands / RNA, keeps standard amino acids, **renumbers to 1-based polymer
     index (gemmi label_seq)** and renames the chain to ``A`` — the same
     coordinate system P2Rank / EquiPNAS use, so the per-residue indices the
     parser later emits line up with step-1 ``binding_protein_residues``
     without any further translation. The PDB is written to
     ``<output-dir>/nucleicnet_inputs/<sample_id>.pdb``;
  3. runs the Server flow above with output under
     ``<output-dir>/nucleicnet_outputs/<sample_id>/``.

NucleicNet's own code uses **relative** paths (``../NucleicNet/util/dssp``,
``../Models/...``) and therefore must run with the current working directory
set to ``<nucleicnet-dir>/Notebooks/``. This script ``chdir``s there itself
after resolving every user-supplied path to an absolute path, so it can be
launched from anywhere as long as ``--nucleicnet-dir`` points at the repo root.

Usage (server, inside the ``Nucl`` conda env)
---------------------------------------------
::

    conda run -n Nucl python scripts/riboseer/run_nucleicnet_batch.py \
        --data-dir       ~/pocket_baseline/data \
        --nucleicnet-dir ~/pocket_baseline/NucleicNet \
        --output-dir     ~/pocket_baseline/data/batch_test_v7/nucleicnet \
        --sample-list    ~/pocket_baseline/data/batch_test_v7/af3_submission_order.csv \
        --resume

Each sample takes ~2-3 min (3-checkpoint ensemble on GPU); the full 107-sample
split is ~4-6 h. Use ``--start/--end`` to shard across runs and ``--resume`` to
skip samples whose ``sxpr/Result_EnsembleAvDf.pkl`` already exists.

See also
--------
src/step4_tool_adapters/external/nucleicnet_parse.py : turns the per-voxel SXPR pkl
    into the step-4 ``<sample_id>.jsonl`` (tool_id="nucleicnet", category="C").
src/step4_tool_adapters/adapters/p2rank_adapter.py : the label_seq renumbering
    convention this script mirrors for index alignment.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

try:
    import gemmi  # type: ignore
except ImportError as _e:  # pragma: no cover - environment dependent
    gemmi = None  # type: ignore
    _GEMMI_IMPORT_ERROR = _e
else:
    _GEMMI_IMPORT_ERROR = None

logger = logging.getLogger("run_nucleicnet_batch")

# Populated lazily by ``import_nucleicnet`` so the module imports (and its
# helpers stay unit-testable) without NucleicNet / torch / a GPU present.
Server = None  # type: ignore
MkdirList = None  # type: ignore

DEFAULT_TIMEOUT = 600  # seconds per sample (10 min)


class SampleTimeout(Exception):
    """Raised when one sample's NucleicNet flow exceeds the time budget."""


@contextmanager
def time_limit(seconds: Optional[int]):
    """Abort the wrapped block with ``SampleTimeout`` after ``seconds``.

    Implemented with ``signal.alarm`` (SIGALRM): the alarm fires in the main
    thread and the handler raises ``SampleTimeout``, unwinding out of whatever
    NucleicNet step was running so the batch loop can move on. This is the right
    fit here because the flow runs synchronously in the main thread; a
    ``threading.Timer`` could only *observe* a hang, not interrupt it.

    ``seconds`` falsy / <= 0 disables the guard. On platforms without SIGALRM
    (e.g. Windows dev boxes) it degrades to a no-op with a one-time warning —
    the production host is Linux, where the guard is real.

    Caveat: SIGALRM interrupts at the next Python bytecode / when a blocking
    call returns control to the interpreter. It reliably breaks Python-level
    spins and interruptible syscalls; a process wedged entirely inside a CUDA
    driver call may only unwind once that call yields. If such hard hangs
    persist, isolating each sample in a subprocess that can be SIGKILLed is the
    stronger remedy — but that's beyond the in-process guard requested here.
    """
    if not seconds or seconds <= 0:
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        logger.warning(
            "SIGALRM unavailable on this platform; per-sample timeout (%ss) "
            "is NOT enforced here (it is on the Linux server).", seconds)
        yield
        return

    def _handler(signum, frame):  # noqa: ARG001
        raise SampleTimeout(f"timed out after {seconds}s")

    try:
        previous = signal.signal(signal.SIGALRM, _handler)
    except ValueError:
        # signal.signal only works in the main thread; degrade to no-op.
        logger.warning("not in main thread; per-sample timeout (%ss) NOT "
                       "enforced", seconds)
        yield
        return
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)

# Minimal standard-amino-acid set (mirrors contact_extractor / p2rank_adapter).


# --------------------------------------------------------------------------
# sample / structure discovery
# --------------------------------------------------------------------------
# Invisible characters that routinely sneak into sample ids when a list is
# copy-pasted, saved with a BOM, or edited on a different OS. ``str.strip()``
# already removes ASCII/Unicode whitespace (incl. \r \n \t and \xa0), but NOT
# these zero-width / directional marks — so a "2czj_E_F" with a leading BOM
# would build a path "﻿2czj_E_F.json" that exists nowhere yet looks
# identical to the eye (and to a terminal that swallows the mark). Strip them.


from step4_tool_adapters.tool_io import (  # noqa: E402
    _is_protein_residue, clean_sample_id, extract_protein_chain_pdb,
    find_raw_structure, load_sample_json, read_sample_list, write_failures,
)

# Re-exported for callers that import them from this module.
__all__ = [
    "clean_sample_id", "read_sample_list", "load_sample_json",
    "find_raw_structure", "extract_protein_chain_pdb", "write_failures",
    "time_limit", "import_nucleicnet", "run_server_flow", "run_one_sample",
    "sxpr_result_path", "main",
]


# --------------------------------------------------------------------------
# NucleicNet driving
# --------------------------------------------------------------------------
def import_nucleicnet(nucleicnet_dir: Path) -> Path:
    """Make ``NucleicNet`` importable and ``chdir`` into ``Notebooks/``.

    NucleicNet's code references its weights / DSSP / FEATURE binaries through
    paths relative to ``Notebooks/`` (``../Models``, ``../NucleicNet/util``),
    so the cwd has to be there for the Server flow to work. Returns the
    notebooks dir. Populates the module-level ``Server`` / ``MkdirList``.
    """
    global Server, MkdirList
    nucleicnet_dir = nucleicnet_dir.expanduser().resolve()
    notebooks_dir = nucleicnet_dir / "Notebooks"
    if not notebooks_dir.is_dir():
        raise FileNotFoundError(
            f"expected Notebooks/ under --nucleicnet-dir: {notebooks_dir}"
        )
    if str(nucleicnet_dir) not in sys.path:
        sys.path.insert(0, str(nucleicnet_dir))
    os.chdir(notebooks_dir)
    from NucleicNet.DatasetBuilding.commandServer import Server as _Server
    from NucleicNet.DatasetBuilding.util import MkdirList as _MkdirList
    Server = _Server
    MkdirList = _MkdirList
    return notebooks_dir


def sxpr_result_path(sample_out_dir: Path) -> Path:
    """The per-sample SXPR ensemble pickle the parser consumes."""
    return sample_out_dir / "sxpr" / "Result_EnsembleAvDf.pkl"


def run_server_flow(pdb_path: Path, sample_out_dir: Path) -> None:
    """Run the verified SXPR-only Server sequence for one sample.

    Paths are passed absolute so they resolve regardless of the cwd that
    ``import_nucleicnet`` switched to.
    """
    if Server is None:
        raise RuntimeError("NucleicNet Server not imported; call import_nucleicnet first")
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    server = Server(
        SaveCleansed=True, SaveDf=True, Select_HeavyAtoms=True,
        DIR_ServerFolder=str(sample_out_dir),
    )
    server.SimpleSanitise(DIR_InputPdbFile=str(pdb_path))
    server.MakeHalo()
    server.MakeDssp()
    server.MakeFeature()
    server.MakeDummyTypi()
    server.MakeSXPR()  # site / nonsite / phosphate / ribose; AUCG skipped


def run_one_sample(
    sample_id: str,
    *,
    samples_dir: Path,
    raw_dir: Path,
    inputs_dir: Path,
    outputs_dir: Path,
    resume: bool,
    timeout: Optional[int] = DEFAULT_TIMEOUT,
) -> dict:
    """Prepare input + run NucleicNet for one sample. Never raises — returns a
    status dict so the batch loop can keep going and tally failures.

    The NucleicNet flow is wrapped in :func:`time_limit`; if a sample wedges
    (the GPU-memory-held / 0%-util hang seen on some inputs) it is aborted after
    ``timeout`` seconds, recorded as a failure with ``timed_out=True``, and the
    loop advances. ``timeout`` of ``None``/``0`` disables the guard.
    """
    sample_out_dir = outputs_dir / sample_id
    result = {"sample_id": sample_id, "status": "ok", "error": None,
              "timed_out": False}

    if resume and sxpr_result_path(sample_out_dir).is_file():
        result["status"] = "skipped"
        return result

    t0 = time.time()
    try:
        sample = load_sample_json(samples_dir, sample_id)
        source_pdb = sample["source_pdb"]
        chain_id = sample["protein"]["chain_id"]
        raw_path = find_raw_structure(raw_dir, source_pdb)
        if raw_path is None:
            raise FileNotFoundError(
                f"no raw structure for {source_pdb!r} under {raw_dir}"
            )
        pdb_path = (inputs_dir / f"{sample_id}.pdb").resolve()
        n_res = extract_protein_chain_pdb(raw_path, chain_id, pdb_path)
        result["n_residues"] = n_res

        # The whole SimpleSanitise -> ... -> MakeSXPR sequence shares one budget.
        with time_limit(timeout):
            run_server_flow(pdb_path, sample_out_dir.resolve())

        if not sxpr_result_path(sample_out_dir).is_file():
            raise RuntimeError(
                f"Server flow finished but {sxpr_result_path(sample_out_dir)} "
                f"is missing"
            )
    except SampleTimeout as e:
        result["status"] = "failed"
        result["timed_out"] = True
        result["error"] = f"SampleTimeout: {e}"
        logger.warning("sample %s TIMED OUT after %ss; skipping",
                       sample_id, timeout)
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
    p.add_argument("--data-dir", required=True, type=Path,
                   help="base data dir (contains processed_quality/ and raw/)")
    p.add_argument("--nucleicnet-dir", required=True, type=Path,
                   help="NucleicNet repo root (the dir holding Notebooks/)")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="base output dir; nucleicnet_inputs/ and "
                        "nucleicnet_outputs/ are created under it")
    p.add_argument("--sample-list", type=Path, default=None,
                   help="txt (one id/line) or csv (sample_id column); "
                        "default: <data-dir>/batch_test_v7/af3_submission_order.csv")
    p.add_argument("--samples-dir", type=Path, default=None,
                   help="override sample-JSON dir "
                        "(default <data-dir>/processed_quality/samples)")
    p.add_argument("--raw-dir", type=Path, default=None,
                   help="override raw-structure dir (default <data-dir>/raw)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--limit", type=int, default=0,
                   help="cap number processed after start/end (0 = no cap)")
    p.add_argument("--resume", action="store_true",
                   help="skip samples whose sxpr/Result_EnsembleAvDf.pkl exists")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help="per-sample wall-clock budget in seconds for the "
                        "NucleicNet flow; on timeout the sample is recorded as "
                        "failed and the batch moves on (0 disables; "
                        f"default {DEFAULT_TIMEOUT})")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Resolve every path to absolute BEFORE chdir-ing into Notebooks/.
    # expanduser() first: argparse/Path never expand a literal "~" themselves,
    # and a shell that didn't expand it would otherwise yield a bogus
    # "<cwd>/~/..." path that silently misses every file.
    def _abs(p: Path) -> Path:
        return p.expanduser().resolve()

    data_dir = _abs(args.data_dir)
    samples_dir = _abs(args.samples_dir or data_dir / "processed_quality" / "samples")
    raw_dir = _abs(args.raw_dir or data_dir / "raw")
    sample_list = _abs(args.sample_list
                       or data_dir / "batch_test_v7" / "af3_submission_order.csv")
    output_dir = _abs(args.output_dir)
    inputs_dir = output_dir / "nucleicnet_inputs"
    outputs_dir = output_dir / "nucleicnet_outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if not samples_dir.is_dir():
        logger.error("samples dir not found: %s", samples_dir)
        return 1
    if not raw_dir.is_dir():
        logger.error("raw dir not found: %s", raw_dir)
        return 1

    samples = read_sample_list(sample_list)
    end = args.end if args.end is not None else len(samples)
    samples = samples[args.start:end]
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        logger.error("no samples selected (check --sample-list/--start/--end)")
        return 1

    # Import NucleicNet + chdir into Notebooks/. Done once, after path resolution.
    try:
        notebooks_dir = import_nucleicnet(args.nucleicnet_dir)
        logger.info("NucleicNet imported; cwd -> %s", notebooks_dir)
    except Exception as e:  # noqa: BLE001
        logger.error("could not import NucleicNet: %s", e)
        return 1

    if args.timeout and args.timeout > 0:
        logger.info("per-sample timeout: %ss", args.timeout)
    else:
        logger.warning("per-sample timeout DISABLED (--timeout %s)", args.timeout)

    n_ok = n_skip = n_fail = n_timeout = 0
    failures: list[dict] = []
    total = len(samples)
    for i, sample_id in enumerate(samples, 1):
        res = run_one_sample(
            sample_id,
            samples_dir=samples_dir, raw_dir=raw_dir,
            inputs_dir=inputs_dir, outputs_dir=outputs_dir,
            resume=args.resume, timeout=args.timeout,
        )
        status = res["status"]
        if status == "ok":
            n_ok += 1
            logger.info("[%d/%d] %s ok (%ss, %s res)", i, total, sample_id,
                        res.get("runtime_seconds"), res.get("n_residues"))
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

    write_failures(failures, outputs_dir / "_failures.jsonl")
    logger.info("done: %d ok, %d skipped, %d failed (%d timed out) of %d",
                n_ok, n_skip, n_fail, n_timeout, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
