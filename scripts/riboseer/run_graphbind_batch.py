#!/usr/bin/env python
"""Batch-run GraphBind over the RiboSeer test split.

GraphBind is a Category-C RNA-binding-residue predictor. Its inputs are the
single-chain protein PDBs already prepared under ``graphbind_inputs/`` (chain
renamed to ``A``, renumbered to 1-based label_seq), so this script does *not*
extract chains — it just drives GraphBind's CLI as verified on the backup host
(``~/pocket_baseline/GraphBind/``, conda env ``GraphBind``)::

    cd GraphBind/scripts
    python prediction.py --querypath <out> --filename <prot.pdb> \
        --chainid A --ligands RNA --cpu 4

``--querypath`` is the per-sample working/output dir and ``--filename`` is a
*basename* GraphBind reads from inside it, so for each sample we copy the input
PDB into ``<output-dir>/<sample_id>/`` and pass its basename. The prediction
lands at ``<output-dir>/<sample_id>/RNA-binding_result.csv``. ``prediction.py``
imports sibling modules by relative path, so it runs with
``cwd=<graphbind-dir>/scripts``.

Each invocation is wrapped in a subprocess timeout; on timeout the whole process
group is killed (HHblits / child searches die too), the sample is recorded as
failed in ``failures.jsonl``, and the batch moves on. Each sample takes ~2-3 min
(HHblits search dominates).

Python compatibility
---------------------
The GraphBind conda env ships **Python 3.6**, so this script is deliberately
self-contained (no imports of the other baseline scripts) and avoids any 3.7+
syntax: no ``from __future__ import annotations``, ``typing`` aliases instead of
PEP 585/604 generics, and ``universal_newlines`` instead of ``text=`` on the
subprocess. It runs unchanged on Python 3.6 through 3.11. (The matching parser
``graphbind_parse.py`` runs in the modern ``pocket`` env — it needs
pydantic, which is 3.8+ only — so it is not subject to this constraint.)

Usage (backup server, inside the ``GraphBind`` conda env)
---------------------------------------------------------
::

    python scripts/riboseer/run_graphbind_batch.py \
        --inputs-dir    ~/pocket_baseline/data/batch_test_v7/graphbind_inputs/ \
        --graphbind-dir ~/pocket_baseline/GraphBind \
        --output-dir    ~/pocket_baseline/graphbind_outputs \
        --sample-list   ~/pocket_baseline/data/processed_quality/splits_tmscore_035/test.txt \
        --cpu 4 --timeout 600 --resume

See also
--------
src/step4_tool_adapters/external/graphbind_parse.py : turns each RNA-binding_result.csv
    into step-4 ``<sample_id>.jsonl`` (tool_id="graphbind", category="C").
"""
import argparse
import csv
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("run_graphbind_batch")

DEFAULT_TIMEOUT = 600  # seconds per sample (HHblits dominates)
DEFAULT_CPU = 4
RESULT_CSV_NAME = "RNA-binding_result.csv"

# Zero-width / directional marks that survive str.strip() and corrupt ids
# copied from a list saved with a BOM or edited cross-OS.
_INVISIBLE = "﻿​‌‍‎‏⁠"


# --------------------------------------------------------------------------
# sample-list reading (self-contained; kept 3.6-compatible)
# --------------------------------------------------------------------------
def clean_sample_id(sample_id):
    # type: (str) -> str
    """Strip surrounding whitespace and invisible/zero-width marks from an id."""
    return sample_id.strip().strip(_INVISIBLE).strip()


def read_sample_list(path):
    # type: (Path) -> List[str]
    """Read sample ids from a ``.txt`` (one id/line) or ``.csv`` (a
    ``sample_id`` column). ``#`` comments and blanks are skipped; every id is
    cleaned of invisible marks."""
    if not path.is_file():
        raise FileNotFoundError("sample-list not found: {}".format(path))
    if path.suffix.lower() == ".csv":
        out = []  # type: List[str]
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "sample_id" not in reader.fieldnames:
                raise ValueError(
                    "{} has no 'sample_id' column (found: {})".format(
                        path, reader.fieldnames))
            for row in reader:
                sid = clean_sample_id(row.get("sample_id") or "")
                if sid:
                    out.append(sid)
        return out
    out = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        s = clean_sample_id(line)
        if s and not s.startswith("#"):
            out.append(clean_sample_id(s.split()[0]))
    return out


# --------------------------------------------------------------------------
# subprocess with process-group timeout (3.6-compatible)
# --------------------------------------------------------------------------
def _kill_tree(proc):
    # type: (subprocess.Popen) -> None
    """SIGKILL the child and, on POSIX, its whole process group (so the HHblits
    grandchild dies too rather than orphaning)."""
    if os.name == "posix" and hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.kill()
    except OSError:
        pass


def run_predict(cmd, cwd, env, timeout):
    # type: (List[str], Path, Dict[str, str], Optional[int]) -> Tuple[Optional[int], str, bool]
    """Run ``cmd`` with a wall-clock ``timeout``. Returns
    ``(returncode, combined_output, timed_out)``. On timeout the process group
    is killed and ``timed_out=True`` (returncode ``None``).

    ``universal_newlines=True`` (not ``text=``) keeps this working on 3.6.
    ``start_new_session`` puts the child in its own process group so a timeout
    can take down the whole tree (python + HHblits) at once.
    """
    popen_kwargs = dict(
        cwd=str(cwd), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **popen_kwargs)
    try:
        out, _ = proc.communicate(timeout=timeout if timeout and timeout > 0 else None)
        return proc.returncode, out or "", False
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, _ = proc.communicate(timeout=15)
        except Exception:  # noqa: BLE001
            out = ""
        return None, out or "", True


# --------------------------------------------------------------------------
# output layout
# --------------------------------------------------------------------------
def query_dir(output_dir, sample_id):
    # type: (Path, str) -> Path
    """Per-sample ``--querypath`` (working + output dir)."""
    return output_dir / sample_id


def result_csv(output_dir, sample_id):
    # type: (Path, str) -> Path
    return query_dir(output_dir, sample_id) / RESULT_CSV_NAME


def is_done(output_dir, sample_id):
    # type: (Path, str) -> bool
    return result_csv(output_dir, sample_id).is_file()


def build_predict_cmd(querypath, filename, chainid, ligands, cpu):
    # type: (Path, str, str, str, int) -> List[str]
    """The ``prediction.py`` argv (python is this env's interpreter)."""
    return [
        sys.executable, "prediction.py",
        "--querypath", str(querypath),
        "--filename", filename,
        "--chainid", chainid,
        "--ligands", ligands,
        "--cpu", str(cpu),
    ]


# --------------------------------------------------------------------------
# per-sample
# --------------------------------------------------------------------------
class _Timeout(Exception):
    """Internal marker so a per-sample timeout is tallied separately from other
    failures (kept local to avoid importing the sibling batch scripts)."""


def run_one_sample(
    sample_id,
    inputs_dir,
    output_dir,
    scripts_dir,
    chainid,
    ligands,
    cpu,
    timeout,
    resume,
):
    # type: (str, Path, Path, Path, str, str, int, Optional[int], bool) -> Dict
    """Run GraphBind for one sample. Never raises — returns a status dict so the
    batch loop keeps going and tallies failures."""
    result = {"sample_id": sample_id, "status": "ok", "error": None,
              "timed_out": False}  # type: Dict

    if resume and is_done(output_dir, sample_id):
        result["status"] = "skipped"
        return result

    t0 = time.time()
    try:
        src_pdb = inputs_dir / "{}.pdb".format(sample_id)
        if not src_pdb.is_file():
            raise FileNotFoundError("input PDB missing: {}".format(src_pdb))

        qdir = query_dir(output_dir, sample_id)
        qdir.mkdir(parents=True, exist_ok=True)
        # GraphBind reads <querypath>/<filename>; copy the prepared chain in.
        filename = "{}.pdb".format(sample_id)
        staged_pdb = qdir / filename
        shutil.copyfile(str(src_pdb), str(staged_pdb))

        cmd = build_predict_cmd(
            qdir.resolve(), filename, chainid, ligands, cpu)
        env = dict(os.environ)
        rc, out, timed_out = run_predict(cmd, scripts_dir, env, timeout)
        (qdir / "prediction.log").write_text(out, encoding="utf-8", errors="replace")

        if timed_out:
            raise _Timeout("prediction.py timed out after {}s".format(timeout))
        if rc != 0:
            tail = "\n".join(out.splitlines()[-15:])
            raise RuntimeError("prediction.py exited {}; tail:\n{}".format(rc, tail))
        if not is_done(output_dir, sample_id):
            tail = "\n".join(out.splitlines()[-15:])
            raise RuntimeError(
                "prediction.py finished but {} is missing; tail:\n{}".format(
                    result_csv(output_dir, sample_id), tail))
    except _Timeout as e:
        result["status"] = "failed"
        result["timed_out"] = True
        result["error"] = "TimeoutError: {}".format(e)
        logger.warning("sample %s TIMED OUT after %ss; skipping", sample_id, timeout)
    except Exception as e:  # noqa: BLE001 - one bad sample must not kill batch
        result["status"] = "failed"
        result["error"] = "{}: {}".format(type(e).__name__, e)
        logger.exception("sample %s failed", sample_id)
    result["runtime_seconds"] = round(time.time() - t0, 1)
    return result


def write_failures(failures, path):
    # type: (List[Dict], Path) -> None
    if not failures:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in failures:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(argv=None):
    # type: (Optional[List[str]]) -> int
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--inputs-dir", required=True, type=Path,
                   help="dir with prepared single-chain <sample_id>.pdb inputs")
    p.add_argument("--graphbind-dir", required=True, type=Path,
                   help="GraphBind install dir (its scripts/ holds prediction.py)")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="per-sample querypath folders are created under here")
    p.add_argument("--sample-list", required=True, type=Path,
                   help="txt (one id/line) or csv (sample_id column)")
    p.add_argument("--cpu", type=int, default=DEFAULT_CPU,
                   help="HHblits threads (prediction.py --cpu)")
    p.add_argument("--chainid", default="A", help="chain id (inputs are 'A')")
    p.add_argument("--ligands", default="RNA", help="ligand type")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help="per-sample wall-clock budget (s); on timeout the "
                        "process tree is killed and the sample marked failed "
                        "(0 disables; default {})".format(DEFAULT_TIMEOUT))
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--limit", type=int, default=0,
                   help="cap number processed after start/end (0 = no cap)")
    p.add_argument("--resume", action="store_true",
                   help="skip samples whose RNA-binding_result.csv exists")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    def _abs(pth):
        # type: (Path) -> Path
        return pth.expanduser().resolve()

    inputs_dir = _abs(args.inputs_dir)
    graphbind_dir = _abs(args.graphbind_dir)
    output_dir = _abs(args.output_dir)
    sample_list = _abs(args.sample_list)
    scripts_dir = graphbind_dir / "scripts"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not inputs_dir.is_dir():
        logger.error("inputs dir not found: %s", inputs_dir)
        return 1
    if not (scripts_dir / "prediction.py").is_file():
        logger.error("prediction.py not found under %s", scripts_dir)
        return 1

    samples = read_sample_list(sample_list)
    end = args.end if args.end is not None else len(samples)
    samples = samples[args.start:end]
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        logger.error("no samples selected (check --sample-list/--start/--end)")
        return 1

    if args.timeout and args.timeout > 0:
        logger.info("per-sample timeout: %ss", args.timeout)
    else:
        logger.warning("per-sample timeout DISABLED (--timeout %s)", args.timeout)

    n_ok = n_skip = n_fail = n_timeout = 0
    failures = []  # type: List[Dict]
    total = len(samples)
    for i, sample_id in enumerate(samples, 1):
        res = run_one_sample(
            clean_sample_id(sample_id),
            inputs_dir, output_dir, scripts_dir,
            args.chainid, args.ligands, args.cpu,
            args.timeout, args.resume,
        )
        status = res["status"]
        if status == "ok":
            n_ok += 1
            logger.info("[%d/%d] %s ok (%ss)", i, total, sample_id,
                        res.get("runtime_seconds"))
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

    write_failures(failures, output_dir / "failures.jsonl")
    logger.info("done: %d ok, %d skipped, %d failed (%d timed out) of %d",
                n_ok, n_skip, n_fail, n_timeout, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
