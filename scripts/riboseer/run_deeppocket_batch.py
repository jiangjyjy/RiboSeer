#!/usr/bin/env python3
"""Batch-run DeepPocket over the RiboSeer test split.

DeepPocket is a Category-B pocket detector (like P2Rank / Fpocket): it runs
fpocket to enumerate candidate pockets, ranks them with a classification CNN,
and segments the top ones. We drive its CLI exactly as verified on the main
server (``/opt/biotools/DeepPocket/``, conda env ``DeepPocket``)::

    CUDA_VISIBLE_DEVICES=<dev> python predict.py \
        -p <protein.pdb> \
        -c first_model_fold1_best_test_auc_85001.pth.tar \
        -s seg0_best_test_IOU_91.pth.tar \
        -r 3

For each requested sample this script:
  1. locates the raw complex for ``sample.source_pdb`` under ``--raw-dir``;
  2. extracts the protein chain (``sample.protein.chain_id``), drops waters /
     ligands / RNA, keeps standard amino acids and **renumbers to 1-based
     polymer index (gemmi label_seq)** with the chain renamed to ``A`` — the
     same coordinate frame P2Rank / EquiPNAS / NucleicNet use, so the residue
     indices the parser emits line up with step-1 ``binding_protein_residues``
     (``clean_pdb`` inside DeepPocket only drops het/non-standard residues, it
     does not renumber, so the numbering survives all the way to the pocket
     atom files);
  3. writes that PDB to ``<output-dir>/<sample_id>/<sample_id>.pdb`` and runs
     ``predict.py`` with ``cwd=<deeppocket-dir>`` (DeepPocket imports sibling
     modules and a ``gninamap`` file by relative path, so it must run from its
     install dir; the ``-p`` path is absolute so fpocket drops its
     ``<stem>_nowat_out/`` next to the input, inside the per-sample folder).

Each ``predict.py`` invocation is wrapped in a subprocess timeout; on timeout
the whole process group is SIGKILLed (so the fpocket grandchild dies too), the
sample is recorded as failed, and the batch moves on.

Usage (server, inside the ``DeepPocket`` conda env)
---------------------------------------------------
::

    python scripts/riboseer/run_deeppocket_batch.py \
        --processed-dir data/processed_quality \
        --sample-list   data/processed_quality/splits_tmscore_035/test.txt \
        --raw-dir       data/raw \
        --deeppocket-dir /opt/biotools/DeepPocket \
        --output-dir    data/batch_test_v7/deeppocket_outputs \
        --device 1 --timeout 300 --resume

See also
--------
src/step4_tool_adapters/external/deeppocket_parse.py : pocket-level -> per-residue,
    emits step-4 ``<sample_id>.jsonl`` (tool_id="deeppocket", category="B").
scripts/riboseer/run_nucleicnet_batch.py : sibling baseline; this script reuses
    its hardened sample-list / chain-extraction helpers.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Make repo root importable so the shared baseline helpers resolve regardless
# of the cwd the script is launched from.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the tested, hardened IO helpers from the NucleicNet batch script
# (sample-list reading with invisible-char cleaning, case-insensitive sample
# JSON lookup, raw-structure discovery, label_seq chain extraction).
from step4_tool_adapters.tool_io import (  # noqa: E402
    clean_sample_id,
    extract_protein_chain_pdb,
    find_raw_structure,
    load_sample_json,
    read_sample_list,
)

logger = logging.getLogger("run_deeppocket_batch")

DEFAULT_TIMEOUT = 300  # seconds per sample
DEFAULT_CLASS_CKPT = "first_model_fold1_best_test_auc_85001.pth.tar"
DEFAULT_SEG_CKPT = "seg0_best_test_IOU_91.pth.tar"
DEFAULT_RANK = 3


# --------------------------------------------------------------------------
# output layout
# --------------------------------------------------------------------------
def sample_dir(output_dir: Path, sample_id: str) -> Path:
    """Per-sample working/output folder."""
    return output_dir / sample_id


def input_pdb_path(output_dir: Path, sample_id: str) -> Path:
    """Where the extracted single-chain protein PDB is written for a sample."""
    return sample_dir(output_dir, sample_id) / f"{sample_id}.pdb"


def pockets_dir(output_dir: Path, sample_id: str) -> Path:
    """The fpocket pockets dir DeepPocket produces (``<stem>_nowat_out/pockets``).

    ``predict.py`` runs fpocket on ``<sample_id>_nowat.pdb`` (clean_pdb output),
    so the directory carries the ``_nowat`` infix.
    """
    return sample_dir(output_dir, sample_id) / f"{sample_id}_nowat_out" / "pockets"


def is_done(output_dir: Path, sample_id: str) -> bool:
    """A sample is complete once the ranked-confidence file exists."""
    direct = pockets_dir(output_dir, sample_id) / "bary_centers_confidence.txt"
    if direct.is_file():
        return True
    # Be tolerant of a different ``_out`` naming across DeepPocket versions.
    sd = sample_dir(output_dir, sample_id)
    if sd.is_dir():
        for cand in sd.glob("*_out/pockets/bary_centers_confidence.txt"):
            if cand.is_file():
                return True
    return False


# --------------------------------------------------------------------------
# subprocess with process-group timeout
# --------------------------------------------------------------------------
def _kill_tree(proc: subprocess.Popen) -> None:
    """SIGKILL the child and, on POSIX, its whole process group (so the fpocket
    grandchild spawned via ``os.system`` dies too rather than orphaning)."""
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


def run_predict(
    cmd: list[str], cwd: Path, env: dict, timeout: Optional[int],
) -> tuple[Optional[int], str, bool]:
    """Run ``cmd`` with a wall-clock ``timeout``. Returns
    ``(returncode, combined_output, timed_out)``. On timeout the process group
    is killed and ``timed_out=True`` (returncode ``None``)."""
    # start_new_session=True puts the child in its own process group so a
    # timeout can take down the whole tree (python + fpocket) at once.
    popen_kwargs = dict(
        cwd=str(cwd), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
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


def build_predict_cmd(
    *, protein_pdb: Path, class_ckpt: str, seg_ckpt: str, rank: int,
) -> list[str]:
    """The ``predict.py`` argv (python is this env's interpreter)."""
    return [
        sys.executable, "predict.py",
        "-p", str(protein_pdb),
        "-c", class_ckpt,
        "-s", seg_ckpt,
        "-r", str(rank),
    ]


# --------------------------------------------------------------------------
# per-sample
# --------------------------------------------------------------------------
def run_one_sample(
    sample_id: str,
    *,
    samples_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    deeppocket_dir: Path,
    class_ckpt: str,
    seg_ckpt: str,
    rank: int,
    device: Optional[str],
    timeout: Optional[int],
    resume: bool,
) -> dict:
    """Prepare input + run DeepPocket for one sample. Never raises — returns a
    status dict so the batch loop keeps going and tallies failures."""
    result = {"sample_id": sample_id, "status": "ok", "error": None,
              "timed_out": False}

    if resume and is_done(output_dir, sample_id):
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

        sd = sample_dir(output_dir, sample_id)
        sd.mkdir(parents=True, exist_ok=True)
        pdb_path = input_pdb_path(output_dir, sample_id).resolve()
        n_res = extract_protein_chain_pdb(raw_path, chain_id, pdb_path)
        result["n_residues"] = n_res

        env = dict(os.environ)
        if device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(device)
        cmd = build_predict_cmd(
            protein_pdb=pdb_path, class_ckpt=class_ckpt,
            seg_ckpt=seg_ckpt, rank=rank,
        )
        rc, out, timed_out = run_predict(cmd, deeppocket_dir, env, timeout)
        # Always keep the tool log for debugging.
        (sd / "predict.log").write_text(out, encoding="utf-8", errors="replace")

        if timed_out:
            raise TimeoutError(f"predict.py timed out after {timeout}s")
        if rc != 0:
            tail = "\n".join(out.splitlines()[-15:])
            raise RuntimeError(f"predict.py exited {rc}; tail:\n{tail}")
        if not is_done(output_dir, sample_id):
            tail = "\n".join(out.splitlines()[-15:])
            raise RuntimeError(
                f"predict.py finished but {pockets_dir(output_dir, sample_id)}/"
                f"bary_centers_confidence.txt is missing; tail:\n{tail}"
            )
    except TimeoutError as e:
        result["status"] = "failed"
        result["timed_out"] = True
        result["error"] = f"TimeoutError: {e}"
        logger.warning("sample %s TIMED OUT after %ss; skipping", sample_id, timeout)
    except Exception as e:  # noqa: BLE001 - one bad sample must not kill batch
        result["status"] = "failed"
        result["error"] = f"{type(e).__name__}: {e}"
        logger.exception("sample %s failed", sample_id)
    result["runtime_seconds"] = round(time.time() - t0, 1)
    return result


def write_failures(failures: list[dict], path: Path) -> None:
    if not failures:
        return
    import json
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
                   help="txt (one id/line) or csv (sample_id column)")
    p.add_argument("--raw-dir", required=True, type=Path,
                   help="dir with raw <source_pdb>.{pdb,cif}[.gz] structures")
    p.add_argument("--deeppocket-dir", required=True, type=Path,
                   help="DeepPocket install dir (cwd for predict.py)")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="per-sample folders are created under here")
    p.add_argument("--samples-dir", type=Path, default=None,
                   help="override sample-JSON dir "
                        "(default <processed-dir>/samples)")
    p.add_argument("--class-checkpoint", default=DEFAULT_CLASS_CKPT,
                   help="classification CNN checkpoint (relative to "
                        "--deeppocket-dir)")
    p.add_argument("--seg-checkpoint", default=DEFAULT_SEG_CKPT,
                   help="segmentation CNN checkpoint (relative to "
                        "--deeppocket-dir)")
    p.add_argument("--rank", type=int, default=DEFAULT_RANK,
                   help="number of top pockets to segment (predict.py -r)")
    p.add_argument("--device", default=None,
                   help="GPU id for CUDA_VISIBLE_DEVICES (unset = inherit)")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help="per-sample wall-clock budget (s); on timeout the "
                        "process tree is killed and the sample marked failed "
                        f"(0 disables; default {DEFAULT_TIMEOUT})")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--limit", type=int, default=0,
                   help="cap number processed after start/end (0 = no cap)")
    p.add_argument("--resume", action="store_true",
                   help="skip samples whose bary_centers_confidence.txt exists")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    def _abs(p: Path) -> Path:
        return p.expanduser().resolve()

    processed_dir = _abs(args.processed_dir)
    samples_dir = _abs(args.samples_dir or processed_dir / "samples")
    raw_dir = _abs(args.raw_dir)
    deeppocket_dir = _abs(args.deeppocket_dir)
    output_dir = _abs(args.output_dir)
    sample_list = _abs(args.sample_list)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, d in (("samples", samples_dir), ("raw", raw_dir),
                     ("deeppocket", deeppocket_dir)):
        if not d.is_dir():
            logger.error("%s dir not found: %s", label, d)
            return 1
    if not (deeppocket_dir / "predict.py").is_file():
        logger.error("predict.py not found under --deeppocket-dir: %s",
                     deeppocket_dir)
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
    logger.info("device (CUDA_VISIBLE_DEVICES): %s",
                args.device if args.device is not None else "<inherit>")

    n_ok = n_skip = n_fail = n_timeout = 0
    failures: list[dict] = []
    total = len(samples)
    for i, sample_id in enumerate(samples, 1):
        res = run_one_sample(
            clean_sample_id(sample_id),
            samples_dir=samples_dir, raw_dir=raw_dir, output_dir=output_dir,
            deeppocket_dir=deeppocket_dir, class_ckpt=args.class_checkpoint,
            seg_ckpt=args.seg_checkpoint, rank=args.rank, device=args.device,
            timeout=args.timeout, resume=args.resume,
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

    write_failures(failures, output_dir / "_failures.jsonl")
    logger.info("done: %d ok, %d skipped, %d failed (%d timed out) of %d",
                n_ok, n_skip, n_fail, n_timeout, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
