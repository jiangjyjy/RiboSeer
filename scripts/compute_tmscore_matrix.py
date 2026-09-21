"""Pairwise US-align TM-score matrix for a list of protein chains.

For each ``sample_id`` in ``--sample-list`` we read its source PDB id and
protein chain id from the sample JSON under ``--processed-dir/samples``,
extract just that one chain from ``<raw-dir>/<pdb>.pdb`` (or
``<raw-dir>/rna2p_balanced/<pdb>.pdb`` as a fallback - matches how
``raw.tar`` is laid out), and cache the single-chain PDB under
``<output>.cache/<sample_id>.pdb`` so subsequent runs skip the
extraction.

For every i<j we invoke::

    USalign struct_i struct_j -ter 0

and parse the two reported normalised TM-scores; the matrix entry is
``max(TM1, TM2)``. The diagonal is 1.0. Results are checkpointed every
``--checkpoint-every`` pairs so a 2 h run survives an SSH drop.

Outputs (alongside ``--output``)::

    tmscore_matrix.npy       float32, shape (N, N), symmetric
    tmscore_matrix.ids.txt   sample_ids in matrix order
    tmscore_matrix.meta.json {usalign_bin, n, computed_pairs, failed, ...}

Usage
-----
::

    python scripts/compute_tmscore_matrix.py \\
        --sample-list   data/processed_quality/splits_tmscore/\\
cluster_representatives.txt \\
        --raw-dir       data/raw/ \\
        --processed-dir data/processed_quality \\
        --output        data/processed_quality/splits_tmscore/\\
tmscore_matrix.npy \\
        --usalign-bin   USalign \\
        --n-workers     8

Add ``--resume`` to keep the existing partial matrix and only fill in
missing pairs (default ON when ``--output`` already exists).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


# ---- input helpers -------------------------------------------------------


def read_sample_list(path: Path) -> list[str]:
    """One sample_id per line; ignores blanks, ``#`` comments, and any
    text after the first whitespace (so the ``<sid>\\t<cluster>`` format
    produced by extract_cluster_representatives.py also parses)."""
    out: list[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        out.append(ln.split()[0])
    return out


def load_sample_chain(processed_dir: Path, sid: str
                      ) -> tuple[str, str]:
    """Returns ``(source_pdb, protein_chain_id)`` from
    ``processed_dir/samples/<sid>.json``; raises KeyError on miss."""
    for p in (processed_dir / "samples" / f"{sid}.json",
              processed_dir / f"{sid}.json"):
        if p.is_file():
            d = json.loads(p.read_text(encoding="utf-8"))
            pdb = d.get("source_pdb")
            ch = (d.get("protein") or {}).get("chain_id")
            if not pdb or not ch:
                raise KeyError(
                    f"{sid}: missing source_pdb/protein.chain_id")
            return str(pdb), str(ch)
    raise KeyError(f"{sid}: sample json not found in {processed_dir}")


def _probe(raw_dir: Path, pdb: str, ext: str) -> Optional[Path]:
    """Probe ``<raw>/<pdb>.<ext>`` + the ``rna2p_balanced/`` subdir,
    case-insensitive on the stem; return the first hit or None."""
    pdb_lc = pdb.lower()
    for c in (raw_dir / f"{pdb_lc}.{ext}",
              raw_dir / f"{pdb}.{ext}",
              raw_dir / "rna2p_balanced" / f"{pdb_lc}.{ext}",
              raw_dir / "rna2p_balanced" / f"{pdb}.{ext}"):
        if c.is_file():
            return c
    return None


def find_raw_pdb(raw_dir: Path, pdb: str) -> Path:
    """Locate the legacy single-char-chain PDB file for ``pdb``.
    Raises FileNotFoundError if absent (CIF-only structures are handled
    by ``find_raw_cif`` + ``extract_chain_from_cif``)."""
    hit = _probe(raw_dir, pdb, "pdb")
    if hit is not None:
        return hit
    raise FileNotFoundError(
        f"no PDB file for {pdb} under {raw_dir} (top level or "
        f"rna2p_balanced/); try the mmCIF fallback")


def find_raw_cif(raw_dir: Path, pdb: str) -> Optional[Path]:
    """Locate the mmCIF file for ``pdb``; None if absent. CIF is the
    only way to recover protein chains whose ``auth_asym_id`` is
    multi-character (e.g. ``'Lm'``, ``'4C'``, ``'U2'`` - PDB columns
    can't store those)."""
    return _probe(raw_dir, pdb, "cif")


# ---- single-chain PDB extraction ----------------------------------------

# Standard PDB columns 22 (0-indexed 21) holds the chain id. We keep
# ATOM / HETATM lines for the requested chain plus terminal markers.

def extract_chain_to_pdb(src: Path, chain_id: str, dst: Path) -> int:
    """Write ATOM/HETATM lines for ``chain_id`` from ``src`` (PDB) to
    ``dst``. Returns the number of atom lines written. Raises ValueError
    if the source is a CIF (call ``extract_chain_from_cif`` instead)
    or if the chain is not present (multi-char chain ids never match
    here - PDB column 22 only stores one char).

    Writes atomically: a ``.tmp`` sibling is filled first and renamed
    over ``dst`` only when at least one atom was found. Earlier versions
    opened ``dst`` directly and left an ``END\\n``-only 5-byte stub on
    disk when the chain was absent; that stub later passed the lax
    "size > 0" cache check and silently broke US-align.
    """
    if src.suffix.lower() == ".cif":
        raise ValueError(
            f"{src} is mmCIF; use extract_chain_from_cif() instead")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    n = 0
    try:
        with src.open("r", encoding="utf-8", errors="replace") as fh, \
                tmp.open("w", encoding="utf-8") as out:
            for line in fh:
                rec = line[:6].strip()
                if rec in ("ATOM", "HETATM"):
                    if len(line) >= 22 and line[21] == chain_id:
                        out.write(line)
                        n += 1
                elif rec == "TER":
                    if len(line) >= 22 and line[21] == chain_id:
                        out.write(line)
            out.write("END\n")
        if n == 0:
            raise ValueError(
                f"chain {chain_id!r} not found in {src}")
        os.replace(tmp, dst)
        return n
    finally:
        # tmp lingers iff we raised before os.replace; nuke it so the
        # cache never sees a half-written file.
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def extract_chain_from_cif(cif_path: Path, chain_id: str,
                           dst: Path) -> int:
    """mmCIF -> single-chain PDB via gemmi.

    Reads ``cif_path`` with gemmi, finds the chain whose ``name``
    matches ``chain_id`` (this is the mmCIF ``auth_asym_id``, which is
    where multi-char ids like ``'Lm'`` / ``'4C'`` live), clones it,
    renames it to ``'A'`` so it fits PDB's one-char chain column, and
    writes the result as a PDB at ``dst``. Returns the atom count
    written. Raises ValueError if no chain in any model carries the
    requested id (or carries it but has zero atoms).
    """
    import gemmi  # lazy: gemmi import is ~100 ms

    st = gemmi.read_structure(str(cif_path))
    for model in st:
        for chain in model:
            if chain.name != chain_id:
                continue
            new_st = gemmi.Structure()
            new_model = gemmi.Model("1")
            new_chain = chain.clone()
            new_chain.name = "A"
            new_model.add_chain(new_chain)
            new_st.add_model(new_model)
            n_atoms = sum(len(res) for res in new_chain)
            if n_atoms == 0:
                raise ValueError(
                    f"chain {chain_id!r} in {cif_path} has no atoms")
            # Write atomically so a mid-write crash can't leave a
            # half-baked file that the cache check then accepts.
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".tmp")
            try:
                new_st.write_pdb(str(tmp))
                os.replace(tmp, dst)
                return n_atoms
            finally:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except OSError:
                    pass
    raise ValueError(
        f"chain {chain_id!r} not found in {cif_path}")


# ---- US-align caller -----------------------------------------------------

# US-align's output wording has drifted across versions:
#
#   ~2020 build:  "TM-score= 0.92345 (if normalized by length of
#                    Chain_1, L=212, d0=5.46)"
#   ~2023 build:  "TM-score= 0.36176 (normalized by length of
#                    Structure_1: L=87, d0=3.36)"
#
# Both forms are accepted: the leading ``if`` is optional, ``Chain`` or
# ``Structure``, and the separator after the index can be ``,`` or
# ``:``. Hard-coding the older variant is what silently turned every
# pair into a parse failure when the server's US-align was newer.
_TM_RE = re.compile(
    r"TM-score\s*=\s*([0-9.]+)\s*\(\s*"
    r"(?:if\s+)?normalized\s+by\s+length\s+of\s+"
    r"(?:Chain|Structure)_([12])\b",
    re.IGNORECASE)


def parse_usalign_output(text: str) -> tuple[float, float]:
    """Return (tm1, tm2); raises ValueError if both are missing.
    Accepts both the legacy ``Chain_1/2`` and current ``Structure_1/2``
    wording."""
    tm1 = tm2 = None
    for m in _TM_RE.finditer(text):
        v = float(m.group(1))
        if m.group(2) == "1" and tm1 is None:
            tm1 = v
        elif m.group(2) == "2" and tm2 is None:
            tm2 = v
    if tm1 is None and tm2 is None:
        # Surface a tiny prefix of the actual stdout to make the next
        # version drift easier to diagnose - silent re-raise was the
        # bug that ate the previous 50-pair smoke run.
        snippet = " | ".join(
            ln.strip() for ln in text.splitlines()
            if "TM-score" in ln)[:240] or text[:240].replace("\n", " ")
        raise ValueError(
            f"no TM-score lines parsed from USalign output "
            f"(saw: {snippet!r})")
    return (tm1 if tm1 is not None else tm2,
            tm2 if tm2 is not None else tm1)


def run_usalign(bin_path: str, a: Path, b: Path,
                timeout: float = 120.0,
                verbose: bool = False) -> float:
    """Invoke US-align on the pair and return ``max(TM1, TM2)``.

    With ``verbose=True``, the exact subprocess command, plus the first
    ~500 chars of stdout and any stderr, are dumped to stderr before
    parsing. The driver uses this for the first ``--verbose`` pairs so
    output-format drift is caught at the first failure, not buried in
    the meta.json after a 2 h run.
    """
    cmd = [bin_path, str(a), str(b), "-ter", "0"]
    if verbose:
        print(f"[verbose] cmd: {' '.join(cmd)}", file=sys.stderr)
    cp = subprocess.run(cmd, capture_output=True, text=True,
                        timeout=timeout)
    if verbose:
        print(f"[verbose] rc={cp.returncode}", file=sys.stderr)
        if cp.stdout:
            print(f"[verbose] stdout (first 500 chars):\n"
                  f"{cp.stdout[:500]}", file=sys.stderr)
        if cp.stderr:
            print(f"[verbose] stderr:\n{cp.stderr[:500]}",
                  file=sys.stderr)
    if cp.returncode != 0:
        raise RuntimeError(
            f"USalign exit {cp.returncode}: {cp.stderr.strip()[:200]}")
    tm1, tm2 = parse_usalign_output(cp.stdout)
    return max(tm1, tm2)


# ---- driver --------------------------------------------------------------


def _pair_iter(n: int) -> Iterable[tuple[int, int]]:
    for i in range(n):
        for j in range(i + 1, n):
            yield i, j


def _undone_pairs(mat: np.ndarray) -> list[tuple[int, int]]:
    """A pair is "done" iff its symmetric entries are both finite and
    not the sentinel -1.0."""
    n = mat.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (np.isfinite(mat[i, j]) and mat[i, j] >= 0):
                out.append((i, j))
    return out


def _save_matrix(mat: np.ndarray, out: Path,
                 ids: list[str], meta: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, mat.astype(np.float32))
    out.with_suffix(".ids.txt").write_text(
        "\n".join(ids) + "\n", encoding="utf-8")
    out.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")


def _cache_valid(p: Path) -> bool:
    """A cached single-chain PDB is valid only if it has at least one
    ATOM line. Earlier buggy versions left ``END\\n``-only stubs when
    chain extraction failed; the previous cache check (existence +
    size > 0) accepted those and silently fed US-align garbage.

    The scan stops at the first ATOM hit, so this is O(few-bytes) for
    a stub and O(one-line) for a real PDB - cheap enough to run for
    every sid on every invocation.
    """
    if not p.is_file():
        return False
    try:
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("ATOM"):
                    return True
    except OSError:
        pass
    return False


def stage_one_chain(sid: str, processed_dir: Path, raw_dir: Path,
                    dst: Path) -> str:
    """Extract one sample's protein chain into a single-chain PDB at
    ``dst``. Strategy:

      1. legacy fast path - line-grep the source PDB by chain id (works
         only for single-char chain ids that the PDB file actually
         carries);
      2. mmCIF fallback via gemmi - required for ~96/332 samples whose
         ``auth_asym_id`` is multi-char (``'Lm'``, ``'4C'``, ``'U2'``);
         the cloned chain is renamed to ``'A'`` so the output is a
         valid PDB.

    Returns the source kind used (``'pdb'`` / ``'cif'``). Raises if
    neither path produces atoms (caller records the sid as skipped).
    """
    pdb, chain = load_sample_chain(processed_dir, sid)
    # 1) PDB fast path
    pdb_src: Optional[Path] = None
    try:
        pdb_src = find_raw_pdb(raw_dir, pdb)
    except FileNotFoundError:
        pdb_src = None
    if pdb_src is not None:
        try:
            extract_chain_to_pdb(pdb_src, chain, dst)
            return "pdb"
        except ValueError:
            # chain not in PDB columns (multi-char id, or PDB lost it):
            # fall through to CIF
            pass
    # 2) CIF fallback
    cif_src = find_raw_cif(raw_dir, pdb)
    if cif_src is not None:
        extract_chain_from_cif(cif_src, chain, dst)
        return "cif"
    raise FileNotFoundError(
        f"no usable structure for {pdb}/{chain} under {raw_dir} "
        f"(PDB chain {chain!r} not found and no mmCIF available)")


def _stage_chains(sids: list[str], processed_dir: Path,
                  raw_dir: Path, cache_dir: Path,
                  *, force_restage: bool = False) -> dict[str, Path]:
    """Extract every needed single-chain PDB once. Returns
    ``{sid: cached_pdb}``. A pre-existing cache file is reused only if
    ``_cache_valid`` says so; invalid stubs are deleted and re-extracted
    (this is what dragged earlier smoke runs into "all-failed" land).

    Set ``force_restage=True`` to nuke every cache file unconditionally
    (used by the ``--force-restage`` CLI flag during a debugging cycle).
    Tallies PDB vs CIF source counts and the invalid-stub eviction count
    so the user can see exactly what the cache layer did."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    errors: list[str] = []
    src_counts = {"pdb": 0, "cif": 0, "cached": 0, "evicted": 0}
    for sid in sids:
        dst = cache_dir / f"{sid}.pdb"
        if force_restage and dst.exists():
            try:
                dst.unlink()
                src_counts["evicted"] += 1
            except OSError:
                pass
        if _cache_valid(dst):
            out[sid] = dst
            src_counts["cached"] += 1
            continue
        # If something exists at dst but is not a valid PDB (typically
        # the legacy "END\n" 5-byte stub) - wipe it so the next
        # extraction starts clean.
        if dst.exists():
            try:
                dst.unlink()
                src_counts["evicted"] += 1
            except OSError:
                pass
        try:
            kind = stage_one_chain(sid, processed_dir, raw_dir, dst)
            out[sid] = dst
            src_counts[kind] += 1
        except (KeyError, FileNotFoundError, ValueError) as e:
            errors.append(f"{sid}: {e}")
    print(f"  staged: pdb={src_counts['pdb']}  "
          f"cif={src_counts['cif']}  cached={src_counts['cached']}  "
          f"evicted_stub={src_counts['evicted']}  "
          f"failed={len(errors)}")
    if errors:
        print(f"WARN: {len(errors)} chains could not be staged:",
              file=sys.stderr)
        for line in errors[:10]:
            print(f"  {line}", file=sys.stderr)
        if len(errors) > 10:
            print(f"  ... +{len(errors) - 10} more", file=sys.stderr)
    return out


def _worker(args):
    # Trailing ``verbose`` flag is set per-task by the driver so the
    # first N pairs dump the exact command + stdout to stderr.
    i, j, pa, pb, bin_path, timeout, verbose = args
    try:
        score = run_usalign(bin_path, pa, pb, timeout, verbose=verbose)
        return i, j, score, None
    except Exception as e:  # subprocess / parse / timeout
        return i, j, float("nan"), f"{type(e).__name__}: {e}"


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample-list", type=Path, required=True)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True,
                   help=".npy path; sibling .ids.txt + .meta.json "
                        "written next to it.")
    p.add_argument("--usalign-bin", default="USalign")
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--timeout", type=float, default=120.0,
                   help="seconds per US-align call.")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="single-chain PDB cache; defaults to "
                        "<output_dir>/chains/  (renamed from the older "
                        "<output_stem>.cache/ to be obvious; if you "
                        "have an old cache pass --cache-dir explicitly).")
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--resume", action="store_true",
                   help="reuse an existing matrix at --output; only "
                        "recompute pairs marked as NaN or -1. Rows for "
                        "sids that failed last time are auto-recovered "
                        "when staging now succeeds.")
    p.add_argument("--force-restage", action="store_true",
                   help="discard all cached chain PDBs and re-extract "
                        "from raw; use this once after upgrading past "
                        "the stub-leaking version.")
    p.add_argument("--limit", type=int, default=None,
                   help="cap the number of pairs computed this run "
                        "(useful for a small smoke before the 2 h job).")
    p.add_argument("--verbose", type=int, default=0, metavar="N",
                   help="print the exact USalign command and the first "
                        "500 chars of stdout/stderr for the first N "
                        "pairs (debug output-format drift). 0 = off.")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    sids = read_sample_list(args.sample_list)
    n = len(sids)
    if n < 2:
        print(f"ERROR: need >= 2 samples, got {n}", file=sys.stderr)
        return 1
    print(f"compute_tmscore_matrix: N={n}, "
          f"pairs={n * (n - 1) // 2}")

    cache_dir = args.cache_dir or args.output.parent / "chains"
    print(f"  cache dir            : {cache_dir.resolve()}")
    chain_pdbs = _stage_chains(sids, args.processed_dir,
                               args.raw_dir, cache_dir,
                               force_restage=args.force_restage)
    usable = [s for s in sids if s in chain_pdbs]
    if len(usable) != n:
        # We KEEP the original ordering for usable sids but mark the
        # missing ones - their row/column is NaN. This is preferable
        # to silently dropping them (matrix indices stay aligned with
        # the input sample list).
        print(f"WARN: {n - len(usable)} of {n} sids had no usable "
              f"chain; their rows/cols will be NaN.", file=sys.stderr)

    # Matrix init / resume
    if args.resume and args.output.is_file():
        mat = np.load(args.output).astype(np.float32)
        if mat.shape != (n, n):
            print(f"ERROR: existing matrix shape {mat.shape} != "
                  f"({n},{n}); rerun without --resume to start fresh.",
                  file=sys.stderr)
            return 1
        print(f"  resuming from {args.output}")
        # If a sid failed last time (its diagonal was clobbered with
        # NaN) but now has a usable chain, reset its row so the
        # previously-skipped pairs get recomputed in this run. Without
        # this, the CIF-fallback rescue would land in a matrix that
        # still considers those reps "missing" and skip their pairs.
        recovered = 0
        for i, s in enumerate(sids):
            if s in chain_pdbs and not np.isfinite(mat[i, i]):
                mat[i, :] = -1.0
                mat[:, i] = -1.0
                mat[i, i] = 1.0
                recovered += 1
        if recovered:
            print(f"  recovered {recovered} previously-skipped rows "
                  f"(staging now succeeds)")
    else:
        mat = np.full((n, n), -1.0, dtype=np.float32)
        np.fill_diagonal(mat, 1.0)

    # Mark un-stageable rows as NaN (skipped). This is the FINAL state
    # for sids whose chain still cannot be staged in this run.
    missing_idx = [i for i, s in enumerate(sids)
                   if s not in chain_pdbs]
    for i in missing_idx:
        mat[i, :] = np.nan
        mat[:, i] = np.nan
        mat[i, i] = np.nan

    pairs = [(i, j) for i, j in _undone_pairs(mat)
             if i not in missing_idx and j not in missing_idx]
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"  pairs to compute     : {len(pairs)}")
    if not pairs:
        _save_matrix(mat, args.output, sids, {
            "n": n, "computed_pairs": 0, "failed": 0,
            "usalign_bin": args.usalign_bin,
            "raw_dir": str(args.raw_dir),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        return 0

    tasks = [(i, j, chain_pdbs[sids[i]], chain_pdbs[sids[j]],
              args.usalign_bin, args.timeout,
              idx < args.verbose)            # verbose for first N pairs
             for idx, (i, j) in enumerate(pairs)]

    done = failed = 0
    t0 = time.monotonic()
    last_ckpt = 0
    failures: list[str] = []

    def _consume(result):
        nonlocal done, failed, last_ckpt
        i, j, score, err = result
        done += 1
        if err:
            failed += 1
            mat[i, j] = mat[j, i] = np.nan
            if len(failures) < 10:
                msg = f"{sids[i]} x {sids[j]}: {err}"
                failures.append(msg)
                # Live print so a regex/version mismatch is visible at
                # the first failure, not buried in meta.json after the
                # run. Capped at 10 lines so a real all-fail scenario
                # doesn't spam.
                print(f"  FAIL[{failed:>3}] {msg}", file=sys.stderr)
                if len(failures) == 10:
                    print("  (further failure messages suppressed; "
                          "full list goes to meta.json)",
                          file=sys.stderr)
        else:
            mat[i, j] = mat[j, i] = float(score)
        if done - last_ckpt >= args.checkpoint_every:
            _save_matrix(mat, args.output, sids, {
                "n": n, "computed_pairs": done, "failed": failed,
                "usalign_bin": args.usalign_bin,
                "raw_dir": str(args.raw_dir),
                "checkpoint_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            last_ckpt = done
            dt = time.monotonic() - t0
            eta = dt / done * (len(pairs) - done)
            print(f"  ckpt {done}/{len(pairs)}  "
                  f"({100 * done / len(pairs):.1f}%)  "
                  f"eta ~{eta / 60:.1f} min  fail={failed}")

    if args.n_workers <= 1:
        # Serial path: keeps mock.patch on subprocess.run effective
        # (ProcessPool would re-import the module in workers and miss
        # the patch) and avoids fork overhead for tiny jobs.
        for task in tasks:
            _consume(_worker(task))
    else:
        with cf.ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            for result in ex.map(_worker, tasks, chunksize=4):
                _consume(result)

    _save_matrix(mat, args.output, sids, {
        "n": n,
        "computed_pairs": done,
        "failed": failed,
        "usalign_bin": args.usalign_bin,
        "raw_dir": str(args.raw_dir),
        "failures_sample": failures,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"  done: {done} pairs, {failed} failed, "
          f"matrix -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
