#!/usr/bin/env python3
"""Parse BindUP output into step-4 ``<sample_id>.jsonl`` (Category C).

BindUP predicts nucleic-acid-binding by ranking a protein's largest positive
electrostatic patches. Its batch mode emits, per PDB+chain, a ``patch_list.txt``::

    ============
    PDB ID: 2czj
    Chain E
    This chain is classified as NA-binding (73% confidence)
    Largest Positive Patches:
    Patch 1: ALA2 PRO3 VAL4 LEU5 ...
    Patch 2: TYR14 GLU18 ...
    Patch 3: GLU30 ILE60 ...

Per-residue scoring (Cat C, like a ranked-pocket tool)
------------------------------------------------------
Patch rank -> score from ``--patch-scores`` (default ``1.0,0.67,0.33``); a
residue in several patches takes the max; residues in no patch are absent
(treated as 0 downstream). ``binding_protein_residues`` = the Patch-1 residues.

Residue numbering / GT alignment
--------------------------------
BindUP reports the **author** residue numbering of the deposited structure
(e.g. ``ALA2`` -> author seqid 2). RiboSeer ground truth is the **1-based
label_seq** polymer index. We build the ``author -> label_seq`` map for the
protein chain with gemmi (``assign_label_seq_id``) from the raw structure under
``--raw-dir`` and translate every patch residue, so the emitted indices line up
with step-1 ``binding_protein_residues``. Tokens that don't map (numbering
mismatch / missing residue) are dropped and counted.

sample_id mapping
-----------------
BindUP keys results by ``(PDB ID, chain)``; RiboSeer keys by ``sample_id``
(``<pdb>_<protchain>_<rnachain>``). We map via each sample JSON's
``source_pdb`` + ``protein.chain_id``. One protein chain can back several samples
(different RNA partners) — they all receive the same BindUP prediction.

Output: ``ToolPrediction(tool_id="bindup", category="C")`` merged into
``<step4-dir>/<sample_id>.jsonl``, replacing any prior ``bindup`` entry and
preserving other tools (same convention as the sibling parsers).

Usage
-----
::

    python -m step4_tool_adapters.external.bindup_parse.py \
        --results-dir   bindup_outputs/ \
        --step4-dir     data/batch_test_v7/step4/ \
        --processed-dir data/processed_quality \
        --raw-dir       data/raw \
        --sample-list   data/processed_quality/splits_tmscore_035/test.txt
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional

# Make repo root + src importable for the schemas and shared helpers.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT / "src", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import gemmi  # type: ignore
except ImportError as _e:  # pragma: no cover - environment dependent
    gemmi = None  # type: ignore
    _GEMMI_IMPORT_ERROR = _e
else:
    _GEMMI_IMPORT_ERROR = None

from step4_tool_adapters.schemas import (  # noqa: E402
    ToolPrediction,
    ToolPredictionSet,
)
from step4_tool_adapters.tool_io import (  # noqa: E402
    _is_protein_residue,
    clean_sample_id,
    find_raw_structure,
    load_sample_json,
    read_sample_list,
)

logger = logging.getLogger("bindup_parse")

TOOL_ID = "bindup"
CATEGORY = "C"
DEFAULT_PATCH_SCORES = [1.0, 0.67, 0.33]

# A residue token like ``ALA2`` / ``TYR14`` / ``GLU30`` / ``HIS12A`` (resname +
# author number + optional insertion code).
_RES_TOKEN_RE = re.compile(r"([A-Za-z]+)(-?\d+)([A-Za-z]?)")
_PDB_RE = re.compile(r"PDB\s*ID[:\s]+(\S+)", re.I)
_CHAIN_RE = re.compile(r"Chain[:\s]+(\S+)", re.I)
_PATCH_RE = re.compile(r"Patch\s*\d+\s*[:\s]\s*(.*)", re.I)


# --------------------------------------------------------------------------
# patch_list.txt parsing
# --------------------------------------------------------------------------
def parse_residue_tokens(text: str) -> list[tuple[int, str]]:
    """Parse ``ALA2 PRO3 ...`` -> ``[(2, ''), (3, ''), ...]`` (author num, icode).
    The residue name is dropped — only the number + insertion code are needed to
    map to label_seq."""
    out: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for _resname, num, icode in _RES_TOKEN_RE.findall(text):
        try:
            n = int(num)
        except ValueError:
            continue
        key = (n, icode.strip())
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def parse_patch_list_text(text: str) -> list[dict]:
    """Parse a ``patch_list.txt`` body into one or more chain blocks.

    Returns ``[{'pdb': str, 'chain': str, 'patches': [[(num, icode), ...], ...]}]``.
    A new block starts at each ``PDB ID:`` line, and also at a ``Chain`` line when
    the current block already accumulated patches (some files list several chains
    under one PDB-ID header)."""
    blocks: list[dict] = []
    cur: Optional[dict] = None

    def flush():
        nonlocal cur
        if cur and cur.get("pdb") and cur.get("chain") is not None:
            blocks.append(cur)
        cur = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _PDB_RE.match(line)
        if m:
            flush()
            cur = {"pdb": m.group(1), "chain": None, "patches": []}
            continue
        m = _CHAIN_RE.match(line)
        if m:
            if cur is None:
                cur = {"pdb": None, "chain": None, "patches": []}
            if cur.get("chain") is not None and cur["patches"]:
                pdb = cur["pdb"]
                flush()
                cur = {"pdb": pdb, "chain": None, "patches": []}
            cur["chain"] = m.group(1)
            continue
        m = _PATCH_RE.match(line)
        if m and cur is not None:
            cur["patches"].append(parse_residue_tokens(m.group(1)))
            continue
    flush()
    return blocks


def build_bindup_index(results_dir: Path) -> dict:
    """Scan ``results_dir`` for ``*patch_list*.txt`` and index every chain block
    by ``(pdb_id.lower(), chain)`` -> ``patches`` (list of token lists)."""
    index: dict = {}
    files = sorted(results_dir.rglob("*patch_list*.txt"))
    for f in files:
        try:
            blocks = parse_patch_list_text(f.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        for b in blocks:
            key = (b["pdb"].lower(), b["chain"])
            if key in index:
                logger.warning("duplicate BindUP block for %s in %s; keeping first",
                               key, f.name)
                continue
            index[key] = b["patches"]
    logger.info("indexed %d BindUP chain block(s) from %d file(s)",
                len(index), len(files))
    return index


def lookup_patches(index: dict, pdb_id: str, chain: str):
    """Look up patches for a (pdb, chain), tolerating chain-case differences."""
    pdb = pdb_id.lower()
    for ch in (chain, chain.upper(), chain.lower()):
        if (pdb, ch) in index:
            return index[(pdb, ch)]
    return None


# --------------------------------------------------------------------------
# author -> label_seq mapping (gemmi)
# --------------------------------------------------------------------------
def build_auth_to_label(raw_path: Path, chain_id: str) -> dict:
    """Map author residue numbering -> 1-based label_seq for one protein chain.

    Returns a dict keyed by ``(auth_num, icode)`` *and* by bare ``auth_num``
    (empty-icode residue preferred) so a token without an insertion code still
    resolves. Built with the same gemmi setup as ``extract_protein_chain_pdb``.
    """
    if gemmi is None:
        raise RuntimeError(f"gemmi is required for residue mapping: {_GEMMI_IMPORT_ERROR}")
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw structure not found: {raw_path}")
    structure = gemmi.read_structure(str(raw_path), merge_chain_parts=True)
    structure.setup_entities()
    structure.assign_label_seq_id(True)
    if len(structure) == 0:
        raise ValueError(f"no models in {raw_path}")
    model = structure[0]

    chain = None
    for ch in model:
        if ch.name == chain_id:
            chain = ch
            break
    if chain is None:  # case-insensitive fallback
        for ch in model:
            if ch.name.lower() == chain_id.lower():
                chain = ch
                break
    if chain is None:
        raise ValueError(
            f"chain {chain_id!r} not found in {raw_path} "
            f"(available: {[c.name for c in model]})")

    amap: dict = {}
    for res in chain:
        if not _is_protein_residue(res.name):
            continue
        if res.label_seq is None:
            continue
        num = int(res.seqid.num)
        icode = (res.seqid.icode or " ").strip()
        label = int(res.label_seq)
        amap[(num, icode)] = label
        # bare-number key: prefer the empty-icode residue, else first seen.
        if icode == "" or num not in amap:
            amap[num] = label
    return amap


def map_token(amap: dict, num: int, icode: str) -> Optional[int]:
    """Translate one (author num, icode) to label_seq via ``amap``."""
    if (num, icode) in amap:
        return amap[(num, icode)]
    return amap.get(num)


# --------------------------------------------------------------------------
# record assembly + merge
# --------------------------------------------------------------------------
def build_prediction(
    sample_id: str,
    patch_label_lists: list[list[int]],
    *,
    patch_scores: list[float],
    raw_output_dir: Optional[str] = None,
) -> ToolPrediction:
    """Assemble the ``bindup`` ToolPrediction from per-patch label_seq residues.

    ``patch_label_lists[i]`` is the (already mapped) residue list of rank ``i+1``.
    Score of rank ``i`` = ``patch_scores[i]`` (0 beyond the list); per-residue =
    max over containing patches; binding = Patch 1.
    """
    per_res: dict[int, float] = {}
    for rank, residues in enumerate(patch_label_lists):
        score = patch_scores[rank] if rank < len(patch_scores) else 0.0
        if score <= 0.0:
            continue
        for r in residues:
            if r >= 1 and score > per_res.get(r, 0.0):
                per_res[r] = score
    per_res = {int(k): round(float(v), 4) for k, v in per_res.items()}

    binding = (sorted({r for r in patch_label_lists[0] if r >= 1})
               if patch_label_lists else [])

    return ToolPrediction(
        tool_id=TOOL_ID,
        category=CATEGORY,
        sample_id=sample_id,
        success=True,
        binding_protein_residues=binding,          # [] is valid ("no binding")
        per_residue_confidence=per_res or None,
        raw_output_dir=raw_output_dir,
    )


def merge_into_set(
    step4_dir: Path, sample_id: str, prediction: ToolPrediction,
) -> Path:
    """Insert ``prediction`` into ``<step4_dir>/<sample_id>.jsonl``, dropping any
    prior ``bindup`` entry and preserving other tools."""
    step4_dir.mkdir(parents=True, exist_ok=True)
    out_path = step4_dir / f"{sample_id}.jsonl"

    pset = ToolPredictionSet(sample_id=sample_id)
    if out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    pset = ToolPredictionSet.model_validate_json(line)
                except Exception:  # noqa: BLE001
                    pass
    pset.sample_id = sample_id
    pset.predictions = [p for p in pset.predictions if p.tool_id != TOOL_ID]
    pset.predictions.append(prediction)
    pset.tools_run = sorted({p.tool_id for p in pset.predictions})

    tmp = out_path.with_suffix(".jsonl.tmp")
    tmp.write_text(pset.model_dump_json() + "\n", encoding="utf-8")
    tmp.replace(out_path)
    return out_path


# --------------------------------------------------------------------------
# per-sample
# --------------------------------------------------------------------------
def parse_one_sample(
    sample_id: str,
    *,
    bindup_index: dict,
    samples_dir: Path,
    raw_dir: Path,
    patch_scores: list[float],
    amap_cache: dict,
) -> ToolPrediction:
    """Map one sample's BindUP patches to a ToolPrediction (raises on missing
    input). ``amap_cache`` memoises the author->label map per (pdb, chain)."""
    sample = load_sample_json(samples_dir, sample_id)
    source_pdb = sample["source_pdb"]
    chain_id = sample["protein"]["chain_id"]

    patches = lookup_patches(bindup_index, source_pdb, chain_id)
    if patches is None:
        raise FileNotFoundError(
            f"no BindUP result for ({source_pdb}, chain {chain_id})")

    cache_key = (source_pdb.lower(), chain_id)
    amap = amap_cache.get(cache_key)
    if amap is None:
        raw_path = find_raw_structure(raw_dir, source_pdb)
        if raw_path is None:
            raise FileNotFoundError(
                f"no raw structure for {source_pdb!r} under {raw_dir}")
        amap = build_auth_to_label(raw_path, chain_id)
        amap_cache[cache_key] = amap

    patch_label_lists: list[list[int]] = []
    n_tok = n_unmapped = 0
    for patch in patches:
        labels: list[int] = []
        for num, icode in patch:
            n_tok += 1
            lab = map_token(amap, num, icode)
            if lab is None:
                n_unmapped += 1
                continue
            labels.append(lab)
        patch_label_lists.append(labels)
    if n_tok and n_unmapped == n_tok:
        raise ValueError(
            f"{sample_id}: none of {n_tok} BindUP residues mapped to label_seq "
            f"(author numbering mismatch?)")
    if n_unmapped:
        logger.warning("%s: %d/%d patch residues unmapped to label_seq",
                       sample_id, n_unmapped, n_tok)

    return build_prediction(
        sample_id, patch_label_lists, patch_scores=patch_scores,
        raw_output_dir=str((raw_dir).resolve()))


def parse_patch_scores(s: str) -> list[float]:
    out: list[float] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    if not out:
        raise argparse.ArgumentTypeError("empty --patch-scores")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results-dir", required=True, type=Path,
                   help="dir containing *_patch_list.txt files (recursed)")
    p.add_argument("--step4-dir", required=True, type=Path,
                   help="step-4 JSONL dir to merge into")
    p.add_argument("--processed-dir", required=True, type=Path,
                   help="step-1 processed dir (its samples/ gives source_pdb + chain)")
    p.add_argument("--raw-dir", required=True, type=Path,
                   help="raw structures for the author->label_seq residue map")
    p.add_argument("--samples-dir", type=Path, default=None,
                   help="override sample-JSON dir (default <processed-dir>/samples)")
    p.add_argument("--sample-list", type=Path, default=None,
                   help="txt/csv of sample ids; default: every sample JSON whose "
                        "(pdb,chain) is found in the BindUP results")
    p.add_argument("--patch-scores", type=parse_patch_scores,
                   default=DEFAULT_PATCH_SCORES,
                   help="comma-separated per-rank scores (default 1.0,0.67,0.33)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level)

    results_dir = args.results_dir.expanduser().resolve()
    step4_dir = args.step4_dir.expanduser().resolve()
    samples_dir = (args.samples_dir or args.processed_dir / "samples").expanduser().resolve()
    raw_dir = args.raw_dir.expanduser().resolve()

    if not results_dir.is_dir():
        logger.error("results dir not found: %s", results_dir)
        return 1
    if not samples_dir.is_dir():
        logger.error("samples dir not found: %s", samples_dir)
        return 1

    bindup_index = build_bindup_index(results_dir)
    if not bindup_index:
        logger.error("no BindUP patch blocks parsed under %s", results_dir)
        return 1

    if args.sample_list is not None:
        samples = read_sample_list(args.sample_list.expanduser().resolve())
    else:
        # discover: every sample JSON whose (pdb, chain) is in the index
        samples = []
        for jf in sorted(samples_dir.glob("*.json")):
            try:
                d = load_sample_json(samples_dir, jf.stem)
                if lookup_patches(bindup_index, d["source_pdb"],
                                  d["protein"]["chain_id"]) is not None:
                    samples.append(jf.stem)
            except Exception:  # noqa: BLE001
                continue
    if not samples:
        logger.error("no samples selected")
        return 1

    amap_cache: dict = {}
    n_ok = n_missing = n_fail = 0
    total = len(samples)
    for i, sid in enumerate(samples, 1):
        sid = clean_sample_id(sid)
        try:
            pred = parse_one_sample(
                sid, bindup_index=bindup_index, samples_dir=samples_dir,
                raw_dir=raw_dir, patch_scores=args.patch_scores,
                amap_cache=amap_cache)
        except FileNotFoundError as e:
            n_missing += 1
            logger.warning("[%d/%d] %s missing: %s", i, total, sid, e)
            continue
        except Exception:  # noqa: BLE001
            n_fail += 1
            logger.exception("[%d/%d] %s failed", i, total, sid)
            continue
        out = merge_into_set(step4_dir, sid, pred)
        n_res = len(pred.per_residue_confidence or {})
        n_bind = len(pred.binding_protein_residues or [])
        n_ok += 1
        logger.info("[%d/%d] %s -> %s (%d residues scored, %d binding)",
                    i, total, sid, out, n_res, n_bind)

    logger.info("done: %d ok, %d missing, %d failed (of %d)",
                n_ok, n_missing, n_fail, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
