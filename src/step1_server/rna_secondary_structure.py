#!/usr/bin/env python
"""Stage 1.3b — RNA secondary structure prediction via ViennaRNA (Linux server).

⚠️  LINUX SERVER TASK — DO NOT RUN ON WINDOWS.
    ViennaRNA's Python bindings are only packaged for Linux/macOS through
    bioconda. This script is written for the server side of the RiboSeer
    step 1 pipeline and must not be launched from the local Windows env.

What it does
------------
For every sample JSON produced by stages 1.2 / 1.3a:

  1. Load the RNA sequence from `rna.sequence`
  2. Sanitize the sequence for ViennaRNA:
       * strip gap characters '-' (missing-density positions)
       * map inosine 'I' → 'G' (base-pair behavior closest to G; the
         7wv3 / 7wv4 poly-inosine samples flagged in stage 1.3a need this)
       * map any other non-{A,C,G,U} character to 'N'
  3. Run MFE fold (`RNA.fold_compound(...).mfe()`) to get a dot-bracket
     structure and free energy
  4. Classify every nucleotide into one of
       paired / hairpin / interior / multiloop / external
     based on the pair-table topology, and compute per-element fractions
  5. Write back into:
       sample["rna"]["features"]["ss_status"]
         = "done" | "skipped_too_long" | "skipped_empty" | "error"
           (a concise state flag — absence of this field means "pending")
       sample["rna"]["features"]["secondary_structure_pred"]
         = {
             "structure":              "...(((...)))...",   # dot-bracket
             "mfe":                    -42.30,               # kcal/mol
             "fold_sequence":          "GGCCAU...",          # sanitized seq
             "sanitized_from_inosine": bool,
             "skipped_reason":         null | "too_long_XXXX_nt" | ...
           }
       sample["rna"]["features"]["structure_composition"]
         = {
             "paired_frac":    0.523,
             "hairpin_frac":   0.105,
             "interior_frac":  0.085,
             "multiloop_frac": 0.031,
             "external_frac":  0.012
           }
  6. For sequences longer than `--max-length` (default 4000), `structure`
     and `structure_composition` are left None with `ss_status =
     "skipped_too_long"` and `skipped_reason = "too_long_XXXX_nt"`
     (RNAfold is O(n³); 4000 nt ≈ 30-60 s/sequence, 5000 nt ≈ minutes).
  7. Resume: samples with `ss_status in {"done", "error", "skipped_empty"}`
     are skipped unless `--force` is given. Samples with
     `ss_status == "skipped_too_long"` are ALWAYS re-checked because
     `--max-length` is a CLI param that may be raised on a later run.

Why not forgi?
--------------
forgi would give us BulgeGraph element classification in one line, but it
pulls in extra deps (numpy, biopython, networkx). The hand-written classifier
below is ~40 lines and matches forgi's `s/h/i/m/f+t` buckets with
"interior" = "bulge + internal loop" and "external" = "5'/3' dangling ends".

Caching
-------
In a ribosome or spliceosome structure, many (protein_chain, rna_chain) pairs
share the SAME RNA chain — only the protein partner differs. So we hash the
sanitized sequence and cache MFE results; 45468 samples likely fold only a
few thousand unique sequences.

Install on server
-----------------
    conda create -n riboseer python=3.11 -y
    conda activate riboseer
    conda config --add channels defaults
    conda config --add channels bioconda
    conda config --add channels conda-forge
    conda config --set channel_priority strict
    conda install -c conda-forge tqdm -y
    # ViennaRNA via pip, NOT conda: bioconda's viennarna 2.7.2 recipe is
    # pinned to Python 3.10 and will clash with a 3.11 env. The PyPI
    # `viennarna` 2.7.2 package ships manylinux wheels for Python 3.8-3.14.
    python -m pip install viennarna
    # (no forgi needed — classifier is inline)

    # Verify:
    #   python -c "import RNA; fc=RNA.fold_compound('GGCCAU'); print(fc.mfe())"

Usage
-----
    # dry run on a small subset first
    python rna_secondary_structure.py \
        --processed-dir /srv/pocket/data/processed \
        --stats-dir     /srv/pocket/data/stats \
        --limit 50

    # full run (idempotent; can Ctrl-C and re-run)
    python rna_secondary_structure.py \
        --processed-dir /srv/pocket/data/processed \
        --stats-dir     /srv/pocket/data/stats

    # cap at 3000 nt if 4000 is too slow
    python rna_secondary_structure.py \
        --processed-dir /srv/pocket/data/processed \
        --stats-dir     /srv/pocket/data/stats \
        --max-length 3000

    # force refold (ignore resume state)
    python rna_secondary_structure.py --force ...
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

# ViennaRNA is imported lazily inside `fold_rna()` so that `--selftest` (which
# only exercises the pure-Python classifier) runs on any machine, including
# Windows, without needing the bioconda package installed.
RNA = None  # type: ignore[assignment]


def _require_viennarna():
    global RNA
    if RNA is not None:
        return
    try:
        import RNA as _RNA
    except ImportError:
        sys.exit(
            "ERROR: ViennaRNA Python bindings not found.\n"
            "Install via pip (NOT bioconda — recipe is pinned to Python 3.10):\n"
            "    python -m pip install viennarna\n"
            "Verify:\n"
            "    python -c \"import RNA; print(RNA.fold_compound('GGCCAU').mfe())\"\n"
        )
    RNA = _RNA


# tqdm is used in main() only; import is deferred to keep --selftest
# dependency-free.


# ---------- sid_to_filename (copied from step1_local/_common.py) ------------
#
# Duplicated here so the server script is self-contained and doesn't need the
# local side of the repo on sys.path. Must stay in sync with
# step1_local/_common.py::sid_to_filename. Unit test below guards that.


def _encode_case_marked(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalpha() and ch.islower():
            out.append("-")
        out.append(ch)
    return "".join(out)


def sid_to_filename(sample_id: str) -> str:
    pdb, _, tail = sample_id.partition("_")
    if not tail:
        return _encode_case_marked(sample_id)
    return f"{pdb}_{_encode_case_marked(tail)}"


# ---------- sequence sanitization -------------------------------------------


def sanitize_sequence(seq: str) -> tuple[str, bool]:
    """Make `seq` safe for ViennaRNA.

    Returns (cleaned_sequence, sanitized_from_inosine_flag).

    Rules (in order):
      - drop gap '-' (stage 1.2 stores unresolved positions as gaps)
      - map 'I' → 'G'      (poly-inosine samples from 7wv3/7wv4 etc.)
      - keep 'A','C','G','U'
      - map anything else → 'N'
    """
    inosine_present = False
    out = []
    for ch in seq:
        if ch == '-':
            continue
        if ch == 'I':
            inosine_present = True
            out.append('G')
        elif ch in "ACGU":
            out.append(ch)
        else:
            out.append('N')
    return "".join(out), inosine_present


# ---------- structure classification ----------------------------------------


def _build_pair_table(dot_bracket: str) -> list[int]:
    """Return 0-indexed pair table: pair[i] = j if (i,j) paired else -1."""
    n = len(dot_bracket)
    pair = [-1] * n
    stack: list[int] = []
    for i, c in enumerate(dot_bracket):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if not stack:
                raise ValueError(f"unbalanced brackets at pos {i}")
            j = stack.pop()
            pair[i] = j
            pair[j] = i
    if stack:
        raise ValueError("unbalanced brackets: unmatched '('")
    return pair


def _loop_type_for_closing_pair(pair: list[int], i: int, j: int) -> str:
    """Classify the loop closed by the pair (i, j).

    Walks positions i+1 .. j-1 and counts how many NESTED pairs are adjacent
    to this loop (pairs whose open bracket is at some k with i<k<j and whose
    close bracket is at pair[k] < j).

      - 0 nested pairs  → hairpin loop
      - 1 nested pair   → interior loop (covers symmetric internal loops AND
                          asymmetric bulges — both have 2 closing pairs on
                          the loop boundary)
      - ≥2 nested pairs → multiloop (the "junction" bucket)
    """
    k = i + 1
    nested = 0
    while k < j:
        if pair[k] > k:
            nested += 1
            k = pair[k] + 1
        else:
            k += 1
    if nested == 0:
        return "hairpin"
    if nested == 1:
        return "interior"
    return "multiloop"


def classify_structure(dot_bracket: str) -> dict:
    """Compute fraction of nucleotides in each element type.

    Categories:
      - paired     : '(' or ')'
      - hairpin    : unpaired, inside a closing pair with 0 nested pairs
      - interior   : unpaired, inside a closing pair with 1 nested pair
                     (symmetric interior loops AND asymmetric bulges)
      - multiloop  : unpaired, inside a closing pair with ≥2 nested pairs
      - external   : unpaired, not inside any closing pair (5'/3' dangles)

    Fractions sum to ~1.0 (rounding).
    """
    n = len(dot_bracket)
    if n == 0:
        return None
    pair = _build_pair_table(dot_bracket)

    # Pre-compute loop type for every closing pair (i, j) with i < j.
    loop_type: dict[tuple[int, int], str] = {}
    for i in range(n):
        j = pair[i]
        if j > i:
            loop_type[(i, j)] = _loop_type_for_closing_pair(pair, i, j)

    # Walk L→R, maintain stack of currently-open pairs; unpaired positions
    # are classified by the innermost enclosing pair's loop type.
    counts: Counter[str] = Counter()
    stack: list[tuple[int, int]] = []
    for k in range(n):
        while stack and stack[-1][1] < k:
            stack.pop()
        if pair[k] >= 0:
            counts["paired"] += 1
            if pair[k] > k:
                stack.append((k, pair[k]))
        else:
            if not stack:
                counts["external"] += 1
            else:
                counts[loop_type[stack[-1]]] += 1

    return {
        "paired_frac":    round(counts["paired"]    / n, 4),
        "hairpin_frac":   round(counts["hairpin"]   / n, 4),
        "interior_frac":  round(counts["interior"]  / n, 4),
        "multiloop_frac": round(counts["multiloop"] / n, 4),
        "external_frac":  round(counts["external"]  / n, 4),
    }


# ---------- folding wrapper + cache -----------------------------------------


SS_STATUS_DONE = "done"
SS_STATUS_SKIPPED_TOO_LONG = "skipped_too_long"
SS_STATUS_SKIPPED_EMPTY = "skipped_empty"
SS_STATUS_ERROR = "error"
# "pending" is represented by absence of `ss_status` (or an explicit null).
_SS_RESUME_SKIP = {SS_STATUS_DONE, SS_STATUS_ERROR, SS_STATUS_SKIPPED_EMPTY}


def fold_rna(seq: str, max_len: int) -> tuple[dict, dict | None, str]:
    """Run RNAfold MFE on `seq` and classify.

    Returns (pred_dict, composition_dict_or_None, ss_status_string).
    pred_dict is always populated; composition is None when structure is None.
    ss_status is one of SS_STATUS_DONE / SKIPPED_TOO_LONG / SKIPPED_EMPTY /
    ERROR — always mirrored into the sample JSON's `rna.features.ss_status`
    so downstream filtering doesn't need to inspect the nested pred dict.
    """
    _require_viennarna()
    cleaned, inosine = sanitize_sequence(seq)
    pred = {
        "structure": None,
        "mfe": None,
        "fold_sequence": cleaned,
        "sanitized_from_inosine": inosine,
        "skipped_reason": None,
    }

    if not cleaned:
        pred["skipped_reason"] = "empty_after_sanitize"
        return pred, None, SS_STATUS_SKIPPED_EMPTY
    if len(cleaned) > max_len:
        pred["skipped_reason"] = f"too_long_{len(cleaned)}_nt"
        return pred, None, SS_STATUS_SKIPPED_TOO_LONG

    try:
        fc = RNA.fold_compound(cleaned)
        structure, mfe = fc.mfe()
    except Exception as e:
        pred["skipped_reason"] = f"rnafold_error: {type(e).__name__}: {e}"
        return pred, None, SS_STATUS_ERROR

    pred["structure"] = structure
    pred["mfe"] = round(float(mfe), 3)
    try:
        composition = classify_structure(structure)
    except Exception as e:
        pred["skipped_reason"] = f"classify_error: {type(e).__name__}: {e}"
        return pred, None, SS_STATUS_ERROR
    return pred, composition, SS_STATUS_DONE


def _seq_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode("ascii")).hexdigest()


# ---------- main -----------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1.3b: RNA secondary structure prediction (server)."
    )
    parser.add_argument("--processed-dir", type=Path, required=True,
                        help="root containing samples/*.json and index.csv")
    parser.add_argument("--stats-dir", type=Path, required=True,
                        help="where rna_secondary_structure_report.md goes")
    parser.add_argument("--max-length", type=int, default=4000,
                        help="skip sequences longer than this (nt). "
                             "RNAfold is O(n^3); 4000 nt ≈ ~1 min/seq.")
    parser.add_argument("--force", action="store_true",
                        help="refold even if secondary_structure_pred is already set")
    parser.add_argument("--limit", type=int, default=0,
                        help="process only first N samples (debug)")
    parser.add_argument("--sample-ids", type=str, default="",
                        help="comma-separated sample_ids to process (debug)")
    args = parser.parse_args()

    _require_viennarna()
    from tqdm import tqdm  # deferred so --selftest doesn't need it

    samples_dir = args.processed_dir / "samples"
    index_csv = args.processed_dir / "index.csv"
    args.stats_dir.mkdir(parents=True, exist_ok=True)

    if not index_csv.exists():
        sys.exit(f"index.csv not found at {index_csv} — stages 1.2 / 1.3a first")

    with index_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.sample_ids:
        wanted = {s.strip() for s in args.sample_ids.split(",") if s.strip()}
        rows = [r for r in rows if r["sample_id"] in wanted]
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} sample rows from {index_csv}")

    totals = {
        "n_rows": len(rows),
        "n_folded": 0,
        "n_skipped_resume_done": 0,
        "n_rechecked_too_long": 0,
        "n_skipped_too_long": 0,
        "n_skipped_empty": 0,
        "n_errors": 0,
        "n_cache_hits": 0,
        "n_inosine_sanitized": 0,
    }
    tier_folded: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    comp_sums: Counter[str] = Counter()
    comp_n = 0
    cache: dict[str, tuple[dict, dict | None, str]] = {}

    for row in tqdm(rows, desc="rnafold", mininterval=1.0):
        sid = row["sample_id"]
        tier = row["quality_tier"]
        filename = sid_to_filename(sid) + ".json"
        sample_path = samples_dir / filename

        with sample_path.open("r", encoding="utf-8") as f:
            sample = json.load(f)

        rna = sample["rna"]
        features = rna.setdefault("features", {})
        prior_status = features.get("ss_status")

        # Resume: skip terminal states ("done" / "error" / "skipped_empty").
        # Re-check "skipped_too_long" because --max-length may have changed.
        if not args.force and prior_status in _SS_RESUME_SKIP:
            totals["n_skipped_resume_done"] += 1
            status_counts[prior_status] += 1
            continue
        if prior_status == SS_STATUS_SKIPPED_TOO_LONG:
            totals["n_rechecked_too_long"] += 1

        seq = rna.get("sequence", "")
        cleaned_seq, _ = sanitize_sequence(seq)
        cache_key = _seq_hash(cleaned_seq) if cleaned_seq else None

        if cache_key and cache_key in cache:
            pred, composition, ss_status = cache[cache_key]
            totals["n_cache_hits"] += 1
        else:
            pred, composition, ss_status = fold_rna(seq, args.max_length)
            if cache_key:
                cache[cache_key] = (pred, composition, ss_status)

        features["ss_status"] = ss_status
        features["secondary_structure_pred"] = pred
        features["structure_composition"] = composition

        status_counts[ss_status] += 1
        if pred.get("sanitized_from_inosine"):
            totals["n_inosine_sanitized"] += 1

        if ss_status == SS_STATUS_DONE:
            totals["n_folded"] += 1
            tier_folded[tier] += 1
            if composition is not None:
                for k, v in composition.items():
                    comp_sums[k] += v
                comp_n += 1
        elif ss_status == SS_STATUS_SKIPPED_TOO_LONG:
            totals["n_skipped_too_long"] += 1
        elif ss_status == SS_STATUS_SKIPPED_EMPTY:
            totals["n_skipped_empty"] += 1
        else:  # SS_STATUS_ERROR
            totals["n_errors"] += 1

        tmp_path = sample_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, sample_path)

    # Report
    lines = [
        "# Stage 1.3b — RNA Secondary Structure Report",
        "",
        f"- Total samples scanned: **{totals['n_rows']}**",
        f"- Samples actually folded this run (ss_status=done): **{totals['n_folded']}**",
        f"- Resume-skipped (ss_status in {{done, error, skipped_empty}}): **{totals['n_skipped_resume_done']}**",
        f"- Re-checked samples previously skipped as too_long (a larger --max-length may rescue them): **{totals['n_rechecked_too_long']}**",
        f"- Marked skipped_too_long this run (>{args.max_length} nt): **{totals['n_skipped_too_long']}**",
        f"- Marked skipped_empty this run: **{totals['n_skipped_empty']}**",
        f"- Marked error this run: **{totals['n_errors']}**",
        f"- Cache hits (sequence hash): **{totals['n_cache_hits']}**",
        f"- Samples sanitized poly-inosine → G: **{totals['n_inosine_sanitized']}**",
        "",
        "## Final ss_status distribution (all samples)",
        "",
    ]
    for s in (SS_STATUS_DONE, SS_STATUS_SKIPPED_TOO_LONG,
              SS_STATUS_SKIPPED_EMPTY, SS_STATUS_ERROR):
        lines.append(f"- {s}: {status_counts.get(s, 0)}")
    missing = totals["n_rows"] - sum(status_counts.values())
    if missing:
        lines.append(f"- pending (not yet processed): {missing}")

    lines += ["", "## Successfully folded samples by tier", ""]
    for t in ("strict", "standard", "low", "discard"):
        lines.append(f"- {t}: {tier_folded.get(t, 0)}")

    if comp_n:
        lines += ["", "## Mean secondary-structure composition (n={0})".format(comp_n), ""]
        for k in ("paired_frac", "hairpin_frac", "interior_frac",
                  "multiloop_frac", "external_frac"):
            avg = comp_sums[k] / comp_n
            lines.append(f"- {k}: {avg:.3f}")

    lines += [
        "",
        "## Notes",
        "",
        "- `rna.features.ss_status` is the canonical state flag on every sample:",
        "  - `done`: fold succeeded; both `secondary_structure_pred.structure` and `structure_composition` are non-null",
        "  - `skipped_too_long`: sequence > `--max-length`; **it will be re-run the next time max-length is raised**",
        "  - `skipped_empty`: sequence is empty after sanitize (extremely rare); not retried",
        "  - `error`: RNAfold or the classifier raised; not retried (unless --force)",
        "  - field missing / `ss_status == null`: pending, never processed by 1.3b",
        "- `secondary_structure_pred.skipped_reason` carries a finer-grained reason string (for example `too_long_4823_nt`)",
        "- For poly-inosine samples (7wv3 / 7wv4 and others, the 29 flagged in the stage 1.3a report) `fold_sequence` is the version with `I` replaced by `G`; `sanitized_from_inosine` is true so downstream code can tell",
        "- Classification rule: interior covers both symmetric internal loops and asymmetric bulges (in the secondary-structure graph they are all loops with 2 closing pairs); junction = multiloop = ≥3 closing pairs",
    ]
    report_path = args.stats_dir / "rna_secondary_structure_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"Folded         : {totals['n_folded']}")
    print(f"Resume-skipped : {totals['n_skipped_resume_done']}")
    print(f"Too long       : {totals['n_skipped_too_long']}")
    print(f"Errors         : {totals['n_errors']}")
    print(f"Cache hits     : {totals['n_cache_hits']}")
    print(f"Inosine fixes  : {totals['n_inosine_sanitized']}")
    print(f"Report         : {report_path}")


# ---------- self-test (doesn't import RNA; runs on any machine) -------------


def _selftest():
    """Sanity check the classifier — runs without ViennaRNA.

    Invoke via: python rna_secondary_structure.py --selftest
    """
    # hairpin: ((...))
    c = classify_structure("((...))")
    assert c["paired_frac"] == round(4/7, 4), c
    assert c["hairpin_frac"] == round(3/7, 4), c
    # external only
    c = classify_structure(".....")
    assert c["external_frac"] == 1.0, c
    # multiloop:  (..(...)..(...)..)
    c = classify_structure("(..(...)..(...)..)")
    assert c["multiloop_frac"] > 0, c
    assert c["hairpin_frac"] > 0, c
    # interior loop: (..(...)..)
    c = classify_structure("(..(...)..)")
    assert c["interior_frac"] > 0, c
    # sanitize_sequence
    cleaned, ino = sanitize_sequence("ACGUI-NX")
    assert cleaned == "ACGUGNN" and ino is True, (cleaned, ino)
    cleaned, ino = sanitize_sequence("AUGC")
    assert cleaned == "AUGC" and ino is False
    # sid_to_filename round-trip
    assert sid_to_filename("3j46_y_1") == "3j46_-y_1"
    assert sid_to_filename("3j46_Y_1") == "3j46_Y_1"
    assert sid_to_filename("8ppl_Aj_A2") == "8ppl_A-j_A2"
    print("selftest: OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
