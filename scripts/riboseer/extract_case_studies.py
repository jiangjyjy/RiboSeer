#!/usr/bin/env python3
"""Pick the best-matching test samples for the paper's Section 5.9 case
studies (before = best single tool, after = RiboSeer full system).

Three cases, each with its own selection rubric (see the per-case
``score_case_*`` helpers):

* Case 1 — **RRM + single-stranded / stem-loop**: SCOPE family ``RRM``,
  rna_context ``single-strand``/``stem-loop``, the three mandatory Cat-A
  tools (boltz2 / chai1 / rosettafold2na) agree on their binding gate
  (mean pairwise Jaccard > 0.5), POLISH action = ``extend``, and RiboSeer
  beats the best single tool on Pearson R.
* Case 2 — **multi-domain + tool disagreement**: SCOPE family ``Multi``
  (or GT residues split across ≥ 2 discontiguous sequence segments),
  Cat-A mean pairwise Jaccard < 0.3, POLISH action = ``relocate``, and
  RiboSeer beats *every* single tool.
* Case 3 — **junction / long RNA + large interface**: RNA length > 60 nt
  (or SCOPE rna_context ``junction``), GT binding-residue count above the
  test-set mean, and a large RiboSeer-over-best-tool Pearson R gain.

No case is forced to match perfectly. Each candidate is scored by how
many of its criteria it satisfies (criteria are reported pass/fail), so
when the perfect sample doesn't exist the closest one still surfaces and
the output flags which criteria were relaxed. The script prints the
**top-3 candidates per case** for the user to choose from.

All metric math reuses the existing evaluation modules so the numbers
match the paper tables exactly:

* per-residue Pearson R — ``table04_main_results`` convention
  (``_pearson`` over ``resolved_residues``);
* pocket-level DCC / DCA / IoU — ``table05_pocket_geometry``
  (top-k cluster centroid distance);
* DockQ (optional, ``--usalign-bin``) — ``table06_complex_quality``
  (only computed for the printed candidates, so it stays cheap).

Usage
-----
::

    python scripts/riboseer/extract_case_studies.py \\
        --data-dir        data/processed_quality \\
        --step4-dir       data/batch_test_v7/step4 \\
        --predictions-dir data/batch_test_v7/fullsystem_predictions \\
        --scope-dir       data/batch_test_v7/scope_profiles_llm \\
        --polish-dir      data/batch_test_v7/polish_actions_llm_v2 \\
        --raw-dir         data/raw \\
        --split-file      data/processed_quality/splits_tmscore_035/test.txt

Optional
--------
``--usalign-bin PATH``  also compute DockQ for the printed candidates
(needs the USalign binary + the DockQ python package on the server);
omitted → DockQ cells read ``n/a``.
``--work-dir DIR``  scratch dir for the auto-built reference complex PDBs
(DockQ only; default ``<predictions-dir>/../case_study_refs``).
``--top-k N``  how many candidates to print per case (default 3).
``--output PATH``  also dump the full candidate table as JSON.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step5_fusion.data_collector import load_sample_json  # noqa: E402
from step5_fusion.enriched_fusion import _pearson  # noqa: E402
from step5_fusion.prediction_io import load_predictions_dir  # noqa: E402
from scripts.tables.table05_pocket_geometry import (  # noqa: E402
    _read_last_jsonl_record, _tool_scores, _resolve_structure_path,
    extract_chain_coords, evaluate_sample_method,
)

# Mandatory Cat-A co-folders — the three whose binding-gate agreement
# defines Case 1 / disagreement defines Case 2.
CAT_A_TOOLS = ("boltz2", "chai1", "rosettafold2na")
CASE1_CONTEXTS = {"single-strand", "stem-loop"}


# ---- small helpers -------------------------------------------------------


def _load_sample_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"split file not found: {path}")
    return [ln.split()[0] for ln in path.read_text(encoding="utf-8")
            .splitlines() if ln.strip() and not ln.strip().startswith("#")]


def _load_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def jaccard(a: set, b: set) -> Optional[float]:
    """|A∩B| / |A∪B|; None when both sets are empty (undefined)."""
    u = a | b
    if not u:
        return None
    return len(a & b) / len(u)


def mean_pairwise_jaccard(gates: dict[str, set]) -> Optional[float]:
    """Mean Jaccard over the C(n,2) pairs of tools that produced a
    non-empty binding gate. None if < 2 tools have a gate."""
    ids = [t for t in CAT_A_TOOLS if gates.get(t)]
    if len(ids) < 2:
        return None
    vals = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            jc = jaccard(gates[ids[i]], gates[ids[j]])
            if jc is not None:
                vals.append(jc)
    return statistics.fmean(vals) if vals else None


def count_segments(residues: list[int], max_gap: int = 3) -> int:
    """Number of discontiguous sequence runs in a residue set (gap >
    ``max_gap`` starts a new run). Proxy for 'binding site spread across
    multiple domains'."""
    rs = sorted(set(int(r) for r in residues))
    if not rs:
        return 0
    segs = 1
    for prev, cur in zip(rs, rs[1:]):
        if cur - prev > max_gap:
            segs += 1
    return segs


def _pearson_over(scores: dict[int, float], residues: list[int],
                  gt_set: set[int]) -> Optional[float]:
    """Per-residue Pearson R of a score map vs the binary GT vector, over
    ``residues`` — same construction as ``table04_main_results`` (missing
    residues score 0; undefined when GT or prediction has no variance)."""
    if len(residues) < 2:
        return None
    pred = [float(scores.get(r, 0.0)) for r in residues]
    gt = [1.0 if r in gt_set else 0.0 for r in residues]
    if (max(pred) - min(pred)) <= 1e-8 or (max(gt) - min(gt)) <= 1e-8:
        return None
    pr = _pearson(pred, gt)
    return None if pr is None else round(pr, 4)


# ---- per-sample collection ----------------------------------------------


def collect_sample(
    sid: str, *, data_dir: Path, step4_dir: Path, raw_dir: Path,
    scope_dir: Path, polish_dir: Path,
    fs_preds: dict[str, dict[int, float]],
    hit_threshold: float = 4.0,
) -> Optional[dict]:
    """Build the full feature + metric record for one sample, or None if
    the sample lacks the data needed to evaluate it (no step4, no sample
    JSON, no GT, or no RiboSeer prediction)."""
    s4 = _read_last_jsonl_record(step4_dir / f"{sid}.jsonl")
    if s4 is None:
        return None
    sample = load_sample_json(data_dir, sid)
    if sample is None:
        return None
    prot = sample.get("protein") or {}
    rna = sample.get("rna") or sample.get("ligand_rna") or {}
    inter = sample.get("interaction") or {}

    gt_residues = [int(r) for r in (inter.get("binding_protein_residues")
                                    or [])]
    gt_set = set(gt_residues)
    if not gt_set:
        return None

    residues = [int(r) for r in (prot.get("resolved_residues") or [])]
    if not residues:
        plen0 = prot.get("length") or len(prot.get("sequence") or "")
        residues = list(range(1, int(plen0 or 0) + 1))
    if len(residues) < 2:
        return None

    lp = int(prot.get("length") or len(prot.get("sequence") or "") or 0)
    lr = int(rna.get("length") or len(rna.get("sequence") or "") or 0)

    # ---- structure coords for DCC (shared across methods) ----
    ca: dict[int, tuple[float, float, float]] = {}
    rna_atoms: list[tuple[float, float, float]] = []
    src_pdb = sample.get("source_pdb")
    prot_chain = prot.get("chain_id")
    rna_chain = rna.get("chain_id")
    if src_pdb and prot_chain and rna_chain:
        struct = _resolve_structure_path(raw_dir, str(src_pdb))
        if struct is not None:
            try:
                ca, rna_atoms = extract_chain_coords(
                    struct, str(prot_chain), str(rna_chain))
            except (FileNotFoundError, ValueError):
                ca, rna_atoms = {}, []

    def _dcc(scores: dict[int, float]) -> Optional[float]:
        if not ca:
            return None
        pack = evaluate_sample_method(
            scores=scores, gt_residues=gt_residues, ca=ca,
            rna_atoms=rna_atoms, hit_threshold=hit_threshold)
        return None if pack is None else pack.get("dcc")

    # ---- per-tool metrics + Cat-A binding gates ----
    preds = s4.get("predictions") or []
    tools: dict[str, dict] = {}
    gates: dict[str, set] = {}
    for p in preds:
        tid = p.get("tool_id")
        if not tid or not p.get("success"):
            continue
        if tid in CAT_A_TOOLS:
            gates[tid] = set(int(r) for r in
                             (p.get("binding_protein_residues") or []))
        scores = _tool_scores(p)
        if not scores:
            continue
        pr = _pearson_over(scores, residues, gt_set)
        tools[tid] = {
            "pearson_r": pr,
            "dcc": _dcc(scores),
            "predicted_structure_path": p.get("predicted_structure_path"),
            "is_cat_a": tid in CAT_A_TOOLS,
        }

    # best single tool = highest per-residue Pearson R (primary metric)
    scored = {t: d for t, d in tools.items() if d["pearson_r"] is not None}
    if not scored:
        return None
    best_tool = max(scored, key=lambda t: scored[t]["pearson_r"])
    best_pr = scored[best_tool]["pearson_r"]
    max_tool_pr = max(d["pearson_r"] for d in scored.values())

    # ---- RiboSeer full-system metrics ----
    fs = fs_preds.get(sid)
    if not fs:
        return None
    ribo_pr = _pearson_over(fs, residues, gt_set)
    ribo_dcc = _dcc(fs)
    if ribo_pr is None:
        return None

    scope = _load_json(scope_dir / f"{sid}.json") or {}
    polish = _load_json(polish_dir / f"{sid}.json") or {}

    catA_jaccard = mean_pairwise_jaccard(gates)
    delta_pr = round(ribo_pr - best_pr, 4)
    best_dcc = scored[best_tool]["dcc"]
    delta_dcc = (round(best_dcc - ribo_dcc, 4)
                 if (best_dcc is not None and ribo_dcc is not None)
                 else None)

    return {
        "sample_id": sid,
        "source_pdb": src_pdb,
        "scope_family": scope.get("protein_family"),
        "rna_context": scope.get("rna_context"),
        "difficulty": scope.get("difficulty"),
        "scope_source": scope.get("source"),
        "Lp": lp, "Lr": lr,
        "n_gt": len(gt_set),
        "gt_segments": count_segments(gt_residues),
        "catA_jaccard": catA_jaccard,
        "polish_action": polish.get("action"),
        "polish_residues": polish.get("residues") or [],
        "polish_target_residues": polish.get("target_residues") or [],
        "polish_source": polish.get("source"),
        "best_tool": best_tool,
        "best_tool_is_cat_a": scored[best_tool]["is_cat_a"],
        "best_tool_pearson": best_pr,
        "best_tool_dcc": best_dcc,
        "max_tool_pearson": max_tool_pr,
        "riboseer_pearson": ribo_pr,
        "riboseer_dcc": ribo_dcc,
        "delta_pearson": delta_pr,
        "delta_dcc": delta_dcc,
        "beats_best": ribo_pr > best_pr,
        "beats_all": ribo_pr > max_tool_pr,
        # carried for optional DockQ on the printed candidates
        "_predictions": preds,
        "_fs": fs,
        "_prot_chain": prot_chain,
        "_rna_chain": rna_chain,
    }


# ---- case scoring --------------------------------------------------------
#
# Each scorer returns (score, criteria) where ``criteria`` is an ordered
# list of (label, passed) pairs. ``score`` = #criteria passed + a small
# improvement-magnitude tiebreak, so a perfect match sorts first and the
# closest partial match still surfaces. Candidates that show no
# improvement (RiboSeer not better) are dropped from a case entirely.


def _improve_tiebreak(rec: dict) -> float:
    """Small [0, ~0.5) bonus rewarding a larger Pearson R gain — only a
    tiebreak between equal-criteria candidates, never enough to outrank a
    candidate that passes more criteria."""
    d = rec.get("delta_pearson") or 0.0
    return max(0.0, min(d, 0.5))


def score_case1(rec: dict) -> Optional[tuple[float, list]]:
    if not rec["beats_best"]:
        return None
    crit = [
        ("SCOPE family = RRM", rec["scope_family"] == "RRM"),
        ("rna_context single-strand/stem-loop",
         rec["rna_context"] in CASE1_CONTEXTS),
        ("Cat-A mean Jaccard > 0.5",
         rec["catA_jaccard"] is not None and rec["catA_jaccard"] > 0.5),
        ("POLISH action = extend", rec["polish_action"] == "extend"),
        ("RiboSeer > best single tool (Pearson)", rec["beats_best"]),
    ]
    return sum(p for _, p in crit) + _improve_tiebreak(rec), crit


def score_case2(rec: dict) -> Optional[tuple[float, list]]:
    if not rec["beats_best"]:
        return None
    multi = (rec["scope_family"] == "Multi") or (rec["gt_segments"] >= 2)
    crit = [
        ("Multi family OR GT in ≥2 segments", multi),
        ("Cat-A mean Jaccard < 0.3 (disagreement)",
         rec["catA_jaccard"] is not None and rec["catA_jaccard"] < 0.3),
        ("POLISH action = relocate", rec["polish_action"] == "relocate"),
        ("RiboSeer > ALL single tools", rec["beats_all"]),
    ]
    return sum(p for _, p in crit) + _improve_tiebreak(rec), crit


def score_case3(rec: dict, gt_mean: float) -> Optional[tuple[float, list]]:
    if not rec["beats_best"]:
        return None
    long_rna = rec["Lr"] > 60 or rec["rna_context"] == "junction"
    crit = [
        ("RNA length > 60 nt OR junction context", long_rna),
        ("SCOPE rna_context = junction", rec["rna_context"] == "junction"),
        ("GT binding residues > test-set mean "
         f"({gt_mean:.1f})", rec["n_gt"] > gt_mean),
        ("RNA length > 70 nt (neighborhood/streak signal)",
         rec["Lr"] > 70),
        ("RiboSeer > best single tool (Pearson)", rec["beats_best"]),
    ]
    # Case 3's whole point is a large gain from G3 neighborhood features,
    # so weight the improvement tiebreak a touch higher here.
    return sum(p for _, p in crit) + _improve_tiebreak(rec), crit


# ---- optional DockQ for the printed candidates --------------------------


def dockq_for_candidate(rec: dict, *, raw_dir: Path, work_dir: Path,
                        usalign_bin: str, timeout: float) -> dict:
    """Best-single-tool DockQ + RiboSeer-consensus DockQ for one printed
    candidate. Returns ``{}`` (silently) on any failure — DockQ is a
    nice-to-have, never a hard requirement. Reuses
    ``table06_complex_quality`` so the numbers match Table 6."""
    try:
        from scripts.tables.table06_complex_quality import (
            build_ref_complex_pdb, evaluate_one, select_consensus_tool,
        )
    except Exception:  # noqa: BLE001
        return {}
    src_pdb = rec.get("source_pdb")
    pc, rc = rec.get("_prot_chain"), rec.get("_rna_chain")
    if not (src_pdb and pc and rc):
        return {}
    try:
        ref = build_ref_complex_pdb(
            raw_dir, str(src_pdb), str(pc), str(rc),
            work_dir / f"{rec['sample_id']}_ref.pdb")
    except (FileNotFoundError, ValueError):
        return {}

    preds = rec.get("_predictions") or []

    def _path_for(tid: str) -> Optional[Path]:
        p = next((pp for pp in preds if pp.get("tool_id") == tid
                  and pp.get("success")
                  and pp.get("predicted_structure_path")), None)
        if p is None:
            return None
        sp = Path(str(p["predicted_structure_path"]))
        return sp if sp.is_file() else None

    out: dict = {}
    # best single tool (only Cat-A tools emit a complex)
    if rec.get("best_tool_is_cat_a"):
        bp = _path_for(rec["best_tool"])
        if bp is not None:
            try:
                out["best_tool_dockq"] = evaluate_one(
                    pred_path=bp, ref_path=ref,
                    usalign_bin=usalign_bin, timeout=timeout).get("dockq")
            except Exception:  # noqa: BLE001
                pass
    # RiboSeer consensus pick
    consensus = select_consensus_tool(preds, rec.get("_fs") or {})
    if consensus is not None:
        cp = _path_for(consensus)
        if cp is not None:
            try:
                out["consensus_tool"] = consensus
                out["consensus_dockq"] = evaluate_one(
                    pred_path=cp, ref_path=ref,
                    usalign_bin=usalign_bin, timeout=timeout).get("dockq")
            except Exception:  # noqa: BLE001
                out.pop("consensus_tool", None)
    return out


# ---- output --------------------------------------------------------------


def _fmt(v, suffix: str = "") -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}{suffix}"
    return f"{v}{suffix}"


def _print_candidate(rank: int, rec: dict, crit: list, dockq: dict) -> None:
    n_pass = sum(p for _, p in crit)
    relaxed = [lbl for lbl, p in crit if not p]
    print(f"  [{rank}] {rec['sample_id']}  "
          f"(criteria {n_pass}/{len(crit)} matched)")
    print(f"      SCOPE profile: family={rec['scope_family']}, "
          f"rna_context={rec['rna_context']}, "
          f"difficulty={rec['difficulty']} (src={rec['scope_source']})")
    print(f"      RNA length: {rec['Lr']} nt, "
          f"Protein length: {rec['Lp']} aa")
    print(f"      GT binding residues |Bp|: {rec['n_gt']} "
          f"(in {rec['gt_segments']} sequence segment(s))")
    print(f"      Cat-A mean pairwise Jaccard: "
          f"{_fmt(rec['catA_jaccard'])}")
    print(f"      Best single tool: {rec['best_tool']}"
          f"{' [Cat-A]' if rec['best_tool_is_cat_a'] else ''}")
    print(f"        Pearson R: {_fmt(rec['best_tool_pearson'])}")
    print(f"        DCC:       {_fmt(rec['best_tool_dcc'], ' A')}")
    if "best_tool_dockq" in dockq:
        print(f"        DockQ:     {_fmt(dockq['best_tool_dockq'])}")
    print(f"      RiboSeer (full system):")
    print(f"        Pearson R: {_fmt(rec['riboseer_pearson'])}")
    print(f"        DCC:       {_fmt(rec['riboseer_dcc'], ' A')}")
    if "consensus_dockq" in dockq:
        print(f"        DockQ:     {_fmt(dockq['consensus_dockq'])} "
              f"(consensus pick = {dockq.get('consensus_tool')})")
    pres = rec['polish_residues']
    pres_s = (str(pres[:15]) + (" ..." if len(pres) > 15 else ""))
    print(f"      POLISH action: {rec['polish_action']} "
          f"on residues {pres_s} (src={rec['polish_source']})")
    ddcc = rec['delta_dcc']
    ddcc_s = "n/a" if ddcc is None else f"-{ddcc:.4f} A"
    print(f"      Improvement: Pearson R {rec['delta_pearson']:+.4f}, "
          f"DCC {ddcc_s}")
    if relaxed:
        print(f"      [relaxed criteria]: {'; '.join(relaxed)}")
    print()


# ---- driver --------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="processed_quality root (with samples/).")
    p.add_argument("--step4-dir", type=Path, required=True,
                   help="step4 JSONL dir (one .jsonl per sample).")
    p.add_argument("--predictions-dir", type=Path, required=True,
                   help="full-system per-sample predictions <sid>.json.")
    p.add_argument("--scope-dir", type=Path, required=True,
                   help="SCOPE profile JSON dir (<sid>.json).")
    p.add_argument("--polish-dir", type=Path, required=True,
                   help="POLISH action JSON dir (<sid>.json).")
    p.add_argument("--raw-dir", type=Path, required=True,
                   help="raw PDB / mmCIF dir (for DCC + DockQ).")
    p.add_argument("--split-file", type=Path, required=True,
                   help="test split (one sample_id per line).")
    p.add_argument("--top-k", type=int, default=3,
                   help="candidates to print per case (default 3).")
    p.add_argument("--hit-threshold", type=float, default=4.0,
                   help="DCC cutoff in A for pocket clustering (default 4.0).")
    p.add_argument("--usalign-bin", type=str, default=None,
                   help="USalign binary; when set, DockQ is computed for "
                        "the printed candidates (else 'n/a').")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="scratch dir for DockQ reference complexes "
                        "(default <predictions-dir>/../case_study_refs).")
    p.add_argument("--dockq-timeout", type=float, default=120.0,
                   help="per-structure USalign/DockQ timeout (s).")
    p.add_argument("--output", type=Path, default=None,
                   help="optional JSON dump of the full candidate table.")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for d in (args.data_dir, args.step4_dir, args.raw_dir):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 1
    try:
        sample_ids = _load_sample_ids(args.split_file)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    fs_preds = load_predictions_dir(args.predictions_dir)
    if not fs_preds:
        print(f"ERROR: no full-system predictions under "
              f"{args.predictions_dir}", file=sys.stderr)
        return 1
    print(f"loaded {len(fs_preds)} full-system predictions; "
          f"{len(sample_ids)} samples in split")

    records: list[dict] = []
    skipped = 0
    for sid in sample_ids:
        rec = collect_sample(
            sid, data_dir=args.data_dir, step4_dir=args.step4_dir,
            raw_dir=args.raw_dir, scope_dir=args.scope_dir,
            polish_dir=args.polish_dir, fs_preds=fs_preds,
            hit_threshold=args.hit_threshold)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)
    print(f"evaluated {len(records)} samples ({skipped} skipped: missing "
          f"step4 / sample / GT / RiboSeer prediction)")

    if not records:
        print("ERROR: no evaluable samples", file=sys.stderr)
        return 1

    gt_mean = statistics.fmean(r["n_gt"] for r in records)
    n_dcc = sum(1 for r in records if r["riboseer_dcc"] is not None)
    print(f"test-set mean GT |Bp| = {gt_mean:.1f}; "
          f"DCC available for {n_dcc}/{len(records)} samples")
    print(f"missing SCOPE profiles: "
          f"{sum(1 for r in records if r['scope_family'] is None)}; "
          f"missing POLISH actions: "
          f"{sum(1 for r in records if r['polish_action'] is None)}")
    print()

    cases = [
        ("Case 1: RRM + Single-Stranded/Stem-loop "
         "(high Cat-A agreement, POLISH extend)",
         lambda r: score_case1(r)),
        ("Case 2: Multi-Domain + Tool Disagreement "
         "(low Cat-A Jaccard, POLISH relocate)",
         lambda r: score_case2(r)),
        ("Case 3: Junction/Long RNA + Large Interface "
         "(neighborhood/streak gain)",
         lambda r: score_case3(r, gt_mean)),
    ]

    dockq_enabled = args.usalign_bin is not None
    work_dir = args.work_dir or (args.predictions_dir.parent
                                 / "case_study_refs")
    if dockq_enabled:
        work_dir.mkdir(parents=True, exist_ok=True)
        print(f"DockQ enabled (USalign={args.usalign_bin}); refs in "
              f"{work_dir}")
        print()

    table_dump: dict = {}
    for title, scorer in cases:
        scored = []
        for r in records:
            res = scorer(r)
            if res is None:
                continue
            score, crit = res
            scored.append((score, r, crit))
        scored.sort(key=lambda t: -t[0])
        top = scored[:args.top_k]

        print("=" * 72)
        print(title)
        print("=" * 72)
        if not top:
            print("  (no candidate showed a RiboSeer improvement -- "
                  "nothing to report)\n")
            table_dump[title] = []
            continue
        if top[0][2] and sum(p for _, p in top[0][2]) < len(top[0][2]):
            print("  NOTE: no perfect match; showing closest candidates "
                  "(relaxed criteria flagged per row).\n")

        dumped = []
        for rank, (score, rec, crit) in enumerate(top, 1):
            dockq = ({} if not dockq_enabled else dockq_for_candidate(
                rec, raw_dir=args.raw_dir, work_dir=work_dir,
                usalign_bin=args.usalign_bin, timeout=args.dockq_timeout))
            _print_candidate(rank, rec, crit, dockq)
            dumped.append({
                "rank": rank,
                "match_score": round(score, 4),
                "criteria": [{"label": l, "passed": bool(pp)}
                             for l, pp in crit],
                **{k: v for k, v in rec.items()
                   if not k.startswith("_")},
                "dockq": dockq,
            })
        table_dump[title] = dumped

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(table_dump, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"wrote full candidate table -> {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
