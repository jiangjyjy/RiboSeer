"""Table 28 — Failure-mode analysis.

Finds RiboSeer's worst test samples (bottom 10% by per-sample Pearson R)
and auto-attributes each to one or more failure modes. A sample can hit
several modes, so the per-mode counts may sum to more than the number of
failures (and the percentages to >100%).

Pipeline reuse: per-sample Pearson R is computed exactly as in Tables
16-25 — ``collect_sample_data`` (GT + resolved mask) → ``riboseer_predict``
→ ``corr_on_eval_subset``. Bottom 10% = samples with R ≤ the 10th
percentile of the defined R values.

Six modes (detectors)
----------------------
1. **All Cat. A tools low confidence (avg pLDDT < 60)** — mean global
   pLDDT (``plddt_mean`` → mean of ``per_residue_confidence``) across the
   *present* Cat A structure tools (boltz2 / chai1 / rosettafold2na) < 60.
2. **Tool disagreement (mean Jaccard < 0.2)** — mean pairwise Jaccard of
   every successful tool's gate set (``binding_protein_residues``) < 0.2
   (needs ≥2 tools).
3. **Multi-domain interface, wrong domain selected** — SCOPE profile
   ``protein_family`` contains "multi", OR (fallback) the GT binding
   residues split into segments separated by a >50-residue gap.
4. **RNA secondary structure mis-predicted** — SCOPE ``rna_context`` is a
   hard minority class (junction / G-quadruplex), the structures tools
   predict worst.
5. **POLISH relocate action failed** — the sample's POLISH action (or any
   round's action) is ``relocate`` — the most aggressive op.
6. **Other or unattributed** — none of 1-5 fired.

Unreadable inputs (no step4 record, missing SCOPE/POLISH dir, no Cat A
tool) print a warning and simply don't fire their mode — never crash.

Run
---
::

    python scripts/tables/table27_failure_modes.py \\
        --data-dir   data/processed_quality \\
        --step4-dir  data/batch_test_v7/step4 \\
        --model-dir  data/enriched_v7_lgbm \\
        --split-file data/processed_quality/splits_tmscore_035/test.txt \\
        --scope-dir  data/batch_test_v7/scope_profiles_llm \\
        --polish-dir data/batch_test_v7/polish_actions_llm_v2

``--scope-dir`` / ``--polish-dir`` are optional; their modes are skipped
(with a warning) when not supplied.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from step5_fusion.data_collector import (  # noqa: E402
    read_jsonl_record, per_residue_to_int_dict,
)
from step5_fusion.enriched_fusion import (  # noqa: E402
    EnrichedFusion, TOOL_ORDER, _CAT_A, _canonical_tool_id, _global_plddt,
)
from scripts.riboseer.ablation_fusion_method import (  # noqa: E402
    SampleData, collect_sample_data, corr_on_eval_subset,
)
from scripts.tables.table15_rna_length import riboseer_predict  # noqa: E402
from step5_fusion.prediction_io import (  # noqa: E402
    load_predictions_dir, predictions_to_vector,
)

# (mode key, display label) in table row order.
MODES: list[tuple[str, str]] = [
    ("low_plddt", "All Cat. A tools low confidence (avg pLDDT < 60)"),
    ("disagreement", "Tool disagreement (mean Jaccard < 0.2)"),
    ("multi_domain", "Multi-domain interface, wrong domain selected"),
    ("rna_struct", "RNA secondary structure mis-predicted"),
    ("polish_relocate", "POLISH relocate action failed"),
    ("other", "Other or unattributed"),
]

PLDDT_THRESHOLD = 60.0
JACCARD_THRESHOLD = 0.2
DOMAIN_GAP = 50          # GT-segment gap (residues) → multi-domain fallback
# Hard RNA-context minority classes (tools predict these worst).
_HARD_RNA = ("junction", "quad")  # 'quad' matches 'g-quadruplex'


# ---------------------------------------------------------------------------
# step4 per-tool view (pLDDT + gate set), read fresh per failure sample
# ---------------------------------------------------------------------------


def read_step4_tools(step4_dir: Path, sid: str) -> dict[str, dict]:
    """``{canonical_tool_id: {'plddt': float, 'has_plddt': bool,
    'gate': set[int]}}`` for the successful tools of one sample."""
    rec = read_jsonl_record(step4_dir / f"{sid}.jsonl")
    out: dict[str, dict] = {}
    if rec is None:
        return out
    for p in rec.get("predictions") or []:
        if not p.get("success"):
            continue
        tid = _canonical_tool_id(p.get("tool_id") or "")
        conf = per_residue_to_int_dict(p.get("per_residue_confidence"))
        has_plddt = (p.get("plddt_mean") is not None) or bool(conf)
        gate: set[int] = set()
        for r in p.get("binding_protein_residues") or []:
            try:
                gate.add(int(r))
            except (TypeError, ValueError):
                continue
        out[tid] = {"plddt": _global_plddt(p, conf),
                    "has_plddt": has_plddt, "gate": gate}
    return out


# ---------------------------------------------------------------------------
# Mode detectors
# ---------------------------------------------------------------------------


def mean_pairwise_jaccard(sets: list[set]) -> Optional[float]:
    """Mean Jaccard over every unordered pair; None if < 2 sets."""
    if len(sets) < 2:
        return None
    js = []
    for a, b in combinations(sets, 2):
        union = len(a | b)
        js.append(len(a & b) / union if union > 0 else 0.0)
    return float(np.mean(js))


def detect_low_plddt(tools: dict[str, dict]) -> Optional[bool]:
    """True/False, or None if no Cat A tool's pLDDT is readable."""
    plddts = [t["plddt"] for tid, t in tools.items()
              if tid in _CAT_A and t["has_plddt"]]
    if not plddts:
        return None
    return float(np.mean(plddts)) < PLDDT_THRESHOLD


def detect_disagreement(tools: dict[str, dict]) -> Optional[bool]:
    """True/False over the successful tools' gate sets; None if < 2."""
    gates = [t["gate"] for t in tools.values()]
    mj = mean_pairwise_jaccard(gates)
    if mj is None:
        return None
    return mj < JACCARD_THRESHOLD


def gt_residues(sample: SampleData) -> list[int]:
    return [rid for rid, yy in zip(sample.residue_ids, sample.y)
            if yy > 0.5]


def detect_multi_domain(profile: Optional[dict], gt: list[int]) -> bool:
    fam = str((profile or {}).get("protein_family", "")).lower()
    if "multi" in fam:
        return True
    # Fallback: GT residues split into >50-gap segments.
    rs = sorted(gt)
    return any(rs[i + 1] - rs[i] > DOMAIN_GAP for i in range(len(rs) - 1))


def detect_rna_struct(profile: Optional[dict]) -> bool:
    ctx = str((profile or {}).get("rna_context", "")).lower()
    return any(key in ctx for key in _HARD_RNA)


def detect_polish_relocate(action: Optional[dict]) -> bool:
    if not action:
        return False
    if str(action.get("action", "")).lower() == "relocate":
        return True
    # iterative POLISH schema: any round's action.
    for rnd in action.get("rounds") or []:
        if str(rnd.get("action", "")).lower() == "relocate":
            return True
    return False


# ---------------------------------------------------------------------------
# Per-sample Pearson R (RiboSeer) + bottom-10% selection
# ---------------------------------------------------------------------------


def riboseer_per_sample_r(samples: list[SampleData],
                          model: Optional[EnrichedFusion],
                          preds: Optional[dict[str, dict[int, float]]] = None
                          ) -> dict[str, float]:
    """RiboSeer per-sample Pearson R. The per-residue vector comes from
    ``preds`` (pre-computed full-system predictions) when supplied, else
    from the loaded ``model``."""
    out: dict[str, float] = {}
    for s in samples:
        if preds is not None:
            vec = predictions_to_vector(preds.get(s.sid), s.residue_ids)
        else:
            vec = riboseer_predict(model, s.X)
        corr = corr_on_eval_subset(vec, s)
        if corr is not None and corr["pearson_r"] is not None:
            out[s.sid] = corr["pearson_r"]
    return out


def bottom_decile(per_r: dict[str, float]) -> tuple[float, list[str]]:
    """(threshold, [sample_ids with R ≤ 10th percentile])."""
    vals = np.asarray(list(per_r.values()), dtype=np.float64)
    thr = float(np.percentile(vals, 10))
    failed = sorted((sid for sid, r in per_r.items() if r <= thr),
                    key=lambda s: per_r[s])
    return thr, failed


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _load_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"split file not found: {path}")
    return [ln.split()[0] for ln in
            path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def _load_json_dir(directory: Optional[Path]) -> dict[str, dict]:
    """``{sid: parsed_json}`` from ``<dir>/<sid>.json`` (SCOPE / POLISH)."""
    out: dict[str, dict] = {}
    if directory is None or not directory.is_dir():
        return out
    for f in directory.glob("*.json"):
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def analyse(failed: list[str], samples_by_sid: dict[str, SampleData],
            step4_dir: Path, scope: dict[str, dict],
            polish: dict[str, dict],
            *, have_scope: bool, have_polish: bool
            ) -> tuple[dict[str, int], list[dict]]:
    """Returns (mode_counts, per-sample detail rows)."""
    counts = {k: 0 for k, _ in MODES}
    detail: list[dict] = []
    for sid in failed:
        tools = read_step4_tools(step4_dir, sid)
        if not tools:
            print(f"  WARN: no step4 tools for {sid}", file=sys.stderr)
        sample = samples_by_sid[sid]

        hits: list[str] = []
        if detect_low_plddt(tools) is True:
            hits.append("low_plddt")
        if detect_disagreement(tools) is True:
            hits.append("disagreement")
        if have_scope and detect_multi_domain(scope.get(sid),
                                              gt_residues(sample)):
            hits.append("multi_domain")
        elif not have_scope and detect_multi_domain(None,
                                                    gt_residues(sample)):
            # No SCOPE dir → fall back to the GT-geometry heuristic only.
            hits.append("multi_domain")
        if have_scope and detect_rna_struct(scope.get(sid)):
            hits.append("rna_struct")
        if have_polish and detect_polish_relocate(polish.get(sid)):
            hits.append("polish_relocate")

        if not hits:
            hits.append("other")
        for h in hits:
            counts[h] += 1
        detail.append({"sid": sid, "modes": hits})
    return counts, detail


def print_report(thr: float, failed: list[str], per_r: dict[str, float],
                 counts: dict[str, int], detail: list[dict]) -> None:
    n = len(failed)
    print("=== Table 28: Failure Mode Analysis ===")
    print(f"Bottom 10% threshold: Pearson R <= {thr:.3f} ({n} samples)")
    print()
    label_w = max(len(lbl) for _, lbl in MODES)
    hdr = f"{'Failure mode':<{label_w}s}  {'# samples':>9s}  {'% of failures':>13s}"
    print(hdr)
    print("-" * len(hdr))
    for key, lbl in MODES:
        c = counts[key]
        pct = (100.0 * c / n) if n else 0.0
        print(f"{lbl:<{label_w}s}  {c:>9d}  {pct:>12.1f}%")
    print()
    print("Failed samples detail:")
    print(f"  {'sample_id':<18s} {'Pearson_R':>9s}  modes")
    for d in detail:
        modes = "[" + ", ".join(d["modes"]) + "]"
        print(f"  {d['sid']:<18s} {per_r[d['sid']]:>9.3f}  {modes}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="processed dir (has samples/<sid>.json)")
    p.add_argument("--step4-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=None,
                   help="RiboSeer EnrichedFusion bundle; not needed when "
                        "--predictions-dir is given")
    p.add_argument("--predictions-dir", type=Path, default=None,
                   help="pre-computed full-system (on/on/on) predictions "
                        "<sid>.json; when set RiboSeer reads these instead "
                        "of loading the model")
    p.add_argument("--split-file", type=Path, required=True)
    p.add_argument("--scope-dir", type=Path, default=None,
                   help="SCOPE profile dir (<sid>.json); optional")
    p.add_argument("--polish-dir", type=Path, default=None,
                   help="POLISH action dir (<sid>.json); optional")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for d in (args.data_dir, args.step4_dir):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 1
    try:
        ids = _load_split(args.split_file)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    preds = None
    model = None
    if args.predictions_dir is not None:
        preds = load_predictions_dir(args.predictions_dir)
        if not preds:
            print(f"ERROR: no predictions under {args.predictions_dir}",
                  file=sys.stderr)
            return 1
        print(f"loaded {len(preds)} full-system predictions from "
              f"{args.predictions_dir}")
    else:
        if args.model_dir is None:
            print("ERROR: pass either --predictions-dir or --model-dir",
                  file=sys.stderr)
            return 1
        try:
            model = EnrichedFusion.load(args.model_dir)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: failed to load model from {args.model_dir}: {e}",
                  file=sys.stderr)
            return 1

    have_scope = args.scope_dir is not None and args.scope_dir.is_dir()
    have_polish = args.polish_dir is not None and args.polish_dir.is_dir()
    if args.scope_dir is not None and not have_scope:
        print(f"WARN: --scope-dir not found: {args.scope_dir} "
              f"(multi_domain falls back to GT geometry; rna_struct off)",
              file=sys.stderr)
    elif args.scope_dir is None:
        print("WARN: no --scope-dir (multi_domain uses GT geometry only; "
              "rna_struct off)", file=sys.stderr)
    if not have_polish:
        print("WARN: no usable --polish-dir (polish_relocate off)",
              file=sys.stderr)

    scope = _load_json_dir(args.scope_dir) if have_scope else {}
    polish = _load_json_dir(args.polish_dir) if have_polish else {}

    samples = collect_sample_data(
        args.step4_dir, args.data_dir, ids,
        feature_set=(model.feature_set if model else "full"),
        use_context=(model.use_context if model else True))
    if not samples:
        print("ERROR: no usable test samples", file=sys.stderr)
        return 1
    samples_by_sid = {s.sid: s for s in samples}
    print(f"usable test samples: {len(samples)}")

    per_r = riboseer_per_sample_r(samples, model, preds)
    if not per_r:
        print("ERROR: no defined per-sample Pearson R", file=sys.stderr)
        return 1
    thr, failed = bottom_decile(per_r)
    print(f"samples with defined R: {len(per_r)}")
    print()

    counts, detail = analyse(
        failed, samples_by_sid, args.step4_dir, scope, polish,
        have_scope=have_scope, have_polish=have_polish)
    print_report(thr, failed, per_r, counts, detail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
