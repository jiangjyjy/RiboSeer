"""Table 16 — RNA-length stratified performance analysis.

Groups the test set by RNA length and reports each method's mean
per-sample Pearson R within each group. No retraining: it reuses the
already-computed step4 tool scores and a pre-trained RiboSeer fusion
bundle, then re-buckets the per-sample correlations by RNA length.

Groups (RNA length in nucleotides, inclusive):

    Short    20-40 nt
    Medium   41-70 nt
    Long     71-100 nt

Samples whose RNA length falls outside ``[20, 100]`` are dropped (and
counted) so the three columns sum to the in-range usable test set.

Methods
-------
Five individual tools plus the RiboSeer fusion, all evaluated with the
**same per-sample / resolved-residue convention** as
``ablation_fusion_method`` / ``table04_main_results.py`` (correlate the
method's per-residue scores against the binary GT on the resolved-residue
subset, then mean across the samples in the group):

* **Boltz-2 / EquiPNAS / Chai-1 / P2Rank / RF2NA** — the tool's raw
  per-residue score (``per_residue_pae_score`` → ``per_residue_confidence``
  fallback), exactly the signal Table 4's per-tool rows use. A sample
  where the tool produced no score (constant 0 vector) is undefined and
  drops out of that tool's group count.
* **RiboSeer** — the loaded ``EnrichedFusion`` bundle's per-residue
  prediction on the enriched feature matrix (same model the headline
  Table 4 RiboSeer row reports).

Because every method reuses the per-sample correlations on the identical
resolved-residue subset, the per-group means here are an exact partition
of each method's overall Table-4 number — no metric drift between tables.

Usage
-----
::

    python scripts/tables/table15_rna_length.py \\
        --step4-dir          data/batch_test_v7/step4/ \\
        --processed-dir      data/processed_quality \\
        --sample-list        data/processed_quality/splits_tmscore_035/test.txt \\
        --enriched-model-dir data/enriched_v7_model/ \\
        --output             data/batch_test_v7/table16_stratified.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from step5_fusion.data_collector import load_sample_json  # noqa: E402
from step5_fusion.enriched_fusion import EnrichedFusion  # noqa: E402
from scripts.riboseer.ablation_fusion_method import (  # noqa: E402
    SampleData, collect_sample_data, corr_on_eval_subset, _tool_raw_vector,
)
from step5_fusion.features_15tool import auto_scope_profile  # noqa: E402
from step5_fusion.prediction_io import (  # noqa: E402
    load_predictions_dir, predictions_to_vector,
)

from typing import Callable  # noqa: E402


# ---------------------------------------------------------------------------
# Length groups + method roster
# ---------------------------------------------------------------------------

# (label, lo, hi) — inclusive nucleotide-length bounds, in display order.
GROUPS: list[tuple[str, int, int]] = [
    ("Short", 20, 40),
    ("Medium", 41, 70),
    ("Long", 71, 100),
]
GROUP_LABELS = [g[0] for g in GROUPS]

# (display name, tool_id). tool_id matches step4 / TOOL_ORDER canonical ids
# (RF2NA's record id is ``rosettafold2na``). RiboSeer is handled separately.
TOOL_METHODS: list[tuple[str, str]] = [
    ("Boltz-2", "boltz2"),
    ("EquiPNAS", "equipnas"),
    ("Chai-1", "chai1"),
    ("P2Rank", "p2rank"),
    ("RF2NA", "rosettafold2na"),
]
RIBOSEER = "RiboSeer"
METHOD_ORDER = [m[0] for m in TOOL_METHODS] + [RIBOSEER]


def group_for(rna_len: int) -> Optional[str]:
    """Length label for ``rna_len``, or None if outside all groups."""
    for label, lo, hi in GROUPS:
        if lo <= rna_len <= hi:
            return label
    return None


# ---------------------------------------------------------------------------
# RNA length lookup
# ---------------------------------------------------------------------------


def _rna_length(sample: dict) -> int:
    rna = sample.get("rna") or sample.get("ligand_rna") or {}
    return int(rna.get("length") or len(rna.get("sequence") or "") or 0)


def rna_lengths(processed_dir: Path, sample_ids: list[str]) -> dict[str, int]:
    """``{sample_id: rna_length}`` for every sample whose JSON loads."""
    out: dict[str, int] = {}
    for sid in sample_ids:
        sample = load_sample_json(processed_dir, sid)
        if sample is None:
            continue
        out[sid] = _rna_length(sample)
    return out


# ---------------------------------------------------------------------------
# RiboSeer prediction (model-type aware)
# ---------------------------------------------------------------------------


def riboseer_predict(model: EnrichedFusion, X: np.ndarray) -> np.ndarray:
    """Per-residue RiboSeer score for one sample's feature matrix.

    Routes to the loaded bundle's estimator: XGBoost → ``predict_proba``
    positive column; LightGBM regressor → ``predict``; Ridge → the
    closed-form predict. Mirrors ``EnrichedFusion.predict_sample`` so the
    RiboSeer row equals its Table-4 number."""
    mt = model.model_type
    if mt == "xgboost":
        if model.model is None:
            raise RuntimeError("RiboSeer model not loaded")
        return np.asarray(model.model.predict_proba(X)[:, 1], dtype=np.float64)
    if mt == "lightgbm":
        if model.model is None:
            raise RuntimeError("RiboSeer model not loaded")
        return np.asarray(model.model.predict(X), dtype=np.float64)
    return np.asarray(model._ridge_predict(X), dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-method, per-sample correlation
# ---------------------------------------------------------------------------


def method_corrs(
    samples: list[SampleData], model: Optional[EnrichedFusion],
    preds: Optional[dict[str, dict[int, float]]] = None,
) -> dict[str, dict[str, float]]:
    """``{method: {sample_id: pearson_r}}`` over the test samples.

    A sample with an undefined correlation for a method (constant
    prediction — e.g. a tool that didn't run) is simply absent from that
    method's inner dict, so it never enters a group mean.

    RiboSeer's vector comes from ``preds`` (pre-computed full-system
    predictions) when supplied, else from the loaded ``model``.
    """
    out: dict[str, dict[str, float]] = {m: {} for m in METHOD_ORDER}
    for s in samples:
        # Individual tools: raw per-residue score vs GT on the eval subset.
        for name, tool_id in TOOL_METHODS:
            corr = corr_on_eval_subset(_tool_raw_vector(s, tool_id), s)
            if corr is not None and corr["pearson_r"] is not None:
                out[name][s.sid] = corr["pearson_r"]
        # RiboSeer fusion.
        if preds is not None:
            vec = predictions_to_vector(preds.get(s.sid), s.residue_ids)
        else:
            vec = riboseer_predict(model, s.X)
        corr = corr_on_eval_subset(vec, s)
        if corr is not None and corr["pearson_r"] is not None:
            out[RIBOSEER][s.sid] = corr["pearson_r"]
    return out


def resolve_riboseer_source(
    predictions_dir: Optional[Path], model_dir: Optional[Path],
) -> tuple[Optional[EnrichedFusion], Optional[dict[str, dict[int, float]]]]:
    """Resolve the RiboSeer prediction source for the stratified tables.

    ``--predictions-dir`` (pre-computed full-system on/on/on predictions)
    takes precedence; otherwise the EnrichedFusion bundle at ``model_dir``
    is loaded. Returns ``(model, preds)`` with exactly one populated.
    Raises ``ValueError`` with a user-facing message on misconfiguration /
    load failure (caller prints it and exits non-zero)."""
    if predictions_dir is not None:
        preds = load_predictions_dir(predictions_dir)
        if not preds:
            raise ValueError(f"no predictions under {predictions_dir}")
        return None, preds
    if model_dir is None:
        raise ValueError("pass either --predictions-dir or "
                         "--enriched-model-dir")
    try:
        model = EnrichedFusion.load(model_dir)
    except (OSError, ValueError, ImportError, json.JSONDecodeError) as e:
        raise ValueError(f"--enriched-model-dir load failed: {e}") from e
    return model, None


# ---------------------------------------------------------------------------
# Generic stratification core (shared by Table 16 / 17 / 18)
# ---------------------------------------------------------------------------


def stratify_by(
    corrs: dict[str, dict[str, float]],
    labels_by_sid: dict[str, Optional[str]],
    group_labels: list[str],
) -> dict[str, dict[str, dict]]:
    """``{method: {group: {"n": int, "pearson_r_mean": float|None}}}``.

    Buckets each method's per-sample Pearson R by ``labels_by_sid[sid]``.
    A sample whose label is not in ``group_labels`` (``None`` / out of
    range) is dropped from every method. The dimension that produces the
    label (RNA length / protein family / RNA context) lives in the caller —
    this core is dimension-agnostic.
    """
    result: dict[str, dict[str, dict]] = {}
    for method in METHOD_ORDER:
        by_group: dict[str, list[float]] = {g: [] for g in group_labels}
        for sid, pr in corrs.get(method, {}).items():
            label = labels_by_sid.get(sid)
            if label in by_group:
                by_group[label].append(pr)
        result[method] = {
            g: {"n": len(vals),
                "pearson_r_mean": (round(statistics.fmean(vals), 4)
                                   if vals else None)}
            for g, vals in by_group.items()
        }
    return result


def group_counts(labels_by_sid: dict[str, Optional[str]],
                 usable_ids: set[str],
                 group_labels: list[str]) -> dict[str, int]:
    """How many usable samples fall in each group (method-agnostic).
    Samples whose label is outside ``group_labels`` count toward
    ``_dropped``."""
    counts = {g: 0 for g in group_labels}
    dropped = 0
    for sid in usable_ids:
        label = labels_by_sid.get(sid)
        if label in counts:
            counts[label] += 1
        else:
            dropped += 1
    counts["_dropped"] = dropped
    return counts


def load_profiles(directory: Optional[Path]) -> dict[str, dict]:
    """``{sample_id: parsed_json}`` from ``<dir>/<sid>.json`` (SCOPE
    profiles). Missing dir → empty (caller falls back to the auto
    profile)."""
    out: dict[str, dict] = {}
    if directory is None or not directory.is_dir():
        return out
    for f in directory.glob("*.json"):
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def profile_labels(
    processed_dir: Path, sample_ids: list[str],
    profiles: dict[str, dict], field: str,
    normalize: Callable[[Optional[str]], str],
) -> dict[str, str]:
    """``{sample_id: group_label}`` derived from a SCOPE-profile field
    (Table 17 ``protein_family`` / Table 18 ``rna_context``).

    Priority per sample: the LLM SCOPE profile's ``field`` → the heuristic
    ``auto_scope_profile`` field (when the profile or field is missing) →
    ``normalize(None)`` (→ the caller's catch-all bucket). ``normalize``
    maps any raw string onto one of the table's group labels, so the result
    is always a valid bucket (never dropped)."""
    out: dict[str, str] = {}
    for sid in sample_ids:
        prof = profiles.get(sid)
        raw = (prof or {}).get(field)
        if not raw:
            sample = load_sample_json(processed_dir, sid)
            if sample is not None:
                raw = auto_scope_profile(sample).get(field)
        out[sid] = normalize(raw)
    return out


# Table-16-specific thin wrappers (preserve the original API).

def stratify(corrs: dict[str, dict[str, float]],
             lengths: dict[str, int]) -> dict[str, dict[str, dict]]:
    """RNA-length stratification (Table 16) via the generic core."""
    labels = {sid: group_for(length) for sid, length in lengths.items()}
    return stratify_by(corrs, labels, GROUP_LABELS)


def group_sample_counts(lengths: dict[str, int],
                        usable_ids: set[str]) -> dict[str, int]:
    labels = {sid: group_for(lengths.get(sid, -1)) for sid in usable_ids}
    return group_counts(labels, usable_ids, GROUP_LABELS)


# ---------------------------------------------------------------------------
# Output (generic — group labels + optional column spans supplied by caller)
# ---------------------------------------------------------------------------

_COLUMNS = ["method", "group", "n", "pearson_r_mean"]


def write_csv(path: Path, table: dict[str, dict[str, dict]],
              group_labels: list[str] = GROUP_LABELS) -> None:
    """Long-format CSV: one row per (method, group)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for method in METHOD_ORDER:
            for g in group_labels:
                cell = table[method][g]
                w.writerow({"method": method, "group": g,
                            "n": cell["n"],
                            "pearson_r_mean": cell["pearson_r_mean"]})


def print_table(table: dict[str, dict[str, dict]],
                counts: dict[str, int],
                group_labels: list[str] = GROUP_LABELS,
                spans: Optional[dict[str, str]] = None,
                dropped_label: str = "out-of-range") -> None:
    """Wide table: a (n, PearsonR) pair per group column. ``spans`` adds a
    parenthetical to each column header (e.g. length ranges)."""
    header = f"{'':10s}"
    for g in group_labels:
        title = g + (f"({spans[g]})" if spans and g in spans else "")
        header += f"  {title:>16s}"
    print(header)
    sub = f"{'Method':10s}"
    for _g in group_labels:
        sub += f"  {'n':>4s} {'PearsonR':>9s}"
    print(sub)
    print("-" * len(sub))
    for method in METHOD_ORDER:
        line = f"{method:10s}"
        for g in group_labels:
            cell = table[method][g]
            pr = cell["pearson_r_mean"]
            pr_s = f"{pr:.3f}" if pr is not None else "–"
            line += f"  {cell['n']:>4d} {pr_s:>9s}"
        print(line)
    print()
    print("group sizes (usable test samples): " + ", ".join(
        f"{g}={counts[g]}" for g in group_labels)
        + f", dropped({dropped_label})={counts['_dropped']}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _load_sample_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"sample list not found: {path}")
    return [ln.split()[0] for ln in
            path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step4-dir", type=Path, required=True,
                   help="test-split step4 JSONL dir")
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--sample-list", type=Path, required=True)
    p.add_argument("--enriched-model-dir", type=Path, default=None,
                   help="pre-trained RiboSeer EnrichedFusion bundle; not "
                        "needed when --predictions-dir is given")
    p.add_argument("--predictions-dir", type=Path, default=None,
                   help="pre-computed full-system (on/on/on) predictions "
                        "<sid>.json; when set RiboSeer reads these instead "
                        "of loading the model")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for d in (args.step4_dir, args.processed_dir):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 1
    try:
        sample_ids = _load_sample_ids(args.sample_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    try:
        model, preds = resolve_riboseer_source(
            args.predictions_dir, args.enriched_model_dir)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Collect feature matrices with the bundle's own feature config so the
    # RiboSeer prediction matches how the model was trained.
    samples = collect_sample_data(
        args.step4_dir, args.processed_dir, sample_ids,
        feature_set=(model.feature_set if model else "full"),
        use_context=(model.use_context if model else True))
    if not samples:
        print("ERROR: no usable test samples", file=sys.stderr)
        return 1
    lengths = rna_lengths(args.processed_dir, [s.sid for s in samples])
    print(f"usable test samples: {len(samples)}")

    corrs = method_corrs(samples, model, preds)
    table = stratify(corrs, lengths)
    counts = group_sample_counts(lengths, {s.sid for s in samples})

    write_csv(args.output, table)
    print(f"wrote {args.output}  "
          f"({len(METHOD_ORDER)} methods × {len(GROUP_LABELS)} groups)")
    print()
    print_table(table, counts,
                spans={"Short": "20-40", "Medium": "41-70", "Long": "71-100"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
