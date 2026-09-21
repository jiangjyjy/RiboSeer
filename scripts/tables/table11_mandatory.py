#!/usr/bin/env python3
"""Table 11 (v2) — MANDATORY-tools guardrail on the headline pipeline.

Why a v2
--------
The original Table 11 used MANDATORY = the 3 Category-A structure tools
{Boltz-2, Chai-1, RoseTTAFold2NA}; that policy was the *worst* of the
three (Pearson R = 0.565). Forward selection found a stronger 7-tool core

    boltz2, chai1, equipnas, nucleicnet, rnabindrplus, hdock, deeppocket

which reaches Pearson R = 0.596 when always selected (UCB full). v2 makes
*that* set the MANDATORY guardrail and re-scores the three selection
policies on the **headline on/on/on pipeline** (SCOPE profile + MAESTRO
selection + LightGBM fusion + POLISH), exactly as Table 10 v2 — only the
per-sample tool subset changes between rows.

Three selection policies (one retrained LightGBM each)
------------------------------------------------------
1. ``Free LLM selection (3-15 tools)`` — each sample uses *only* MAESTRO's
   own picks (no MANDATORY injection, no union).
2. ``MANDATORY optimal-7 + LLM extras`` — force-include the 7-tool core,
   keep MAESTRO's extra picks on top: ``K_sel = K_LLM ∪ MANDATORY_7``.
3. ``All 15 tools always`` — every library tool active, selection ignored.

Each policy runs the full flow: resolve the per-sample tool set → SCOPE G4
block from the LLM profile → G1/G2/G3 features over the tool set
(un-selected tools → NaN) → retrain LightGBM (train split uses the *same*
policy) → predict on test → apply saved POLISH actions → Pearson / R².

No LLM call is made; the saved MAESTRO selections / SCOPE profiles /
POLISH actions are reused verbatim.

Reported columns
----------------
* **PearsonR / R²** — per-residue, mean over test samples (same metric as
  Table 9 / 10).
* **#tools/sample** — mean policy tool-set size over the test split.
* **runtime(min)** — mean per-sample wall-clock of the policy's tools.
  Per tool: real step4 ``runtime_seconds`` when available, else the Table-1
  timeout (``TOOL_TIMEOUT``, an upper bound); summed sequentially over the
  sample's tools, averaged, /60. ``--runtime-mode timeout`` forces the
  pure-timeout upper bound for every tool.

Usage
-----
::

    python scripts/tables/table11_mandatory.py \\
        --processed-dir      data/processed_quality \\
        --train-step4-dir    data/batch_train_v7/step4 \\
        --test-step4-dir     data/batch_test_v7/step4 \\
        --train-list         data/processed_quality/splits_tmscore_035/train.txt \\
        --test-list          data/processed_quality/splits_tmscore_035/test.txt \\
        --scope-dir-train    data/batch_train_v7/scope_profiles_llm \\
        --scope-dir-test     data/batch_test_v7/scope_profiles_llm \\
        --maestro-dir-train  data/batch_train_v7/maestro_selections_llm_v4 \\
        --maestro-dir-test   data/batch_test_v7/maestro_selections_llm_v4 \\
        --polish-dir         data/batch_test_v7/polish_actions_llm_v2 \\
        --output             data/batch_test_v7/table11_v2.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Reuse the headline (on/on/on) machinery verbatim so this table stays
# byte-for-byte comparable to the Table 9 / Table 10 v2 full-system row.
from step5_fusion.lightgbm_fusion import (  # noqa: E402
    LGBM_OK,
    SampleT9,
    _corr_from_probdict,
    _load_json_dir,
    _train_model,
    apply_polish,
    collect_samples,
    resolve_scope_vector,
    resolve_selected_tools,
)
from scripts.tables.table09_llm_modules import (  # noqa: E402
    _load_sample_ids,
)
from step5_fusion.features_15tool import (  # noqa: E402
    ALL_KNOWN_TOOLS, build_15tool_features,
)
from scripts.riboseer.make_table11_selections import (  # noqa: E402
    clean_tools, _LIB_INDEX,
)
from scripts.tables.table11_runtime import (  # noqa: E402
    extract_tool_runtimes, sample_runtime,
)
from scripts.riboseer.config import MANDATORY_TOOLS  # noqa: E402

# Forward-selection optimal 7-tool core (Pearson R = 0.596 at UCB full).
# Single source of truth lives in ``config.MANDATORY_TOOLS`` — this is the
# same guardrail now used by the headline pipeline (Table 9 / 10 v2).
MANDATORY_7: tuple[str, ...] = tuple(MANDATORY_TOOLS)

# Table-1 per-tool timeouts (seconds) — the runtime upper bound used when a
# tool has no real step4 timing (external-service tools log none).
TOOL_TIMEOUT: dict[str, float] = {
    "boltz2": 900, "chai1": 900, "rosettafold2na": 600, "rfaa": 1200,
    "alphafold3": 900,
    "p2rank": 60, "fpocket": 60, "deeppocket": 300,
    "equipnas": 300, "nucleicnet": 600, "graphbind": 600,
    "rnabindrplus": 300, "bindup": 300,
    "hdock": 600, "haddock3": 600,
}

# Policy id → (display label, resolver tag).
POLICIES: tuple[tuple[str, str], ...] = (
    ("free", "Free LLM selection (3-15 tools)"),
    ("mandatory7", "MANDATORY optimal-7 + LLM extras"),
    ("all", "All 15 tools always"),
)


@dataclass
class PolicyResult:
    policy: str
    label: str
    n_samples: int
    pearson_r: Optional[float]
    spearman_r: Optional[float]
    r_squared: Optional[float]
    avg_tools: Optional[float]
    runtime_min: Optional[float]


# ---------------------------------------------------------------------------
# Per-sample tool set per policy
# ---------------------------------------------------------------------------


def resolve_policy_tools(sample: SampleT9, selections: dict[str, dict],
                         policy: str) -> list[str]:
    """The active tool subset for one sample under a selection policy.

    * ``all``        → every library tool.
    * ``free``       → MAESTRO's own picks, exactly (no MANDATORY union).
    * ``mandatory7`` → ``K_LLM ∪ MANDATORY_7``.
    """
    if policy == "all":
        return list(ALL_KNOWN_TOOLS)
    llm = clean_tools(resolve_selected_tools(sample, True, selections))
    if policy == "free":
        return llm
    if policy == "mandatory7":
        merged = set(llm) | set(MANDATORY_7)
        return sorted(merged, key=lambda t: _LIB_INDEX[t])
    raise ValueError(f"unknown policy: {policy}")


def build_matrix(sample: SampleT9, profiles: dict[str, dict],
                 selections: dict[str, dict], policy: str) -> np.ndarray:
    """Headline feature matrix (SCOPE on) for one sample under ``policy``.
    Column layout is the constant 154-D ``table9`` contract regardless of
    the subset (un-selected tools → NaN columns)."""
    selected = resolve_policy_tools(sample, selections, policy)
    scope_vec = resolve_scope_vector(sample, True, profiles)  # SCOPE on
    return build_15tool_features(
        sample.step4_data, sample.residue_ids,
        selected_tools=selected, scope_vector=scope_vec,
        use_context=True, use_scope=True)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


def resolve_runtimes(step4_dir: Optional[Path], mode: str
                     ) -> tuple[dict[str, float], dict[str, str]]:
    """Per-tool runtime (seconds) plus a per-tool source tag.

    ``mode='real'``  → real step4 ``runtime_seconds`` when present, else the
    Table-1 timeout. ``mode='timeout'`` → always the Table-1 timeout.
    """
    real = {} if mode == "timeout" else extract_tool_runtimes(step4_dir)
    runtimes: dict[str, float] = {}
    source: dict[str, str] = {}
    for t in ALL_KNOWN_TOOLS:
        if t in real:
            runtimes[t] = round(real[t], 1)
            source[t] = "real"
        else:
            runtimes[t] = float(TOOL_TIMEOUT[t])
            source[t] = "timeout"
    return runtimes, source


def policy_runtime_minutes(test: list[SampleT9], selections: dict[str, dict],
                           policy: str, runtimes: dict[str, float]
                           ) -> Optional[float]:
    secs = [sample_runtime(resolve_policy_tools(s, selections, policy),
                           runtimes, "sequential") for s in test]
    return round(statistics.fmean(secs) / 60.0, 2) if secs else None


# ---------------------------------------------------------------------------
# Train / eval per policy
# ---------------------------------------------------------------------------


def _mean(vs: list[float]) -> Optional[float]:
    return round(statistics.fmean(vs), 4) if vs else None


def eval_policy(
    policy: str, label: str,
    train: list[SampleT9], test: list[SampleT9],
    profiles_train: dict[str, dict], profiles_test: dict[str, dict],
    selections_train: dict[str, dict], selections_test: dict[str, dict],
    polish_actions: dict[str, dict],
    runtimes: dict[str, float],
    polish_on: bool = True,
) -> PolicyResult:
    """Retrain the fusion LightGBM under ``policy`` (same policy on the
    train split), predict on test, optionally apply POLISH (``polish_on``),
    and aggregate. ``polish_on=False`` keeps the raw fusion prediction
    (POLISH removed from the headline)."""
    X_train = np.vstack([
        build_matrix(s, profiles_train, selections_train, policy)
        for s in train])
    y_train = np.concatenate([s.y for s in train])
    model = _train_model(X_train, y_train)

    corrs: list[dict] = []
    for s in test:
        X = build_matrix(s, profiles_test, selections_test, policy)
        vec = np.asarray(model.predict(X), dtype=np.float64)
        prob = {rid: float(vec[i]) for i, rid in enumerate(s.residue_ids)}
        prob = apply_polish(prob, s, polish_on, polish_actions)
        corr = _corr_from_probdict(prob, s)
        if corr is not None:
            corrs.append(corr)

    prs = [c["pearson_r"] for c in corrs if c["pearson_r"] is not None]
    srs = [c["spearman_r"] for c in corrs if c["spearman_r"] is not None]
    r2s = [c["r_squared"] for c in corrs if c["r_squared"] is not None]
    avg_tools = statistics.fmean(
        [len(resolve_policy_tools(s, selections_test, policy)) for s in test]
    ) if test else None
    return PolicyResult(
        policy=policy, label=label, n_samples=len(prs),
        pearson_r=_mean(prs), spearman_r=_mean(srs), r_squared=_mean(r2s),
        avg_tools=round(avg_tools, 2) if avg_tools is not None else None,
        runtime_min=policy_runtime_minutes(
            test, selections_test, policy, runtimes))


def run_table11_v2(
    train: list[SampleT9], test: list[SampleT9],
    profiles_train: dict[str, dict], profiles_test: dict[str, dict],
    selections_train: dict[str, dict], selections_test: dict[str, dict],
    polish_actions: dict[str, dict],
    runtimes: dict[str, float],
    polish_on: bool = True,
) -> list[PolicyResult]:
    return [
        eval_policy(pid, label, train, test, profiles_train, profiles_test,
                    selections_train, selections_test, polish_actions,
                    runtimes, polish_on=polish_on)
        for pid, label in POLICIES
    ]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_COLUMNS = ["policy", "label", "n_samples", "pearson_r", "spearman_r",
            "r_squared", "avg_tools", "runtime_min"]


def write_csv(path: Path, rows: list[PolicyResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: getattr(r, c) for c in _COLUMNS})


def _fnum(v: Optional[float], nd: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def print_runtimes(runtimes: dict[str, float], source: dict[str, str]) -> None:
    print("Per-tool runtime (seconds):")
    for t in ALL_KNOWN_TOOLS:
        print(f"  {t:16s} {runtimes[t]:8.1f}s  ({source[t]})")
    print()


def print_table(rows: list[PolicyResult]) -> None:
    print("=== Table 11 (v2): MANDATORY Tools Guardrail ===")
    hdr = (f"{'Selection policy':<36s} {'PearsonR':>8s} {'R2':>7s} "
           f"{'#tools/sample':>14s} {'runtime(min)':>13s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r.label:<36s} {_fnum(r.pearson_r):>8s} "
              f"{_fnum(r.r_squared):>7s} "
              f"{_fnum(r.avg_tools, 1):>14s} {_fnum(r.runtime_min, 1):>13s}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--train-step4-dir", type=Path, required=True)
    p.add_argument("--test-step4-dir", type=Path, required=True)
    p.add_argument("--train-list", type=Path, required=True)
    p.add_argument("--test-list", type=Path, required=True)
    p.add_argument("--scope-dir-train", type=Path, default=None)
    p.add_argument("--scope-dir-test", type=Path, default=None)
    p.add_argument("--maestro-dir-train", type=Path, default=None)
    p.add_argument("--maestro-dir-test", type=Path, default=None)
    p.add_argument("--polish-dir", type=Path, default=None,
                   help="saved POLISH actions; omit (or pass --no-polish) to "
                        "run with POLISH off (the new headline).")
    p.add_argument("--no-polish", action="store_true",
                   help="force POLISH off even if --polish-dir is given.")
    p.add_argument("--runtime-mode", choices=("real", "timeout"),
                   default="real",
                   help="'real' uses step4 runtime_seconds with the Table-1 "
                        "timeout as fallback; 'timeout' forces the timeout "
                        "upper bound for every tool (default: real).")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not LGBM_OK:
        print("ERROR: lightgbm not installed (pip install lightgbm)",
              file=sys.stderr)
        return 1
    for d in (args.train_step4_dir, args.test_step4_dir, args.processed_dir):
        if not d.is_dir():
            print(f"ERROR: not a directory: {d}", file=sys.stderr)
            return 1
    try:
        train_ids = _load_sample_ids(args.train_list)
        test_ids = _load_sample_ids(args.test_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    train = collect_samples(args.train_step4_dir, args.processed_dir,
                            train_ids)
    test = collect_samples(args.test_step4_dir, args.processed_dir, test_ids)
    print(f"train usable: {len(train)}, test usable: {len(test)}")
    if not train or not test:
        print("ERROR: no usable train/test samples", file=sys.stderr)
        return 1

    profiles_train = _load_json_dir(args.scope_dir_train)
    profiles_test = _load_json_dir(args.scope_dir_test)
    selections_train = _load_json_dir(args.maestro_dir_train)
    selections_test = _load_json_dir(args.maestro_dir_test)
    polish_actions = _load_json_dir(args.polish_dir)
    polish_on = (args.polish_dir is not None) and not args.no_polish
    print(f"loaded SCOPE: train={len(profiles_train)} test={len(profiles_test)}"
          f"  MAESTRO: train={len(selections_train)} "
          f"test={len(selections_test)}  POLISH: "
          f"{'on (' + str(len(polish_actions)) + ' actions)' if polish_on else 'off'}")

    runtimes, source = resolve_runtimes(args.test_step4_dir, args.runtime_mode)
    rows = run_table11_v2(
        train, test, profiles_train, profiles_test,
        selections_train, selections_test, polish_actions, runtimes,
        polish_on=polish_on)
    write_csv(args.output, rows)
    print(f"\nwrote {args.output}  ({len(rows)} policies)\n")
    print_runtimes(runtimes, source)
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
