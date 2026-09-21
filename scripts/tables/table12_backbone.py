#!/usr/bin/env python3
"""Table 13 — LLM backbone comparison (SCOPE × MAESTRO × POLISH).

Runs the full SCOPE + MAESTRO + POLISH LLM pipeline under three swappable
backbones, all served through the **the LLM relay** aggregator (one base URL +
one API key, only the model id differs):

    DeepSeek V4 Flash   deepseek-v4-flash
    Qwen 3.7 Max        qwen3.7-max
    Gemini 3.5 Flash    gemini-3.5-flash

and reports each one's on/on/on per-residue Pearson / Spearman / R² plus
its schema validation failure rate, bracketed by two reference rows read
from the shipped v4 ablation CSV:

    No LLM (UCB only)   Table 9  off/off/off   (deterministic control)
    LLM 5.1 (ours)      Table 9  on/on/on      (headline backbone)

**Test-only backbone swap.** The train side is backbone-independent — the
LightGBM fusion model trains on the train split's SCOPE/MAESTRO features,
which come from the existing LLM/deterministic run, NOT the backbone
(verified: Gemini's off/off/off = 0.5542 = the No-LLM baseline). So the
backbone's API is only spent on the **107 test samples**; the train-side
LLM JSONs are reused via ``--train-scope-dir`` / ``--train-maestro-dir``.

Per backbone this driver runs, into ``<work-dir>/<tag>/``:

1. SCOPE profiles      (test only)     — generate_scope_profiles  --mode llm
2. MAESTRO selections  (test only)     — generate_maestro_selections --mode llm
3. POLISH iterative    (test only)     — generate_polish_iterative --mode llm
                                          --max-rounds 3  (call-budget saver)
4. table09_llm_modules.py             — trains on the EXISTING train-side
                                          SCOPE/MAESTRO dirs, evaluates the
                                          backbone's test side; reads on/on/on

each generator pointed at the LLM relay via ``--api-base/--api-key/--model``
(the OpenAI-compatible ``BackboneClient`` — the LLM relay speaks
``/v1/chat/completions``). The intermediate JSON dirs are persisted, so a
backbone whose ``table09_llm_modules.csv`` already exists is skipped on
re-run (coarse resume; pass ``--force`` to regenerate).

Call budget (per backbone, test-only): SCOPE 107 + MAESTRO 107 + POLISH
107×≤3 rounds ≈ 320-535 calls — down from ~985 (the old train+test run
spent ~660 on the train side alone).

The API key is the single shared ``LLM_RELAY_API_KEY`` (never hard-coded /
never passed on the driver CLI); it falls back to the legacy per-backbone
env var (``DEEPSEEK_API_KEY`` / ``QWEN_API_KEY`` / ``GEMINI_API_KEY``) if
the shared one is unset. A backbone with no key resolves is skipped with a
one-line note. The base URL defaults to ``https://api.llm-relay.example/v1`` and
is overridable with ``--base-url`` (the LLM relay's exact path is uncertain).

Usage
-----
::

    export LLM_RELAY_API_KEY=sk-...
    python scripts/tables/table12_backbone.py \\
        --processed-dir      data/processed_quality \\
        --train-list         data/processed_quality/splits_tmscore_035/train.txt \\
        --test-list          data/processed_quality/splits_tmscore_035/test.txt \\
        --train-step4-dir    data/batch_train_v7/step4 \\
        --test-step4-dir     data/batch_test_v7/step4 \\
        --weight-tensor      data/batch_train_v7/W_v2.json \\
        --enriched-model-dir data/enriched_v7_lgbm \\
        --train-scope-dir    data/batch_train_v7/scope_profiles_llm \\
        --train-maestro-dir  data/batch_train_v7/maestro_selections_llm_v4 \\
        --work-dir           data/batch_test_v7/table13 \\
        --done-ablation-csv  data/batch_test_v7/table09_llm_modules_v4.csv \\
        --output             data/batch_test_v7/table13_backbone.csv

    # one backbone at a time:  --only deepseek   (or qwen / gemini)
    # alternate base URL:      --base-url https://api.llm-relay.example/v1
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scripts.riboseer import generate_scope_profiles as gsp  # noqa: E402
from scripts.riboseer import generate_maestro_selections as gm  # noqa: E402
from scripts.riboseer import generate_polish_iterative as gpi  # noqa: E402
from scripts.tables import table09_llm_modules as alm  # noqa: E402
from scripts.tables.table13_prompt import (  # noqa: E402
    _scan_flat_source, _scan_round_source,
)

# POLISH rounds — 3 (not 5) keeps the per-backbone call budget < 1000.
POLISH_MAX_ROUNDS = 3


def failure_rate(dirs: dict) -> dict:
    """Schema-validation failure rate over the backbone's **test-side** LLM
    output only (SCOPE + MAESTRO test dirs by ``source`` tag, POLISH test
    rounds). The train side is reused/not backbone-generated, so it's
    excluded — this rate reflects exactly the calls this backbone made."""
    a = f = 0
    for d, tag in ((dirs["scope_test"], "cauto_fallback"),
                   (dirs["maestro_test"], "llm_fallback")):
        da, df = _scan_flat_source(d, tag)
        a += da
        f += df
    pa, pf = _scan_round_source(dirs["polish_test"])
    a += pa
    f += pf
    return {"attempts": a, "fails": f,
            "pct_fail": round(100.0 * f / a, 1) if a else None}

# All three backbones now go through the LLM relay (one aggregator, one key, one
# base URL — only the model id differs). Base URL is overridable via
# --base-url (the LLM relay's exact path is uncertain). Key from $LLM_RELAY_API_KEY.
LLM_RELAY_BASE_URL = "https://api.llm-relay.example/v1"

# Backbones in display order (between the two reference rows). ``key_env`` is
# kept only as a legacy per-backbone fallback for the shared the LLM relay key.
BACKBONES: list[dict] = [
    {"name": "DeepSeek V4 Flash", "tag": "deepseek",
     "model": "deepseek-v4-flash", "key_env": "DEEPSEEK_API_KEY"},
    {"name": "Qwen 3.7 Max", "tag": "qwen",
     "model": "qwen3.7-max", "key_env": "QWEN_API_KEY"},
    {"name": "Gemini 3.5 Flash", "tag": "gemini",
     "model": "gemini-3.5-flash", "key_env": "GEMINI_API_KEY"},
]

# Reference rows read from --done-ablation-csv; (label, scope, maestro,
# polish) selects the Table-9 combo. Fallback metrics (used only if the row
# is absent) come from the paper's reported numbers.
_REF_NO_LLM = {"label": "No LLM (UCB only)", "combo": ("off", "off", "off"),
               "fallback": (0.554, 0.436, 0.389)}
_REF_LLM = {"label": "LLM 5.1 (ours)", "combo": ("on", "on", "on"),
            "fallback": (0.565, 0.446, 0.395)}


# ---------------------------------------------------------------------------
# Per-backbone directory layout + generator argv
# ---------------------------------------------------------------------------


def cell_dirs(work_dir: Path, tag: str) -> dict[str, Path]:
    # Test-only: no per-backbone train dirs — the train side is reused from
    # --train-scope-dir / --train-maestro-dir.
    d = work_dir / tag
    return {
        "scope_test": d / "scope_profiles_test",
        "maestro_test": d / "maestro_selections_test",
        "polish_test": d / "polish_actions_iterative",
        "ablation_csv": d / "table09_llm_modules.csv",
    }


def _backbone_argv(bk: dict, base_url: str, api_key: str) -> list[str]:
    # the LLM relay is OpenAI-compatible (/v1/chat/completions) — no Anthropic
    # fallback / thinking toggle needed; the shared base URL + key + the
    # backbone's model id is all the generators need.
    return ["--api-base", base_url, "--api-key", api_key,
            "--model", bk["model"]]


def build_stage_argvs(bk: dict, base_url: str, api_key: str, paths: dict,
                      dirs: dict) -> dict[str, list]:
    style = _backbone_argv(bk, base_url, api_key)
    scope_common = ["--processed-dir", str(paths["processed_dir"]),
                    "--mode", "llm"]
    maestro_common = ["--processed-dir", str(paths["processed_dir"]),
                      "--mode", "llm",
                      "--weight-tensor", str(paths["weight_tensor"])]
    # Train side reuses existing LLM JSONs (backbone-independent); only the
    # test side is generated with the backbone.
    ablation = [
        "--train-step4-dir", str(paths["train_step4_dir"]),
        "--test-step4-dir", str(paths["test_step4_dir"]),
        "--processed-dir", str(paths["processed_dir"]),
        "--train-list", str(paths["train_list"]),
        "--test-list", str(paths["test_list"]),
        "--scope-profiles-test", str(dirs["scope_test"]),
        "--maestro-selections-test", str(dirs["maestro_test"]),
        "--polish-actions", str(dirs["polish_test"]),
        "--output", str(dirs["ablation_csv"]),
    ]
    if paths.get("train_scope_dir") is not None:
        ablation += ["--scope-profiles-train", str(paths["train_scope_dir"])]
    if paths.get("train_maestro_dir") is not None:
        ablation += ["--maestro-selections-train",
                     str(paths["train_maestro_dir"])]

    return {
        "scope_test": scope_common + [
            "--sample-list", str(paths["test_list"]),
            "--out-dir", str(dirs["scope_test"])] + style,
        "maestro_test": maestro_common + [
            "--sample-list", str(paths["test_list"]),
            "--step4-dir", str(paths["test_step4_dir"]),
            "--scope-profiles", str(dirs["scope_test"]),
            "--out-dir", str(dirs["maestro_test"])] + style,
        "polish_test": [
            "--processed-dir", str(paths["processed_dir"]),
            "--sample-list", str(paths["test_list"]),
            "--step4-dir", str(paths["test_step4_dir"]),
            "--enriched-model-dir", str(paths["enriched_model_dir"]),
            "--out-dir", str(dirs["polish_test"]),
            "--mode", "llm",
            "--max-rounds", str(POLISH_MAX_ROUNDS)] + style,
        "ablation": ablation,
    }


def run_backbone_generation(bk: dict, base_url: str, api_key: str,
                            paths: dict, dirs: dict) -> None:
    """Run the 3 test-only generation stages + ablation for one backbone
    (server-side; each stage hits the the LLM relay API). Raises if any stage
    returns non-zero."""
    argvs = build_stage_argvs(bk, base_url, api_key, paths, dirs)
    for stage, mod in (("scope_test", gsp),
                       ("maestro_test", gm),
                       ("polish_test", gpi)):
        rc = mod.main(argvs[stage])
        if rc != 0:
            raise RuntimeError(f"stage {stage} failed (rc={rc})")
    rc = alm.main(argvs["ablation"])
    if rc != 0:
        raise RuntimeError(f"ablation failed (rc={rc})")


# ---------------------------------------------------------------------------
# Read ablation rows
# ---------------------------------------------------------------------------


def read_combo_row(csv_path: Path,
                   combo: tuple[str, str, str]) -> Optional[dict]:
    """The (scope, maestro, polish) row from an ablation CSV, or None."""
    if not csv_path.is_file():
        return None
    want = {"scope": combo[0], "maestro": combo[1], "polish": combo[2]}
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            if all(rec.get(k) == v for k, v in want.items()):
                return rec
    return None


def _num(s):
    try:
        return round(float(s), 4)
    except (TypeError, ValueError):
        return None


def _metrics_from_row(rec: Optional[dict],
                      fallback: tuple[float, float, float]) -> dict:
    if rec is None:
        return {"pearson_r_mean": fallback[0], "spearman_r_mean": fallback[1],
                "r2_mean": fallback[2], "n_samples": None}
    return {
        "pearson_r_mean": _num(rec.get("pearson_r_mean")),
        "spearman_r_mean": _num(rec.get("spearman_r_mean")),
        "r2_mean": _num(rec.get("r2_mean")),
        "n_samples": (int(rec["n_samples"]) if rec.get("n_samples")
                      else None),
    }


def _reference_row(ref: dict, done_csv: Optional[Path]) -> dict:
    rec = read_combo_row(done_csv, ref["combo"]) if done_csv else None
    m = _metrics_from_row(rec, ref["fallback"])
    return {"backbone": ref["label"], **m, "pct_fail": None,
            "status": "reference"}


def _backbone_row(bk: dict, dirs: dict, status: str) -> dict:
    m = _metrics_from_row(read_combo_row(dirs["ablation_csv"],
                                         ("on", "on", "on")),
                          (None, None, None))
    fail = failure_rate(dirs)
    return {"backbone": bk["name"], **m,
            "pct_fail": fail.get("pct_fail"), "status": status}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_COLUMNS = ["backbone", "pearson_r_mean", "spearman_r_mean", "r2_mean",
            "pct_fail", "n_samples", "status"]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in _COLUMNS})


def print_table(rows: list[dict]) -> None:
    hdr = (f"{'LLM backbone':21s} {'PearsonR':>9s} {'SpearmanR':>10s} "
           f"{'R2':>8s} {'%Fail':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        pf = (f"{r['pct_fail']}%" if r.get("pct_fail") is not None else "–")
        print(f"{r['backbone']:21s} {str(r['pearson_r_mean']):>9s} "
              f"{str(r['spearman_r_mean']):>10s} {str(r['r2_mean']):>8s} "
              f"{pf:>6s}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--train-list", type=Path, required=True)
    p.add_argument("--test-list", type=Path, required=True)
    p.add_argument("--train-step4-dir", type=Path, required=True)
    p.add_argument("--test-step4-dir", type=Path, required=True)
    p.add_argument("--weight-tensor", type=Path, required=True)
    p.add_argument("--enriched-model-dir", type=Path, required=True)
    p.add_argument("--train-scope-dir", type=Path, default=None,
                   help="existing train-split SCOPE profiles dir (reused; "
                        "backbone-independent). Omit → ablation uses its "
                        "deterministic train fallback.")
    p.add_argument("--train-maestro-dir", type=Path, default=None,
                   help="existing train-split MAESTRO selections dir (reused)")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--done-ablation-csv", type=Path, default=None,
                   help="shipped v4 ablation CSV; supplies the No-LLM "
                        "(off/off/off) and LLM 5.1 (on/on/on) reference rows")
    p.add_argument("--only", type=str, default=None,
                   help="comma list of backbone tags to run "
                        "(gemini, deepseek, qwen); default = all")
    p.add_argument("--base-url", type=str, default=LLM_RELAY_BASE_URL,
                   help=f"the LLM relay base URL (default {LLM_RELAY_BASE_URL}; "
                        "try https://api.llm-relay.example/v1 or "
                        "https://api.llm-relay.example/api/v1 if it 404s)")
    p.add_argument("--force", action="store_true",
                   help="regenerate even if a backbone's ablation CSV exists; "
                        "also wipes that backbone's scope/maestro/polish dirs "
                        "so every test sample is re-sent to the API")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    paths = {
        "processed_dir": args.processed_dir, "train_list": args.train_list,
        "test_list": args.test_list, "train_step4_dir": args.train_step4_dir,
        "test_step4_dir": args.test_step4_dir,
        "weight_tensor": args.weight_tensor,
        "enriched_model_dir": args.enriched_model_dir,
        "train_scope_dir": args.train_scope_dir,
        "train_maestro_dir": args.train_maestro_dir,
    }
    if args.train_scope_dir is None or args.train_maestro_dir is None:
        print("NOTE: --train-scope-dir/--train-maestro-dir not both set; "
              "ablation will use its deterministic train-side fallback for "
              "the missing one (still backbone-independent).")
    only = {t.strip() for t in args.only.split(",")} if args.only else None

    backbone_rows: list[dict] = []
    for bk in BACKBONES:
        if only is not None and bk["tag"] not in only:
            continue
        dirs = cell_dirs(args.work_dir, bk["tag"])

        if (not args.force and dirs["ablation_csv"].is_file()):
            print(f"[{bk['tag']}] ablation CSV exists — reading "
                  f"(--force to regenerate)")
            backbone_rows.append(_backbone_row(bk, dirs, "cached"))
            continue

        # Shared the LLM relay key; fall back to the legacy per-backbone env var.
        api_key = (os.environ.get("LLM_RELAY_API_KEY")
                   or os.environ.get(bk["key_env"]))
        if not api_key:
            print(f"[{bk['tag']}] SKIP — neither LLM_RELAY_API_KEY nor "
                  f"{bk['key_env']} set")
            backbone_rows.append({"backbone": bk["name"],
                                  "pearson_r_mean": None,
                                  "spearman_r_mean": None, "r2_mean": None,
                                  "pct_fail": None, "n_samples": None,
                                  "status": "skipped_no_key"})
            continue

        # --force: wipe ALL of this backbone's intermediate products (scope /
        # maestro / polish dirs + ablation CSV) so nothing stale is reused and
        # every test sample is re-sent to the API. The generators don't skip
        # existing per-sample JSONs, but a clean slate also drops outputs for
        # any sample no longer in the list.
        if args.force:
            cell_root = args.work_dir / bk["tag"]
            if cell_root.exists():
                shutil.rmtree(cell_root)
                print(f"[{bk['tag']}] --force: cleared {cell_root} "
                      f"(scope/maestro/polish + ablation CSV)")

        print(f"[{bk['tag']}] generating + ablating via {bk['model']} "
              f"@ {args.base_url} ...")
        try:
            run_backbone_generation(bk, args.base_url, api_key, paths, dirs)
            status = "run"
        except RuntimeError as e:
            print(f"[{bk['tag']}] FAILED: {e}", file=sys.stderr)
            backbone_rows.append({"backbone": bk["name"],
                                  "pearson_r_mean": None,
                                  "spearman_r_mean": None, "r2_mean": None,
                                  "pct_fail": None, "n_samples": None,
                                  "status": f"error: {e}"})
            continue
        backbone_rows.append(_backbone_row(bk, dirs, status))

    # Assemble final table: No-LLM reference, the backbones, LLM reference.
    rows = [_reference_row(_REF_NO_LLM, args.done_ablation_csv)]
    rows += backbone_rows
    rows.append(_reference_row(_REF_LLM, args.done_ablation_csv))

    write_csv(args.output, rows)
    print(f"wrote {args.output}  ({len(rows)} rows)")
    print()
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
