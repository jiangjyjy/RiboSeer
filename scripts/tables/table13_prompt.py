#!/usr/bin/env python3
"""Table 14 — prompt-strategy ablation (CoT × Temperature).

Ablates two prompt-engineering knobs shared by the SCOPE / MAESTRO / POLISH
LLM prompts:

* **CoT** — chain-of-thought on/off (``--cot`` / ``--no-cot``).
* **T**   — sampling temperature (``--temperature``).

Five cells (the CoT=on, T=0.7 cell is the shipped default — Table 9
on/on/on = 0.565 — and is reused, not re-run, when ``--done-ablation-csv``
is given):

    CoT  T
    –    0.0
    –    0.7
    ✓    0.0
    ✓    0.7   ← default (reused)
    ✓    1.0

For each cell this driver runs, into ``<work-dir>/<tag>/``:

1. SCOPE profiles  (train + test)   — generate_scope_profiles --mode llm
2. MAESTRO selections (train + test)— generate_maestro_selections --mode llm
3. POLISH iterative actions (test)  — generate_polish_iterative --mode llm --max-rounds 5
4. table09_llm_modules.py          — reads the scope=on/maestro=on/polish=on row

each generator invoked with this cell's ``--cot/--no-cot`` and
``--temperature``. It then reads the on/on/on Pearson/Spearman/R² and the
**% schema-validation failure rate** (the fraction of LLM calls that fell
back because the response didn't parse to the expected JSON — tracked via
the generators' ``source`` tags: ``cauto_fallback`` / ``llm_fallback``).

Cost: 4 fresh cells × ~800 LLM calls ≈ 3200 (LLM 5.1). Needs ``LLM_API_KEY``.
Use ``--only`` to run one cell at a time.

Usage
-----
::

    python scripts/tables/table13_prompt.py \\
        --processed-dir   data/processed_quality \\
        --train-list      data/processed_quality/splits_tmscore_035/train.txt \\
        --test-list       data/processed_quality/splits_tmscore_035/test.txt \\
        --train-step4-dir data/batch_train_v7/step4/ \\
        --test-step4-dir  data/batch_test_v7/step4/ \\
        --weight-tensor   data/batch_train_v7/W_v2.json \\
        --enriched-model-dir data/enriched_v7_lgbm/ \\
        --step6-dir       data/batch_test_v7/step6/ \\
        --scope-config    configs/step2_config.yaml \\
        --maestro-config  configs/step3_config.yaml \\
        --polish-config   configs/step7_config.yaml \\
        --work-dir        data/batch_test_v7/table14/ \\
        --done-ablation-csv data/batch_test_v7/table09_llm_modules.csv \\
        --output          data/batch_test_v7/table14_prompt.csv
"""
from __future__ import annotations

import argparse
import csv
import json
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

# (cot, temperature, is_shipped_default)
CONFIGS: list[dict] = [
    {"cot": False, "temperature": 0.0},
    {"cot": False, "temperature": 0.7},
    {"cot": True, "temperature": 0.0},
    {"cot": True, "temperature": 0.7, "default": True},
    {"cot": True, "temperature": 1.0},
]

_ON_ON_ON = {"scope": "on", "maestro": "on", "polish": "on"}


def config_tag(cfg: dict) -> str:
    """Filesystem-safe tag, e.g. CoT=on/T=0.7 → ``cot1_t07``."""
    return (f"cot{1 if cfg['cot'] else 0}"
            f"_t{int(round(cfg['temperature'] * 10)):02d}")


def is_default(cfg: dict) -> bool:
    return bool(cfg.get("default"))


# ---------------------------------------------------------------------------
# Per-cell directory layout + generator argv
# ---------------------------------------------------------------------------


def cell_dirs(work_dir: Path, cfg: dict) -> dict[str, Path]:
    d = work_dir / config_tag(cfg)
    return {
        "scope_train": d / "scope_profiles_train",
        "scope_test": d / "scope_profiles_test",
        "maestro_train": d / "maestro_selections_train",
        "maestro_test": d / "maestro_selections_test",
        "polish_test": d / "polish_actions_iterative",
        "ablation_csv": d / "table09_llm_modules.csv",
    }


def _cot_flag(cfg: dict) -> str:
    return "--cot" if cfg["cot"] else "--no-cot"


def _style_args(cfg: dict) -> list[str]:
    return [_cot_flag(cfg), "--temperature", str(cfg["temperature"])]


def build_stage_argvs(cfg: dict, paths: dict, dirs: dict) -> dict[str, list]:
    """The argv list for each generation stage + the ablation (all strings)."""
    style = _style_args(cfg)
    scope_common = ["--processed-dir", str(paths["processed_dir"]),
                    "--mode", "llm", "--config", str(paths["scope_config"])]
    maestro_common = ["--processed-dir", str(paths["processed_dir"]),
                      "--mode", "llm", "--config", str(paths["maestro_config"]),
                      "--weight-tensor", str(paths["weight_tensor"])]
    return {
        "scope_train": scope_common + [
            "--sample-list", str(paths["train_list"]),
            "--out-dir", str(dirs["scope_train"])] + style,
        "scope_test": scope_common + [
            "--sample-list", str(paths["test_list"]),
            "--out-dir", str(dirs["scope_test"])] + style,
        "maestro_train": maestro_common + [
            "--sample-list", str(paths["train_list"]),
            "--step4-dir", str(paths["train_step4_dir"]),
            "--scope-profiles", str(dirs["scope_train"]),
            "--out-dir", str(dirs["maestro_train"])] + style,
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
            "--step6-dir", str(paths["step6_dir"]),
            "--out-dir", str(dirs["polish_test"]),
            "--mode", "llm", "--config", str(paths["polish_config"]),
            "--max-rounds", "5"] + style,
        "ablation": [
            "--train-step4-dir", str(paths["train_step4_dir"]),
            "--test-step4-dir", str(paths["test_step4_dir"]),
            "--processed-dir", str(paths["processed_dir"]),
            "--train-list", str(paths["train_list"]),
            "--test-list", str(paths["test_list"]),
            "--scope-profiles-train", str(dirs["scope_train"]),
            "--scope-profiles-test", str(dirs["scope_test"]),
            "--maestro-selections-train", str(dirs["maestro_train"]),
            "--maestro-selections-test", str(dirs["maestro_test"]),
            "--polish-actions", str(dirs["polish_test"]),
            "--output", str(dirs["ablation_csv"])],
    }


def run_cell_generation(cfg: dict, paths: dict, dirs: dict) -> None:
    """Run the 4 generation stages + ablation for one cell (server-side;
    each call hits the LLM API). Raises if any stage returns non-zero."""
    argvs = build_stage_argvs(cfg, paths, dirs)
    for stage, mod in (("scope_train", gsp), ("scope_test", gsp),
                       ("maestro_train", gm), ("maestro_test", gm),
                       ("polish_test", gpi)):
        rc = mod.main(argvs[stage])
        if rc != 0:
            raise RuntimeError(f"stage {stage} failed (rc={rc})")
    rc = alm.main(argvs["ablation"])
    if rc != 0:
        raise RuntimeError(f"ablation failed (rc={rc})")


# ---------------------------------------------------------------------------
# Schema-validation failure rate (from the generators' ``source`` tags)
# ---------------------------------------------------------------------------


def _iter_json(directory: Path):
    if not directory.is_dir():
        return
    for f in sorted(directory.glob("*.json")):
        try:
            yield json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue


def _scan_flat_source(directory: Path, fail_tag: str) -> tuple[int, int]:
    """(attempts, fails) over files carrying a top-level ``source`` that is
    either an LLM success or ``fail_tag``."""
    attempts = fails = 0
    for rec in _iter_json(directory):
        src = rec.get("source")
        if src == "llm":
            attempts += 1
        elif src == fail_tag:
            attempts += 1
            fails += 1
    return attempts, fails


def _scan_round_source(directory: Path) -> tuple[int, int]:
    """(attempts, fails) over every round of the iterative POLISH records."""
    attempts = fails = 0
    for rec in _iter_json(directory):
        for rd in rec.get("rounds") or []:
            src = rd.get("source")
            if src == "llm":
                attempts += 1
            elif src == "llm_fallback":
                attempts += 1
                fails += 1
    return attempts, fails


def failure_rate(dirs: dict) -> dict:
    """Combined schema-validation failure rate across SCOPE + MAESTRO +
    POLISH for one cell."""
    a, f = 0, 0
    for d, tag in ((dirs["scope_train"], "cauto_fallback"),
                   (dirs["scope_test"], "cauto_fallback"),
                   (dirs["maestro_train"], "llm_fallback"),
                   (dirs["maestro_test"], "llm_fallback")):
        da, df = _scan_flat_source(d, tag)
        a += da
        f += df
    pa, pf = _scan_round_source(dirs["polish_test"])
    a += pa
    f += pf
    return {"attempts": a, "fails": f,
            "pct_fail": round(100.0 * f / a, 1) if a else None}


# ---------------------------------------------------------------------------
# Read the on/on/on ablation row
# ---------------------------------------------------------------------------


def read_onon_row(csv_path: Path) -> Optional[dict]:
    if not csv_path.is_file():
        return None
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            if all(rec.get(k) == v for k, v in _ON_ON_ON.items()):
                return rec
    return None


def _num(s):
    try:
        return round(float(s), 4)
    except (TypeError, ValueError):
        return None


def assemble_row(cfg: dict, ablation_csv: Path, fail: dict,
                 reused: bool) -> dict:
    rec = read_onon_row(ablation_csv)
    return {
        "cot": int(cfg["cot"]),
        "temperature": cfg["temperature"],
        "pearson_r_mean": _num(rec.get("pearson_r_mean")) if rec else None,
        "spearman_r_mean": _num(rec.get("spearman_r_mean")) if rec else None,
        "r2_mean": _num(rec.get("r2_mean")) if rec else None,
        "pct_fail": fail.get("pct_fail") if fail else None,
        "n_samples": int(rec["n_samples"]) if rec and rec.get("n_samples")
        else None,
        "status": "reused" if reused else "run",
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_COLUMNS = ["cot", "temperature", "pearson_r_mean", "spearman_r_mean",
            "r2_mean", "pct_fail", "n_samples", "status"]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in _COLUMNS})


def print_table(rows: list[dict]) -> None:
    hdr = (f"{'CoT':>3s} {'T':>4s} {'PearsonR':>9s} {'SpearmanR':>10s} "
           f"{'R2':>8s} {'%Fail':>6s} {'status':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cot = "✓" if r["cot"] else "–"
        pf = f"{r['pct_fail']}%" if r["pct_fail"] is not None else "–"
        print(f"{cot:>3s} {r['temperature']:>4} "
              f"{str(r['pearson_r_mean']):>9s} "
              f"{str(r['spearman_r_mean']):>10s} {str(r['r2_mean']):>8s} "
              f"{pf:>6s} {r['status']:>7s}")


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
    p.add_argument("--step6-dir", type=Path, required=True)
    p.add_argument("--scope-config", type=Path, required=True)
    p.add_argument("--maestro-config", type=Path, required=True)
    p.add_argument("--polish-config", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--done-ablation-csv", type=Path, default=None,
                   help="existing on/on/on ablation CSV for the shipped "
                        "default cell (CoT=on, T=0.7) — reused, not re-run")
    p.add_argument("--only", type=str, default=None,
                   help="comma list of cell tags to run (e.g. cot0_t00); "
                        "default = all cells")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    paths = {
        "processed_dir": args.processed_dir, "train_list": args.train_list,
        "test_list": args.test_list, "train_step4_dir": args.train_step4_dir,
        "test_step4_dir": args.test_step4_dir,
        "weight_tensor": args.weight_tensor,
        "enriched_model_dir": args.enriched_model_dir,
        "step6_dir": args.step6_dir, "scope_config": args.scope_config,
        "maestro_config": args.maestro_config,
        "polish_config": args.polish_config,
    }
    only = {t.strip() for t in args.only.split(",")} if args.only else None

    rows: list[dict] = []
    for cfg in CONFIGS:
        tag = config_tag(cfg)
        if only is not None and tag not in only:
            continue
        dirs = cell_dirs(args.work_dir, cfg)
        reused = is_default(cfg) and args.done_ablation_csv is not None
        if reused:
            dirs["ablation_csv"] = args.done_ablation_csv
            print(f"[{tag}] reuse default ablation {args.done_ablation_csv}")
        else:
            print(f"[{tag}] generating + ablating (cot={cfg['cot']}, "
                  f"T={cfg['temperature']}) ...")
            run_cell_generation(cfg, paths, dirs)
        fail = failure_rate(dirs)
        rows.append(assemble_row(cfg, dirs["ablation_csv"], fail, reused))

    write_csv(args.output, rows)
    print(f"wrote {args.output}  ({len(rows)} cells)")
    print()
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
