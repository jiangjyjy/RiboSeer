"""Full pipeline end-to-end smoke (server-only).

NOT a unittest module. The final acceptance script for the project:
runs steps 1-8 in sequence on ONE sample and prints a per-step
summary. Designed to be the "does the whole thing work?" check before
batch runs.

Pipeline flow (paper figure 2)::

    [step 1] processed/<id>.json    (sample loader, already cached)
        ↓
    [step 2] target_char           (LLM → category)
        ↓
    [step 3] tool selection        (LLM → tool plan)
        ↓
    [step 4] run tools             (boltz2 / rf2na / p2rank / equipnas)
        ↓
    [step 5] fuse predictions      (LLM → tool weights → noisy-OR)
        ↓
    [step 6] pocket QA             (5 sub-scores → total)
        ↓
    [step 7] iterate               (LLM → accept / refine / restart)
        ↓
    [step 8] update W              (EMA + optional γ correction)

Each step runs in a separate subprocess (so module state can't
contaminate the next step) and writes its JSONL into a per-step
``data/step{N}_outputs/<sample>.jsonl`` file. The script prints what
landed there before invoking the next step.

Run on the server, e.g.::

    cd ~/riboseer
    conda activate riboseer
    export LLM_API_KEY=<your key>

    python tests/test_full_pipeline_e2e.py \\
        --sample-id 1un6_B_F \\
        --processed-dir data/processed \\
        --output-root data/full_pipeline_smoke

The script never starts step N+1 if step N produced no record for the
sample id (it would only fail later anyway). On the first failure it
prints the offending step's stderr and exits non-zero.

Cost
----
Per sample, end-to-end with all four tools and meta-correction OFF:
  - LLM tokens: ~6000-12000 (steps 2, 3, 5, 7)
  - Tool runtime: dominated by step 4 (Cat A tools take hours; CPU-only
    Cat B/C take minutes). For a quick smoke, restrict --tools to
    p2rank,equipnas to skip the structure-prediction Cat A path.

Skipping flags
--------------
- ``--skip <step>``  repeatable. Re-uses the existing JSONL under
  ``--output-root/step{N}_outputs/<sid>.jsonl`` if you don't want to
  re-run a step (handy for smoke runs that re-iterate steps 7 / 8).
- ``--no-llm``  Disables LLM in steps 7 and 8 (steps 2 / 3 / 5 always
  need LLM; if you want a tool-free smoke, also pass ``--skip 4``).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"


# ---------- IO helpers -----------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    return rows


def _record_for(records: Iterable[dict], sid: str) -> Optional[dict]:
    for r in records:
        if r.get("sample_id") == sid:
            return r
    return None


def _print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def _run_step(label: str, argv: list[str], *, env: Optional[dict] = None) -> int:
    """Run a step in a subprocess.

    Returns the exit code; on non-zero, dumps stderr and returns the code
    so the caller can decide whether to abort.
    """
    print(f"[{label}] $ python {' '.join(shlex.quote(a) for a in argv)}")
    sub_env = dict(os.environ)
    sub_env["PYTHONPATH"] = str(SRC) + os.pathsep + sub_env.get("PYTHONPATH", "")
    if env:
        sub_env.update(env)
    proc = subprocess.run(
        [sys.executable, *argv],
        env=sub_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.stdout:
        # Indent so the parent's output stays readable.
        for line in proc.stdout.rstrip("\n").splitlines():
            print(f"  | {line}")
    if proc.returncode != 0:
        print(f"[{label}] EXIT {proc.returncode}")
        if proc.stderr:
            for line in proc.stderr.rstrip("\n").splitlines():
                print(f"  ! {line}")
    return proc.returncode


# ---------- per-step summarisers ------------------------------------------


def _summary_step1(sample_path: Path) -> str:
    if not sample_path.is_file():
        return "(no sample file)"
    obj = json.loads(sample_path.read_text(encoding="utf-8"))
    p_seq = (obj.get("protein") or {}).get("sequence") or ""
    r_seq = (obj.get("rna") or {}).get("sequence") or ""
    return (f"protein {len(p_seq)} aa, rna {len(r_seq)} nt; "
            f"sample_id={obj.get('sample_id')!r}")


def _summary_step2(rec: dict) -> str:
    out = rec.get("output") or {}
    return (f"category={out.get('category')!r}  "
            f"confidence={out.get('confidence')}  "
            f"tokens={(rec.get('api_usage') or {}).get('total_tokens', 0)}")


def _summary_step3(rec: dict) -> str:
    plan = rec.get("output") or {}
    selected = plan.get("selected_tools") or []
    return (f"selected={selected}  "
            f"tokens={(rec.get('api_usage') or {}).get('total_tokens', 0)}")


def _summary_step4(rec: dict) -> str:
    preds = rec.get("predictions") or []
    ok = [p for p in preds if p.get("success")]
    fail = [p for p in preds if not p.get("success")]
    return (f"{len(ok)}/{len(preds)} succeeded "
            f"({[p.get('tool_id') for p in ok]}); "
            f"{len(fail)} failed")


def _summary_step5(rec: dict) -> str:
    weights = rec.get("tool_weights") or {}
    bp = rec.get("binding_protein_residues") or []
    return (f"weights={weights}  "
            f"binding_protein={len(bp)} res; "
            f"f1={rec.get('f1')}  "
            f"tokens={(rec.get('api_usage') or {}).get('total_tokens', 0)}")


def _summary_step6(rec: dict) -> str:
    return (f"total={rec.get('total_score'):.3f}  "
            f"q1={rec.get('structural_plausibility')}  "
            f"q2={rec.get('physicochemical_complementarity')}  "
            f"q3={rec.get('evolutionary_conservation')}  "
            f"q4={rec.get('cross_tool_consensus')}  "
            f"q5={rec.get('known_motif_consistency')}  "
            f"({rec.get('n_metrics_computed')}/5 computed)")


def _summary_step7(rec: dict) -> str:
    return (f"final_score={rec.get('final_score'):.3f}  "
            f"reason={rec.get('termination_reason')}  "
            f"iters={rec.get('total_iterations')}  "
            f"trajectory={[round(s, 3) for s in rec.get('score_trajectory') or []]}")


def _summary_step8(rec: dict) -> str:
    tools = rec.get("tools_updated") or []
    meta = "META" if rec.get("meta_correction_applied") else "ema-only"
    return (f"updated={tools}  mode={meta}  "
            f"category={rec.get('category')!r}")


# ---------- main -----------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-id", type=str, required=True,
                   help="single sample id to drive end-to-end")
    p.add_argument("--processed-dir", type=Path,
                   default=REPO / "data" / "processed",
                   help="step 1 output dir (sample json lives at "
                        "samples/<id>.json)")
    p.add_argument("--output-root", type=Path,
                   default=REPO / "data" / "full_pipeline_smoke",
                   help="root directory for all step{N}_outputs/<id>.jsonl")
    p.add_argument("--configs", type=Path, default=REPO / "configs")
    p.add_argument("--tools", type=str, default=None,
                   help="comma-separated tool list for step 4 "
                        "(default: full plan from step 3)")
    p.add_argument("--skip", type=int, action="append", default=[],
                   help="step N to skip (1..8, repeatable). Re-uses the "
                        "existing JSONL under --output-root for that step.")
    p.add_argument("--no-llm", action="store_true",
                   help="disable LLM in step 7 / 8 (steps 2/3/5 still "
                        "need it)")
    p.add_argument("--abort-on-failure", action="store_true", default=True,
                   help="(default) stop the pipeline on first non-zero "
                        "exit; pass --no-abort-on-failure to keep going")
    p.add_argument("--no-abort-on-failure", dest="abort_on_failure",
                   action="store_false")
    args = p.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    sid = args.sample_id
    out = args.output_root
    cfg = args.configs

    # Per-step JSONL outputs (one record per call, sample-keyed).
    #
    # Output-arg conventions across steps:
    #   - step 2 / 3            : --output is always a FILE
    #   - step 4 (run_all.py)   : --output is always a DIRECTORY; step
    #     4 writes <dir>/<sample_id>.jsonl per sample. Passing a file
    #     path here would create a directory with that file's name and
    #     write the jsonl INSIDE it, leaving the read-back at the wrong
    #     path. Hence ``s4_dir`` (passed in) vs ``s4_path`` (read from).
    #   - step 5 / 6 / 7 / 8    : --output auto-detects from suffix:
    #     ``.jsonl`` → file mode; otherwise dir mode + per-sample file.
    #     We pass file paths here for those four steps.
    s1_path = args.processed_dir / "samples" / f"{sid}.json"
    s2_path = out / "step2_outputs" / f"{sid}.jsonl"
    s3_path = out / "step3_outputs" / f"{sid}.jsonl"
    s4_dir  = out / "step4_outputs"
    s4_path = s4_dir / f"{sid}.jsonl"
    s5_path = out / "step5_outputs" / f"{sid}.jsonl"
    s6_path = out / "step6_outputs" / f"{sid}.jsonl"
    s7_path = out / "step7_outputs" / f"{sid}.jsonl"
    s8_path = out / "step8_outputs" / f"{sid}.jsonl"

    weight_tensor_path = out / "weights" / "tensor.json"
    history_path = out / "history" / "history.jsonl"

    # ---- step 1: just verify the sample exists -------------------------
    _print_header(f"STEP 1 — sample loader (cached)  sid={sid}")
    if not s1_path.is_file():
        print(f"ERROR: step 1 sample file missing: {s1_path}", file=sys.stderr)
        print("       Run extract_pairs / scan_dataset first.", file=sys.stderr)
        return 1
    print(f"  {_summary_step1(s1_path)}")

    # ---- step 2: target_char --------------------------------------------
    _print_header("STEP 2 — target characterisation (LLM)")
    if 2 in args.skip:
        print(f"  [skipped]  (re-using {s2_path})")
    else:
        rc = _run_step("step2", [
            "-m", "step2_target_char.run",
            "--processed-dir", str(args.processed_dir),
            "--sample-id", sid,
            "--config", str(cfg / "step2_config.yaml"),
            "--output", str(s2_path),
        ])
        if rc != 0 and args.abort_on_failure:
            return rc
    s2_rec = _record_for(_load_jsonl(s2_path), sid)
    if s2_rec is None:
        print(f"ERROR: step 2 produced no record for {sid}", file=sys.stderr)
        return 1
    print(f"  → {_summary_step2(s2_rec)}")

    # ---- step 3: tool selection ----------------------------------------
    _print_header("STEP 3 — tool selection (LLM + UCB)")
    # NOTE: step 3 reads its weight tensor path from step3_config.yaml
    # (``weight_tensor_path``), not a CLI flag. The full-pipeline tensor
    # at ``--output-root/weights/tensor.json`` is therefore NOT shared
    # with step 3 in this smoke; step 3's UCB uses the in-repo default
    # path. For a reinforcement-loop test (W feeding back into step 3
    # selection across runs), point step3_config's weight_tensor_path
    # to the same path step 8 writes to.
    if 3 in args.skip:
        print(f"  [skipped]  (re-using {s3_path})")
    else:
        rc = _run_step("step3", [
            "-m", "step3_tool_selection.run",
            "--processed-dir", str(args.processed_dir),
            "--step2-output", str(s2_path),
            "--sample-id", sid,
            "--config", str(cfg / "step3_config.yaml"),
            "--output", str(s3_path),
        ])
        if rc != 0 and args.abort_on_failure:
            return rc
    s3_rec = _record_for(_load_jsonl(s3_path), sid)
    if s3_rec is None:
        print(f"ERROR: step 3 produced no record for {sid}", file=sys.stderr)
        return 1
    print(f"  → {_summary_step3(s3_rec)}")

    # ---- step 4: run tools ---------------------------------------------
    _print_header("STEP 4 — run tools (boltz2 / rf2na / p2rank / equipnas)")
    if 4 in args.skip:
        print(f"  [skipped]  (re-using {s4_path})")
    else:
        argv4 = [
            "-m", "step4_tool_adapters.run_all",
            "--processed-dir", str(args.processed_dir),
            "--step3-output", str(s3_path),
            "--sample-id", sid,
            "--config", str(cfg / "step4_config.yaml"),
            # ``run_all.py --output`` is always interpreted as a
            # directory; the per-sample JSONL lands at
            # ``<dir>/<sample_id>.jsonl`` (which is ``s4_path``).
            "--output", str(s4_dir),
        ]
        if args.tools:
            argv4 += ["--tools", args.tools]
        rc = _run_step("step4", argv4)
        if rc != 0 and args.abort_on_failure:
            return rc
    s4_rec = _record_for(_load_jsonl(s4_path), sid)
    if s4_rec is None:
        print(f"ERROR: step 4 produced no record for {sid}", file=sys.stderr)
        return 1
    print(f"  → {_summary_step4(s4_rec)}")

    # ---- step 5: fusion -------------------------------------------------
    _print_header("STEP 5 — fusion (LLM tool weighting + noisy-OR)")
    if 5 in args.skip:
        print(f"  [skipped]  (re-using {s5_path})")
    else:
        rc = _run_step("step5", [
            "-m", "step5_fusion.run",
            "--processed-dir", str(args.processed_dir),
            "--step2-output", str(s2_path),
            "--step4-output", str(s4_path),
            "--sample-id", sid,
            "--config", str(cfg / "step5_config.yaml"),
            "--output", str(s5_path),
        ])
        if rc != 0 and args.abort_on_failure:
            return rc
    s5_rec = _record_for(_load_jsonl(s5_path), sid)
    if s5_rec is None:
        print(f"ERROR: step 5 produced no record for {sid}", file=sys.stderr)
        return 1
    print(f"  → {_summary_step5(s5_rec)}")

    # ---- step 6: pocket QA ---------------------------------------------
    _print_header("STEP 6 — pocket QA (5 sub-scores)")
    if 6 in args.skip:
        print(f"  [skipped]  (re-using {s6_path})")
    else:
        rc = _run_step("step6", [
            "-m", "step6_pocket_qa.run",
            "--processed-dir", str(args.processed_dir),
            "--step2-output", str(s2_path),
            "--step4-output", str(s4_path),
            "--step5-output", str(s5_path),
            "--sample-id", sid,
            "--config", str(cfg / "step6_config.yaml"),
            "--output", str(s6_path),
        ])
        if rc != 0 and args.abort_on_failure:
            return rc
    s6_rec = _record_for(_load_jsonl(s6_path), sid)
    if s6_rec is None:
        print(f"ERROR: step 6 produced no record for {sid}", file=sys.stderr)
        return 1
    print(f"  → {_summary_step6(s6_rec)}")

    # ---- step 7: iteration loop ----------------------------------------
    _print_header("STEP 7 — accept / refine / restart loop (LLM)")
    if 7 in args.skip:
        print(f"  [skipped]  (re-using {s7_path})")
    else:
        argv7 = [
            "-m", "step7_iteration.run",
            "--processed-dir", str(args.processed_dir),
            "--step2-output", str(s2_path),
            "--step4-output", str(s4_path),
            "--step5-output", str(s5_path),
            "--step6-output", str(s6_path),
            "--sample-id", sid,
            "--config", str(cfg / "step7_config.yaml"),
            "--output", str(s7_path),
        ]
        if args.no_llm:
            argv7.append("--no-llm")
        rc = _run_step("step7", argv7)
        if rc != 0 and args.abort_on_failure:
            return rc
    s7_rec = _record_for(_load_jsonl(s7_path), sid)
    if s7_rec is None:
        print(f"ERROR: step 7 produced no record for {sid}", file=sys.stderr)
        return 1
    print(f"  → {_summary_step7(s7_rec)}")

    # ---- step 8: weight update -----------------------------------------
    _print_header("STEP 8 — weight tensor update (EMA + optional γ)")
    if 8 in args.skip:
        print(f"  [skipped]  (re-using {s8_path})")
    else:
        argv8 = [
            "-m", "step8_weight_update.run",
            "--processed-dir", str(args.processed_dir),
            "--step2-output", str(s2_path),
            "--step4-output", str(s4_path),
            "--step5-output", str(s5_path),
            "--step6-output", str(s6_path),
            "--step7-output", str(s7_path),
            "--weight-tensor", str(weight_tensor_path),
            "--history", str(history_path),
            "--sample-id", sid,
            "--config", str(cfg / "step8_config.yaml"),
            "--output", str(s8_path),
        ]
        if args.no_llm:
            argv8.append("--no-llm")
        rc = _run_step("step8", argv8)
        if rc != 0 and args.abort_on_failure:
            return rc
    s8_rec = _record_for(_load_jsonl(s8_path), sid)
    if s8_rec is None:
        print(f"ERROR: step 8 produced no record for {sid}", file=sys.stderr)
        return 1
    print(f"  → {_summary_step8(s8_rec)}")

    # ---- final summary --------------------------------------------------
    _print_header("FULL PIPELINE COMPLETE")
    print(f"  sample        : {sid}")
    print(f"  category      : {(s2_rec.get('output') or {}).get('category')}")
    print(f"  selected tools: {(s3_rec.get('output') or {}).get('selected_tools')}")
    print(f"  step 4 ok     : "
          f"{[p.get('tool_id') for p in s4_rec.get('predictions') or [] if p.get('success')]}")
    print(f"  step 5 weights: {s5_rec.get('tool_weights')}")
    print(f"  step 6 total  : {s6_rec.get('total_score'):.3f}  "
          f"(precision/recall/f1: {s5_rec.get('precision')}/"
          f"{s5_rec.get('recall')}/{s5_rec.get('f1')})")
    print(f"  step 7 final  : {s7_rec.get('final_score'):.3f}  "
          f"({s7_rec.get('termination_reason')})")
    print(f"  step 8 updated: {s8_rec.get('tools_updated')}")
    print()
    print(f"  artifacts root: {out}")
    print(f"  weight tensor : {weight_tensor_path}")
    print(f"  history       : {history_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
