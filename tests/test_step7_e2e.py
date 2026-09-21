"""Real end-to-end smoke for Step 7 — accept/refine/restart loop (server-only).

NOT a unittest module. A manual script that drives ``run_iteration_loop``
on one or more real samples, using existing step 2 / 4 / 5 / 6 outputs and
a real ``LLMClient``.

Run on the server, e.g.::

    cd ~/riboseer
    conda activate riboseer
    export LLM_API_KEY=<your key>

    python tests/test_step7_e2e.py \\
        --processed-dir data/processed \\
        --step2-output data/step2_outputs/e2e_smoke.jsonl \\
        --step4-output data/step4_outputs/1un6_B_F.jsonl \\
        --step5-output data/step5_outputs/e2e_smoke.jsonl \\
        --step6-output data/step6_outputs/e2e_smoke.jsonl \\
        --sample-id 1un6_B_F \\
        --config configs/step7_config.yaml \\
        --output data/step7_outputs/e2e_smoke.jsonl

What it does
------------
- For each requested sample id, loads:
    * step 1 sample JSON (``--processed-dir/samples/<id>.json``)
    * step 2 record (target_char) — optional but recommended
    * step 4 record (ToolPredictionSet)
    * step 5 record (CompositeResult)
    * step 6 record (PocketQAResult — seeds the iteration's score)
- Calls ``run_iteration_loop`` with a real LLMClient (built from the
  step 2 config so transport settings stay consistent with steps 2/3/5).
- Prints a per-iteration block:
    * iteration index, action, score before → after, delta
    * the LLM's rationale (truncated)
    * cumulative token usage
- Writes the final ``IterationResult`` JSONL record to ``--output``.

This is the manually-driven counterpart to the unit-tested
``tests/test_step7_mock.py``. Heavy I/O and real-API calls live here so
the unit suite stays fast and offline.

Notes
-----
- Token budget per sample: ~1500-2500 tokens × N iterations (cap 3).
- ``--no-llm`` skips the LLM call entirely and synthesises an iter-0
  accept; useful for a transport-free sanity check before paying for
  real tokens.
- The script does not refresh upstream outputs. Make sure step 4 / 5 /
  6 already ran for each sample id you pass in.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step2_target_char.llm_client import LLMClient  # noqa: E402
from step2_target_char.run import build_client, load_config as load_step2_config  # noqa: E402
# Re-use step 6's record-rebuild helpers — same encoding/decoding logic.
from step6_pocket_qa.run import (  # noqa: E402
    _category_for as _step6_category_for,
    _index_by_sid,
    _load_jsonl,
    _load_sample,
    _step4_record_to_predictions,
    _step5_record_to_composite,
)
from step7_iteration.iterator import run_iteration_loop  # noqa: E402
from step7_iteration.run import _step6_record_to_qa  # noqa: E402


# ---------- I/O helpers -----------------------------------------------------


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _fmt_score(v: Optional[float]) -> str:
    return "  -  " if v is None else f"{v:.3f}"


def _fmt_delta(v: Optional[float]) -> str:
    if v is None:
        return "  n/a"
    return f"{v:+.3f}"


def _truncate(s: str, n: int = 200) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + " ..."


# ---------- main ------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed-dir", type=Path, required=True,
                    help="step 1 output dir containing samples/<id>.json")
    ap.add_argument("--step4-output", type=Path, required=True,
                    help="JSONL from step 4")
    ap.add_argument("--step5-output", type=Path, required=True,
                    help="JSONL from step 5 (CompositeResult per line)")
    ap.add_argument("--step6-output", type=Path, required=True,
                    help="JSONL from step 6 (PocketQAResult per line)")
    ap.add_argument("--step2-output", type=Path, default=None,
                    help="JSONL from step 2 (target_char per line); when "
                         "supplied, lets the prompt include the category. "
                         "Otherwise falls back to step 5's step2_category.")
    ap.add_argument("--config", type=Path,
                    default=REPO / "configs" / "step7_config.yaml",
                    help="step7_config.yaml")
    ap.add_argument("--step2-config", type=Path,
                    default=REPO / "configs" / "step2_config.yaml",
                    help="step2_config.yaml (for LLMClient transport)")
    ap.add_argument("--step5-config", type=Path,
                    default=REPO / "configs" / "step5_config.yaml",
                    help="step5_config.yaml (used for the inner re-fuse)")
    ap.add_argument("--step6-config", type=Path,
                    default=REPO / "configs" / "step6_config.yaml",
                    help="step6_config.yaml (used for the inner re-score)")
    ap.add_argument("--sample-id", type=str, action="append", default=None,
                    help="restrict to this sample id (repeatable). If "
                         "omitted, processes every sample present in "
                         "--step6-output.")
    ap.add_argument("--output", type=Path,
                    default=REPO / "data" / "step7_outputs" / "e2e_smoke.jsonl")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip LLM and synthesise iter-0 accept (offline check)")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # ----- configs --------------------------------------------------------
    cfg = _load_yaml(args.config)
    step2_cfg = load_step2_config(args.step2_config) if args.step2_config.is_file() else {}
    step5_cfg = _load_yaml(args.step5_config)
    step6_cfg = _load_yaml(args.step6_config)

    # ----- client ---------------------------------------------------------
    client: Optional[LLMClient]
    if args.no_llm:
        client = None
        print("[mode] --no-llm: LLM disabled; iter-0 will synthesise accept.")
    else:
        if not os.environ.get("LLM_API_KEY"):
            print("ERROR: LLM_API_KEY is not set. "
                  "Run with --no-llm for an offline check.", file=sys.stderr)
            return 1
        try:
            client = build_client(step2_cfg)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR building LLMClient: {e}", file=sys.stderr)
            return 1

    # ----- load JSONL inputs ---------------------------------------------
    s4_index = _index_by_sid(_load_jsonl(args.step4_output))
    s5_index = _index_by_sid(_load_jsonl(args.step5_output))
    s6_index = _index_by_sid(_load_jsonl(args.step6_output))
    s2_index = (
        _index_by_sid(_load_jsonl(args.step2_output))
        if args.step2_output is not None and args.step2_output.is_file()
        else {}
    )

    sample_ids = args.sample_id or sorted(s6_index.keys())
    if not sample_ids:
        print("ERROR: no sample ids resolved from --step6-output",
              file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.output.open("w", encoding="utf-8")

    cumulative_tokens = 0
    n_done = 0
    n_skipped = 0
    by_reason: dict[str, int] = {
        "accepted": 0, "converged": 0, "max_iterations": 0,
    }

    try:
        for sid in sample_ids:
            print()
            print("=" * 80)
            print(f"sample: {sid}")
            print("=" * 80)

            s4_record = s4_index.get(sid)
            s5_record = s5_index.get(sid)
            s6_record = s6_index.get(sid)
            if not s4_record:
                print("  SKIP — no step 4 record")
                n_skipped += 1
                continue
            if not s5_record:
                print("  SKIP — no step 5 record")
                n_skipped += 1
                continue
            if not s6_record:
                print("  SKIP — no step 6 record")
                n_skipped += 1
                continue

            try:
                sample_json = _load_sample(args.processed_dir, sid)
            except FileNotFoundError as e:
                print(f"  SKIP — {e}")
                n_skipped += 1
                continue
            try:
                composite = _step5_record_to_composite(s5_record)
                predictions = _step4_record_to_predictions(s4_record)
                qa_result = _step6_record_to_qa(s6_record)
            except Exception as e:  # noqa: BLE001
                print(f"  SKIP — record invalid: {e}")
                n_skipped += 1
                continue

            # Inject category for q5 if available (matches step 6 e2e behaviour).
            category = _step6_category_for(sid, s2_index, s5_record)
            scoring_input = dict(sample_json)
            s2_record = s2_index.get(sid) or {}
            target_char = s2_record.get("output") or s2_record or None
            if category:
                tc = dict(scoring_input.get("target_char") or {})
                tc.setdefault("category", category)
                scoring_input["target_char"] = tc

            # ----- print initial state ---------------------------------------
            print()
            print("--- initial state ---")
            print(f"  category        : {category or '(none)'}")
            print(f"  binding(prot)   : "
                  f"{len(composite.binding_protein_residues)} residues")
            print(f"  binding(rna)    : "
                  f"{len(composite.binding_rna_nucleotides)} nucleotides")
            print(f"  initial total_q : {qa_result.total_score:.4f}")
            print(f"  per-q           :  "
                  f"q1={_fmt_score(qa_result.structural_plausibility)}  "
                  f"q2={_fmt_score(qa_result.physicochemical_complementarity)}  "
                  f"q3={_fmt_score(qa_result.evolutionary_conservation)}  "
                  f"q4={_fmt_score(qa_result.cross_tool_consensus)}  "
                  f"q5={_fmt_score(qa_result.known_motif_consistency)}")

            # ----- run loop --------------------------------------------------
            result = run_iteration_loop(
                sample_json=scoring_input,
                target_char=target_char,
                tool_predictions=predictions,
                composite_result=composite,
                qa_result=qa_result,
                client=client,
                config=cfg,
                fusion_config=step5_cfg,
                qa_config=step6_cfg,
            )

            # ----- print iterations ------------------------------------------
            print()
            print("--- iterations ---")
            sample_tokens = 0
            for rec in result.iterations:
                a = rec.action
                extras = ""
                if a.action == "refine":
                    extras = f"  tool={a.refine_tool}"
                elif a.action == "restart":
                    pass
                # api_usage may have suppressed_action / status / etc.
                status = (rec.api_usage or {}).get("status", "")
                status_str = f"  [{status}]" if status else ""
                tok = (rec.api_usage or {}).get("total_tokens")
                if isinstance(tok, int):
                    sample_tokens += tok
                print(
                    f"  iter {rec.iteration}: {a.action:<8} "
                    f"score: {_fmt_score(rec.score_before)} -> "
                    f"{_fmt_score(rec.score_after)}  "
                    f"delta={_fmt_delta(rec.delta)}{extras}{status_str}"
                )
                print(f"    confidence: {a.confidence:.2f}")
                print(f"    rationale : {_truncate(a.rationale)}")
                if a.action == "refine" and a.refine_reason:
                    print(f"    refine_reason: {_truncate(a.refine_reason)}")
                if a.action == "restart" and a.restart_reason:
                    print(f"    restart_reason: {_truncate(a.restart_reason)}")

            # ----- print summary ---------------------------------------------
            print()
            print("--- summary ---")
            print(f"  total_iterations : {result.total_iterations}")
            print(f"  termination      : {result.termination_reason}")
            print(f"  trajectory       : "
                  f"{[round(s, 4) for s in result.score_trajectory]}")
            print(f"  final_score      : {result.final_score:.4f}")
            print(f"  tokens this run  : {sample_tokens}")

            cumulative_tokens += sample_tokens
            by_reason[result.termination_reason] = (
                by_reason.get(result.termination_reason, 0) + 1
            )
            n_done += 1

            out_f.write(
                json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
                + "\n"
            )
            out_f.flush()
    finally:
        out_f.close()

    print()
    print("=" * 80)
    print(
        f"Processed: {n_done}  skipped: {n_skipped}  "
        f"accepted={by_reason['accepted']}  "
        f"converged={by_reason['converged']}  "
        f"max_iter={by_reason['max_iterations']}"
    )
    print(f"Total tokens : {cumulative_tokens}")
    print(f"Wrote        : {args.output}")
    return 0 if n_skipped == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
