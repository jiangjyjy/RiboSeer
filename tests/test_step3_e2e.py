"""End-to-end real-API test for Step 3 — RUN ON SERVER ONLY.

Reads the 5 samples from step 2's e2e output and runs tool selection on each.
Expected cost: ~15-25k tokens (5 samples × ~3-5k tokens each).

Usage (on the server):
    cd ~/riboseer
    conda activate riboseer
    export LLM_API_KEY="<your real key>"
    python tests/test_step3_e2e.py \
        --processed-dir data/processed \
        --step2-output data/step2_outputs/e2e_smoke.jsonl \
        --config configs/step3_config.yaml \
        --output data/step3_outputs/e2e_smoke.jsonl

What we're checking
-------------------
  - Do different categories get different tool selections?
  - Does RRM_x_stem_loop prioritize Category C tools?
  - Does novel_fold_x_junction lean toward A/D tools?
  - Does the all-X protein edge case (3j92_x_5) get reasonable handling?
  - Is the rationale biologically coherent?
  - Any retries needed? (0 is ideal)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from step2_target_char.llm_client import LLMClient  # noqa: E402
from step2_target_char.run import build_client, load_config as load_step2_config  # noqa: E402
from step3_tool_selection.tool_selector import select_tools  # noqa: E402
from step3_tool_selection.weight_tensor import WeightTensor  # noqa: E402


def _banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _dump_record(record: dict) -> None:
    plan = record.get("tool_plan") or {}
    usage = record.get("api_usage") or {}
    status = "OK" if record.get("success") else "FAIL"
    print(f"  status:     {status}")
    print(f"  category:   {record.get('step2_category')}")
    print(f"  tools:      {plan.get('selected_tools')}")
    print(f"  strategy:   {plan.get('execution_strategy')}")
    print(f"  confidence: {plan.get('confidence')}")
    print(f"  retries:    {record.get('retries')}")
    print(f"  tokens:     prompt={usage.get('prompt_tokens')} "
          f"completion={usage.get('completion_tokens')} "
          f"total={usage.get('total_tokens')}")
    print(f"  rationale:\n    {plan.get('rationale')}")
    overrides = plan.get("param_overrides") or []
    if overrides:
        print(f"  param_overrides:")
        for ov in overrides:
            print(f"    - {ov}")
    es = plan.get("early_stop_threshold")
    if es is not None:
        print(f"  early_stop: {es}")
    if not record.get("success"):
        print(f"  FAILURE: {record.get('failure_reason')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--processed-dir", type=Path, required=True)
    ap.add_argument("--step2-output", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=REPO / "configs" / "step3_config.yaml")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not os.environ.get("LLM_API_KEY"):
        print("ERROR: LLM_API_KEY is not exported.")
        return 2

    # Load configs
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    step2_cfg_path = args.config.parent / "step2_config.yaml"
    step2_cfg = load_step2_config(step2_cfg_path)
    client = build_client(step2_cfg)

    wt_path = Path(config.get("weight_tensor_path", "data/step3_weights/weight_tensor.json"))
    wt = WeightTensor.load(wt_path) if wt_path.is_file() else WeightTensor.from_config(config)

    # Load step 2 records
    s2_records = []
    with args.step2_output.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                s2_records.append(json.loads(line))
    print(f"Loaded {len(s2_records)} step-2 records from {args.step2_output}")

    out_f = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_f = args.output.open("w", encoding="utf-8")

    n_ok = n_fail = 0
    total_tokens = 0
    try:
        for s2 in s2_records:
            sid = s2.get("sample_id", "?")
            tc = s2.get("output") or {}
            tf = s2.get("input_features") or {}

            _banner(f"SAMPLE {sid}  (step2: {tc.get('category')})")
            record = select_tools(
                sample_id=sid,
                target_char=tc,
                target_features=tf,
                client=client,
                weight_tensor=wt,
                config=config,
            )
            _dump_record(record)

            if out_f:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

            if record.get("success"):
                n_ok += 1
            else:
                n_fail += 1
            total_tokens += (record.get("api_usage") or {}).get("total_tokens", 0) or 0
    finally:
        if out_f:
            out_f.close()

    _banner("SUMMARY")
    print(f"  samples:    {n_ok + n_fail}  (ok={n_ok}  fail={n_fail})")
    print(f"  tokens:     {total_tokens}")
    if args.output:
        print(f"  jsonl:      {args.output}")
    print(f"  usage log:  {client.usage_log}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
