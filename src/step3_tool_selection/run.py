"""CLI entry point for Step 3 — tool selection.

Usage
-----
    # From step 2 JSONL output
    python -m src.step3_tool_selection.run \
        --processed-dir data/processed \
        --step2-output data/step2_outputs/e2e_smoke.jsonl \
        --config configs/step3_config.yaml \
        --output data/step3_outputs/tool_plan.jsonl

    # Single sample with inline step 2 run
    python -m src.step3_tool_selection.run \
        --processed-dir data/processed \
        --sample-id 1un6_B_F \
        --run-step2 \
        --config configs/step3_config.yaml

Requires `LLM_API_KEY` exported. Run on the server.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from step2_target_char.llm_client import LLMClient
from step2_target_char.run import build_client, load_config as load_step2_config
from .tool_selector import select_tools
from .weight_tensor import WeightTensor


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_step2_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _summarize(record: dict) -> str:
    plan = record.get("tool_plan") or {}
    usage = record.get("api_usage") or {}
    status = "OK " if record.get("success") else "FAIL"
    tools = plan.get("selected_tools", [])
    conf = plan.get("confidence", 0.0)
    return (
        f"[{status}] {record.get('sample_id', '?'):<16} "
        f"cat={record.get('step2_category', '?'):<28} "
        f"tools={','.join(tools):<50} "
        f"strategy={plan.get('execution_strategy', '?'):<10} "
        f"conf={conf:.2f}  "
        f"tokens={usage.get('total_tokens', 0)}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage 3 — run LLM tool selection over samples",
    )
    ap.add_argument("--processed-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True,
                    help="step3_config.yaml")
    ap.add_argument("--step2-config", type=Path, default=None,
                    help="step2_config.yaml (needed for --run-step2 mode)")
    ap.add_argument("--step2-output", type=Path, default=None,
                    help="JSONL from step 2 e2e run")
    ap.add_argument("--sample-id", type=str, default=None)
    ap.add_argument("--run-step2", action="store_true",
                    help="run step 2 inline before step 3 (needs API)")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_config(args.config)

    # Build LLM client from step2 config (reuse same API settings)
    step2_cfg_path = args.step2_config or (args.config.parent / "step2_config.yaml")
    step2_cfg = load_step2_config(step2_cfg_path)
    client = build_client(step2_cfg)

    # Load weight tensor
    wt_path = Path(config.get("weight_tensor_path", "data/step3_weights/weight_tensor.json"))
    wt = WeightTensor.load(wt_path) if wt_path.is_file() else WeightTensor.from_config(config)

    # Collect step 2 results
    step2_records: list[dict] = []
    if args.step2_output:
        step2_records = _load_step2_records(args.step2_output)
        if args.sample_id:
            step2_records = [r for r in step2_records if r.get("sample_id") == args.sample_id]
    elif args.run_step2 and args.sample_id:
        from step2_target_char.feature_adapter import extract_target_features, sid_to_filename
        from step2_target_char.target_char import characterize_target
        samples_dir = args.processed_dir / "samples"
        path = samples_dir / (sid_to_filename(args.sample_id) + ".json")
        with path.open("r", encoding="utf-8") as f:
            sample = json.load(f)
        s2_record = characterize_target(sample, client, step2_cfg)
        print(f"Step 2: {args.sample_id} → {s2_record.get('output', {}).get('category')}")
        step2_records = [s2_record]
    else:
        raise SystemExit("specify --step2-output or (--sample-id + --run-step2)")

    if not step2_records:
        raise SystemExit("no step 2 records to process")

    print(f"Processing {len(step2_records)} sample(s) for tool selection...")

    out_f = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_f = args.output.open("w", encoding="utf-8")

    n_ok = n_fail = 0
    total_tokens = 0
    try:
        for s2 in step2_records:
            sid = s2.get("sample_id", "?")
            tc = s2.get("output") or {}
            tf = s2.get("input_features") or {}

            record = select_tools(
                sample_id=sid,
                target_char=tc,
                target_features=tf,
                client=client,
                weight_tensor=wt,
                config=config,
            )
            print(_summarize(record))
            if out_f:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
            if record.get("success"):
                n_ok += 1
            else:
                n_fail += 1
            total_tokens += (record.get("api_usage") or {}).get("total_tokens", 0)
    finally:
        if out_f:
            out_f.close()

    print()
    print(f"Processed: {len(step2_records)}  (ok={n_ok}  fail={n_fail})")
    print(f"Total tokens: {total_tokens}")
    if args.output:
        print(f"Wrote: {args.output}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
