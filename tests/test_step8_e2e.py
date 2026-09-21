"""Real end-to-end smoke for Step 8 — weight tensor update (server-only).

NOT a unittest module. A manual driver that runs ``step8_weight_update.run.main``
against real upstream JSONLs (steps 2 / 4 / 5 / 6 / 7) and prints a
human-readable side-by-side of the W slice before / after the EMA pass
(and meta-correction if enabled).

Run on the server, e.g.::

    cd ~/riboseer
    conda activate riboseer
    export LLM_API_KEY=<your key>     # only needed if meta-correction is on

    python tests/test_step8_e2e.py \\
        --processed-dir data/processed \\
        --step2-output data/step2_outputs/e2e_smoke.jsonl \\
        --step4-output data/step4_outputs/1un6_B_F.jsonl \\
        --step5-output data/step5_outputs/e2e_smoke.jsonl \\
        --step6-output data/step6_outputs/e2e_smoke.jsonl \\
        --step7-output data/step7_outputs/e2e_smoke.jsonl \\
        --weight-tensor data/weights/weight_tensor.json \\
        --history data/history/prediction_history.jsonl \\
        --sample-id 1un6_B_F \\
        --config configs/step8_config.yaml \\
        --output data/step8_outputs/e2e_smoke.jsonl

What it does
------------
- Calls ``step8_weight_update.run.main`` directly (the same code path
  the production CLI uses).
- Prints a per-sample block:
    * tools_updated + ema_deltas
    * before / after W slice for the sample's category
    * meta-correction status + (if applied) γ_k factors + rationale
- Saves the updated WeightTensor + appended history JSONL.

Notes
-----
- Token budget per sample with meta-correction enabled: ~1500-3000.
  EMA-only path: 0 tokens (no LLM call).
- ``--no-llm`` disables meta-correction even if config has it enabled.
- The script is a thin wrapper around ``run.main`` so any change to
  the CLI semantics flows through automatically. Use it in CI / smoke
  pipelines to validate the wire-up before trusting batch runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import step8_weight_update.run as step8_run  # noqa: E402
from step3_tool_selection.weight_tensor import METRICS, WeightTensor  # noqa: E402
from step8_weight_update.schemas import WeightUpdateResult  # noqa: E402


def _load_yaml_keys(path: Path) -> dict:
    """Tiny YAML reader for the few fields we need (avoids hard PyYAML dep)."""
    import yaml
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _print_slice(slice_dict: dict[str, dict[str, float]], header: str) -> None:
    print(f"  {header}")
    if not slice_dict:
        print("    (empty)")
        return
    for tid, mdict in sorted(slice_dict.items()):
        cells = "  ".join(
            f"{m.split('_')[0][:4]}={v:.3f}" for m, v in mdict.items()
        )
        print(f"    {tid:<16} {cells}")


def _print_record(record: dict) -> None:
    sid = record.get("sample_id", "?")
    cat = record.get("category", "?")
    eta = record.get("learning_rate", 0.0)
    tools = record.get("tools_updated") or []
    print()
    print("=" * 80)
    print(f"sample: {sid}    category: {cat}    eta: {eta}")
    print("=" * 80)
    if not tools:
        print("  (no tools updated — every tool failed, or per-tool decomposition empty)")
        return

    print(f"  tools_updated: {tools}")
    print(f"  ema_deltas:")
    for tid, mdict in record.get("ema_deltas", {}).items():
        cells = "  ".join(f"{m.split('_')[0][:4]}={d:+.4f}"
                          for m, d in mdict.items())
        print(f"    {tid:<16} {cells}")

    print()
    _print_slice(record.get("weights_before", {}), "weights_before:")
    print()
    _print_slice(record.get("weights_after", {}), "weights_after:")

    print()
    if record.get("meta_correction_applied"):
        print("  meta-correction APPLIED:")
        for tid, g in (record.get("correction_factors") or {}).items():
            print(f"    {tid:<16} γ = {g:.3f}")
        if record.get("meta_rationale"):
            print(f"    rationale: {record['meta_rationale'][:300]}")
    else:
        status = (record.get("api_usage") or {}).get("status", "?")
        print(f"  meta-correction NOT applied (status={status!r})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    # Pass through every step 8 CLI flag — we reuse run.main directly.
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--step4-output", type=Path, required=True)
    p.add_argument("--step5-output", type=Path, required=True)
    p.add_argument("--step6-output", type=Path, required=True)
    p.add_argument("--step7-output", type=Path, required=True)
    p.add_argument("--step2-output", type=Path, default=None)
    p.add_argument("--weight-tensor", type=Path, required=True)
    p.add_argument("--history", type=Path, default=None)
    p.add_argument("--config", type=Path,
                   default=REPO / "configs" / "step8_config.yaml")
    p.add_argument("--step2-config", type=Path,
                   default=REPO / "configs" / "step2_config.yaml")
    p.add_argument("--sample-id", type=str, action="append", default=None)
    p.add_argument("--output", type=Path,
                   default=REPO / "data" / "step8_outputs" / "e2e_smoke.jsonl")
    p.add_argument("--no-llm", action="store_true")
    args = p.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Pre-flight: warn if meta-correction will require an API key.
    cfg = _load_yaml_keys(args.config)
    enable_meta = bool((cfg.get("weight_update") or {})
                       .get("enable_meta_correction", False))
    if enable_meta and not args.no_llm and not os.environ.get("LLM_API_KEY"):
        print("ERROR: enable_meta_correction=True but LLM_API_KEY is not set. "
              "Pass --no-llm for an offline run.", file=sys.stderr)
        return 1

    # Build CLI argv for run.main (forwarding everything verbatim).
    argv: list[str] = [
        "--processed-dir", str(args.processed_dir),
        "--step4-output", str(args.step4_output),
        "--step5-output", str(args.step5_output),
        "--step6-output", str(args.step6_output),
        "--step7-output", str(args.step7_output),
        "--weight-tensor", str(args.weight_tensor),
        "--config", str(args.config),
    ]
    if args.step2_output:
        argv += ["--step2-output", str(args.step2_output)]
    if args.step2_config:
        argv += ["--step2-config", str(args.step2_config)]
    if args.history:
        argv += ["--history", str(args.history)]
    if args.sample_id:
        for sid in args.sample_id:
            argv += ["--sample-id", sid]
    argv += ["--output", str(args.output)]
    if args.no_llm:
        argv += ["--no-llm"]

    rc = step8_run.main(argv)
    print(f"\n[run.main returned {rc}]")

    # Read back the JSONL we just wrote and pretty-print every record.
    if args.output.is_file():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            try:
                # Round-trip through the schema as a final sanity check.
                WeightUpdateResult.model_validate(rec)
            except Exception as e:  # noqa: BLE001
                print(f"WARN: emitted record failed schema validation: {e}",
                      file=sys.stderr)
            _print_record(rec)

    # Show the post-update tensor's UCB ranking (so a human can spot
    # whether the update meaningfully shifted tool ranking).
    if args.weight_tensor.is_file():
        print()
        print("=" * 80)
        print("Post-update tensor — UCB scores per category")
        print("=" * 80)
        wt = WeightTensor.load(args.weight_tensor)
        # Pull category from the first emitted record (single-sample
        # smoke is the common case; multi-sample callers can ignore
        # this section if they don't care).
        cats: list[str] = []
        if args.output.is_file():
            for line in args.output.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("category") not in cats:
                        cats.append(rec.get("category"))
        for cat in cats:
            print(f"\n  category: {cat}")
            scores = wt.compute_utility_scores(cat)
            for tid, s in scores.items():
                print(f"    {tid:<16}  UCB={s:.4f}  evals={wt.get_count(tid, cat)}")

    return rc


if __name__ == "__main__":
    sys.exit(main())
