"""End-to-end real-API test for Step 2 — RUN ON SERVER ONLY.

This script does real LLM calls. Expected total cost per run: ~10-20k tokens
for 5 samples (a few cents at time of writing).

Usage (on the server):
    cd ~/riboseer
    conda activate riboseer
    export LLM_API_KEY="<your real key>"
    python tests/test_step2_e2e.py \
        --processed-dir data/processed \
        --config configs/step2_config.yaml

Optional flags:
    --output data/step2_outputs/e2e_smoke.jsonl   (save full records)
    --sample-ids a,b,c                             (override the curated list)

What's tested
-------------
Picks five deliberately-diverse samples covering the main edge cases seen in
step 1 (if any are missing on your server copy, pass `--sample-ids` to
substitute). For each it runs `characterize_target` end-to-end and prints:

  1. `1un6_B_F`       strict tier X-RAY zinc-finger-like RNA-binding protein
                       (87 aa, pI 8.87, C/H enriched, RNA 61 nt GC 0.66)
  2. `7z20_w_a`       strict tier cryo-EM typical basic RBP
                       (94 aa, pI 9.60, K top-3, RNA 119 nt)
  3. `7ods_K_A`       strict tier, RNA has modifications
                       (178 aa, 1589 nt RNA with modifications — likely rRNA)
  4. `7mpi_BI_A1`     very long RNA (3137 nt) + basic protein (pI 10.92)
                       — near-max RNA length, expect junction
  5. `3j92_x_5`       all-X protein edge case (pI=None, aa_composition=None)
                       — tests null handling and low-confidence fallback

Each sample's console output shows:
  - the rendered TargetFeatures (so we can eyeball what the LLM saw)
  - the LLM's analysis / category / confidence / notes
  - total tokens spent + retries needed
  - any failure reason

After the 5 samples, a summary line shows overall ok/fail counts and total
token consumption so we can sanity-check the pricing estimate.

Paste the full output back to Claude Code. What we're jointly checking:
  - Are the category assignments biologically plausible?
  - Is the LLM using the domain-knowledge hints or ignoring them?
  - Does the null-field handling work (sample #5)?
  - Are there systematic biases (e.g. always predicting "RRM"?)
  - Is the retry path hit? If yes for a sample whose output *should* be
    trivial, the prompt needs tightening.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step2_target_char.feature_adapter import (  # noqa: E402
    extract_target_features, sid_to_filename,
)
from step2_target_char.llm_client import LLMClient  # noqa: E402
from step2_target_char.prompts import format_features_for_prompt  # noqa: E402
from step2_target_char.run import build_client, load_config  # noqa: E402
from step2_target_char.target_char import characterize_target  # noqa: E402


DEFAULT_SAMPLES = [
    "1un6_B_F",    # X-RAY strict, zinc-finger candidate
    "7z20_w_a",    # cryo-EM strict, basic RBP short RNA
    "7ods_K_A",    # cryo-EM strict, modified long RNA (rRNA-like)
    "7mpi_BI_A1",  # very long RNA (3137 nt)
    "3j92_x_5",    # all-X protein edge case
]


def _banner(title: str, char: str = "=") -> None:
    print()
    print(char * 72)
    print(f"  {title}")
    print(char * 72)


def _dump_features(features) -> None:
    print(format_features_for_prompt(features))


def _dump_record(record: dict) -> None:
    out = record.get("output") or {}
    usage = record.get("api_usage") or {}
    status = "OK" if record.get("success") else "FAIL"
    print(f"\n  status:     {status}")
    print(f"  category:   {out.get('category')}")
    print(f"  confidence: {out.get('confidence')}")
    print(f"  retries:    {record.get('retries')}")
    print(f"  tokens:     prompt={usage.get('prompt_tokens')} "
          f"completion={usage.get('completion_tokens')} "
          f"total={usage.get('total_tokens')}")
    print(f"  analysis:\n{out.get('analysis')}")
    notes = out.get("notes")
    if notes:
        print(f"  notes: {notes}")
    if not record.get("success"):
        print(f"  FAILURE REASON: {record.get('failure_reason')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--processed-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    default=REPO / "configs" / "step2_config.yaml")
    ap.add_argument("--sample-ids", type=str, default=None,
                    help="comma-separated override for the default list")
    ap.add_argument("--output", type=Path, default=None,
                    help="optional JSONL output path")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not os.environ.get("LLM_API_KEY"):
        print("ERROR: LLM_API_KEY is not exported.")
        return 2

    sample_ids = (
        [s.strip() for s in args.sample_ids.split(",") if s.strip()]
        if args.sample_ids else list(DEFAULT_SAMPLES)
    )

    samples_dir = args.processed_dir / "samples"
    config = load_config(args.config)
    client = build_client(config)

    out_f = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_f = args.output.open("w", encoding="utf-8")

    n_ok = n_fail = 0
    total_tokens = 0
    try:
        for sid in sample_ids:
            path = samples_dir / (sid_to_filename(sid) + ".json")
            if not path.exists():
                _banner(f"{sid} — SKIPPED (no file at {path})")
                continue
            with path.open("r", encoding="utf-8") as f:
                sample = json.load(f)

            features = extract_target_features(sample)
            _banner(f"SAMPLE {sid}")
            _dump_features(features)

            record = characterize_target(sample, client, config)
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
