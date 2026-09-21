"""CLI entry point for Step 2.

Usage
-----
    # single sample
    python -m src.step2_target_char.run \
        --processed-dir data/processed \
        --sample-id 1un6_B_F \
        --config configs/step2_config.yaml

    # random N samples, write JSONL
    python -m src.step2_target_char.run \
        --processed-dir data/processed \
        --n 10 \
        --output data/step2_outputs/batch_test.jsonl \
        --config configs/step2_config.yaml

    # explicit sample list
    python -m src.step2_target_char.run \
        --processed-dir data/processed \
        --sample-ids 1un6_B_F,7z20_w_a,3j92_x_5 \
        --output data/step2_outputs/curated.jsonl \
        --config configs/step2_config.yaml

Requires `LLM_API_KEY` exported in the shell (the API client reads it
directly from os.environ). Run this on the server, not the laptop.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import yaml

from .feature_adapter import sid_to_filename
from .llm_client import LLMClient
from .target_char import characterize_target


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_client(config: dict) -> LLMClient:
    api = config.get("api") or {}
    kwargs = {}
    if "model" in api:
        kwargs["model"] = api["model"]
    if "url" in api:
        kwargs["url"] = api["url"]
    if "max_retries" in api:
        kwargs["max_retries"] = int(api["max_retries"])
    if "retry_base_delay_seconds" in api:
        kwargs["retry_base_delay"] = float(api["retry_base_delay_seconds"])
    if "timeout_seconds" in api:
        kwargs["timeout_seconds"] = float(api["timeout_seconds"])
    if "usage_log" in api:
        kwargs["usage_log"] = Path(api["usage_log"])
    return LLMClient(**kwargs)


def pick_rows(
    index_csv: Path,
    *,
    sample_id: str | None,
    sample_ids: str | None,
    n: int,
    seed: int,
) -> list[dict]:
    with index_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if sample_id:
        matched = [r for r in rows if r["sample_id"] == sample_id]
        if not matched:
            raise SystemExit(f"sample_id {sample_id!r} not in {index_csv}")
        return matched
    if sample_ids:
        wanted = {s.strip() for s in sample_ids.split(",") if s.strip()}
        matched = [r for r in rows if r["sample_id"] in wanted]
        missing = wanted - {r["sample_id"] for r in matched}
        if missing:
            raise SystemExit(f"sample_ids not in index: {sorted(missing)}")
        return matched
    if n > 0:
        random.Random(seed).shuffle(rows)
        return rows[:n]
    raise SystemExit("specify one of: --sample-id / --sample-ids / --n")


def _summarize(record: dict) -> str:
    out = record.get("output") or {}
    usage = record.get("api_usage") or {}
    status = "OK " if record.get("success") else "FAIL"
    conf = out.get("confidence", 0.0)
    return (
        f"[{status}] {record.get('sample_id'):<16} "
        f"category={out.get('category', '?'):<28} "
        f"conf={conf:.2f}  "
        f"retries={record.get('retries', 0)}  "
        f"tokens={usage.get('total_tokens', 0)}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage 2 — run LLM target characterization over samples",
    )
    ap.add_argument("--processed-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--sample-id", type=str, default=None,
                    help="single sample_id from index.csv")
    ap.add_argument("--sample-ids", type=str, default=None,
                    help="comma-separated sample_ids")
    ap.add_argument("--n", type=int, default=0,
                    help="random-pick N samples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=None,
                    help="JSONL output path; stdout-summary printed either way")
    args = ap.parse_args()

    # Windows GBK console can't print Å / ≲ embedded in the feature dump
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    client = build_client(config)

    samples_dir = args.processed_dir / "samples"
    index_csv = args.processed_dir / "index.csv"
    rows = pick_rows(
        index_csv,
        sample_id=args.sample_id,
        sample_ids=args.sample_ids,
        n=args.n, seed=args.seed,
    )
    print(f"Processing {len(rows)} sample(s)...")

    out_f = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_f = args.output.open("w", encoding="utf-8")

    n_ok = n_fail = 0
    total_tokens = 0
    try:
        for r in rows:
            sid = r["sample_id"]
            path = samples_dir / (sid_to_filename(sid) + ".json")
            with path.open("r", encoding="utf-8") as f:
                sample = json.load(f)
            record = characterize_target(sample, client, config)
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
    print(f"Processed: {len(rows)}  (ok={n_ok}  fail={n_fail})")
    print(f"Total tokens: {total_tokens}")
    if args.output:
        print(f"Wrote: {args.output}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
