"""CLI entry point for Step 7 — iterative decision loop.

Reads existing step 2 / 4 / 5 / 6 JSONL outputs and runs
``run_iteration_loop`` on each sample, producing one ``IterationResult``
JSONL line per sample under ``--output``.

Typical invocation::

    python -m step7_iteration.run \\
        --processed-dir data/processed \\
        --step2-output data/step2_outputs/e2e_smoke.jsonl \\
        --step4-output data/step4_outputs/1un6_B_F.jsonl \\
        --step5-output data/step5_outputs/e2e_smoke.jsonl \\
        --step6-output data/step6_outputs/e2e_smoke.jsonl \\
        --config configs/step7_config.yaml \\
        --output data/step7_outputs/

Modes
-----
- Default uses ``LLMClient`` (requires ``LLM_API_KEY`` env var).
- ``--no-llm`` skips the LLM call entirely; iteration 0 synthesises an
  accept (useful for offline smoke / CI).

Output
------
- ``--output`` is a single ``.jsonl`` file → all records concatenated
  there (file truncated at start).
- ``--output`` is a directory → one ``<sample_id>.jsonl`` per sample.
- Writes are atomic (``.tmp`` + ``replace``) so a SIGINT mid-write
  cannot leave partial JSON.

Exit codes
----------
- 0  every requested sample produced an IterationResult
- 1  setup error (missing inputs, malformed records)
- 2  at least one sample was skipped
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import yaml

from step2_target_char.llm_client import LLMClient
from step2_target_char.run import build_client, load_config as load_step2_config
from step4_tool_adapters.schemas import ToolPrediction, ToolPredictionSet
from step5_fusion.schemas import CompositeResult
from step6_pocket_qa.schemas import PocketQAResult
# Reuse step 6's CLI helpers for the step5 / step4 record-rebuild path —
# they encode the same "writer-added fields → schema-strict reverse"
# logic and we don't want to duplicate it.
from step6_pocket_qa.run import (
    _category_for as _step6_category_for,
    _step4_record_to_predictions,
    _step5_record_to_composite,
)

from .iterator import run_iteration_loop


# ---------- IO helpers -----------------------------------------------------


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def _index_by_sid(rows: Iterable[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        sid = r.get("sample_id")
        if sid:
            out[sid] = r
    return out


def _load_sample(processed_dir: Path, sample_id: str) -> dict:
    candidates = [
        processed_dir / "samples" / f"{sample_id}.json",
        processed_dir / f"{sample_id}.json",
    ]
    for p in candidates:
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"sample {sample_id!r} not found in {[str(p) for p in candidates]}"
    )


def _step6_record_to_qa(record: dict) -> PocketQAResult:
    """Reverse of ``PocketQAResult.model_dump(mode='json')``."""
    return PocketQAResult.model_validate(record)


# ---------- output writer --------------------------------------------------


def _resolve_output_path(out_arg: Optional[Path], sample_id: str) -> Optional[Path]:
    if out_arg is None:
        return None
    if out_arg.suffix in (".jsonl", ".json"):
        out_arg.parent.mkdir(parents=True, exist_ok=True)
        return out_arg
    out_arg.mkdir(parents=True, exist_ok=True)
    return out_arg / f"{sample_id}.jsonl"


def _write_record_atomic(path: Path, record: dict, *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if append and path.is_file():
        existing = path.read_text(encoding="utf-8")
    else:
        existing = ""
    body = existing + json.dumps(record, ensure_ascii=False) + "\n"
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


def _summarize(record: dict) -> str:
    sid = record.get("sample_id", "?")
    n = record.get("total_iterations", 0)
    final = record.get("final_score", 0.0)
    reason = record.get("termination_reason", "?")
    traj = record.get("score_trajectory") or []
    if traj:
        traj_str = " -> ".join(f"{s:.3f}" for s in traj)
    else:
        traj_str = "(empty)"
    return (
        f"{sid:<16}  iters={n}  final={final:.3f}  "
        f"reason={reason:<14}  traj=[{traj_str}]"
    )


# ---------- main -----------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Step 7 — accept/refine/restart iteration loop",
    )
    ap.add_argument("--processed-dir", type=Path, required=True,
                    help="step 1 output dir containing samples/<id>.json")
    ap.add_argument("--step4-output", type=Path, required=True,
                    help="JSONL from step 4 (one ToolPredictionSet per line)")
    ap.add_argument("--step5-output", type=Path, required=True,
                    help="JSONL from step 5 (one CompositeResult per line)")
    ap.add_argument("--step6-output", type=Path, required=True,
                    help="JSONL from step 6 (one PocketQAResult per line)")
    ap.add_argument("--step2-output", type=Path, default=None,
                    help="JSONL from step 2 (optional; lets q5 read category)")
    ap.add_argument("--config", type=Path, required=True,
                    help="step7_config.yaml")
    ap.add_argument("--step2-config", type=Path, default=None,
                    help="step2_config.yaml (for LLMClient transport settings)")
    ap.add_argument("--step5-config", type=Path, default=None,
                    help="step5_config.yaml (for the inner re-fuse on refine/restart)")
    ap.add_argument("--step6-config", type=Path, default=None,
                    help="step6_config.yaml (for the inner re-score on refine/restart)")
    ap.add_argument("--sample-id", type=str, action="append", default=None,
                    help="restrict to this sample id (repeatable). If "
                         "omitted, processes every sample present in "
                         "--step6-output.")
    ap.add_argument("--output", type=Path, default=None,
                    help="output dir or single .jsonl file")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip LLM and synthesise accept on iteration 0 "
                         "(useful for offline smoke runs)")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # ----- configs --------------------------------------------------------
    config = load_config(args.config)
    cfg_dir = args.config.parent

    step2_cfg_path = args.step2_config or (cfg_dir / "step2_config.yaml")
    step2_cfg = load_step2_config(step2_cfg_path) if step2_cfg_path.is_file() else {}

    step5_cfg = (
        load_config(args.step5_config) if args.step5_config and args.step5_config.is_file()
        else (load_config(cfg_dir / "step5_config.yaml")
              if (cfg_dir / "step5_config.yaml").is_file() else {})
    )
    step6_cfg = (
        load_config(args.step6_config) if args.step6_config and args.step6_config.is_file()
        else (load_config(cfg_dir / "step6_config.yaml")
              if (cfg_dir / "step6_config.yaml").is_file() else {})
    )

    # ----- client ---------------------------------------------------------
    client: Optional[LLMClient]
    if args.no_llm:
        client = None
    else:
        try:
            client = build_client(step2_cfg)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR building LLMClient: {e}", file=sys.stderr)
            return 1

    # ----- load JSONL outputs --------------------------------------------
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
        print("ERROR: no sample ids resolved from --step6-output", file=sys.stderr)
        return 1

    # If --output is a single file, truncate once at start so multiple
    # samples concatenate cleanly.
    write_to_one_file = (
        args.output is not None and args.output.suffix in (".jsonl", ".json")
    )
    one_file_path: Optional[Path] = None
    if write_to_one_file:
        one_file_path = args.output
        one_file_path.parent.mkdir(parents=True, exist_ok=True)
        one_file_path.write_text("", encoding="utf-8")

    n_total = 0
    n_skipped = 0
    n_by_reason: dict[str, int] = {"accepted": 0, "converged": 0, "max_iterations": 0}
    total_tokens = 0
    for sid in sample_ids:
        s5_record = s5_index.get(sid)
        s4_record = s4_index.get(sid)
        s6_record = s6_index.get(sid)
        if not s5_record or not s4_record or not s6_record:
            print(
                f"[SKIP] {sid}: missing "
                f"{'step4 ' if not s4_record else ''}"
                f"{'step5 ' if not s5_record else ''}"
                f"{'step6 ' if not s6_record else ''}".rstrip(),
                file=sys.stderr,
            )
            n_skipped += 1
            continue

        try:
            sample_json = _load_sample(args.processed_dir, sid)
        except FileNotFoundError as e:
            print(f"[SKIP] {sid}: {e}", file=sys.stderr)
            n_skipped += 1
            continue

        try:
            composite = _step5_record_to_composite(s5_record)
            predictions = _step4_record_to_predictions(s4_record)
            qa_result = _step6_record_to_qa(s6_record)
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {sid}: record invalid ({e})", file=sys.stderr)
            n_skipped += 1
            continue

        # Inject category for q5 (so the inner re-score on refine/restart
        # can see it) and pull the target_char dict for the prompt.
        category = _step6_category_for(sid, s2_index, s5_record)
        scoring_input = dict(sample_json)
        s2_record = s2_index.get(sid) or {}
        target_char = s2_record.get("output") or s2_record or None
        if category:
            tc = dict(scoring_input.get("target_char") or {})
            tc.setdefault("category", category)
            scoring_input["target_char"] = tc

        result = run_iteration_loop(
            sample_json=scoring_input,
            target_char=target_char,
            tool_predictions=predictions,
            composite_result=composite,
            qa_result=qa_result,
            client=client,
            config=config,
            fusion_config=step5_cfg,
            qa_config=step6_cfg,
        )
        record = result.model_dump(mode="json")

        print(_summarize(record))

        out_path = (one_file_path if write_to_one_file
                    else _resolve_output_path(args.output, sid))
        if out_path is not None:
            _write_record_atomic(out_path, record, append=write_to_one_file)

        n_total += 1
        n_by_reason[result.termination_reason] = (
            n_by_reason.get(result.termination_reason, 0) + 1
        )
        # Sum token usage across all iteration records.
        for rec in result.iterations:
            usage = rec.api_usage or {}
            t = usage.get("total_tokens")
            if isinstance(t, int):
                total_tokens += t

    print()
    print(
        f"Processed: {n_total}  skipped: {n_skipped}  "
        f"accepted={n_by_reason['accepted']}  "
        f"converged={n_by_reason['converged']}  "
        f"max_iter={n_by_reason['max_iterations']}"
    )
    print(f"Total tokens: {total_tokens}")
    if args.output:
        print(f"Wrote: {args.output}")
    return 0 if n_skipped == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
