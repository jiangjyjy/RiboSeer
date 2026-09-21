"""CLI entry point for Step 6 — pocket QA scoring.

Reads existing step 4 / step 5 JSONL outputs and produces a per-sample
``PocketQAResult`` line under ``--output``. No LLM call.

Typical invocation::

    python -m step6_pocket_qa.run \\
        --processed-dir data/processed \\
        --step4-output data/step4_outputs/1un6_B_F.jsonl \\
        --step5-output data/step5_outputs/e2e_smoke.jsonl \\
        --config configs/step6_config.yaml \\
        --output data/step6_outputs/

Optional ``--step2-output`` is the step 2 JSONL; when supplied, the
sample's ``target_char.category`` is injected into the sample dict so q5
(known_motif_consistency) can pick its rule. Without it, q5 falls back
to neutral.

Output layout
-------------
- ``--output`` is a single ``.jsonl`` file → all records concatenated
  there (file is truncated at start).
- ``--output`` is a directory (or has no .jsonl suffix) → one
  ``<sample_id>.jsonl`` per sample inside that directory.
- Writes are atomic (``.tmp`` + ``replace``) so a SIGINT mid-write
  cannot leave partial JSON.

Exit codes
----------
0  every requested sample scored (n_metrics_computed > 0 for each)
2  at least one sample produced n_metrics_computed=0
1  setup error (missing inputs, malformed records)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import yaml

from step4_tool_adapters.schemas import ToolPrediction, ToolPredictionSet
from step5_fusion.schemas import CompositeResult

from .scorer import score_prediction


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


def _index_by_sid(rows: Iterable[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        sid = r.get("sample_id")
        if sid:
            out[sid] = r
    return out


# ---------- step 5 record → CompositeResult --------------------------------

# Fields the step 5 writer adds on top of CompositeResult; we strip them
# before model_validate to keep ``extra="forbid"`` happy.
_COMPOSITE_KEYS: frozenset[str] = frozenset({
    "sample_id", "tool_weights", "binding_protein_residues",
    "binding_rna_nucleotides", "per_residue_probability",
    "per_nucleotide_probability", "threshold", "fusion_rationale",
    "confidence", "tools_fused", "api_usage", "timestamp",
})


def _step5_record_to_composite(record: dict) -> CompositeResult:
    """Reconstruct ``CompositeResult`` from a flattened step 5 JSONL row.

    The writer adds bookkeeping fields (``step2_category``, GT metrics,
    etc.) and stringifies the per-residue probability keys; we reverse
    both before handing off to the strict schema validator.
    """
    payload = {k: v for k, v in record.items() if k in _COMPOSITE_KEYS}

    prp = payload.get("per_residue_probability") or {}
    payload["per_residue_probability"] = {int(k): float(v) for k, v in prp.items()}

    pnp = payload.get("per_nucleotide_probability")
    if pnp is not None:
        payload["per_nucleotide_probability"] = {
            int(k): float(v) for k, v in pnp.items()
        }

    return CompositeResult.model_validate(payload)


def _step4_record_to_predictions(record: dict) -> list[ToolPrediction]:
    pset = ToolPredictionSet.model_validate(record)
    return list(pset.predictions)


def _category_for(
    sample_id: str, step2_index: dict[str, dict], step5_record: dict,
) -> str:
    """Resolve target category for q5.

    Prefers the step 2 record's ``output.category`` (or top-level
    ``category`` for older mock format); falls back to the
    ``step2_category`` field that step 5 writer copied in.
    """
    s2 = step2_index.get(sample_id) or {}
    target_char = s2.get("output") or s2
    cat = (target_char or {}).get("category")
    if cat:
        return str(cat)
    s5_cat = step5_record.get("step2_category")
    return str(s5_cat) if s5_cat else ""


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
    def _f(v: Optional[float]) -> str:
        return "  -  " if v is None else f"{v:.3f}"
    return (
        f"{record['sample_id']:<16}  "
        f"q1={_f(record.get('structural_plausibility'))}  "
        f"q2={_f(record.get('physicochemical_complementarity'))}  "
        f"q3={_f(record.get('evolutionary_conservation'))}  "
        f"q4={_f(record.get('cross_tool_consensus'))}  "
        f"q5={_f(record.get('known_motif_consistency'))}  "
        f"=> total={record.get('total_score', 0.0):.3f}  "
        f"({record.get('n_metrics_computed', 0)}/5)"
    )


# ---------- main -----------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Step 6 — pocket QA scoring (5 sub-scores + weighted total)",
    )
    ap.add_argument("--processed-dir", type=Path, required=True,
                    help="step 1 output dir containing samples/<id>.json")
    ap.add_argument("--step4-output", type=Path, required=True,
                    help="JSONL from step 4 (one ToolPredictionSet per line)")
    ap.add_argument("--step5-output", type=Path, required=True,
                    help="JSONL from step 5 (one CompositeResult per line)")
    ap.add_argument("--step2-output", type=Path, default=None,
                    help="JSONL from step 2; lets q5 read target_char.category")
    ap.add_argument("--config", type=Path, required=True,
                    help="step6_config.yaml")
    ap.add_argument("--sample-id", type=str, action="append", default=None,
                    help="restrict to this sample id (repeatable). If "
                         "omitted, processes every sample present in "
                         "--step5-output.")
    ap.add_argument("--output", type=Path, default=None,
                    help="output dir or single .jsonl file")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_config(args.config)

    s4_index = _index_by_sid(_load_jsonl(args.step4_output))
    s5_index = _index_by_sid(_load_jsonl(args.step5_output))
    s2_index = (
        _index_by_sid(_load_jsonl(args.step2_output))
        if args.step2_output is not None and args.step2_output.is_file()
        else {}
    )

    sample_ids = args.sample_id or sorted(s5_index.keys())
    if not sample_ids:
        print("ERROR: no sample ids resolved from --step5-output", file=sys.stderr)
        return 1

    # If --output is a single file, truncate it once at start so multiple
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
    n_zero = 0
    for sid in sample_ids:
        s5_record = s5_index.get(sid)
        if not s5_record:
            print(f"[SKIP] {sid}: no step 5 record", file=sys.stderr)
            continue
        s4_record = s4_index.get(sid)
        if not s4_record:
            print(f"[SKIP] {sid}: no step 4 record", file=sys.stderr)
            continue
        try:
            sample_json = _load_sample(args.processed_dir, sid)
        except FileNotFoundError as e:
            print(f"[SKIP] {sid}: {e}", file=sys.stderr)
            continue

        try:
            composite = _step5_record_to_composite(s5_record)
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {sid}: step5 record invalid ({e})", file=sys.stderr)
            continue
        try:
            predictions = _step4_record_to_predictions(s4_record)
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {sid}: step4 record invalid ({e})", file=sys.stderr)
            continue

        # Inject category from step 2 (if available) so q5 can use it
        # without changing the sample JSON on disk.
        category = _category_for(sid, s2_index, s5_record)
        scoring_input = dict(sample_json)
        if category:
            tc = dict(scoring_input.get("target_char") or {})
            tc.setdefault("category", category)
            scoring_input["target_char"] = tc

        result = score_prediction(
            sample_json=scoring_input,
            composite_result=composite,
            tool_predictions=predictions,
            config=config,
        )
        record = result.model_dump(mode="json")

        print(_summarize(record))

        out_path = (one_file_path if write_to_one_file
                    else _resolve_output_path(args.output, sid))
        if out_path is not None:
            _write_record_atomic(out_path, record, append=write_to_one_file)

        n_total += 1
        if result.n_metrics_computed == 0:
            n_zero += 1

    print()
    print(f"Processed: {n_total}  zero_metric_samples: {n_zero}")
    if args.output:
        print(f"Wrote: {args.output}")
    return 0 if n_zero == 0 and n_total > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
