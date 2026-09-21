"""Real end-to-end smoke for Step 5 — fusion (server-only).

NOT a unittest module. A manual script that drives ``fuse_predictions``
on one or two real samples using existing step 4 output. Calls the LLM
API for the weight assignment.

Run on the server, e.g.::

    cd ~/riboseer
    conda activate riboseer
    export LLM_API_KEY="your_key"
    python tests/test_step5_e2e.py \\
        --processed-dir data/processed \\
        --step2-output data/step2_outputs/e2e_smoke.jsonl \\
        --step4-output data/step4_outputs/1un6_B_F.jsonl \\
        --sample-id 1un6_B_F \\
        --output data/step5_outputs/e2e_smoke.jsonl

Behaviour
---------
- For each requested sample id, loads the step 1 sample JSON
  (``--processed-dir/samples/<id>.json``), the matching step 2 record
  (target_char), and the matching step 4 record (ToolPredictionSet).
- Calls ``fuse_predictions`` with the production LLM client.
- Prints a side-by-side block: per-tool prediction summary → LLM weight
  assignment → final fused residue set → ground-truth comparison
  (precision / recall / F1).
- Writes the final JSONL record to ``--output`` (file or directory).

Notes
-----
- This script does not refresh step 4 output. If step 4 hasn't produced
  predictions for a sample, the run fails for that sample. Use
  ``src/step4_tool_adapters/run.py`` first.
- Token budget: ~1.5k prompt tokens per sample (3-tool case).
- The script is intentionally short — actual fusion logic lives in
  ``src/step5_fusion/fusion.py`` and is unit-tested under
  ``tests/test_step5_mock.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step2_target_char.llm_client import LLMClient  # noqa: E402
from step2_target_char.run import build_client, load_config as load_step2_config  # noqa: E402
from step3_tool_selection.weight_tensor import WeightTensor  # noqa: E402
from step4_tool_adapters.schemas import ToolPredictionSet  # noqa: E402
from step5_fusion.fusion import (  # noqa: E402
    build_output_record, fuse_predictions,
)
from step5_fusion.prompts import summarize_predictions  # noqa: E402


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


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _index_by_sid(rows: list[dict]) -> dict[str, dict]:
    return {r["sample_id"]: r for r in rows if r.get("sample_id")}


def _load_yaml(path: Path) -> dict:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--step2-output", type=Path, required=True,
                   help="JSONL from step 2 (target_char per sample)")
    p.add_argument("--step4-output", type=Path, required=True,
                   help="JSONL from step 4 (ToolPredictionSet per sample)")
    p.add_argument("--config", type=Path,
                   default=REPO / "configs" / "step5_config.yaml")
    p.add_argument("--step2-config", type=Path,
                   default=REPO / "configs" / "step2_config.yaml")
    p.add_argument("--step3-config", type=Path,
                   default=REPO / "configs" / "step3_config.yaml")
    p.add_argument("--sample-id", type=str, action="append",
                   default=None,
                   help="restrict to this sample id (repeatable). If "
                        "omitted, processes every sample present in "
                        "--step4-output.")
    p.add_argument("--output", type=Path,
                   default=REPO / "data" / "step5_outputs" / "e2e_smoke.jsonl")
    p.add_argument("--no-llm", action="store_true",
                   help="skip the LLM call (equal-weights fallback) — "
                        "useful for verifying the rest of the pipeline "
                        "without burning tokens")
    args = p.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # ----- load configs ---------------------------------------------------
    step5_cfg = _load_yaml(args.config)
    step2_cfg = load_step2_config(args.step2_config)
    step3_cfg = _load_yaml(args.step3_config) if args.step3_config.is_file() else {}

    # ----- build client ---------------------------------------------------
    client = None if args.no_llm else build_client(step2_cfg)

    # ----- weight tensor (for prompt embedding) ---------------------------
    wt_path = Path(step3_cfg.get(
        "weight_tensor_path",
        "data/step3_weights/weight_tensor.json",
    ))
    weight_tensor = (
        WeightTensor.load(wt_path) if wt_path.is_file()
        else WeightTensor.from_config(step3_cfg) if step3_cfg
        else None
    )

    # ----- load step 2 / step 4 records ----------------------------------
    s2_index = _index_by_sid(_load_jsonl(args.step2_output))
    s4_index = _index_by_sid(_load_jsonl(args.step4_output))

    sample_ids = args.sample_id or sorted(s4_index.keys())
    if not sample_ids:
        print("ERROR: no sample ids resolved from --step4-output")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.output.open("w", encoding="utf-8")

    try:
        for sid in sample_ids:
            print()
            print("=" * 80)
            print(f"sample: {sid}")
            print("=" * 80)

            sample_json = _load_sample(args.processed_dir, sid)

            # step 2 target_char (fall back to whole record if no "output" key)
            s2_rec = s2_index.get(sid) or {}
            target_char = s2_rec.get("output") or s2_rec

            # step 4 predictions
            s4_rec = s4_index.get(sid)
            if not s4_rec:
                print(f"  SKIP — no step 4 record for {sid}")
                continue
            try:
                pset = ToolPredictionSet.model_validate(s4_rec)
            except Exception as e:
                print(f"  SKIP — step 4 record malformed: {e}")
                continue
            predictions = list(pset.predictions)

            # Show what's going into the LLM.
            print()
            print("--- step 2 category ---")
            print(f"  category   : {target_char.get('category')}")
            print(f"  confidence : {target_char.get('confidence')}")
            print()
            print("--- step 4 predictions (input) ---")
            print(summarize_predictions(predictions))

            # Fuse.
            composite = fuse_predictions(
                sample_json=sample_json,
                target_char=target_char,
                tool_predictions=predictions,
                client=client,
                weight_tensor=weight_tensor,
                config=step5_cfg,
            )
            record = build_output_record(composite, sample_json, target_char)

            # Pretty-print key bits.
            print()
            print("--- fusion result ---")
            print(f"  status       : {(record['api_usage'] or {}).get('status', '?')}")
            print(f"  tools_fused  : {record['tools_fused']}")
            print(f"  tool_weights : {record['tool_weights']}")
            print(f"  threshold    : {record['threshold']}")
            n_bind = len(record['binding_protein_residues'])
            n_rna = len(record['binding_rna_nucleotides'])
            print(f"  binding(prot): {n_bind} residues "
                  f"{record['binding_protein_residues'][:20]}"
                  f"{' ...' if n_bind > 20 else ''}")
            print(f"  binding(rna) : {n_rna} nucleotides "
                  f"{record['binding_rna_nucleotides'][:20]}"
                  f"{' ...' if n_rna > 20 else ''}")
            print(f"  confidence   : {record['confidence']}")
            print(f"  rationale    : {record['fusion_rationale'][:300]}"
                  f"{' ...' if len(record['fusion_rationale']) > 300 else ''}")

            # vs ground truth
            print()
            print("--- vs ground truth ---")
            gt = record['ground_truth_protein']
            print(f"  GT residues  : {len(gt)} {gt[:20]}"
                  f"{' ...' if len(gt) > 20 else ''}")
            print(f"  precision    : {record['precision']}")
            print(f"  recall       : {record['recall']}")
            print(f"  f1           : {record['f1']}")
            usage = record['api_usage'] or {}
            print(f"  tokens (cum) : prompt={usage.get('prompt_tokens', 0)}  "
                  f"completion={usage.get('completion_tokens', 0)}  "
                  f"total={usage.get('total_tokens', 0)}")

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
    finally:
        out_f.close()

    print()
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
