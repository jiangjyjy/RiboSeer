"""Real end-to-end smoke for Step 6 — pocket QA scoring (server-only).

NOT a unittest module. A manual script that drives ``score_prediction``
on one or more real samples using existing step 4 / step 5 outputs.
No LLM call (step 6 is pure code).

Run on the server, e.g.::

    cd ~/riboseer
    conda activate riboseer
    python tests/test_step6_e2e.py \\
        --processed-dir data/processed \\
        --step4-output data/step4_outputs/1un6_B_F.jsonl \\
        --step5-output data/step5_outputs/e2e_smoke.jsonl \\
        --step2-output data/step2_outputs/e2e_smoke.jsonl \\
        --sample-id 1un6_B_F \\
        --config configs/step6_config.yaml \\
        --output data/step6_outputs/e2e_smoke.jsonl

What it does
------------
- For each requested sample id, loads the step 1 sample JSON
  (``--processed-dir/samples/<id>.json``), the matching step 2 record
  (target_char), the matching step 4 record (ToolPredictionSet), and the
  matching step 5 record (CompositeResult).
- Calls ``score_prediction`` and prints a side-by-side block:
    * inputs (composite binding, surviving tools, category)
    * each q_m's score + a one-line ``info`` summary (computed vs abstain)
    * weighted total + which weights were renormalised in
- Writes the final JSONL record to ``--output``.

This is the manually-driven counterpart to the unit-tested
``tests/test_step6_mock.py``. Heavy I/O lives here so the unit suite
stays fast and dependency-free.

Notes
-----
- Step 6 needs gemmi only when a Cat A predicted structure is present;
  EquiPNAS/P2Rank-only samples score q1 with the 2-axis (continuity +
  ratio) path and skip compactness silently.
- Token budget: 0 (no LLM).
- The script does not refresh upstream outputs. Make sure step 4 and
  step 5 already ran for the requested sample(s).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step4_tool_adapters.schemas import ToolPredictionSet  # noqa: E402

from step6_pocket_qa.run import (  # noqa: E402
    _category_for,
    _index_by_sid,
    _load_jsonl,
    _load_sample,
    _step4_record_to_predictions,
    _step5_record_to_composite,
)
from step6_pocket_qa.scorer import score_prediction  # noqa: E402


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _fmt_score(v: Optional[float]) -> str:
    return "  -  " if v is None else f"{v:.3f}"


def _info_summary(detail_dict: dict, name: str) -> str:
    """One-line dump of the most diagnostic ``info`` keys per metric."""
    info = detail_dict.get("info") or {}
    if not detail_dict.get("computed"):
        why = detail_dict.get("error") or info.get("reason") or "abstain"
        return f"  reason: {why}"
    if name == "structural_plausibility":
        return (
            f"  clusters={info.get('n_clusters')}  "
            f"ratio={info.get('interface_ratio')}  "
            f"rg={info.get('rg')}  expected_rg={info.get('expected_rg')}"
        )
    if name == "physicochemical_complementarity":
        return (
            f"  +charge={info.get('positive_ratio')}  "
            f"aromatic={info.get('aromatic_ratio')}  "
            f"polar={info.get('polar_ratio')}  "
            f"GP={info.get('gp_ratio')}"
        )
    if name == "evolutionary_conservation":
        return (
            f"  rare={info.get('rare_ratio')}  "
            f"terminal={info.get('terminal_share')}  "
            f"pI={info.get('pI')}  pi_axis={info.get('pi_reason', '?')}"
        )
    if name == "cross_tool_consensus":
        return (
            f"  active={info.get('n_active_tools')}  "
            f"vote={info.get('avg_vote_ratio')}  "
            f"jaccard={info.get('avg_jaccard')}"
        )
    if name == "known_motif_consistency":
        hits = info.get("motif_hits") or []
        return (
            f"  domain={info.get('domain')}  "
            f"hits={len(hits)}  "
            f"coverage={info.get('coverage_ratio')}"
        )
    return ""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--step4-output", type=Path, required=True,
                   help="JSONL from step 4 (ToolPredictionSet per sample)")
    p.add_argument("--step5-output", type=Path, required=True,
                   help="JSONL from step 5 (CompositeResult per sample)")
    p.add_argument("--step2-output", type=Path, default=None,
                   help="JSONL from step 2 (target_char per sample); when "
                        "supplied, q5 reads category from this. Otherwise "
                        "falls back to step5's step2_category field.")
    p.add_argument("--config", type=Path,
                   default=REPO / "configs" / "step6_config.yaml")
    p.add_argument("--sample-id", type=str, action="append", default=None,
                   help="restrict to this sample id (repeatable). If "
                        "omitted, processes every sample present in "
                        "--step5-output.")
    p.add_argument("--output", type=Path,
                   default=REPO / "data" / "step6_outputs" / "e2e_smoke.jsonl")
    args = p.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = _load_yaml(args.config)

    s4_index = _index_by_sid(_load_jsonl(args.step4_output))
    s5_index = _index_by_sid(_load_jsonl(args.step5_output))
    s2_index = (
        _index_by_sid(_load_jsonl(args.step2_output))
        if args.step2_output is not None and args.step2_output.is_file()
        else {}
    )

    sample_ids = args.sample_id or sorted(s5_index.keys())
    if not sample_ids:
        print("ERROR: no sample ids resolved from --step5-output")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_f = args.output.open("w", encoding="utf-8")

    try:
        for sid in sample_ids:
            print()
            print("=" * 80)
            print(f"sample: {sid}")
            print("=" * 80)

            s5_record = s5_index.get(sid)
            if not s5_record:
                print(f"  SKIP — no step 5 record")
                continue
            s4_record = s4_index.get(sid)
            if not s4_record:
                print(f"  SKIP — no step 4 record")
                continue

            try:
                sample_json = _load_sample(args.processed_dir, sid)
            except FileNotFoundError as e:
                print(f"  SKIP — {e}")
                continue
            try:
                composite = _step5_record_to_composite(s5_record)
            except Exception as e:  # noqa: BLE001
                print(f"  SKIP — step5 record invalid: {e}")
                continue
            try:
                predictions = _step4_record_to_predictions(s4_record)
            except Exception as e:  # noqa: BLE001
                print(f"  SKIP — step4 record invalid: {e}")
                continue

            # Inject category for q5.
            category = _category_for(sid, s2_index, s5_record)
            scoring_input = dict(sample_json)
            if category:
                tc = dict(scoring_input.get("target_char") or {})
                tc.setdefault("category", category)
                scoring_input["target_char"] = tc

            # ----- print inputs -------------------------------------------
            print()
            print("--- inputs ---")
            print(f"  category     : {category or '(none)'}")
            print(f"  protein len  : "
                  f"{(scoring_input.get('protein') or {}).get('length')}")
            print(f"  binding(prot): {len(composite.binding_protein_residues)} "
                  f"residues "
                  f"{composite.binding_protein_residues[:20]}"
                  f"{' ...' if len(composite.binding_protein_residues) > 20 else ''}")
            print(f"  binding(rna) : {len(composite.binding_rna_nucleotides)} "
                  f"nucleotides "
                  f"{composite.binding_rna_nucleotides[:20]}"
                  f"{' ...' if len(composite.binding_rna_nucleotides) > 20 else ''}")
            tool_summary = ", ".join(
                f"{p.tool_id}({'ok' if p.success else 'fail'})"
                for p in predictions
            )
            print(f"  tools        : {tool_summary}")

            # ----- run scorer ---------------------------------------------
            result = score_prediction(
                sample_json=scoring_input,
                composite_result=composite,
                tool_predictions=predictions,
                config=config,
            )
            record = result.model_dump(mode="json")

            # ----- print sub-scores ---------------------------------------
            print()
            print("--- sub-scores ---")
            for name in (
                "structural_plausibility",
                "physicochemical_complementarity",
                "evolutionary_conservation",
                "cross_tool_consensus",
                "known_motif_consistency",
            ):
                detail = record["details"][name]
                tag = "✓" if detail["computed"] else "·"
                print(f"  {tag} {name:<33} "
                      f"= {_fmt_score(detail.get('score'))}")
                print(_info_summary(detail, name))

            # ----- aggregate ----------------------------------------------
            print()
            print("--- aggregate ---")
            print(f"  total_score       : {result.total_score:.4f}")
            print(f"  n_metrics_computed: {result.n_metrics_computed} / 5")
            if result.weights_used:
                w_str = "  ".join(
                    f"{k}={v:.2f}" for k, v in sorted(result.weights_used.items())
                )
                print(f"  weights_used      : {w_str}")
            else:
                print(f"  weights_used      : (none — all metrics abstained)")

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
    finally:
        out_f.close()

    print()
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
