#!/usr/bin/env python3
"""Iterative POLISH action generator.

The shipped POLISH (``generate_polish_actions.py``) makes a *single*
post-processing edit per sample; this variant runs the **iterative** loop:

    round t:
      1. call the LLM with the CURRENT per-residue probability + tool scores
      2. LLM returns an action (mask / extend / relocate / accept)
      3. accept            → stop
      4. otherwise apply the action → updated probability
      5. repeat until accept or t == max_rounds

This script runs that loop once per sample at ``--max-rounds`` (default 5)
and records **every** round, so a single run covers every budget: replaying
``rounds[:Tmax]`` reproduces a run capped at Tmax rounds.
That reconstruction is exact because a round's prompt depends only on the
current probability vector — it is unaware of the remaining budget — so the
first ``Tmax`` recorded rounds are exactly what a real Tmax-budget run would
produce.

Output ``<out-dir>/<sample_id>.json``::

    {"sample_id": ...,
     "rounds": [
       {"round": 1, "action": "mask", "residues": [...],
        "target_residues": [], "confidence": 0.7, "reasoning": "...",
        "source": "llm"},
       {"round": 2, "action": "accept", ...}],
     "total_rounds": 2, "max_rounds": 5}

Modes mirror ``generate_polish_actions.py``: ``--mode auto`` is the
deterministic, no-API control (``auto_polish_action`` re-evaluated on the
updated vector each round — it converges to ``accept`` as the binding set
shrinks); ``--mode llm`` calls LLM each round, falling back to the auto
action on any failure (per-round ``source=llm_fallback``).

Base (round-0) probability: ``EnrichedFusion.predict_sample`` when
``--enriched-model-dir`` is given, else the per-residue mean of tool scores.

Usage
-----
::

    python scripts/riboseer/generate_polish_iterative.py \\
        --processed-dir data/processed_quality \\
        --sample-list   data/processed_quality/splits_tmscore_035/test.txt \\
        --step4-dir     data/batch_test_v7/step4/ \\
        --enriched-model-dir data/enriched_v7_lgbm/ \\
        --step6-dir     data/batch_test_v7/step6/ \\
        --out-dir       data/batch_test_v7/polish_actions_iterative/ \\
        --mode llm --config configs/step7_config.yaml --max-rounds 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step5_fusion.data_collector import (  # noqa: E402
    load_sample_json, read_jsonl_record,
)
from step4_tool_adapters.tool_io import (  # noqa: E402
    clean_sample_id, read_sample_list,
)
from step7_iteration.polish_ops import (  # noqa: E402
    apply_polish_to_probability, auto_polish_action,
)
from scripts.riboseer.generate_polish_actions import (  # noqa: E402
    _build_llm_client, _load_enriched, _verdict_score, fused_probability,
    llm_action,
)
from scripts.riboseer.prompt_style import add_prompt_style_args  # noqa: E402
from scripts.riboseer.llm_backbone import (  # noqa: E402
    add_backbone_args, backbone_from_args,
)


def _round_action(plen: int, prob: dict[int, float], step4_data: dict,
                  verdict: Optional[float], mode: str, client,
                  config: dict, temperature: Optional[float] = None,
                  cot: bool = True) -> tuple[dict, str]:
    """One round's action + its source tag. LLM failure → auto fallback."""
    if mode == "llm":
        res = llm_action(plen, prob, step4_data, verdict, client, config,
                         temperature, cot)
        if res is not None:
            return res, "llm"
        act = auto_polish_action(prob)
        act.setdefault("target_residues", [])
        return act, "llm_fallback"
    act = auto_polish_action(prob)
    act.setdefault("target_residues", [])
    return act, "auto"


def iterate_sample(plen: int, base_prob: dict[int, float], step4_data: dict,
                   verdict: Optional[float], mode: str, client, config: dict,
                   max_rounds: int, temperature: Optional[float] = None,
                   cot: bool = True) -> list[dict]:
    """Run the POLISH loop, returning the per-round records (terminating at
    the first ``accept`` or after ``max_rounds`` rounds)."""
    prob = dict(base_prob)
    rounds: list[dict] = []
    for t in range(1, max(1, max_rounds) + 1):
        action, source = _round_action(plen, prob, step4_data, verdict,
                                        mode, client, config, temperature,
                                        cot)
        act_name = str(action.get("action", "accept")).lower()
        rounds.append({
            "round": t,
            "action": act_name,
            "residues": list(action.get("residues") or []),
            "target_residues": list(action.get("target_residues") or []),
            "confidence": action.get("confidence"),
            "reasoning": str(action.get("reasoning", ""))[:1000],
            "source": source,
        })
        if act_name == "accept":
            break
        prob = apply_polish_to_probability(prob, action)
    return rounds


def generate(*, processed_dir: Path, sample_ids: list[str], step4_dir: Path,
             out_dir: Path, mode: str, enriched_model,
             step6_dir: Optional[Path], config_path: Optional[Path],
             max_rounds: int, temperature: Optional[float] = None,
             cot: bool = True,
             backbone: Optional[dict] = None) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = None
    config: dict = {}
    if mode == "llm":
        client, config = _build_llm_client(config_path, backbone)

    stats = {"written": 0, "skipped": 0, "total_rounds": 0, "accepted": 0}
    for sid in sample_ids:
        s4 = read_jsonl_record(step4_dir / f"{sid}.jsonl")
        samples_dir = processed_dir / "samples"
        sample = load_sample_json(
            samples_dir if samples_dir.is_dir() else processed_dir, sid)
        if s4 is None or sample is None:
            stats["skipped"] += 1
            continue
        base = fused_probability(s4, sample, enriched_model)
        if not base:
            stats["skipped"] += 1
            continue
        prot = sample.get("protein") or {}
        plen = int(prot.get("length") or len(prot.get("sequence") or "") or 0)
        verdict = _verdict_score(step6_dir, sid) if mode == "llm" else None

        rounds = iterate_sample(plen, base, s4, verdict, mode, client, config,
                                max_rounds, temperature, cot)
        stats["total_rounds"] += len(rounds)
        if rounds and rounds[-1]["action"] == "accept":
            stats["accepted"] += 1

        (out_dir / f"{sid}.json").write_text(json.dumps({
            "sample_id": sid,
            "rounds": rounds,
            "total_rounds": len(rounds),
            "max_rounds": max_rounds,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        stats["written"] += 1
    return stats


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--sample-list", type=Path, required=True)
    p.add_argument("--step4-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--mode", choices=("auto", "llm"), default="auto")
    p.add_argument("--enriched-model-dir", type=Path, default=None)
    p.add_argument("--step6-dir", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None,
                   help="step7 config yaml (llm mode)")
    p.add_argument("--max-rounds", type=int, default=5)
    add_prompt_style_args(p)
    add_backbone_args(p)
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not args.processed_dir.is_dir() or not args.step4_dir.is_dir():
        print("ERROR: --processed-dir / --step4-dir must be dirs",
              file=sys.stderr)
        return 1
    if args.max_rounds < 1:
        print("ERROR: --max-rounds must be >= 1", file=sys.stderr)
        return 1

    ids = [clean_sample_id(s) for s in read_sample_list(args.sample_list)]
    enriched = _load_enriched(args.enriched_model_dir)

    stats = generate(
        processed_dir=args.processed_dir, sample_ids=ids,
        step4_dir=args.step4_dir, out_dir=args.out_dir, mode=args.mode,
        enriched_model=enriched, step6_dir=args.step6_dir,
        config_path=args.config, max_rounds=args.max_rounds,
        temperature=args.temperature, cot=args.cot,
        backbone=backbone_from_args(args))
    avg_rounds = (stats["total_rounds"] / stats["written"]
                  if stats["written"] else 0.0)
    print(f"mode={args.mode}  wrote {stats['written']} iterative action "
          f"sets to {args.out_dir}  (avg rounds={avg_rounds:.2f}, "
          f"accepted={stats['accepted']}, skipped={stats['skipped']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
