#!/usr/bin/env python3
"""Generate per-sample POLISH actions for the Table 9 ablation.

POLISH reviews the fused prediction and optionally edits it
(mask / extend / relocate). This script produces the action JSON that
``table09_llm_modules.py`` reads via ``--polish-actions``:

    <out-dir>/<sample_id>.json
      {"sample_id": ..., "action": "mask"|"extend"|"relocate"|"accept",
       "residues": [...], "target_residues": [...]?,
       "reasoning": "...", "source": "auto" | "llm" | "llm_fallback"}

(The ablation also accepts a multi-action ``{"actions": [...]}`` wrapper;
this script writes the single-action flat form, which it consumes as a
one-element sequence.)

Inputs per sample: the step4 tool predictions, the fused per-residue
probability, and (optionally) the PocketQA / VERDICT score.

Modes
-----
* ``--mode auto`` (default; **no API**) — deterministic
  ``polish_ops.auto_polish_action``: mask the weakest neighbour-smoothed
  binding residues (conservative false-positive removal). Offline.
* ``--mode llm`` — calls LLM 5.1 with the **continuous** per-residue
  probabilities split into confident (p>0.7) and boundary (0.3–0.7) bands,
  the per-tool score + support count for each boundary residue, and the
  VERDICT score; parses
  ``{"action","residues","target_residues","confidence","reasoning"}``.
  ``confidence`` (0–1) scales how hard the edit hits the probability
  vector downstream (``polish_ops``). Any API/parse failure falls back to
  the auto action (``source=llm_fallback``).

Typically run on the **test split only** (107 samples).

Fused probability: ``EnrichedFusion.predict_sample`` when
``--enriched-model-dir`` is given, else the per-residue mean of the
available tools' scores (pae→confidence priority).

Usage
-----
::

    # offline rule-based actions (test split)
    python scripts/riboseer/generate_polish_actions.py \
        --processed-dir data/processed_quality \
        --sample-list   data/processed_quality/splits_tmscore_035/test.txt \
        --step4-dir     data/batch_test_v7/step4/ \
        --out-dir       data/batch_test_v7/polish_actions/ \
        --mode auto

    # LLM actions (server)
    python scripts/riboseer/generate_polish_actions.py \
        ... --mode llm --config configs/step7_config.yaml \
        --enriched-model-dir data/enriched_v7_model/ \
        --step6-dir data/batch_test_v7/step6/
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
    load_sample_json, per_residue_to_int_dict, read_jsonl_record,
)
from step4_tool_adapters.tool_io import (  # noqa: E402
    clean_sample_id, read_sample_list,
)
from step5_fusion.features_15tool import _index_predictions  # noqa: E402
from step7_iteration.polish_ops import auto_polish_action  # noqa: E402
from scripts.riboseer.prompt_style import (  # noqa: E402
    add_prompt_style_args, resolve_temperature, with_cot,
)
from scripts.riboseer.llm_backbone import client_from_backbone  # noqa: E402

_VALID_ACTIONS = {"mask", "extend", "relocate", "accept"}


# ---------------------------------------------------------------------------
# fused probability
# ---------------------------------------------------------------------------


def _tool_scores(pred: dict) -> dict[int, float]:
    pae = per_residue_to_int_dict(pred.get("per_residue_pae_score"))
    if pae:
        return pae
    return per_residue_to_int_dict(pred.get("per_residue_confidence"))


def mean_fused_probability(step4_data: dict) -> dict[int, float]:
    """Per-residue mean of every successful tool's score (pae→conf).
    Self-contained fallback when no enriched model is supplied."""
    preds = _index_predictions(step4_data)
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for pred in preds.values():
        for rid, sc in _tool_scores(pred).items():
            sums[rid] = sums.get(rid, 0.0) + float(sc)
            counts[rid] = counts.get(rid, 0) + 1
    return {rid: sums[rid] / counts[rid] for rid in sums if counts[rid]}


def fused_probability(step4_data: dict, sample: dict,
                      enriched_model) -> dict[int, float]:
    if enriched_model is not None:
        prot = sample.get("protein") or {}
        length = prot.get("length") or len(prot.get("sequence") or "")
        preds = (step4_data or {}).get("predictions") or []
        try:
            return enriched_model.predict_sample(
                preds, prot.get("sequence") or "", int(length or 0))
        except Exception:  # noqa: BLE001 — fall back to mean
            pass
    return mean_fused_probability(step4_data)


# ---------------------------------------------------------------------------
# LLM action
# ---------------------------------------------------------------------------


def tool_agreement_summary(step4_data: dict,
                           binding: list[int]) -> str:
    """For the (first 20) binding residues, how many tools gate each."""
    preds = _index_predictions(step4_data)
    gates: dict[int, int] = {}
    for pred in preds.values():
        for r in (pred.get("binding_protein_residues") or []):
            try:
                gates[int(r)] = gates.get(int(r), 0) + 1
            except (TypeError, ValueError):
                continue
    n_tools = len(preds)
    parts = [f"r{r}:{gates.get(r, 0)}/{n_tools}" for r in binding[:20]]
    return ", ".join(parts)


_HIGH_CONF = 0.7   # p > 0.7  → confident binding
_BOUNDARY_LO = 0.3  # 0.3 <= p <= 0.7 → ambiguous boundary


def _fmt_residue_probs(residues: list[int], prob: dict[int, float]) -> str:
    return ", ".join(f"{r}({prob.get(r, 0.0):.2f})" for r in residues)


def grouped_probability_lines(prob: dict[int, float]
                              ) -> tuple[list[int], list[int], str, str]:
    """Split residues into confident (p>0.7) and boundary (0.3<=p<=0.7)
    bands and render each as ``<id>(<prob>)`` strings (sorted by id)."""
    high = sorted(r for r, p in prob.items() if p > _HIGH_CONF)
    boundary = sorted(r for r, p in prob.items()
                      if _BOUNDARY_LO <= p <= _HIGH_CONF)
    return high, boundary, _fmt_residue_probs(high, prob), \
        _fmt_residue_probs(boundary, prob)


def per_tool_boundary_lines(step4_data: dict, boundary: list[int], *,
                            max_residues: int = 15,
                            support_threshold: float = 0.5) -> str:
    """For each boundary residue, list every tool's per-residue score and
    how many tools 'support' it (score > 0.5) — e.g.::

        Residue 22: boltz2=0.6, chai1=0.3, equipnas=0.8 → 2/3 support
    """
    preds = _index_predictions(step4_data)
    tool_scores = {tid: _tool_scores(p) for tid, p in preds.items()}
    lines: list[str] = []
    for r in boundary[:max_residues]:
        present = [(tid, sc[r]) for tid, sc in sorted(tool_scores.items())
                   if r in sc]
        if not present:
            continue
        support = sum(1 for _, v in present if v > support_threshold)
        per = ", ".join(f"{tid}={v:.2g}" for tid, v in present)
        lines.append(f"  Residue {r}: {per} → {support}/{len(present)} support")
    return "\n".join(lines)


def build_polish_prompt(protein_len: int, prob: dict[int, float],
                        step4_data: dict, verdict: Optional[float],
                        cot: bool = True) -> str:
    high, boundary, high_txt, bnd_txt = grouped_probability_lines(prob)
    tool_txt = per_tool_boundary_lines(step4_data, boundary)
    vtxt = f"{verdict:.3f}" if verdict is not None else "n/a"
    prompt = (
        "You are a quality reviewer for RNA–protein binding-site "
        "predictions. Review the fused per-residue probabilities below and "
        "decide whether to correct them. Output ONLY a JSON object.\n\n"
        f"Protein length: {protein_len}\n"
        f"VERDICT quality score: {vtxt}\n\n"
        f"High-confidence binding residues (probability > {_HIGH_CONF:.1f}):\n"
        f"  residues {high_txt if high else '(none)'}\n\n"
        f"Boundary residues (probability {_BOUNDARY_LO:.1f}-{_HIGH_CONF:.1f}):\n"
        f"  residues {bnd_txt if boundary else '(none)'}\n\n"
        "Per-tool predictions for the boundary residues:\n"
        f"{tool_txt if tool_txt else '  (no per-tool detail available)'}\n\n"
        "High-confidence residues are usually real and should be kept; boundary residues with little tool support (e.g. 1/3) are more likely false positives.\n"
        "Decide:\n"
        "  1. Clear false positives to mask? (prefer boundary residues with little tool support)\n"
        "  2. A missed contiguous stretch to extend?\n"
        "  3. Relocate the pocket to a more consistent region?\n\n"
        "Also give a confidence (0-1): how sure you are about this edit. The higher the "
        "confidence, the larger the edit applied downstream; if unsure, give a low confidence or just accept.\n\n"
        'Output JSON: {"action": "mask"|"extend"|"relocate"|"accept", '
        '"residues": [...], "target_residues": [...], '
        '"confidence": 0.0-1.0, "reasoning": "..."}'
    )
    return with_cot(prompt, cot)


def _parse_confidence(parsed: dict) -> Optional[float]:
    conf = parsed.get("confidence")
    if conf is None:
        return None
    try:
        return max(0.0, min(1.0, float(conf)))
    except (TypeError, ValueError):
        return None


def llm_action(protein_len: int, prob: dict[int, float],
               step4_data: dict, verdict: Optional[float], client,
               config: dict, temperature: Optional[float] = None,
               cot: bool = True) -> Optional[dict]:
    from step2_target_char.llm_client import (
        LLMError, extract_content, extract_json_object,
    )
    try:
        prompt = build_polish_prompt(protein_len, prob, step4_data, verdict,
                                     cot)
        temperature = resolve_temperature(config, temperature)
        resp = client.call([{"role": "user", "content": prompt}],
                           temperature=temperature)
        parsed = extract_json_object(extract_content(resp) or "")
        if not parsed:
            return None
        action = str(parsed.get("action", "accept")).lower()
        if action not in _VALID_ACTIONS:
            action = "accept"
        return {
            "action": action,
            "residues": [int(r) for r in (parsed.get("residues") or [])
                         if str(r).lstrip("-").isdigit()],
            "target_residues": [int(r) for r in
                                (parsed.get("target_residues") or [])
                                if str(r).lstrip("-").isdigit()],
            "confidence": _parse_confidence(parsed),
            "reasoning": str(parsed.get("reasoning", ""))[:1000],
        }
    except (LLMError, ValueError, KeyError, TypeError):
        return None


def _build_llm_client(config_path: Optional[Path],
                      backbone: Optional[dict] = None):
    client = client_from_backbone(backbone)
    if client is not None:
        return client, {}
    import yaml  # type: ignore
    from step2_target_char.run import build_client
    cfg = {}
    if config_path and config_path.is_file():
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return build_client(cfg), cfg


def _load_enriched(model_dir: Optional[Path]):
    if model_dir is None:
        return None
    try:
        from step5_fusion.enriched_fusion import EnrichedFusion
        return EnrichedFusion.load(model_dir)
    except Exception as e:  # noqa: BLE001
        print(f"WARN: could not load enriched model ({e}); using mean fusion",
              file=sys.stderr)
        return None


def _verdict_score(step6_dir: Optional[Path], sid: str) -> Optional[float]:
    if step6_dir is None:
        return None
    rec = read_jsonl_record(step6_dir / f"{sid}.jsonl")
    if not rec:
        return None
    v = rec.get("total_score")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def generate(*, processed_dir: Path, sample_ids: list[str], step4_dir: Path,
             out_dir: Path, mode: str, enriched_model,
             step6_dir: Optional[Path], config_path: Optional[Path],
             temperature: Optional[float] = None,
             cot: bool = True) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = None
    config: dict = {}
    if mode == "llm":
        client, config = _build_llm_client(config_path)

    stats = {"written": 0, "skipped": 0, "auto": 0, "llm": 0,
             "llm_fallback": 0}
    for sid in sample_ids:
        s4 = read_jsonl_record(step4_dir / f"{sid}.jsonl")
        samples_dir = processed_dir / "samples"
        sample = load_sample_json(
            samples_dir if samples_dir.is_dir() else processed_dir, sid)
        if s4 is None or sample is None:
            stats["skipped"] += 1
            continue
        prob = fused_probability(s4, sample, enriched_model)
        if not prob:
            stats["skipped"] += 1
            continue
        prot = sample.get("protein") or {}
        plen = int(prot.get("length") or len(prot.get("sequence") or "") or 0)

        source = "auto"
        action = auto_polish_action(prob)
        action.setdefault("target_residues", [])
        if mode == "llm":
            verdict = _verdict_score(step6_dir, sid)
            res = llm_action(plen, prob, s4, verdict, client, config,
                             temperature, cot)
            if res is not None:
                action = res
                source = "llm"
            else:
                source = "llm_fallback"
        stats[source] = stats.get(source, 0) + 1

        (out_dir / f"{sid}.json").write_text(json.dumps({
            "sample_id": sid,
            "action": action.get("action", "accept"),
            "residues": action.get("residues", []),
            "target_residues": action.get("target_residues", []),
            "confidence": action.get("confidence"),
            "reasoning": action.get("reasoning", ""),
            "source": source,
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
    p.add_argument("--enriched-model-dir", type=Path, default=None,
                   help="EnrichedFusion bundle for fused probability "
                        "(else per-residue mean of tool scores)")
    p.add_argument("--step6-dir", type=Path, default=None,
                   help="dir of step6 PocketQA JSONL (VERDICT score, llm ctx)")
    p.add_argument("--config", type=Path, default=None,
                   help="step7 config yaml (llm mode)")
    add_prompt_style_args(p)
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not args.processed_dir.is_dir() or not args.step4_dir.is_dir():
        print("ERROR: --processed-dir / --step4-dir must be dirs",
              file=sys.stderr)
        return 1

    ids = [clean_sample_id(s) for s in read_sample_list(args.sample_list)]
    enriched = _load_enriched(args.enriched_model_dir)

    stats = generate(
        processed_dir=args.processed_dir, sample_ids=ids,
        step4_dir=args.step4_dir, out_dir=args.out_dir, mode=args.mode,
        enriched_model=enriched, step6_dir=args.step6_dir,
        config_path=args.config, temperature=args.temperature, cot=args.cot)
    print(f"mode={args.mode}  wrote {stats['written']} actions to "
          f"{args.out_dir}  (auto={stats.get('auto', 0)}, "
          f"llm={stats.get('llm', 0)}, "
          f"llm_fallback={stats.get('llm_fallback', 0)}, "
          f"skipped={stats['skipped']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
