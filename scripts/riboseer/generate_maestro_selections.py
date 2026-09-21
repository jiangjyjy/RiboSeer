#!/usr/bin/env python3
"""Generate per-sample MAESTRO tool selections for the Table 9 ablation.

MAESTRO picks, per sample, which tools from the library to fuse. This
script produces the selection JSON that ``table09_llm_modules.py`` reads
via ``--maestro-selections-{train,test}``:

    <out-dir>/<sample_id>.json
      {"sample_id": ..., "selected_tools": ["boltz2", ...],
       "reasoning": "...", "source": "ucb" | "llm" | "llm_fallback"}

Inputs per sample: the SCOPE profile (→ pocket category), the tools that
actually succeeded in step4 (availability), and the UCB utility scores
from the W tensor.

Modes
-----
* ``--mode auto`` (default; **no API**) — greedy UCB: rank the *available*
  tools by their ``WeightTensor.compute_utility`` score for the sample's
  category (scored per-tool, so non-deployed library tools are included)
  and keep the top-k, guaranteeing ≥1 Category-A structure tool when available
  (mirrors the LLM rule). Deterministic, offline. With no ``--weight-tensor``
  the tensor cold-starts to the per-category priors (A>C≈D>B).
* ``--mode llm`` — calls LLM 5.1 with the profile + the available tools
  listed **UCB-descending with no category labels or mandates** (so the
  model ranks by data-driven UCB, not tool fame/category), parses
  ``{"selected_tools", "reasoning"}``, and intersects the picks with what's
  actually available. Any API/parse failure falls back to the greedy-UCB
  pick (``source=llm_fallback``) so a run never dies on one bad sample.

Category resolution (for the UCB row): ``--step2-dir`` record's category
if present (matches a trained tensor's keys), else a pseudo-category built
from the SCOPE profile, else ``novel_fold_x_unstructured``.

Usage
-----
::

    # offline greedy-UCB selections (train split)
    python scripts/riboseer/generate_maestro_selections.py \
        --processed-dir data/processed_quality \
        --sample-list   data/processed_quality/splits_tmscore_035/train.txt \
        --step4-dir     data/batch_train_v7/step4/ \
        --out-dir       data/batch_train_v7/maestro_selections/ \
        --weight-tensor data/batch_train_v7/W.json \
        --mode auto

    # LLM selections (server)
    python scripts/riboseer/generate_maestro_selections.py \
        ... --mode llm --config configs/step3_config.yaml \
        --scope-profiles data/batch_train_v7/scope_profiles/
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

from step5_fusion.data_collector import load_sample_json, read_jsonl_record  # noqa: E402
from step3_tool_selection.weight_tensor import WeightTensor  # noqa: E402
from step4_tool_adapters.tool_io import (  # noqa: E402
    clean_sample_id, read_sample_list,
)
from step5_fusion.features_15tool import (  # noqa: E402
    ALL_KNOWN_TOOLS, TOOL_CATEGORY, _canonical, _index_predictions,
)
from scripts.riboseer.prompt_style import (  # noqa: E402
    add_prompt_style_args, resolve_temperature, with_cot,
)
from scripts.riboseer.llm_backbone import (  # noqa: E402
    add_backbone_args, backbone_from_args, client_from_backbone,
)

_CAT_A = "A"
DEFAULT_CATEGORY = "novel_fold_x_unstructured"


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def available_tools(step4_data: dict) -> list[str]:
    """Canonical ids of tools that succeeded in step4, in library order."""
    present = set(_index_predictions(step4_data).keys())
    return [t for t in ALL_KNOWN_TOOLS if t in present]


def utility_scores(weight_tensor: WeightTensor, tools: list[str],
                   category: str) -> dict[str, float]:
    """UCB utility for each given tool under ``category``.

    Scores tools directly via ``compute_utility`` rather than
    ``WeightTensor.compute_utility_scores``, which iterates only the
    *deployed* (``available=True``) registry tools and therefore drops the
    newer library tools (nucleicnet/bindup/…) — leaving them at UCB 0.000
    in the prompt. Every tool the caller passes gets a real score.
    """
    return {t: round(weight_tensor.compute_utility(t, category), 4)
            for t in tools}


def category_for(sample_id: str, step2_dir: Optional[Path],
                 profile: Optional[dict]) -> str:
    """Pocket category for the UCB row. step2 record > profile-derived >
    default."""
    if step2_dir is not None:
        rec = read_jsonl_record(step2_dir / f"{sample_id}.jsonl")
        if rec:
            cat = (rec.get("output") or {}).get("category") or rec.get("category")
            if cat:
                return str(cat)
    if profile:
        fam = str(profile.get("protein_family") or "novel").lower()
        rna = str(profile.get("rna_context") or "unstructured").lower()
        return f"{fam}_x_{rna}"
    return DEFAULT_CATEGORY


# ---------------------------------------------------------------------------
# greedy UCB (auto)
# ---------------------------------------------------------------------------


def greedy_ucb_select(available: list[str], utility: dict[str, float],
                      top_k: int) -> list[str]:
    """Top-k available tools by UCB utility, guaranteeing ≥1 Cat-A tool
    when one is available. Order: highest utility first."""
    if not available:
        return []
    ranked = sorted(available, key=lambda t: (-utility.get(t, 0.0), t))
    picked = ranked[:max(1, top_k)]
    # Guarantee a structure tool if the library offers one here.
    cat_a_avail = [t for t in ranked if TOOL_CATEGORY.get(t) == _CAT_A]
    if cat_a_avail and not any(TOOL_CATEGORY.get(t) == _CAT_A for t in picked):
        picked = picked[:-1] + [cat_a_avail[0]]
    # De-dupe preserving order.
    seen: set[str] = set()
    out = []
    for t in picked:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


# ---------------------------------------------------------------------------
# LLM selection
# ---------------------------------------------------------------------------


def build_maestro_prompt(profile: Optional[dict], available: list[str],
                         utility: dict[str, float], max_tools: int,
                         protein_len: int, rna_len: int,
                         cot: bool = True) -> str:
    prof = profile or {}
    # Rank strictly by UCB (descending), NOT by category/library order, and
    # show no category labels — every tool is presented identically so the
    # model judges it only on its data-driven UCB score.
    ranked = sorted(available, key=lambda t: (-utility.get(t, 0.0), t))
    tool_lines = "\n".join(
        f"  - {t}: UCB {utility.get(t, 0.0):.3f}" for t in ranked)
    prompt = (
        "You are an RNA–protein binding-site prediction expert. Select the "
        "best 3-" + str(max_tools) + " tools from the list below to fuse for "
        "this target.\n\n"
        f"Target profile:\n"
        f"  protein_family: {prof.get('protein_family', 'unknown')}\n"
        f"  rna_context: {prof.get('rna_context', 'unknown')}\n"
        f"  difficulty: {prof.get('difficulty', 'unknown')}\n"
        f"  protein_length: {protein_len} aa\n"
        f"  rna_length: {rna_len} nt\n\n"
        "Each tool's UCB score is a data-driven measure of how well that tool "
        "actually performed on the training data (higher = more reliable). "
        "Select strictly by UCB ranking: prefer the tools with the highest "
        "UCB scores. Judge every tool ONLY by its UCB score — not by its "
        "name, model family, or method type. All tools are equally "
        "eligible.\n\n"
        f"Tools, ranked by UCB score (high → low):\n{tool_lines}\n\n"
        "How many to select:\n"
        f"  - Take the highest-UCB tools, between 3 and {max_tools} of them.\n"
        "  - Harder / larger targets → lean toward more tools; easier / "
        "smaller targets → fewer.\n\n"
        'Output JSON: {"selected_tools": ["tool_id", ...], "reasoning": "..."}'
    )
    return with_cot(prompt, cot)


def llm_select(sample_id: str, profile: Optional[dict],
               available: list[str], utility: dict[str, float],
               protein_len: int, rna_len: int, client, config: dict,
               max_tools: int, temperature: Optional[float] = None,
               cot: bool = True) -> Optional[list[str]]:
    """One LLM round-trip → selected tool list (intersected with
    available). None on any failure (caller falls back to greedy)."""
    from step2_target_char.llm_client import (
        LLMError, extract_content, extract_json_object,
    )
    try:
        prompt = build_maestro_prompt(profile, available, utility, max_tools,
                                      protein_len, rna_len, cot)
        temperature = resolve_temperature(config, temperature)
        resp = client.call([{"role": "user", "content": prompt}],
                           temperature=temperature)
        parsed = extract_json_object(extract_content(resp) or "")
        if not parsed:
            return None
        picks = [_canonical(t) for t in (parsed.get("selected_tools") or [])]
        avail = set(available)
        picks = [t for t in picks if t in avail]
        if not picks:
            return None
        return (picks, str(parsed.get("reasoning", ""))[:1000])
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


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _seq_len(sample: dict, key: str) -> int:
    d = sample.get(key) or {}
    return int(d.get("length") or len(d.get("sequence") or "") or 0)


def generate(*, processed_dir: Path, sample_ids: list[str], step4_dir: Path,
             out_dir: Path, mode: str, weight_tensor: WeightTensor,
             scope_profiles: dict[str, dict], step2_dir: Optional[Path],
             config_path: Optional[Path], max_tools: int,
             top_k: int, temperature: Optional[float] = None,
             cot: bool = True,
             backbone: Optional[dict] = None) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = None
    config: dict = {}
    if mode == "llm":
        client, config = _build_llm_client(config_path, backbone)

    stats = {"written": 0, "skipped": 0, "llm": 0, "ucb": 0, "llm_fallback": 0}
    for sid in sample_ids:
        s4 = read_jsonl_record(step4_dir / f"{sid}.jsonl")
        sample = load_sample_json(processed_dir / "samples", sid) \
            if (processed_dir / "samples").is_dir() else \
            load_sample_json(processed_dir, sid)
        if s4 is None or sample is None:
            stats["skipped"] += 1
            continue
        avail = available_tools(s4)
        if not avail:
            stats["skipped"] += 1
            continue
        profile = scope_profiles.get(sid)
        category = category_for(sid, step2_dir, profile)
        utility = utility_scores(weight_tensor, avail, category)
        plen, rlen = _seq_len(sample, "protein"), _seq_len(sample, "rna")

        source = "ucb"
        reasoning = f"greedy UCB top-{top_k} for category {category}"
        selected = greedy_ucb_select(avail, utility, top_k)
        if mode == "llm":
            res = llm_select(sid, profile, avail, utility, plen, rlen,
                             client, config, max_tools, temperature, cot)
            if res is not None:
                selected, reasoning = res
                source = "llm"
            else:
                source = "llm_fallback"
        stats[source if source in stats else "ucb"] = \
            stats.get(source, 0) + 1

        (out_dir / f"{sid}.json").write_text(json.dumps({
            "sample_id": sid,
            "selected_tools": selected,
            "reasoning": reasoning,
            "category": category,
            "available_tools": avail,
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
    p.add_argument("--weight-tensor", type=Path, default=None,
                   help="W tensor JSON (cold-starts to priors if omitted)")
    p.add_argument("--scope-profiles", type=Path, default=None,
                   help="dir of SCOPE profile JSONs (per-sample category/LLM ctx)")
    p.add_argument("--step2-dir", type=Path, default=None,
                   help="dir of step2 JSONL records (authoritative category)")
    p.add_argument("--config", type=Path, default=None,
                   help="step3 config yaml (llm mode)")
    p.add_argument("--max-tools", type=int, default=6)
    p.add_argument("--top-k", type=int, default=5,
                   help="greedy-UCB pick size (auto / fallback)")
    add_prompt_style_args(p)
    add_backbone_args(p)
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not args.processed_dir.is_dir() or not args.step4_dir.is_dir():
        print("ERROR: --processed-dir / --step4-dir must be dirs",
              file=sys.stderr)
        return 1

    ids = [clean_sample_id(s) for s in read_sample_list(args.sample_list)]
    wt = WeightTensor.load(args.weight_tensor) if args.weight_tensor \
        else WeightTensor()
    scope_profiles = _load_json_dir(args.scope_profiles)

    stats = generate(
        processed_dir=args.processed_dir, sample_ids=ids,
        step4_dir=args.step4_dir, out_dir=args.out_dir, mode=args.mode,
        weight_tensor=wt, scope_profiles=scope_profiles,
        step2_dir=args.step2_dir, config_path=args.config,
        max_tools=args.max_tools, top_k=args.top_k,
        temperature=args.temperature, cot=args.cot,
        backbone=backbone_from_args(args))
    print(f"mode={args.mode}  wrote {stats['written']} selections to "
          f"{args.out_dir}  (llm={stats.get('llm', 0)}, "
          f"ucb={stats.get('ucb', 0)}, "
          f"llm_fallback={stats.get('llm_fallback', 0)}, "
          f"skipped={stats['skipped']})")
    return 0


def _load_json_dir(directory: Optional[Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if directory is None or not directory.is_dir():
        return out
    for f in directory.glob("*.json"):
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return out


if __name__ == "__main__":
    raise SystemExit(main())
