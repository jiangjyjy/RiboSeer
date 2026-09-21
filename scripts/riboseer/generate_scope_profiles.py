"""Generate per-sample SCOPE profile JSONs for the Table 9 ablation.

Two modes
---------
* ``--mode auto`` (default; **no API**) — writes the deterministic
  ``cauto`` heuristic profile (``features_15tool.auto_scope_profile``)
  for every sample in the split. This is the SCOPE control arm and runs
  fully offline.
* ``--mode llm`` — calls the project's LLM client (step2 ``LLMClient``)
  once per sample to produce the LLM profile. Intended to run on the server
  where ``LLM_API_KEY`` and full data live. The prompt mirrors the SCOPE
  schema (protein_family / rna_context / difficulty / confidence). On any
  per-sample API/parse failure it falls back to the cauto profile and
  records ``"source": "cauto_fallback"`` so the run never dies on one bad
  sample.

Each profile is written to ``<out-dir>/<sample_id>.json`` with the schema
``encode_scope_profile`` consumes::

    {"sample_id", "protein_family", "rna_context", "difficulty",
     "confidence", "source", "reasoning"?}

Usage
-----
::

    # offline control profiles
    python scripts/riboseer/generate_scope_profiles.py \\
        --processed-dir data/processed_quality \\
        --sample-list   data/processed_quality/splits_tmscore_035/test.txt \\
        --out-dir       data/batch_test_v7/scope_profiles/ \\
        --mode auto

    # LLM profiles (server)
    python scripts/riboseer/generate_scope_profiles.py \\
        ... --mode llm --config config/step2_config.yaml
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

from step5_fusion.data_collector import load_sample_json  # noqa: E402
from step5_fusion.features_15tool import (  # noqa: E402
    SCOPE_FAMILIES, SCOPE_RNA_CONTEXTS, auto_scope_profile,
)
from scripts.riboseer.prompt_style import (  # noqa: E402
    add_prompt_style_args, resolve_temperature, with_cot,
)
from scripts.riboseer.llm_backbone import (  # noqa: E402
    add_backbone_args, backbone_from_args, client_from_backbone,
)


def _load_sample_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"sample list not found: {path}")
    return [ln.split()[0] for ln in
            path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


# ---------------------------------------------------------------------------
# LLM prompt + parsing (server-side mode)
# ---------------------------------------------------------------------------


def build_scope_prompt(sample: dict, cot: bool = True) -> str:
    """SCOPE-profile prompt for one sample (plan §1.3)."""
    prot = sample.get("protein") or {}
    rna = sample.get("rna") or sample.get("ligand_rna") or {}
    plen = prot.get("length") or len(prot.get("sequence") or "")
    rlen = rna.get("length") or len(rna.get("sequence") or "")
    pseq = (prot.get("sequence") or "")[:400]
    rseq = (rna.get("sequence") or "")[:200]
    fam = ", ".join(SCOPE_FAMILIES)
    ctx = ", ".join(SCOPE_RNA_CONTEXTS)
    base = (
        "You are an RNA–protein interaction expert. Characterise the "
        "target below.\n\n"
        f"Protein length: {plen} aa\nProtein sequence (truncated): {pseq}\n"
        f"RNA length: {rlen} nt\nRNA sequence (truncated): {rseq}\n\n"
        f"Choose protein_family from: {fam}.\n"
        f"Choose rna_context from: {ctx}.\n"
        "Choose difficulty from: Easy, Medium, Hard.\n"
        "Give confidence in [0,1].\n\n"
        'Output JSON: {"protein_family": "...", "rna_context": "...", '
        '"difficulty": "...", "confidence": 0.0, "reasoning": "..."}'
    )
    return with_cot(base, cot)


def _normalise_llm_profile(sample_id: str, raw: dict, sample: dict) -> dict:
    """Coerce a parsed LLM response into the canonical profile schema,
    snapping out-of-vocab labels to the nearest valid value."""
    fam = str(raw.get("protein_family", "Novel"))
    if fam not in SCOPE_FAMILIES:
        fam = "Novel"
    ctx = str(raw.get("rna_context", "unstructured"))
    if ctx not in SCOPE_RNA_CONTEXTS:
        ctx = "unstructured"
    diff = str(raw.get("difficulty", "Medium")).strip().capitalize()
    if diff not in ("Easy", "Medium", "Hard"):
        diff = "Medium"
    try:
        conf = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return {
        "sample_id": sample_id,
        "protein_family": fam,
        "rna_context": ctx,
        "difficulty": diff,
        "confidence": min(max(conf, 0.0), 1.0),
        "reasoning": str(raw.get("reasoning", ""))[:1000],
        "source": "llm",
    }


def llm_profile_for_sample(sample_id: str, sample: dict, client,
                           config: dict, temperature: Optional[float] = None,
                           cot: bool = True) -> dict:
    """One LLM round-trip → profile dict. Falls back to cauto on any
    failure (never raises)."""
    from step2_target_char.llm_client import (  # local import: server-only
        LLMError, extract_content, extract_json_object,
    )
    try:
        messages = [{"role": "user",
                     "content": build_scope_prompt(sample, cot)}]
        temperature = resolve_temperature(config, temperature)
        resp = client.call(messages, temperature=temperature)
        parsed = extract_json_object(extract_content(resp) or "")
        if parsed is None:
            raise ValueError("no JSON object in response")
        return _normalise_llm_profile(sample_id, parsed, sample)
    except (LLMError, ValueError, KeyError) as e:
        fb = auto_scope_profile(sample)
        fb["source"] = "cauto_fallback"
        fb["fallback_reason"] = str(e)[:200]
        return fb


def _build_llm_client(config_path: Optional[Path],
                      backbone: Optional[dict] = None):
    """Construct the LLM client (server-side).

    With a ``backbone`` config (Table 13 ``--api-base``) → a
    ``BackboneClient`` for that endpoint, config ``{}``. Otherwise a
    ``LLMClient`` via ``step2_target_char.run.build_client`` so the api_key
    comes from ``$LLM_API_KEY`` and config ``api.{model,url,...}`` map to
    kwargs (defaults: model llm-model + env key)."""
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
# Driver
# ---------------------------------------------------------------------------


def generate(processed_dir: Path, sample_ids: list[str], out_dir: Path,
             mode: str, config_path: Optional[Path],
             temperature: Optional[float] = None, cot: bool = True,
             backbone: Optional[dict] = None
             ) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = None
    config: dict = {}
    if mode == "llm":
        client, config = _build_llm_client(config_path, backbone)

    stats = {"written": 0, "skipped": 0, "llm": 0, "cauto": 0}
    for sid in sample_ids:
        sample = load_sample_json(processed_dir, sid)
        if sample is None:
            stats["skipped"] += 1
            continue
        sample.setdefault("sample_id", sid)
        if mode == "llm":
            profile = llm_profile_for_sample(sid, sample, client, config,
                                             temperature, cot)
        else:
            profile = auto_scope_profile(sample)
        if profile.get("source", "").startswith("llm"):
            stats["llm"] += 1
        else:
            stats["cauto"] += 1
        (out_dir / f"{sid}.json").write_text(
            json.dumps(profile, ensure_ascii=False, indent=2),
            encoding="utf-8")
        stats["written"] += 1
    return stats


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--sample-list", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--mode", choices=("auto", "llm"), default="auto")
    p.add_argument("--config", type=Path, default=None,
                   help="step2 config yaml (llm mode only)")
    add_prompt_style_args(p)
    add_backbone_args(p)
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not args.processed_dir.is_dir():
        print(f"ERROR: not a directory: {args.processed_dir}",
              file=sys.stderr)
        return 1
    try:
        ids = _load_sample_ids(args.sample_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    stats = generate(args.processed_dir, ids, args.out_dir,
                     args.mode, args.config,
                     temperature=args.temperature, cot=args.cot,
                     backbone=backbone_from_args(args))
    print(f"mode={args.mode}  wrote {stats['written']} profiles to "
          f"{args.out_dir}  (llm={stats['llm']}, cauto={stats['cauto']}, "
          f"skipped={stats['skipped']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
