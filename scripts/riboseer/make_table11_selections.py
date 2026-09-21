#!/usr/bin/env python3
"""Table 11 — derive the MANDATORY-tools and All-tools MAESTRO selections.

Table 11 compares three MAESTRO tool-selection policies, holding everything
else fixed.  Read the **scope=on / maestro=on / polish=off** row of
``table09_llm_modules.py`` for each policy (only the ``--maestro-selections``
dir changes between runs):

1. **Free LLM selection**     — the LLM's own picks
   (``maestro_selections_llm_v4/``); this is exactly Table 9's
   (scope=on, maestro=on, polish=off) row, so no rerun is needed.
2. **MANDATORY Cat-A + LLM**  — force-include the 5 Category-A structure
   tools (boltz2 / chai1 / rosettafold2na / rfaa / alphafold3), keep the
   LLM's extra picks on top.
3. **All 15 tools always**    — every library tool, no selection.

This script derives policies 2 and 3 **from the Free-LLM selection dir** so
all three policies score on exactly the same sample set.  It writes the same
per-sample JSON schema ``table09_llm_modules.py`` consumes
(``<out-dir>/<sid>.json`` with a load-bearing ``selected_tools`` list).

Run once per split per policy::

    # MANDATORY (test split)
    python scripts/riboseer/make_table11_selections.py --policy mandatory \\
        --input-dir data/batch_test_v7/maestro_selections_llm_v4/ \\
        --out-dir   data/batch_test_v7/maestro_selections_mandatory/
    # MANDATORY (train split)
    python scripts/riboseer/make_table11_selections.py --policy mandatory \\
        --input-dir data/batch_train_v7/maestro_selections_llm_v4/ \\
        --out-dir   data/batch_train_v7/maestro_selections_mandatory/
    # All 15 tools (test + train, same pattern with --policy all)
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

from step5_fusion.features_15tool import (  # noqa: E402
    ALL_KNOWN_TOOLS, TOOL_CATEGORY, _canonical,
)

# The 5 Category-A structure tools, in library order.
MANDATORY_CAT_A: tuple[str, ...] = tuple(
    t for t in ALL_KNOWN_TOOLS if TOOL_CATEGORY[t] == "A")

POLICIES = ("mandatory", "all")
_LIB_INDEX = {t: i for i, t in enumerate(ALL_KNOWN_TOOLS)}


def clean_tools(tools: list[str]) -> list[str]:
    """Canonicalise step4 aliases, drop unknown ids, de-dupe, return in
    library order. Mirrors ``table09_llm_modules.resolve_selected_tools``."""
    seen: set[str] = set()
    for t in tools or []:
        c = _canonical(t)
        if c in _LIB_INDEX:
            seen.add(c)
    return sorted(seen, key=lambda t: _LIB_INDEX[t])


def mandatory_tools(original: list[str]) -> list[str]:
    """LLM picks ∪ the 5 Category-A tools, in library order."""
    merged = set(clean_tools(original)) | set(MANDATORY_CAT_A)
    return sorted(merged, key=lambda t: _LIB_INDEX[t])


def all_tools() -> list[str]:
    """Every library tool, in library order."""
    return list(ALL_KNOWN_TOOLS)


def selected_for_policy(original: list[str], policy: str) -> list[str]:
    if policy == "mandatory":
        return mandatory_tools(original)
    if policy == "all":
        return all_tools()
    raise ValueError(f"unknown policy: {policy}")


def build_record(record: dict, policy: str) -> dict:
    """New selection record: rewrite ``selected_tools`` per policy, keep the
    rest of the original metadata for traceability."""
    original = list(record.get("selected_tools") or [])
    out = dict(record)
    out["selected_tools"] = selected_for_policy(original, policy)
    out["policy"] = policy
    out["llm_selected_tools"] = clean_tools(original)
    out["source"] = f"table11_{policy}"
    return out


def load_selection_dir(directory: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in sorted(directory.glob("*.json")):
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def avg_tools_per_sample(records: dict[str, dict]) -> Optional[float]:
    counts = [len(clean_tools(r.get("selected_tools") or []))
              for r in records.values()]
    return round(sum(counts) / len(counts), 2) if counts else None


def generate(input_dir: Path, out_dir: Path, policy: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_selection_dir(input_dir)
    written: dict[str, dict] = {}
    for sid, rec in records.items():
        new_rec = build_record(rec, policy)
        (out_dir / f"{sid}.json").write_text(
            json.dumps(new_rec, ensure_ascii=False, indent=2),
            encoding="utf-8")
        written[sid] = new_rec
    return {"written": len(written),
            "avg_tools": avg_tools_per_sample(written)}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=POLICIES, required=True)
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Free-LLM selection dir (maestro_selections_llm_v4/)")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not args.input_dir.is_dir():
        print(f"ERROR: not a directory: {args.input_dir}", file=sys.stderr)
        return 1

    stats = generate(args.input_dir, args.out_dir, args.policy)
    if stats["written"] == 0:
        print(f"ERROR: no selection JSONs found in {args.input_dir}",
              file=sys.stderr)
        return 1
    print(f"policy={args.policy}  wrote {stats['written']} selections to "
          f"{args.out_dir}  (avg tools/sample = {stats['avg_tools']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
