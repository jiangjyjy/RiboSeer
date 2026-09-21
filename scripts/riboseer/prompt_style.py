"""Shared prompt-engineering knobs for paper Table 14 (prompt ablation).

Two parameters are ablated across the SCOPE / MAESTRO / POLISH LLM prompts:

* **CoT** (chain-of-thought) — whether the model is asked to reason step by
  step before emitting the JSON answer. ``cot_clause(cot)`` returns the
  trailing instruction appended to each prompt: a "think step by step,
  reason first" clause when on, a "JSON only, no reasoning" clause when off.
* **Temperature** — the LLM sampling temperature. ``resolve_temperature``
  picks an explicit override (the Table-14 CLI value) over the config's
  ``api.temperature`` over a default.

The generators stay otherwise identical, so a (CoT, T) cell differs from the
default (CoT=on, T=0.7) only by these two knobs.
"""
from __future__ import annotations

import argparse
from typing import Optional

DEFAULT_TEMPERATURE = 0.7

_COT_CLAUSE = (
    "Let's think step by step. First write a brief \"Reasoning:\" analysis "
    "of the evidence, then output the final JSON object after your reasoning."
)
_DIRECT_CLAUSE = (
    "Do not explain. Output ONLY the JSON object, with no reasoning, "
    "preamble, or text before or after it."
)


def cot_clause(cot: bool) -> str:
    """Trailing prompt instruction selecting chain-of-thought vs direct."""
    return _COT_CLAUSE if cot else _DIRECT_CLAUSE


def with_cot(prompt: str, cot: bool) -> str:
    """Append the CoT/direct clause to a base prompt."""
    return f"{prompt}\n\n{cot_clause(cot)}"


def resolve_temperature(config: Optional[dict], override: Optional[float] = None,
                        default: float = 0.1) -> float:
    """Sampling temperature: explicit ``override`` > ``config.api.temperature``
    > ``default``. Tolerates a missing/garbled config."""
    if override is not None:
        try:
            return float(override)
        except (TypeError, ValueError):
            return default
    try:
        return float((config or {}).get("api", {}).get("temperature", default))
    except (TypeError, ValueError, AttributeError):
        return default


def add_prompt_style_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--cot/--no-cot`` (default on) and ``--temperature`` (default
    0.7) to a generator's argparse parser. Avoids ``BooleanOptionalAction``
    for broad Python compatibility."""
    parser.add_argument("--cot", dest="cot", action="store_true", default=True,
                        help="use chain-of-thought prompts (default)")
    parser.add_argument("--no-cot", dest="cot", action="store_false",
                        help="direct JSON-only prompts (no reasoning)")
    parser.add_argument("--temperature", type=float,
                        default=DEFAULT_TEMPERATURE,
                        help="LLM sampling temperature (default 0.7)")
