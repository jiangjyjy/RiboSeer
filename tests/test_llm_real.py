"""Real LLM API capability probe — RUN ON SERVER ONLY.

This script is intentionally NOT run locally by Claude Code. It needs:
  - network access to https://api.llm-provider.example
  - a valid API key exported as `LLM_API_KEY`

Usage (on server):
    cd ~/riboseer
    conda activate riboseer
    export LLM_API_KEY="<your real key>"
    python tests/test_llm_real.py

Each probe prints:
  - whether it succeeded
  - the assistant's raw content (truncated)
  - token usage
  - any tool_calls / structured output if the endpoint returned them

What we're trying to learn
--------------------------
1. `test_connection`  — basic round-trip: can we talk to the LLM endpoint at all?
2. `test_json_mode`   — does `response_format={"type":"json_object"}` do anything,
                        or is it silently ignored?  If honored, Step 2's prompt
                        can lean on JSON mode; if not, we parse JSON from the
                        free-form text ourselves.
3. `test_function_calling` — does the endpoint accept OpenAI-style `tools` +
                             return `tool_calls`?  Useful for step 3 (tool
                             selection), not strictly needed for step 2 — but
                             we probe now so later steps don't have to.

Expected cost: < 500 tokens total. Negligible.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step2_target_char.llm_client import (  # noqa: E402
    LLMClient, LLMError, extract_content, extract_tool_calls,
)


def _banner(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _dump_usage(result: dict) -> None:
    usage = result.get("usage") or {}
    print(f"  usage: prompt={usage.get('prompt_tokens')} "
          f"completion={usage.get('completion_tokens')} "
          f"total={usage.get('total_tokens')}")


def _dump_content(result: dict, max_chars: int = 500) -> None:
    try:
        c = extract_content(result)
    except LLMError as e:
        print(f"  (no content: {e})")
        return
    snippet = c if len(c) <= max_chars else c[:max_chars] + f"... (+{len(c)-max_chars} chars)"
    print(f"  content:\n{snippet}")


def probe_connection(client: LLMClient) -> bool:
    _banner("1. test_connection — basic round trip")
    try:
        result = client.test_connection()
    except LLMError as e:
        print(f"  FAIL: {e}")
        return False
    print("  OK")
    _dump_content(result)
    _dump_usage(result)
    return True


def probe_json_mode(client: LLMClient) -> bool:
    _banner("2. test_json_mode — response_format={'type':'json_object'}")
    try:
        result = client.test_json_mode()
    except LLMError as e:
        print(f"  FAIL (endpoint rejected the request): {e}")
        # Rejection != silent ignore. Many endpoints 400 on unknown params.
        # Report and let the human decide the strategy.
        return False
    print("  request accepted")
    _dump_content(result)
    _dump_usage(result)
    # Try to parse the content as JSON — if json_mode is honored, this
    # should always succeed; if not, the model *usually* still returns
    # valid JSON because we asked for it in the prompt too.
    try:
        parsed = json.loads(extract_content(result))
        print(f"  content parses as JSON: {parsed!r}")
    except (ValueError, LLMError) as e:
        print(f"  content does NOT parse as JSON: {e}")
    return True


def probe_function_calling(client: LLMClient) -> bool:
    _banner("3. test_function_calling — tools + tool_choice=auto")
    try:
        result = client.test_function_calling()
    except LLMError as e:
        print(f"  FAIL (endpoint rejected the request): {e}")
        return False
    print("  request accepted")
    tool_calls = extract_tool_calls(result)
    if tool_calls:
        print(f"  tool_calls returned ({len(tool_calls)}):")
        for tc in tool_calls:
            print(f"    - {json.dumps(tc, ensure_ascii=False)}")
    else:
        print("  NO tool_calls in response — endpoint may ignore `tools` silently")
        _dump_content(result)
    _dump_usage(result)
    return True


def main() -> int:
    if not os.environ.get("LLM_API_KEY"):
        print("ERROR: LLM_API_KEY is not exported. Run:")
        print('    export LLM_API_KEY="<your key>"')
        return 2

    client = LLMClient()
    results = {
        "connection": probe_connection(client),
        "json_mode_accepted": probe_json_mode(client),
        "function_calling_accepted": probe_function_calling(client),
    }

    _banner("summary")
    for k, v in results.items():
        print(f"  {k}: {'OK' if v else 'FAIL'}")
    print(f"\nUsage log appended to: {client.usage_log}")
    print("Paste the full output back to Claude Code to decide Step 2's "
          "prompt / parsing strategy.")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
