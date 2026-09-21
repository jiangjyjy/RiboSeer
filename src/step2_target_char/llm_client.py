"""LLM API client wrapper for RiboSeer Step 2.

Provides:
  - `LLMClient`: thin wrapper around requests.post, with:
      * API key from env var `LLM_API_KEY`
      * automatic retry on transient errors (network / 5xx / 429) with
        exponential backoff
      * per-call token usage logged to `logs/llm_usage.jsonl`
  - capability probes (`test_connection`, `test_json_mode`,
    `test_function_calling`) — run on the server to discover whether the
    deployed LLM endpoint supports JSON mode / tool calls; Step 2's prompt
    strategy branches on the result.

Design notes
------------
- The Authorization header uses "Bearer {api_key}" format as required by the
  the LLM API provider endpoint.
- 4xx responses other than 429 are *not* retried — they indicate a client
  bug (bad payload, auth, model-not-found) where retrying wastes quota.
- Usage log is append-only JSONL so concurrent runs never corrupt it; each
  line is self-describing (has timestamp, latency, success flag).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests


_log = logging.getLogger(__name__)


DEFAULT_URL = "https://api.llm-provider.example/v1/chat/completions"
DEFAULT_USAGE_LOG = Path("logs/llm_usage.jsonl")

# Substrings the upstream API returns when it rejects ``response_format``.
# coding-plan endpoint returns:
#   "The parameter response_format.type specified in the request are not
#    valid: json_object is not supported"
# We match either token to be future-proof against re-wordings.
_RESPONSE_FORMAT_REJECT_TOKENS = ("json_object", "response_format")


class LLMError(RuntimeError):
    """Raised when the API returns a non-retryable error or retries exhaust."""


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llm-model",
        url: str = DEFAULT_URL,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        timeout_seconds: float = 180.0,
        usage_log: Path = DEFAULT_USAGE_LOG,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY")
        if not self.api_key:
            raise LLMError(
                "LLM_API_KEY not set — export it in the shell or pass api_key=... "
                "(don't hard-code keys in source)"
            )
        self.model = model
        self.url = url
        self.max_retries = int(max_retries)
        self.retry_base_delay = float(retry_base_delay)
        self.timeout = float(timeout_seconds)
        self.usage_log = Path(usage_log)

    # -- public API --------------------------------------------------------

    def call(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        response_format: Optional[dict] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[Any] = None,
        extra_body: Optional[dict] = None,
    ) -> dict:
        """POST `messages` to the chat endpoint. Returns the parsed JSON body.

        Retries on network errors, 429, and 5xx. Raises `LLMError` for
        non-retryable HTTP errors or after `max_retries` retries.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            body["response_format"] = response_format
        if tools is not None:
            body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
        if extra_body:
            body.update(extra_body)

        last_error: Optional[str] = None
        # when the endpoint rejects response_format we strip it
        # and retry once. Tracked separately so a single endpoint quirk
        # doesn't burn the transport-retry budget for transient failures.
        response_format_stripped = False
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                resp = requests.post(
                    self.url, headers=self._headers(), json=body, timeout=self.timeout,
                )
            except (requests.ConnectionError, requests.Timeout) as e:
                latency_ms = int((time.time() - t0) * 1000)
                last_error = f"{type(e).__name__}: {e}"
                self._log_usage(None, latency_ms, success=False, error=last_error)
                self._backoff(attempt)
                continue

            latency_ms = int((time.time() - t0) * 1000)
            status = resp.status_code
            if status == 200:
                try:
                    result = resp.json()
                except ValueError as e:
                    # 200 but body isn't JSON — that's a server bug, don't retry
                    self._log_usage(None, latency_ms, success=False,
                                    error=f"200 non-JSON body: {e}")
                    raise LLMError(f"HTTP 200 but non-JSON body: {resp.text[:300]}") from e
                self._log_usage(result, latency_ms, success=True)
                return result
            if status == 429 or 500 <= status < 600:
                last_error = f"HTTP {status}: {resp.text[:300]}"
                self._log_usage(None, latency_ms, success=False, error=last_error)
                self._backoff(attempt)
                continue
            # 400 with a response_format complaint → retry once without it.
            # The system prompts already say "respond with valid JSON only",
            # so most endpoints produce parseable JSON even outside json_object
            # mode (and our extract_json_object handles markdown fences).
            if (status == 400
                    and "response_format" in body
                    and not response_format_stripped):
                err_text = (resp.text or "").lower()
                if any(t in err_text for t in _RESPONSE_FORMAT_REJECT_TOKENS):
                    response_format_stripped = True
                    # Rebind to a fresh dict rather than mutate — the
                    # original ``body`` reference was captured by mock /
                    # logging machinery for the first call, and mutating
                    # it would retroactively rewrite that record.
                    body = {k: v for k, v in body.items()
                            if k != "response_format"}
                    self._log_usage(
                        None, latency_ms, success=False,
                        error=("HTTP 400 response_format unsupported, "
                               f"retrying without: {resp.text[:200]}"),
                    )
                    _log.warning(
                        "LLM endpoint rejected response_format; retrying "
                        "without (set tools.api.use_json_mode: false in "
                        "config to skip this round trip)."
                    )
                    # Don't sleep — this isn't a transient failure.
                    continue
            # non-retryable 4xx
            self._log_usage(None, latency_ms, success=False,
                            error=f"HTTP {status}: {resp.text[:300]}")
            raise LLMError(f"HTTP {status} (non-retryable): {resp.text[:300]}")

        raise LLMError(
            f"exhausted {self.max_retries} retries; last error: {last_error}"
        )

    # -- capability probes (run on server) ---------------------------------

    def test_connection(self) -> dict:
        """Simplest round trip — tells us whether key + url + model work."""
        messages = [
            {"role": "user", "content": "Respond with the single word: pong"},
        ]
        return self.call(messages, temperature=0.0)

    def test_json_mode(self) -> dict:
        """Probe whether `response_format={"type":"json_object"}` is honored.

        If the endpoint silently ignores the param, we just get free-form text
        — caller inspects `choices[0].message.content` and decides.
        """
        messages = [
            {"role": "system", "content": "You return JSON only. No prose."},
            {"role": "user", "content":
                'Return exactly this JSON object: {"status":"ok","value":42}'},
        ]
        return self.call(messages, temperature=0.0,
                         response_format={"type": "json_object"})

    def test_function_calling(self) -> dict:
        """Probe whether the endpoint accepts OpenAI-style `tools`."""
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a given city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            },
        }]
        messages = [
            {"role": "user",
             "content": "What's the weather in Beijing? Use the tool."},
        ]
        return self.call(messages, temperature=0.0, tools=tools, tool_choice="auto")

    # -- internals --------------------------------------------------------

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _backoff(self, attempt: int) -> None:
        delay = self.retry_base_delay * (2 ** attempt)
        time.sleep(delay)

    def _log_usage(
        self,
        result: Optional[dict],
        latency_ms: int,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "latency_ms": latency_ms,
            "success": success,
        }
        if success and result is not None:
            usage = result.get("usage") or {}
            entry["prompt_tokens"] = usage.get("prompt_tokens")
            entry["completion_tokens"] = usage.get("completion_tokens")
            entry["total_tokens"] = usage.get("total_tokens")
        if error is not None:
            entry["error"] = error[:500]
        try:
            self.usage_log.parent.mkdir(parents=True, exist_ok=True)
            with self.usage_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            # Logging must never break the caller.
            pass


# ---------- convenience helpers --------------------------------------------


def extract_content(response: dict) -> str:
    """Pull the assistant text out of an OpenAI-compatible response."""
    choices = response.get("choices") or []
    if not choices:
        raise LLMError("response has no choices")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        raise LLMError("response message has no content")
    return content


def extract_tool_calls(response: dict) -> list:
    """Pull `tool_calls` out (empty list if endpoint doesn't support tools)."""
    choices = response.get("choices") or []
    if not choices:
        return []
    msg = choices[0].get("message") or {}
    return msg.get("tool_calls") or []


# ---------- shared JSON extraction -----------------------------


# Outermost ``{...}`` block — DOTALL + greedy so multi-line objects with
# newlines and nested braces still match. Used as the last-resort layer.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# Markdown code fence with optional ``json`` language tag. when
# the endpoint refuses ``response_format=json_object`` we drop into plain
# text mode where the model commonly wraps JSON in fences. Match each
# fence body and try them in order so prose-then-fenced-JSON outputs are
# parsed before the greedy ``{...}`` regex could overshoot.
_FENCED_JSON_RE = re.compile(
    r"```(?:json)?[ \t]*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE,
)


def extract_json_object(text: str) -> Optional[dict]:
    """Best-effort JSON-object extraction from LLM output.

    Layers, first to succeed wins:

      1. ``json.loads(text)`` — clean json_object-mode response.
      2. Markdown code-fence body (`````json ... ````` or unlabelled).
         Tried before the greedy regex because a fence is a strong signal
         that the JSON the model intended to emit is *exactly* what's
         inside the fence — even when there's prose around it.
      3. Outermost ``{...}`` block — catches the case where the model
         emits a short preamble or trailing comment without any fence.

    Returns ``None`` if no layer parses cleanly. Non-dict roots (lists,
    strings, numbers) are also rejected — callers expect an object.
    """
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except ValueError:
        pass
    for m in _FENCED_JSON_RE.finditer(text):
        body = m.group(1).strip()
        if not body:
            continue
        try:
            obj = json.loads(body)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except ValueError:
        return None
