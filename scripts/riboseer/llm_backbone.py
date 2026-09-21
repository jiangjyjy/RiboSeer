"""Generic LLM backbone client for paper Table 13 (backbone comparison).

The SCOPE / MAESTRO / POLISH generators were written against
``step2_target_char.llm_client.LLMClient`` — an OpenAI-compatible
``POST {url}`` wrapper whose ``.call(messages, temperature=...)`` returns
the parsed JSON body and whose ``extract_content`` reads
``choices[0].message.content``.

Table 13 swaps the LLM backbone for three third-party proxies (Gemini /
DeepSeek / Qwen). ``BackboneClient`` is a **drop-in** for ``LLMClient``:
same ``.call(...)`` signature, same OpenAI-shaped return value, same
``LLMError`` on failure — so the generators need only choose which client
to build, nothing else changes downstream.

What it adds over ``LLMClient``
-------------------------------
* **Endpoint discovery** — most proxies expose ``/chat/completions``;
  some only ``/v1/chat/completions``. We try ``{base}/chat/completions``
  first and fall back to ``{base}/v1/chat/completions`` on a 404, caching
  the winner so discovery costs at most one extra round trip.
* **Anthropic fallback** — when ``allow_anthropic_fallback`` is set (Qwen's
  llm-relay-3.com may speak the ``/v1/messages`` Messages API instead of
  OpenAI chat), we additionally try ``{base}/messages`` /
  ``{base}/v1/messages`` with Anthropic headers + body and normalise the
  ``content[].text`` blocks back into the OpenAI shape the callers expect.
* **thinking disable** — Anthropic-style ``"thinking": {"type":
  "disabled"}`` is added on the Messages transport when
  ``thinking_disabled`` is set (keeps reasoning models from blowing the
  token budget / wrapping the JSON in a reasoning preamble).

Keys are never hard-coded here — the caller passes ``api_key`` (the driver
reads it from a per-backbone env var).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from step2_target_char.llm_client import LLMError  # noqa: E402

DEFAULT_MAX_TOKENS = 4096
DEFAULT_USAGE_LOG = Path("logs/backbone_usage.jsonl")


# ---------------------------------------------------------------------------
# Transports: each knows how to talk one wire format and normalise the reply
# ---------------------------------------------------------------------------


class _Transport:
    """One (wire-format, URL) pair. ``headers``/``body`` build a request and
    ``content`` pulls the assistant text out of a 200 body."""

    kind = "base"

    def __init__(self, url: str) -> None:
        self.url = url

    def headers(self, api_key: str) -> dict:  # pragma: no cover - trivial
        raise NotImplementedError

    def body(self, model: str, messages: list[dict], temperature: float,
             *, max_tokens: int, thinking_disabled: bool,
             response_format: Optional[dict]) -> dict:  # pragma: no cover
        raise NotImplementedError

    def content(self, resp_json: dict) -> str:  # pragma: no cover - trivial
        raise NotImplementedError


class _OpenAITransport(_Transport):
    kind = "openai"

    def headers(self, api_key: str) -> dict:
        return {"Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"}

    def body(self, model, messages, temperature, *, max_tokens,
             thinking_disabled, response_format) -> dict:
        b: dict[str, Any] = {"model": model, "messages": messages,
                             "temperature": temperature}
        if response_format is not None:
            b["response_format"] = response_format
        return b

    def content(self, resp_json: dict) -> str:
        choices = resp_json.get("choices") or []
        if not choices:
            raise LLMError("OpenAI response has no choices")
        msg = choices[0].get("message") or {}
        text = msg.get("content")
        if text is None:
            raise LLMError("OpenAI response message has no content")
        return text


class _AnthropicTransport(_Transport):
    kind = "anthropic"

    def headers(self, api_key: str) -> dict:
        return {"x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"}

    def body(self, model, messages, temperature, *, max_tokens,
             thinking_disabled, response_format) -> dict:
        # Anthropic splits system out of the message list and requires
        # max_tokens; temperature is bounded to [0, 1].
        system_parts = [m["content"] for m in messages
                        if m.get("role") == "system"]
        chat = [m for m in messages if m.get("role") != "system"]
        b: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": chat,
            "temperature": min(max(float(temperature), 0.0), 1.0),
        }
        if system_parts:
            b["system"] = "\n\n".join(system_parts)
        if thinking_disabled:
            b["thinking"] = {"type": "disabled"}
        return b

    def content(self, resp_json: dict) -> str:
        blocks = resp_json.get("content") or []
        texts = [blk.get("text", "") for blk in blocks
                 if isinstance(blk, dict) and blk.get("type") == "text"]
        if not texts:
            raise LLMError("Anthropic response has no text block")
        return "".join(texts)


def _join(base: str, suffix: str) -> str:
    return base.rstrip("/") + suffix


def _candidate_transports(base_url: str,
                          allow_anthropic_fallback: bool) -> list[_Transport]:
    """Ordered transports to probe. OpenAI ``/chat/completions`` first, its
    ``/v1`` variant next (skipped if ``base`` already ends in ``/v1``), then
    the Anthropic Messages endpoints when fallback is enabled. Duplicate
    URLs are dropped so an already-``/v1`` base doesn't probe itself twice."""
    base = base_url.rstrip("/")
    out: list[_Transport] = [_OpenAITransport(_join(base, "/chat/completions"))]
    if not base.endswith("/v1"):
        out.append(_OpenAITransport(_join(base, "/v1/chat/completions")))
    if allow_anthropic_fallback:
        out.append(_AnthropicTransport(_join(base, "/messages")))
        if not base.endswith("/v1"):
            out.append(_AnthropicTransport(_join(base, "/v1/messages")))
    seen: set[str] = set()
    uniq: list[_Transport] = []
    for t in out:
        if t.url not in seen:
            seen.add(t.url)
            uniq.append(t)
    return uniq


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class BackboneClient:
    """OpenAI-compatible (with Anthropic fallback) drop-in for LLMClient."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        *,
        allow_anthropic_fallback: bool = False,
        thinking_disabled: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 4,
        retry_base_delay: float = 2.0,
        timeout_seconds: float = 180.0,
        usage_log: Path = DEFAULT_USAGE_LOG,
    ) -> None:
        if not api_key:
            raise LLMError(
                "backbone api_key not set — pass --api-key or set the "
                "backbone's API-key env var (don't hard-code keys)")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._candidates = _candidate_transports(
            base_url, allow_anthropic_fallback)
        self.thinking_disabled = thinking_disabled
        self.max_tokens = int(max_tokens)
        self.max_retries = int(max_retries)
        self.retry_base_delay = float(retry_base_delay)
        self.timeout = float(timeout_seconds)
        self.usage_log = Path(usage_log)
        # The transport discovered to work, cached after the first success.
        self._active: Optional[_Transport] = None

    # -- public API (LLMClient parity) ------------------------------------

    def call(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        response_format: Optional[dict] = None,
        tools: Optional[list] = None,        # accepted for parity; unused
        tool_choice: Optional[Any] = None,   # accepted for parity; unused
        extra_body: Optional[dict] = None,
    ) -> dict:
        """POST ``messages`` and return an OpenAI-shaped
        ``{"choices":[{"message":{"content": <text>}}], "usage": ...}``.

        Discovers the working transport on the first call (cached after),
        retries transient failures (network / 429 / 5xx) per transport, and
        falls through to the next candidate endpoint on a 404. Raises
        ``LLMError`` when every candidate is exhausted."""
        transports = ([self._active] if self._active is not None
                      else list(self._candidates))
        last_error = "no candidate endpoints"
        for transport in transports:
            ok, result, err, fatal = self._try_transport(
                transport, messages, temperature, response_format, extra_body)
            if ok:
                self._active = transport
                return result
            last_error = err or last_error
            if fatal:
                # auth / explicit model error — trying other paths won't help
                raise LLMError(last_error)
            # else: 404 / persistent transient → move to next candidate
        raise LLMError(f"all backbone endpoints failed; last error: "
                       f"{last_error}")

    # -- internals --------------------------------------------------------

    def _try_transport(
        self, transport: _Transport, messages: list[dict],
        temperature: float, response_format: Optional[dict],
        extra_body: Optional[dict],
    ) -> tuple[bool, Optional[dict], Optional[str], bool]:
        """Returns ``(ok, normalised_result, error, fatal)``. ``fatal`` =
        raise immediately (auth/non-retryable, non-404); otherwise the caller
        advances to the next candidate transport."""
        body = transport.body(
            self.model, messages, temperature,
            max_tokens=self.max_tokens,
            thinking_disabled=self.thinking_disabled,
            response_format=response_format)
        if extra_body:
            body.update(extra_body)
        headers = transport.headers(self.api_key)

        last_error: Optional[str] = None
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                resp = requests.post(transport.url, headers=headers,
                                     json=body, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = f"{type(e).__name__}: {e}"
                self._log(transport, None, t0, False, last_error)
                self._backoff(attempt)
                continue

            status = resp.status_code
            if status == 200:
                try:
                    rj = resp.json()
                    content = transport.content(rj)
                except (ValueError, LLMError) as e:
                    # 200 but unparseable / wrong shape → this transport is
                    # the wrong wire format; let caller try the next one.
                    last_error = (f"{transport.kind} 200 unparseable "
                                  f"({e}): {resp.text[:200]}")
                    self._log(transport, None, t0, False, last_error)
                    return False, None, last_error, False
                self._log(transport, rj, t0, True, None)
                return (True, {"choices": [{"message": {"content": content}}],
                               "usage": rj.get("usage") or {}}, None, False)
            if status == 404:
                last_error = f"HTTP 404 at {transport.url}"
                self._log(transport, None, t0, False, last_error)
                return False, None, last_error, False     # try next endpoint
            if status in (401, 403):
                last_error = f"HTTP {status} (auth): {resp.text[:200]}"
                self._log(transport, None, t0, False, last_error)
                return False, None, last_error, True       # fatal
            if status == 429 or 500 <= status < 600:
                last_error = f"HTTP {status}: {resp.text[:200]}"
                self._log(transport, None, t0, False, last_error)
                self._backoff(attempt)
                continue
            # Other 4xx (e.g. 400 wrong body for this wire format): not
            # retryable on this transport, but a different transport might
            # accept it → advance rather than abort.
            last_error = f"HTTP {status}: {resp.text[:200]}"
            self._log(transport, None, t0, False, last_error)
            return False, None, last_error, False

        # retries exhausted on transient errors → let caller try next endpoint
        return False, None, (last_error or "retries exhausted"), False

    def _backoff(self, attempt: int) -> None:
        time.sleep(self.retry_base_delay * (2 ** attempt))

    def _log(self, transport: _Transport, result: Optional[dict],
             t0: float, success: bool, error: Optional[str]) -> None:
        entry: dict[str, Any] = {
            "model": self.model,
            "endpoint": transport.url,
            "wire": transport.kind,
            "latency_ms": int((time.time() - t0) * 1000),
            "success": success,
        }
        if success and result is not None:
            entry["usage"] = result.get("usage")
        if error is not None:
            entry["error"] = error[:500]
        try:
            self.usage_log.parent.mkdir(parents=True, exist_ok=True)
            with self.usage_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Factory used by the generators
# ---------------------------------------------------------------------------


def make_backbone_client(
    api_base: str, api_key: Optional[str], model: str, *,
    allow_anthropic_fallback: bool = False,
    thinking_disabled: bool = False,
) -> BackboneClient:
    """Build a ``BackboneClient`` from the generator CLI args. ``api_base``
    presence is what makes a generator pick this over the LLM client."""
    return BackboneClient(
        api_key=api_key or "", model=model, base_url=api_base,
        allow_anthropic_fallback=allow_anthropic_fallback,
        thinking_disabled=thinking_disabled)


# ---------------------------------------------------------------------------
# Shared CLI plumbing for the SCOPE / MAESTRO / POLISH generators
# ---------------------------------------------------------------------------


def add_backbone_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--api-base/--api-key/--model`` (+ wire-format toggles) to a
    generator. When ``--api-base`` is given the generator builds a
    ``BackboneClient`` for that endpoint instead of the default LLM client;
    ``--model`` then names the backbone model id."""
    g = parser.add_argument_group("LLM backbone (Table 13 — overrides LLM)")
    g.add_argument("--api-base", type=str, default=None,
                   help="backbone API base URL; presence switches the "
                        "generator off the default LLM client")
    g.add_argument("--api-key", type=str, default=None,
                   help="backbone API key (prefer an env var over a literal)")
    g.add_argument("--model", type=str, default=None,
                   help="backbone model id (used with --api-base)")
    g.add_argument("--anthropic-fallback", action="store_true",
                   help="also probe the Anthropic /messages endpoint")
    g.add_argument("--thinking-disabled", action="store_true",
                   help="send Anthropic-style thinking:{type:disabled}")


def backbone_from_args(args: argparse.Namespace) -> Optional[dict]:
    """Backbone-config dict from parsed args, or None when ``--api-base`` was
    not supplied (→ generator keeps its default LLM client)."""
    if not getattr(args, "api_base", None):
        return None
    return {
        "api_base": args.api_base,
        "api_key": args.api_key,
        "model": args.model,
        "allow_anthropic_fallback": bool(
            getattr(args, "anthropic_fallback", False)),
        "thinking_disabled": bool(getattr(args, "thinking_disabled", False)),
    }


def client_from_backbone(backbone: Optional[dict]) -> Optional[BackboneClient]:
    """Build the client for a backbone-config dict (``backbone_from_args``),
    or None when there's no backbone (caller falls back to the LLM client)."""
    if not backbone:
        return None
    return make_backbone_client(
        backbone["api_base"], backbone.get("api_key"), backbone["model"],
        allow_anthropic_fallback=bool(
            backbone.get("allow_anthropic_fallback")),
        thinking_disabled=bool(backbone.get("thinking_disabled")))
