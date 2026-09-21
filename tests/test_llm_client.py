"""Mock tests for `llm_client.py` — runs locally, does NOT hit the real API.

Run:
    cd riboseer
    python -m pytest tests/test_llm_client.py -v
    # or without pytest:
    python tests/test_llm_client.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import requests

# Make src/ importable when running standalone (`python tests/test_llm_client.py`)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step2_target_char.llm_client import (  # noqa: E402
    LLMClient, LLMError, extract_content, extract_json_object,
    extract_tool_calls,
)


def _ok_response(content: str = "pong", usage: dict | None = None) -> MagicMock:
    """Build a MagicMock that quacks like a successful requests.Response."""
    r = MagicMock()
    r.status_code = 200
    body = {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    r.json.return_value = body
    r.text = json.dumps(body)
    return r


def _err_response(status: int, text: str = "server error") -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.text = text
    r.json.side_effect = ValueError("not json")
    return r


class TestLLMClientInit(unittest.TestCase):
    def test_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LLMError):
                LLMClient()

    def test_accepts_explicit_key(self):
        c = LLMClient(api_key="test-key", usage_log=Path(tempfile.mktemp()))
        self.assertEqual(c.api_key, "test-key")

    def test_reads_env_var(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "env-key"}):
            c = LLMClient(usage_log=Path(tempfile.mktemp()))
            self.assertEqual(c.api_key, "env-key")


class TestLLMClientCall(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.log = self.tmp / "usage.jsonl"
        self.client = LLMClient(
            api_key="k", usage_log=self.log,
            max_retries=3, retry_base_delay=0.0,  # zero delay for test speed
        )

    def _read_log(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines() if line]

    @patch("step2_target_char.llm_client.requests.post")
    def test_successful_call_returns_parsed_body(self, mock_post):
        mock_post.return_value = _ok_response("hello")
        result = self.client.call([{"role": "user", "content": "hi"}])
        self.assertEqual(extract_content(result), "hello")
        mock_post.assert_called_once()

    @patch("step2_target_char.llm_client.requests.post")
    def test_records_token_usage_on_success(self, mock_post):
        mock_post.return_value = _ok_response(
            "x", usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        self.client.call([{"role": "user", "content": "hi"}])
        entries = self._read_log()
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertTrue(e["success"])
        self.assertEqual(e["prompt_tokens"], 100)
        self.assertEqual(e["completion_tokens"], 50)
        self.assertEqual(e["total_tokens"], 150)

    @patch("step2_target_char.llm_client.requests.post")
    def test_retries_on_connection_error(self, mock_post):
        # First two calls raise ConnectionError, third succeeds
        mock_post.side_effect = [
            requests.ConnectionError("net down"),
            requests.ConnectionError("net down"),
            _ok_response("ok after retry"),
        ]
        result = self.client.call([{"role": "user", "content": "hi"}])
        self.assertEqual(extract_content(result), "ok after retry")
        self.assertEqual(mock_post.call_count, 3)
        # Failure log entries + 1 success
        entries = self._read_log()
        self.assertEqual(len(entries), 3)
        self.assertFalse(entries[0]["success"])
        self.assertFalse(entries[1]["success"])
        self.assertTrue(entries[2]["success"])

    @patch("step2_target_char.llm_client.requests.post")
    def test_retries_on_timeout(self, mock_post):
        mock_post.side_effect = [
            requests.Timeout("slow"),
            _ok_response("recovered"),
        ]
        result = self.client.call([{"role": "user", "content": "hi"}])
        self.assertEqual(extract_content(result), "recovered")

    @patch("step2_target_char.llm_client.requests.post")
    def test_retries_on_429(self, mock_post):
        mock_post.side_effect = [_err_response(429, "rate limited"), _ok_response("ok")]
        result = self.client.call([{"role": "user", "content": "hi"}])
        self.assertEqual(extract_content(result), "ok")
        self.assertEqual(mock_post.call_count, 2)

    @patch("step2_target_char.llm_client.requests.post")
    def test_retries_on_5xx(self, mock_post):
        mock_post.side_effect = [_err_response(503, "bad gateway"), _ok_response("ok")]
        result = self.client.call([{"role": "user", "content": "hi"}])
        self.assertEqual(extract_content(result), "ok")

    @patch("step2_target_char.llm_client.requests.post")
    def test_does_not_retry_on_400(self, mock_post):
        mock_post.return_value = _err_response(400, "bad request")
        with self.assertRaises(LLMError):
            self.client.call([{"role": "user", "content": "hi"}])
        self.assertEqual(mock_post.call_count, 1)

    @patch("step2_target_char.llm_client.requests.post")
    def test_does_not_retry_on_401(self, mock_post):
        mock_post.return_value = _err_response(401, "unauthorized")
        with self.assertRaises(LLMError):
            self.client.call([{"role": "user", "content": "hi"}])
        self.assertEqual(mock_post.call_count, 1)

    @patch("step2_target_char.llm_client.requests.post")
    def test_exhausts_retries(self, mock_post):
        mock_post.side_effect = [requests.ConnectionError("x")] * 3
        with self.assertRaises(LLMError) as ctx:
            self.client.call([{"role": "user", "content": "hi"}])
        self.assertIn("exhausted", str(ctx.exception))
        self.assertEqual(mock_post.call_count, 3)

    @patch("step2_target_char.llm_client.requests.post")
    def test_sends_temperature_and_format(self, mock_post):
        mock_post.return_value = _ok_response("{}")
        self.client.call(
            [{"role": "user", "content": "hi"}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        call_kwargs = mock_post.call_args.kwargs
        body = call_kwargs["json"]
        self.assertEqual(body["temperature"], 0.3)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["model"], "llm-model")

    @patch("step2_target_char.llm_client.requests.post")
    def test_authorization_header_uses_bearer_prefix(self, mock_post):
        mock_post.return_value = _ok_response("x")
        self.client.call([{"role": "user", "content": "hi"}])
        headers = mock_post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer k")


class TestHelpers(unittest.TestCase):
    def test_extract_content_ok(self):
        resp = {"choices": [{"message": {"content": "hi"}}]}
        self.assertEqual(extract_content(resp), "hi")

    def test_extract_content_no_choices(self):
        with self.assertRaises(LLMError):
            extract_content({"choices": []})

    def test_extract_tool_calls_empty(self):
        self.assertEqual(extract_tool_calls({"choices": []}), [])
        self.assertEqual(
            extract_tool_calls({"choices": [{"message": {"content": "x"}}]}), [],
        )

    def test_extract_tool_calls_present(self):
        tc = [{"id": "t1", "function": {"name": "f", "arguments": "{}"}}]
        resp = {"choices": [{"message": {"tool_calls": tc}}]}
        self.assertEqual(extract_tool_calls(resp), tc)


# ---------- response_format 400 fallback -----------------------


def _err_response_with_text(status: int, text: str) -> MagicMock:
    """Like ``_err_response`` but with a controllable body text."""
    r = MagicMock()
    r.status_code = status
    r.text = text
    r.json.side_effect = ValueError("not json")
    return r


class TestResponseFormatFallback(unittest.TestCase):
    """When the endpoint rejects ``response_format=json_object`` with HTTP
    400, ``LLMClient.call`` strips the param and retries once. Subsequent
    calls in the same invocation must NOT re-add it."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.client = LLMClient(
            api_key="k", usage_log=self.tmp / "usage.jsonl",
            max_retries=3, retry_base_delay=0.0,
        )

    @patch("step2_target_char.llm_client.requests.post")
    def test_strips_response_format_on_400_and_retries(self, mock_post):
        reject = _err_response_with_text(
            400,
            'The parameter response_format.type specified in the request '
            'are not valid: json_object is not supported',
        )
        mock_post.side_effect = [reject, _ok_response("after retry")]
        result = self.client.call(
            [{"role": "user", "content": "hi"}],
            response_format={"type": "json_object"},
        )
        self.assertEqual(extract_content(result), "after retry")
        self.assertEqual(mock_post.call_count, 2)
        # First call: response_format present.
        first_body = mock_post.call_args_list[0].kwargs["json"]
        self.assertIn("response_format", first_body)
        # Second call: stripped.
        second_body = mock_post.call_args_list[1].kwargs["json"]
        self.assertNotIn("response_format", second_body)

    @patch("step2_target_char.llm_client.requests.post")
    def test_strips_only_once_then_propagates(self, mock_post):
        # Endpoint persistently returns 400 even without response_format
        # (unlikely in practice, but the strip path must not loop forever).
        reject = _err_response_with_text(
            400,
            'response_format json_object is not supported',
        )
        # After stripping we still get a 400 (the second one has nothing
        # to do with response_format anymore — could be any 400). The
        # client must surface it as LLMError, not strip again.
        second_400 = _err_response_with_text(400, "some other 400")
        mock_post.side_effect = [reject, second_400]
        with self.assertRaises(LLMError):
            self.client.call(
                [{"role": "user", "content": "hi"}],
                response_format={"type": "json_object"},
            )
        self.assertEqual(mock_post.call_count, 2)

    @patch("step2_target_char.llm_client.requests.post")
    def test_does_not_strip_on_unrelated_400(self, mock_post):
        # 400 without response_format / json_object in the body is a real
        # bad-request and must be raised immediately.
        mock_post.return_value = _err_response_with_text(
            400, "model not found",
        )
        with self.assertRaises(LLMError):
            self.client.call(
                [{"role": "user", "content": "hi"}],
                response_format={"type": "json_object"},
            )
        self.assertEqual(mock_post.call_count, 1)

    @patch("step2_target_char.llm_client.requests.post")
    def test_does_not_strip_when_response_format_not_set(self, mock_post):
        # If the caller didn't send response_format, a 400 is just a 400.
        mock_post.return_value = _err_response_with_text(
            400, "json_object error",  # body still mentions the token
        )
        with self.assertRaises(LLMError):
            self.client.call([{"role": "user", "content": "hi"}])
        self.assertEqual(mock_post.call_count, 1)


# ---------- shared JSON extractor ------------------------------


class TestExtractJsonObject(unittest.TestCase):
    """Shared JSON extractor used by step 2/3/5. Verifies the fast path,
    markdown-fence handling, and the greedy regex fallback."""

    def test_clean_json(self):
        self.assertEqual(extract_json_object('{"a": 1}'), {"a": 1})

    def test_returns_none_for_empty(self):
        self.assertIsNone(extract_json_object(""))
        self.assertIsNone(extract_json_object("   "))
        self.assertIsNone(extract_json_object("not json at all"))

    def test_rejects_non_dict_root(self):
        # Lists, strings, numbers — callers expect {...}, not just "JSON".
        self.assertIsNone(extract_json_object("[1, 2, 3]"))
        self.assertIsNone(extract_json_object('"a string"'))
        self.assertIsNone(extract_json_object("42"))

    def test_fenced_with_json_lang(self):
        text = 'Here is the result:\n```json\n{"a": 1, "b": 2}\n```\nHope that helps!'
        self.assertEqual(extract_json_object(text), {"a": 1, "b": 2})

    def test_fenced_without_lang_tag(self):
        text = '```\n{"x": "y"}\n```'
        self.assertEqual(extract_json_object(text), {"x": "y"})

    def test_fenced_with_nested_object(self):
        text = '```json\n{"outer": {"inner": [1, 2]}}\n```'
        self.assertEqual(
            extract_json_object(text),
            {"outer": {"inner": [1, 2]}},
        )

    def test_fenced_skips_non_json_fence(self):
        # First fence is a code snippet, second is the actual JSON.
        text = (
            "```python\nprint('hi')\n```\n"
            "Now the answer:\n"
            '```json\n{"answer": 42}\n```'
        )
        self.assertEqual(extract_json_object(text), {"answer": 42})

    def test_unfenced_with_preamble(self):
        # Last-resort regex: find the outermost {...}.
        text = "I think the result is: {\"k\": \"v\"} — let me know."
        self.assertEqual(extract_json_object(text), {"k": "v"})

    def test_invalid_json_inside_fence_falls_back_to_regex(self):
        # A broken fence body should not block the regex fallback from
        # finding the real {...} elsewhere.
        text = (
            '```json\nnot really json\n```\n'
            'Actual answer: {"ok": true}'
        )
        self.assertEqual(extract_json_object(text), {"ok": True})


if __name__ == "__main__":
    unittest.main(verbosity=2)
