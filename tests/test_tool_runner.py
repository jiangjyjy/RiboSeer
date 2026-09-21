"""Unit tests for step4_tool_adapters.tool_runner.

Uses real subprocess calls but only against universally-available
shell builtins / python so the tests run on Windows + Linux + macOS.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step4_tool_adapters.tool_runner import (  # noqa: E402
    ToolRunResult, run_command, run_in_conda_env,
)


class TestRunCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.log_dir = Path(self.tmp) / "logs"

    def test_python_version(self):
        res = run_command(
            "python --version",
            timeout=15, log_dir=self.log_dir, log_tag="ver",
        )
        self.assertIsInstance(res, ToolRunResult)
        self.assertTrue(res.success, msg=res.stderr)
        # CPython prints "Python X.Y.Z"; old versions printed to stderr —
        # tolerate both streams.
        combined = (res.stdout + res.stderr).lower()
        self.assertIn("python", combined)

    def test_log_file_written(self):
        res = run_command(
            "python -c \"print('hi')\"",
            timeout=10, log_dir=self.log_dir, log_tag="hello",
        )
        self.assertTrue(res.success)
        self.assertIsNotNone(res.log_path)
        log_p = Path(res.log_path)
        self.assertTrue(log_p.is_file())
        text = log_p.read_text(encoding="utf-8")
        self.assertIn("# rc:", text)
        self.assertIn("hi", text)

    def test_nonzero_exit_not_raised(self):
        res = run_command(
            "python -c \"import sys; sys.exit(7)\"",
            timeout=10, log_dir=self.log_dir, log_tag="rc7",
        )
        self.assertFalse(res.success)
        self.assertEqual(res.returncode, 7)

    def test_timeout_caught(self):
        # python sleep — works on every platform unlike `sleep` shell cmd
        res = run_command(
            "python -c \"import time; time.sleep(5)\"",
            timeout=1, log_dir=self.log_dir, log_tag="sleep",
        )
        self.assertTrue(res.timed_out)
        self.assertFalse(res.success)
        self.assertEqual(res.returncode, 124)

    def test_runtime_seconds_recorded(self):
        res = run_command(
            "python --version",
            timeout=10, log_dir=self.log_dir, log_tag="rt",
        )
        self.assertGreater(res.runtime_seconds, 0.0)
        self.assertLess(res.runtime_seconds, 30.0)

    def test_extra_env_propagated(self):
        res = run_command(
            "python -c \"import os; print(os.environ.get('POCKET_TEST', 'missing'))\"",
            timeout=10, log_dir=self.log_dir, log_tag="env",
            env={"POCKET_TEST": "ok"},
        )
        self.assertTrue(res.success)
        self.assertIn("ok", res.stdout)


class TestRunInCondaEnv(unittest.TestCase):
    def test_empty_env_rejected(self):
        with self.assertRaises(ValueError):
            run_in_conda_env("", "python --version")

    def test_command_wrapping(self):
        """Verify run_in_conda_env wraps the command without actually
        spawning conda (which may not be on PATH in CI)."""
        captured = {}

        def fake_run_command(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return ToolRunResult(
                command=cmd, cwd=None, returncode=0,
                stdout="", stderr="", runtime_seconds=0.0,
                log_path=None,
            )

        with patch(
            "step4_tool_adapters.tool_runner.run_command",
            side_effect=fake_run_command,
        ):
            run_in_conda_env(
                "boltz",
                "boltz predict input.yaml --out_dir out/",
                timeout=42,
                log_tag="b2",
            )

        self.assertIn("conda run -n", captured["cmd"])
        self.assertIn("boltz", captured["cmd"])
        self.assertIn("--no-capture-output", captured["cmd"])
        self.assertIn("boltz predict input.yaml", captured["cmd"])
        self.assertEqual(captured["kwargs"]["timeout"], 42)
        self.assertEqual(captured["kwargs"]["log_tag"], "b2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
