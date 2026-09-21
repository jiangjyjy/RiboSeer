"""Subprocess wrappers for invoking external tools.

Two helpers:
  - ``run_command``: plain subprocess call (used by P2Rank, which is
    a Java executable and needs no environment switch).
  - ``run_in_conda_env``: prepend ``conda run -n <env> --no-capture-output``
    so the tool runs inside its dedicated conda environment.

Both return a ``ToolRunResult`` carrying stdout/stderr/exit code/runtime
plus the path of an on-disk log (so we can debug long-running server
runs without keeping huge strings in memory).

Behaviour
---------
- never raises on tool failure; ``ToolRunResult.success`` reflects exit code
- ``TimeoutExpired`` is caught and surfaced via ``timed_out=True``
- log files live under ``log_dir`` (defaults to ``./logs/step4/``)
"""
from __future__ import annotations

import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DEFAULT_LOG_DIR = Path("logs") / "step4"


@dataclass
class ToolRunResult:
    """Result of one subprocess invocation."""
    command: str
    cwd: Optional[str]
    returncode: int
    stdout: str
    stderr: str
    runtime_seconds: float
    timed_out: bool = False
    log_path: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out


# ---------- internal helpers ----------------------------------------------


def _ensure_log_dir(log_dir: Optional[Path]) -> Path:
    target = log_dir or DEFAULT_LOG_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_log(
    log_dir: Path,
    tag: str,
    command: str,
    cwd: Optional[str],
    stdout: str,
    stderr: str,
    returncode: int,
    runtime_seconds: float,
    timed_out: bool,
) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    safe_tag = "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)[:60]
    log_path = log_dir / f"{ts}_{safe_tag}.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# command: {command}\n")
        f.write(f"# cwd:     {cwd}\n")
        f.write(f"# rc:      {returncode}\n")
        f.write(f"# elapsed: {runtime_seconds:.2f}s\n")
        f.write(f"# timed_out: {timed_out}\n")
        f.write("\n--- stdout ---\n")
        f.write(stdout or "")
        f.write("\n--- stderr ---\n")
        f.write(stderr or "")
    return log_path


# ---------- public API -----------------------------------------------------


def run_command(
    command: str,
    *,
    cwd: Optional[str | Path] = None,
    timeout: int = 600,
    log_dir: Optional[Path] = None,
    log_tag: str = "cmd",
    env: Optional[dict] = None,
) -> ToolRunResult:
    """Run ``command`` in a shell.

    Parameters
    ----------
    command : str
        Full command line. Pass already-quoted paths if needed.
    cwd : str | Path, optional
        Working directory for the child process.
    timeout : int
        Hard timeout in seconds; if exceeded the call returns
        ``timed_out=True`` instead of raising.
    log_dir : Path, optional
        Where to write a per-invocation log file.
    log_tag : str
        Short tag prepended to the log filename.
    env : dict, optional
        Extra env vars (merged on top of os.environ).
    """
    cwd_str = str(cwd) if cwd is not None else None
    log_dir_p = _ensure_log_dir(log_dir)

    full_env = None
    if env:
        import os
        full_env = {**os.environ, **{k: str(v) for k, v in env.items()}}

    start = time.monotonic()
    timed_out = False
    stdout = ""
    stderr = ""
    returncode = -1
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd_str,
            timeout=timeout,
            env=full_env,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        returncode = proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout = (e.stdout.decode() if isinstance(e.stdout, bytes)
                  else (e.stdout or ""))
        stderr = (e.stderr.decode() if isinstance(e.stderr, bytes)
                  else (e.stderr or ""))
        stderr = (stderr + f"\n[TIMEOUT after {timeout}s]").strip()
        returncode = 124  # GNU `timeout` convention
    except Exception as e:  # noqa: BLE001 — must never raise
        stderr = f"[run_command exception] {type(e).__name__}: {e}"
        returncode = -1
    elapsed = time.monotonic() - start

    log_path = _write_log(
        log_dir_p, log_tag, command, cwd_str,
        stdout, stderr, returncode, elapsed, timed_out,
    )

    return ToolRunResult(
        command=command,
        cwd=cwd_str,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        runtime_seconds=elapsed,
        timed_out=timed_out,
        log_path=str(log_path),
    )


def run_in_conda_env(
    env_name: str,
    command: str,
    *,
    cwd: Optional[str | Path] = None,
    timeout: int = 3600,
    log_dir: Optional[Path] = None,
    log_tag: Optional[str] = None,
    extra_env: Optional[dict] = None,
) -> ToolRunResult:
    """Run ``command`` inside conda env ``env_name``.

    Wraps the call in ``conda run -n <env> --no-capture-output``. The
    ``--no-capture-output`` flag is critical: without it conda buffers
    stdout/stderr and Python's ``capture_output=True`` gets nothing
    until the child exits, which can hide long-running progress (and
    confuses some tools that detect a TTY).
    """
    if not env_name:
        raise ValueError("env_name must be a non-empty string")
    full_cmd = f"conda run -n {shlex.quote(env_name)} --no-capture-output {command}"
    return run_command(
        full_cmd,
        cwd=cwd,
        timeout=timeout,
        log_dir=log_dir,
        log_tag=log_tag or f"conda_{env_name}",
        env=extra_env,
    )


# ---------- demo -----------------------------------------------------------


def _demo_main() -> None:
    """Trivial smoke that just runs `python --version` to prove the path."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    res = run_command("python --version", timeout=10, log_tag="version")
    print(f"rc={res.returncode}  elapsed={res.runtime_seconds:.3f}s  "
          f"stdout={res.stdout.strip()!r}  log={res.log_path}")


if __name__ == "__main__":
    _demo_main()
