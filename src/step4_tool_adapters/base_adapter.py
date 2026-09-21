"""Abstract base class for tool adapters.

Concrete adapters override ``prepare_input``, ``run_tool``, and
``parse_output``. The full pipeline is wrapped by ``predict``, which
catches every exception and returns a ``success=False`` ToolPrediction
on failure (failure must never propagate — step 4 needs to keep going
across the rest of the batch).
"""
from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from .schemas import ToolPrediction, make_failure_prediction


class BaseAdapter(ABC):
    """Common interface for every step-4 adapter.

    Subclasses must set ``tool_id`` and ``category`` as class attributes
    (matching the values in ``step3_tool_selection.tool_registry``).
    """

    tool_id: str = ""
    category: str = ""  # "A" | "B" | "C" | "D"

    # ------------------------------------------------------------------ abstract

    @abstractmethod
    def prepare_input(
        self,
        sample_json: dict,
        work_dir: Path,
        config: dict,
    ) -> dict:
        """Materialise the tool's input files under ``work_dir``.

        Returns a dict of named paths (e.g. ``{"pdb": Path(...),
        "fasta": Path(...)}``) that ``run_tool`` will consume. Subclasses
        may raise on missing required fields — ``predict`` catches it.
        """

    @abstractmethod
    def run_tool(
        self,
        input_paths: dict,
        work_dir: Path,
        config: dict,
    ) -> Path:
        """Invoke the tool. Returns the directory holding raw outputs."""

    @abstractmethod
    def parse_output(
        self,
        output_dir: Path,
        sample_json: dict,
        config: dict,
    ) -> ToolPrediction:
        """Parse native output → unified ``ToolPrediction``."""

    # ------------------------------------------------------------------ pipeline

    def predict(
        self,
        sample_json: dict,
        work_dir: Path,
        config: dict,
    ) -> ToolPrediction:
        """Run the full prepare → run → parse pipeline.

        Never raises: any exception is folded into a ``success=False``
        ToolPrediction so a single bad sample doesn't abort batch runs.
        """
        if not self.tool_id:
            raise NotImplementedError("subclass must set class attr tool_id")
        if self.category not in {"A", "B", "C", "D"}:
            raise NotImplementedError(
                f"subclass must set class attr category in 'ABCD' "
                f"(got {self.category!r})"
            )
        sample_id = sample_json.get("sample_id", "<unknown>")
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        start = time.monotonic()
        try:
            input_paths = self.prepare_input(sample_json, work_dir, config)
        except Exception as e:  # noqa: BLE001
            return make_failure_prediction(
                tool_id=self.tool_id,
                category=self.category,
                sample_id=sample_id,
                error_message=f"prepare_input failed: {type(e).__name__}: {e}\n"
                              + traceback.format_exc(limit=4),
                runtime_seconds=time.monotonic() - start,
            )

        try:
            output_dir = self.run_tool(input_paths, work_dir, config)
        except Exception as e:  # noqa: BLE001
            return make_failure_prediction(
                tool_id=self.tool_id,
                category=self.category,
                sample_id=sample_id,
                error_message=f"run_tool failed: {type(e).__name__}: {e}\n"
                              + traceback.format_exc(limit=4),
                runtime_seconds=time.monotonic() - start,
                raw_output_dir=str(work_dir),
            )

        try:
            prediction = self.parse_output(output_dir, sample_json, config)
        except Exception as e:  # noqa: BLE001
            return make_failure_prediction(
                tool_id=self.tool_id,
                category=self.category,
                sample_id=sample_id,
                error_message=f"parse_output failed: {type(e).__name__}: {e}\n"
                              + traceback.format_exc(limit=4),
                runtime_seconds=time.monotonic() - start,
                raw_output_dir=str(output_dir),
            )

        # Stamp runtime if the adapter didn't provide its own.
        if prediction.runtime_seconds is None:
            object.__setattr__(  # bypass pydantic frozen if any
                prediction, "runtime_seconds", time.monotonic() - start,
            )
        return prediction

    # ------------------------------------------------------------------ utility

    def fail(
        self,
        sample_id: str,
        error_message: str,
        *,
        runtime_seconds: Optional[float] = None,
        raw_output_dir: Optional[str] = None,
    ) -> ToolPrediction:
        """Build a failure record from inside a subclass (e.g. when
        ``parse_output`` decides the run was a soft failure)."""
        return make_failure_prediction(
            tool_id=self.tool_id,
            category=self.category,
            sample_id=sample_id,
            error_message=error_message,
            runtime_seconds=runtime_seconds,
            raw_output_dir=raw_output_dir,
        )
