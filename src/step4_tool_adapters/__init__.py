"""Step 4 — tool adapters.

Each adapter wraps one of the deployed prediction tools (formula 5 in the paper):

    A_k : f_raw_k(r, p; θ_k) → P_k = (B_p^(k), B_r^(k), s_k, G^(k))

The adapter is responsible for:
  1. preparing the tool-specific input files (PDB / FASTA / YAML / ...);
  2. running the tool via subprocess (each tool lives in its own conda env);
  3. parsing the native output into a unified ``ToolPrediction`` schema.

Public surface re-exported here for convenience:
  - ``ToolPrediction`` / ``ToolPredictionSet`` (schemas)
  - ``BaseAdapter`` (abstract base class)
  - ``run_command`` / ``run_in_conda_env`` (subprocess helpers)
"""
from .base_adapter import BaseAdapter
from .schemas import ToolPrediction, ToolPredictionSet
from .tool_runner import run_command, run_in_conda_env, ToolRunResult

__all__ = [
    "BaseAdapter",
    "ToolPrediction",
    "ToolPredictionSet",
    "ToolRunResult",
    "run_command",
    "run_in_conda_env",
]
