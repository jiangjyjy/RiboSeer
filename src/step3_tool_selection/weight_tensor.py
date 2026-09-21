"""Weight tensor W[K, M, J] for tool reliability tracking (Section 3.5 of the paper).

Dimensions:
  K = tool_ids (the 15-tool library from tool_registry, paper Table 1 —
      UCB ranks all of them)
  M = 5 quality metrics (PocketQA dimensions)
  J = pocket categories ("{protein_domain}_x_{rna_structure}", up to 48)

Cold start: every (tool, metric, category) cell is initialized from a
per-tool-category default (Category A=0.7, B=0.5, C=0.6, D=0.6). As
step 8 feeds evaluation results back, individual cells diverge.

UCB scoring (formula 9 in the paper):
  U_k(j) = Σ_m α_m * W[k,m,j] + β * exploration_bonus(k, j)

At cold start n_k_j = 0 for all tools, so the exploration bonus is a
uniform constant β. The ranking is therefore entirely driven by the
weighted sum of default W values, which in turn reflects the category-level
priors (A tools score higher than B tools for instance).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from . import tool_registry


METRICS = [
    "structural_plausibility",
    "physicochemical_complementarity",
    "evolutionary_conservation",
    "cross_tool_consensus",
    "known_motif_consistency",
]

_DEFAULT_CATEGORY_WEIGHTS = {"A": 0.7, "B": 0.5, "C": 0.6, "D": 0.6}

# Category-agnostic fallback cell. A weight/count written under this key
# applies to *every* category query that lacks a category-specific cell —
# used to broadcast a global per-tool reliability (e.g. a training-set μ̂)
# across all pocket categories. Absent from cold-start / per-category
# tensors, so the fallback is a no-op for them (fully backward compatible).
GLOBAL_CATEGORY = "__global__"


class WeightTensor:
    """Sparse representation of W[K, M, J].

    Internal storage: ``{tool_id: {metric: {category: float}}}``.
    Cells that have never been written return the cold-start default
    for the tool's category.
    """

    def __init__(
        self,
        data: Optional[dict] = None,
        counts: Optional[dict] = None,
        default_weights: Optional[dict[str, float]] = None,
        alpha: Optional[dict[str, float]] = None,
        beta: float = 1.0,
    ) -> None:
        # W values — sparse dict-of-dict-of-dict
        self._data: dict[str, dict[str, dict[str, float]]] = data or {}
        # n_k_j: how many times tool k has been evaluated on category j
        self._counts: dict[str, dict[str, int]] = counts or {}
        # total evaluations per category (N_j)
        self._total_counts: dict[str, int] = {}
        self._rebuild_total_counts()
        # per-category cold-start default
        self._defaults = default_weights or dict(_DEFAULT_CATEGORY_WEIGHTS)
        # per-metric importance weights (should sum to ~1)
        self._alpha = alpha or {m: 1.0 / len(METRICS) for m in METRICS}
        # UCB exploration coefficient
        self.beta = beta

    def _rebuild_total_counts(self) -> None:
        totals: dict[str, int] = defaultdict(int)
        for tool_id, cat_counts in self._counts.items():
            for cat, n in cat_counts.items():
                totals[cat] += n
        self._total_counts = dict(totals)

    # -- read ---------------------------------------------------------------

    def _default_for(self, tool_id: str) -> float:
        try:
            cat = tool_registry.get_tool(tool_id).category
        except KeyError:
            return 0.5
        return self._defaults.get(cat, 0.5)

    def get_weight(self, tool_id: str, metric: str, category: str) -> float:
        cell = self._data.get(tool_id, {}).get(metric, {})
        if category in cell:
            return cell[category]
        if GLOBAL_CATEGORY in cell:          # category-agnostic fallback
            return cell[GLOBAL_CATEGORY]
        return self._default_for(tool_id)

    def get_weights(self, tool_id: str, category: str) -> dict[str, float]:
        """Return {metric: weight} for one (tool, category) pair."""
        return {m: self.get_weight(tool_id, m, category) for m in METRICS}

    def get_count(self, tool_id: str, category: str) -> int:
        cc = self._counts.get(tool_id, {})
        if category in cc:
            return cc[category]
        return cc.get(GLOBAL_CATEGORY, 0)    # category-agnostic fallback

    # -- write --------------------------------------------------------------

    def update(
        self,
        tool_id: str,
        category: str,
        metric: str,
        value: float,
    ) -> None:
        """Set W[tool_id, metric, category] = value."""
        self._data.setdefault(tool_id, {}).setdefault(metric, {})[category] = value

    def increment_count(self, tool_id: str, category: str) -> None:
        """Record one evaluation of `tool_id` on `category`."""
        self._counts.setdefault(tool_id, {})[category] = (
            self._counts.get(tool_id, {}).get(category, 0) + 1
        )
        self._total_counts[category] = self._total_counts.get(category, 0) + 1

    # -- UCB scoring --------------------------------------------------------

    def _exploration_bonus(self, tool_id: str, category: str) -> float:
        """UCB exploration term: β * sqrt(ln(N_j) / n_k_j).

        Cold start (N_j=0 or n_k_j=0): return β as a flat bonus so all
        unexplored tools get the same exploration incentive.
        """
        n_j = (self._total_counts.get(category)
               or self._total_counts.get(GLOBAL_CATEGORY, 0))
        n_k = self.get_count(tool_id, category)
        if n_j == 0 or n_k == 0:
            return self.beta
        return self.beta * math.sqrt(math.log(n_j) / n_k)

    def compute_utility(self, tool_id: str, category: str) -> float:
        """UCB utility score for one (tool, category) pair.

        U_k(j) = Σ_m α_m * W[k,m,j] + exploration_bonus
        """
        weights = self.get_weights(tool_id, category)
        weighted_sum = sum(
            self._alpha.get(m, 0.0) * weights.get(m, 0.0)
            for m in METRICS
        )
        return weighted_sum + self._exploration_bonus(tool_id, category)

    def compute_utility_scores(self, category: str) -> dict[str, float]:
        """Compute UCB utility for all tools, sorted desc."""
        scores = {
            t.tool_id: round(self.compute_utility(t.tool_id, category), 4)
            for t in tool_registry.get_all_tools()
        }
        return dict(sorted(scores.items(), key=lambda kv: -kv[1]))

    # -- prompt rendering ---------------------------------------------------

    def get_category_summary(self, category: str) -> str:
        """Format W slice for `category` as prompt-embeddable text."""
        lines = [f"Tool reliability weights for category '{category}':"]
        scores = self.compute_utility_scores(category)
        for tool_id, score in scores.items():
            try:
                tool = tool_registry.get_tool(tool_id)
            except KeyError:
                continue
            n = self.get_count(tool_id, category)
            w = self.get_weights(tool_id, category)
            avg_w = sum(w.values()) / len(w) if w else 0
            lines.append(
                f"  {tool.name} ({tool_id}): "
                f"UCB={score:.3f}, avg_weight={avg_w:.3f}, "
                f"evaluations={n}, category={tool.category}"
            )
        return "\n".join(lines)

    # -- persistence --------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = {
            "data": self._data,
            "counts": self._counts,
            "defaults": self._defaults,
            "alpha": self._alpha,
            "beta": self.beta,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "WeightTensor":
        if not path.is_file():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return cls(
            data=obj.get("data"),
            counts=obj.get("counts"),
            default_weights=obj.get("defaults"),
            alpha=obj.get("alpha"),
            beta=obj.get("beta", 1.0),
        )

    @classmethod
    def from_config(cls, config: dict) -> "WeightTensor":
        """Build from step3_config.yaml's weight/UCB sections."""
        wc = config.get("default_weights") or {}
        ucb = config.get("ucb") or {}
        alpha = ucb.get("alpha") or {}
        beta = float(ucb.get("beta", 1.0))
        return cls(default_weights=wc, alpha=alpha, beta=beta)
