"""Prediction history for Step 2 — target characterization.

Stores past characterization results so subsequent runs can condition the
LLM prompt on the distribution of categories / confidence seen so far.

Current state: cold start (H is empty). The framework is wired up so that
once step 7/8 produce feedback, records flow in and `get_summary()` feeds
a useful prior into `build_messages(history_summary=...)`.

Storage format: one JSONL line per record, each line is a self-describing
dict with at least {sample_id, category, confidence, timestamp}. Extra
fields (tools_used, scores) are preserved but not required — they'll be
populated by later pipeline steps.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


class PredictionHistory:
    """In-memory history with JSONL persistence.

    Usage::

        h = PredictionHistory.load(Path("data/history/prediction_history.jsonl"))
        h.add_record({
            "sample_id": "1un6_B_F",
            "category": "RRM_x_stem_loop",
            "confidence": 0.9,
            "timestamp": "2026-04-18T...",
            "tools_used": [...],
            "scores": {...},
        })
        summary = h.get_summary(limit=10)
        h.save(Path("data/history/prediction_history.jsonl"))
    """

    def __init__(self, records: Optional[list[dict]] = None) -> None:
        self._records: list[dict] = list(records) if records else []

    # -- mutators -----------------------------------------------------------

    def add_record(self, record: dict) -> None:
        """Append one prediction record. Must contain at least `sample_id`
        and `category`; other fields are optional."""
        if "sample_id" not in record or "category" not in record:
            raise ValueError("record must have 'sample_id' and 'category'")
        self._records.append(record)

    # -- queries ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[dict]:
        return list(self._records)

    def get_category_stats(self) -> dict[str, dict[str, Any]]:
        """Return ``{category: {"count": int, "avg_confidence": float}}``
        sorted by count descending."""
        counts: Counter[str] = Counter()
        conf_sums: defaultdict[str, float] = defaultdict(float)
        for r in self._records:
            cat = r.get("category", "unknown")
            counts[cat] += 1
            conf = r.get("confidence")
            if isinstance(conf, (int, float)):
                conf_sums[cat] += float(conf)
        stats: dict[str, dict[str, Any]] = {}
        for cat, n in counts.most_common():
            avg = conf_sums[cat] / n if n else 0.0
            stats[cat] = {"count": n, "avg_confidence": round(avg, 3)}
        return stats

    def get_summary(
        self,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """Return a compact text summary suitable for injection into the
        system prompt via ``build_messages(history_summary=...)``.

        If `category` is given, only summarise records matching that category.
        `limit` caps the number of individual records mentioned (the
        aggregate stats always cover all records).
        """
        pool = self._records
        if category:
            pool = [r for r in pool if r.get("category") == category]
        if not pool:
            return ""

        stats = self.get_category_stats()
        lines = [f"Total predictions so far: {len(self._records)}"]

        # top-level category distribution (always full, not filtered)
        top_cats = list(stats.items())[:8]
        dist_parts = [f"{cat} {s['count']}×(avg conf {s['avg_confidence']:.2f})"
                      for cat, s in top_cats]
        lines.append("Category distribution: " + ", ".join(dist_parts))

        # recent examples from the filtered pool
        recent = pool[-limit:]
        if recent:
            lines.append(f"Recent {len(recent)} example(s):")
            for r in recent:
                sid = r.get("sample_id", "?")
                cat = r.get("category", "?")
                conf = r.get("confidence", "?")
                lines.append(f"  - {sid}: {cat} (conf={conf})")

        return "\n".join(lines)

    # -- persistence --------------------------------------------------------

    def save(self, path: Path) -> None:
        """Write all records to JSONL (overwrites)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in self._records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> "PredictionHistory":
        """Load from JSONL. Returns empty history if file doesn't exist."""
        records: list[dict] = []
        if path.is_file():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return cls(records)
