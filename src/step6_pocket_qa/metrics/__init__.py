"""Per-metric implementations for step 6 pocket QA scoring.

Each module exposes a single ``def metric_fn(...) -> tuple[Optional[float], dict]``
that returns ``(score_in_[0,1] or None, info_dict)``. The scorer wraps each
call in ``try/except`` so a single failing metric never aborts the others.
"""
