"""Step 7 — iteration / decision loop (Section 3.8 of the paper, formulas 23-24).

Public surface:

  - ``schemas`` — Pydantic models for the LLM action and the
    iteration record/result.
  - ``prompts`` — built in phase 7.1.
  - ``iterator`` — built in phase 7.2 (main loop).
  - ``run`` — CLI entry point, built in phase 7.2.

Phase 7.0 only ships the schemas + a placeholder ``run`` so
``python -m step7_iteration.run`` doesn't crash before 7.2.
"""
