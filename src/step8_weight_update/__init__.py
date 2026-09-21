"""Step 8 — weight tensor update (Section 3.6, Phase 2, of the paper; formulas 14-16).

Two pieces:

  - ``ema_updater``  — EMA update of W[k, m, j] from one PocketQAResult.
    Pure code, no LLM. The MVP path.
  - ``meta_correction`` — optional LLM call that returns a per-tool
    correction factor γ_k ∈ [0.5, 1.5] applied on top of the EMA result.
    Disabled by default; kicks in once enough history accrues.

Phase 8.0 ships ``schemas`` + ``ema_updater``; phase 8.1 adds
``meta_correction`` + ``run`` (CLI) + a full-pipeline e2e script.
"""
