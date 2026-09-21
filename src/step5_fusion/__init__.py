"""Step 5 — Fusion (LLM assigns weights, noisy-OR composes per-residue probability).

Implements formulas (11)(12)(13) from Section 3.6, Phase 1, of the paper:

  c_k       = LLM(P_fuse, {P_k}, W[:,:,j], T_ctx)              ... (12)
  b_hat(i)  = 1 - ∏_k (1 - c_k · b_k(i))                        ... (11)
  B_p_hat   = {i : b_hat(i) > τ}                                ... (13)

Public surface:
  - schemas.ToolWeightAssignment / CompositeResult
  - noisy_or.noisy_or_fusion
  - fusion.fuse_predictions
  - run.main (CLI entry)
"""
