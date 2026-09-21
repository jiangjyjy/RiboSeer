"""Step 6 — Pocket QA scoring (Section 3.7 of the paper, formulas 17-22).

Five unsupervised quality sub-scores for one fused prediction
(``CompositeResult`` from step 5), aggregated as a weighted total in
``PocketQAResult``. No ground truth required; no LLM call.

Public surface:
  - ``score_prediction`` — top-level entry point in ``scorer``
  - ``PocketQAResult`` / ``MetricDetail`` — output schemas
"""
