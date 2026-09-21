#!/usr/bin/env python3
"""RiboSeer — run the whole pipeline over a sample list with one command.

    python run_pipeline.py \\
        --processed-dir data/processed \\
        --sample-list   data/splits/test.txt \\
        --output-dir    data/run/test \\
        --weight-tensor data/W.json \\
        --tools local

Stages, in order (steps 1 and 2 of the paper's GENESIS are offline — build
the corpus first, see the README):

    SCOPE    step 2  target profiling            (LLM)
    MAESTRO  step 3  tool selection              (LLM + UCB weight tensor)
    RELAY    step 4  15-tool dispatch            (adapters, see below)
    HARMONY  step 5  learned fusion
    VERDICT  step 6  quality scoring             (+ step 7 refinement loop)
    MEMORY   step 8  weight-tensor update

Each sample writes one JSONL per stage under ``<output-dir>/step<N>/`` plus
one summary line under ``<output-dir>/summary/``, so a crashed sample can be
resumed with ``--resume`` and re-examined stage by stage.

Tool selection
--------------
MAESTRO picks per sample, the 7-tool MANDATORY core (paper Table 1) is
appended, and RELAY runs the result:

* ``--tools all`` (default) — every planned tool.
* ``--tools local`` — everything except AlphaFold 3, RNABindRPlus and
  BindUP, which only run on their authors' web servers. Submit those by
  hand (their adapters write the payload) and drop the results under
  ``data/external/<tool>/``; the README's "manual web submission" section
  lists the addresses. A tool that produced no result is treated like any
  other failure: HARMONY zero-fills its columns.
* ``--tools boltz2,chai1,p2rank`` — an explicit subset.

Every other flag is passed straight through to the stage runner
(``scripts/batch_predict.py``); ``--help`` on that module lists them all.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.batch_predict import main as batch_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return batch_main(argv, description=__doc__, prog="run_pipeline.py")


if __name__ == "__main__":
    raise SystemExit(main())
