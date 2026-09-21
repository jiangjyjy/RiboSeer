"""Real P2Rank invocation script (server-only).

NOT a unittest module — this is a manual smoke that calls the real
P2Rank binary on one sample. Run by hand on the server, e.g.::

    cd ~/riboseer
    conda activate riboseer
    python tests/test_p2rank_real.py \\
        --processed-dir data/processed \\
        --raw-dir /opt/raw_pdb \\
        --sample-id 1un6_B_F \\
        --p2rank-install-dir /opt/biotools/p2rank_2.5

What it does:
  1. Load the sample JSON for the requested sample_id.
  2. Run ``P2RankAdapter.predict`` end-to-end (real subprocess call).
  3. Print: input PDB path, P2Rank's raw output dir, the unified
     ``ToolPrediction`` dump, and a quick per-pocket summary.

Why an argparse script instead of a unittest:
  - the real binary lives only on the server;
  - no point shipping a test that always skips locally;
  - hand-driven smoke is easier to read line-by-line than a
    "skipUnless" test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step4_tool_adapters.adapters.p2rank_adapter import (  # noqa: E402
    P2RankAdapter,
)


def _load_sample(processed_dir: Path, sample_id: str) -> dict:
    candidates = [
        processed_dir / "samples" / f"{sample_id}.json",
        processed_dir / f"{sample_id}.json",
    ]
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"sample {sample_id!r} not found in {[str(p) for p in candidates]}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed-dir", type=Path, required=True,
                   help="step 1 output dir containing samples/<id>.json")
    p.add_argument("--raw-dir", type=Path, required=True,
                   help="directory holding raw <pdb_id>.pdb / .cif files")
    p.add_argument("--sample-id", type=str, default="1un6_B_F")
    p.add_argument("--p2rank-install-dir", type=Path, required=True,
                   help="P2Rank installation directory (contains the "
                        "'prank' launcher script)")
    p.add_argument("--work-dir", type=Path,
                   default=REPO / "data" / "step4_workdir",
                   help="staging dir for prepared inputs and tool outputs")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--score-threshold", type=float, default=0.5,
                   help="ligandability threshold for binding_protein_residues")
    p.add_argument("--top-pockets", type=int, default=5)
    args = p.parse_args()

    sample = _load_sample(args.processed_dir, args.sample_id)
    sample_work = args.work_dir / args.sample_id
    sample_work.mkdir(parents=True, exist_ok=True)

    config = {
        "tools": {
            "p2rank": {
                "install_dir": str(args.p2rank_install_dir),
                "timeout": args.timeout,
                "residue_score_threshold": args.score_threshold,
                "top_pockets": args.top_pockets,
            },
        },
        "structure_source": {"raw_dir": str(args.raw_dir)},
        "log_dir": str(REPO / "logs" / "step4"),
    }

    print("=" * 72)
    print(f"sample_id   : {args.sample_id}")
    print(f"source_pdb  : {sample['source_pdb']}")
    print(f"protein     : chain={sample['protein']['chain_id']} "
          f"len={sample['protein']['length']}")
    print(f"raw_dir     : {args.raw_dir}")
    print(f"work_dir    : {sample_work}")
    print(f"install_dir : {args.p2rank_install_dir}")
    print("=" * 72)

    adapter = P2RankAdapter()
    pred = adapter.predict(sample, sample_work, config)

    print()
    print(f"success     : {pred.success}")
    print(f"runtime     : {pred.runtime_seconds:.2f}s"
          if pred.runtime_seconds else "runtime     : (none)")
    print(f"raw_output  : {pred.raw_output_dir}")

    if not pred.success:
        print()
        print("ERROR")
        print(pred.error_message)
        return 1

    print()
    print(f"binding_protein_residues ({len(pred.binding_protein_residues)}): "
          f"{pred.binding_protein_residues}")
    print()
    if pred.pockets:
        print(f"pockets ({len(pred.pockets)}):")
        for p_ in pred.pockets:
            print(f"  rank={p_.rank}  score={p_.score:.2f}  "
                  f"residues={p_.residues}")

    print()
    print("--- ToolPrediction (JSON) ---")
    print(json.dumps(pred.model_dump(mode="json"), indent=2))

    # Optional: compare against ground truth from sample JSON.
    gt = sample.get("interaction", {}).get("binding_protein_residues") or []
    if gt and pred.binding_protein_residues:
        gt_set = set(gt)
        pred_set = set(pred.binding_protein_residues)
        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        print()
        print(f"ground truth ({len(gt_set)} residues): {sorted(gt_set)}")
        print(f"TP={tp}  FP={fp}  FN={fn}  "
              f"precision={precision:.3f}  recall={recall:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
