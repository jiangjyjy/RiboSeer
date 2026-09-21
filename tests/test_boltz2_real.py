"""Real Boltz-2 invocation script (server-only).

NOT a unittest module — a manual smoke that calls the real ``boltz``
CLI on one sample inside the ``boltz`` conda env.

Run by hand on the server, e.g.::

    cd ~/riboseer
    conda activate riboseer          # for python deps (gemmi, pydantic, ...)
    python tests/test_boltz2_real.py \\
        --processed-dir data/processed \\
        --sample-id 1un6_B_F \\
        --boltz-env boltz \\
        --work-dir data/step4_workdir

Notes:
  - First run on a sample takes several minutes due to the MSA HTTP
    server lookup. Default timeout is 30 minutes.
  - The script does NOT need network access from the python side; the
    ``boltz`` CLI itself contacts the MSA server.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step4_tool_adapters.adapters.boltz2_adapter import (  # noqa: E402
    Boltz2Adapter,
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
    p.add_argument("--sample-id", type=str, default="1un6_B_F")
    p.add_argument("--boltz-env", type=str, default="boltz",
                   help="conda env name where the boltz CLI is installed")
    p.add_argument("--flags", type=str,
                   default="--use_msa_server --no_kernels",
                   help="extra flags passed to `boltz predict`")
    p.add_argument("--work-dir", type=Path,
                   default=REPO / "data" / "step4_workdir",
                   help="staging dir for prepared inputs and tool outputs")
    p.add_argument("--timeout", type=int, default=1800,
                   help="max seconds for the Boltz-2 run")
    p.add_argument("--contact-threshold", type=float, default=4.5)
    args = p.parse_args()

    sample = _load_sample(args.processed_dir, args.sample_id)
    sample_work = args.work_dir / args.sample_id
    sample_work.mkdir(parents=True, exist_ok=True)

    config = {
        "tools": {
            "boltz2": {
                "conda_env": args.boltz_env,
                "flags": args.flags,
                "timeout": args.timeout,
            },
        },
        "contact_threshold": args.contact_threshold,
        "log_dir": str(REPO / "logs" / "step4"),
    }

    print("=" * 72)
    print(f"sample_id    : {args.sample_id}")
    print(f"protein      : chain={sample['protein']['chain_id']} "
          f"len={sample['protein']['length']}")
    print(f"rna          : chain={sample['rna']['chain_id']} "
          f"len={sample['rna']['length']}")
    print(f"work_dir     : {sample_work}")
    print(f"conda env    : {args.boltz_env}")
    print(f"flags        : {args.flags}")
    print(f"timeout      : {args.timeout}s")
    print("=" * 72)

    adapter = Boltz2Adapter()
    pred = adapter.predict(sample, sample_work, config)

    print()
    print(f"success      : {pred.success}")
    rt = f"{pred.runtime_seconds:.1f}" if pred.runtime_seconds else "?"
    print(f"runtime      : {rt}s")
    print(f"raw_output   : {pred.raw_output_dir}")

    if not pred.success:
        print()
        print("ERROR")
        print(pred.error_message)
        return 1

    n_prot = len(pred.binding_protein_residues or [])
    n_rna = len(pred.binding_rna_nucleotides or [])
    print(f"plddt_mean   : {pred.plddt_mean}")
    print(f"iptm_score   : {pred.iptm_score}")
    print(f"pae_mean     : {pred.pae_mean}")
    print(f"binding_protein_residues ({n_prot}): {pred.binding_protein_residues}")
    print(f"binding_rna_nucleotides  ({n_rna}): {pred.binding_rna_nucleotides}")

    print()
    print("--- ToolPrediction (JSON) ---")
    print(json.dumps(pred.model_dump(mode="json"), indent=2))

    # Optional: compare against ground truth
    gt_p = (sample.get("interaction") or {}).get("binding_protein_residues") or []
    gt_r = (sample.get("interaction") or {}).get("binding_rna_nucleotides") or []
    if (gt_p or gt_r) and (pred.binding_protein_residues
                            or pred.binding_rna_nucleotides):
        print()
        print(f"GT protein residues ({len(gt_p)}): {sorted(gt_p)}")
        print(f"GT rna nucleotides  ({len(gt_r)}): {sorted(gt_r)}")
        for label, gt, pr in (
            ("protein", gt_p, pred.binding_protein_residues or []),
            ("rna", gt_r, pred.binding_rna_nucleotides or []),
        ):
            gt_set, pr_set = set(gt), set(pr)
            tp = len(gt_set & pr_set)
            fp = len(pr_set - gt_set)
            fn = len(gt_set - pr_set)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            print(f"  {label:7s}  TP={tp}  FP={fp}  FN={fn}  "
                  f"precision={precision:.3f}  recall={recall:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
