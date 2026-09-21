"""Real RoseTTAFold2NA invocation script (server-only — DATABASE-BLOCKED).

NOT a unittest module — manual smoke that calls ``run_RF2NA.sh`` on
one sample inside the ``RF2NA2`` conda env.

⚠ BLOCKED ON DATABASE DOWNLOAD ⚠
---------------------------------
The RF2NA sequence database (~230 GB: BFD / UniRef / Rfam / etc.) is
still being downloaded onto the server. ``run_RF2NA.sh`` cannot run
without it — it will fail at the first MSA step. Do NOT invoke this
script until the database is fully staged. The adapter and this smoke
script are written so that the moment the DB lands you can just run
this file; until then it stays as code-only insurance.

When the database is ready, run by hand on the server::

    cd ~/riboseer
    conda activate riboseer
    python tests/test_rf2na_real.py \\
        --processed-dir data/processed \\
        --sample-id 1un6_B_F \\
        --rf2na-env RF2NA2 \\
        --rf2na-install-dir /opt/biotools/RoseTTAFold2NA \\
        --timeout 14400

Notes
-----
- First run on a sample takes 30 min – several hours: RF2NA does
  HHblits/HMMER searches against the full database, then folds.
- The python side does not need network access; the launcher script
  itself runs all DB-backed lookups locally.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step4_tool_adapters.adapters.rf2na_adapter import (  # noqa: E402
    RF2NAAdapter,
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
    p.add_argument("--rf2na-env", type=str, default="RF2NA2",
                   help="conda env where RoseTTAFold2NA is installed")
    p.add_argument("--rf2na-install-dir", type=Path, required=True,
                   help="RoseTTAFold2NA install dir (contains run_RF2NA.sh)")
    p.add_argument("--skip-script-patch", action="store_true",
                   help="skip the run_RF2NA.sh `conda activate RF2NA` "
                        "rewrite AND the make_protein_msa.sh BFD comment-"
                        "out (use only when both upstream files are "
                        "already correct for your environment)")
    p.add_argument("--bfd-available", action="store_true",
                   help="set if the BFD database is staged on disk; "
                        "passes skip_bfd=false so make_protein_msa.sh "
                        "keeps its BFD-using lines (default: skip_bfd=true)")
    p.add_argument("--work-dir", type=Path,
                   default=REPO / "data" / "step4_workdir",
                   help="staging dir for prepared inputs and tool outputs")
    p.add_argument("--contact-threshold", type=float, default=4.5)
    p.add_argument("--timeout", type=int, default=14400,
                   help="max seconds (default 4h; full MSA + folding can "
                        "exceed this on long proteins)")
    args = p.parse_args()

    sample = _load_sample(args.processed_dir, args.sample_id)
    sample_work = args.work_dir / args.sample_id
    sample_work.mkdir(parents=True, exist_ok=True)

    config = {
        "tools": {
            "rosettafold2na": {
                "conda_env": args.rf2na_env,
                "install_dir": str(args.rf2na_install_dir),
                "timeout": args.timeout,
                "skip_script_patch": args.skip_script_patch,
                "skip_bfd": not args.bfd_available,
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
    print(f"install_dir  : {args.rf2na_install_dir}")
    print(f"conda env    : {args.rf2na_env}")
    print(f"timeout      : {args.timeout}s")
    print("=" * 72)
    print("⚠ Reminder: this run requires the RF2NA database (~230 GB).")
    print("  If the DB is missing the call will fail at the MSA step.")
    print("=" * 72)

    adapter = RF2NAAdapter()
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
    print(f"pae_mean     : {pred.pae_mean}")
    print(f"binding_protein_residues ({n_prot}): {pred.binding_protein_residues}")
    print(f"binding_rna_nucleotides  ({n_rna}): {pred.binding_rna_nucleotides}")

    print()
    print("--- ToolPrediction (JSON) ---")
    print(json.dumps(pred.model_dump(mode="json"), indent=2))

    # Optional: compare against ground truth.
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
