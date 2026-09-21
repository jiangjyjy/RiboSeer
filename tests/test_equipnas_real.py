"""Real EquiPNAS invocation script (server-only).

NOT a unittest module — a manual smoke that runs the full EquiPNAS
pipeline (preprocessing + prediction) on one sample inside the
``EquiPNAS`` conda env.

Run by hand on the server, e.g.::

    cd ~/riboseer
    conda activate riboseer            # for python deps (gemmi, pydantic, ...)
    python tests/test_equipnas_real.py \\
        --processed-dir data/processed \\
        --raw-dir /opt/raw_pdb \\
        --sample-id 1un6_B_F \\
        --equipnas-env EquiPNAS \\
        --equipnas-install-dir /opt/biotools/EquiPNAS \\
        --model-path models/EquiPNAS-RNA/E-l12-768.pt

Notes
-----
- EquiPNAS preprocessing assumes PSSM and ESM-2 inputs already exist
  on disk for each target. If they're missing, ``prepare_input`` will
  raise during the second preprocessing script. Pass
  ``--skip-preprocess`` if you've already staged the preprocessed
  feature directory by hand.
- Preprocessing also requires DSSP (``mkdssp``) inside the EquiPNAS
  conda env. The adapter probes for it via
  ``command -v mkdssp || command -v dssp`` and aborts with an install
  hint if absent. Use ``--skip-dssp-check`` only if DSSP lives at a
  non-standard path that ``which`` can't see inside ``conda run``.
- Default total timeout is 30 min (preprocessing) + 20 min (main run);
  override with ``--preprocess-timeout`` / ``--timeout`` if needed.
- The script does not need network access from the python side.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from step4_tool_adapters.adapters.equipnas_adapter import (  # noqa: E402
    EquiPNASAdapter,
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
    p.add_argument("--equipnas-env", type=str, default="EquiPNAS",
                   help="conda env where EquiPNAS is installed")
    p.add_argument("--equipnas-install-dir", type=Path, required=True,
                   help="EquiPNAS install dir (contains EquiPNAS.py and "
                        "Preprocessing/)")
    p.add_argument("--model-path", type=str,
                   default="models/EquiPNAS-RNA/E-l12-768.pt",
                   help="path to model state dict, relative to install_dir")
    p.add_argument("--work-dir", type=Path,
                   default=REPO / "data" / "step4_workdir",
                   help="staging dir for prepared inputs and tool outputs")
    p.add_argument("--prob-threshold", type=float, default=0.5,
                   help="probability threshold for binding_protein_residues")
    p.add_argument("--preprocess-timeout", type=int, default=1800)
    p.add_argument("--timeout", type=int, default=1200,
                   help="max seconds for the main EquiPNAS.py run")
    p.add_argument("--skip-preprocess", action="store_true",
                   help="assume the preprocessed dir is already populated "
                        "(skip the 3 preprocessing scripts)")
    p.add_argument("--skip-dssp-check", action="store_true",
                   help="skip the up-front `which mkdssp` probe (use only "
                        "if DSSP lives somewhere `which` can't reach)")
    args = p.parse_args()

    sample = _load_sample(args.processed_dir, args.sample_id)
    sample_work = args.work_dir / args.sample_id
    sample_work.mkdir(parents=True, exist_ok=True)

    config = {
        "tools": {
            "equipnas": {
                "conda_env": args.equipnas_env,
                "install_dir": str(args.equipnas_install_dir),
                "model_path": args.model_path,
                "timeout": args.timeout,
                "preprocess_timeout": args.preprocess_timeout,
                "residue_prob_threshold": args.prob_threshold,
                "skip_preprocess": args.skip_preprocess,
                "skip_dssp_check": args.skip_dssp_check,
            },
        },
        "structure_source": {"raw_dir": str(args.raw_dir)},
        "log_dir": str(REPO / "logs" / "step4"),
    }

    print("=" * 72)
    print(f"sample_id    : {args.sample_id}")
    print(f"source_pdb   : {sample['source_pdb']}")
    print(f"protein      : chain={sample['protein']['chain_id']} "
          f"len={sample['protein']['length']}")
    print(f"raw_dir      : {args.raw_dir}")
    print(f"work_dir     : {sample_work}")
    print(f"install_dir  : {args.equipnas_install_dir}")
    print(f"model_path   : {args.model_path}")
    print(f"conda env    : {args.equipnas_env}")
    print(f"skip preproc : {args.skip_preprocess}")
    print(f"prob thresh  : {args.prob_threshold}")
    print("=" * 72)

    adapter = EquiPNASAdapter()
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

    n_bind = len(pred.binding_protein_residues or [])
    n_scored = len(pred.per_residue_confidence or {})
    print(f"per-residue scored : {n_scored}")
    print(f"binding_protein_residues ({n_bind}): {pred.binding_protein_residues}")

    print()
    print("--- ToolPrediction (JSON) ---")
    print(json.dumps(pred.model_dump(mode="json"), indent=2))

    # Optional: compare against ground truth.
    gt = (sample.get("interaction") or {}).get("binding_protein_residues") or []
    if gt and pred.binding_protein_residues:
        gt_set = set(gt)
        pred_set = set(pred.binding_protein_residues)
        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        print()
        print(f"ground truth ({len(gt_set)} residues): {sorted(gt_set)}")
        print(f"TP={tp}  FP={fp}  FN={fn}  "
              f"precision={precision:.3f}  recall={recall:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
