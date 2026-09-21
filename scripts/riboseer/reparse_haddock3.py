"""Re-parse existing HADDOCK 3 run dirs to backfill per-residue scores.

Context
-------
``structure_utils.compute_distance_binding_scores`` used to filter the
docked pose strictly on chain ids ``A``/``B``. But ``prepare_input``
only renames *multi-char* source chains to A/B — a single-char source
chain (protein "A", RNA "E") is written under its original name and
HADDOCK 3 preserves it in the pose. So every sample whose RNA chain
wasn't literally "B" got ``per_residue_pae_score = null`` even though
the docking succeeded (``binding_protein_residues`` still populated via
``extract_contacts``'s content fallback). The fix added the same
content-based chain selection to ``compute_distance_binding_scores``.

This script applies that fix to an *already-docked* batch WITHOUT
re-running HADDOCK 3: it re-invokes ``Haddock3Adapter.parse_output`` on
each sample's existing ``haddock_run/`` dir and splices the refreshed
``haddock3`` entry back into the step4 JSONL. Re-docking 66 samples at
~10 min each would be ~11 h; re-parsing is seconds.

Usage (on the server, after deploying the fixed structure_utils.py)
-------------------------------------------------------------------
::

    python scripts/riboseer/reparse_haddock3.py \\
        --step4-dir     data/batch_test_v7/step4/ \\
        --processed-dir data/processed_quality \\
        --work-dir      data/batch_test_v7/work/ \\
        --config-dir    configs/ \\
        --sample-list   data/processed_quality/splits/test.txt

``--dry-run`` reports what WOULD change (how many samples gain a
non-null ``per_residue_pae_score``) without rewriting any JSONL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from step4_tool_adapters.adapters.haddock3_adapter import (  # noqa: E402
    Haddock3Adapter,
)
from step5_fusion.data_collector import (  # noqa: E402
    load_sample_json,
)


def _load_config(config_dir: Optional[Path]) -> dict:
    """Merge the step4 yaml (for contact_threshold / haddock3 tool cfg).
    Missing file → empty config; parse_output has sane defaults
    (contact_threshold 4.5, distance_scale 8.0)."""
    if config_dir is None:
        return {}
    path = Path(config_dir) / "step4_config.yaml"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _read_last_record(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    rec = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
    return rec


def _find_run_dir(work_dir: Path, sample_id: str) -> Optional[Path]:
    """The HADDOCK 3 run dir prepare_input created:
    ``<work-dir>/<sample_id>/haddock_run/``. Falls back to scanning for
    any ``haddock_run`` under the sample's work subtree."""
    direct = work_dir / sample_id / "haddock_run"
    if direct.is_dir():
        return direct
    sample_work = work_dir / sample_id
    if sample_work.is_dir():
        for cand in sample_work.rglob("haddock_run"):
            if cand.is_dir():
                return cand
    return None


def _per_res_len(entry: Optional[dict]) -> int:
    if not entry:
        return 0
    pae = entry.get("per_residue_pae_score")
    return len(pae) if isinstance(pae, dict) else 0


def reparse_sample(
    *,
    sample_id: str,
    step4_dir: Path,
    processed_dir: Path,
    work_dir: Path,
    config: dict,
    dry_run: bool,
) -> dict:
    """Re-parse one sample's HADDOCK 3 run dir. Returns a status dict
    describing what happened (for the summary table)."""
    jsonl = step4_dir / f"{sample_id}.jsonl"
    rec = _read_last_record(jsonl)
    if rec is None:
        return {"sample_id": sample_id, "status": "no_step4"}
    preds = rec.get("predictions") or []
    idx = next((i for i, p in enumerate(preds)
                if p.get("tool_id") == "haddock3"), None)
    if idx is None:
        return {"sample_id": sample_id, "status": "no_haddock3_entry"}
    old_entry = preds[idx]
    if not old_entry.get("success"):
        # The 24 timed-out / failed samples: nothing to re-parse.
        return {"sample_id": sample_id, "status": "haddock3_failed"}

    old_n = _per_res_len(old_entry)

    sample = load_sample_json(processed_dir, sample_id)
    if sample is None:
        return {"sample_id": sample_id, "status": "no_sample_json"}
    run_dir = _find_run_dir(work_dir, sample_id)
    if run_dir is None:
        return {"sample_id": sample_id, "status": "no_run_dir"}

    adapter = Haddock3Adapter()
    pred = adapter.parse_output(run_dir, sample, config)
    new_entry = pred.model_dump() if hasattr(pred, "model_dump") \
        else pred.dict()
    new_n = _per_res_len(new_entry)

    status = "unchanged"
    if new_n > 0 and old_n == 0:
        status = "fixed"
    elif new_n != old_n:
        status = "changed"

    if status in ("fixed", "changed") and not dry_run:
        preds[idx] = new_entry
        rec["predictions"] = preds
        tmp = jsonl.with_suffix(".jsonl.tmp")
        tmp.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        tmp.replace(jsonl)

    return {"sample_id": sample_id, "status": status,
            "old_per_res": old_n, "new_per_res": new_n}


def _load_sample_ids(path: Optional[Path]) -> Optional[list[str]]:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"sample-list not found: {path}")
    return [ln.split()[0] for ln in path.read_text(encoding="utf-8")
            .splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step4-dir", type=Path, required=True)
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True,
                   help="batch work dir holding "
                        "<sample_id>/haddock_run/.")
    p.add_argument("--config-dir", type=Path, default=None,
                   help="dir with step4_config.yaml (optional; "
                        "parse defaults used if absent).")
    p.add_argument("--sample-list", type=Path, default=None,
                   help="restrict to these ids; default = every "
                        "*.jsonl under --step4-dir.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not args.step4_dir.is_dir():
        print(f"ERROR: --step4-dir not a directory: {args.step4_dir}",
              file=sys.stderr)
        return 1
    if not args.work_dir.is_dir():
        print(f"ERROR: --work-dir not a directory: {args.work_dir}",
              file=sys.stderr)
        return 1
    try:
        sample_ids = _load_sample_ids(args.sample_list)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if sample_ids is None:
        sample_ids = sorted(p.stem for p in args.step4_dir.glob("*.jsonl"))

    config = _load_config(args.config_dir)

    from collections import Counter
    tally: Counter = Counter()
    fixed_rows: list[dict] = []
    for sid in sample_ids:
        res = reparse_sample(
            sample_id=sid, step4_dir=args.step4_dir,
            processed_dir=args.processed_dir, work_dir=args.work_dir,
            config=config, dry_run=args.dry_run)
        tally[res["status"]] += 1
        if res["status"] == "fixed":
            fixed_rows.append(res)

    mode = "DRY-RUN (no writes)" if args.dry_run else "applied"
    print(f"reparse haddock3 ({mode}) over {len(sample_ids)} samples:")
    for status, n in sorted(tally.items()):
        print(f"  {status:20s} {n}")
    if fixed_rows:
        print(f"\n{'fixed (null → scored):':22s}")
        for r in fixed_rows[:30]:
            print(f"  {r['sample_id']:16s} "
                  f"per_res {r['old_per_res']} → {r['new_per_res']}")
        if len(fixed_rows) > 30:
            print(f"  ... +{len(fixed_rows) - 30} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
