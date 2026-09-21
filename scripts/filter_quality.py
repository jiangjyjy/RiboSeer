"""Quality / "easy-sample" filter for the RNA-protein pocket dataset.

Why
---
``scripts/filter_samples.py`` only bands by RNA length. Boltz-2 / Chai-1
/ RF2NA still do badly on samples that are *intrinsically hard* for a
structure predictor even inside 20-100 nt RNA:

  - very long protein chains (slow, OOM, lower pLDDT),
  - almost no ground-truth binding residues (|GT| tiny -> the pocket
    is a needle; recall/precision are noisy and usually ~0),
  - poor experimental resolution (the "answer" itself is fuzzy).

This script keeps the leakage-safe cluster split from
``filter_samples.py`` but adds tunable quality gates::

    --max-protein-length 400      protein.length <= N
    --min-protein-length 0        protein.length >= N
    --min-gt-binding 5            |interaction.binding_protein_residues| >= N
    --max-resolution 3.5          data_availability.resolution <= R (A)
    --min-binding-ratio 0.0       |GT| / protein.length >= r
    --min-rna-length 20           rna length band (same as filter_samples)
    --max-rna-length 100
    --drop-missing-resolution     also drop samples whose resolution is
                                  null (NMR / cryo-EM / predicted). Default
                                  is to KEEP them (only the numeric cap is
                                  enforced when a value exists).

Usage
-----
::

    # 1. just look at the distribution + a combo table, write nothing
    python scripts/filter_quality.py \\
        --input-dir data/processed_filtered_100/samples/ \\
        --output-dir data/processed_quality/ \\
        --dry-run

    # 2. commit a chosen combo
    python scripts/filter_quality.py \\
        --input-dir data/processed_filtered_100/samples/ \\
        --output-dir data/processed_quality/ \\
        --max-protein-length 400 \\
        --min-gt-binding 5 \\
        --max-resolution 3.5 \\
        --min-rna-length 20 --max-rna-length 100

``--dry-run`` always prints the multi-combo comparison table
(kept / train / test for a ladder of threshold combos) so you can
pick a target before materialising anything.

Layout produced under ``--output-dir`` (mirrors filter_samples.py)::

    samples/<id>.json            (copied, or symlinked with --symlink)
    splits/train.txt val.txt test.txt
    splits/train_200.txt test_200.txt
    filter_quality_stats.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Optional


# ---- field extractors (pure, unit-tested) --------------------------------


def rna_length_of(sample: dict) -> Optional[int]:
    """``rna.length`` if a positive int, else ``len(rna.sequence)``,
    else ``None``."""
    rna = sample.get("rna") or {}
    n = rna.get("length")
    if isinstance(n, int) and n > 0:
        return n
    seq = rna.get("sequence")
    if isinstance(seq, str) and seq:
        return len(seq)
    return None


def protein_length_of(sample: dict) -> Optional[int]:
    """``protein.length`` if a positive int, else ``len(protein.sequence)``,
    else ``None``."""
    prot = sample.get("protein") or {}
    n = prot.get("length")
    if isinstance(n, int) and n > 0:
        return n
    seq = prot.get("sequence")
    if isinstance(seq, str) and seq:
        return len(seq)
    return None


def gt_binding_count(sample: dict) -> int:
    """Number of ground-truth binding protein residues (0 if absent)."""
    gt = (sample.get("interaction") or {}).get(
        "binding_protein_residues") or []
    return len(gt) if isinstance(gt, list) else 0


def resolution_of(sample: dict) -> Optional[float]:
    """``data_availability.resolution`` as float, else ``None``
    (NMR / cryo-EM / predicted structures often have no resolution)."""
    r = (sample.get("data_availability") or {}).get("resolution")
    if isinstance(r, (int, float)) and r > 0:
        return float(r)
    return None


def binding_ratio_of(sample: dict) -> Optional[float]:
    """``|GT| / protein.length`` (None if protein length unknown)."""
    pl = protein_length_of(sample)
    if not pl:
        return None
    return gt_binding_count(sample) / pl


def domain_of(sample: dict) -> str:
    """``protein.domain`` bucketed; ``"<none>"`` when null/absent."""
    d = (sample.get("protein") or {}).get("domain")
    if d is None or d == "":
        return "<none>"
    return str(d)


def method_of(sample: dict) -> str:
    m = (sample.get("data_availability") or {}).get("experimental_method")
    return str(m) if m else "<none>"


def tier_of(sample: dict) -> str:
    t = (sample.get("data_availability") or {}).get("quality_tier")
    return str(t) if t else "<none>"


# ---- split map (leakage-safe, reused from the original cluster split) ----


def load_splits_map(splits_json: Path) -> dict[str, str]:
    """``{sample_id: split}`` from splits.json's ``samples`` block,
    keys also stored lower-cased. ``split`` may be None -> ``""``."""
    if not splits_json.is_file():
        return {}
    data = json.loads(splits_json.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for sid, meta in (data.get("samples") or {}).items():
        sp = meta.get("split") if isinstance(meta, dict) else meta
        sp = sp or ""
        out[sid] = sp
        out.setdefault(sid.lower(), sp)
    return out


# ---- quality predicate ---------------------------------------------------


class QualityThresholds:
    """One immutable bundle of gates. ``passes()`` is the single source
    of truth used by both the real filter and the dry-run combo table."""

    __slots__ = ("min_rna", "max_rna", "min_prot", "max_prot",
                 "min_gt", "max_res", "min_ratio", "drop_missing_res",
                 "allowed_tiers")

    def __init__(self, *, min_rna=20, max_rna=100, min_prot=0,
                 max_prot=10**9, min_gt=0, max_res=None,
                 min_ratio=0.0, drop_missing_res=False,
                 allowed_tiers=None):
        self.min_rna = min_rna
        self.max_rna = max_rna
        self.min_prot = min_prot
        self.max_prot = max_prot
        self.min_gt = min_gt
        self.max_res = max_res
        self.min_ratio = min_ratio
        self.drop_missing_res = drop_missing_res
        # None = accept any tier; else a set of acceptable tier strings
        self.allowed_tiers = (set(allowed_tiers)
                              if allowed_tiers else None)

    def passes(self, sample: dict) -> tuple[bool, str]:
        """Return ``(kept, reason)``. ``reason`` is ``"ok"`` when kept,
        else the first failing gate (deterministic order)."""
        rlen = rna_length_of(sample)
        if rlen is None:
            return False, "no_rna_length"
        if rlen < self.min_rna:
            return False, "rna_too_short"
        if rlen > self.max_rna:
            return False, "rna_too_long"

        plen = protein_length_of(sample)
        if plen is None:
            return False, "no_protein_length"
        if plen < self.min_prot:
            return False, "protein_too_short"
        if plen > self.max_prot:
            return False, "protein_too_long"

        if gt_binding_count(sample) < self.min_gt:
            return False, "gt_too_few"

        if self.min_ratio > 0.0:
            br = binding_ratio_of(sample)
            if br is None or br < self.min_ratio:
                return False, "binding_ratio_low"

        if self.max_res is not None:
            res = resolution_of(sample)
            if res is None:
                if self.drop_missing_res:
                    return False, "resolution_missing"
            elif res > self.max_res:
                return False, "resolution_poor"

        if self.allowed_tiers is not None:
            if tier_of(sample) not in self.allowed_tiers:
                return False, "tier_excluded"

        return True, "ok"


# ---- scan ----------------------------------------------------------------


def scan_dir(input_dir: Path) -> list[tuple[str, Path, dict]]:
    """Load every ``*.json``; skip unreadable. Returns
    ``[(sample_id, path, sample_dict), ...]`` sorted by filename."""
    out: list[tuple[str, Path, dict]] = []
    for jf in sorted(input_dir.glob("*.json")):
        try:
            s = json.loads(jf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append((s.get("sample_id") or jf.stem, jf, s))
    return out


def apply_filter(
    scanned: list[tuple[str, Path, dict]],
    splits_map: dict[str, str],
    thr: QualityThresholds,
) -> dict:
    """Apply ``thr`` to a pre-scanned list. Returns kept records,
    per-split id lists, reason histogram. Pure / read-only."""
    kept: list[tuple[str, Path, int, str]] = []
    by_split: dict[str, list[str]] = {"train": [], "val": [],
                                      "test": [], "unknown": []}
    reasons: dict[str, int] = {}
    rna_lens: list[int] = []

    for sid, jf, sample in scanned:
        ok, why = thr.passes(sample)
        if not ok:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        reasons["ok"] = reasons.get("ok", 0) + 1
        split = (splits_map.get(sid)
                 or splits_map.get(sid.lower()) or "")
        bucket = split if split in ("train", "val", "test") else "unknown"
        rlen = rna_length_of(sample) or 0
        kept.append((sid, jf, rlen, bucket))
        by_split[bucket].append(sid)
        rna_lens.append(rlen)

    return {
        "kept": kept,
        "by_split": by_split,
        "reasons": reasons,
        "rna_lens": rna_lens,
        "counts": {
            "total_scanned": len(scanned),
            "kept": len(kept),
        },
    }


def sample_subset(ids: list[str], n: int, seed: int) -> list[str]:
    """Reproducible subset: sort for determinism, then seeded sample."""
    pool = sorted(ids)
    if len(pool) <= n:
        return pool
    return sorted(random.Random(seed).sample(pool, n))


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "min": None, "max": None,
                "mean": None, "median": None}
    return {
        "n": len(vals),
        "min": round(min(vals), 3),
        "max": round(max(vals), 3),
        "mean": round(statistics.fmean(vals), 3),
        "median": round(statistics.median(vals), 3),
    }


# ---- dry-run combo ladder ------------------------------------------------

# (label, kwargs for QualityThresholds). Edited freely - this is the
# table the user picks a row from. Every combo keeps the 20-100 RNA band.
DEFAULT_COMBOS: list[tuple[str, dict]] = [
    ("prot<=500, GT>=3",
     dict(max_prot=500, min_gt=3)),
    ("prot<=400, GT>=5",
     dict(max_prot=400, min_gt=5)),
    ("prot<=400, GT>=5, res<=3.5",
     dict(max_prot=400, min_gt=5, max_res=3.5)),
    ("prot<=300, GT>=5, res<=3.5",
     dict(max_prot=300, min_gt=5, max_res=3.5)),
    ("prot<=300, GT>=5, res<=3.0",
     dict(max_prot=300, min_gt=5, max_res=3.0)),
    ("prot<=250, GT>=8, res<=3.0",
     dict(max_prot=250, min_gt=8, max_res=3.0)),
    ("prot<=200, GT>=10, res<=3.0",
     dict(max_prot=200, min_gt=10, max_res=3.0)),
]


def combo_table(scanned, splits_map, combos, base_kw) -> list[dict]:
    """For each (label, kw): merge over ``base_kw`` (RNA band etc.),
    return rows with kept/train/val/test counts."""
    rows = []
    for label, kw in combos:
        merged = dict(base_kw)
        merged.update(kw)
        thr = QualityThresholds(**merged)
        r = apply_filter(scanned, splits_map, thr)
        bs = r["by_split"]
        rows.append({
            "label": label,
            "kept": r["counts"]["kept"],
            "train": len(bs["train"]),
            "val": len(bs["val"]),
            "test": len(bs["test"]),
            "unknown": len(bs["unknown"]),
        })
    return rows


def _print_combo_table(rows: list[dict]) -> None:
    w = max((len(r["label"]) for r in rows), default=10)
    print()
    print("  threshold combo (all keep RNA 20-100 nt)")
    print(f"  {'combo'.ljust(w)}  {'kept':>6} {'train':>6} "
          f"{'val':>5} {'test':>5} {'unk':>5}")
    print(f"  {'-' * w}  {'-' * 6} {'-' * 6} {'-' * 5} "
          f"{'-' * 5} {'-' * 5}")
    for r in rows:
        print(f"  {r['label'].ljust(w)}  {r['kept']:>6} "
              f"{r['train']:>6} {r['val']:>5} {r['test']:>5} "
              f"{r['unknown']:>5}")
    print()


# ---- IO ------------------------------------------------------------------


def _materialise(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        try:
            os.symlink(src.resolve(), dst)
            return
        except (OSError, NotImplementedError):
            pass
    dst.write_bytes(src.read_bytes())


def _write_lines(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""),
                    encoding="utf-8")


# ---- main ----------------------------------------------------------------


def _build_thresholds(args) -> QualityThresholds:
    return QualityThresholds(
        min_rna=args.min_rna_length,
        max_rna=args.max_rna_length,
        min_prot=args.min_protein_length,
        max_prot=args.max_protein_length,
        min_gt=args.min_gt_binding,
        max_res=args.max_resolution,
        min_ratio=args.min_binding_ratio,
        drop_missing_res=args.drop_missing_resolution,
        allowed_tiers=([t.strip() for t in args.quality_tier.split(",")
                        if t.strip()] if args.quality_tier else None),
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path,
                   default=Path("data/processed_filtered_100/samples"),
                   help="dir of <sample_id>.json.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="filtered dataset root (samples/ + splits/).")
    p.add_argument("--splits-json", type=Path,
                   default=Path("data/processed/splits.json"),
                   help="source of the leakage-safe cluster split.")
    p.add_argument("--max-protein-length", type=int, default=400)
    p.add_argument("--min-protein-length", type=int, default=0)
    p.add_argument("--min-gt-binding", type=int, default=5,
                   help="keep iff |binding_protein_residues| >= N.")
    p.add_argument("--max-resolution", type=float, default=None,
                   help="keep iff resolution <= R angstrom (only "
                        "enforced when a numeric resolution exists; "
                        "see --drop-missing-resolution).")
    p.add_argument("--min-binding-ratio", type=float, default=0.0,
                   help="keep iff |GT|/protein_length >= r (0 = off).")
    p.add_argument("--drop-missing-resolution", action="store_true",
                   help="also drop samples with null resolution.")
    p.add_argument("--quality-tier", default=None,
                   help="comma list of data_availability.quality_tier "
                        "to KEEP, e.g. 'strict,standard' (default: "
                        "accept any tier).")
    p.add_argument("--min-rna-length", type=int, default=20)
    p.add_argument("--max-rna-length", type=int, default=100)
    p.add_argument("--train-sample-n", type=int, default=200)
    p.add_argument("--test-sample-n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--symlink", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="print stats + combo table, write nothing.")
    p.add_argument("--no-combo-table", action="store_true",
                   help="skip the multi-combo ladder in --dry-run.")
    args = p.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.min_rna_length < 0 or args.max_rna_length <= 0:
        print("ERROR: RNA length bounds must be >=0 / >0",
              file=sys.stderr)
        return 1
    if args.min_rna_length > args.max_rna_length:
        print(f"ERROR: --min-rna-length ({args.min_rna_length}) > "
              f"--max-rna-length ({args.max_rna_length})",
              file=sys.stderr)
        return 1
    if args.max_protein_length <= 0:
        print("ERROR: --max-protein-length must be > 0", file=sys.stderr)
        return 1
    if not args.input_dir.is_dir():
        print(f"ERROR: --input-dir not a directory: {args.input_dir}",
              file=sys.stderr)
        return 1

    splits_map = load_splits_map(args.splits_json)
    if not splits_map:
        print(f"WARN: no splits loaded from {args.splits_json}; kept "
              f"samples will land in 'unknown'.", file=sys.stderr)

    scanned = scan_dir(args.input_dir)
    thr = _build_thresholds(args)
    res = apply_filter(scanned, splits_map, thr)
    by_split = res["by_split"]

    # Distribution snapshot over the *input* pool (pre-filter) so the
    # user can sanity-check the thresholds against reality.
    plens = [protein_length_of(s) for _, _, s in scanned]
    plens = [v for v in plens if v]
    gts = [gt_binding_count(s) for _, _, s in scanned]
    ress = [resolution_of(s) for _, _, s in scanned]
    ress = [v for v in ress if v is not None]
    n_res_missing = sum(1 for _, _, s in scanned
                        if resolution_of(s) is None)

    train_200 = sample_subset(by_split["train"],
                              args.train_sample_n, args.seed)
    test_200 = sample_subset(by_split["test"],
                             args.test_sample_n, args.seed)

    if not args.dry_run:
        samples_out = args.output_dir / "samples"
        splits_out = args.output_dir / "splits"
        for sid, src, _r, _b in res["kept"]:
            _materialise(src, samples_out / f"{sid}.json", args.symlink)
        for name in ("train", "val", "test"):
            _write_lines(splits_out / f"{name}.txt",
                         sorted(by_split[name]))
        _write_lines(splits_out / "train_200.txt", train_200)
        _write_lines(splits_out / "test_200.txt", test_200)
        (args.output_dir / "filter_quality_stats.json").write_text(
            json.dumps({
                "thresholds": {
                    "min_rna_length": args.min_rna_length,
                    "max_rna_length": args.max_rna_length,
                    "min_protein_length": args.min_protein_length,
                    "max_protein_length": args.max_protein_length,
                    "min_gt_binding": args.min_gt_binding,
                    "max_resolution": args.max_resolution,
                    "min_binding_ratio": args.min_binding_ratio,
                    "drop_missing_resolution":
                        args.drop_missing_resolution,
                    "quality_tier": args.quality_tier,
                },
                "seed": args.seed,
                "counts": res["counts"],
                "drop_reasons": res["reasons"],
                "split_counts": {k: len(v)
                                 for k, v in by_split.items()},
                "train_200_n": len(train_200),
                "test_200_n": len(test_200),
                "input_pool_stats": {
                    "protein_length": _stats(plens),
                    "gt_binding": _stats([float(x) for x in gts]),
                    "resolution": _stats(ress),
                    "n_resolution_missing": n_res_missing,
                },
            }, indent=2), encoding="utf-8")

    tag = "DRY-RUN (no writes)" if args.dry_run else "written"
    print(f"filter_quality [{tag}]")
    print(f"  input dir            : {args.input_dir}")
    print(f"  scanned              : {res['counts']['total_scanned']}")
    print(f"  kept                 : {res['counts']['kept']}")
    print(f"  split: train={len(by_split['train'])}  "
          f"val={len(by_split['val'])}  test={len(by_split['test'])}  "
          f"unknown={len(by_split['unknown'])}")
    print(f"  sampled: train_200={len(train_200)}  "
          f"test_200={len(test_200)}  (seed={args.seed})")
    print("  drop reasons         : " + (", ".join(
        f"{k}={v}" for k, v in sorted(res["reasons"].items())
        if k != "ok") or "(none)"))
    ps, gs, rs = _stats(plens), _stats([float(x) for x in gts]), \
        _stats(ress)
    print(f"  pool protein_length  : min={ps['min']} "
          f"med={ps['median']} mean={ps['mean']} max={ps['max']}")
    print(f"  pool |GT|            : min={gs['min']} "
          f"med={gs['median']} mean={gs['mean']} max={gs['max']}")
    print(f"  pool resolution (A)  : min={rs['min']} "
          f"med={rs['median']} max={rs['max']}  "
          f"missing={n_res_missing}")
    if args.dry_run and not args.no_combo_table:
        base_kw = dict(min_rna=args.min_rna_length,
                       max_rna=args.max_rna_length)
        rows = combo_table(scanned, splits_map,
                           DEFAULT_COMBOS, base_kw)
        _print_combo_table(rows)
        print("  pick a combo, then re-run WITHOUT --dry-run using "
              "the matching\n  --max-protein-length / --min-gt-binding "
              "/ --max-resolution flags.")
    if not args.dry_run:
        print(f"  output               : {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
