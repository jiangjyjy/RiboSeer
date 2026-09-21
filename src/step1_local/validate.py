"""Stage 1.6a — validate every sample JSON against the RiboSeer schema.

Uses pydantic v2 for shape/type validation, then runs a handful of semantic
checks that pydantic can't express (cross-field invariants, e.g.
`len(contact_pairs) == len(contact_distances)`).

Fields that the **server stages** (1.3b / 1.5) still need to fill are
allowed to be `null` here. They're tracked separately and surfaced in the
report's "server-pending field checklist" so a glance at the document tells you exactly
what's left to do before step 2.

Additionally, this script writes a sidecar
`data/processed/_feature_flags.csv` with one row per sample:

    sample_id, quality_tier, has_pi, has_bfactor, has_gc,
    ss_status, split, protein_cluster_id, rna_cluster_id

`dataset_loader.py` reads this sidecar (when it exists) to filter at the
index level without having to open 45,468 JSONs. Re-run `validate.py` after
the server stages finish to refresh it.

Usage:
    python src/step1_local/validate.py \\
        --processed-dir /path/to/riboseer/data/processed \\
        --stats-dir     /path/to/riboseer/data/stats
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from _common import sid_to_filename  # noqa: E402 — Windows DLL path fix
from tqdm import tqdm  # noqa: E402

from pydantic import BaseModel, ConfigDict, ValidationError  # noqa: E402


# ---------- pydantic schema -------------------------------------------------


class _Strict(BaseModel):
    """Base class: reject unknown fields so accidental drift shows up."""
    model_config = ConfigDict(extra="forbid")


class ProteinFeatures(_Strict):
    pI: float | None
    mean_bfactor: float | None
    aa_composition: dict[str, float] | None


class Protein(_Strict):
    chain_id: str
    sequence: str
    length: int
    resolved_residues: list[int]
    features: ProteinFeatures
    domain: None | str
    plddt_mean: float | None


class SecondaryStructurePred(_Strict):
    structure: str | None
    mfe: float | None
    fold_sequence: str | None
    sanitized_from_inosine: bool
    skipped_reason: str | None


class StructureComposition(_Strict):
    paired_frac: float
    hairpin_frac: float
    interior_frac: float
    multiloop_frac: float
    external_frac: float


class RNAFeatures(_Strict):
    gc_content: float | None
    secondary_structure_pred: SecondaryStructurePred | None
    structure_composition: StructureComposition | None
    # ss_status is only written once stage 1.3b runs; keep it optional here.
    ss_status: str | None = None


class RNA(_Strict):
    chain_id: str
    sequence: str
    length: int
    resolved_residues: list[int]
    has_modification: bool
    modifications: list[str]
    features: RNAFeatures
    rfam_family: str | None


class Interaction(_Strict):
    binding_protein_residues: list[int]
    binding_rna_nucleotides: list[int]
    contact_pairs: list[list[int]]
    contact_distances: list[float]
    distance_cutoff: float
    has_homolog: bool | None
    eclip_count: int | None


class DataAvailability(_Strict):
    has_experimental_structure: bool
    resolution: float | None
    experimental_method: str
    quality_tier: str
    msa_depth: int | None


class SplitInfo(_Strict):
    protein_cluster_id: str | None
    rna_cluster_id: str | None
    split: str | None


class Sample(_Strict):
    sample_id: str
    source_pdb: str
    protein: Protein
    rna: RNA
    interaction: Interaction
    data_availability: DataAvailability
    split_info: SplitInfo


VALID_TIERS = {"strict", "standard", "low", "discard"}
VALID_SPLITS = {"train", "val", "test"}
VALID_SS_STATUS = {"done", "skipped_too_long", "skipped_empty", "error", None}


# ---------- semantic checks -------------------------------------------------


def semantic_check(sample: Sample) -> list[str]:
    """Cross-field invariants that pydantic can't express.

    Returns a list of human-readable error strings (empty = all clean).
    """
    errs: list[str] = []

    if sample.protein.length != len(sample.protein.sequence):
        errs.append(
            f"protein.length ({sample.protein.length}) != "
            f"len(protein.sequence) ({len(sample.protein.sequence)})"
        )
    if sample.rna.length != len(sample.rna.sequence):
        errs.append(
            f"rna.length ({sample.rna.length}) != "
            f"len(rna.sequence) ({len(sample.rna.sequence)})"
        )

    n_pairs = len(sample.interaction.contact_pairs)
    n_dists = len(sample.interaction.contact_distances)
    if n_pairs != n_dists:
        errs.append(
            f"contact_pairs len ({n_pairs}) != contact_distances len ({n_dists})"
        )
    for i, pair in enumerate(sample.interaction.contact_pairs):
        if len(pair) != 2:
            errs.append(f"contact_pairs[{i}] is not length 2: {pair}")
            break  # one message is enough

    if sample.data_availability.quality_tier not in VALID_TIERS:
        errs.append(
            f"unknown quality_tier: {sample.data_availability.quality_tier!r}"
        )

    if sample.split_info.split not in VALID_SPLITS and sample.split_info.split is not None:
        errs.append(f"unknown split value: {sample.split_info.split!r}")

    ss = sample.rna.features.ss_status
    if ss not in VALID_SS_STATUS:
        errs.append(f"unknown ss_status: {ss!r}")

    # Consistency: ss_status == "done" iff secondary_structure_pred.structure is a string
    pred = sample.rna.features.secondary_structure_pred
    if ss == "done":
        if pred is None or pred.structure is None:
            errs.append("ss_status=done but secondary_structure_pred.structure is None")
        if sample.rna.features.structure_composition is None:
            errs.append("ss_status=done but structure_composition is None")
    elif ss in ("skipped_too_long", "skipped_empty", "error"):
        if pred is None:
            errs.append(f"ss_status={ss!r} but secondary_structure_pred is None (dict expected)")

    return errs


# ---------- sidecar writer --------------------------------------------------


def flag_row(sample: Sample) -> dict:
    prot_feats = sample.protein.features
    rna_feats = sample.rna.features
    return {
        "sample_id": sample.sample_id,
        "quality_tier": sample.data_availability.quality_tier,
        "has_pi": prot_feats.pI is not None,
        "has_bfactor": prot_feats.mean_bfactor is not None,
        "has_gc": rna_feats.gc_content is not None,
        "ss_status": rna_feats.ss_status or "pending",
        "split": sample.split_info.split or "",
        "protein_cluster_id": sample.split_info.protein_cluster_id or "",
        "rna_cluster_id": sample.split_info.rna_cluster_id or "",
    }


FLAG_FIELDS = ("sample_id", "quality_tier", "has_pi", "has_bfactor", "has_gc",
               "ss_status", "split", "protein_cluster_id", "rna_cluster_id")


# ---------- main loop -------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1.6a: validate sample JSONs + emit feature-flag sidecar."
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="validate only first N samples (debug)")
    parser.add_argument("--no-sidecar", action="store_true",
                        help="skip writing _feature_flags.csv")
    args = parser.parse_args()

    samples_dir = args.processed_dir / "samples"
    index_csv = args.processed_dir / "index.csv"
    args.stats_dir.mkdir(parents=True, exist_ok=True)

    if not index_csv.exists():
        sys.exit(f"index.csv not found at {index_csv}")

    with index_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Validating {len(rows)} samples from {index_csv}")

    flag_rows: list[dict] = []
    schema_errors: list[tuple[str, str]] = []
    semantic_errors: list[tuple[str, list[str]]] = []
    error_category_counts: Counter = Counter()

    null_fields = Counter()
    tier_counts: Counter = Counter()
    ss_status_counts: Counter = Counter()
    split_counts: Counter = Counter()
    n_pass = 0

    for row in tqdm(rows, desc="validate", mininterval=1.0):
        sid = row["sample_id"]
        path = samples_dir / (sid_to_filename(sid) + ".json")
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            schema_errors.append((sid, f"read: {type(e).__name__}: {e}"))
            error_category_counts["read_error"] += 1
            continue

        try:
            sample = Sample.model_validate(raw)
        except ValidationError as e:
            schema_errors.append((sid, str(e)))
            error_category_counts["schema_error"] += 1
            continue

        errs = semantic_check(sample)
        if errs:
            semantic_errors.append((sid, errs))
            for err in errs:
                # bucket by first two words
                bucket = " ".join(err.split()[:2])
                error_category_counts[f"semantic: {bucket}"] += 1
            continue

        n_pass += 1
        tier_counts[sample.data_availability.quality_tier] += 1
        ss_status_counts[sample.rna.features.ss_status or "pending"] += 1
        split_counts[sample.split_info.split or "unassigned"] += 1

        if sample.protein.features.pI is None:
            null_fields["protein.features.pI"] += 1
        if sample.protein.features.mean_bfactor is None:
            null_fields["protein.features.mean_bfactor"] += 1
        if sample.protein.features.aa_composition is None:
            null_fields["protein.features.aa_composition"] += 1
        if sample.protein.domain is None:
            null_fields["protein.domain"] += 1
        if sample.protein.plddt_mean is None:
            null_fields["protein.plddt_mean"] += 1
        if sample.rna.features.gc_content is None:
            null_fields["rna.features.gc_content"] += 1
        if sample.rna.features.secondary_structure_pred is None:
            null_fields["rna.features.secondary_structure_pred"] += 1
        if sample.rna.features.structure_composition is None:
            null_fields["rna.features.structure_composition"] += 1
        if sample.rna.features.ss_status is None:
            null_fields["rna.features.ss_status"] += 1
        if sample.rna.rfam_family is None:
            null_fields["rna.rfam_family"] += 1
        if sample.data_availability.msa_depth is None:
            null_fields["data_availability.msa_depth"] += 1
        if sample.interaction.has_homolog is None:
            null_fields["interaction.has_homolog"] += 1
        if sample.interaction.eclip_count is None:
            null_fields["interaction.eclip_count"] += 1
        if sample.split_info.protein_cluster_id is None:
            null_fields["split_info.protein_cluster_id"] += 1
        if sample.split_info.rna_cluster_id is None:
            null_fields["split_info.rna_cluster_id"] += 1
        if sample.split_info.split is None:
            null_fields["split_info.split"] += 1

        flag_rows.append(flag_row(sample))

    if not args.no_sidecar:
        flags_path = args.processed_dir / "_feature_flags.csv"
        with flags_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FLAG_FIELDS)
            w.writeheader()
            w.writerows(flag_rows)
        print(f"Wrote sidecar {flags_path}")

    # Render report
    lines = [
        "# Stage 1.6a — Validation Report",
        "",
        f"- Total samples scanned: **{len(rows)}**",
        f"- Passed pydantic schema + semantic checks: **{n_pass}**",
        f"- schema / read errors: **{sum(1 for _ in schema_errors)}**",
        f"- Semantic check errors: **{len(semantic_errors)}**",
        "",
        "## Quality tier distribution (passing samples)",
        "",
    ]
    for tier in ("strict", "standard", "low", "discard"):
        lines.append(f"- {tier}: {tier_counts.get(tier, 0)}")

    lines += [
        "",
        "## ss_status distribution",
        "",
    ]
    for s in ("done", "skipped_too_long", "skipped_empty", "error", "pending"):
        lines.append(f"- {s}: {ss_status_counts.get(s, 0)}")

    lines += [
        "",
        "## Split distribution",
        "",
    ]
    for s in ("train", "val", "test", "unassigned"):
        lines.append(f"- {s}: {split_counts.get(s, 0)}")

    lines += [
        "",
        "## Server-pending field checklist (number of null samples)",
        "",
        "These fields are **allowed** to be null (the local stages have not run the whole pipeline yet). "
        "Once the owning stage has run, the null counts should drop to the handful of real edge cases.",
        "",
        "| field | n null | owning stage |",
        "|------|--------|---------|",
    ]
    stage_owner = {
        "protein.features.pI": "1.4 (edge case: all-X protein)",
        "protein.features.mean_bfactor": "1.4 (edge case: cryo-EM B=0 placeholder)",
        "protein.features.aa_composition": "1.4 (edge case: all-X protein)",
        "protein.domain": "TODO (InterPro query, later step)",
        "protein.plddt_mean": "TODO (AlphaFold, step 3+)",
        "rna.features.gc_content": "1.3a (edge case: poly-inosine)",
        "rna.features.secondary_structure_pred": "**1.3b pending on server**",
        "rna.features.structure_composition": "**1.3b pending on server**",
        "rna.features.ss_status": "**1.3b pending on server**",
        "rna.rfam_family": "TODO (Rfam query, later step)",
        "data_availability.msa_depth": "TODO (MMseqs profile, later step)",
        "interaction.has_homolog": "TODO (homology search)",
        "interaction.eclip_count": "TODO (eCLIP dataset integration)",
        "split_info.protein_cluster_id": "**1.5 pending on server**",
        "split_info.rna_cluster_id": "**1.5 pending on server**",
        "split_info.split": "**1.5 pending on server**",
    }
    for field, n in sorted(null_fields.items(), key=lambda kv: -kv[1]):
        owner = stage_owner.get(field, "")
        lines.append(f"| `{field}` | {n} | {owner} |")

    if schema_errors:
        lines += ["", "## Schema errors (first 20)", ""]
        for sid, err in schema_errors[:20]:
            lines.append(f"- **{sid}**:")
            for line in str(err).splitlines()[:6]:
                lines.append(f"  ```\n  {line}\n  ```")
        if len(schema_errors) > 20:
            lines.append(f"- …{len(schema_errors)} schema errors in total")

    if semantic_errors:
        lines += ["", "## Semantic errors (first 20)", ""]
        for sid, errs in semantic_errors[:20]:
            lines.append(f"- **{sid}**: {'; '.join(errs)}")
        if len(semantic_errors) > 20:
            lines.append(f"- …{len(semantic_errors)} semantic errors in total")

    report_path = args.stats_dir / "validation_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"Passed       : {n_pass} / {len(rows)}")
    print(f"Schema errors: {len(schema_errors)}")
    print(f"Semantic err : {len(semantic_errors)}")
    print(f"Report       : {report_path}")

    if schema_errors or semantic_errors:
        print("  (see validation_report.md for details)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
