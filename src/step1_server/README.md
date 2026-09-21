# GENESIS — step1_server

The two curation steps that need Linux-only tooling (ViennaRNA for RNA
secondary structure, MMseqs2 / CD-HIT for sequence clustering). Everything
else in step 1 runs anywhere — see `src/step1_local/`.

## Setup

The same environment as the rest of the pipeline, plus:

```bash
conda activate riboseer

# channel priority — required, otherwise the solver may downgrade shared
# deps and break `import RNA` / `import pydantic`
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
conda config --set channel_priority strict

conda install -c conda-forge -c bioconda mmseqs2 cd-hit -y

# ViennaRNA via pip: bioconda's viennarna 2.7.2 recipe is pinned to
# Python 3.10 and conflicts with a 3.11 env, while the PyPI `viennarna`
# package ships manylinux wheels for Python 3.8-3.14.
python -m pip install viennarna
```

Verify:

```bash
mmseqs version
python -c "import RNA; fc=RNA.fold_compound('GGCCAU'); print(fc.mfe())"
python -c "import pydantic; print(pydantic.VERSION)"
```

Run the scripts in the order below. Each is idempotent, atomic-write and
resume-safe, so Ctrl-C + re-run is always the fix for an interruption.

## Run order

| # | Stage | Script                              | Wall clock (est) | Writes to                                                   |
|---|-------|-------------------------------------|------------------|-------------------------------------------------------------|
| 1 | 1.3b  | `rna_secondary_structure.py`        | 1-4 h            | `rna.features.{ss_status, secondary_structure_pred, structure_composition}` |
| 2 | 1.5a  | `cluster_and_split.py`              | 10-30 min        | `data/processed/splits.json`, `data/stats/cluster_stats.md` |
| 3 | 1.5b  | `apply_splits.py`                   | 2-5 min          | `split_info.{protein_cluster_id, rna_cluster_id, split}` on every sample |

**1.5b depends on 1.5a's `splits.json`. 1.3b is independent — you can run it
before or after the 1.5 pair; they touch disjoint fields.**

End-to-end command sequence (after setup):

```bash
cd riboseer
conda activate riboseer

# 1.3b — RNA secondary structure (feel free to Ctrl-C + re-run; fully resumable)
python src/step1_server/rna_secondary_structure.py \
    --processed-dir data/processed \
    --stats-dir     data/stats

# 1.5a — MMseqs2 clustering + split
python src/step1_server/cluster_and_split.py \
    --processed-dir data/processed \
    --stats-dir     data/stats \
    --work-dir      data/_cluster_work \
    --threads       16

# 1.5b — merge splits.json into every sample JSON
python src/step1_server/apply_splits.py \
    --processed-dir data/processed
```

After all three finish, the processed corpus (and the new `data/stats/*.md`
reports) is complete and the local stages can resume.

---

## Scripts

### `rna_secondary_structure.py` (stage 1.3b)

MFE structure prediction via ViennaRNA for every sample's RNA chain.

**Reads**: `sample["rna"]["sequence"]`
**Writes**:
- `sample["rna"]["features"]["ss_status"]` — canonical string enum
  (`done` / `skipped_too_long` / `skipped_empty` / `error`; absence = `pending`)
- `sample["rna"]["features"]["secondary_structure_pred"]` — dot-bracket,
  MFE, sanitized fold sequence, and detailed `skipped_reason`
- `sample["rna"]["features"]["structure_composition"]` — per-element
  fractions (null when `ss_status != "done"`)

`ss_status` is the single field downstream code should filter on. The two
nested dicts carry the detail, but `ss_status` alone answers "is this
sample's SS prediction usable?".

**Pre-fold sanitization** (important — stage 1.3a flagged this):

| input char | action | note                                 |
| ---------- | ------ | ------------------------------------ |
| `A/C/G/U`  | keep   | standard bases                       |
| `-`        | drop   | missing-density gap                  |
| `I`        | → `G`  | inosine; 29 samples from 7wv3 / 7wv4 |
| anything else | → `N` | unknown / other mods                 |

Samples whose RNA sequence contained inosine are flagged with
`secondary_structure_pred.sanitized_from_inosine = true` so downstream code
can find them.

**Max length**: RNAfold MFE is O(n³), so sequences longer than `--max-length`
(default **4000 nt**) are **skipped** with `skipped_reason =
"too_long_XXXX_nt"`; their `structure_composition` stays null. Adjust if the
server has more time than compute.

**Resume rules** (keyed off `ss_status`):

| prior status        | default behavior      | `--force`        |
| ------------------- | --------------------- | ---------------- |
| `done`              | skip                  | refold           |
| `error`             | skip (don't retry)    | retry            |
| `skipped_empty`     | skip (immutable)      | retry            |
| `skipped_too_long`  | **re-evaluate**       | re-evaluate      |
| missing / `null`    | fold                  | fold             |

`skipped_too_long` gets re-evaluated every run because `--max-length` is a
CLI parameter that may be raised on a subsequent run. If the new cap still
doesn't fit, the sample is re-stamped `skipped_too_long` with no data loss.

**Sequence cache**: 45,468 samples come from ~1800 PDBs and many ribosome /
spliceosome structures reuse the same RNA across multiple protein partners.
Results are cached by SHA-1 of the sanitized sequence, so unique folds are
probably <5000 even for the full dataset.

**Classification**: each nucleotide is bucketed into one of

- `paired`       — inside a `(` / `)`
- `hairpin`      — unpaired, inside a loop closed by **1** pair
- `interior`     — unpaired, inside a loop closed by **2** pairs
  (covers symmetric interior loops AND asymmetric bulges)
- `multiloop`    — unpaired, inside a loop closed by **≥3** pairs (junction)
- `external`     — unpaired, not inside any closing pair (5'/3' dangles)

`structure_composition` stores the fraction of nucleotides in each bucket.

**Run**:

```bash
# small smoke test first (50 samples, no writes beyond these)
python src/step1_server/rna_secondary_structure.py \
    --processed-dir data/processed \
    --stats-dir data/stats \
    --limit 50

# full run
python src/step1_server/rna_secondary_structure.py \
    --processed-dir data/processed \
    --stats-dir data/stats

# if 4000 nt is too slow, lower the cap
python src/step1_server/rna_secondary_structure.py \
    --processed-dir data/processed \
    --stats-dir data/stats \
    --max-length 3000

# run the pure-Python classifier unit tests (no ViennaRNA needed)
python src/step1_server/rna_secondary_structure.py --selftest
```

**Estimated wall clock (full dataset)**: dominated by the longest unique
sequences. With the sequence cache and 4000 nt cap, expect **1-4 hours** on
a modern server. Without the cache it would be ~10× slower.

**Report**: `data/stats/rna_secondary_structure_report.md`.

---

### `cluster_and_split.py` (stage 1.5a)

MMseqs2 clustering on both sides of every (protein, RNA) sample, followed by
a cluster-aware split into train / val / test.

**Reads**: `data/processed/index.csv` and every `samples/*.json`
(sequences only — features and contacts aren't needed here).

**Writes**:
- `data/processed/splits.json` — canonical split map (see below)
- `data/stats/cluster_stats.md` — human-readable report

**What gets clustered**

1. **Dedup**: the same protein chain often appears in many samples (one
   protein bound to several RNAs in a ribosome), so unique chains are
   extracted first by `(pdb_id, chain_id)` before clustering. Same for RNA.
2. **Pre-cluster filter**:
   - all-X protein sequences (the 122 unclusterable edge cases from 1.4)
     are dropped — they can't provide meaningful identity matches
   - RNA sequences are sanitized with the same rule as 1.3b
     (`- → drop`, `I → G`, other non-ACGU → `N`) and empty results are
     dropped
   - optional `--max-rna-len` drops ultra-long rRNAs from clustering too
3. **MMseqs2 `easy-cluster`**:
   - protein: `--min-seq-id 0.3 -c 0.8 --cov-mode 0` (30 % identity,
     80 % mutual coverage — standard "remote-homologs in same cluster")
   - RNA:     `--min-seq-id 0.8 -c 0.8 --cov-mode 0 --search-type 3`
     (80 % identity in nucleotide mode; same coverage rule)

**Double-cluster definition for splitting**

Two samples are "homologous" iff **both** their protein cluster IDs and
their RNA cluster IDs match. The script groups samples by the tuple
`(prot_cid, rna_cid)` and treats each group as atomic — all members go to
the same split. A greedy largest-first bin-packer distributes groups into
train / val / test targeting `--split-ratio` (default `0.8,0.1,0.1`).

**Leakage audit**: the double-cluster rule allows *single-factor* overlap
(e.g. same protein cluster across val and train but different RNA). That's
by design, but the script counts those overlaps and reports them in
`cluster_stats.md` so you can tighten the rule later if needed.

**Unclusterable samples**: filtered-out chains produce `split = null` and
a descriptive `protein_cluster_id` / `rna_cluster_id` marker:

| marker                             | cause                                     |
|------------------------------------|-------------------------------------------|
| `unclusterable_X_only`             | protein sequence is all X (122 samples)   |
| `unclusterable_rna_empty`          | RNA sanitizes to empty                    |
| `unclusterable_rna_too_long`       | RNA > `--max-rna-len` and opt-in filtered |
| `unclusterable_protein_missing`    | chain not found in cluster tsv (should never happen in normal runs) |
| `unclusterable_rna_missing`        | same, RNA side                            |

**Run**:

```bash
# default (30 % / 80 % identity, 8:1:1 split, seed 42)
python src/step1_server/cluster_and_split.py \
    --processed-dir data/processed \
    --stats-dir     data/stats \
    --work-dir      data/_cluster_work \
    --threads       16

# tighter split (25 % protein id, 90 % RNA id, different seed)
python src/step1_server/cluster_and_split.py \
    --processed-dir data/processed \
    --stats-dir     data/stats \
    --work-dir      data/_cluster_work \
    --protein-min-seq-id 0.25 \
    --rna-min-seq-id 0.90 \
    --seed 7

# skip ultra-long rRNAs from the RNA clustering (still appear in splits.json as unclusterable_rna_too_long)
python src/step1_server/cluster_and_split.py \
    --processed-dir data/processed \
    --stats-dir     data/stats \
    --work-dir      data/_cluster_work \
    --max-rna-len 4000

# keep FASTA + MMseqs2 intermediates for debugging
python src/step1_server/cluster_and_split.py ... --keep-work
```

The work-dir is wiped at the end unless `--keep-work` is given. MMseqs2 is
shelled out to via subprocess, so install it first (`conda install -c
bioconda mmseqs2`).

**Estimated wall clock**: ~1-3 min for FASTA export, ~5-15 min each for the
two MMseqs2 runs on a 16-thread box. Full pipeline 10-30 min.

**`splits.json` schema**:

```json
{
  "config": {
    "protein_min_seq_id": 0.3,
    "rna_min_seq_id": 0.8,
    "coverage": 0.8,
    "split_ratio": [0.8, 0.1, 0.1],
    "seed": 42,
    "max_rna_len": 0,
    "mmseqs_bin": "/opt/conda/envs/pocket/bin/mmseqs",
    "rna_sanitize_rule": "I->G, -dropped, else non-ACGU -> N"
  },
  "stats": { ... same numbers as cluster_stats.md ... },
  "cluster_index": {
    "protein": { "prot_000001": "1un6_B", "prot_000002": "2xxa_A", ... },
    "rna":     { "rna_000001":  "1un6_F", "rna_000002":  "2xxa_F", ... }
  },
  "samples": {
    "1un6_B_F": {
      "protein_cluster_id": "prot_000001",
      "rna_cluster_id":     "rna_000001",
      "split":              "train"
    },
    "3j92_x_5": {
      "protein_cluster_id": "unclusterable_X_only",
      "rna_cluster_id":     "rna_000123",
      "split":              null
    },
    ...
  }
}
```

---

### `apply_splits.py` (stage 1.5b)

Merges `splits.json` back into each sample JSON's `split_info` block:

```
sample["split_info"]["protein_cluster_id"] = "prot_000042"
sample["split_info"]["rna_cluster_id"]     = "rna_000017"
sample["split_info"]["split"]              = "train" | "val" | "test" | null
```

Pure Python, no MMseqs2 dependency. Idempotent (skips samples where
`split_info` already matches `splits.json`), atomic write, `--dry-run`
available. If `index.csv` has samples that `splits.json` doesn't know about,
it **fails loudly** — that's a config error and shouldn't be silently
rewritten to `unclusterable`; pass `--allow-missing` to downgrade to a
warning.

**Run**:

```bash
python src/step1_server/apply_splits.py --processed-dir data/processed

# dry run
python src/step1_server/apply_splits.py --processed-dir data/processed --dry-run

# force-rewrite every sample (e.g. after re-running clustering with new params)
python src/step1_server/apply_splits.py --processed-dir data/processed --force
```

**Estimated wall clock**: 2-5 min for 45,468 samples (mostly JSON parse/write,
already shown on the local box to do ~20k files/min). No reporting file —
the console output is the report.

---

## Filename encoding reminder

The sample JSONs on disk use a case-safe filename encoding, so a sample id
is not a filename: `3j46_y_1` lives on disk as `samples/3j46_-y_1.json` (the
rationale is case-insensitive filesystems). All server scripts here carry a
local copy of `sid_to_filename()` — kept in sync with
`src/step1_local/_common.py` — so they resolve filenames correctly on Linux
too.
