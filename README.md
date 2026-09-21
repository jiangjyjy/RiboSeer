# RiboSeer

Code release for **Agentic Orchestration of Heterogeneous Predictors for RNA–Protein
Binding Pocket Discovery**.

RiboSeer predicts RNA–protein binding pockets by coordinating a library of 15
open-source structure-prediction, pocket-detection, binding-residue and docking tools
under an LLM planner. Rather than training another end-to-end model, it learns *which*
predictors to trust for a given class of target and fuses their outputs into a single
per-residue contact prediction for both the protein and the RNA.

This repository is an **anonymized release** accompanying a double-blind submission. It
contains the code needed to reproduce the results reported in the paper.

---

## Overview

Given an RNA sequence and a protein sequence — optionally with experimentally resolved
coordinates — RiboSeer predicts a per-residue binding probability
`F : (r, p, X_r, X_p) → y ∈ [0,1]^{L_p}`, where `y_i` is the probability that protein
residue `i` contacts the RNA. Ground-truth labels use a heavy-atom distance cutoff of
`d_cut = 4.5 Å`.

The pipeline is composed of seven named modules. In the configuration reported in the
paper, two of them (SCOPE and MAESTRO) call an LLM — **GLM 5.1** — and the remaining
five are ordinary Python. Every LLM decision is schema-validated before it is used.

| Module | § | Role | Code |
|--------|---|------|------|
| **GENESIS** | 3.4 | Data curation: enumerates protein–RNA chain pairs, computes heavy-atom contact sets, applies the length / contact / resolution / quality-tier cascade | `src/step1_local/`, `src/step1_server/` |
| **SCOPE** | 3.5 | LLM-driven target profiling: serializes a per-target feature dictionary and queries the LLM for a target profile `T_prof` and a difficulty label `j` | `src/step2_target_char/` |
| **MAESTRO** | 3.6 | LLM-driven tool selection: given the profile and a UCB-style weight tensor `W ∈ R^{K×J}`, proposes an ordered tool plan | `src/step3_tool_selection/` |
| **RELAY** | 3.7 | Concurrent tool execution: dispatches all selected adapters in parallel, each within its own timeout budget, normalizing native output into a shared prediction record | `src/step4_tool_adapters/` |
| **HARMONY** | 3.8 | Multi-feature learned fusion: a LightGBM model over a four-group engineered feature space built from every active tool | `src/step5_fusion/` |
| **VERDICT** | 3.9 | Quality assessment: scores a fused prediction on structural plausibility, physicochemical complementarity, evolutionary conservation, cross-tool consensus and canonical-motif overlap, and drives the refinement decision | `src/step6_pocket_qa/`, `src/step7_iteration/` |
| **MEMORY** | 3.10 | Weight-tensor update: exponential moving average `W^(n)_{k,j} ← (1−η)W^(n−1)_{k,j} + η·r_{n,k,j}`, fed back to MAESTRO on later samples | `src/step8_weight_update/` |

The tool library, the MANDATORY core and the per-tool timeouts all live in one place,
[`src/step3_tool_selection/tool_registry.py`](src/step3_tool_selection/tool_registry.py) —
the source of truth every other component reads from.

---

## The MAESTRO tool library (paper Table 1)

Fifteen tools in four categories. `✓M` marks the **7-tool MANDATORY core**, the set
chosen by forward greedy search on the training set, which MAESTRO force-includes in
every sample's plan (paper Table 11). The timeout column is the per-sample budget RELAY
enforces.

| Tool | Cat. | ✓M | Run by | Timeout (s) |
|------|------|----|--------|-------------|
| Boltz-2 | A | ✓M | RELAY (conda env `boltz`) | 900 |
| Chai-1 | A | ✓M | RELAY (conda env `chai1`) | 900 |
| RoseTTAFold2NA | A | | RELAY (conda env `RF2NA2`) | 600 |
| RoseTTAFold-All-Atom | A | | RELAY (conda env `RFAA`) | 1200 |
| AlphaFold 3 | A | | **web submission** (AlphaFold Server) | 900 |
| P2Rank | B | | RELAY (installed binary) | 60 |
| Fpocket | B | | RELAY (installed binary) | 60 |
| DeepPocket | B | ✓M | RELAY (conda env `DeepPocket`, GPU) | 300 |
| EquiPNAS | C | ✓M | RELAY (conda env `EquiPNAS`, GPU) | 300 |
| NucleicNet | C | ✓M | RELAY (local checkout, in-process, GPU) | 600 |
| GraphBind | C | | RELAY (conda env `GraphBind`) | 600 |
| RNABindRPlus | C | ✓M | **web submission** (results by email) | 300 |
| BindUP | C | | **web submission** (batch form) | 300 |
| HDOCK | D | ✓M | RELAY (HDOCKlite binaries, CPU) | 600 |
| HADDOCK 3 | D | | RELAY (conda env `haddock3`) | 600 |

RELAY dispatches the cheapest category first (`C → B → D → A`) so an early stop is
possible. Tools that are not installed, fail, or exceed their budget are zero-filled;
HARMONY handles the resulting missing feature columns natively through missing-value
splits, so the pipeline runs with whatever subset of the library a given host provides.

### Tools that require manual web submission

Four tools are not distributed for local execution. Their adapters run nothing — they
write the submission payload for you, then read back the results you download:

| Tool | Submit at | Results arrive | Put them under |
|------|-----------|----------------|----------------|
| AlphaFold 3 | AlphaFold Server (https://alphafoldserver.com) | job-folder download, 30 jobs/day | `data/external/af3/<sample_id>/` |
| RNABindRPlus | http://ailab-projects2.ist.psu.edu/RNABindRPlus/ | by email | `data/external/rnabindrplus/` |
| BindUP | https://bindup.technion.ac.il/ (batch form takes a PDB-ID list) | job page, or by email | `data/external/bindup/` |
| GraphBind | http://www.csbio.sjtu.edu.cn/bioinf/BindWeb/ | result page (the local CLI route is preferred) | `data/external/graphbind/` |

Run the pipeline with `--tools local` to skip these four; if you run with `all`, a tool
that produced no result is treated like any other failure and HARMONY zero-fills it.
`scripts/riboseer/` holds the helpers for the manual flow — `make_af3_train_inputs.py`,
`reorganize_af3_jobs.py`, `submit_rnabindrplus.py`, `submit_bindup_batch.py`,
`submit_bindup_single.py`, `submit_graphbind_batch.py` — and the per-tool parsers live
in `src/step4_tool_adapters/external/`.

---

## Repository structure

```
RiboSeer/
├── run_pipeline.py            # one-command entry point: steps 2 → 8 over a sample list
├── src/
│   ├── step1_local/           # GENESIS — pure-Python curation, runs anywhere
│   ├── step1_server/          # GENESIS — needs Linux-only tooling (MMseqs2, ViennaRNA, CD-HIT)
│   ├── step2_target_char/     # SCOPE    — target profiling + the LLM client
│   ├── step3_tool_selection/  # MAESTRO  — tool registry, selection, weight tensor
│   ├── step4_tool_adapters/   # RELAY    — one adapter per predictor
│   │   ├── adapters/          #   the 15 tool adapters (base_adapter defines the interface)
│   │   ├── external/          #   parsers + submission-payload builders per tool
│   │   └── tool_io.py         #   shared sample/structure resolution
│   ├── step5_fusion/          # HARMONY  — the learned fusion
│   │   ├── features_15tool.py #   154-D per-residue feature builder (paper §3.8)
│   │   ├── lightgbm_fusion.py #   training + per-sample inference
│   │   ├── prediction_io.py   #   prediction / model persistence
│   │   └── metrics.py         #   per-sample Pearson / Spearman / R²
│   ├── step6_pocket_qa/       # VERDICT  — scorers and metrics
│   ├── step7_iteration/       # VERDICT  — refinement loop (+ polish_ops)
│   └── step8_weight_update/   # MEMORY   — EMA update + meta-correction
├── scripts/
│   ├── tables/                # one script per paper table: tableNN_<name>.py
│   ├── riboseer/              # experiment drivers: cached LLM outputs, case studies, helpers
│   └── *.py                   # corpus preparation and fusion-model training
├── tests/                     # pytest suite (see "Tests")
├── configs/                   # one YAML per stage; no paths hard-coded in source
└── docs/figures/              # paper figures
```

---

## Setup

### Environment

Conda (recommended):

```bash
conda env create -f environment.yml
conda activate riboseer
```

or pip:

```bash
python -m pip install -r requirements.txt
```

Python 3.11 or newer is required.

### Prediction tools

RELAY invokes each predictor as an external process, so the tools are **not** vendored
here. Install the subset you need from each project's own repository, then point the
corresponding entry in [`configs/step4_config.yaml`](configs/step4_config.yaml) at its
installation directory and conda environment. Each adapter calls its tool with an
explicit argument list and parses the native output into the shared record
(`tool_name`, `binding_protein_residues`, `binding_rna_nucleotides`,
`per_residue_confidence`, `geometry`).

One caveat worth knowing: the NucleicNet adapter imports the tool's Python API, which
resolves its weights through paths relative to its own `Notebooks/` directory, so that
adapter changes the process working directory while it runs. Set
`execution.parallel_workers: 1` in the step-4 config when NucleicNet is part of the plan.

### Model access

SCOPE and MAESTRO call an OpenAI-compatible chat-completions endpoint. The client is
provider-agnostic: endpoint, model name and sampling parameters live in
[`configs/step2_config.yaml`](configs/step2_config.yaml), and the credential is read
from the environment.

```bash
export LLM_API_KEY="your-key-here"
```

The key is never read from a config file and must not be committed — `.env` is
gitignored and `.env.example` is the template. The paper's reasoner is **GLM 5.1**; set
`api.model` in the stage configs to the identifier your provider expects.

---

## Data

Datasets are **not** included and are not redistributed. GENESIS builds the corpus from
publicly available PDB/mmCIF entries; the curation cascade (paper Table 2) is:

1. Enumerate all `(protein, RNA)` chain pairs per entry.
2. Compute the heavy-atom contact set at `d_cut = 4.5 Å`.
3. Apply the length, contact-count, resolution and quality-tier filters — retaining
   **2,018 pairs**, grouping into **332 protein sequence clusters** at 30% identity.
4. Split by sequence cluster, so that no cluster spans the train/test boundary. The
   reported residue-level results use a **107-sample structurally disjoint test split**.

To rebuild the corpus, edit [`configs/paths.yaml`](configs/paths.yaml) to point at your
local copy of the structures and run the GENESIS scripts in order:

```bash
python src/step1_local/scan_dataset.py      --raw-dir /path/to/structures --out-dir data/stats
python src/step1_local/extract_pairs.py     --raw-dir /path/to/structures \
                                            --processed-dir data/processed --stats-dir data/stats
python src/step1_local/protein_features.py  --raw-dir /path/to/structures \
                                            --processed-dir data/processed --stats-dir data/stats
python src/step1_local/rna_features_basic.py --processed-dir data/processed --stats-dir data/stats
python src/step1_local/validate.py          --processed-dir data/processed --stats-dir data/stats
```

Two curation steps need Linux-only tooling and live in `src/step1_server/`:
`rna_secondary_structure.py` (ViennaRNA) and `cluster_and_split.py` (MMseqs2, plus
CD-HIT for the double-clustering variant). Until they run, the corresponding fields
stay `null` in the sample JSON.

> **Note on sample identifiers.** Identifiers are case-folded when files are written, so
> a naive `sample_id + ".json"` path is wrong on a case-sensitive filesystem. Always
> resolve disk paths through `load_sample_json()` in
> [`src/step4_tool_adapters/tool_io.py`](src/step4_tool_adapters/tool_io.py).

---

## Usage

### Whole pipeline, one command

```bash
export LLM_API_KEY="..."
export PYTHONPATH=src

python run_pipeline.py \
    --processed-dir data/processed \
    --sample-list   data/splits/test.txt \
    --output-dir    data/run/test \
    --weight-tensor data/W.json \
    --tools local
```

This runs SCOPE → MAESTRO → RELAY → HARMONY → VERDICT → MEMORY for every sample: one
JSONL per stage under `<output-dir>/step<N>/`, one summary line per sample under
`<output-dir>/summary/`. `--resume` skips samples that already finished;
`--no-weight-update` freezes MEMORY (evaluation mode); `--no-llm` replaces every LLM
call with its deterministic fallback, which is enough to smoke-test the plumbing without
credentials. `--tools` accepts `all`, `local`, or an explicit comma-separated id list.

HARMONY is the LightGBM model reported in the paper. Point it at a fitted model, or let
it fit one on the training split and cache it:

```bash
# reuse a saved model
python run_pipeline.py ... --fusion lightgbm --fusion-model-dir data/harmony_model

# fit on the training split (cached in <output-dir>/harmony_model for later runs)
python run_pipeline.py ... --fusion lightgbm     --fusion-train-step4-dir data/batch_train/step4     --fusion-train-list      data/splits/train.txt
```

`--fusion auto` (the default) uses a model when one is available and otherwise falls back
to the prototype noisy-OR fusion; `--fusion noisy-or` forces the prototype.

### Stage by stage

Each stage also runs on its own, consuming the previous stages' outputs:

```bash
# SCOPE — target profiling
python -m step2_target_char.run --processed-dir data/processed --config configs/step2_config.yaml

# MAESTRO — tool selection (optionally chains SCOPE via --run-step2)
python -m step3_tool_selection.run --processed-dir data/processed --config configs/step3_config.yaml

# RELAY — runs one sample at a time; --tools takes a space-separated list
python -m step4_tool_adapters.run --processed-dir data/processed --sample-id <id> \
    --tools boltz2 chai1 p2rank --config configs/step4_config.yaml

# HARMONY — fusion
python -m step5_fusion.run --processed-dir data/processed --config configs/step5_config.yaml

# VERDICT — scoring, then the refinement decision
python -m step6_pocket_qa.run --processed-dir data/processed \
    --step4-output data/step4_outputs --step5-output data/step5_outputs \
    --config configs/step6_config.yaml
python -m step7_iteration.run --processed-dir data/processed \
    --step4-output data/step4_outputs --step5-output data/step5_outputs \
    --step6-output data/step6_outputs --config configs/step7_config.yaml

# MEMORY — weight-tensor update
python -m step8_weight_update.run --processed-dir data/processed \
    --step4-output data/step4_outputs --step5-output data/step5_outputs \
    --step6-output data/step6_outputs --step7-output data/step7_outputs \
    --weight-tensor data/W.json --config configs/step8_config.yaml
```

`--help` on any stage prints its full argument set.

---

## Reproduction

Every script that produces a paper table is named after it: `scripts/tables/tableNN_*`
holds paper Table NN. (Table 1 is the tool registry itself, Table 2 is assembled from the
GENESIS reports below, and Table 3 is descriptive.)

### Main results

| Paper | Contents | Script |
|-------|----------|--------|
| Table 1 | MAESTRO tool library | *no generator* — the data is [`src/step3_tool_selection/tool_registry.py`](src/step3_tool_selection/tool_registry.py); the MANDATORY core was found with `scripts/riboseer/search_best_tool_subset.py` |
| Table 2 | GENESIS curation cascade | `scripts/filter_quality.py` (survivors and per-filter drop reasons), `scripts/extract_cluster_representatives.py`, `scripts/compute_tmscore_matrix.py` + `scripts/spectral_split.py` (the split) |
| Table 3 | Full baseline catalogue | *no generator* — descriptive |
| Table 4 | Main residue-level results | `scripts/tables/table04_main_results.py` (per-tool rows) + `scripts/tables/table04_naive_ensembles.py` (naive-ensemble rows) |
| Table 5 | Pocket-level geometric metrics | `scripts/tables/table05_pocket_geometry.py` |
| Table 6 | Complex-level structural metrics | `scripts/tables/table06_complex_quality.py` |

### Ablations

| Paper | Contents | Script |
|-------|----------|--------|
| Table 7 | HARMONY fusion-method ablation | `scripts/tables/table07_fusion_method.py` |
| Table 8 | Feature-group ablation | `scripts/tables/table08_feature_groups.py` |
| Table 9 | LLM module ablation (SCOPE × MAESTRO) | `scripts/tables/table09_llm_modules.py` |
| Table 10 | Leave-one-out tool ablation | `scripts/tables/table10_leave_one_out.py` |
| Table 11 | MANDATORY_TOOLS guardrail effect | `scripts/tables/table11_mandatory.py`; runtime column: `scripts/tables/table11_runtime.py` |
| Table 12 | LLM backbone ablation | `scripts/tables/table12_backbone.py` |
| Table 13 | Prompt-engineering ablation | `scripts/tables/table13_prompt.py` |
| Table 14 | MEMORY EMA learning-rate ablation | `scripts/tables/table14_memory_eta.py` |
| Table 19 | Sensitivity to the ground-truth cutoff `d_cut` | `scripts/tables/table19_distance_cutoff.py` |

### Stratified analyses

| Paper | Contents | Script |
|-------|----------|--------|
| Table 15 | Stratified by RNA length | `scripts/tables/table15_rna_length.py` |
| Table 16 | Stratified by protein domain family | `scripts/tables/table16_protein_family.py` |
| Table 17 | Stratified by RNA structural context | `scripts/tables/table17_rna_context.py` |
| Table 18 | Stratified by SCOPE-assigned difficulty | `scripts/tables/table18_difficulty.py` |

### Additional analyses

| Paper | Contents | Script |
|-------|----------|--------|
| Table 20 | Top-k precision / recall / F1 / MCC | `scripts/tables/table20_topk.py` |
| Table 21 | Per-sample AUROC and AUPRC | `scripts/tables/table21_auroc_auprc.py` |
| Table 22 | Robustness across train/test partitions | `scripts/tables/table22_split_robustness.py` |
| Table 23 | Variance across five random seeds | `scripts/tables/table23_seed_variance.py` |
| Table 24 | Per-sample Pearson R distribution | `scripts/tables/table24_distribution.py` |
| Table 25 | Wall-clock cost and LLM API usage per module | `scripts/tables/table25_cost.py` |
| Table 26 | Top-10 HARMONY feature importances | `scripts/tables/table26_feature_importance.py` |
| Table 27 | Failure-mode breakdown | `scripts/tables/table27_failure_modes.py` |

### Order of operations

Most table scripts do **not** train a model themselves — they consume frozen
intermediate artefacts. A full reproduction therefore runs in this order:

1. **GENESIS** — build the corpus (see [Data](#data)), producing the frozen split lists
   under `splits_tmscore_035/`.
2. **Frozen LLM outputs** — `scripts/riboseer/generate_scope_profiles.py`,
   `generate_maestro_selections.py` and `generate_polish_actions.py` cache the
   SCOPE / MAESTRO / VERDICT decisions. Every ablation reads these rather than
   re-querying the model, which is what makes the ablations cheap and deterministic.
3. **RELAY outputs** — per-tool predictions for the train and test splits. The batch
   drivers in `scripts/riboseer/` (`run_deeppocket_batch.py`, `run_nucleicnet_batch.py`,
   `run_graphbind_batch.py`, `run_hdock_batch.py`, …) write the same step-4 record format
   as RELAY; `merge_external_step4.py` folds results produced elsewhere into the
   canonical step-4 directory.
4. **HARMONY** — `scripts/riboseer/generate_fullsystem_predictions.py` trains the
   full-system LightGBM and writes the headline predictions that Tables 4–6 and 20–27
   consume. Tables 7, 8, 10, 11 and 22–23 retrain per row/split themselves via
   `scripts/riboseer/retrain_eval_common.py`.
5. **Tables** — the scripts listed above.

Two caveats before you run anything:

- The headline rows read the full-system predictions through
  `--predictions-dir <fullsystem_predictions>`; see the usage block in
  `scripts/riboseer/generate_fullsystem_predictions.py`.
- Tables 12 and 13 require **live LLM API calls** (three alternative backbones, and four
  fresh prompt configurations — roughly 3,200 calls each). Every other table runs from
  cached artefacts.

Run `--help` on any script for its full argument set; argument names are consistent
across scripts (`--step4-dir`, `--processed-dir`, `--sample-list`, `--predictions-dir`).

### Figures

| Figure | Contents | File |
|--------|----------|------|
| Architecture | The seven-module pipeline: offline curation, then (a) SCOPE profiling, (b) MAESTRO selection, (c) RELAY parallel dispatch, HARMONY fusion, VERDICT scoring and the MEMORY feedback loop | [`docs/figures/pipeline.pdf`](docs/figures/pipeline.pdf) |
| Case study — 3X1L | Ground truth vs. RiboSeer (Pearson r = 0.962) vs. best baseline, Boltz-2 (r = 0.737) | [`docs/figures/case_study_3x1l.png`](docs/figures/case_study_3x1l.png) |
| Case study — 4WKR | Ground truth vs. RiboSeer (Pearson r = 0.919) vs. best baseline, HDOCK (r = 0.678) | [`docs/figures/case_study_4wkr.png`](docs/figures/case_study_4wkr.png) |

Each case-study panel shows the same pocket in three views — ground truth, RiboSeer, and
the strongest single-tool baseline for that sample — with per-residue confidence in
parentheses and hydrogen bonds marked. The cases are selected by
`scripts/riboseer/find_best_cases.py` and rendered by
`scripts/riboseer/extract_case_studies.py`.

---

## Tests

```bash
python -m pytest tests/ -q
```

The suite is split into two kinds of test:

- **Mock tests** (`test_*.py`, `test_*_mock.py`) — the bulk of the suite. No network, no
  API key, no deployed tools; the LLM stages run against recorded fixtures and the
  adapters against captured tool output.
- **Live tests** (`test_*_real.py`) — argparse scripts rather than pytest cases, so they
  are not collected. Each is meant to be run by hand on a machine with the live endpoint
  or the deployed tool, and prints what it observed so an unexpected output format can
  be diagnosed.

Run only the mock suite:

```bash
python -m pytest tests/ -q -k "not real and not e2e"
```

---

## Citation

This is an anonymous release under double-blind review; author and venue information is
withheld. Please cite the paper once it is published.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

The prediction tools invoked by the adapters are separate works under their own
licenses and are not redistributed here; check each tool's terms before use.
