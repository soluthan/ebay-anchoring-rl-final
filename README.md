# Offline and Simulated Online RL for eBay Best Offer Anchoring

Offline and simulated online reinforcement learning project for studying buyer anchoring in eBay Best Offer negotiations.

The project models a single-step bargaining MDP where the state describes a listing and seller, the action is the buyer's opening offer ratio, and the reward is realized savings for accepted offers. If an offer is accepted, savings are mechanically `1 - anchor_ratio`; model-estimated expected savings are therefore `P(accept | state, anchor) * (1 - anchor)`. The project compares historical buyer behavior with supervised baselines, Conservative Q-Learning (CQL), and PPO in a simulator learned from the Phase 1 acceptance model.

Important evidence boundary: historical rows are observed logged outcomes, supervised/CQL rows are Phase-1 model estimates, and PPO rows are simulator-only estimates. Simulator-predicted deal rates are not directly comparable to real observed deal rates, and no simulator result should be read as evidence of live marketplace lift.

## Repository Layout

The code intentionally uses a flat module layout so `run_pipeline.py` can import the phase modules directly.

| File | Purpose |
| --- | --- |
| `data_preprocess.py` | Builds train/validation/test parquet splits, with random, temporal, and leaf-holdout robustness modes. |
| `project_constants.py` | Shared schema, seed, action-bound, and feature constants for the flat module layout. |
| `phase1_supervised.py` | Trains the XGBoost deal-probability model used by the baselines and simulator. |
| `phase2_cql.py` | Trains a PyTorch CQL policy on the offline one-step MDP. |
| `phase3_ppo.py` | Trains PPO inside a Phase-1 model-based simulator, with a faithful-support variant. |
| `ope.py` | Runs estimated-propensity OPE diagnostics for historical, greedy, CQL, and external target policies. |
| `recommend.py` | Produces batch offer recommendations and support diagnostics. |
| `recommend_one.py` | Produces an offer menu for a single listing. |
| `results.py` | Builds comparison tables and result figures from generated model artifacts. |
| `run_pipeline.py` | Runs preprocessing, Phase 1, Phase 2, Phase 3, and results in sequence. |
| `configs/default_experiment.json` | Documents default paths, seeds, hyperparameters, outputs, and run order; scripts still read environment variables. |
| `docs/data_schema.md` | Documents the raw inputs, engineered MDP columns, and generated artifacts. |
| `docs/repository_structure_appendix.md` | Explains the folder structure, module responsibilities, core constants, evidence types, and Git artifact boundary. |
| `REPRODUCIBILITY.md` | Documents data boundaries, configuration, seeds, and submission tagging. |
| `tests/test_static_repo.py` | Lightweight static checks that can run without the private dataset. |

## Data and Artifact Policy

Raw data, processed parquet files, trained models, generated figures, and local virtual environments are intentionally not committed. The repository expects these to live locally in ignored paths such as `data/`, `models/`, `outputs/`, and `report/`.

Ignored examples include:

- `*.csv`
- `*.parquet`
- `*.pt`
- `*.pkl`
- `*.ubj`
- `data/`
- `data_*/`
- `models/`
- `models_*/`
- `outputs/`
- `outputs_*/`
- `.venv/`

The `report/` directory is reserved for final submission-facing deliverables. Working literature reviews, audit notes, model-comparison memos, and other private process documents should stay local and untracked. Datasets and model artifacts remain ignored.

## Reproducibility

The default experiment settings are recorded in `configs/default_experiment.json`, including seeds, expected paths, key hyperparameters, generated outputs, and the command order used to reproduce the pipeline. This file is documentation, not an automatically consumed runtime config; the scripts use environment variables so individual phases can be run independently.

See `REPRODUCIBILITY.md` for the full reproducibility checklist. The repository also includes a GitHub Actions smoke test that compiles the flat Python modules and validates static project structure without requiring the private dataset.

## Setup

Create and activate a virtual environment, then install the existing dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For exact reproduction of the final clean smoke-test environment, use:

```bash
pip install -r requirements-lock.txt
```

No dataset is included in the repository. Place the merged source file at `clean_master_dataset.parquet` or point `DATA_DIR` to the directory that contains it.

## Running the Pipeline

Build train/validation/test splits:

```bash
DATA_DIR=. OUT_DIR=./data python data_preprocess.py
```

Build robustness splits:

```bash
DATA_DIR=. OUT_DIR=./data_time SPLIT_MODE=temporal python data_preprocess.py
DATA_DIR=. OUT_DIR=./data_leaf SPLIT_MODE=leaf_holdout python data_preprocess.py
```

`data_preprocess.py` fits seller-score normalization, seller-positive imputation,
and the top leaf-category vocabulary on the training split only, then applies
those statistics to validation/test. Each run writes `split_summary.csv` and
`preprocess_stats.json` beside the generated parquet files.

Run all phases:

```bash
python run_pipeline.py
```

By default, Phase 3 writes faithful-simulator PPO artifacts such as `ppo_metrics_faithful.json`. `results.py` prefers those faithful artifacts when available and falls back to legacy untagged PPO outputs for older local runs.

Run a small smoke pipeline without touching production artifacts:

```bash
DATA_DIR=. OUT_DIR=./data_smoke MODEL_DIR=./models_smoke OUTPUT_DIR=./outputs_smoke \
PREPROCESS_MAX_ROWS=50000 PHASE1_MAX_ROWS=20000 XGB_N_ESTIMATORS=40 \
CQL_EPOCHS=1 CQL_MAX_ROWS=20000 PPO_STEPS=4096 PPO_ROLLOUT=512 PPO_EVAL_EPISODES=500 \
OPE_BOOTSTRAP=20 OPE_MAX_ROWS=20000 OPE_BEHAVIOR_MAX_ROWS=20000 \
python run_pipeline.py --phase all
```

Run one phase at a time:

```bash
python run_pipeline.py --phase prep
python run_pipeline.py --phase 1
python run_pipeline.py --phase 2
python run_pipeline.py --phase 3
```

Generate recommendations after training:

```bash
python recommend.py
python recommend_one.py --price 120 --seller_score 4500 --pos_pct 99.2
```

Run off-policy evaluation diagnostics after Phase 1:

```bash
DATA_DIR=./data MODEL_DIR=./models OUTPUT_DIR=./outputs python ope.py
```

The OPE script estimates behavior propensities from the observed data, then
reports SNIPS, DR-style estimates, bootstrap confidence intervals, effective
sample size, clipped/unclipped weights, and kernel-bandwidth sensitivity. When
`models/cql_best.pt` and `models/cql_scaler.pkl` exist, it also evaluates the
CQL target policy by default. These numbers are diagnostics for support
mismatch, not causal evidence of live marketplace lift.

For PPO support-edge checks, `PPO_FAITHFUL_P5=<anchor>` overrides the default
historical 5th-percentile support threshold. Use this only for smoke/sensitivity
investigations; do not treat short-run threshold-sweep numbers as findings.

## Branch Workflow

The stable branch is `main`. This clean submission repository uses curated
milestone branches for the course code-management trail, and the final submitted
state is tagged for reproducibility:

- `feature/supervised-baseline`: preprocessing and supervised acceptance model.
- `feature/offline-rl-cql`: offline RL, CQL, OPE diagnostics, and result tables.
- `feature/ppo-sim`: faithful PPO simulator, recommendations, and final report integration.
