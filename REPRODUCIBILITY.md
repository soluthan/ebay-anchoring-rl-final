# Reproducibility Notes

This project follows the course guidance that GitHub is part of scientific reproducibility, not just code storage.

## Version Control

- Stable code lives on `main`.
- The final submitted state is identified by the latest `course-submission-*`
  tag.
- This clean submission repository keeps curated milestone branches:
  - `feature/supervised-baseline`: preprocessing and supervised acceptance model.
  - `feature/offline-rl-cql`: offline RL, CQL, OPE diagnostics, and result tables.
  - `feature/ppo-sim`: faithful PPO simulator, recommendations, and final report integration.

## Data Boundary

The repository does not commit raw data, processed parquet splits, trained checkpoints, metrics, generated figures, or local environments. Those are intentionally ignored by `.gitignore`.

Expected local inputs and generated outputs:

- Input: `clean_master_dataset.parquet`
- Processed splits: `data/train.parquet`, `data/val.parquet`, `data/test.parquet`
- Split audit files: `data/split_summary.csv`, `data/preprocess_stats.json`
- Robustness split directories such as `data_time/` and `data_leaf/`
- Models and metrics: `models/`
- Smoke-test model directories such as `models_smoke/`
- Recommendation tables and figures: `outputs/`
- Smoke-test output directories such as `outputs_smoke/`
- Human-facing report and selected figures: `report/`

See `docs/data_schema.md` for the raw-to-engineered column mapping and the one-step MDP framing.

## Configuration

The default experiment settings are documented in `configs/default_experiment.json`. The current scripts read configuration through environment variables to keep the flat module layout simple.

Example full run:

```bash
DATA_DIR=. OUT_DIR=./data python data_preprocess.py
DATA_DIR=./data MODEL_DIR=./models python phase1_supervised.py
DATA_DIR=./data MODEL_DIR=./models python phase2_cql.py
DATA_DIR=./data MODEL_DIR=./models PPO_FAITHFUL=1 python phase3_ppo.py
MODEL_DIR=./models OUTPUT_DIR=./outputs python results.py
DATA_DIR=./data MODEL_DIR=./models OUTPUT_DIR=./outputs python ope.py
DATA_DIR=./data MODEL_DIR=./models OUTPUT_DIR=./outputs python recommend.py
```

With `PPO_FAITHFUL=1`, Phase 3 writes `ppo_*_faithful` artifacts. `results.py` prefers those faithful outputs when present and falls back to legacy untagged PPO outputs for older local runs.

Robustness split examples:

```bash
DATA_DIR=. OUT_DIR=./data_time SPLIT_MODE=temporal python data_preprocess.py
DATA_DIR=. OUT_DIR=./data_leaf SPLIT_MODE=leaf_holdout python data_preprocess.py
```

The preprocessing script fits seller-score normalization, seller-positive
imputation, and leaf-category vocabulary on train only. Temporal and
leaf-holdout splits are intended as robustness checks for chronological shift
and unseen fashion sub-categories; cross-meta-category generalization remains a
future-work extension.

`ope.py` estimates behavior propensities because logged randomized propensities
are unavailable. Its SNIPS/DR estimates, bootstrap confidence intervals,
effective sample sizes, and weight diagnostics should be read as support
mismatch evidence rather than causal online-lift estimates. If CQL artifacts are
present, the OPE table includes the CQL target policy automatically.

## Random Seeds

The scripts use seed `42` for data splitting, model sampling, CQL, PPO, and recommendation sampling. See `configs/default_experiment.json` for the seed map.

## Dependency Policy

The project keeps the provided `requirements.txt` as the dependency source of truth. For a formal archival submission, export exact local package versions from the environment used to produce the final results and include that as an appendix or separate lock file if the course permits it.

## Smoke Checks

The CI workflow runs lightweight checks that do not require the private dataset:

- Python bytecode compilation for the flat modules.
- Static tests for expected files, JSON config validity, report allowlist rules, and flat imports in `run_pipeline.py`.
