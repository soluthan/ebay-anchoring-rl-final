# Migration from the previous pipeline

The corrected pipeline is intentionally incompatible with old artifacts.

## Before running

Move or delete local `data/`, `models/`, and `outputs/` directories produced by
the previous implementation. In particular, do not reuse:

- `deal_classifier.ubj` trained on `status_id == 2`;
- row-level train/validation/test Parquet files;
- old CQL or Gaussian-PPO checkpoints;
- old OPE tables, dashboard figures, report PDF, or presentation numbers.

## Corrected run

```bash
DATA_DIR=. OUT_DIR=./data MODEL_DIR=./models OUTPUT_DIR=./outputs \
PPO_RUN_BOTH=1 python run_pipeline.py --phase all
```

The new classifier is named `opening_acceptance_classifier.ubj` so an old
status-2 model cannot be loaded accidentally.

## Required checks

```bash
python -m unittest discover -s tests -v
cat data/item_split_overlap.json
head data/acceptance_by_anchor_bin.csv
cat data/status_price_diagnostics.csv
```

All three item-overlap values must be zero. Only after these checks and the full
rerun should numerical results be copied into `report/technical_report.md` and
used to regenerate the PDF and presentation.
