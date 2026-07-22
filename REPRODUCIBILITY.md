# Reproducibility Notes

## Required semantic audit

Before accepting any model results, verify:

1. `preprocess_stats.json` records accepted statuses `[1, 9]` and split group
   `anon_item_id`.
2. `item_split_overlap.json` contains three zeros.
3. Every processed row has `offr_type_id == 0` and a unique
   `(anon_item_id, anon_byr_id)` pair.
4. Every `anchor_ratio` is in `(0.01, 1.0]`.
5. `savings_pct == opening_accepted * (1-anchor_ratio)`.
6. `acceptance_by_anchor_bin.csv` has plausible status-1/status-9 behavior.
7. In `status_price_diagnostics.csv`, statuses treated as accepted have
   `median_item_to_offer` near 1 and high `item_offer_within_1pct`.

Status trends are diagnostics; the official data dictionary/companion code is
the authority for status semantics.

## Full run

```bash
DATA_DIR=. OUT_DIR=./data MODEL_DIR=./models OUTPUT_DIR=./outputs \
PPO_RUN_BOTH=1 python run_pipeline.py --phase all
```

Run order is preprocessing, Phase 1, CQL, both PPO variants, optional OPE,
recommendations, then the dashboard. Each model phase uses seed 42. PPO resets
the seed for both variants to make the H3 comparison less sensitive to random
initialization.

## Evaluation

- Classifier: ROC AUC, Brier score, log loss, and reliability bins.
- H1: anchor-response curve and interior-optimum fraction.
- H2: fraction of recommended actions within train p5-p95 support; OPE is an
  optional diagnostic rather than causal evidence.
- Report CQL's `alpha` (default `5.0`) because the strength of conservatism
  directly affects the H2 result.
- H3: basic versus robust PPO simulator reward and relative shrinkage.

## Artifact incompatibility

All models, metrics, figures, report numbers, and recommendation outputs created
before the opening-event/status/group-split correction must be deleted or stored
outside the submission repository and regenerated. Old checkpoints are not
compatible with the Beta PPO architecture.

## Data boundary

No private eBay data, processed splits, model checkpoints, or generated output
tables are committed. The pipeline starts from a local merged
`clean_master_dataset.parquet`; raw CSV joining is intentionally outside scope.
