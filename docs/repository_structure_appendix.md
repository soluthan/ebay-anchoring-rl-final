# Repository Structure Appendix

## Research design

The repository implements a one-step, support-aware opening-offer study:

```text
state  -> first buyer-offer ratio -> immediate acceptance / zero
reward = opening_accepted * (1 - anchor_ratio)
```

The primary hypotheses are:

1. Model-estimated immediate expected savings is non-monotonic in the anchor
   and has an interior optimum for a meaningful share of listings.
2. Supervised greedy optimization selects less historically supported actions
   than one-step CQL.
3. PPO's simulator-only advantage shrinks under probability noise and support
   penalties.

## Shared constants

`project_constants.py` is the source of truth for:

- accepted statuses `(1, 9)`;
- first buyer-offer type `0`;
- action range `(0.01, 1.00]`;
- four state features;
- model/audit column names;
- the shared 50-point action grid.

`policy_utils.py` centralizes candidate-action scoring and the true supervised
greedy policy so Phase 1, OPE, and recommendations cannot silently diverge.

## Phase responsibilities

| Phase | Reads | Writes |
| --- | --- | --- |
| Preprocess | merged event/listing Parquet | grouped splits and semantic audits |
| Phase 1 | grouped splits | classifier, AUC/Brier/log loss, calibration, fixed/greedy metrics |
| Phase 2 | grouped splits and classifier | CQL checkpoint/scaler/history/metrics |
| Phase 3 | test states, classifier, scaler | basic and robust Beta-PPO artifacts |
| OPE | test rewards and learned policies | optional support/value diagnostics |
| Recommend | trained classifier/CQL and train support | listing-level policy table |
| Results | corrected metrics only | comparison CSV, H3 summary, dashboard |

## Non-features

`anon_item_id`, `anon_byr_id`, `anon_thread_id`, `offr_type_id`, timestamps,
raw status, and eventual success are used only to construct or audit the sample.
They are never policy-state inputs.

The default study scope is fashion only (`FILTER_FASHION=1`,
`meta_categ_id == 11450`). Status-price diagnostics empirically cross-check the
official accepted-status mapping before model training results are interpreted.

## Reproducibility boundary

Private data, processed Parquet files, model checkpoints, and generated outputs
remain ignored. Tracked report files must be regenerated after the corrected
full run; artifacts from the old status-2/row-split pipeline are invalid.
