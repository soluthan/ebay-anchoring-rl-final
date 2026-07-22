# Data Schema and Estimand

## Raw event columns

| Column | Use |
| --- | --- |
| `anon_item_id` | Listing group and leakage-safe split key |
| `anon_byr_id` | Combined with item ID to define a bargaining thread |
| `anon_thread_id` | Audit identifier when present |
| `src_cre_date` or `src_cre_dt` | Chronological event ordering |
| `offr_type_id` | `0` identifies a first buyer-offer event |
| `status_id` | Opening-event status; `1` and `9` are accepted |
| `offr_price` | Buyer offer amount |
| `start_price_usd` | Listing-price denominator |
| `meta_categ_id` | Fashion-scope filter |
| `anon_leaf_categ_id` | Compact product-category feature |
| `fdbk_score_src` | Seller feedback score |
| `fdbk_pstv_src` | Seller positive-feedback percentage |

`item_price` is retained for status adjudication. For accepted event codes it
should coincide with `offr_price`; this is checked by
`status_price_diagnostics.csv`. It is not a model feature or reward input.

## Opening-offer extraction

1. Group events by `(anon_item_id, anon_byr_id)`.
2. Sort by event timestamp.
3. Retain the first event only.
4. Keep it only when `offr_type_id == 0`.
5. Set `opening_accepted = 1[status_id in {1, 9}]`.
6. Keep `0.01 < anchor_ratio <= 1.0`.

This produces one contextual-bandit observation per buyer-listing thread.

## Engineered columns

| Column | Meaning |
| --- | --- |
| `log_list_price` | `log(start_price_usd + 1)` |
| `seller_score_norm` | Seller score divided by train-set p99 and clipped to `[0,1]` |
| `seller_pos_pct` | Train-median-filled seller positive percentage |
| `categ_id_clean` | Train-set top leaf category or `-1` |
| `anchor_ratio` | `offr_price / start_price_usd` |
| `opening_accepted` | Immediate opening-event acceptance label |
| `savings_pct` | `opening_accepted * (1-anchor_ratio)` |
| `thread_eventual_accepted` | Audit-only indicator that any event later succeeded |
| `thread_event_count` | Audit-only number of events in the thread |
| `item_price` | Audit-only status-adjudication price |

Preprocessing identifiers and audit fields are not model state features.

## Outputs

- `train.parquet`, `val.parquet`, `test.parquet`
- `preprocess_stats.json` (including train-fitted normalization, category, and
  anchor p5-p95 support metadata)
- `split_summary.csv`
- `item_split_overlap.json` (all overlaps must equal zero)
- `opening_status_counts.csv`
- `acceptance_by_anchor_bin.csv`
- `status_price_diagnostics.csv`

The binned status curve is a semantic diagnostic, not proof of causality or a
requirement that raw acceptance be perfectly monotone under confounding.

The default `FILTER_FASHION=1` filter restricts all rows to
`meta_categ_id == 11450`; conclusions are conditional on that fashion category.

## Evidence boundary

The data are observational and contain no randomized propensities. Phase-1 and
CQL values are model estimates, OPE remains a support diagnostic, and PPO values
are simulator-only. The reward ignores the option value of a seller counteroffer.
