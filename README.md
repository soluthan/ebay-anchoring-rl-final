# Support-Aware Opening Offers for eBay Best Offer

This course project studies one narrow decision:

> Given listing and seller characteristics, which **first buyer-offer ratio**
> maximizes model-estimated immediate expected savings while remaining supported
> by historical behavior?

For state `s` and opening-offer ratio `a`:

```text
P(opening offer accepted | s, a) * (1 - a)
```

The source is observational. Results are policy diagnostics, not causal effects
of changing an offer and not estimates of buyer welfare, resale value, or the
value of later counteroffers.

## Correct observation unit

`data_preprocess.py` follows the official companion-code semantics:

- bargaining thread: `(anon_item_id, anon_byr_id)`;
- opening event: chronological event `order == 1` with `offr_type_id == 0`;
- immediate acceptance: opening-row `status_id` in `{1, 9}`;
- action support: `0.01 < offr_price / start_price_usd <= 1.00`;
- reward: `opening_accepted * (1 - anchor_ratio)`.

A countered opening receives zero immediate reward even if the buyer later buys
the item. `thread_eventual_accepted` is retained only for auditing. Random and
temporal splits assign complete `anon_item_id` groups, preventing listing leakage.

## Models

| Component | Role |
| --- | --- |
| Historical behavior | Observed immediate-outcome benchmark |
| Fixed anchor 0.70 | Rule-based baseline, not a greedy policy |
| Supervised greedy | Grid maximizer of the Phase-1 expected-savings surface |
| One-step CQL | Support-conservative offline policy; every row is terminal and `gamma=0` |
| PPO basic | Simulator-only optimizer using a bounded Beta policy |
| PPO robust | PPO with probability noise and penalties outside historical p5-p95 support |

The Phase-1 classifier reports AUC, Brier score, log loss, and a calibration
table. PPO results are never interpreted as live-marketplace lift.

The shared 67-point action grid contains the fixed 0.70 anchor exactly. This
prevents a tree-model discontinuity at a common offer ratio from making the
nominal grid-greedy policy score below the fixed baseline solely because its
grid missed 0.70.

## Features

The state remains deliberately compact:

```python
[
    "log_list_price",
    "seller_score_norm",
    "seller_pos_pct",
    "categ_id_clean",
]
```

`anchor_ratio` is the action, not a pre-decision state feature. Buyer/item IDs,
offer type, timestamp, and raw status are used only for extraction, splitting,
and audits.

By default `FILTER_FASHION=1` restricts the entire study to
`meta_categ_id == 11450`. Every reported result is therefore conditional on the
fashion sample. Set `FILTER_FASHION=0` only for a separately documented rerun.

## Repository layout

| File | Purpose |
| --- | --- |
| `data_preprocess.py` | Opening-offer extraction, reward, grouped splits, semantic audits |
| `phase1_supervised.py` | Acceptance model, calibration, fixed and true-greedy baselines |
| `phase2_cql.py` | One-step terminal CQL |
| `phase3_ppo.py` | Basic and robust simulator-only PPO with a Beta policy |
| `policy_utils.py` | Shared action grid, scoring, and supervised greedy policy |
| `ope.py` | Optional estimated-propensity support diagnostics |
| `recommend.py` | Batch greedy/CQL recommendation table |
| `recommend_one.py` | One-listing fixed/greedy/CQL menu |
| `results.py` | Evidence-separated comparison table and dashboard |
| `run_pipeline.py` | Flat-module orchestration |
| `MIGRATION.md` | Required cleanup and rerun steps for replacing the old pipeline |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place `clean_master_dataset.parquet` in the project root. It must include at
least `anon_item_id`, `anon_byr_id`, `offr_type_id`, `src_cre_date`,
`status_id`, `offr_price`, `start_price_usd`, and `meta_categ_id`.

## Run

Complete corrected pipeline:

```bash
DATA_DIR=. OUT_DIR=./data MODEL_DIR=./models OUTPUT_DIR=./outputs \
PPO_RUN_BOTH=1 python run_pipeline.py --phase all
```

The raw CSV merge is intentionally outside this command; the pipeline begins
from the merged Parquet.

Small smoke run:

```bash
DATA_DIR=. OUT_DIR=./data_smoke MODEL_DIR=./models_smoke OUTPUT_DIR=./outputs_smoke \
PREPROCESS_MAX_ROWS=50000 PHASE1_MAX_ROWS=20000 XGB_N_ESTIMATORS=40 \
CQL_EPOCHS=1 CQL_MAX_ROWS=20000 PPO_STEPS=2048 PPO_ROLLOUT=512 \
PPO_EVAL_EPISODES=500 OPE_BOOTSTRAP=10 OPE_MAX_ROWS=10000 \
python run_pipeline.py --phase all
```

One-listing recommendation:

```bash
DATA_DIR=./data MODEL_DIR=./models python recommend_one.py \
  --price 120 --seller_score 4500 --pos_pct 99.2 --leaf_category 12345
```

`recommend_one.py` needs the trained classifier, optional CQL artifacts, and
`data/preprocess_stats.json`. The latter persists the train-fitted category,
normalization, and p5-p95 support metadata, so row-level training data are not
required for one-listing inference.

## Full-data result

On the corrected fashion sample (5,705,893 opening offers), the classifier
reached test AUC 0.8339 and Brier score 0.1539. Observed historical immediate
expected savings was 9.73%. Phase-1 estimates were 12.68% for fixed 0.70,
13.71% for supervised greedy, and 12.19% for CQL. Greedy kept 90.99% of actions
inside train p5-p95 support versus 99.92% for CQL. Robust PPO simulator reward
was 4.99% below basic PPO. See `report/technical_report.md` for evidence types,
OPE diagnostics, and limitations.

## Interpretation boundaries

- Low offers that trigger a counteroffer are scored zero, so the objective may
  understate their option value and favor higher immediate-close offers.
- Buyers choose anchors endogenously. Unobserved item quality, perceived
  overpricing, and buyer information can confound the acceptance surface.
- Historical, model-estimated, OPE-diagnostic, and simulator-only results are
  different evidence types and must not be ranked as if they were interchangeable.
- Old checkpoints and report numbers created with `status_id == 2`, row-level
  splits, or actions above listing price are incompatible and must be regenerated.
