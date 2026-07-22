# Support-Aware Optimization of Buyer Opening Offers

## Final technical report

**Nathan Nozik**<br>
Advanced Seminar (WIHN0043SE): Real-World Applications of Deep Learning and
Reinforcement Learning<br>
TUM School of Management | Chair of Business Analytics<br>
Heilbronn, 31.07.2026

### Research question

How can a buyer's opening-offer ratio in eBay Best Offer negotiations be chosen
to maximize model-estimated immediate expected savings while remaining within
the support of historically observed offers?

For listing and seller state `s` and opening-offer ratio `a`, the objective is:

```text
P(opening offer immediately accepted | s, a) * (1 - a)
```

The study is a one-step contextual-bandit analysis. It does not estimate the
causal psychological effect of anchoring, buyer welfare, resale profit, or the
value of later counteroffers.

### Hypotheses

- **H1 - Interior optimum:** Expected immediate savings is non-monotonic in the
  opening-offer ratio and has an interior optimum for a meaningful share of
  listings.
- **H2 - Support conservatism:** Supervised greedy optimization selects less
  historically supported offers and has a larger model-versus-OPE gap than
  one-step Conservative Q-Learning (CQL).
- **H3 - Simulator sensitivity:** PPO's apparent simulator reward is higher in
  the basic simulator than under probability noise and historical-support
  penalties.

## Data and semantic corrections

The source Parquet contained 47,377,200 event rows. A bargaining thread is the
pair `(anon_item_id, anon_byr_id)`. Events were sorted chronologically and a
thread was retained only when its first event was a buyer opening offer
(`offr_type_id == 0`). This produced 5,705,893 valid fashion opening offers.

The immediate opening offer is accepted when the status of that opening row is
1 or 9. A later accepted counteroffer does not change this label. This mapping
was checked empirically: the median `item_price / offr_price` is exactly 1.00
for statuses 1 and 9, and 100% of their non-null observations are within 1% of
equality. By contrast, status 2 has a median ratio of 1.54 and only 1.1% are
within 1%, so status 2 is not treated as an accepted opening offer.

The action is `offr_price / start_price_usd`, restricted to `(0.01, 1.00]`.
The realized immediate reward is:

```text
opening_accepted * (1 - anchor_ratio)
```

This corrects the earlier use of `item_price` as the acquisition price.
Listing IDs were assigned as complete groups to train, validation, and test
partitions. The resulting split contained 4,564,662 / 570,404 / 570,827 rows,
with zero listing overlap. The test immediate-acceptance rate was 33.46% and
the mean historical opening ratio was 60.62%.

The study is restricted to fashion (`meta_categ_id == 11450`). All conclusions
are conditional on this category.

## Features and practical output

The state is deliberately compact:

1. log listing price;
2. normalized seller feedback score;
3. seller positive-feedback percentage;
4. cleaned fashion leaf category (25 frequent categories plus other).

The action is not included as a state feature. The system outputs a recommended
opening ratio, a dollar offer, predicted immediate acceptance, predicted
immediate expected savings, and a historical-support flag. Thus there is
differentiation within fashion through price, seller reputation, and leaf
category, but not through unobserved brand, condition, material, demand, or
resale value.

## Methods

### Phase 1: calibrated supervised acceptance model

An XGBoost classifier estimates immediate opening acceptance. Calibration is
central because the objective multiplies the estimated probability by the
discount. A fixed 0.70 rule and a genuine grid-search supervised greedy policy
are evaluated separately.

### Phase 2: one-step CQL

CQL directly regresses the observed one-step reward. Every row is terminal,
`gamma=0`, the conservative penalty is `alpha=5.0`, and ten sampled negative
actions are used. The intent is not to force CQL to win, but to measure the
reward-support trade-off.

### Phase 3: PPO sensitivity experiment

PPO is trained only in the learned Phase-1 simulator. A Beta actor produces
bounded actions without clipping a Gaussian sample. The basic simulator is
compared with a robust variant that adds probability noise (`sigma=0.03`) and
penalizes actions outside the training p5-p95 support interval.

### OPE diagnostics

A learned behavioral density supports kernel IPS, SNIPS, doubly robust (DR),
and self-normalized DR diagnostics. Propensities are estimated rather than
logged, so these results indicate support sensitivity and are not causal
marketplace lift. The main diagnostic below uses bandwidth 0.05, clipped
weights, 50,000 test rows, and 30 bootstrap replicates.

## Results

The acceptance model obtained test ROC AUC 0.8339, Brier score 0.1539, and log
loss 0.4660. Calibration bins closely followed the diagonal. The expected-
savings curve rose with acceptance at low anchors, reached an interior maximum,
and then fell as the discount approached zero.

| Policy | Evidence | Mean anchor | Acceptance | Expected savings | Within p5-p95 support |
| --- | --- | ---: | ---: | ---: | ---: |
| Historical behavior | Observed outcomes | 0.6062 | 0.3346 | 0.0973 | 1.0000 |
| Fixed anchor 0.70 | Phase-1 estimate | 0.7000 | 0.4226 | 0.1268 | 1.0000 |
| Supervised greedy | Phase-1 estimate | 0.7333 | 0.5007 | 0.1371 | 0.9099 |
| One-step CQL | Phase-1 estimate | 0.7025 | 0.4214 | 0.1219 | 0.9992 |
| PPO basic | Simulator only | 0.7455 | 0.4823 | 0.1273 | not comparable |
| PPO robust | Simulator only | 0.7359 | 0.4515 | 0.1209 | not comparable |

These rows are not a single causal leaderboard because their evidence types
differ.

### H1: supported

The supervised greedy policy selected an interior grid action for 99.981% of
test listings; only 0.019% selected the lower boundary and none selected the
upper boundary. The population response curve peaked around an opening ratio of
0.67. The data therefore support a real acceptance-discount trade-off rather
than the tautology that lower accepted offers imply larger discounts.

### H2: partially supported

The supervised greedy policy kept 90.99% of actions in the historical p5-p95
band, whereas CQL kept 99.92% in support. CQL achieved this by lowering its mean
anchor from 0.733 to 0.702 and sacrificing 1.52 percentage points of Phase-1
model-estimated reward.

At OPE bandwidth 0.05 with clipped weights, supervised greedy had reward-model
value 0.1373, SNIPS 0.1277 (95% bootstrap interval 0.1257-0.1299), and effective
sample size 25.6% of the diagnostic sample. CQL had reward-model value 0.1221,
SNIPS 0.1235 (0.1216-0.1252), and effective sample size 36.2%. The supervised
model-minus-SNIPS gap was +0.0095, while CQL's was -0.0014. This is consistent
with reduced optimism and better support under CQL. However, DR estimates and
bandwidth sensitivity do not establish causal overstatement; therefore H2 is
only partially, not conclusively, supported.

### H3: supported as a simulator sensitivity result

Basic PPO reported expected simulator savings of 0.1273. Under probability
noise and support penalties this fell to 0.1209, an absolute decline of 0.0063
or 4.99%. The robust estimate is close to CQL's 0.1219. This supports the claim
that part of PPO's apparent advantage is simulator sensitivity, but it does not
measure a real marketplace effect.

## Practical recommendation

The simplest defensible default is an opening offer near 70% of the listing
price. It is fully inside the central historical support band and its model-
estimated expected savings (12.68%) exceeded CQL's (12.19%). The complex CQL
policy therefore did not beat the simple fixed rule on predicted reward.

For individualized decision support, use the supervised greedy recommendation
only when it is marked in-support; its average recommendation was about 73.3%
of listing price. If the greedy recommendation is flagged as extrapolated,
prefer the CQL recommendation or the fixed 70% rule. The supplied recommender
implements this menu and reports ratios, dollar offers, acceptance estimates,
expected savings, and support flags.

This guidance is predictive, not causal. It should be presented as a model-
based opening-offer aid, not a guaranteed bargaining strategy.

## Limitations

1. Declined, expired, and countered openings all receive zero reward. This
   omits the option value of eliciting a counteroffer and may favor higher
   immediate-close offers.
2. Buyers choose anchors endogenously. Unobserved item quality, perceived
   overpricing, buyer information, and seller reservation values can confound
   the estimated acceptance surface.
3. The compact feature set omits brand, condition, detailed demand, buyer
   history, and resale value. The project cannot estimate resale margin.
4. Fashion-only results may not generalize to other categories or periods.
5. PPO inherits the classifier's errors and is simulator-only.

## Academic contribution

The contribution is not a claim that reinforcement learning automatically
beats simpler baselines. It is a reproducible comparison of a calibrated
supervised optimizer, support-conservative offline learning, and simulator-only
PPO under one coherent opening-offer objective. The main findings are an
empirical interior optimum, a measurable reward-support trade-off, and a
quantified simulator sensitivity. The non-win of CQL against a fixed 70% rule
is itself an informative result about when added model complexity is not
practically justified.

## Reproducibility

The full corrected run used seed 42. It generated semantic-audit tables,
grouped split checks, classifier calibration, CQL and PPO histories, OPE weight
diagnostics, a 5,000-row recommendation table, and the comparison dashboard.
Seventeen automated semantic and static tests pass.

## References

- Kumar, A. et al. (2020). Conservative Q-Learning for Offline Reinforcement
  Learning. NeurIPS 2020. https://arxiv.org/abs/2006.04779
- Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms.
  https://arxiv.org/abs/1707.06347
- Guo, C. et al. (2017). On Calibration of Modern Neural Networks.
  ICML 2017. https://proceedings.mlr.press/v70/guo17a.html
- Dudik, M., Langford, J., and Li, L. (2011). Doubly Robust Policy Evaluation
  and Learning. ICML 2011. https://arxiv.org/abs/1103.4601
