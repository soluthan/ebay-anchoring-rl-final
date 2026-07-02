# Offline and Simulated Online Reinforcement Learning for eBay Best Offer Anchoring

## Abstract

This project studies buyer-side anchoring in eBay Best Offer negotiations as a decision-support problem. The buyer's opening offer is represented as an anchor ratio relative to the seller's list price, and the main question is whether supervised learning, offline reinforcement learning, and simulated online reinforcement learning recommend meaningfully different anchors under observational support constraints. The empirical design uses a one-step Markov decision process, close to a contextual bandit, where the state describes a listing and seller, the action is the opening offer ratio, and the reward is accepted-offer savings. Because accepted-offer savings are mechanically `1 - anchor_ratio`, the final pipeline uses a single supervised acceptance model rather than a separate realized-price regressor.

The main result is not that an RL policy can safely outperform historical behavior in a live marketplace. Instead, the result is more cautious and scientifically useful: aggressive supervised maximization can propose anchors outside the historical support of the logged data, while Conservative Q-Learning (CQL) produces a more conservative target policy whose off-policy evaluation diagnostics have substantially healthier effective sample size. PPO trained in a learned simulator produces high simulator-only savings, but those results are framed as a stress test of the learned environment rather than evidence of real-world lift. The project therefore contributes a reproducible, flat-module pipeline for anchoring research, a support-aware comparison of supervised, offline RL, and simulated online RL policies, and a clear set of limitations for future sequential negotiation modeling.

## 1. Introduction and Motivation

Best Offer marketplaces expose an economically interesting decision: a buyer must choose how low to anchor an opening offer. Behavioral research suggests that first offers can shape bargaining outcomes by setting a reference point, yet a very low offer also risks rejection. This creates a natural trade-off between savings conditional on acceptance and the probability of reaching a deal. For an eBay buyer, the practical question is not simply "what is the lowest possible offer?" but "what opening offer has an acceptable chance of success for this listing and seller?"

This project models that trade-off for eBay Best Offer data. The action is the buyer's opening offer ratio, defined as opening offer divided by list price. Lower anchors produce larger potential savings if accepted, while higher anchors are usually safer. The immediate reward for an accepted offer is therefore direct and interpretable: `1 - anchor_ratio`. If the offer is rejected, the realized reward is zero. Expected savings can then be written as:

`E[savings | state, anchor] = P(accept | state, anchor) * (1 - anchor)`.

This formulation makes the project a decision-support system rather than a claim about manipulating seller behavior in production. The system recommends candidate anchors and reports whether those anchors are supported by historical data. That support condition is central because the dataset is observational: buyers did not randomize offers, and logged behavior may not cover all anchors equally. A purely supervised optimizer can exploit errors in the learned acceptance model by choosing extreme actions that were rarely observed. Offline RL and off-policy diagnostics are introduced specifically to address this risk.

The project has four contributions.

1. It defines eBay Best Offer anchoring as a one-step MDP/contextual-bandit problem with an interpretable action and reward.
2. It implements a supervised acceptance model, a fixed-anchor baseline, CQL offline RL, and PPO in a learned faithful simulator using a flat, reproducible Python module layout.
3. It adds robustness controls for temporal and leaf-category splits, train-only preprocessing, and off-policy diagnostics with SNIPS, DR-style estimates, bootstrap confidence intervals, effective sample size, and bandwidth sensitivity.
4. It separates evidence types: historical rows are logged outcomes, supervised and CQL rows are Phase-1 model estimates, and PPO rows are simulator-only estimates.

## 2. Related Work

### Anchoring and Negotiation

The behavioral motivation comes from anchoring-and-adjustment research. Tversky and Kahneman (1974) show that numeric anchors can systematically influence judgment under uncertainty. In negotiation contexts, Galinsky and Mussweiler (2001) argue that first offers can anchor counterpart expectations and final agreements, while Northcraft and Neale (1987) show anchoring effects in pricing judgments. Backus, Blake, Larsen, and Tadelis (2020) provide the closest marketplace precedent: a field study of sequential bargaining on eBay Best Offer. Together, these papers motivate the idea that an opening offer is not merely a mechanical bid but also a reference point within a bargaining process.

The present project uses that literature cautiously. The eBay dataset does not identify seller beliefs, seller reservation values, or the counterfactual response to a different buyer offer. Therefore, the project does not claim to estimate a causal anchoring effect. Instead, anchoring is used as a behavioral and economic framing for the action variable: the opening offer ratio is the buyer's chosen reference point, and the model evaluates historically supported decision rules over that action.

### Contextual Bandits and One-Step MDPs

Because only the opening offer is modeled, the decision problem is effectively a contextual bandit or one-step MDP. The state is observed before action, the action is continuous, and the reward is immediate. This connects to contextual bandit work such as Li et al. (2010), which studies policy learning from logged contextual interaction data, and to counterfactual learning frameworks such as Bottou et al. (2013), Swaminathan and Joachims (2015), and Dudik et al. (2011). The key shared issue is that evaluation depends on whether the target policy chooses actions that are sufficiently represented in the logged data.

### Policy Gradients, Actor-Critic Methods, and PPO

Policy-gradient methods provide the online RL foundation. Sutton et al. (1999) establish the policy-gradient theorem, while Konda and Tsitsiklis (2000) analyze actor-critic methods. Schulman et al. (2015) introduce generalized advantage estimation, and Schulman et al. (2017) introduce Proximal Policy Optimization (PPO), which stabilizes policy updates through a clipped surrogate objective. In this project, PPO is not trained online in the real marketplace. It is trained inside a learned simulator based on the Phase-1 acceptance model, so its empirical role is exploratory: it shows what a simulated online optimizer does under the learned acceptance surface and support threshold. This evidence boundary is consistent with model-based offline RL work such as MOPO (Yu et al., 2020) and COMBO (Yu et al., 2021), which explicitly treat learned-model rollouts and out-of-support actions as sources of bias that require pessimism or penalties.

### Offline RL, CQL, and Support Mismatch

Offline RL is directly relevant because the project learns from a fixed logged dataset. A central risk is extrapolation error: policies may choose actions rarely observed in the data, causing value estimates to rely on unsupported model generalization. Fujimoto et al. (2019) discuss this problem in batch-constrained deep RL, and Levine et al. (2020) review broader offline RL challenges. Conservative Q-Learning (Kumar et al., 2020) addresses extrapolation by penalizing high Q-values on out-of-distribution actions. Implicit Q-Learning (Kostrikov, Nair, and Levine, 2021) is a closely related modern alternative because it avoids directly querying values for unseen actions. This makes CQL a natural methodological fit for buyer anchoring, where low anchors may look attractive under a supervised model but may not be supported by historical behavior.

### Off-Policy Evaluation

Off-policy evaluation (OPE) is necessary because the project compares target policies using logged data. The report uses estimated propensities because true randomized logging propensities are unavailable. This makes the OPE diagnostic rather than causal. SNIPS is emphasized over raw IPS because self-normalization is more stable under heavy-tailed weights. Doubly robust estimators are included because they combine a reward model with importance weighting, following ideas in Dudik et al. (2011) and Jiang and Li (2016). Uehara, Shi, and Kallus (2022) provide a modern overview of OPE and its statistical limits, while Lee et al. (2022) is especially relevant here because it studies deterministic target policies with continuous actions through kernel-based OPE. Effective sample size (ESS), clipping, and bandwidth sensitivity are treated as core outputs rather than technical appendices: if ESS collapses, the target policy is not credibly evaluable from the logged support.

### Recommender and Decision-Support Framing

The final system is best interpreted as a recommender or decision-support tool. It does not autonomously bargain, and it does not claim production lift. Instead, it generates offer anchors and support diagnostics for a buyer-facing decision. This framing is closer to logged-feedback recommendation and decision support than to autonomous marketplace control. Xiao and Wang (2023), for example, frame offline RL for recommendation around logged feedback, distribution mismatch, and support constraints. This also makes the ethical and methodological boundary clearer: recommendations should be accompanied by evidence type, uncertainty, and support warnings.

## 3. Problem Formulation

For each Best Offer interaction, the observed state vector includes listing and seller features:

- `log_list_price`: log-transformed seller list price.
- `seller_score_norm`: seller feedback score normalized by the training-set 99th percentile.
- `seller_pos_pct`: seller positive-feedback percentage with train-median imputation.
- `categ_id_clean`: a cleaned anonymized leaf-category feature.

The action is:

`a = anchor_ratio = opening_offer / list_price`.

The reward is:

`r = 1 - a` if the offer is accepted, and `r = 0` otherwise.

This gives the expected reward objective:

`E[r | s, a] = P(accept | s, a) * (1 - a)`.

The final design deliberately removes the earlier redundant price-regression path. Because `offr_price` is the opening offer and accepted-offer savings are already determined by the anchor ratio, a second model for realized price would add complexity without adding identification. The retained supervised model estimates only acceptance probability.

The production random split contains 7,668,237 train rows, 958,530 validation rows, and 958,530 test rows. The observed test deal rate is 0.1278, the mean historical anchor is 0.6654, and the historical 5th to 95th percentile anchor range is approximately 0.3125 to 0.9270. The preprocessing pipeline also supports temporal splits and leaf-holdout splits. Importantly, category vocabulary, seller-score normalization, and missing-value imputation are fitted on the training split only before being applied to validation and test, reducing leakage.

## 4. Methods

### Phase 1: Supervised Acceptance Model

Phase 1 trains an XGBoost binary classifier to predict acceptance:

`P(accept | state, anchor)`.

The classifier is used in three places: the fixed-anchor supervised baseline, the CQL evaluation surface, and the PPO simulator. The production model obtains a test AUC of 0.7011. This is strong enough to support comparative diagnostics, but it is not treated as a causal response model. The model may still encode confounding from buyer selection, seller heterogeneity, and unobserved listing quality.

The simplest supervised baseline is a fixed anchor of 0.70. A more aggressive greedy optimizer is also used inside OPE diagnostics, but its support collapses because it tends to select very low anchors. That collapse is important: it demonstrates why pure supervised maximization is risky in this domain.

### Phase 2: Offline RL with CQL

Phase 2 trains a one-step CQL agent on the logged offline data. Actions are continuous anchor ratios, and rewards are accepted-offer savings. The CQL objective is appropriate because the dataset is observational and the model should avoid unsupported high-value actions. In practical terms, CQL searches for anchors with favorable expected savings while penalizing value estimates on actions outside the logged support.

The production CQL policy selects a mean anchor of 0.7690 with standard deviation 0.0507. Under the Phase-1 model, this policy has expected savings of 0.0200 and predicted deal probability of 0.0865. It is more conservative than the fixed 0.70 anchor, and its expected savings is lower under the model estimate. The reason it remains scientifically valuable is not headline reward but support: CQL is much more evaluable than the aggressive greedy policy under OPE diagnostics.

### Phase 3: PPO in a Faithful Simulator

Phase 3 trains PPO inside a learned simulator. The simulator uses the Phase-1 acceptance model and a faithful-support threshold derived from the historical anchor distribution. The production faithful threshold is the empirical 5th percentile of the training anchors, 0.3114. This prevents the simulator from rewarding anchors far below the observed support boundary.

The PPO result is intentionally labeled simulator-only. In production, PPO selects a mean anchor of 0.3176 and obtains simulator-estimated expected savings of 0.1840 with predicted acceptance of 0.2699. These numbers are much higher than the model-estimated supervised and CQL rows, but they are not directly comparable to observed logged outcomes. They depend on the learned simulator and should be interpreted as an exploratory upper-bound/stress-test behavior, not as evidence of live-marketplace improvement.

### Off-Policy Evaluation

The OPE module estimates behavior propensities from logged state-action pairs using a continuous-action kernel. It reports IPS, SNIPS, DR-style estimates, bootstrap confidence intervals, ESS, clipped/unclipped weights, and bandwidth sensitivity over `h = 0.03`, `0.05`, and `0.10`. Because true logging propensities are unavailable, these values are diagnostic rather than causal.

The central OPE question is whether a target policy can be evaluated with enough support. This is why ESS is reported beside every value estimate. A high reward estimate with tiny ESS is not persuasive; it is a warning sign that the target policy is asking the logged data to answer a counterfactual it barely contains.

## 5. Numerical Studies

### Policy Comparison

The production pipeline yields the following comparison table.

| Policy | Evidence type | Expected savings | Deal / acceptance rate | Mean anchor |
| --- | --- | ---: | ---: | ---: |
| Behavioral historical | Observed logged outcome | 0.0502 | 0.1278 | 0.6654 |
| Supervised fixed anchor 0.70 | Phase-1 model estimate | 0.0268 | 0.0893 | 0.7000 |
| CQL offline RL | Phase-1 model estimate | 0.0200 | 0.0865 | 0.7690 |
| PPO faithful simulator | Simulator-only estimate | 0.1840 | 0.2699 | 0.3176 |

The table should not be read as a single leaderboard. Each row has a different evidence type. The behavioral row is an observed historical average. The supervised and CQL rows are evaluated under the Phase-1 acceptance model. PPO is evaluated inside a learned simulator. The most reliable comparison is therefore not "which row is largest?" but "which claims are supported by which evidence?"

The observed historical policy has higher realized savings than the fixed-anchor and CQL model estimates, partly because the historical average includes the realized outcomes of actual selected buyer offers. The CQL policy is conservative in anchor choice and sacrifices expected savings under the Phase-1 surface. That result is negative but useful: under the current features and reward definition, conservative offline RL does not automatically improve estimated savings over the logged behavior.

### OPE Support Diagnostics

The OPE diagnostics explain why the aggressive supervised greedy policy should not be used as a central result. At bandwidth `h = 0.05`, CQL has clipped ESS fraction 0.3619, while the greedy model has clipped ESS fraction 0.0083. Across the bandwidth sweep, CQL remains in a usable range, while the greedy target remains badly unsupported.

| Target policy | Mean target anchor | ESS fraction at h=0.03 | ESS fraction at h=0.05 | ESS fraction at h=0.10 | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| CQL offline RL | 0.7690 | 0.2254 | 0.3619 | 0.6018 | Supported enough for diagnostic OPE. |
| Fixed anchor 0.70 | 0.7000 | 0.2162 | 0.3621 | 0.6872 | Supported enough for diagnostic OPE. |
| Greedy model | 0.0102 | 0.0044 | 0.0083 | 0.0187 | Unsupported; estimates are extrapolative. |

The OPE value estimates match this interpretation. For CQL, SNIPS and DR-style estimates are stable and close to the Phase-1 model estimate: at `h = 0.05`, SNIPS is 0.0191 and DR is 0.0197. For the fixed 0.70 anchor, SNIPS is 0.0257 and DR is 0.0268. For the greedy model, reward estimates are high, but the ESS collapse makes them unreliable. The scientifically responsible conclusion is therefore not that greedy anchoring is superior, but that it is not credibly evaluable from this logged data.

### Robustness Design

The codebase includes random, temporal, and leaf-holdout preprocessing modes. Temporal splitting checks sensitivity to chronological drift. Leaf-holdout splitting checks whether the model generalizes to unseen anonymized fashion subcategories. The current production numbers above come from the random split; future report revisions should add a small robustness table for temporal and leaf-holdout runs if time permits. Cross-meta-category evaluation would be a stronger test of the standardized-versus-idiosyncratic product thesis, but it would require expanding beyond the current fashion-only scope.

## 6. Discussion and Limitations

The project is intentionally cautious because several identification limits remain.

First, the data are observational. Buyers choose anchors strategically, and sellers choose whether to accept under unobserved constraints. Without randomized logging propensities or a natural experiment, OPE cannot identify causal marketplace lift. Estimated propensities help diagnose support mismatch, but they do not remove confounding.

Second, the one-step formulation abstracts away the sequential structure of negotiation. Best Offer interactions can include counteroffers, repeated messages, expiration dynamics, and seller learning. A one-step MDP is a useful first approximation because the opening offer is the most visible anchor, but it cannot represent multi-turn bargaining strategy. A future Phase 4 model could use sequence models or a partially observable MDP to represent negotiation histories.

Third, the state representation is intentionally compact. It includes list price, seller feedback, seller positive percentage, and anonymized category. It omits richer product features such as title embeddings, condition, brand, images, seller inventory context, and time-on-market. These missing variables may explain acceptance and support patterns that the current model attributes to the anchor.

Fourth, PPO depends on simulator fidelity. Even with a faithful-support threshold, the simulator is learned from the Phase-1 acceptance model and inherits its biases. Model-based offline RL work makes this risk explicit: MOPO penalizes uncertain model rollouts to manage the trade-off between generalization and leaving the batch-data support (Yu et al., 2020), while COMBO regularizes values on model-generated out-of-support tuples (Yu et al., 2021). PPO can exploit smooth regions of the learned acceptance surface in ways that would not survive real marketplace interaction. For that reason, PPO is reported as simulator-only. Its main value is exploratory: it shows what the learned environment incentivizes, and it highlights where the model would need stronger validation before deployment.

Fifth, the CQL result is not a dramatic performance win. It is arguably more important as a methodological result: conservatism improves evaluability and support alignment, but under the current model it also lowers expected savings. That negative finding strengthens the report because it prevents a decorative "RL beats baseline" narrative. The honest claim is that support-aware RL changes the recommendation profile and makes unsupported extrapolation visible.

## 7. Conclusion and Future Work

This project shows how eBay Best Offer anchoring can be formulated as a support-aware decision problem. The final pipeline compares logged behavior, supervised baselines, CQL offline RL, and PPO in a learned faithful simulator while preserving clear evidence boundaries. The strongest finding is diagnostic: aggressive supervised optimization can leave the support of the logged data, whereas CQL remains much more evaluable under OPE. PPO produces high simulator-only rewards, but those numbers should be treated as a learned-environment stress test rather than a production claim.

The next steps are straightforward. First, run and report temporal and leaf-holdout robustness tables using the existing preprocessing modes. Second, expand product features beyond compact seller/listing variables. Third, test cross-meta-category generalization to distinguish standardized from idiosyncratic goods. Fourth, move from a one-step anchor model to a sequential negotiation model that can represent counteroffers and partial observability. These extensions would make the project stronger without changing its core scientific discipline: every recommendation should be tied to support, uncertainty, and evidence type.

## References

Backus, M., Blake, T., Larsen, B., and Tadelis, S. (2020). Sequential bargaining in the field: Evidence from millions of online bargaining interactions. The Quarterly Journal of Economics, 135(3), 1319-1361. doi:10.1093/qje/qjaa003.

Bottou, L., Peters, J., Quinonero-Candela, J., Charles, D. X., Chickering, D. M., Portugaly, E., Ray, D., Simard, P., and Snelson, E. (2013). Counterfactual reasoning and learning systems: The example of computational advertising. Journal of Machine Learning Research.

Dudik, M., Langford, J., and Li, L. (2011). Doubly robust policy evaluation and learning. International Conference on Machine Learning.

Fujimoto, S., Meger, D., and Precup, D. (2019). Off-policy deep reinforcement learning without exploration. International Conference on Machine Learning.

Galinsky, A. D., and Mussweiler, T. (2001). First offers as anchors: The role of perspective-taking and negotiator focus. Journal of Personality and Social Psychology.

Jiang, N., and Li, L. (2016). Doubly robust off-policy value evaluation for reinforcement learning. International Conference on Machine Learning.

Konda, V. R., and Tsitsiklis, J. N. (2000). Actor-critic algorithms. Advances in Neural Information Processing Systems.

Kostrikov, I., Nair, A., and Levine, S. (2021). Offline reinforcement learning with implicit Q-learning. arXiv:2110.06169.

Kumar, A., Zhou, A., Tucker, G., and Levine, S. (2020). Conservative Q-Learning for offline reinforcement learning. Advances in Neural Information Processing Systems.

Lee, H., Lee, J., Choi, Y., Jeon, W., Lee, B.-J., Noh, Y.-K., and Kim, K.-E. (2022). Local metric learning for off-policy evaluation in contextual bandits with continuous actions. arXiv:2210.13373.

Levine, S., Kumar, A., Tucker, G., and Fu, J. (2020). Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv.

Li, L., Chu, W., Langford, J., and Schapire, R. E. (2010). A contextual-bandit approach to personalized news article recommendation. International World Wide Web Conference.

Northcraft, G. B., and Neale, M. A. (1987). Experts, amateurs, and real estate: An anchoring-and-adjustment perspective on property pricing decisions. Organizational Behavior and Human Decision Processes.

Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. (2015). Trust Region Policy Optimization. International Conference on Machine Learning.

Schulman, J., Moritz, P., Levine, S., Jordan, M., and Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. International Conference on Learning Representations.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017). Proximal Policy Optimization algorithms. arXiv.

Sutton, R. S., McAllester, D., Singh, S., and Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. Advances in Neural Information Processing Systems.

Swaminathan, A., and Joachims, T. (2015). Counterfactual risk minimization: Learning from logged bandit feedback. International Conference on Machine Learning.

Tversky, A., and Kahneman, D. (1974). Judgment under uncertainty: Heuristics and biases. Science.

Uehara, M., Shi, C., and Kallus, N. (2022). A review of off-policy evaluation in reinforcement learning. arXiv:2212.06355.

Xiao, T., and Wang, D. (2023). A general offline reinforcement learning framework for interactive recommendation. arXiv:2310.00678.

Yu, T., Thomas, G., Yu, L., Ermon, S., Zou, J., Levine, S., Finn, C., and Ma, T. (2020). MOPO: Model-based offline policy optimization. Advances in Neural Information Processing Systems.

Yu, T., Kumar, A., Rafailov, R., Rajeswaran, A., Levine, S., and Finn, C. (2021). COMBO: Conservative offline model-based policy optimization. Advances in Neural Information Processing Systems.
