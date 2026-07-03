# Appendix: Repository Structure and Core Values

This appendix documents how the repository is organized and what each core
project value means. It is intended as a quick guide for reviewers who want to
understand the project structure without reading every source file first.

## 1. Repository Structure

The project intentionally uses a flat Python module layout. The phase scripts
sit next to each other at the repository root so that `run_pipeline.py` can
import them directly without packaging the project into a Python package.

```text
.
|-- README.md
|-- REPRODUCIBILITY.md
|-- requirements.txt
|-- requirements-lock.txt
|-- configs/
|   `-- default_experiment.json
|-- docs/
|   |-- data_schema.md
|   `-- repository_structure_appendix.md
|-- report/
|   |-- README.md
|   |-- technical_report.md
|   |-- technical_report.pdf
|   |-- technical_report_results_dashboard.png
|   `-- ebay_anchoring_rl_presentation.pptx
|-- tests/
|   `-- test_static_repo.py
|-- project_constants.py
|-- data_preprocess.py
|-- phase1_supervised.py
|-- phase2_cql.py
|-- phase3_ppo.py
|-- ope.py
|-- results.py
|-- recommend.py
|-- recommend_one.py
`-- run_pipeline.py
```

Ignored local folders such as `data/`, `models/`, `outputs/`, `.venv/`, and
smoke-test variants are not part of the committed repository. They are local
runtime products.

## 2. Folder Responsibilities

| Path | Role |
| --- | --- |
| `configs/` | Documents the default experiment configuration, run order, paths, seeds, and expected artifacts. |
| `docs/` | Contains reviewer-facing documentation for data schema, repository structure, and project semantics. |
| `report/` | Contains final human-facing deliverables only: report, presentation, and selected final figures. |
| `tests/` | Contains lightweight static tests that run without the private dataset. |
| root Python files | Implement the flat preprocessing, modeling, evaluation, recommendation, and orchestration pipeline. |

## 3. Core Module Responsibilities

| Module | Core responsibility | Main output when run locally |
| --- | --- | --- |
| `project_constants.py` | Centralizes shared schema names, seed, action bounds, and feature columns. | No files; imported by other modules. |
| `data_preprocess.py` | Converts the merged raw source into train/validation/test one-step MDP rows. | `data/train.parquet`, `data/val.parquet`, `data/test.parquet`, split audit files. |
| `phase1_supervised.py` | Trains the supervised acceptance model `P(accept | state, anchor)`. | `models/deal_classifier.ubj`. |
| `phase2_cql.py` | Trains a conservative offline RL policy using CQL. | `models/cql_best.pt`, `models/cql_scaler.pkl`, CQL metrics. |
| `phase3_ppo.py` | Trains PPO in a learned simulator based on Phase 1 acceptance probabilities. | PPO checkpoint and simulator metrics, usually faithful-simulator variants. |
| `ope.py` | Runs off-policy evaluation diagnostics and support checks. | `outputs/ope_policy_eval.csv`, `outputs/ope_weight_diagnostics.csv`. |
| `results.py` | Aggregates final policy-comparison tables and dashboard figures. | `outputs/policy_comparison.csv`, final report figure inputs. |
| `recommend.py` | Produces batch recommendations and support flags for listings. | `outputs/offer_recommendations.csv`. |
| `recommend_one.py` | Produces an offer menu for one listing from command-line inputs. | Console output. |
| `run_pipeline.py` | Runs the phase modules in sequence while preserving flat imports. | The combined outputs of selected phases. |

## 4. Core Values and Terms

These names appear repeatedly in the code, report, and presentation.

| Name | Meaning | Why it matters |
| --- | --- | --- |
| `SEED = 42` | Shared random seed for reproducible splits, model sampling, CQL, PPO, and recommendation sampling. | Keeps local reruns as stable as possible. |
| `STATE_COLS` | The compact pre-offer state: `log_list_price`, `seller_score_norm`, `seller_pos_pct`, `categ_id_clean`. | Defines the listing/seller context given to all policies. |
| `ACTION_COL = anchor_ratio` | Buyer's opening offer divided by list price. | This is the policy decision variable, also called the anchor. |
| `REWARD_COL = savings_pct` | `1 - anchor_ratio` if accepted, otherwise `0`. | Defines the one-step reward used for policy comparison. |
| `DEAL_COL = status_id` | Original acceptance status. | Used to identify accepted offers and estimate acceptance. |
| `DEAL_STATUS = 2` | Status value denoting an accepted offer in the dataset. | Converts raw status codes into the binary deal target. |
| `FASHION_CATEG_IDS = [11450]` | Current high-level category scope. | Keeps the empirical study within fashion Best Offer listings. |
| `ANCHOR_MIN = 0.01` | Lower action bound for candidate anchors. | Prevents degenerate zero-offer actions. |
| `ANCHOR_MAX = 1.50` | Upper action bound for candidate anchors. | Allows offers above list price in the action grid if needed by data or diagnostics. |
| `N_GRID = 50` | Number of candidate anchors used in grid search/recommendation routines. | Controls resolution of greedy and recommendation scans. |
| `N_ACTIONS_DISC = 50` | Number of discrete sampled actions for CQL-style action comparisons. | Controls action discretization in offline RL training/evaluation. |

## 5. State Feature Selection

The state is deliberately compact and uses only information observed before the
buyer chooses an opening offer:

| Feature | Interpretation |
| --- | --- |
| `log_list_price` | Price context, log-transformed to reduce scale effects. |
| `seller_score_norm` | Seller feedback score normalized by a train-fitted 99th percentile. |
| `seller_pos_pct` | Seller positive-feedback percentage, median-filled from the training split when missing. |
| `categ_id_clean` | Train-fitted anonymized leaf-category feature. |

This feature choice is not copied directly from the behavioral anchoring
literature. The anchoring literature motivates the opening offer as a reference
point, while the eBay bargaining literature motivates the marketplace setting.
The features here are selected for a one-step contextual-bandit/RL formulation:
they condition `P(accept | state, anchor)` without using post-action or
outcome-leaking information.

## 6. Evidence Types

The final report and presentation separate evidence types carefully.

| Evidence type | Meaning | Example |
| --- | --- | --- |
| Observed | Directly logged historical behavior. | Historical buyer behavior row in result tables. |
| Model estimate | Computed with the Phase 1 acceptance model. | Fixed anchor, greedy baseline, and CQL model-estimated values. |
| OPE diagnostic | Support/evaluability check from estimated behavior propensities. | Effective sample size and weight diagnostics. |
| Simulator-only | PPO trained and evaluated inside the learned simulator. | Faithful PPO row in the comparison table. |

This separation is central to the project. PPO simulator results should not be
read as live marketplace lift. OPE values should be read as support diagnostics,
not causal treatment effects.

## 7. Core Reported Numbers

These are the headline values used in the final report and presentation. They
are included here to clarify what each number represents and what evidence type
it belongs to.

| Value | Meaning | Evidence boundary |
| --- | --- | --- |
| `AUC = 0.701` | Phase 1 acceptance-model discrimination on held-out data. | Predictive supervised metric, not a policy value. |
| `ESS = 36.2%` for fixed `0.70` | Effective sample size fraction for evaluating a fixed 0.70 anchor at bandwidth `h = 0.05`. | OPE support diagnostic. |
| `ESS = 36.2%` for CQL | Effective sample size fraction for evaluating the CQL target policy at bandwidth `h = 0.05`. | OPE support diagnostic. |
| `ESS = 0.8%` for greedy | Effective sample size fraction for the greedy supervised target policy. | Support-collapse warning. |
| Historical savings `0.050` | Observed average logged savings. | Historical logged behavior. |
| Fixed-anchor savings `0.027` | Expected savings estimate for the fixed 0.70 baseline. | Phase 1 model estimate. |
| CQL savings `0.020` | Expected savings estimate for the conservative offline RL policy. | Phase 1 model estimate with conservative target policy. |
| PPO savings `0.184` | Simulator-only average reward from faithful PPO. | Learned simulator result, not live-marketplace lift. |

The central interpretation is that greedy supervised maximization is not trusted
because its effective sample size collapses. CQL is valuable as a conservative
policy because it remains evaluable from the logged data, even though its
estimated reward is lower than historical behavior in the reported table.

## 8. Generated Artifacts and Git Boundary

The repository tracks code, documentation, and final human-facing deliverables.
It does not track private data or local model artifacts.

| Local artifact | Git policy |
| --- | --- |
| `clean_master_dataset.parquet` | Local private input; ignored. |
| `data/*.parquet`, `data/*.csv` | Generated preprocessing outputs; ignored. |
| `models/*.ubj`, `models/*.pt`, `models/*.pkl` | Trained models/checkpoints/scalers; ignored. |
| `outputs/*.csv`, `outputs/*.png` | Generated diagnostics, recommendations, and figures; ignored unless manually promoted to final report assets. |
| `report/technical_report.*` | Final report deliverables; tracked. |
| `report/ebay_anchoring_rl_presentation.pptx` | Final presentation deliverable; tracked. |

## 9. Reading Order for Reviewers

Recommended path through the repository:

1. `README.md` for the project overview and run commands.
2. `docs/data_schema.md` for raw-to-engineered data semantics.
3. `docs/repository_structure_appendix.md` for folder/module/core-value mapping.
4. `configs/default_experiment.json` for reproducibility settings.
5. `technical_report.pdf` for the scientific narrative.
6. `run_pipeline.py` and phase modules for implementation details.
