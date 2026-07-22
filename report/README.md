# Report Artifacts

This directory contains the corrected full-data technical report. Its numerical
claims come from the status-{1,9}, first-event, listing-grouped production run.
Do not restore claims from the old status-2, all-events, row-split pipeline.

Recommended tracked files:

- `technical_report.md`
- `technical_report.pdf`
- `ebay_anchoring_rl_presentation.pptx`
- Final figures used directly in the report

Do not place raw data, parquet files, trained models, private credentials, audit memos, model-comparison transcripts, or working literature-review PDFs here. Those can remain local, but they should not be part of the public submission repository.

## Report outline

1. Problem definition and motivation
2. Prior work: anchoring, bargaining, offline RL, and policy-gradient methods
3. Data and one-step MDP formulation
4. Supervised environment model and behavioral baseline
5. Offline RL with CQL
6. Simulated online RL with faithful PPO
7. Evaluation: expected savings, immediate acceptance, support diagnostics, and sensitivity checks
8. Limitations: one-step framing, simulator bias, support mismatch, and no real online experimentation
9. Conclusion and next steps
