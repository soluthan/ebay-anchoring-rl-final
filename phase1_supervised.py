"""Phase 1: supervised immediate-acceptance model and simple baselines.

The classifier estimates ``P(opening offer accepted | state, anchor)``.  The
fixed 0.70 policy is a rule-based baseline; the supervised greedy policy is the
actual grid-search maximizer of ``P(accept) * (1-anchor)``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from policy_utils import ACTION_GRID, greedy_policy, score_actions, support_flags
from project_constants import ACTION_COL, CLASSIFIER_FILE, LABEL_COL, SEED, STATE_COLS


np.random.seed(SEED)

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = STATE_COLS + [ACTION_COL]
FIXED_ANCHOR = 0.70
N_ESTIMATORS = int(os.environ.get("XGB_N_ESTIMATORS", "400"))
PHASE1_MAX_ROWS = int(os.environ.get("PHASE1_MAX_ROWS", "0"))


def _features(frame: pd.DataFrame) -> np.ndarray:
    return frame[FEATURES].values.astype(np.float32)


def _acceptance_label(frame: pd.DataFrame) -> np.ndarray:
    return frame[LABEL_COL].values.astype(np.int32)


def maybe_sample(frame: pd.DataFrame, max_rows: int, name: str) -> pd.DataFrame:
    if max_rows and len(frame) > max_rows:
        sampled = frame.sample(max_rows, random_state=SEED).reset_index(drop=True)
        print(f"  [smoke] sampled {name} to {len(sampled):,} rows")
        return sampled
    return frame


def train_classifier(train: pd.DataFrame, val: pd.DataFrame):
    model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(
        _features(train),
        _acceptance_label(train),
        eval_set=[(_features(val), _acceptance_label(val))],
        verbose=False,
    )
    return model


def calibration_table(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> pd.DataFrame:
    table = pd.DataFrame({"y": y_true, "p": probability})
    # Rank-based bins remain stable when the model emits repeated probabilities.
    bins = max(1, min(bins, len(table)))
    table["bin"] = pd.qcut(
        table["p"].rank(method="first"), q=bins, labels=False, duplicates="drop"
    )
    return (
        table.groupby("bin", as_index=False)
        .agg(n=("y", "size"), mean_predicted_probability=("p", "mean"), observed_acceptance=("y", "mean"))
    )


def classifier_metrics(model, test: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    y_true = _acceptance_label(test)
    probability = model.predict_proba(_features(test))[:, 1]
    auc = float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) > 1 else float("nan")
    metrics = {
        "test": {
            "auc": auc,
            "brier": float(brier_score_loss(y_true, probability)),
            "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
            "opening_acceptance_rate": float(y_true.mean()),
            "mean_predicted_acceptance": float(probability.mean()),
            "n": int(len(y_true)),
        }
    }
    return metrics, calibration_table(y_true, probability)


def behavioral_benchmark(frame: pd.DataFrame) -> dict:
    accepted = frame[LABEL_COL].values.astype(bool)
    anchors = frame[ACTION_COL].values.astype(np.float32)
    rewards = accepted.astype(np.float32) * (1.0 - anchors)
    return {
        "evidence_type": "observed immediate opening-offer outcomes",
        "mean_savings_all": float(rewards.mean()),
        "mean_savings_accepted_only": float(rewards[accepted].mean()) if accepted.any() else 0.0,
        "opening_acceptance_rate": float(accepted.mean()),
        "mean_anchor_ratio": float(anchors.mean()),
        "std_anchor_ratio": float(anchors.std()),
        "n_test": int(len(frame)),
    }


def fixed_anchor_benchmark(model, test: pd.DataFrame, p5: float, p95: float) -> dict:
    states = test[STATE_COLS].values.astype(np.float32)
    anchors = np.full(len(states), FIXED_ANCHOR, dtype=np.float32)
    probability, expected = score_actions(model, states, anchors)
    return {
        "policy": "fixed_anchor_0.70",
        "evidence_type": "Phase-1 model estimate",
        "mean_anchor": FIXED_ANCHOR,
        "mean_p_accept": float(probability.mean()),
        "mean_expected_savings": float(expected.mean()),
        "within_p5_p95_support_fraction": float(support_flags(anchors, p5, p95).mean()),
    }


def supervised_greedy_benchmark(model, test: pd.DataFrame, p5: float, p95: float) -> dict:
    states = test[STATE_COLS].values.astype(np.float32)
    anchors, probability, expected, best_indices = greedy_policy(model, states)
    interior = (best_indices > 0) & (best_indices < len(ACTION_GRID) - 1)
    return {
        "policy": "supervised_greedy",
        "evidence_type": "Phase-1 model estimate",
        "mean_anchor": float(anchors.mean()),
        "std_anchor": float(anchors.std()),
        "mean_p_accept": float(probability.mean()),
        "mean_expected_savings": float(expected.mean()),
        "within_p5_p95_support_fraction": float(support_flags(anchors, p5, p95).mean()),
        "interior_optimum_fraction": float(interior.mean()),
        "lower_boundary_optimum_fraction": float((best_indices == 0).mean()),
        "upper_boundary_optimum_fraction": float((best_indices == len(ACTION_GRID) - 1).mean()),
    }


def anchor_response_curve(model, test: pd.DataFrame) -> pd.DataFrame:
    states = test[STATE_COLS].values.astype(np.float32)
    # A deterministic sample keeps the diagnostic cheap on the production data.
    if len(states) > 100_000:
        rng = np.random.default_rng(SEED)
        states = states[rng.choice(len(states), 100_000, replace=False)]
    rows = []
    for anchor in ACTION_GRID:
        actions = np.full(len(states), anchor, dtype=np.float32)
        probability, expected = score_actions(model, states, actions)
        rows.append(
            {
                "anchor_ratio": float(anchor),
                "mean_predicted_acceptance": float(probability.mean()),
                "mean_predicted_expected_savings": float(expected.mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    started = time.time()
    train = maybe_sample(pd.read_parquet(DATA_DIR / "train.parquet"), PHASE1_MAX_ROWS, "train")
    val = maybe_sample(pd.read_parquet(DATA_DIR / "val.parquet"), PHASE1_MAX_ROWS, "val")
    test = maybe_sample(pd.read_parquet(DATA_DIR / "test.parquet"), PHASE1_MAX_ROWS, "test")
    print(f"Loaded train={len(train):,} val={len(val):,} test={len(test):,}")

    print("Training immediate opening-offer acceptance classifier ...")
    model = train_classifier(train, val)
    model.save_model(str(MODEL_DIR / CLASSIFIER_FILE))

    metrics, calibration = classifier_metrics(model, test)
    behavior = behavioral_benchmark(test)
    p5, p95 = np.percentile(train[ACTION_COL].values, [5, 95])
    fixed = fixed_anchor_benchmark(model, test, float(p5), float(p95))
    greedy = supervised_greedy_benchmark(model, test, float(p5), float(p95))

    with open(MODEL_DIR / "clf_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with open(MODEL_DIR / "behavioral_benchmark.json", "w", encoding="utf-8") as file:
        json.dump(behavior, file, indent=2)
    with open(MODEL_DIR / "fixed_anchor_metrics.json", "w", encoding="utf-8") as file:
        json.dump(fixed, file, indent=2)
    with open(MODEL_DIR / "greedy_metrics.json", "w", encoding="utf-8") as file:
        json.dump(greedy, file, indent=2)
    calibration.to_csv(MODEL_DIR / "clf_calibration.csv", index=False)
    anchor_response_curve(model, test).to_csv(MODEL_DIR / "anchor_response_curve.csv", index=False)

    test_metrics = metrics["test"]
    print("\n-- Phase 1 summary --------------------------------")
    print(f"  Test AUC                       : {test_metrics['auc']:.4f}")
    print(f"  Test Brier score               : {test_metrics['brier']:.4f}")
    print(f"  Observed immediate acceptance  : {behavior['opening_acceptance_rate']:.4f}")
    print(f"  Observed E[immediate savings]  : {behavior['mean_savings_all']:.4f}")
    print(f"  Fixed 0.70 model E[savings]    : {fixed['mean_expected_savings']:.4f}")
    print(f"  Greedy model E[savings]        : {greedy['mean_expected_savings']:.4f}")
    print(f"  Greedy interior optima         : {greedy['interior_optimum_fraction']:.1%}")
    print(f"\nPhase 1 complete in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
