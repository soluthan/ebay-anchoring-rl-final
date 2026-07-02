"""
phase1_supervised.py — Supervised Baselines (XGBoost)
=====================================================
Trains the predictive model the rest of the pipeline depends on:

    deal_classifier.ubj   :  P(deal | s, a)

Because an accepted Best Offer is paid at the buyer's offer, accepted-deal
savings are mechanically:

    savings = 1 - anchor_ratio

Expected savings for candidate offers are therefore:

    P(deal | s, anchor) * (1 - anchor)

No separate price model is trained. This phase also writes:

    clf_metrics.json          : test AUC
    behavioral_benchmark.json : historical policy stats (E[savings], deal rate, anchor dist)
    greedy_metrics.json        : expected savings of a fixed greedy anchor=0.70 policy

Run:
    DATA_DIR=./data MODEL_DIR=./models python phase1_supervised.py
"""

import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from project_constants import (
    ACTION_COL,
    DEAL_COL,
    DEAL_STATUS,
    SEED,
    STATE_COLS,
)

# ── reproducibility ───────────────────────────────────────────────
np.random.seed(SEED)

# ── paths ─────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = STATE_COLS + [ACTION_COL]

GREEDY_ANCHOR = 0.70

N_ESTIMATORS = int(os.environ.get("XGB_N_ESTIMATORS", "400"))
PHASE1_MAX_ROWS = int(os.environ.get("PHASE1_MAX_ROWS", "0"))


# ── helpers ───────────────────────────────────────────────────────
def _features(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURES].values.astype(np.float32)


def _deal_label(df: pd.DataFrame) -> np.ndarray:
    return (df[DEAL_COL].values == DEAL_STATUS).astype(np.int32)


def expected_savings(clf, X: np.ndarray, anchors: np.ndarray):
    p = clf.predict_proba(X)[:, 1]
    savings = np.clip(1.0 - anchors, 0.0, 0.99)
    return p * savings, p, savings


def maybe_sample(df: pd.DataFrame, max_rows: int, name: str) -> pd.DataFrame:
    if max_rows and len(df) > max_rows:
        out = df.sample(max_rows, random_state=SEED).reset_index(drop=True)
        print(f"  [smoke] sampled {name} to {len(out):,} rows (PHASE1_MAX_ROWS)")
        return out
    return df


# ── model training ────────────────────────────────────────────────
def train_classifier(train, val):
    Xtr, ytr = _features(train), _deal_label(train)
    Xv, yv = _features(val), _deal_label(val)
    clf = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", early_stopping_rounds=30,
        random_state=SEED, n_jobs=-1,
    )
    clf.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    return clf


# ── metrics / benchmarks ──────────────────────────────────────────
def classifier_metrics(clf, test) -> dict:
    Xte, yte = _features(test), _deal_label(test)
    auc = (float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
           if len(np.unique(yte)) > 1 else float("nan"))
    return {"test": {"auc": auc}}


def behavioral_benchmark(df) -> dict:
    deals = df[DEAL_COL].values == DEAL_STATUS
    anchors = df[ACTION_COL].values.astype(np.float32)
    savings = np.where(
        deals,
        1.0 - anchors,
        0.0,
    )
    savings = np.clip(savings, 0.0, 0.99)
    return {
        "mean_savings_all": float(savings.mean()),
        "mean_savings_deals_only": float(savings[deals].mean()) if deals.any() else 0.0,
        "deal_rate": float(deals.mean()),
        "mean_anchor_ratio": float(anchors.mean()),
        "std_anchor_ratio": float(anchors.std()),
        "n_test": int(len(df)),
    }


def greedy_benchmark(clf, test) -> dict:
    X = test[STATE_COLS].values.astype(np.float32)
    anchors = np.full(len(X), GREEDY_ANCHOR, dtype=np.float32)
    X = np.column_stack([X, anchors])
    es, p, sv = expected_savings(clf, X, anchors)
    return {
        "greedy_anchor": GREEDY_ANCHOR,
        "greedy_e_savings": float(es.mean()),
        "greedy_mean_p_deal": float(p.mean()),
        "greedy_mean_savings_if_deal": float(sv.mean()),
    }


# ── main ──────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val = pd.read_parquet(DATA_DIR / "val.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    train = maybe_sample(train, PHASE1_MAX_ROWS, "train")
    val = maybe_sample(val, PHASE1_MAX_ROWS, "val")
    test = maybe_sample(test, PHASE1_MAX_ROWS, "test")
    print(f"Loaded train={len(train):,} val={len(val):,} test={len(test):,}")

    print("Training deal classifier …")
    clf = train_classifier(train, val)

    clf.save_model(str(MODEL_DIR / "deal_classifier.ubj"))

    clf_m = classifier_metrics(clf, test)
    bench = behavioral_benchmark(test)
    greedy = greedy_benchmark(clf, test)

    with open(MODEL_DIR / "clf_metrics.json", "w") as f:
        json.dump(clf_m, f, indent=2)
    with open(MODEL_DIR / "behavioral_benchmark.json", "w") as f:
        json.dump(bench, f, indent=2)
    with open(MODEL_DIR / "greedy_metrics.json", "w") as f:
        json.dump(greedy, f, indent=2)

    print("\n── Phase 1 summary ─────────────────────────────────")
    print(f"  Test AUC (P deal)         : {clf_m['test']['auc']:.4f}")
    print(f"  Behavioural E[savings]    : {bench['mean_savings_all']:.4f}")
    print(f"  Behavioural deal rate     : {bench['deal_rate']:.4f}")
    print(f"  Greedy(0.70) E[savings]   : {greedy['greedy_e_savings']:.4f}")
    print(f"\n✅ Phase 1 complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
