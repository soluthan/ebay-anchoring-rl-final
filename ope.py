"""
ope.py - Continuous-action off-policy evaluation diagnostics
============================================================

This module estimates how much support deterministic offer policies have under
the logged buyer behavior. It is intentionally diagnostic rather than causal:
the eBay data does not contain randomized propensities, so propensities are
estimated from a behavior model pi_b(a | s).

Outputs:
    outputs/ope_policy_eval.csv
    outputs/ope_weight_diagnostics.csv
    outputs/ope_summary.json
    models/ope_behavior_model.pkl

Run after Phase 1:
    DATA_DIR=./data MODEL_DIR=./models OUTPUT_DIR=./outputs python ope.py

Useful knobs:
    OPE_BANDWIDTHS=0.03,0.05,0.10
    OPE_BOOTSTRAP=200
    OPE_MAX_ROWS=200000
    OPE_BEHAVIOR_MAX_ROWS=300000
    OPE_WEIGHT_CLIP=20
"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from project_constants import (
    ACTION_COL,
    ANCHOR_MAX,
    ANCHOR_MIN,
    LIST_COL,
    N_GRID,
    REWARD_COL,
    SEED,
    STATE_COLS,
)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BANDWIDTHS = [
    float(x) for x in os.environ.get("OPE_BANDWIDTHS", "0.03,0.05,0.10").split(",")
    if x.strip()
]
N_BOOTSTRAP = int(os.environ.get("OPE_BOOTSTRAP", "200"))
MAX_ROWS = int(os.environ.get("OPE_MAX_ROWS", "200000"))
BEHAVIOR_MAX_ROWS = int(os.environ.get("OPE_BEHAVIOR_MAX_ROWS", "300000"))
WEIGHT_CLIP = float(os.environ.get("OPE_WEIGHT_CLIP", "20"))
INCLUDE_CQL = os.environ.get("OPE_INCLUDE_CQL", "1") == "1"


@dataclass
class BehaviorPolicy:
    scaler: StandardScaler
    model: HistGradientBoostingRegressor
    residual_sigma: float

    def predict_mu(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        x = self.scaler.transform(states)
        return np.clip(self.model.predict(x), ANCHOR_MIN, ANCHOR_MAX)

    def density(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        mu = self.predict_mu(states)
        sigma = max(float(self.residual_sigma), 1e-3)
        return np.maximum(norm.pdf(actions, loc=mu, scale=sigma), 1e-12)


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def train_behavior_policy(train: pd.DataFrame) -> BehaviorPolicy:
    if len(train) > BEHAVIOR_MAX_ROWS:
        train = train.sample(BEHAVIOR_MAX_ROWS, random_state=SEED).reset_index(drop=True)

    states = train[STATE_COLS].values.astype(np.float32)
    actions = train[ACTION_COL].values.astype(np.float32)

    scaler = StandardScaler()
    x = scaler.fit_transform(states)
    model = HistGradientBoostingRegressor(
        max_iter=180,
        max_leaf_nodes=31,
        learning_rate=0.06,
        l2_regularization=0.01,
        random_state=SEED,
    )
    model.fit(x, actions)
    pred = np.clip(model.predict(x), ANCHOR_MIN, ANCHOR_MAX)
    residual_sigma = float(np.std(actions - pred))
    residual_sigma = max(residual_sigma, 0.02)
    return BehaviorPolicy(scaler=scaler, model=model, residual_sigma=residual_sigma)


def reward_model_values(clf, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    feat = np.column_stack([states, actions]).astype(np.float32)
    p_deal = clf.predict_proba(feat)[:, 1]
    savings = np.clip(1.0 - actions, 0.0, 0.99)
    return np.clip(p_deal * savings, 0.0, 0.99)


def greedy_actions(clf, states: np.ndarray) -> np.ndarray:
    grid = np.linspace(ANCHOR_MIN, ANCHOR_MAX, N_GRID, dtype=np.float32)
    outs = []
    chunk = 4096
    for start in range(0, len(states), chunk):
        s = states[start:start + chunk]
        b = len(s)
        s_rep = np.repeat(s, N_GRID, axis=0)
        a_rep = np.tile(grid, b)
        feat = np.column_stack([s_rep, a_rep]).astype(np.float32)
        p = clf.predict_proba(feat)[:, 1].reshape(b, N_GRID)
        expected = p * (1.0 - grid)[None, :]
        outs.append(grid[expected.argmax(axis=1)])
    return np.concatenate(outs)


def cql_artifacts_available() -> bool:
    return (MODEL_DIR / "cql_best.pt").exists() and (MODEL_DIR / "cql_scaler.pkl").exists()


def preload_torch_for_cql():
    if not (INCLUDE_CQL and cql_artifacts_available()):
        return None
    import torch

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    return torch


def cql_actions(states: np.ndarray) -> np.ndarray | None:
    if not (INCLUDE_CQL and cql_artifacts_available()):
        return None

    try:
        import torch
        from phase2_cql import QNetwork

        with open(MODEL_DIR / "cql_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        net = QNetwork(len(STATE_COLS))
        net.load_state_dict(torch.load(MODEL_DIR / "cql_best.pt", map_location="cpu"))
        net.eval()

        grid = torch.linspace(ANCHOR_MIN, ANCHOR_MAX, N_GRID)
        outs = []
        chunk = 4096
        for start in range(0, len(states), chunk):
            s_np = scaler.transform(states[start:start + chunk]).astype(np.float32)
            s = torch.tensor(s_np)
            b = s.shape[0]
            s_exp = s.unsqueeze(1).expand(-1, N_GRID, -1).reshape(b * N_GRID, -1)
            a_exp = grid.unsqueeze(0).expand(b, -1).reshape(-1)
            with torch.no_grad():
                q = net(s_exp, a_exp).reshape(b, N_GRID)
            outs.append(grid[q.argmax(dim=1)].numpy())
        return np.concatenate(outs).astype(np.float32)
    except Exception as exc:
        print(f"  [ope] skipped CQL target policy ({exc})")
        return None


def policy_action_table(test: pd.DataFrame, clf) -> dict[str, np.ndarray]:
    states = test[STATE_COLS].values.astype(np.float32)
    policies = {
        "behavioral_logged": test[ACTION_COL].values.astype(np.float32),
        "fixed_anchor_0.70": np.full(len(test), 0.70, dtype=np.float32),
        "greedy_model": greedy_actions(clf, states),
    }
    cql = cql_actions(states)
    if cql is not None:
        policies["cql_offline_rl"] = cql

    external_path = os.environ.get("OPE_TARGET_ACTIONS_CSV")
    if external_path:
        column = os.environ.get("OPE_TARGET_ACTION_COL", "target_anchor")
        ext = pd.read_csv(external_path)
        if column not in ext.columns:
            raise ValueError(f"{external_path} must contain column {column!r}.")
        if len(ext) != len(test):
            raise ValueError(
                "External OPE action CSV must have the same row count/order as test.parquet."
            )
        policies[os.environ.get("OPE_TARGET_POLICY_NAME", "external_policy")] = (
            ext[column].values.astype(np.float32)
        )
    return policies


def effective_sample_size(weights: np.ndarray) -> float:
    denom = float(np.square(weights).sum())
    if denom <= 0:
        return 0.0
    return float(np.square(weights.sum()) / denom)


def kernel_weights(
    behavior: BehaviorPolicy,
    states: np.ndarray,
    logged_actions: np.ndarray,
    target_actions: np.ndarray,
    bandwidth: float,
    clip: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    behavior_density = behavior.density(states, logged_actions)
    kernel_density = norm.pdf(logged_actions, loc=target_actions, scale=bandwidth)
    raw = kernel_density / behavior_density
    if clip is None:
        return raw, behavior_density
    return np.clip(raw, 0.0, clip), behavior_density


def summarize_weights(weights: np.ndarray) -> dict:
    return {
        "weight_mean": float(weights.mean()),
        "weight_std": float(weights.std()),
        "weight_p50": float(np.percentile(weights, 50)),
        "weight_p90": float(np.percentile(weights, 90)),
        "weight_p95": float(np.percentile(weights, 95)),
        "weight_p99": float(np.percentile(weights, 99)),
        "weight_max": float(weights.max()),
        "zero_weight_frac": float((weights <= 1e-12).mean()),
        "ess": effective_sample_size(weights),
        "ess_frac": effective_sample_size(weights) / max(len(weights), 1),
    }


def point_estimates(
    rewards: np.ndarray,
    q_logged: np.ndarray,
    q_target: np.ndarray,
    weights: np.ndarray,
) -> dict:
    weight_sum = float(weights.sum())
    ips = float(np.mean(weights * rewards))
    snips = float(np.sum(weights * rewards) / weight_sum) if weight_sum > 0 else float("nan")
    dr_terms = q_target + weights * (rewards - q_logged)
    dr = float(np.mean(dr_terms))
    sndr = (
        float(np.mean(q_target) + np.sum(weights * (rewards - q_logged)) / weight_sum)
        if weight_sum > 0
        else float("nan")
    )
    return {
        "logged_reward_mean": float(np.mean(rewards)),
        "reward_model_mean": float(np.mean(q_target)),
        "ips": ips,
        "snips": snips,
        "dr": dr,
        "sndr": sndr,
    }


def bootstrap_ci(
    rewards: np.ndarray,
    q_logged: np.ndarray,
    q_target: np.ndarray,
    weights: np.ndarray,
    n_bootstrap: int,
) -> dict:
    if n_bootstrap <= 0:
        return {}

    rng = np.random.default_rng(SEED)
    n = len(rewards)
    values = {"ips": [], "snips": [], "dr": [], "sndr": []}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        est = point_estimates(rewards[idx], q_logged[idx], q_target[idx], weights[idx])
        for key in values:
            values[key].append(est[key])

    out = {}
    for key, vals in values.items():
        arr = np.asarray(vals, dtype=np.float64)
        out[f"{key}_ci_low"] = float(np.nanpercentile(arr, 2.5))
        out[f"{key}_ci_high"] = float(np.nanpercentile(arr, 97.5))
    return out


def evaluate_policy(
    policy_name: str,
    target_actions: np.ndarray,
    behavior: BehaviorPolicy,
    clf,
    test: pd.DataFrame,
    bandwidth: float,
    clipped: bool,
) -> tuple[dict, dict]:
    states = test[STATE_COLS].values.astype(np.float32)
    logged_actions = test[ACTION_COL].values.astype(np.float32)
    rewards = test[REWARD_COL].values.astype(np.float32)
    target_actions = np.clip(target_actions.astype(np.float32), ANCHOR_MIN, ANCHOR_MAX)

    clip = WEIGHT_CLIP if clipped else None
    weights, behavior_density = kernel_weights(
        behavior, states, logged_actions, target_actions, bandwidth, clip
    )
    q_logged = reward_model_values(clf, states, logged_actions)
    q_target = reward_model_values(clf, states, target_actions)

    estimates = point_estimates(rewards, q_logged, q_target, weights)
    estimates.update(bootstrap_ci(rewards, q_logged, q_target, weights, N_BOOTSTRAP))
    estimates.update(
        {
            "policy": policy_name,
            "bandwidth": bandwidth,
            "weights": "clipped" if clipped else "unclipped",
            "n": int(len(test)),
            "target_anchor_mean": float(target_actions.mean()),
            "target_anchor_std": float(target_actions.std()),
            "behavior_density_p5": float(np.percentile(behavior_density, 5)),
            "behavior_density_p50": float(np.percentile(behavior_density, 50)),
        }
    )

    diagnostics = summarize_weights(weights)
    diagnostics.update(
        {
            "policy": policy_name,
            "bandwidth": bandwidth,
            "weights": "clipped" if clipped else "unclipped",
            "n": int(len(test)),
        }
    )
    return estimates, diagnostics


def maybe_sample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows > 0 and len(df) > max_rows:
        return df.sample(max_rows, random_state=SEED).reset_index(drop=True)
    return df.reset_index(drop=True)


def main():
    t0 = time.time()
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    test = maybe_sample(test, MAX_ROWS)

    preload_torch_for_cql()

    import xgboost as xgb

    clf = xgb.XGBClassifier()
    clf.load_model(str(MODEL_DIR / "deal_classifier.ubj"))

    print("Training estimated behavior policy pi_b(a | s) ...")
    behavior = train_behavior_policy(train)
    with open(MODEL_DIR / "ope_behavior_model.pkl", "wb") as f:
        pickle.dump(behavior, f)

    print(f"Behavior residual sigma: {behavior.residual_sigma:.4f}")
    print("Building target policy actions ...")
    policies = policy_action_table(test, clf)

    eval_rows = []
    diagnostic_rows = []
    for policy_name, target_actions in policies.items():
        for bandwidth in BANDWIDTHS:
            for clipped in (True, False):
                estimates, diagnostics = evaluate_policy(
                    policy_name, target_actions, behavior, clf, test, bandwidth, clipped
                )
                eval_rows.append(estimates)
                diagnostic_rows.append(diagnostics)

    eval_df = pd.DataFrame(eval_rows).sort_values(["policy", "bandwidth", "weights"])
    diag_df = pd.DataFrame(diagnostic_rows).sort_values(["policy", "bandwidth", "weights"])
    eval_df.to_csv(OUTPUT_DIR / "ope_policy_eval.csv", index=False)
    diag_df.to_csv(OUTPUT_DIR / "ope_weight_diagnostics.csv", index=False)

    summary = {
        "framing": (
            "Propensities are estimated, not logged. Treat SNIPS/DR as diagnostics "
            "for support mismatch rather than causal marketplace lift."
        ),
        "bandwidths": BANDWIDTHS,
        "bootstrap_replicates": N_BOOTSTRAP,
        "weight_clip": WEIGHT_CLIP,
        "n_test_evaluated": int(len(test)),
        "policies": sorted(policies.keys()),
        "files": {
            "policy_eval": str(OUTPUT_DIR / "ope_policy_eval.csv"),
            "weight_diagnostics": str(OUTPUT_DIR / "ope_weight_diagnostics.csv"),
            "behavior_model": str(MODEL_DIR / "ope_behavior_model.pkl"),
        },
    }
    with open(OUTPUT_DIR / "ope_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nOPE diagnostics saved:")
    print(f"  {OUTPUT_DIR / 'ope_policy_eval.csv'}")
    print(f"  {OUTPUT_DIR / 'ope_weight_diagnostics.csv'}")
    print("\nReminder: small ESS or wide CIs are a finding, not a failure.")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
