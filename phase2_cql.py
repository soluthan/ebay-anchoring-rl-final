"""
phase2_cql.py — Offline RL: Conservative Q-Learning (CQL)
=========================================================
Single-step continuous-action CQL, from scratch in PyTorch.

Q(s, a) is trained to regress the observed reward (1-step Bellman target, since
every episode is terminal) while a CQL penalty pushes DOWN the Q-values of
out-of-distribution anchor ratios — addressing the OOD extrapolation problem.

Saves: cql_best.pt, cql_scaler.pkl, cql_history.json, cql_metrics.json

Compute knobs (env vars):
    CQL_EPOCHS     epochs (default 30)
    CQL_MAX_ROWS   subsample training rows for speed on CPU (default 400000)

Run:
    DATA_DIR=./data MODEL_DIR=./models python phase2_cql.py
"""

import os
import json
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from project_constants import (
    ACTION_COL,
    ANCHOR_MAX,
    ANCHOR_MIN,
    DEAL_COL,
    N_ACTIONS_DISC,
    REWARD_COL,
    SEED,
    STATE_COLS,
)

# ── reproducibility ───────────────────────────────────────────────
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── paths ─────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── MDP columns (match Phase 1 output) ────────────────────────────
EPOCHS = int(os.environ.get("CQL_EPOCHS", "30"))
MAX_ROWS = int(os.environ.get("CQL_MAX_ROWS", "400000"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Q network ─────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, s, a):
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        x = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)


# ── CQL Agent ─────────────────────────────────────────────────────
class CQLAgent:
    def __init__(self, state_dim, hidden=256, lr=3e-4, alpha=5.0,
                 n_neg=10, batch_size=2048):
        self.alpha = alpha
        self.n_neg = n_neg
        self.batch_size = batch_size
        self.q = QNetwork(state_dim, hidden).to(DEVICE)
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.scaler = None
        self.action_grid = torch.linspace(
            ANCHOR_MIN, ANCHOR_MAX, N_ACTIONS_DISC, device=DEVICE
        )
        self.history = {"td_loss": [], "cql_loss": [], "val_loss": []}

    # ── training ──────────────────────────────────────────────────
    def fit(self, S, A, R, S_val, A_val, R_val, epochs=EPOCHS):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        S = self.scaler.fit_transform(S).astype(np.float32)
        Sv = self.scaler.transform(S_val).astype(np.float32)

        S = torch.tensor(S, device=DEVICE)
        A = torch.tensor(A, device=DEVICE).float()
        R = torch.tensor(R, device=DEVICE).float()
        S_v = torch.tensor(Sv, device=DEVICE)
        A_v = torch.tensor(A_val, device=DEVICE).float()
        R_v = torch.tensor(R_val, device=DEVICE).float()

        loader = DataLoader(
            TensorDataset(S, A, R), batch_size=self.batch_size, shuffle=True
        )

        best = float("inf")
        for ep in range(epochs):
            self.q.train()
            ep_td, ep_cql = [], []
            for s_b, a_b, r_b in loader:
                q_sa = self.q(s_b, a_b)
                td = nn.functional.mse_loss(q_sa, r_b)

                B = s_b.shape[0]
                rand_a = torch.rand(B, self.n_neg, device=DEVICE)
                rand_a = ANCHOR_MIN + rand_a * (ANCHOR_MAX - ANCHOR_MIN)
                s_exp = s_b.unsqueeze(1).expand(-1, self.n_neg, -1).reshape(B * self.n_neg, -1)
                a_exp = rand_a.reshape(-1)
                q_ood = self.q(s_exp, a_exp).reshape(B, self.n_neg)
                cql = (torch.logsumexp(q_ood, dim=1) - q_sa).mean()

                loss = td + self.alpha * cql
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                ep_td.append(td.item())
                ep_cql.append(cql.item())

            self.q.eval()
            with torch.no_grad():
                val_q = self.q(S_v, A_v)
                val_loss = nn.functional.mse_loss(val_q, R_v).item()

            self.history["td_loss"].append(float(np.mean(ep_td)))
            self.history["cql_loss"].append(float(np.mean(ep_cql)))
            self.history["val_loss"].append(float(val_loss))

            if val_loss < best:
                best = val_loss
                torch.save(self.q.state_dict(), MODEL_DIR / "cql_best.pt")
            if ep % 5 == 0:
                print(f"Epoch {ep:>3} | td={self.history['td_loss'][-1]:.4f} "
                      f"cql={self.history['cql_loss'][-1]:.4f} val={val_loss:.4f}")

        self.q.load_state_dict(torch.load(MODEL_DIR / "cql_best.pt", map_location=DEVICE))

    # ── policy (chunked to bound memory) ──────────────────────────
    def act_batch(self, S_np, chunk=2048):
        self.q.eval()
        outs = []
        G = N_ACTIONS_DISC
        for i in range(0, len(S_np), chunk):
            s = self.scaler.transform(S_np[i:i + chunk]).astype(np.float32)
            s = torch.tensor(s, device=DEVICE)
            B = s.shape[0]
            s_exp = s.unsqueeze(1).expand(-1, G, -1).reshape(B * G, -1)
            a_exp = self.action_grid.unsqueeze(0).expand(B, -1).reshape(-1)
            with torch.no_grad():
                q = self.q(s_exp, a_exp).reshape(B, G)
            outs.append(self.action_grid[q.argmax(dim=1)].cpu().numpy())
        return np.concatenate(outs)

    def save_scaler(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)


# ── Legacy proxy diagnostic ───────────────────────────────────────
def heuristic_ope_ips(a_hist, a_pred, r):
    """Very rough continuous-action IPS proxy.

    The report-grade OPE lives in ope.py, where propensities are estimated from
    state, kernel bandwidth sensitivity is reported, and ESS/CIs are written.
    This shortcut remains only as a lightweight Phase-2 smoke diagnostic.
    """
    from scipy.stats import norm
    mu, sig = a_hist.mean(), a_hist.std() + 1e-8
    p_b = norm.pdf(a_hist, mu, sig) + 1e-8
    p_p = norm.pdf(a_hist, a_pred, 0.05) + 1e-8
    w = np.clip(p_p / p_b, 0, 10)
    return float((w * r).mean())


# ── evaluation ────────────────────────────────────────────────────
def evaluate(agent, test, clf) -> dict:
    S = test[STATE_COLS].values.astype(np.float32)
    A = test[ACTION_COL].values.astype(np.float32)
    R = test[REWARD_COL].values.astype(np.float32)

    a_cql = agent.act_batch(S)
    feat = np.column_stack([S, a_cql]).astype(np.float32)   # RAW features for XGBoost
    p = clf.predict_proba(feat)[:, 1]

    savings = np.clip(1.0 - a_cql, 0.0, 0.99)
    e_savings_sim = float((p * savings).mean())
    ope = heuristic_ope_ips(A, a_cql, R)

    metrics = {
        "cql_e_savings_sim": e_savings_sim,
        "cql_ope_ips": ope,
        "cql_ope_ips_note": "Legacy heuristic only; use ope.py for SNIPS/DR/ESS/CIs.",
        "cql_mean_anchor": float(a_cql.mean()),
        "cql_std_anchor": float(a_cql.std()),
        "cql_mean_p_deal": float(p.mean()),
    }
    print("\n── CQL evaluation ─────────────────────────────────")
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.floating)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    return metrics


# ── main ──────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    if len(train) > MAX_ROWS:
        train = train.sample(MAX_ROWS, random_state=SEED).reset_index(drop=True)
        print(f"Subsampled training to {len(train):,} rows (CQL_MAX_ROWS).")
    val = pd.read_parquet(DATA_DIR / "val.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")

    S, A, R = (train[STATE_COLS].values, train[ACTION_COL].values, train[REWARD_COL].values)
    Sv, Av, Rv = (val[STATE_COLS].values, val[ACTION_COL].values, val[REWARD_COL].values)

    agent = CQLAgent(len(STATE_COLS))
    agent.fit(S, A, R, Sv, Av, Rv)

    agent.save_scaler(MODEL_DIR / "cql_scaler.pkl")
    with open(MODEL_DIR / "cql_history.json", "w") as f:
        json.dump(agent.history, f, indent=2)

    import xgboost as xgb
    clf = xgb.XGBClassifier(); clf.load_model(str(MODEL_DIR / "deal_classifier.ubj"))

    metrics = evaluate(agent, test, clf)
    with open(MODEL_DIR / "cql_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Phase 2 complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
