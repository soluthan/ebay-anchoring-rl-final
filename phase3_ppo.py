"""
phase3_ppo.py — Online RL: PPO on a simulated marketplace (+ faithful-sim robustness)
=====================================================================================
1. EbayBargainingEnv — acceptance dynamics use the Phase-1 deal model.
2. PPO agent (clipped surrogate) from scratch in PyTorch.

ROBUSTNESS VARIANT (H3). The naive simulator lets PPO exploit out-of-distribution
blind spots: it offers ~1% of list because the classifier still predicts non-trivial
acceptance there. The *faithful* simulator discounts P(accept) for anchors below the
historical 5th percentile — encoding "we don't believe acceptance predictions for
offers no buyer ever made." Run both and compare:

    PPO_FAITHFUL=0 python phase3_ppo.py     # naive sim  -> PPO exploits (anchor ~0.01)
    PPO_FAITHFUL=1 python phase3_ppo.py     # faithful   -> PPO stays realistic   (default)
    PPO_FAITHFUL_P5=0.35 python phase3_ppo.py  # smoke/investigation threshold override

Outputs are tagged: faithful runs write ppo_*_faithful.json so both survive for the
comparison figure.

FIX (#5, retained): env feeds RAW features to XGBoost, normalises only the policy obs.
NOTE: torch imported before xgboost (xgboost is imported inside main) to avoid the
macOS OpenMP deadlock.

Run:  DATA_DIR=./data MODEL_DIR=./models python phase3_ppo.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from gymnasium import spaces

from project_constants import (
    ACTION_COL,
    ANCHOR_MAX,
    ANCHOR_MIN,
    SEED,
    STATE_COLS,
)

# ── reproducibility ────────────────────────────────────────────────────────────
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── MDP constants ───────────────────────────────────────────────────────────────
N_STATE = len(STATE_COLS)

TOTAL_STEPS = int(os.environ.get("PPO_STEPS", "200000"))
ROLLOUT_LEN = int(os.environ.get("PPO_ROLLOUT", "2048"))
EVAL_EPISODES = int(os.environ.get("PPO_EVAL_EPISODES", "5000"))
FAITHFUL = os.environ.get("PPO_FAITHFUL", "1") == "1"
FAITHFUL_BETA = float(os.environ.get("PPO_FAITHFUL_BETA", "8.0"))
FAITHFUL_P5_OVERRIDE = os.environ.get("PPO_FAITHFUL_P5")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ────────────────────────────────────────────────────────────────────────────────
class EbayBargainingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, states_raw, clf, scaler,
                 noise_std=0.03, hard_reward=False,
                 faithful=False, anchor_p5=0.30, faithful_beta=8.0):
        super().__init__()
        self.states_raw = np.asarray(states_raw, dtype=np.float32)
        self.clf = clf
        self.scaler = scaler
        self.noise_std = noise_std
        self.hard_reward = hard_reward
        self.faithful = faithful
        self.anchor_p5 = float(anchor_p5)
        self.faithful_beta = float(faithful_beta)
        self.rng = np.random.default_rng(SEED)

        self.observation_space = spaces.Box(-10.0, 10.0, (N_STATE,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self._raw = self._obs = None

    @staticmethod
    def _unnorm_action(a_norm: float) -> float:
        a_norm = np.clip(a_norm, -1.0, 1.0)
        return ANCHOR_MIN + (a_norm + 1.0) / 2.0 * (ANCHOR_MAX - ANCHOR_MIN)

    def _to_obs(self, raw_row):
        return self.scaler.transform(raw_row.reshape(1, -1)).astype(np.float32)[0]

    def _support_weight(self, anchor):
        """1.0 within support; decays toward 0 for anchors below the historical p5."""
        if not self.faithful or anchor >= self.anchor_p5:
            return 1.0
        gap = (self.anchor_p5 - anchor) / max(self.anchor_p5, 1e-6)
        return float(np.exp(-self.faithful_beta * gap))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = self.rng.integers(0, len(self.states_raw))
        self._raw = self.states_raw[idx].copy()
        self._obs = self._to_obs(self._raw)
        return self._obs, {}

    def step(self, action):
        anchor = self._unnorm_action(float(action[0]))
        feat = np.append(self._raw, anchor).reshape(1, -1).astype(np.float32)

        p_deal = float(self.clf.predict_proba(feat)[0, 1])
        p_deal = float(np.clip(p_deal + self.rng.normal(0, self.noise_std), 0.0, 0.99))
        p_deal *= self._support_weight(anchor)              # faithfulness discount
        savings = float(np.clip(1.0 - anchor, 0.0, 0.99))

        if self.hard_reward:
            reward = float(int(self.rng.random() < p_deal) * savings)
        else:
            reward = float(p_deal * savings)

        info = {"anchor_ratio": anchor, "p_deal": p_deal,
                "pred_savings": savings, "reward": reward}
        return self._obs, reward, True, False, info


# ────────────────────────────────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor_mu = nn.Linear(hidden, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))
        self.critic = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mu.weight, gain=0.01)

    def forward(self, state):
        x = self.backbone(state)
        mu = torch.tanh(self.actor_mu(x))
        std = torch.exp(self.actor_log_std).expand_as(mu).clamp(0.01, 1.0)
        val = self.critic(x).squeeze(-1)
        return mu, std, val

    def get_dist(self, state):
        mu, std, val = self.forward(state)
        return Normal(mu, std), val


class PPOAgent:
    def __init__(self, state_dim=N_STATE, hidden=128, lr=3e-4, clip_eps=0.2,
                 n_epochs=10, gae_lambda=0.95, gamma=0.99, vf_coef=0.5,
                 entropy_coef=0.01, max_grad_norm=0.5, batch_size=64, tag=""):
        self.clip_eps = clip_eps; self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda; self.gamma = gamma
        self.vf_coef = vf_coef; self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm; self.batch_size = batch_size
        self.tag = tag
        self.ac = ActorCritic(state_dim, hidden).to(DEVICE)
        self.opt = optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)

    def collect_rollout(self, env, n_steps):
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        obs, _ = env.reset()
        for _ in range(n_steps):
            s_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                dist, val = self.ac.get_dist(s_t)
                act = dist.sample().clamp(-1.0, 1.0)
                lp = dist.log_prob(act).sum(-1)
            act_np = act.cpu().numpy()[0]
            next_obs, rew, term, trunc, _ = env.step(act_np)
            states.append(obs); actions.append(act_np); log_probs.append(lp.item())
            rewards.append(rew); values.append(val.item()); dones.append(float(term or trunc))
            obs, _ = env.reset() if (term or trunc) else (next_obs, None)
        return {"states": np.array(states, np.float32), "actions": np.array(actions, np.float32),
                "log_probs": np.array(log_probs, np.float32), "rewards": np.array(rewards, np.float32),
                "values": np.array(values, np.float32), "dones": np.array(dones, np.float32)}

    def compute_gae(self, rollout, last_val=0.0):
        T = len(rollout["rewards"]); advs = np.zeros(T, np.float32); gae = 0.0
        for t in reversed(range(T)):
            next_val = rollout["values"][t + 1] if t < T - 1 else last_val
            delta = rollout["rewards"][t] + self.gamma * next_val * (1 - rollout["dones"][t]) - rollout["values"][t]
            gae = delta + self.gamma * self.gae_lambda * (1 - rollout["dones"][t]) * gae
            advs[t] = gae
        return advs

    def update(self, rollout):
        advs = self.compute_gae(rollout); returns = advs + rollout["values"]
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        S = torch.tensor(rollout["states"], device=DEVICE)
        A = torch.tensor(rollout["actions"], device=DEVICE)
        LP = torch.tensor(rollout["log_probs"], device=DEVICE)
        RET = torch.tensor(returns, device=DEVICE); ADV = torch.tensor(advs, device=DEVICE)
        N = len(S); metrics = {"pg_loss": [], "vf_loss": [], "entropy": [], "total_loss": []}
        for _ in range(self.n_epochs):
            idx = np.random.permutation(N)
            for start in range(0, N, self.batch_size):
                mb = idx[start: start + self.batch_size]
                dist, val = self.ac.get_dist(S[mb])
                new_lp = dist.log_prob(A[mb]).sum(-1); entropy = dist.entropy().mean()
                ratio = torch.exp(new_lp - LP[mb])
                pg1 = ratio * ADV[mb]
                pg2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * ADV[mb]
                pg_loss = -torch.min(pg1, pg2).mean()
                vf_loss = nn.functional.mse_loss(val, RET[mb])
                loss = pg_loss + self.vf_coef * vf_loss - self.entropy_coef * entropy
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm); self.opt.step()
                metrics["pg_loss"].append(pg_loss.item()); metrics["vf_loss"].append(vf_loss.item())
                metrics["entropy"].append(entropy.item()); metrics["total_loss"].append(loss.item())
        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def train(self, env, total_steps=TOTAL_STEPS, rollout_len=ROLLOUT_LEN):
        history = {"step": [], "mean_reward": [], "pg_loss": [], "entropy": []}
        steps, best = 0, -float("inf")
        print(f"PPO training for {total_steps:,} steps  (faithful={env.faithful}) …")
        while steps < total_steps:
            rollout = self.collect_rollout(env, rollout_len)
            metrics = self.update(rollout)
            mean_rew = float(rollout["rewards"].mean()); steps += rollout_len
            history["step"].append(steps); history["mean_reward"].append(mean_rew)
            history["pg_loss"].append(metrics["pg_loss"]); history["entropy"].append(metrics["entropy"])
            if mean_rew > best:
                best = mean_rew; torch.save(self.ac.state_dict(), MODEL_DIR / f"ppo_best{self.tag}.pt")
            if steps % 20_000 < rollout_len:
                print(f"  Steps={steps:>7,} MeanRew={mean_rew:.4f} PG={metrics['pg_loss']:.4f} Ent={metrics['entropy']:.4f}")
        self.ac.load_state_dict(torch.load(MODEL_DIR / f"ppo_best{self.tag}.pt", map_location=DEVICE))
        print(f"  Best mean reward: {best:.4f}")
        return history

    def evaluate(self, env, n_episodes=EVAL_EPISODES):
        self.ac.eval(); rewards, anchors, p_deals = [], [], []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            s_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                dist, _ = self.ac.get_dist(s_t)
                act = dist.mean.clamp(-1.0, 1.0)
            _, rew, _, _, info = env.step(act.cpu().numpy()[0])
            rewards.append(rew); anchors.append(info["anchor_ratio"]); p_deals.append(info["p_deal"])
        return {"ppo_e_savings": float(np.mean(rewards)), "ppo_mean_anchor": float(np.mean(anchors)),
                "ppo_std_anchor": float(np.std(anchors)), "ppo_mean_p_deal": float(np.mean(p_deals))}


# ────────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    tag = "_faithful" if FAITHFUL else ""

    import xgboost as xgb
    clf = xgb.XGBClassifier(); clf.load_model(str(MODEL_DIR / "deal_classifier.ubj"))
    with open(MODEL_DIR / "cql_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    test = pd.read_parquet(DATA_DIR / "test.parquet")
    states_raw = test[STATE_COLS].values.astype(np.float32)

    # historical 5th-percentile anchor — the support floor used by the faithful sim
    empirical_anchor_p5 = float(np.percentile(
        pd.read_parquet(DATA_DIR / "train.parquet", columns=[ACTION_COL])[ACTION_COL].values, 5))
    anchor_p5 = (
        float(FAITHFUL_P5_OVERRIDE)
        if FAITHFUL_P5_OVERRIDE not in {None, ""}
        else empirical_anchor_p5
    )
    override_note = f", empirical p5 = {empirical_anchor_p5:.3f}" if anchor_p5 != empirical_anchor_p5 else ""
    print(
        f"Faithful simulator: {FAITHFUL}  "
        f"(support threshold = {anchor_p5:.3f}{override_note}, beta = {FAITHFUL_BETA})"
    )

    env = EbayBargainingEnv(states_raw, clf, scaler,
                            noise_std=0.03, hard_reward=False,
                            faithful=FAITHFUL, anchor_p5=anchor_p5, faithful_beta=FAITHFUL_BETA)

    ppo = PPOAgent(state_dim=N_STATE, tag=tag)
    history = ppo.train(env)
    with open(MODEL_DIR / f"ppo_history{tag}.json", "w") as f:
        json.dump(history, f)

    metrics = ppo.evaluate(env)
    metrics["faithful"] = FAITHFUL
    metrics["faithful_support_threshold"] = anchor_p5
    metrics["faithful_empirical_anchor_p5"] = empirical_anchor_p5
    metrics["faithful_beta"] = FAITHFUL_BETA
    print("\n── PPO policy evaluation ──────────────────────────")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    with open(MODEL_DIR / f"ppo_metrics{tag}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Phase 3 complete in {time.time() - t0:.1f}s  "
          f"(metrics -> ppo_metrics{tag}.json)")


if __name__ == "__main__":
    main()
