"""Phase 3: PPO in basic and support-penalized learned simulators.

PPO is exploratory and simulator-only.  The environment is a one-step
contextual bandit whose acceptance surface comes from Phase 1.  A Beta policy
generates actions directly on (0, 1), avoiding the biased density/gradient that
results from sampling a Gaussian and clipping it at the action boundaries.

By default one invocation trains both variants required for H3:

* basic: no support penalty and no output noise;
* robust: Gaussian probability noise plus exponential penalties outside the
  historical 5th--95th percentile anchor range.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from torch.distributions import Beta

from project_constants import (
    ACTION_COL,
    ANCHOR_MAX,
    ANCHOR_MIN,
    CLASSIFIER_FILE,
    SEED,
    STATE_COLS,
)


np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_STATE = len(STATE_COLS)
TOTAL_STEPS = int(os.environ.get("PPO_STEPS", "200000"))
ROLLOUT_LEN = int(os.environ.get("PPO_ROLLOUT", "2048"))
EVAL_EPISODES = int(os.environ.get("PPO_EVAL_EPISODES", "5000"))
RUN_BOTH = os.environ.get("PPO_RUN_BOTH", "1") == "1"
FAITHFUL = os.environ.get("PPO_FAITHFUL", "1") == "1"
FAITHFUL_BETA = float(os.environ.get("PPO_FAITHFUL_BETA", "8.0"))
FAITHFUL_P5_OVERRIDE = os.environ.get("PPO_FAITHFUL_P5")
FAITHFUL_P95_OVERRIDE = os.environ.get("PPO_FAITHFUL_P95")
ROBUST_NOISE_STD = float(os.environ.get("PPO_ROBUST_NOISE_STD", "0.03"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_EPS = 1e-6


def unit_to_anchor(unit_action):
    return ANCHOR_MIN + unit_action * (ANCHOR_MAX - ANCHOR_MIN)


class EbayBargainingEnv(gym.Env):
    """Single-step learned environment; XGBoost always receives raw features."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        states_raw,
        clf,
        scaler,
        noise_std=0.0,
        hard_reward=False,
        faithful=False,
        anchor_p5=0.30,
        anchor_p95=0.95,
        faithful_beta=8.0,
    ):
        super().__init__()
        self.states_raw = np.asarray(states_raw, dtype=np.float32)
        self.clf = clf
        self.scaler = scaler
        self.noise_std = float(noise_std)
        self.hard_reward = hard_reward
        self.faithful = faithful
        self.anchor_p5 = float(anchor_p5)
        self.anchor_p95 = float(anchor_p95)
        self.faithful_beta = float(faithful_beta)
        self.rng = np.random.default_rng(SEED)

        self.observation_space = spaces.Box(-10.0, 10.0, (N_STATE,), np.float32)
        # PPO acts in unit space; the environment maps it to [ANCHOR_MIN, ANCHOR_MAX].
        self.action_space = spaces.Box(0.0, 1.0, (1,), np.float32)
        self._raw = self._obs = None

    def _to_obs(self, raw_row):
        return self.scaler.transform(raw_row.reshape(1, -1)).astype(np.float32)[0]

    def _support_weight(self, anchor: float) -> float:
        if not self.faithful:
            return 1.0
        if anchor < self.anchor_p5:
            gap = (self.anchor_p5 - anchor) / max(self.anchor_p5 - ANCHOR_MIN, 1e-6)
        elif anchor > self.anchor_p95:
            gap = (anchor - self.anchor_p95) / max(ANCHOR_MAX - self.anchor_p95, 1e-6)
        else:
            return 1.0
        return float(np.exp(-self.faithful_beta * gap))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        index = self.rng.integers(0, len(self.states_raw))
        self._raw = self.states_raw[index].copy()
        self._obs = self._to_obs(self._raw)
        return self._obs, {}

    def step(self, action):
        unit_action = float(np.clip(action[0], 0.0, 1.0))
        anchor = float(unit_to_anchor(unit_action))
        features = np.append(self._raw, anchor).reshape(1, -1).astype(np.float32)

        p_accept = float(self.clf.predict_proba(features)[0, 1])
        if self.noise_std:
            p_accept += float(self.rng.normal(0.0, self.noise_std))
        p_accept = float(np.clip(p_accept, 0.0, 0.99))
        p_accept *= self._support_weight(anchor)
        discount = 1.0 - anchor
        reward = (
            float(int(self.rng.random() < p_accept) * discount)
            if self.hard_reward
            else float(p_accept * discount)
        )
        info = {
            "anchor_ratio": anchor,
            "p_accept": p_accept,
            "discount_if_accepted": discount,
            "support_weight": self._support_weight(anchor),
            "reward": reward,
        }
        return self._obs, reward, True, False, info


class ActorCritic(nn.Module):
    """Actor-Critic with a bounded Beta policy in unit action space."""

    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_alpha = nn.Linear(hidden, 1)
        self.actor_beta = nn.Linear(hidden, 1)
        self.critic = nn.Linear(hidden, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_alpha.weight, gain=0.01)
        nn.init.orthogonal_(self.actor_beta.weight, gain=0.01)

    def forward(self, state):
        latent = self.backbone(state)
        # +1 keeps the initial density unimodal and away from singular boundaries.
        alpha = nn.functional.softplus(self.actor_alpha(latent)) + 1.0
        beta = nn.functional.softplus(self.actor_beta(latent)) + 1.0
        value = self.critic(latent).squeeze(-1)
        return alpha, beta, value

    def get_dist(self, state):
        alpha, beta, value = self.forward(state)
        return Beta(alpha, beta), value


class PPOAgent:
    def __init__(
        self,
        state_dim=N_STATE,
        hidden=128,
        lr=3e-4,
        clip_eps=0.2,
        n_epochs=10,
        gamma=0.0,
        gae_lambda=0.0,
        vf_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        batch_size=64,
        tag="",
    ):
        # gamma=0 makes the one-step terminal formulation explicit.
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.tag = tag
        self.ac = ActorCritic(state_dim, hidden).to(DEVICE)
        self.opt = optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)

    def collect_rollout(self, env, n_steps):
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        observation, _ = env.reset()
        for _ in range(n_steps):
            state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                distribution, value = self.ac.get_dist(state)
                # Numerical guard only: Beta samples are already bounded in
                # theory; epsilon avoids log(0) under finite precision.
                action = distribution.sample().clamp(ACTION_EPS, 1.0 - ACTION_EPS)
                log_probability = distribution.log_prob(action).sum(-1)
            action_np = action.cpu().numpy()[0]
            next_observation, reward, terminated, truncated, _ = env.step(action_np)
            states.append(observation)
            actions.append(action_np)
            log_probs.append(log_probability.item())
            rewards.append(reward)
            values.append(value.item())
            dones.append(float(terminated or truncated))
            observation, _ = env.reset() if (terminated or truncated) else (next_observation, None)
        return {
            "states": np.asarray(states, np.float32),
            "actions": np.asarray(actions, np.float32),
            "log_probs": np.asarray(log_probs, np.float32),
            "rewards": np.asarray(rewards, np.float32),
            "values": np.asarray(values, np.float32),
            "dones": np.asarray(dones, np.float32),
        }

    def compute_gae(self, rollout, last_value=0.0):
        count = len(rollout["rewards"])
        advantages = np.zeros(count, np.float32)
        gae = 0.0
        for step in reversed(range(count)):
            next_value = rollout["values"][step + 1] if step < count - 1 else last_value
            nonterminal = 1.0 - rollout["dones"][step]
            delta = (
                rollout["rewards"][step]
                + self.gamma * next_value * nonterminal
                - rollout["values"][step]
            )
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[step] = gae
        return advantages

    def update(self, rollout):
        advantages = self.compute_gae(rollout)
        returns = advantages + rollout["values"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.tensor(rollout["states"], device=DEVICE)
        actions = torch.tensor(rollout["actions"], device=DEVICE).clamp(
            ACTION_EPS, 1.0 - ACTION_EPS
        )
        old_log_probs = torch.tensor(rollout["log_probs"], device=DEVICE)
        returns_t = torch.tensor(returns, device=DEVICE)
        advantages_t = torch.tensor(advantages, device=DEVICE)
        count = len(states)
        metrics = {"pg_loss": [], "vf_loss": [], "entropy": [], "total_loss": []}

        for _ in range(self.n_epochs):
            permutation = np.random.permutation(count)
            for start in range(0, count, self.batch_size):
                indices = permutation[start:start + self.batch_size]
                distribution, values = self.ac.get_dist(states[indices])
                new_log_probs = distribution.log_prob(actions[indices]).sum(-1)
                entropy = distribution.entropy().sum(-1).mean()
                ratio = torch.exp(new_log_probs - old_log_probs[indices])
                objective = ratio * advantages_t[indices]
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                ) * advantages_t[indices]
                policy_loss = -torch.min(objective, clipped).mean()
                value_loss = nn.functional.mse_loss(values, returns_t[indices])
                loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.opt.step()
                metrics["pg_loss"].append(policy_loss.item())
                metrics["vf_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.item())
                metrics["total_loss"].append(loss.item())
        return {key: float(np.mean(values)) for key, values in metrics.items()}

    def train(self, env, total_steps=TOTAL_STEPS, rollout_len=ROLLOUT_LEN):
        history = {"step": [], "mean_reward": [], "pg_loss": [], "entropy": []}
        steps, best = 0, -float("inf")
        print(f"PPO training for {total_steps:,} steps (robust={env.faithful}) ...")
        while steps < total_steps:
            rollout = self.collect_rollout(env, min(rollout_len, total_steps - steps))
            metrics = self.update(rollout)
            mean_reward = float(rollout["rewards"].mean())
            steps += len(rollout["rewards"])
            history["step"].append(steps)
            history["mean_reward"].append(mean_reward)
            history["pg_loss"].append(metrics["pg_loss"])
            history["entropy"].append(metrics["entropy"])
            if mean_reward > best:
                best = mean_reward
                torch.save(self.ac.state_dict(), MODEL_DIR / f"ppo_best{self.tag}.pt")
        self.ac.load_state_dict(
            torch.load(MODEL_DIR / f"ppo_best{self.tag}.pt", map_location=DEVICE)
        )
        return history

    def evaluate(self, env, n_episodes=EVAL_EPISODES):
        self.ac.eval()
        rewards, anchors, probabilities, support_weights = [], [], [], []
        for _ in range(n_episodes):
            observation, _ = env.reset()
            state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                distribution, _ = self.ac.get_dist(state)
                unit_action = distribution.mean
            _, reward, _, _, info = env.step(unit_action.cpu().numpy()[0])
            rewards.append(reward)
            anchors.append(info["anchor_ratio"])
            probabilities.append(info["p_accept"])
            support_weights.append(info["support_weight"])
        return {
            "ppo_e_savings": float(np.mean(rewards)),
            "ppo_mean_anchor": float(np.mean(anchors)),
            "ppo_std_anchor": float(np.std(anchors)),
            "ppo_mean_p_accept": float(np.mean(probabilities)),
            "ppo_mean_support_weight": float(np.mean(support_weights)),
        }


def run_variant(states_raw, clf, scaler, p5: float, p95: float, faithful: bool):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    tag = "_faithful" if faithful else ""
    noise = ROBUST_NOISE_STD if faithful else 0.0
    environment = EbayBargainingEnv(
        states_raw,
        clf,
        scaler,
        noise_std=noise,
        hard_reward=False,
        faithful=faithful,
        anchor_p5=p5,
        anchor_p95=p95,
        faithful_beta=FAITHFUL_BETA,
    )
    agent = PPOAgent(state_dim=N_STATE, tag=tag)
    history = agent.train(environment)
    metrics = agent.evaluate(environment)
    metrics.update(
        {
            "evidence_type": "simulator-only estimate",
            "policy_distribution": "Beta on unit action space",
            "one_step_terminal": True,
            "gamma": 0.0,
            "faithful": faithful,
            "noise_std": noise,
            "support_p5": p5,
            "support_p95": p95,
            "faithful_beta": FAITHFUL_BETA,
        }
    )
    with open(MODEL_DIR / f"ppo_history{tag}.json", "w", encoding="utf-8") as file:
        json.dump(history, file)
    with open(MODEL_DIR / f"ppo_metrics{tag}.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(f"  {'Robust' if faithful else 'Basic'} PPO E[savings]: {metrics['ppo_e_savings']:.4f}")
    return metrics


def main() -> None:
    started = time.time()
    import xgboost as xgb

    classifier = xgb.XGBClassifier()
    classifier.load_model(str(MODEL_DIR / CLASSIFIER_FILE))
    with open(MODEL_DIR / "cql_scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    test = pd.read_parquet(DATA_DIR / "test.parquet")
    train_anchors = pd.read_parquet(DATA_DIR / "train.parquet", columns=[ACTION_COL])[ACTION_COL].values
    empirical_p5, empirical_p95 = np.percentile(train_anchors, [5, 95])
    p5 = float(FAITHFUL_P5_OVERRIDE) if FAITHFUL_P5_OVERRIDE else float(empirical_p5)
    p95 = float(FAITHFUL_P95_OVERRIDE) if FAITHFUL_P95_OVERRIDE else float(empirical_p95)
    states_raw = test[STATE_COLS].values.astype(np.float32)

    variants = [False, True] if RUN_BOTH else [FAITHFUL]
    for faithful in variants:
        run_variant(states_raw, classifier, scaler, p5, p95, faithful)

    print(f"\nPhase 3 complete in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
