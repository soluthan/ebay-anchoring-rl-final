"""Shared deterministic policy helpers.

The supervised greedy policy is defined once here and reused by Phase 1, OPE,
and the recommendation scripts.  This prevents the action grid or objective
from drifting between evaluation paths.
"""

from __future__ import annotations

import numpy as np

from project_constants import ANCHOR_MAX, ANCHOR_MIN, N_GRID


ACTION_GRID = np.linspace(ANCHOR_MIN, ANCHOR_MAX, N_GRID, dtype=np.float32)


def score_actions(clf, states: np.ndarray, actions: np.ndarray):
    """Return P(immediate acceptance) and expected immediate savings."""
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32).reshape(-1)
    features = np.column_stack([states, actions]).astype(np.float32)
    p_accept = clf.predict_proba(features)[:, 1]
    expected = p_accept * (1.0 - actions)
    return p_accept, expected


def greedy_policy(clf, states: np.ndarray, chunk: int = 4096):
    """argmax_a P(accept | state, a) * (1-a) over the shared action grid."""
    states = np.asarray(states, dtype=np.float32)
    anchors_out, probs_out, values_out, indices_out = [], [], [], []

    for start in range(0, len(states), chunk):
        state_chunk = states[start:start + chunk]
        batch = len(state_chunk)
        repeated_states = np.repeat(state_chunk, N_GRID, axis=0)
        repeated_actions = np.tile(ACTION_GRID, batch)
        features = np.column_stack([repeated_states, repeated_actions]).astype(np.float32)
        probabilities = clf.predict_proba(features)[:, 1].reshape(batch, N_GRID)
        values = probabilities * (1.0 - ACTION_GRID)[None, :]
        best = values.argmax(axis=1)
        rows = np.arange(batch)
        anchors_out.append(ACTION_GRID[best])
        probs_out.append(probabilities[rows, best])
        values_out.append(values[rows, best])
        indices_out.append(best)

    return (
        np.concatenate(anchors_out),
        np.concatenate(probs_out),
        np.concatenate(values_out),
        np.concatenate(indices_out),
    )


def support_flags(actions: np.ndarray, p5: float, p95: float) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    return (actions >= p5) & (actions <= p95)
