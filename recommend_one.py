"""Generate a transparent offer menu for one fashion listing.

PPO is intentionally excluded from the actionable menu because it is trained
and evaluated only inside a learned simulator.  The menu compares a fixed rule,
the supervised model optimum, and the support-conservative CQL policy.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

try:
    import torch

    torch.set_num_threads(1)
except Exception:
    torch = None

from policy_utils import ACTION_GRID, greedy_policy, score_actions
from project_constants import ACTION_COL, CLASSIFIER_FILE, N_GRID, STATE_COLS


DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models"))


def load_preprocess_stats() -> dict:
    path = DATA_DIR / "preprocess_stats.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run data_preprocess.py before requesting a recommendation."
        )
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def support_band(stats: dict) -> tuple[float, float]:
    """Load the train-fitted support band without requiring row-level data."""
    if "anchor_p5" in stats and "anchor_p95" in stats:
        return float(stats["anchor_p5"]), float(stats["anchor_p95"])

    # Backward-compatible fallback for artifacts produced before the support
    # quantiles were added to preprocess_stats.json.
    train_path = DATA_DIR / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(
            "preprocess_stats.json lacks anchor_p5/anchor_p95 and train.parquet "
            "is unavailable. Rerun data_preprocess.py with the corrected code."
        )
    anchors = pd.read_parquet(train_path, columns=[ACTION_COL])[ACTION_COL].values
    p5, p95 = np.percentile(anchors, [5, 95])
    return float(p5), float(p95)


def support_label(anchor: float, p5: float, p95: float) -> str:
    return "inside" if p5 <= anchor <= p95 else "outside"


def build_state(price: float, score: float, pos_pct: float, leaf_category, stats: dict):
    top_categories = set(stats.get("top_leaf_categories", []))
    category = int(leaf_category) if leaf_category in top_categories else -1
    return np.asarray(
        [[
            np.log(price + 1.0),
            np.clip(score / stats["seller_score_p99"], 0.0, 1.0),
            pos_pct,
            category,
        ]],
        dtype=np.float32,
    )


def cql_anchor(state: np.ndarray):
    checkpoint = MODEL_DIR / "cql_best.pt"
    scaler_path = MODEL_DIR / "cql_scaler.pkl"
    if torch is None or not checkpoint.exists() or not scaler_path.exists():
        return None
    from phase2_cql import QNetwork

    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)
    network = QNetwork(len(STATE_COLS))
    network.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    network.eval()
    scaled_state = torch.tensor(scaler.transform(state).astype(np.float32))
    with torch.no_grad():
        q_values = network(
            scaled_state.expand(N_GRID, -1), torch.tensor(ACTION_GRID)
        )
    return float(ACTION_GRID[int(q_values.argmax())])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--price", type=float, required=True)
    parser.add_argument("--seller_score", type=float, default=1000.0)
    parser.add_argument("--pos_pct", type=float, default=99.0)
    parser.add_argument(
        "--leaf_category",
        type=int,
        default=None,
        help="Optional anonymized fashion leaf-category ID; unknown categories map to -1.",
    )
    args = parser.parse_args()
    if args.price <= 0:
        raise ValueError("--price must be positive")

    import xgboost as xgb

    classifier = xgb.XGBClassifier()
    classifier.load_model(str(MODEL_DIR / CLASSIFIER_FILE))
    stats = load_preprocess_stats()
    p5, p95 = support_band(stats)
    state = build_state(
        args.price, args.seller_score, args.pos_pct, args.leaf_category, stats
    )

    greedy_anchor, _, _, _ = greedy_policy(classifier, state)
    candidates = [("Fixed 0.70 baseline", 0.70)]
    cql = cql_anchor(state)
    if cql is not None:
        candidates.append(("CQL support-aware", cql))
    candidates.append(("Supervised greedy", float(greedy_anchor[0])))

    print(
        f"\nListing ${args.price:,.2f} | seller score {args.seller_score:,.0f} "
        f"| {args.pos_pct}% positive"
    )
    category_note = args.leaf_category if args.leaf_category is not None else "unknown"
    print(f"Leaf category: {category_note}; historical p5-p95: {p5:.1%}-{p95:.1%}\n")
    print(
        f"  {'Policy':<23}{'Anchor':>9}{'Offer':>11}{'P(accept)':>12}"
        f"{'Discount':>11}{'E[savings]':>13}{'Support':>10}"
    )
    print("  " + "-" * 89)
    for name, anchor in candidates:
        probability, expected = score_actions(
            classifier, state, np.asarray([anchor], dtype=np.float32)
        )
        offer_text = f"${anchor * args.price:,.2f}"
        print(
            f"  {name:<23}{anchor:>9.1%}{offer_text:>11}"
            f"{probability[0]:>12.1%}{1.0 - anchor:>11.1%}"
            f"{expected[0]:>13.1%}{support_label(anchor, p5, p95):>10}"
        )

    print(
        "\nP(accept) and E[savings] are observational model estimates, not causal "
        "guarantees. CQL is support-conservative; it is not guaranteed to have "
        "the highest acceptance probability. PPO is omitted because it is simulator-only.\n"
    )


if __name__ == "__main__":
    main()
