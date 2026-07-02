"""
Shared constants for the flat-module eBay Best Offer anchoring RL project.

Keeping these values in one file avoids quiet drift between preprocessing,
supervised baselines, offline RL, PPO, results, and recommendation scripts while
preserving the simple side-by-side module layout used by the course project.
"""

SEED = 42

STATE_COLS = [
    "log_list_price",
    "seller_score_norm",
    "seller_pos_pct",
    "categ_id_clean",
]
ACTION_COL = "anchor_ratio"
REWARD_COL = "savings_pct"
DEAL_COL = "status_id"
LIST_COL = "start_price_usd"
ITEM_COL = "anon_item_id"
FINAL_OFFER_COL = "offr_price"

DEAL_STATUS = 2
FASHION_CATEG_IDS = [11450]

ANCHOR_MIN = 0.01
ANCHOR_MAX = 1.50
N_GRID = 50
N_ACTIONS_DISC = 50
