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
LABEL_COL = "opening_accepted"
STATUS_COL = "status_id"
LIST_COL = "start_price_usd"
ITEM_COL = "anon_item_id"
BUYER_COL = "anon_byr_id"
THREAD_COL = "anon_thread_id"
OFFER_TYPE_COL = "offr_type_id"
OFFER_PRICE_COL = "offr_price"
ITEM_PRICE_COL = "item_price"
CLASSIFIER_FILE = "opening_acceptance_classifier.ubj"

# Dataset-specific opening-acceptance mapping. Preprocessing writes status/price
# diagnostics so this assumption is auditable rather than silently trusted.
ACCEPTED_STATUSES = (1, 9)
FIRST_BUYER_OFFER_TYPE = 0
FASHION_CATEG_IDS = [11450]

ANCHOR_MIN = 0.01
ANCHOR_MAX = 1.00
# Sixty-seven evenly spaced points from 0.01 to 1.00 use a 0.015 step and
# therefore include the economically important fixed 0.70 baseline exactly.
# This guarantees that the grid-greedy policy cannot score below the baseline
# merely because a tree model jumps at 0.70 while a coarser grid misses it.
ACTION_GRID_SIZE = 67

# Backward-compatible names used by the individual phase scripts.
N_GRID = ACTION_GRID_SIZE
N_ACTIONS_DISC = ACTION_GRID_SIZE
