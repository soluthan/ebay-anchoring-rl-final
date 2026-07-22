"""Build the one-step opening-offer dataset.

Input
-----
``clean_master_dataset.parquet``: merged Best Offer events and listing data.

Output
------
``train.parquet``, ``val.parquet`` and ``test.parquet`` contain one observation
per buyer-listing bargaining thread.  The observation is the *first event* in
the thread and is retained only when it is a first buyer offer
(``offr_type_id == 0``).

The estimand is deliberately narrow:

    action  = opening offer / listing price
    outcome = whether that opening-offer event itself was accepted
    reward  = outcome * (1 - action)

Later counteroffers and eventual thread success are not part of the reward.
They are retained only as audit fields.  All normal random and temporal splits
assign complete ``anon_item_id`` groups, preventing the same listing from
appearing in both training and evaluation data.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

from project_constants import (
    ACCEPTED_STATUSES,
    ACTION_COL,
    ANCHOR_MAX,
    ANCHOR_MIN,
    BUYER_COL,
    FASHION_CATEG_IDS,
    FIRST_BUYER_OFFER_TYPE,
    ITEM_COL,
    ITEM_PRICE_COL,
    LABEL_COL,
    OFFER_PRICE_COL,
    OFFER_TYPE_COL,
    REWARD_COL,
    SEED,
    STATUS_COL,
    THREAD_COL,
)


DATA_DIR = Path(os.environ.get("DATA_DIR", "."))
OUT_DIR = Path(os.environ.get("OUT_DIR", "./data"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE = DATA_DIR / "clean_master_dataset.parquet"

SPLIT_MODE = os.environ.get("SPLIT_MODE", "random").strip().lower()
TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", "0.80"))
VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.10"))
TOP_LEAF_N = int(os.environ.get("TOP_LEAF_N", "25"))
FILTER_FASHION = os.environ.get("FILTER_FASHION", "1") == "1"
PREPROCESS_MAX_ROWS = int(os.environ.get("PREPROCESS_MAX_ROWS", "0"))

DATE_CANDIDATES = ["src_cre_date", "src_cre_dt"]

COL = {
    "item_id": ITEM_COL,
    "buyer_id": BUYER_COL,
    "thread_id": THREAD_COL,
    "offer_type": OFFER_TYPE_COL,
    "list_price": "start_price_usd",
    "offer_price": OFFER_PRICE_COL,
    "item_price": ITEM_PRICE_COL,
    "status": STATUS_COL,
    "meta_categ": "meta_categ_id",
    "leaf_categ": "anon_leaf_categ_id",
    "slr_score": "fdbk_score_src",
    "slr_pos_pct": "fdbk_pstv_src",
}


def check_columns(lf: pl.LazyFrame) -> None:
    columns = set(lf.collect_schema().names())
    required = {
        COL["item_id"],
        COL["buyer_id"],
        COL["offer_type"],
        COL["list_price"],
        COL["offer_price"],
        COL["item_price"],
        COL["status"],
        COL["meta_categ"],
    }
    missing = required - columns
    if missing:
        raise ValueError(
            "Opening-offer extraction requires these missing columns: "
            f"{sorted(missing)}\nAvailable: {sorted(columns)}"
        )
    if not any(name in columns for name in DATE_CANDIDATES):
        raise ValueError(
            "Opening-offer extraction needs an event timestamp. "
            f"Expected one of: {DATE_CANDIDATES}"
        )


def available_date_column(columns: list[str]) -> str:
    for name in DATE_CANDIDATES:
        if name in columns:
            return name
    raise ValueError(f"No event timestamp found; tried {DATE_CANDIDATES}")


def add_interaction_time(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    date_col = available_date_column(schema.names())
    dtype = schema[date_col]
    if dtype in (pl.String, pl.Utf8):
        text = pl.col(date_col).cast(pl.String)
        parsed = pl.coalesce(
            text.str.strptime(pl.Datetime, "%d%b%Y %H:%M:%S", strict=False),
            text.str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
            text.str.strptime(pl.Date, "%d%b%Y", strict=False).cast(pl.Datetime),
            text.str.strptime(pl.Date, "%Y-%m-%d", strict=False).cast(pl.Datetime),
        )
    else:
        parsed = pl.col(date_col).cast(pl.Datetime, strict=False)
    return lf.with_columns(parsed.alias("interaction_ts"))


def extract_opening_offers(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Return one genuine opening-buyer-offer row per buyer-listing thread."""
    c = COL
    source_columns = lf.collect_schema().names()
    date_column = available_date_column(source_columns)
    # The merged source has many unused listing columns. Project only fields
    # needed for extraction/features/audits before sorting millions of events.
    wanted = [
        c["item_id"],
        c["buyer_id"],
        c["thread_id"],
        c["offer_type"],
        c["list_price"],
        c["offer_price"],
        c["item_price"],
        c["status"],
        c["meta_categ"],
        c["leaf_categ"],
        c["slr_score"],
        c["slr_pos_pct"],
        date_column,
    ]
    lf = lf.select([name for name in dict.fromkeys(wanted) if name in source_columns])
    if FILTER_FASHION:
        lf = lf.filter(pl.col(c["meta_categ"]).is_in(FASHION_CATEG_IDS))

    lf = add_interaction_time(lf).filter(
        pl.col(c["item_id"]).is_not_null()
        & pl.col(c["buyer_id"]).is_not_null()
        & pl.col("interaction_ts").is_not_null()
        & (pl.col(c["list_price"]) > 0)
    )

    group_cols = [c["item_id"], c["buyer_id"]]
    all_columns = lf.collect_schema().names()
    first_columns = [name for name in all_columns if name not in group_cols]

    # Select the first observed buyer-item event, then keep it only when it is
    # a buyer opening offer (offr_type_id == 0). This prevents a later thread
    # outcome from being attached to the opening action.
    lf = (
        lf.sort(group_cols + ["interaction_ts"])
        .group_by(group_cols, maintain_order=True)
        .agg(
            *[pl.col(name).first().alias(name) for name in first_columns],
            pl.col(c["status"])
            .is_in(ACCEPTED_STATUSES)
            .any()
            .alias("thread_eventual_accepted"),
            pl.len().alias("thread_event_count"),
        )
        .filter(pl.col(c["offer_type"]) == FIRST_BUYER_OFFER_TYPE)
    )
    return lf


def build_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    c = COL
    lf = extract_opening_offers(lf)
    lf = lf.filter(pl.col(c["offer_price"]) > 0).with_columns(
        (pl.col(c["offer_price"]) / pl.col(c["list_price"])).alias(ACTION_COL),
        pl.col(c["status"])
        .is_in(ACCEPTED_STATUSES)
        .cast(pl.Int8)
        .alias(LABEL_COL),
        pl.col(c["meta_categ"]).alias("meta_categ_id_raw"),
        (
            pl.col(c["leaf_categ"])
            if c["leaf_categ"] in lf.collect_schema().names()
            else pl.lit(None).cast(pl.Int64)
        ).alias("leaf_categ_id_raw"),
    )
    lf = lf.filter(
        (pl.col(ACTION_COL) > ANCHOR_MIN) & (pl.col(ACTION_COL) <= ANCHOR_MAX)
    )
    return lf.with_columns(
        (pl.col(LABEL_COL).cast(pl.Float32) * (1.0 - pl.col(ACTION_COL)))
        .alias(REWARD_COL),
        (pl.col(c["list_price"]) + 1.0).log().alias("log_list_price"),
    )


def _split_positions(n_groups: int) -> tuple[int, int]:
    if n_groups < 3:
        raise ValueError("At least three unique listings are required for train/val/test.")
    train_end = max(1, int(TRAIN_FRAC * n_groups))
    val_end = max(train_end + 1, int((TRAIN_FRAC + VAL_FRAC) * n_groups))
    val_end = min(val_end, n_groups - 1)
    return train_end, val_end


def _frames_from_item_ids(
    df: pl.DataFrame, train_ids: np.ndarray, val_ids: np.ndarray, test_ids: np.ndarray
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    item = COL["item_id"]

    def select(ids: np.ndarray) -> pl.DataFrame:
        return df.join(pl.DataFrame({item: ids}), on=item, how="inner")

    return select(train_ids), select(val_ids), select(test_ids)


def split_random(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Randomly split unique listings, never individual offer rows."""
    item_ids = df[COL["item_id"]].unique().to_numpy()
    rng = np.random.default_rng(SEED)
    item_ids = item_ids[rng.permutation(len(item_ids))]
    train_end, val_end = _split_positions(len(item_ids))
    return _frames_from_item_ids(
        df, item_ids[:train_end], item_ids[train_end:val_end], item_ids[val_end:]
    )


def split_temporal(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Chronologically split complete listings by their earliest opening time."""
    if "interaction_ts" not in df.columns or df["interaction_ts"].null_count() == len(df):
        raise ValueError("SPLIT_MODE=temporal requires a parseable event timestamp.")
    item = COL["item_id"]
    item_order = (
        df.filter(pl.col("interaction_ts").is_not_null())
        .group_by(item)
        .agg(pl.col("interaction_ts").min().alias("listing_opening_ts"))
        .sort(["listing_opening_ts", item])
    )
    item_ids = item_order[item].to_numpy()
    train_end, val_end = _split_positions(len(item_ids))
    return _frames_from_item_ids(
        df, item_ids[:train_end], item_ids[train_end:val_end], item_ids[val_end:]
    )


def split_leaf_holdout(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Optional robustness split with disjoint anonymized leaf categories."""
    leaf = "leaf_categ_id_raw"
    if leaf not in df.columns or df[leaf].null_count() == len(df):
        raise ValueError("SPLIT_MODE=leaf_holdout requires anon_leaf_categ_id.")
    leaves = df[leaf].drop_nulls().unique().to_numpy()
    if len(leaves) < 3:
        raise ValueError("Need at least three leaf categories for leaf_holdout.")
    rng = np.random.default_rng(SEED)
    leaves = leaves[rng.permutation(len(leaves))]
    train_end, val_end = _split_positions(len(leaves))
    return (
        df.filter(pl.col(leaf).is_in(leaves[:train_end])),
        df.filter(pl.col(leaf).is_in(leaves[train_end:val_end])),
        df.filter(pl.col(leaf).is_in(leaves[val_end:])),
    )


def split_frame(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if SPLIT_MODE == "random":
        frames = split_random(df)
    elif SPLIT_MODE == "temporal":
        frames = split_temporal(df)
    elif SPLIT_MODE in {"leaf_holdout", "leaf-holdout"}:
        frames = split_leaf_holdout(df)
    else:
        raise ValueError(
            f"Unknown SPLIT_MODE={SPLIT_MODE!r}. Use random, temporal, or leaf_holdout."
        )
    assert_disjoint_items(*frames)
    return frames


def assert_disjoint_items(train: pl.DataFrame, val: pl.DataFrame, test: pl.DataFrame) -> None:
    item = COL["item_id"]
    train_ids = set(train[item].to_list())
    val_ids = set(val[item].to_list())
    test_ids = set(test[item].to_list())
    overlap = {
        "train_val": len(train_ids & val_ids),
        "train_test": len(train_ids & test_ids),
        "val_test": len(val_ids & test_ids),
    }
    if any(overlap.values()):
        raise AssertionError(f"Listing leakage detected: {overlap}")


def fit_preprocess_stats(train: pl.DataFrame) -> dict:
    slr, slr_pos, leaf = COL["slr_score"], COL["slr_pos_pct"], "leaf_categ_id_raw"
    seller_score_p99 = (
        float(train[slr].quantile(0.99) or 1.0) if slr in train.columns else 1.0
    )
    seller_pos_median = (
        float(train[slr_pos].median() or 0.95) if slr_pos in train.columns else 0.95
    )
    top_leaves = (
        train[leaf]
        .drop_nulls()
        .value_counts(sort=True)
        .head(TOP_LEAF_N)[leaf]
        .to_list()
        if leaf in train.columns
        else []
    )
    anchor_p5 = float(train[ACTION_COL].quantile(0.05))
    anchor_p95 = float(train[ACTION_COL].quantile(0.95))
    return {
        "split_mode": SPLIT_MODE,
        "split_group": COL["item_id"],
        "accepted_statuses": list(ACCEPTED_STATUSES),
        "opening_offer_type": FIRST_BUYER_OFFER_TYPE,
        "filter_fashion": FILTER_FASHION,
        "seller_score_p99": max(seller_score_p99, 1e-6),
        "seller_pos_median": seller_pos_median,
        "top_leaf_n": TOP_LEAF_N,
        "top_leaf_categories": [int(x) for x in top_leaves],
        # Persist the deployable support band so recommend_one.py can operate
        # with aggregate preprocessing metadata and trained models only.
        "anchor_p5": anchor_p5,
        "anchor_p95": anchor_p95,
    }


def apply_preprocess_stats(df: pl.DataFrame, stats: dict) -> pl.DataFrame:
    slr, slr_pos, leaf = COL["slr_score"], COL["slr_pos_pct"], "leaf_categ_id_raw"
    seller_score = (
        (pl.col(slr) / stats["seller_score_p99"]).clip(0, 1)
        if slr in df.columns
        else pl.lit(0.5)
    )
    seller_positive = (
        pl.col(slr_pos).fill_null(stats["seller_pos_median"])
        if slr_pos in df.columns
        else pl.lit(0.95)
    )
    top_leaves = stats["top_leaf_categories"]
    category = (
        pl.when(pl.col(leaf).is_in(top_leaves))
        .then(pl.col(leaf))
        .otherwise(-1)
        .cast(pl.Int32)
        if leaf in df.columns
        else pl.lit(-1).cast(pl.Int32)
    )
    return df.with_columns(
        seller_score.fill_null(0.5).alias("seller_score_norm"),
        seller_positive.fill_null(stats["seller_pos_median"]).alias("seller_pos_pct"),
        category.fill_null(-1).alias("categ_id_clean"),
    )


def select_model_columns(df: pl.DataFrame) -> pl.DataFrame:
    keep = [
        "log_list_price",
        "seller_score_norm",
        "seller_pos_pct",
        "categ_id_clean",
        ACTION_COL,
        REWARD_COL,
        LABEL_COL,
        STATUS_COL,
        COL["list_price"],
        COL["offer_price"],
        COL["item_price"],
        ITEM_COL,
        BUYER_COL,
        THREAD_COL,
        OFFER_TYPE_COL,
        "interaction_ts",
        "thread_event_count",
        "thread_eventual_accepted",
        "meta_categ_id_raw",
        "leaf_categ_id_raw",
    ]
    return df.select([name for name in keep if name in df.columns])


def summarize_split(name: str, df: pl.DataFrame) -> dict:
    if len(df) == 0:
        return {"split": name, "n": 0}
    dates = df["interaction_ts"].drop_nulls() if "interaction_ts" in df.columns else []
    return {
        "split": name,
        "n": int(len(df)),
        "n_items": int(df[ITEM_COL].n_unique()),
        "opening_acceptance_rate": float(df[LABEL_COL].mean()),
        "mean_anchor": float(df[ACTION_COL].mean()),
        "anchor_p5": float(df[ACTION_COL].quantile(0.05)),
        "anchor_p95": float(df[ACTION_COL].quantile(0.95)),
        "date_min": str(dates.min()) if len(dates) else "",
        "date_max": str(dates.max()) if len(dates) else "",
    }


def write_status_diagnostics(df: pl.DataFrame) -> None:
    """Cheap semantic checks; trends are diagnostic, not status definitions."""
    (
        df.group_by(STATUS_COL)
        .agg(pl.len().alias("n"))
        .sort(STATUS_COL)
        .write_csv(OUT_DIR / "opening_status_counts.csv")
    )
    n_bins = 20
    diagnostic = (
        df.with_columns(
            (
                ((pl.col(ACTION_COL) - ANCHOR_MIN) / (ANCHOR_MAX - ANCHOR_MIN) * n_bins)
                .floor()
                .clip(0, n_bins - 1)
                .cast(pl.Int16)
            ).alias("anchor_bin")
        )
        .group_by("anchor_bin")
        .agg(
            pl.len().alias("n"),
            pl.col(ACTION_COL).min().alias("anchor_min"),
            pl.col(ACTION_COL).max().alias("anchor_max"),
            pl.col(ACTION_COL).mean().alias("anchor_mean"),
            (pl.col(STATUS_COL) == 1).mean().alias("status_1_rate"),
            (pl.col(STATUS_COL) == 9).mean().alias("status_9_rate"),
            pl.col(LABEL_COL).mean().alias("accepted_1_or_9_rate"),
        )
        .sort("anchor_bin")
    )
    diagnostic.write_csv(OUT_DIR / "acceptance_by_anchor_bin.csv")

    # Empirical status adjudication: for genuinely accepted rows, the transaction
    # price should coincide with the accepted offer. This supplements (rather
    # than replaces) the official codebook/companion-code definition.
    if COL["item_price"] in df.columns:
        price_diagnostic = (
            df.group_by(STATUS_COL)
            .agg(
                pl.len().alias("n"),
                pl.col(COL["item_price"]).is_not_null().mean().alias("item_price_present"),
                (pl.col(COL["item_price"]) / pl.col(COL["list_price"]))
                .median()
                .alias("median_item_to_list"),
                (pl.col(COL["item_price"]) / pl.col(COL["offer_price"]))
                .median()
                .alias("median_item_to_offer"),
                (pl.col(COL["offer_price"]) / pl.col(COL["list_price"]))
                .median()
                .alias("median_offer_to_list"),
                (
                    (pl.col(COL["item_price"]) - pl.col(COL["offer_price"])).abs()
                    / pl.col(COL["offer_price"])
                )
                .lt(0.01)
                .mean()
                .alias("item_offer_within_1pct"),
            )
            .sort(STATUS_COL)
        )
        price_diagnostic.write_csv(OUT_DIR / "status_price_diagnostics.csv")


def write_split_summary(
    train: pl.DataFrame, val: pl.DataFrame, test: pl.DataFrame, stats: dict
) -> None:
    pl.DataFrame(
        [summarize_split("train", train), summarize_split("val", val), summarize_split("test", test)]
    ).write_csv(OUT_DIR / "split_summary.csv")
    with open(OUT_DIR / "preprocess_stats.json", "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)

    sets = [set(frame[ITEM_COL].to_list()) for frame in (train, val, test)]
    overlap = {
        "train_val": len(sets[0] & sets[1]),
        "train_test": len(sets[0] & sets[2]),
        "val_test": len(sets[1] & sets[2]),
    }
    with open(OUT_DIR / "item_split_overlap.json", "w", encoding="utf-8") as file:
        json.dump(overlap, file, indent=2)


def main() -> None:
    started = time.time()
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing file: {SOURCE.resolve()}")

    print("Scanning merged event parquet ...")
    source = pl.scan_parquet(SOURCE)
    check_columns(source)
    opening = build_features(source)
    if PREPROCESS_MAX_ROWS:
        opening = opening.head(PREPROCESS_MAX_ROWS)
        print(f"Smoke cap: {PREPROCESS_MAX_ROWS:,} extracted opening offers.")
    frame = opening.collect()
    if frame.is_empty():
        raise ValueError("No valid first buyer offers remained after filtering.")

    duplicate_threads = frame.select([ITEM_COL, BUYER_COL]).is_duplicated().sum()
    if duplicate_threads:
        raise AssertionError(f"Found {duplicate_threads} duplicate buyer-listing threads.")
    if frame[ACTION_COL].max() > ANCHOR_MAX:
        raise AssertionError("Opening-offer action exceeds ANCHOR_MAX.")

    print(f"Valid first buyer offers: {len(frame):,}")
    print(f"Splitting complete listings with SPLIT_MODE={SPLIT_MODE!r} ...")
    train_raw, val_raw, test_raw = split_frame(frame)
    stats = fit_preprocess_stats(train_raw)
    train = select_model_columns(apply_preprocess_stats(train_raw, stats))
    val = select_model_columns(apply_preprocess_stats(val_raw, stats))
    test = select_model_columns(apply_preprocess_stats(test_raw, stats))

    train.write_parquet(OUT_DIR / "train.parquet")
    val.write_parquet(OUT_DIR / "val.parquet")
    test.write_parquet(OUT_DIR / "test.parquet")
    write_split_summary(train, val, test, stats)
    write_status_diagnostics(frame)

    print("\nDONE")
    print(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    print(f"Immediate opening-offer acceptance: {float(frame[LABEL_COL].mean()):.3f}")
    print(f"Listing overlap audit: {OUT_DIR / 'item_split_overlap.json'}")
    print(f"Status diagnostic: {OUT_DIR / 'acceptance_by_anchor_bin.csv'}")
    if COL["item_price"] in frame.columns:
        print(f"Price adjudication: {OUT_DIR / 'status_price_diagnostics.csv'}")
    print(f"Time: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
