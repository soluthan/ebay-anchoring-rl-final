"""
data_preprocess.py - Feature engineering / MDP dataset builder
==============================================================
Input : clean_master_dataset.parquet   (the threads + lists merge - assumed done)
Output: data/train.parquet, data/val.parquet, data/test.parquet

Each output row is one (state, action, reward) tuple of the single-step
anchoring MDP:

    state  s = [log_list_price, seller_score_norm, seller_pos_pct, categ_id_clean]
    action a =  anchor_ratio   = buyer_offer / listing_price   in (0.01, 1.50]
    reward r =  savings_pct    = (1 - anchor_ratio) if deal else 0

Robustness split modes:

    SPLIT_MODE=random        reproducible random 80/10/10 split (default)
    SPLIT_MODE=temporal      chronological 80/10/10 split
    SPLIT_MODE=leaf_holdout  disjoint anonymized leaf-category split

All preprocessing statistics used by the model features are fit on train only:
seller-score p99, seller-positive median, and top leaf-category vocabulary.

Run:
    DATA_DIR=. OUT_DIR=./data python data_preprocess.py
    DATA_DIR=. OUT_DIR=./data_time SPLIT_MODE=temporal python data_preprocess.py
    DATA_DIR=. OUT_DIR=./data_leaf SPLIT_MODE=leaf_holdout python data_preprocess.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

from project_constants import (
    ANCHOR_MAX,
    ANCHOR_MIN,
    DEAL_STATUS,
    FASHION_CATEG_IDS,
    SEED,
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

DATE_CANDIDATES = ["src_cre_date", "src_cre_dt", "auct_start_dt", "auct_end_dt"]


# Column map (matches the merged parquet).
COL = {
    "item_id": "anon_item_id",
    "thread_id": "anon_thread_id",
    "list_price": "start_price_usd",
    "first_offer": "offr_price",
    "final_price": "item_price",
    "deal_status": "status_id",
    "meta_categ": "meta_categ_id",
    "leaf_categ": "anon_leaf_categ_id",
    "slr_score": "fdbk_score_src",
    "slr_pos_pct": "fdbk_pstv_src",
}


def check_columns(lf: pl.LazyFrame) -> None:
    cols = set(lf.collect_schema().names())
    required = {
        COL["item_id"],
        COL["list_price"],
        COL["first_offer"],
        COL["deal_status"],
        COL["meta_categ"],
    }
    missing = required - cols
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}\n"
            f"Available: {sorted(cols)}"
        )


def available_date_column(columns: list[str]) -> str | None:
    for name in DATE_CANDIDATES:
        if name in columns:
            return name
    return None


def add_interaction_time(lf: pl.LazyFrame) -> pl.LazyFrame:
    cols = lf.collect_schema().names()
    date_col = available_date_column(cols)
    if date_col is None:
        return lf.with_columns(pl.lit(None).cast(pl.Datetime).alias("interaction_ts"))

    if date_col in {"src_cre_date"}:
        parsed = pl.col(date_col).str.strptime(
            pl.Datetime, "%d%b%Y %H:%M:%S", strict=False
        )
    else:
        parsed = pl.col(date_col).str.strptime(
            pl.Date, "%d%b%Y", strict=False
        ).cast(pl.Datetime)
    return lf.with_columns(parsed.alias("interaction_ts"))


def build_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    c = COL
    if FILTER_FASHION:
        lf = lf.filter(pl.col(c["meta_categ"]).is_in(FASHION_CATEG_IDS))
    lf = lf.filter(pl.col(c["list_price"]) > 0)
    lf = lf.filter(pl.col(c["first_offer"]) > 0)

    lf = add_interaction_time(lf)
    lf = lf.with_columns(
        (pl.col(c["first_offer"]) / pl.col(c["list_price"])).alias("anchor_ratio"),
        pl.col(c["meta_categ"]).alias("meta_categ_id_raw"),
        pl.col(c["leaf_categ"]).alias("leaf_categ_id_raw")
        if c["leaf_categ"] in lf.collect_schema().names()
        else pl.lit(None).cast(pl.Int64).alias("leaf_categ_id_raw"),
    )
    lf = lf.filter(
        (pl.col("anchor_ratio") > ANCHOR_MIN) & (pl.col("anchor_ratio") <= ANCHOR_MAX)
    )

    # On an accepted Best Offer the buyer pays the accepted offer (offr_price),
    # so realized savings = 1 - offer/list. Non-deals receive zero savings.
    lf = lf.with_columns(
        pl.when(pl.col(c["deal_status"]) == DEAL_STATUS)
        .then(1.0 - pl.col(c["first_offer"]) / pl.col(c["list_price"]))
        .otherwise(0.0)
        .fill_null(0.0)
        .fill_nan(0.0)
        .clip(-0.5, 0.99)
        .alias("savings_pct"),
        (pl.col(c["list_price"]) + 1).log().alias("log_list_price"),
    )
    return lf


def split_random(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    n = len(df)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    train_end = int(TRAIN_FRAC * n)
    val_end = int((TRAIN_FRAC + VAL_FRAC) * n)
    return df[idx[:train_end]], df[idx[train_end:val_end]], df[idx[val_end:]]


def split_temporal(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if "interaction_ts" not in df.columns or df["interaction_ts"].null_count() == len(df):
        raise ValueError(
            "SPLIT_MODE=temporal requires one parseable date column. "
            f"Tried: {DATE_CANDIDATES}"
        )
    df = df.filter(pl.col("interaction_ts").is_not_null()).sort("interaction_ts")
    n = len(df)
    train_end = int(TRAIN_FRAC * n)
    val_end = int((TRAIN_FRAC + VAL_FRAC) * n)
    return df[:train_end], df[train_end:val_end], df[val_end:]


def split_leaf_holdout(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    leaf = "leaf_categ_id_raw"
    if leaf not in df.columns or df[leaf].null_count() == len(df):
        raise ValueError("SPLIT_MODE=leaf_holdout requires anon_leaf_categ_id.")

    counts = df.group_by(leaf).len().rename({"len": "n"})
    counts = counts.filter(pl.col(leaf).is_not_null())
    if len(counts) < 3:
        raise ValueError("Need at least three leaf categories for leaf_holdout split.")

    rng = np.random.default_rng(SEED)
    leaves = counts[leaf].to_numpy()
    leaves = leaves[rng.permutation(len(leaves))]

    n_leaf = len(leaves)
    n_train = max(1, int(round(TRAIN_FRAC * n_leaf)))
    n_val = max(1, int(round(VAL_FRAC * n_leaf)))
    if n_train + n_val >= n_leaf:
        n_train = max(1, n_leaf - 2)
        n_val = 1

    # Split whole leaf IDs, preserving train/validation/test disjointness. Row
    # proportions can be imbalanced when leaf sizes are imbalanced; the summary
    # file makes that visible.
    train_leaves = leaves[:n_train].tolist()
    val_leaves = leaves[n_train:n_train + n_val].tolist()
    test_leaves = leaves[n_train + n_val:].tolist()

    train = df.filter(pl.col(leaf).is_in(train_leaves))
    val = df.filter(pl.col(leaf).is_in(val_leaves))
    test = df.filter(pl.col(leaf).is_in(test_leaves))
    return train, val, test


def split_frame(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if SPLIT_MODE == "random":
        return split_random(df)
    if SPLIT_MODE == "temporal":
        return split_temporal(df)
    if SPLIT_MODE in {"leaf_holdout", "leaf-holdout"}:
        return split_leaf_holdout(df)
    raise ValueError(
        f"Unknown SPLIT_MODE={SPLIT_MODE!r}. Use random, temporal, or leaf_holdout."
    )


def fit_preprocess_stats(train: pl.DataFrame) -> dict:
    slr = COL["slr_score"]
    slr_pos = COL["slr_pos_pct"]
    leaf = "leaf_categ_id_raw"

    if slr in train.columns:
        seller_score_p99 = float(train[slr].quantile(0.99) or 1.0)
    else:
        seller_score_p99 = 1.0
    seller_score_p99 = max(seller_score_p99, 1e-6)

    if slr_pos in train.columns:
        seller_pos_median = float(train[slr_pos].median() or 0.95)
    else:
        seller_pos_median = 0.95

    if leaf in train.columns:
        top_leaves = (
            train[leaf]
            .drop_nulls()
            .value_counts(sort=True)
            .head(TOP_LEAF_N)[leaf]
            .to_list()
        )
    else:
        top_leaves = []

    return {
        "split_mode": SPLIT_MODE,
        "filter_fashion": FILTER_FASHION,
        "seller_score_p99": seller_score_p99,
        "seller_pos_median": seller_pos_median,
        "top_leaf_n": TOP_LEAF_N,
        "top_leaf_categories": [int(x) for x in top_leaves],
    }


def apply_preprocess_stats(df: pl.DataFrame, stats: dict) -> pl.DataFrame:
    slr = COL["slr_score"]
    slr_pos = COL["slr_pos_pct"]
    leaf = "leaf_categ_id_raw"

    if slr in df.columns:
        seller_score_expr = (pl.col(slr) / stats["seller_score_p99"]).clip(0, 1)
    else:
        seller_score_expr = pl.lit(0.5)

    if slr_pos in df.columns:
        seller_pos_expr = pl.col(slr_pos).fill_null(stats["seller_pos_median"])
    else:
        seller_pos_expr = pl.lit(0.95)

    top_leaves = stats["top_leaf_categories"]
    if leaf in df.columns:
        category_expr = (
            pl.when(pl.col(leaf).is_in(top_leaves))
            .then(pl.col(leaf))
            .otherwise(-1)
            .cast(pl.Int32)
        )
    else:
        category_expr = pl.lit(-1).cast(pl.Int32)

    return df.with_columns(
        seller_score_expr.fill_null(0.5).alias("seller_score_norm"),
        seller_pos_expr.fill_null(0.95).alias("seller_pos_pct"),
        category_expr.fill_null(-1).alias("categ_id_clean"),
    )


def select_mdp_columns(df: pl.DataFrame) -> pl.DataFrame:
    c = COL
    keep = [
        "log_list_price",
        "seller_score_norm",
        "seller_pos_pct",
        "categ_id_clean",
        "anchor_ratio",
        "savings_pct",
        c["deal_status"],
        c["list_price"],
        c["final_price"],
        c["first_offer"],
        c["item_id"],
        c["thread_id"],
        "interaction_ts",
        "meta_categ_id_raw",
        "leaf_categ_id_raw",
    ]
    return df.select([col for col in keep if col in df.columns])


def summarize_split(name: str, df: pl.DataFrame) -> dict:
    if len(df) == 0:
        return {"split": name, "n": 0}

    dates = df["interaction_ts"] if "interaction_ts" in df.columns else None
    non_null_dates = dates.drop_nulls() if dates is not None else []
    leaf_n = (
        int(df["leaf_categ_id_raw"].n_unique())
        if "leaf_categ_id_raw" in df.columns
        else 0
    )
    meta_n = (
        int(df["meta_categ_id_raw"].n_unique())
        if "meta_categ_id_raw" in df.columns
        else 0
    )

    return {
        "split": name,
        "n": int(len(df)),
        "deal_rate": float((df[COL["deal_status"]] == DEAL_STATUS).mean()),
        "mean_anchor": float(df["anchor_ratio"].mean()),
        "anchor_p5": float(df["anchor_ratio"].quantile(0.05)),
        "anchor_p95": float(df["anchor_ratio"].quantile(0.95)),
        "leaf_categories": leaf_n,
        "meta_categories": meta_n,
        "date_min": str(non_null_dates.min()) if len(non_null_dates) else "",
        "date_max": str(non_null_dates.max()) if len(non_null_dates) else "",
    }


def write_split_summary(train: pl.DataFrame, val: pl.DataFrame, test: pl.DataFrame, stats: dict) -> None:
    rows = [
        summarize_split("train", train),
        summarize_split("val", val),
        summarize_split("test", test),
    ]
    summary = pl.DataFrame(rows)
    summary.write_csv(OUT_DIR / "split_summary.csv")
    with open(OUT_DIR / "preprocess_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if SPLIT_MODE in {"leaf_holdout", "leaf-holdout"}:
        leaf = "leaf_categ_id_raw"
        train_leaves = set(train[leaf].drop_nulls().to_list())
        val_leaves = set(val[leaf].drop_nulls().to_list())
        test_leaves = set(test[leaf].drop_nulls().to_list())
        overlap = {
            "train_val": len(train_leaves & val_leaves),
            "train_test": len(train_leaves & test_leaves),
            "val_test": len(val_leaves & test_leaves),
        }
        with open(OUT_DIR / "leaf_holdout_overlap.json", "w", encoding="utf-8") as f:
            json.dump(overlap, f, indent=2)


def main():
    t0 = time.time()
    assert SOURCE.exists(), f"Missing file: {SOURCE.resolve()}"

    print("Scanning parquet ...")
    lf = pl.scan_parquet(SOURCE)
    check_columns(lf)

    print("Building base features ...")
    base = build_features(lf)
    if PREPROCESS_MAX_ROWS:
        base = base.head(PREPROCESS_MAX_ROWS)
        print(f"Smoke cap: reading at most {PREPROCESS_MAX_ROWS:,} rows (PREPROCESS_MAX_ROWS).")
    df = base.collect()
    print(f"Rows after feature filters: {len(df):,}")

    print(f"Splitting data with SPLIT_MODE={SPLIT_MODE!r} ...")
    train_raw, val_raw, test_raw = split_frame(df)
    stats = fit_preprocess_stats(train_raw)

    train = select_mdp_columns(apply_preprocess_stats(train_raw, stats))
    val = select_mdp_columns(apply_preprocess_stats(val_raw, stats))
    test = select_mdp_columns(apply_preprocess_stats(test_raw, stats))

    train.write_parquet(OUT_DIR / "train.parquet")
    val.write_parquet(OUT_DIR / "val.parquet")
    test.write_parquet(OUT_DIR / "test.parquet")
    write_split_summary(train, val, test, stats)

    deal_rate = float((df[COL["deal_status"]] == DEAL_STATUS).mean())
    print("\nDONE")
    print(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    print(f"Overall deal rate: {deal_rate:.3f}")
    print(f"Summary: {OUT_DIR / 'split_summary.csv'}")
    print(f"Time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
