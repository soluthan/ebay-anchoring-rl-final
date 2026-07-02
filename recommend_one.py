"""
recommend_one.py — offer MENU for a single (real) eBay fashion listing
======================================================================
Instead of one "optimal" number, this shows a spectrum of strategies and tells
you WHEN to use each, because the right discount depends on how much you want the
item vs. the price:

    Conservative (CQL)        ~highest acceptance, smallest discount
    Balanced (typical buyer)  ~what experienced buyers do
    Aggressive (faithful PPO) ~bigger discount, lower acceptance (edge of support)

Each row shows the $ offer, model P(accept), the discount, and a support flag.

Example:
    python recommend_one.py --price 120 --seller_score 4500 --pos_pct 99.2

Caveats: category is anonymised in training (ignored); p_accept is a MODEL estimate;
the aggressive row sits at the edge of historical support, so treat it as the
risk-tolerant end, not a guarantee. This is decision support, not a pricing oracle.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch                       # torch-first to avoid the macOS OpenMP deadlock
    torch.set_num_threads(1)
except Exception:
    torch = None

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl

from project_constants import ANCHOR_MAX, ANCHOR_MIN, N_GRID, STATE_COLS

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models"))
SOURCE = Path(os.environ.get("DATA_DIR_RAW", ".")) / "clean_master_dataset.parquet"

GRID = np.linspace(ANCHOR_MIN, ANCHOR_MAX, N_GRID).astype(np.float32)


def norm_p99():
    if SOURCE.exists():
        s = pl.scan_parquet(SOURCE).select("fdbk_score_src").collect()["fdbk_score_src"]
        return float(s.quantile(0.99) or 5000.0)
    return 5000.0


def support_band():
    a = pl.scan_parquet(DATA_DIR / "train.parquet").select("anchor_ratio").collect()["anchor_ratio"].to_numpy()
    return float(np.percentile(a, 5)), float(np.percentile(a, 95))


def flag(anchor, p5, p95):
    return "OK" if (p5 <= anchor <= p95) else "edge/out-of-support"


def build_state(price, score, pos_pct, p99, category=-1):
    return np.array([[np.log(price + 1.0), np.clip(score / p99, 0, 1), pos_pct, category]], dtype=np.float32)


def p_accept(clf, state, anchor):
    feat = np.column_stack([state, [[anchor]]]).astype(np.float32)
    return float(clf.predict_proba(feat)[0, 1])


def cql_anchor(state):
    ckpt, scl = MODEL_DIR / "cql_best.pt", MODEL_DIR / "cql_scaler.pkl"
    if torch is None or not (ckpt.exists() and scl.exists()):
        return None
    from phase2_cql import QNetwork
    scaler = pickle.load(open(scl, "rb"))
    net = QNetwork(len(STATE_COLS)); net.load_state_dict(torch.load(ckpt, map_location="cpu")); net.eval()
    s = torch.tensor(scaler.transform(state).astype(np.float32))
    with torch.no_grad():
        q = net(s.expand(N_GRID, -1), torch.tensor(GRID))
    return float(GRID[int(q.argmax())])


def ppo_anchor(state, faithful=True):
    tag = "_faithful" if faithful else ""
    ckpt, scl = MODEL_DIR / f"ppo_best{tag}.pt", MODEL_DIR / "cql_scaler.pkl"
    if torch is None or not (ckpt.exists() and scl.exists()):
        return None
    try:
        from phase3_ppo import ActorCritic
        scaler = pickle.load(open(scl, "rb"))
        ac = ActorCritic(len(STATE_COLS)); ac.load_state_dict(torch.load(ckpt, map_location="cpu")); ac.eval()
        s = torch.tensor(scaler.transform(state).astype(np.float32))
        with torch.no_grad():
            mu, _, _ = ac.forward(s)
        a_norm = float(np.clip(mu.item(), -1, 1))
        return ANCHOR_MIN + (a_norm + 1.0) / 2.0 * (ANCHOR_MAX - ANCHOR_MIN)
    except Exception as e:
        print(f"  [ppo] skipped ({e})")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", type=float, required=True)
    ap.add_argument("--seller_score", type=float, default=1000)
    ap.add_argument("--pos_pct", type=float, default=99.0)
    args = ap.parse_args()

    import xgboost as xgb
    clf = xgb.XGBClassifier(); clf.load_model(str(MODEL_DIR / "deal_classifier.ubj"))
    p99 = norm_p99(); p5, p95 = support_band()
    state = build_state(args.price, args.seller_score, args.pos_pct, p99)

    # typical anchor from the behavioural benchmark
    bench = json.load(open(MODEL_DIR / "behavioral_benchmark.json"))
    typical = float(bench.get("mean_anchor_ratio", 0.67))

    rows = []
    a_cql = cql_anchor(state)
    if a_cql is not None:
        rows.append(("Conservative (CQL)", a_cql, "you want THIS item; maximise chance it closes"))
    rows.append(("Balanced (typical buyer)", typical, "sensible default — what experienced buyers do"))
    a_ppo = ppo_anchor(state, faithful=True)
    if a_ppo is not None:
        rows.append(("Aggressive (faithful PPO)", a_ppo, "price-first / reselling; OK to risk losing the deal"))

    rows.sort(key=lambda r: -r[1])   # safe (high anchor) -> aggressive (low anchor)

    print(f"\nListing ${args.price:,.2f}  |  seller score {args.seller_score:,.0f}  |  {args.pos_pct}% positive")
    print(f"(typical offers run {p5:.0%}–{p95:.0%} of list)\n")
    print(f"  {'Strategy':<26}{'Offer':>10}{'Discount':>10}{'P(accept)':>11}{'Support':>10}   Use when")
    print("  " + "-" * 96)
    for name, a, use in rows:
        pa = p_accept(clf, state, a)
        print(f"  {name:<26}{('$%.2f' % (a*args.price)):>10}{('%.0f%%' % ((1-a)*100)):>10}"
              f"{('%.0f%%' % (pa*100)):>11}{flag(a,p5,p95):>10}   {use}")
    print("\n  Read it as a risk dial: higher offer = more likely to close, smaller saving.")
    print("  P(accept) is a model estimate. The aggressive row sits at the edge of what")
    print("  buyers historically tried — treat it as the risk-tolerant end, not a promise.\n")


if __name__ == "__main__":
    main()
