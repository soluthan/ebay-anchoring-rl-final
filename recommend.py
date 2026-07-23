"""
recommend.py — practical offer-ratio recommender + support diagnostics
======================================================================
Per-listing recommendation table + the H2 support/extrapolation diagnostics.

For each sampled test listing it reports, for the Greedy and (if available) CQL
policies: recommended anchor ratio, recommended $ offer, model P(accept), model
expected savings, and a SUPPORT WARNING flag.

Honest framing: p_accept / exp_savings are MODEL estimates (greedy maximises the
model, so its numbers are optimistic). The SUPPORT flag is the trustworthy part —
it marks anchors outside the historical 5–95% band or in sparse bins (model
extrapolating). Prefer CQL where greedy is flagged.

NOTE (macOS): torch is imported BEFORE xgboost on purpose. XGBoost and PyTorch
each ship an OpenMP runtime and deadlock if XGBoost initialises first; importing
torch first (and KMP_DUPLICATE_LIB_OK) avoids the hang.

Run (after Phase 1; CQL column appears once Phase 2 has run):
    DATA_DIR=./data MODEL_DIR=./models python recommend.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # avoid libomp double-load deadlock
try:
    import torch                                          # torch-first (see note above)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
except Exception:
    torch = None

import json
from pathlib import Path

import numpy as np
import pandas as pd

from policy_utils import ACTION_GRID, greedy_policy, score_actions
from project_constants import (
    ACTION_COL,
    ANCHOR_MAX,
    ANCHOR_MIN,
    CLASSIFIER_FILE,
    ITEM_COL,
    LIST_COL,
    N_GRID,
    SEED,
    STATE_COLS,
)

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models"))
OUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRID = ACTION_GRID

N_SAMPLE = int(os.environ.get("RECO_N", "5000"))
SPARSE_FRAC = 0.01


# ── historical support model (from training anchors) ──────────────
def build_support(train):
    a = train[ACTION_COL].values
    p5, p95 = np.percentile(a, [5, 95])
    edges = np.linspace(ANCHOR_MIN, ANCHOR_MAX, 21)
    counts, _ = np.histogram(a, bins=edges)
    dens = counts / max(counts.sum(), 1)
    sparse = dens < SPARSE_FRAC

    def warn(anchors):
        anchors = np.asarray(anchors)
        out_band = (anchors < p5) | (anchors > p95)
        bin_idx = np.clip(np.digitize(anchors, edges) - 1, 0, len(sparse) - 1)
        in_sparse = sparse[bin_idx]
        return np.where(out_band | in_sparse, "EXTRAPOLATED", "OK")

    return {"p5": float(p5), "p95": float(p95), "warn": warn}


def score_anchor(clf, states, anchors):
    return score_actions(clf, states, anchors)


# ── optional CQL policy ───────────────────────────────────────────
def load_cql():
    ckpt = MODEL_DIR / "cql_best.pt"
    scl = MODEL_DIR / "cql_scaler.pkl"
    if not (ckpt.exists() and scl.exists()) or torch is None:
        return None
    try:
        import pickle
        from phase2_cql import QNetwork
        with open(scl, "rb") as f:
            scaler = pickle.load(f)
        net = QNetwork(len(STATE_COLS))
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        net.eval()
        grid = torch.tensor(GRID)

        def policy(states, chunk=4096):
            outs = []
            for i in range(0, len(states), chunk):
                S = scaler.transform(states[i:i + chunk]).astype(np.float32)
                S = torch.tensor(S)
                B = S.shape[0]
                s_exp = S.unsqueeze(1).expand(-1, N_GRID, -1).reshape(B * N_GRID, -1)
                a_exp = grid.unsqueeze(0).expand(B, -1).reshape(-1)
                with torch.no_grad():
                    q = net(s_exp, a_exp).reshape(B, N_GRID)
                outs.append(GRID[q.argmax(1).numpy()])
            return np.concatenate(outs)
        return policy
    except Exception as e:
        print(f"  [cql] skipped ({e})")
        return None


def main():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    import xgboost as xgb
    clf = xgb.XGBClassifier(); clf.load_model(str(MODEL_DIR / CLASSIFIER_FILE))

    support = build_support(train)

    sample = test.sample(min(N_SAMPLE, len(test)), random_state=SEED).reset_index(drop=True)
    states = sample[STATE_COLS].values.astype(np.float32)
    list_px = sample[LIST_COL].values.astype(np.float32)

    out = pd.DataFrame({
        "item_id": sample[ITEM_COL].values if ITEM_COL in sample else np.arange(len(sample)),
        "list_price": np.round(list_px, 2),
        "historical_anchor": np.round(sample[ACTION_COL].values, 3),
    })

    ga, gp, ges, _ = greedy_policy(clf, states)
    gflag = support["warn"](ga)
    out["greedy_anchor"] = np.round(ga, 3)
    out["greedy_offer_usd"] = np.round(ga * list_px, 2)
    out["greedy_p_accept"] = np.round(gp, 3)
    out["greedy_exp_savings"] = np.round(ges, 3)
    out["greedy_support"] = gflag

    cql = load_cql()
    ca = cflag = None
    if cql is not None:
        ca = cql(states)
        cp, ces = score_anchor(clf, states, ca)
        cflag = support["warn"](ca)
        out["cql_anchor"] = np.round(ca, 3)
        out["cql_offer_usd"] = np.round(ca * list_px, 2)
        out["cql_p_accept"] = np.round(cp, 3)
        out["cql_exp_savings"] = np.round(ces, 3)
        out["cql_support"] = cflag

    out.to_csv(OUT_DIR / "offer_recommendations.csv", index=False)

    print(f"\nHistorical support band: anchors p5={support['p5']:.3f}  p95={support['p95']:.3f}")
    print(f"Listings in table: {len(out):,}  ->  {OUT_DIR/'offer_recommendations.csv'}\n")

    def row(name, a, flag):
        share = float((np.asarray(flag) == "EXTRAPOLATED").mean())
        print(f"  {name:<10} mean_anchor={np.mean(a):.3f}  std={np.std(a):.3f}  "
              f"%extrapolated={share*100:5.1f}%")

    print("── Support / extrapolation diagnostics (H2) ──")
    row("Historical", sample[ACTION_COL].values, support["warn"](sample[ACTION_COL].values))
    row("Greedy", ga, gflag)
    if cql is not None:
        row("CQL", ca, cflag)
    else:
        print("  CQL        (run phase2_cql.py to add this row + the cql_* columns)")

    print(
        "\nReminder: p_accept / exp_savings are MODEL estimates; trust both support flags."
        "\nUse CQL when greedy is EXTRAPOLATED only if CQL is inside support; otherwise "
        "use fixed 0.70."
    )


if __name__ == "__main__":
    main()
