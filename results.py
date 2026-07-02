"""
results.py — Comparison dashboard
=================================
Loads the artifacts from Phases 1–3 and produces:
  1. policy_comparison.csv  — policy diagnostics with evidence-type labels
  2. results_dashboard.png  — CQL/PPO loss curves, anchor distributions, savings bars

Run AFTER phases 1–3.
    MODEL_DIR=./models python results.py
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def artifact(name, faithful_name=None):
    """Prefer faithful PPO artifacts when available, then fall back to legacy names."""
    if faithful_name is not None:
        faithful_path = MODEL_DIR / faithful_name
        if faithful_path.exists():
            return faithful_path
    return MODEL_DIR / name


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_comparison_table():
    bench = load_json(MODEL_DIR / "behavioral_benchmark.json")
    clf_m = load_json(MODEL_DIR / "clf_metrics.json")
    greedy = load_json(MODEL_DIR / "greedy_metrics.json")
    cql_m = load_json(MODEL_DIR / "cql_metrics.json")
    ppo_m = load_json(artifact("ppo_metrics.json", "ppo_metrics_faithful.json"))
    ppo_label = "PPO Faithful Sim" if ppo_m.get("faithful") else "PPO Simulated"

    rows = [
        {"Policy": "Behavioral (historical)",
         "Evidence Type": "observed logged outcome",
         "E[Savings]": bench["mean_savings_all"],
         "Deal Rate": bench["deal_rate"],
         "Mean Anchor": bench["mean_anchor_ratio"],
         "AUC (Phase 1)": clf_m["test"]["auc"]},
        {"Policy": f"Supervised Greedy (anchor={greedy['greedy_anchor']:.2f})",
         "Evidence Type": "Phase-1 model estimate",
         "E[Savings]": greedy["greedy_e_savings"],
         "Deal Rate": greedy["greedy_mean_p_deal"],
         "Mean Anchor": greedy["greedy_anchor"],
         "AUC (Phase 1)": None},
        {"Policy": "CQL (Offline RL)",
         "Evidence Type": "Phase-1 model estimate",
         "E[Savings]": cql_m["cql_e_savings_sim"],
         "Deal Rate": cql_m.get("cql_mean_p_deal"),
         "Mean Anchor": cql_m["cql_mean_anchor"],
         "AUC (Phase 1)": None},
        {"Policy": ppo_label,
         "Evidence Type": "simulator-only estimate",
         "E[Savings]": ppo_m["ppo_e_savings"],
         "Deal Rate": ppo_m["ppo_mean_p_deal"],
         "Mean Anchor": ppo_m["ppo_mean_anchor"],
         "AUC (Phase 1)": None},
    ]
    df = pd.DataFrame(rows).set_index("Policy")
    print("\n═══ Policy Comparison Table ════════════════════════════════")
    print(df.to_string())
    print(
        "\nNote: evidence types are not interchangeable. Behavioral rows are "
        "observed logged outcomes; PPO rows are simulator-only estimates."
    )
    df.to_csv(OUTPUT_DIR / "policy_comparison.csv")
    return df


def plot_all():
    cql_hist = load_json(MODEL_DIR / "cql_history.json")
    ppo_hist = load_json(artifact("ppo_history.json", "ppo_history_faithful.json"))
    bench = load_json(MODEL_DIR / "behavioral_benchmark.json")
    cql_m = load_json(MODEL_DIR / "cql_metrics.json")
    ppo_m = load_json(artifact("ppo_metrics.json", "ppo_metrics_faithful.json"))
    greedy = load_json(MODEL_DIR / "greedy_metrics.json")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("eBay Buyer Anchoring — RL Training Results",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # 1. CQL loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(cql_hist["td_loss"], label="TD loss", linewidth=1.5)
    ax1.plot(cql_hist["cql_loss"], label="CQL penalty", linewidth=1.5)
    ax1.plot(cql_hist["val_loss"], label="Val loss", linewidth=1.5, linestyle="--")
    ax1.set_title("CQL Training Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # 2. PPO mean reward
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ppo_hist["step"], ppo_hist["mean_reward"], color="darkorange", linewidth=1.5)
    ax2.set_title("PPO Mean Episode Reward"); ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Mean Reward (E[Savings])"); ax2.grid(alpha=0.3)

    # 3. PPO entropy
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(ppo_hist["step"], ppo_hist["entropy"], color="green", linewidth=1.5)
    ax3.set_title("PPO Policy Entropy (Exploration)"); ax3.set_xlabel("Environment Steps")
    ax3.set_ylabel("Entropy"); ax3.grid(alpha=0.3)

    # 4. Anchor ratio distributions
    ax4 = fig.add_subplot(gs[1, :2])
    from scipy.stats import norm
    x = np.linspace(0.0, 1.5, 300)
    mu_b, std_b = bench["mean_anchor_ratio"], max(bench["std_anchor_ratio"], 1e-3)
    mu_c, std_c = cql_m["cql_mean_anchor"], max(cql_m["cql_std_anchor"], 1e-3)
    mu_p, std_p = ppo_m["ppo_mean_anchor"], max(ppo_m["ppo_std_anchor"], 1e-3)
    ax4.fill_between(x, norm.pdf(x, mu_b, std_b), alpha=0.35, label=f"Behavioral (μ={mu_b:.2f})")
    ax4.fill_between(x, norm.pdf(x, mu_c, std_c), alpha=0.35, label=f"CQL (μ={mu_c:.2f})")
    ax4.fill_between(x, norm.pdf(x, mu_p, std_p), alpha=0.35, label=f"PPO (μ={mu_p:.2f})")
    ax4.axvline(1.0, color="gray", linestyle=":", alpha=0.6, label="Listing price (=1.0)")
    ax4.set_title("Anchor Ratio Distributions by Policy")
    ax4.set_xlabel("Anchor Ratio (buyer offer / listing price)")
    ax4.set_ylabel("Density"); ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    # 5. E[Savings] bar chart
    ax5 = fig.add_subplot(gs[1, 2])
    labels = ["Behavioral", "Greedy", "CQL", "PPO"]
    savings = [bench["mean_savings_all"], greedy["greedy_e_savings"],
               cql_m["cql_e_savings_sim"], ppo_m["ppo_e_savings"]]
    colors = ["steelblue", "goldenrod", "coral", "mediumseagreen"]
    bars = ax5.bar(labels, savings, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, savings):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax5.set_title("Expected Savings by Policy (mixed evidence types)")
    ax5.set_ylabel("E[Savings] (fraction of listing price)")
    ax5.set_ylim(0, max(savings) * 1.25 + 1e-6); ax5.grid(axis="y", alpha=0.3)

    plt.savefig(OUTPUT_DIR / "results_dashboard.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {OUTPUT_DIR / 'results_dashboard.png'}")
    plt.close()


def main():
    print("Building results comparison …")
    try:
        build_comparison_table()
        plot_all()
        print(f"\n✅ All outputs saved to {OUTPUT_DIR}/")
    except FileNotFoundError as e:
        print(f"⚠️  Missing artifact: {e}")
        print("   Run phases 1–3 first before generating results.")


if __name__ == "__main__":
    main()
