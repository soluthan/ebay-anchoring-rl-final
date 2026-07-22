"""Build the final policy table and evidence-separated dashboard."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_optional(path: Path):
    return load_json(path) if path.exists() else None


def build_comparison_table() -> pd.DataFrame:
    behavior = load_json(MODEL_DIR / "behavioral_benchmark.json")
    classifier = load_json(MODEL_DIR / "clf_metrics.json")["test"]
    fixed = load_json(MODEL_DIR / "fixed_anchor_metrics.json")
    greedy = load_json(MODEL_DIR / "greedy_metrics.json")
    cql = load_json(MODEL_DIR / "cql_metrics.json")
    ppo_basic = load_optional(MODEL_DIR / "ppo_metrics.json")
    ppo_robust = load_optional(MODEL_DIR / "ppo_metrics_faithful.json")

    rows = [
        {
            "Policy": "Historical behavior",
            "Evidence Type": "observed immediate outcomes",
            "Expected Savings": behavior["mean_savings_all"],
            "Acceptance": behavior["opening_acceptance_rate"],
            "Mean Anchor": behavior["mean_anchor_ratio"],
            "Within p5-p95 Support": 1.0,
        },
        {
            "Policy": "Fixed anchor 0.70",
            "Evidence Type": "Phase-1 model estimate",
            "Expected Savings": fixed["mean_expected_savings"],
            "Acceptance": fixed["mean_p_accept"],
            "Mean Anchor": fixed["mean_anchor"],
            "Within p5-p95 Support": fixed["within_p5_p95_support_fraction"],
        },
        {
            "Policy": "Supervised greedy",
            "Evidence Type": "Phase-1 model estimate",
            "Expected Savings": greedy["mean_expected_savings"],
            "Acceptance": greedy["mean_p_accept"],
            "Mean Anchor": greedy["mean_anchor"],
            "Within p5-p95 Support": greedy["within_p5_p95_support_fraction"],
        },
        {
            "Policy": "CQL support-conservative",
            "Evidence Type": "Phase-1 model estimate",
            "Expected Savings": cql["cql_e_savings_sim"],
            "Acceptance": cql["cql_mean_p_accept"],
            "Mean Anchor": cql["cql_mean_anchor"],
            "Within p5-p95 Support": cql["cql_within_p5_p95_support_fraction"],
        },
    ]
    for name, metrics in (("PPO basic", ppo_basic), ("PPO robust", ppo_robust)):
        if metrics is not None:
            rows.append(
                {
                    "Policy": name,
                    "Evidence Type": "simulator-only estimate",
                    "Expected Savings": metrics["ppo_e_savings"],
                    "Acceptance": metrics["ppo_mean_p_accept"],
                    "Mean Anchor": metrics["ppo_mean_anchor"],
                    "Within p5-p95 Support": None,
                }
            )

    table = pd.DataFrame(rows).set_index("Policy")
    table["Classifier AUC"] = classifier["auc"]
    table["Classifier Brier"] = classifier["brier"]
    table.to_csv(OUTPUT_DIR / "policy_comparison.csv")
    print("\nPolicy comparison (evidence types must not be mixed):")
    print(table.to_string())

    if ppo_basic and ppo_robust:
        basic_value = float(ppo_basic["ppo_e_savings"])
        robust_value = float(ppo_robust["ppo_e_savings"])
        shrinkage = (basic_value - robust_value) / basic_value if basic_value else float("nan")
        with open(OUTPUT_DIR / "ppo_robustness.json", "w", encoding="utf-8") as file:
            json.dump(
                {
                    "basic_simulator_expected_savings": basic_value,
                    "robust_simulator_expected_savings": robust_value,
                    "relative_shrinkage": shrinkage,
                    "interpretation": "Simulator-only H3 sensitivity; not live-marketplace lift.",
                },
                file,
                indent=2,
            )
    return table


def plot_dashboard(table: pd.DataFrame) -> None:
    calibration = pd.read_csv(MODEL_DIR / "clf_calibration.csv")
    response = pd.read_csv(MODEL_DIR / "anchor_response_curve.csv")
    cql_history = load_json(MODEL_DIR / "cql_history.json")
    ppo_basic_history = load_optional(MODEL_DIR / "ppo_history.json")
    ppo_robust_history = load_optional(MODEL_DIR / "ppo_history_faithful.json")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Opening-offer policy diagnostics", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(response["anchor_ratio"], response["mean_predicted_acceptance"], label="P(accept)")
    ax.plot(
        response["anchor_ratio"],
        response["mean_predicted_expected_savings"],
        label="Expected savings",
    )
    ax.set(title="H1: response over anchor grid", xlabel="Opening-offer ratio")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(
        calibration["mean_predicted_probability"],
        calibration["observed_acceptance"],
        marker="o",
        label="classifier",
    )
    ax.set(title="Acceptance calibration", xlabel="Predicted", ylabel="Observed")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(cql_history["td_loss"], label="reward regression")
    ax.plot(cql_history["cql_loss"], label="CQL penalty")
    ax.plot(cql_history["val_loss"], "--", label="validation")
    ax.set(title="One-step CQL training", xlabel="Epoch")
    ax.legend(); ax.grid(alpha=0.3)

    model_rows = table[table["Evidence Type"] == "Phase-1 model estimate"]
    ax = axes[1, 0]
    ax.bar(model_rows.index, model_rows["Expected Savings"], color=["#d9a441", "#c85a54", "#4f81bd"])
    ax.set(title="Model-estimated policies", ylabel="Expected immediate savings")
    ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    ax.bar(model_rows.index, model_rows["Within p5-p95 Support"], color=["#d9a441", "#c85a54", "#4f81bd"])
    ax.set(title="H2: historical support", ylabel="Fraction inside p5-p95", ylim=(0, 1.05))
    ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 2]
    if ppo_basic_history:
        ax.plot(ppo_basic_history["step"], ppo_basic_history["mean_reward"], label="basic")
    if ppo_robust_history:
        ax.plot(ppo_robust_history["step"], ppo_robust_history["mean_reward"], label="robust")
    ax.set(title="H3: PPO simulator sensitivity", xlabel="Simulator steps", ylabel="Mean reward")
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "results_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    try:
        table = build_comparison_table()
        plot_dashboard(table)
        print(f"\nOutputs saved to {OUTPUT_DIR}")
    except FileNotFoundError as error:
        print(f"Missing corrected-pipeline artifact: {error}")
        print("Rerun preprocessing and Phases 1-3; old artifacts are not compatible.")


if __name__ == "__main__":
    main()
