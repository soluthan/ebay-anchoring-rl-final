"""
run_pipeline.py — master orchestration
=======================================
Runs the phases in sequence using the FLAT module layout in this folder
(fixes the original "phantom import" bug — there is no data/, models/,
agents/, envs/ package; everything lives side-by-side here).

Usage:
    python run_pipeline.py             # prep -> 1 -> 2 -> 3 -> OPE -> recommend -> results
    OUT_DIR=./data_time SPLIT_MODE=temporal python run_pipeline.py
    python run_pipeline.py --phase 1                  # just Phase 1
    python run_pipeline.py --phase prep               # just feature engineering

Environment variables:
    DATA_DIR   : where clean_master_dataset.parquet lives (for --phase prep)
                 and where data/{train,val,test}.parquet are read from.
                 NOTE: data_preprocess writes to OUT_DIR (default ./data),
                 while phases read DATA_DIR (default ./data). Keep them aligned:
                 run prep with DATA_DIR=<folder containing the merged parquet>,
                 then run phases with DATA_DIR=./data (the default).
    MODEL_DIR  : where model files / metrics are saved (default ./models)

The threads+lists CSV merge ("Phase 0") is a one-off and is intentionally NOT
orchestrated here — it is assumed clean_master_dataset.parquet already exists.
"""

import argparse
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    # Import torch before Phase 1 imports XGBoost. On macOS, initializing
    # XGBoost's OpenMP runtime first can freeze the later CQL/PPO torch phases.
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
except Exception:
    torch = None


def run_phase(name: str, fn):
    print(f"\n{'=' * 60}\n {name}\n{'=' * 60}")
    t0 = time.time()
    fn()
    print(f"\n ✅ {name} finished in {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["prep", "1", "2", "3", "ope", "recommend", "results", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    args = parser.parse_args()
    phases = (
        ["prep", "1", "2", "3", "ope", "recommend", "results"]
        if args.phase == "all"
        else [args.phase]
    )

    if "prep" in phases:
        from data_preprocess import main as prep
        run_phase("Feature Engineering (build train/val/test)", prep)
        if args.phase == "all":
            os.environ["DATA_DIR"] = os.environ.get("OUT_DIR", "./data")

    if "1" in phases:
        from phase1_supervised import main as phase1
        run_phase("Phase 1: Supervised Baselines (XGBoost)", phase1)

    if "2" in phases:
        from phase2_cql import main as phase2
        run_phase("Phase 2: Offline RL — CQL", phase2)

    if "3" in phases:
        from phase3_ppo import main as phase3
        run_phase("Phase 3: Online RL — PPO Simulator", phase3)

    if "ope" in phases:
        from ope import main as ope
        run_phase("OPE: Propensity-Weighted Diagnostics", ope)

    if "recommend" in phases:
        from recommend import main as recommend
        run_phase("Recommendations: Policy Menu and Support", recommend)

    if "results" in phases:
        from results import main as results
        run_phase("Results: Comparison Dashboard", results)

    print("\n🎉 Pipeline complete.")


if __name__ == "__main__":
    main()
