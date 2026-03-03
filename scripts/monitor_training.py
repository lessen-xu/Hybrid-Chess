# -*- coding: utf-8 -*-
"""Live training monitor — periodically reads metrics.csv and saves a dashboard PNG.

Usage:
  python -m scripts.monitor_training --run-dir runs/az_grand_run
  python -m scripts.monitor_training --run-dir runs/az_grand_run --interval 30
"""

from __future__ import annotations

import argparse
import os
import time

import matplotlib.pyplot as plt
import pandas as pd


def update_dashboard(csv_path: str, png_path: str) -> bool:
    """Read metrics CSV and overwrite the dashboard PNG. Returns True on success."""
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return False
    except Exception:
        return False  # concurrent-write conflict; retry next tick

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    # --- Panel 1: Loss curves ---
    if {"iter", "total_loss", "value_loss", "policy_loss"}.issubset(df.columns):
        axes[0].plot(df["iter"], df["total_loss"], label="Total", color="black", lw=2)
        axes[0].plot(df["iter"], df["value_loss"], label="Value", color="red", ls="--")
        axes[0].plot(df["iter"], df["policy_loss"], label="Policy", color="blue", ls=":")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Iteration")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # --- Panel 2: Evaluation win counts ---
    if {"iter", "eval_random_w", "eval_ab_w"}.issubset(df.columns):
        # eval columns may be empty when eval is skipped; coerce to numeric
        rw = pd.to_numeric(df["eval_random_w"], errors="coerce")
        aw = pd.to_numeric(df["eval_ab_w"], errors="coerce")
        axes[1].plot(df["iter"], rw, label="vs Random W", color="green", marker="o", ms=4)
        axes[1].plot(df["iter"], aw, label="vs AlphaBeta W", color="blue", marker="^", ms=4)
        axes[1].set_title("Evaluation Wins")
        axes[1].set_xlabel("Iteration")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # --- Panel 3: Self-play game length ---
    if {"iter", "sp_avg_ply"}.issubset(df.columns):
        axes[2].plot(df["iter"], df["sp_avg_ply"], color="purple", marker="x", ms=4,
                     label="Avg Ply")
        if "sp_p90_ply" in df.columns:
            axes[2].plot(df["iter"], df["sp_p90_ply"], color="orange", ls="--",
                         label="P90 Ply")
        axes[2].set_title("Self-Play Game Length")
        axes[2].set_xlabel("Iteration")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Live training dashboard monitor")
    parser.add_argument("--run-dir", type=str, default="runs/az_grand_run",
                        help="training run directory (default: runs/az_grand_run)")
    parser.add_argument("--interval", type=int, default=60,
                        help="refresh interval in seconds (default: 60)")
    args = parser.parse_args()

    csv_path = os.path.join(args.run_dir, "metrics.csv")
    png_path = os.path.join(args.run_dir, "training_progress.png")
    os.makedirs(args.run_dir, exist_ok=True)

    print(f"[Monitor] Watching: {csv_path}")
    print(f"[Monitor] Output:   {png_path}")
    print(f"[Monitor] Interval: {args.interval}s")
    print(f"[Monitor] Press Ctrl+C to stop.\n")

    while True:
        ok = update_dashboard(csv_path, png_path)
        if ok:
            n = len(pd.read_csv(csv_path))
            print(f"  [{time.strftime('%H:%M:%S')}] Refreshed ({n} iterations plotted)")
        else:
            print(f"  [{time.strftime('%H:%M:%S')}] Waiting for data ...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
