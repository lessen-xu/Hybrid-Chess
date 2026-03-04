# -*- coding: utf-8 -*-
"""Experiment analysis and figure generation.

Reads metrics.csv from one or more training runs and produces the
4 completion-criterion figures for the evaluation protocol:

  1. Imbalance Diagnosis — Faction win-rate curve over training
  2. Branching Factor Disparity — Per-side histogram
  3. Strategic Depth — Game length distribution
  4. Equilibrium Convergence — Cross-run comparison

Usage:
  # Single run analysis
  python -m scripts.analyze_experiment \
      --run-dirs runs/experiment_vanilla \
      --labels Vanilla \
      --outdir runs/analysis

  # Three-run comparison
  python -m scripts.analyze_experiment \
      --run-dirs runs/experiment_vanilla runs/experiment_extra_cannon runs/experiment_no_queen \
      --labels "Vanilla" "Extra Cannon" "No Queen" \
      --outdir runs/analysis
"""

from __future__ import annotations

import scripts._fix_encoding  # noqa: F401
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ====================================================================
# Styling
# ====================================================================

COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ====================================================================
# Data loading
# ====================================================================

def load_metrics(run_dir: str) -> List[Dict[str, str]]:
    """Load metrics.csv from a run directory."""
    csv_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No metrics.csv in {run_dir}")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def safe_float(val: str, default: float = 0.0) -> float:
    """Convert string to float, returning default for empty/invalid."""
    try:
        return float(val) if val.strip() else default
    except (ValueError, AttributeError):
        return default


def safe_int(val: str, default: int = 0) -> int:
    try:
        return int(val) if val.strip() else default
    except (ValueError, AttributeError):
        return default


# ====================================================================
# Figure 1: Imbalance Diagnosis
# ====================================================================

def plot_imbalance(
    all_metrics: Dict[str, List[dict]],
    outdir: str,
):
    """Plot faction win-rate curve over training iterations.

    X = iteration, Y = Chess faction win rate from self-play data.
    """
    fig, ax = plt.subplots()

    for idx, (label, rows) in enumerate(all_metrics.items()):
        iters, chess_wr = [], []
        for r in rows:
            sp_games = safe_int(r.get("sp_games", "0"))
            cw = safe_int(r.get("sp_chess_wins", "0"))
            if sp_games > 0:
                iters.append(safe_int(r["iter"]))
                chess_wr.append(cw / sp_games)

        if iters:
            color = COLORS[idx % len(COLORS)]
            ax.plot(iters, chess_wr, "o-", color=color, label=label,
                    markersize=5, linewidth=2)

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.5,
               label="Balanced (50%)")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Chess Faction Win Rate (Self-Play)")
    ax.set_title("Faction Imbalance Diagnosis")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")

    out_path = os.path.join(outdir, "fig1_imbalance_diagnosis.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig1] Saved: {out_path}")


# ====================================================================
# Figure 2: Branching Factor Disparity
# ====================================================================

def plot_branching(
    all_metrics: Dict[str, List[dict]],
    outdir: str,
):
    """Plot per-side branching factor over training iterations."""
    fig, ax = plt.subplots()

    bar_width = 0.35
    labels_list = list(all_metrics.keys())
    x = np.arange(len(labels_list))

    chess_means, xiangqi_means = [], []

    for label, rows in all_metrics.items():
        chess_vals = [safe_float(r.get("sp_avg_legal_chess", "0"))
                      for r in rows if r.get("sp_avg_legal_chess")]
        xiangqi_vals = [safe_float(r.get("sp_avg_legal_xiangqi", "0"))
                        for r in rows if r.get("sp_avg_legal_xiangqi")]

        chess_mean = np.mean(chess_vals) if chess_vals else 0
        xiangqi_mean = np.mean(xiangqi_vals) if xiangqi_vals else 0
        chess_means.append(chess_mean)
        xiangqi_means.append(xiangqi_mean)

    ax.bar(x - bar_width / 2, chess_means, bar_width,
           label="Chess (International)", color="#2196F3", alpha=0.85)
    ax.bar(x + bar_width / 2, xiangqi_means, bar_width,
           label="Xiangqi (Chinese)", color="#FF5722", alpha=0.85)

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Average Legal Moves per Turn")
    ax.set_title("Branching Factor Disparity by Faction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, rotation=15)
    ax.legend()

    # Annotate values
    for i, (c, xq) in enumerate(zip(chess_means, xiangqi_means)):
        ax.text(i - bar_width / 2, c + 0.3, f"{c:.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + bar_width / 2, xq + 0.3, f"{xq:.1f}", ha="center", va="bottom", fontsize=9)

    out_path = os.path.join(outdir, "fig2_branching_factor.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig2] Saved: {out_path}")


# ====================================================================
# Figure 3: Strategic Depth (Game Length)
# ====================================================================

def plot_strategic_depth(
    all_metrics: Dict[str, List[dict]],
    outdir: str,
):
    """Plot game length distribution across experiments.

    Shows evolution of average ply over training for each run,
    plus a boxplot of final-phase game lengths.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Game length over training
    for idx, (label, rows) in enumerate(all_metrics.items()):
        iters = [safe_int(r["iter"]) for r in rows]
        avg_plies = [safe_float(r.get("sp_avg_ply", "0")) for r in rows]

        color = COLORS[idx % len(COLORS)]
        ax1.plot(iters, avg_plies, "o-", color=color, label=label,
                 markersize=5, linewidth=2)

    ax1.set_xlabel("Training Iteration")
    ax1.set_ylabel("Average Game Length (plies)")
    ax1.set_title("Strategic Depth Evolution")
    ax1.legend(loc="best")

    # Right: Boxplot of game lengths (all iterations)
    box_data = []
    box_labels = []
    for idx, (label, rows) in enumerate(all_metrics.items()):
        plies = [safe_float(r.get("sp_avg_ply", "0")) for r in rows
                 if safe_float(r.get("sp_avg_ply", "0")) > 0]
        if plies:
            box_data.append(plies)
            box_labels.append(label)

    if box_data:
        bp = ax2.boxplot(box_data, patch_artist=True, labels=box_labels)
        for patch, color in zip(bp["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
    ax2.set_ylabel("Game Length (plies)")
    ax2.set_title("Game Length Distribution")

    out_path = os.path.join(outdir, "fig3_strategic_depth.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig3] Saved: {out_path}")


# ====================================================================
# Figure 4: Equilibrium Convergence
# ====================================================================

def plot_equilibrium(
    all_metrics: Dict[str, List[dict]],
    outdir: str,
):
    """Compare final faction win rates across runs.

    Grouped bar chart: each run's converged Chess/Xiangqi/Draw rates.
    """
    fig, ax = plt.subplots()

    labels_list = list(all_metrics.keys())
    n_runs = len(labels_list)
    x = np.arange(n_runs)
    bar_width = 0.25

    chess_rates, xiangqi_rates, draw_rates = [], [], []

    for label, rows in all_metrics.items():
        # Use last 5 iterations as "converged" region
        recent = rows[-5:] if len(rows) >= 5 else rows
        total_cw = sum(safe_int(r.get("sp_chess_wins", "0")) for r in recent)
        total_xw = sum(safe_int(r.get("sp_xiangqi_wins", "0")) for r in recent)
        total_dr = sum(safe_int(r.get("sp_draws", "0")) for r in recent)
        total_g = total_cw + total_xw + total_dr

        if total_g > 0:
            chess_rates.append(total_cw / total_g)
            xiangqi_rates.append(total_xw / total_g)
            draw_rates.append(total_dr / total_g)
        else:
            chess_rates.append(0)
            xiangqi_rates.append(0)
            draw_rates.append(0)

    ax.bar(x - bar_width, chess_rates, bar_width,
           label="Chess Wins", color="#2196F3", alpha=0.85)
    ax.bar(x, draw_rates, bar_width,
           label="Draws", color="#9E9E9E", alpha=0.85)
    ax.bar(x + bar_width, xiangqi_rates, bar_width,
           label="Xiangqi Wins", color="#FF5722", alpha=0.85)

    ax.axhline(y=0.5, color="green", linestyle="--", linewidth=1,
               alpha=0.7, label="Balanced")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Faction Outcome Rate (Converged)")
    ax.set_title("Equilibrium Convergence Across Rule Variants")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")

    out_path = os.path.join(outdir, "fig4_equilibrium_convergence.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Fig4] Saved: {out_path}")


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation protocol figures from training metrics."
    )
    parser.add_argument("--run-dirs", nargs="+", required=True,
                        help="One or more run directories containing metrics.csv")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Display labels for each run (default: directory names)")
    parser.add_argument("--outdir", type=str, default="runs/analysis",
                        help="Output directory for figures (default: runs/analysis)")
    args = parser.parse_args()

    labels = args.labels or [Path(d).name for d in args.run_dirs]
    if len(labels) != len(args.run_dirs):
        parser.error("Number of labels must match number of run directories")

    # Load all metrics
    all_metrics = {}
    for run_dir, label in zip(args.run_dirs, labels):
        try:
            rows = load_metrics(run_dir)
            all_metrics[label] = rows
            print(f"  Loaded {len(rows)} rows from {run_dir} ({label})")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")

    if not all_metrics:
        print("ERROR: No metrics loaded. Exiting.")
        return

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print(f"\n  Generating figures in: {outdir}\n")

    plot_imbalance(all_metrics, outdir)
    plot_branching(all_metrics, outdir)
    plot_strategic_depth(all_metrics, outdir)
    plot_equilibrium(all_metrics, outdir)

    print(f"\n  All 4 figures generated in: {outdir}")


if __name__ == "__main__":
    main()
