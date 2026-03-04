# -*- coding: utf-8 -*-
"""Live experiment monitor — tracks progress of three-universe training runs.

Reads metrics.csv from each run directory and the experiment_progress.json
to display a unified dashboard.

Usage:
  python -m scripts.monitor_experiment
  python -m scripts.monitor_experiment --interval 120
"""

from __future__ import annotations

import scripts._fix_encoding  # noqa: F401
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ====================================================================
# Config
# ====================================================================

RUN_DIRS = [
    ("Vanilla",       "runs/experiment_vanilla"),
    ("Extra Cannon",  "runs/experiment_extra_cannon"),
    ("No Queen",      "runs/experiment_no_queen"),
]

PROGRESS_FILE = "runs/experiment_progress.json"
DASHBOARD_PNG = "runs/experiment_dashboard.png"

COLORS = {
    "Vanilla": "#2196F3",
    "Extra Cannon": "#FF5722",
    "No Queen": "#4CAF50",
}


# ====================================================================
# Data loading
# ====================================================================

def _load_csv(run_dir: str) -> dict:
    """Load metrics.csv into lists of values."""
    import csv
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return {"rows": rows, "count": len(rows)}
    except Exception:
        return {}


def _safe_float(val, default=0.0):
    try:
        return float(val) if val and str(val).strip() else default
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    try:
        return int(float(val)) if val and str(val).strip() else default
    except (ValueError, TypeError):
        return default


# ====================================================================
# Dashboard
# ====================================================================

def update_dashboard() -> bool:
    """Generate a 2x3 dashboard PNG. Returns True if any data was found."""
    has_data = False

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), dpi=150)
    fig.suptitle("Three-Universe Experiment Dashboard", fontsize=16, fontweight="bold")

    # Load progress
    progress = {}
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                progress = json.load(f)
        except Exception:
            pass

    all_data = {}
    for label, run_dir in RUN_DIRS:
        data = _load_csv(run_dir)
        if data:
            all_data[label] = data
            has_data = True

    # === Row 1: Training metrics ===

    # Panel 1: Total Loss
    ax = axes[0][0]
    for label, data in all_data.items():
        iters = [_safe_int(r["iter"]) for r in data["rows"]]
        losses = [_safe_float(r.get("total_loss")) for r in data["rows"]]
        ax.plot(iters, losses, "o-", color=COLORS.get(label, "gray"),
                label=f"{label} ({data['count']} iters)", markersize=3, linewidth=1.5)
    ax.set_title("Training Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: vs Random Wins
    ax = axes[0][1]
    for label, data in all_data.items():
        iters = [_safe_int(r["iter"]) for r in data["rows"]]
        rw = [_safe_int(r.get("eval_random_w")) for r in data["rows"]]
        ax.plot(iters, rw, "o-", color=COLORS.get(label, "gray"),
                label=label, markersize=3, linewidth=1.5)
    ax.set_title("vs Random: Wins (out of 20)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Wins")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: vs AB Wins
    ax = axes[0][2]
    for label, data in all_data.items():
        iters = [_safe_int(r["iter"]) for r in data["rows"]]
        aw = [_safe_int(r.get("eval_ab_w")) for r in data["rows"]]
        ax.plot(iters, aw, "o-", color=COLORS.get(label, "gray"),
                label=label, markersize=3, linewidth=1.5)
    ax.axhline(y=10, color="gray", linestyle="--", alpha=0.5, label="Breakthrough (10W)")
    ax.set_title("vs AlphaBeta-d1: Wins (out of 20)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Wins")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # === Row 2: Telemetry ===

    # Panel 4: Chess Faction Win Rate (self-play)
    ax = axes[1][0]
    for label, data in all_data.items():
        iters, chess_wr = [], []
        for r in data["rows"]:
            sg = _safe_int(r.get("sp_games"))
            cw = _safe_int(r.get("sp_chess_wins"))
            if sg > 0:
                iters.append(_safe_int(r["iter"]))
                chess_wr.append(cw / sg)
        if iters:
            ax.plot(iters, chess_wr, "o-", color=COLORS.get(label, "gray"),
                    label=label, markersize=3, linewidth=1.5)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Balanced")
    ax.set_title("Chess Faction Win Rate (Self-Play)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 5: Branching Factor
    ax = axes[1][1]
    for label, data in all_data.items():
        iters = [_safe_int(r["iter"]) for r in data["rows"]]
        bc = [_safe_float(r.get("sp_avg_legal_chess")) for r in data["rows"]]
        bx = [_safe_float(r.get("sp_avg_legal_xiangqi")) for r in data["rows"]]
        color = COLORS.get(label, "gray")
        if any(v > 0 for v in bc):
            ax.plot(iters, bc, "o-", color=color, label=f"{label} Chess",
                    markersize=3, linewidth=1.5)
            ax.plot(iters, bx, "s--", color=color, label=f"{label} Xiangqi",
                    markersize=3, linewidth=1.5, alpha=0.6)
    ax.set_title("Avg Legal Moves per Turn")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Legal Moves")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 6: Game Length
    ax = axes[1][2]
    for label, data in all_data.items():
        iters = [_safe_int(r["iter"]) for r in data["rows"]]
        avg_ply = [_safe_float(r.get("sp_avg_ply")) for r in data["rows"]]
        ax.plot(iters, avg_ply, "o-", color=COLORS.get(label, "gray"),
                label=label, markersize=3, linewidth=1.5)
    ax.set_title("Self-Play Game Length")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Avg Plies")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Status text
    status_text = _format_status(progress, all_data)
    fig.text(0.5, 0.01, status_text, ha="center", fontsize=9, style="italic",
             color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(DASHBOARD_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return has_data


def _format_status(progress: dict, all_data: dict) -> str:
    parts = []
    if progress:
        status = progress.get("status", "unknown")
        completed = progress.get("completed_runs", 0)
        total = progress.get("total_runs", 3)
        current = progress.get("current_run")
        if current:
            parts.append(f"Running: {current['label']} ({current.get('ablation', '')})")
        parts.append(f"Progress: {completed}/{total} runs | Status: {status}")
        if progress.get("start_time"):
            parts.append(f"Started: {progress['start_time']}")
    for label, data in all_data.items():
        parts.append(f"{label}: {data['count']} iters")
    return "  |  ".join(parts) if parts else "Waiting for data..."


# ====================================================================
# Main loop
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Live experiment monitor dashboard")
    parser.add_argument("--interval", type=int, default=120,
                        help="Refresh interval in seconds (default: 120)")
    args = parser.parse_args()

    print(f"[Monitor] Watching: {', '.join(d for _, d in RUN_DIRS)}")
    print(f"[Monitor] Progress: {PROGRESS_FILE}")
    print(f"[Monitor] Dashboard: {DASHBOARD_PNG}")
    print(f"[Monitor] Interval: {args.interval}s")
    print(f"[Monitor] Press Ctrl+C to stop.\n")

    while True:
        try:
            ok = update_dashboard()
            ts = time.strftime("%H:%M:%S")
            if ok:
                # Load progress for status line
                status_parts = []
                if os.path.exists(PROGRESS_FILE):
                    try:
                        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                            p = json.load(f)
                        completed = p.get("completed_runs", 0)
                        total = p.get("total_runs", 3)
                        current = p.get("current_run")
                        status_parts.append(f"Runs: {completed}/{total}")
                        if current:
                            status_parts.append(f"Current: {current['label']}")
                    except Exception:
                        pass

                for label, run_dir in RUN_DIRS:
                    csv_path = os.path.join(run_dir, "metrics.csv")
                    if os.path.exists(csv_path):
                        import csv
                        with open(csv_path, "r", encoding="utf-8") as f:
                            n = sum(1 for _ in csv.reader(f)) - 1  # subtract header
                        status_parts.append(f"{label}:{n}it")

                print(f"  [{ts}] Refreshed | {' | '.join(status_parts)}")
            else:
                print(f"  [{ts}] Waiting for data ...")
        except Exception as e:
            print(f"  [{time.strftime('%H:%M:%S')}] Error: {e}")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
