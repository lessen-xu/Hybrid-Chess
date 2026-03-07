# -*- coding: utf-8 -*-
"""Sprint 6 Step 3+4: Generate statistical tables and paper figures.

Produces:
  - CI-annotated gate tables
  - table_side_masking.csv
  - Figure 1-3 (matplotlib)

Usage:
  python -m scripts.add_stats
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from scripts.stats_utils import wilson_ci, bootstrap_ci

PAPER_DIR = Path("paper")
FIG_DIR = PAPER_DIR / "figures"


def _load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ====================================================================
# 1. Gate tables with Wilson CI
# ====================================================================

def annotate_gate_csv(in_path, out_path):
    """Add Wilson 95% CI to gate CSV."""
    rows = _load_csv(in_path)
    out_rows = []
    for r in rows:
        n = int(r["total_games"])
        w = int(r["wins"])
        lo, hi = wilson_ci(w, n)
        r["ci_lo"] = f"{lo:.3f}"
        r["ci_hi"] = f"{hi:.3f}"
        r["ci_label"] = f"{float(r['conversion_rate']):.0%} [{lo:.0%}, {hi:.0%}]"
        out_rows.append(r)

    fields = list(out_rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"  Annotated: {out_path}")
    return out_rows


# ====================================================================
# 2. Side masking table
# ====================================================================

def build_side_masking_table():
    """Build table showing symmetric avg vs role-separated breakout."""
    sym_path = PAPER_DIR / "M_v4_clean.csv"
    role_path = PAPER_DIR / "M_v4_clean_role.csv"

    # Parse symmetric
    sym_rows = _load_csv(sym_path)
    labels = [k for k in sym_rows[0].keys() if k]
    n = len(labels)
    M_sym = np.zeros((n, n))
    for i, row in enumerate(sym_rows):
        for j, l in enumerate(labels):
            M_sym[i, j] = float(row[l])

    # Parse role-separated (first col is row label)
    role_rows = _load_csv(role_path)
    all_keys = list(role_rows[0].keys())
    row_label_key = all_keys[0]  # "Row=Chess / Col=Xiangqi"
    role_col_labels = all_keys[1:]
    M_role = np.zeros((n, n))
    for i, row in enumerate(role_rows):
        for j, l in enumerate(role_col_labels):
            M_role[i, j] = float(row[l])

    out_rows = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sym_val = M_sym[i, j]
            chess_val = M_role[i, j]  # i plays Chess vs j plays Xiangqi
            xiangqi_val = 1.0 - M_role[j, i]  # i plays Xiangqi vs j plays Chess
            gap = abs(chess_val - xiangqi_val)
            out_rows.append({
                "agent_i": labels[i],
                "agent_j": labels[j],
                "symmetric_avg": f"{sym_val:.3f}",
                "i_as_chess_score": f"{chess_val:.3f}",
                "i_as_xiangqi_score": f"{xiangqi_val:.3f}",
                "masking_gap": f"{gap:.3f}",
            })

    out_path = PAPER_DIR / "table_side_masking.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"  Side masking table: {out_path}")
    return out_rows


# ====================================================================
# 3. Figures
# ====================================================================

def fig1_gate_comparison():
    """Figure 1: Gate conversion rate by agent family with Wilson CI."""
    # Collect all gate data
    agents = []

    # Baselines
    for r in _load_csv(PAPER_DIR / "gate_baselines_v1.csv"):
        agents.append({
            "label": r["agent"].upper().replace("PURE_MCTS_", "MCTS_"),
            "conv": float(r["conversion_rate"]),
            "n": int(r["total_games"]),
            "w": int(r["wins"]),
            "family": "baseline",
        })

    # AB (from existing Sprint 4 data — all 0%)
    for depth in ["D4", "D6", "D8", "D10"]:
        agents.append({
            "label": f"AB_{depth}",
            "conv": 0.0, "n": 40, "w": 0,
            "family": "ab",
        })

    # AZ (representative)
    az_rows = _load_csv(PAPER_DIR / "az_gate_v4.csv")
    for r in az_rows:
        label = r["label"].replace("ckpt_", "")
        if label in ("iter0", "iter2", "iter9", "iter19"):
            display = {"iter0": "AZ-0", "iter2": "AZ-Early",
                        "iter9": "AZ-Mid", "iter19": "AZ-Late"}[label]
            agents.append({
                "label": display,
                "conv": float(r["conversion_rate"]),
                "n": int(r["total_games"]),
                "w": int(r["wins"]),
                "family": "az",
            })

    # Sort
    order = ["RANDOM", "GREEDY", "MCTS_100", "MCTS_400",
             "AB_D4", "AB_D6", "AB_D8", "AB_D10",
             "AZ-0", "AZ-Early", "AZ-Mid", "AZ-Late"]
    agents.sort(key=lambda a: order.index(a["label"]) if a["label"] in order else 99)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"baseline": "#5B9BD5", "ab": "#FF6B6B", "az": "#4CAF50"}
    x = range(len(agents))
    bars = ax.bar(x, [a["conv"] for a in agents],
                  color=[colors[a["family"]] for a in agents],
                  edgecolor="white", linewidth=0.5)

    # CI error bars
    for i, a in enumerate(agents):
        lo, hi = wilson_ci(a["w"], a["n"])
        ax.plot([i, i], [lo, hi], color="black", linewidth=1.5)
        ax.plot([i-0.15, i+0.15], [lo, lo], color="black", linewidth=1)
        ax.plot([i-0.15, i+0.15], [hi, hi], color="black", linewidth=1)

    # Gate threshold line
    ax.axhline(y=0.80, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Gate threshold (80%)")

    ax.set_xticks(list(x))
    ax.set_xticklabels([a["label"] for a in agents], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Conversion Rate", fontsize=11)
    ax.set_title("Figure 1: Endgame Gate — Conversion Rate by Agent Family", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = FIG_DIR / "fig1_gate_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  Figure 1: {out}")


def fig2_ungated_vs_gated():
    """Figure 2: Side-by-side ungated vs gated heatmaps."""
    # Ungated
    ungated_rows = _load_csv(PAPER_DIR / "M_v4_ungated.csv")
    u_labels = [k for k in ungated_rows[0].keys() if k]
    n_u = len(u_labels)
    M_u = np.zeros((n_u, n_u))
    for i, row in enumerate(ungated_rows):
        for j, l in enumerate(u_labels):
            M_u[i, j] = float(row[l])

    # Gated
    gated_rows = _load_csv(PAPER_DIR / "M_v4_clean.csv")
    g_labels = [k for k in gated_rows[0].keys() if k]
    n_g = len(g_labels)
    M_g = np.zeros((n_g, n_g))
    for i, row in enumerate(gated_rows):
        for j, l in enumerate(g_labels):
            M_g[i, j] = float(row[l])

    # Rename gated labels
    rename = {"ckpt_iter2": "AZ-Early", "ckpt_iter9": "AZ-Mid", "ckpt_iter19": "AZ-Late"}
    g_labels_display = [rename.get(l, l) for l in g_labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    cmap = plt.cm.RdYlGn

    # Ungated
    im1 = ax1.imshow(M_u, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax1.set_xticks(range(n_u))
    ax1.set_yticks(range(n_u))
    ax1.set_xticklabels(u_labels, rotation=45, ha="right", fontsize=8)
    ax1.set_yticklabels(u_labels, fontsize=8)
    for i in range(n_u):
        for j in range(n_u):
            ax1.text(j, i, f"{M_u[i,j]:.2f}", ha="center", va="center", fontsize=6,
                     color="white" if M_u[i,j] < 0.3 or M_u[i,j] > 0.7 else "black")
    ax1.set_title(f"Ungated (9 agents)\nDraw rate: ~70%", fontsize=11, fontweight="bold")

    # Gated
    im2 = ax2.imshow(M_g, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(n_g))
    ax2.set_yticks(range(n_g))
    ax2.set_xticklabels(g_labels_display, rotation=45, ha="right", fontsize=10)
    ax2.set_yticklabels(g_labels_display, fontsize=10)
    for i in range(n_g):
        for j in range(n_g):
            ax2.text(j, i, f"{M_g[i,j]:.2f}", ha="center", va="center", fontsize=12,
                     color="white" if M_g[i,j] < 0.3 or M_g[i,j] > 0.7 else "black")
    ax2.set_title(f"Gated (3 agents)\nDraw rate: 33%", fontsize=11, fontweight="bold")

    fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, label="Row agent payoff")
    fig.suptitle("Figure 2: Ungated vs Gated Payoff Matrix", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = FIG_DIR / "fig2_ungated_vs_gated.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 2: {out}")


def fig3_role_separated():
    """Figure 3: Role-separated heatmap (main result)."""
    role_rows = _load_csv(PAPER_DIR / "M_v4_clean_role.csv")
    header_key = [k for k in role_rows[0].keys()][0]  # first column header
    col_labels_raw = [k for k in role_rows[0].keys() if k != header_key]
    n = len(col_labels_raw)

    M = np.zeros((n, n))
    row_labels = []
    for i, row in enumerate(role_rows):
        row_labels.append(row[header_key])
        for j, cl in enumerate(col_labels_raw):
            M[i, j] = float(row[cl])

    # Rename
    rename_row = {"ckpt_iter2@Chess": "AZ-Early\n(Chess)", "ckpt_iter9@Chess": "AZ-Mid\n(Chess)",
                  "ckpt_iter19@Chess": "AZ-Late\n(Chess)"}
    rename_col = {"ckpt_iter2@Xiangqi": "AZ-Early\n(Xiangqi)", "ckpt_iter9@Xiangqi": "AZ-Mid\n(Xiangqi)",
                  "ckpt_iter19@Xiangqi": "AZ-Late\n(Xiangqi)"}
    row_display = [rename_row.get(l, l) for l in row_labels]
    col_display = [rename_col.get(l, l) for l in col_labels_raw]

    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = plt.cm.RdYlGn
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(col_display, fontsize=10)
    ax.set_yticklabels(row_display, fontsize=10)

    for i in range(n):
        for j in range(n):
            val = M[i, j]
            color = "white" if val < 0.3 or val > 0.7 else "black"
            fontweight = "bold" if val == 1.0 else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=16, color=color, fontweight=fontweight)

    # Annotate the upper-triangular pattern
    ax.annotate("100% win\n(skill gap)", xy=(0, 1), xytext=(-0.8, 1.7),
                fontsize=9, color="darkgreen", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkgreen"))

    ax.set_xlabel("Xiangqi Side →", fontsize=12, fontweight="bold")
    ax.set_ylabel("← Chess Side", fontsize=12, fontweight="bold")
    ax.set_title("Figure 3: Role-Separated Payoff Matrix\n(Chess advantage activates only with skill gap)",
                 fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Chess-side payoff")
    fig.tight_layout()

    out = FIG_DIR / "fig3_role_separated.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 3: {out}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Annotate gate CSVs with CI
    print("\n=== Gate CI Annotation ===")
    if (PAPER_DIR / "az_gate_v4.csv").exists():
        annotate_gate_csv(PAPER_DIR / "az_gate_v4.csv",
                          PAPER_DIR / "az_gate_v4_ci.csv")
    if (PAPER_DIR / "gate_baselines_v1.csv").exists():
        annotate_gate_csv(PAPER_DIR / "gate_baselines_v1.csv",
                          PAPER_DIR / "gate_baselines_v1_ci.csv")

    # 2. Side masking table
    print("\n=== Side Masking Table ===")
    build_side_masking_table()

    # 3. Figures
    print("\n=== Figures ===")
    fig1_gate_comparison()
    fig2_ungated_vs_gated()
    fig3_role_separated()

    print("\n  All done.\n")


if __name__ == "__main__":
    main()
