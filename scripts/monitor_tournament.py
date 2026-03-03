# -*- coding: utf-8 -*-
"""Live dashboard for AB tournament with termination reason tracking.

Reads runs/<tag>_progress.json -> generates runs/<tag>_dashboard.png

Usage:
  python -m scripts.monitor_tournament --watch 30
  python -m scripts.monitor_tournament --tag ab_termination_d2 --watch 30
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Palette ──
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
TEXT    = "#e6edf3"
DIM     = "#8b949e"
CHESS_C = "#f85149"
XQ_C    = "#58a6ff"
DRAW_C  = "#d29922"
GREEN   = "#3fb950"

# Termination reason colors
TERM_COLORS = {
    "Threefold repetition":      "#f0883e",   # orange
    "Max plies reached":         "#a371f7",   # purple
    "Stalemate (draw by rule)":  "#768390",   # gray
    "Checkmate":                 "#3fb950",   # green
    "Chess king captured":       "#f85149",   # red
    "Xiangqi general captured":  "#58a6ff",   # blue
}

# Short labels for chart
TERM_SHORT = {
    "Threefold repetition":      "3-fold Rep",
    "Max plies reached":         "Move Limit",
    "Stalemate (draw by rule)":  "Stalemate",
    "Checkmate":                 "Checkmate",
    "Chess king captured":       "King Captured",
    "Xiangqi general captured":  "General Captured",
}

TERM_EMOJI = {
    "Threefold repetition":      "[R]",
    "Max plies reached":         "[T]",
    "Stalemate (draw by rule)":  "[S]",
    "Checkmate":                 "[#]",
    "Chess king captured":       "[K]",
    "Xiangqi general captured":  "[G]",
}


def wilson(wins, n, z=1.96):
    if n == 0: return 0, 0, 0
    p = wins / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    m = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n) / d
    return p, max(0, c-m), min(1, c+m)


def render(data: dict, out: str):
    conditions = data.get("conditions", {})
    cond_keys = list(conditions.keys())
    n_conds = len(cond_keys)
    if n_conds == 0:
        n_conds = 1  # at least 1 row

    depth = data.get("depth", 2)
    rand_plies = data.get("random_plies", 4)
    status = data.get("status", "running")
    gpg = data.get("games_per_condition", 100)

    # Dynamic layout: 4 columns per condition row
    fig = plt.figure(figsize=(22, 5 * n_conds + 2), facecolor=BG)

    # ── Title ──
    status_icon = "[DONE]" if status == "done" else "[...]"
    fig.suptitle(
        f"Termination Analysis -- AB-d{depth} vs AB-d{depth}  "
        f"({rand_plies} random opening plies)",
        color=TEXT, fontsize=18, fontweight="bold", fontfamily="monospace", y=0.97,
    )
    fig.text(0.5, 0.94,
             f"{status_icon} {status.upper()}   |   {data.get('last_update', '--')}",
             ha="center", color=DIM, fontsize=11, fontfamily="monospace")

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(n_conds, 4, figure=fig, left=0.04, right=0.97,
                  top=0.90, bottom=0.06, hspace=0.50, wspace=0.30,
                  width_ratios=[0.8, 1.0, 1.2, 1.0])

    for idx, key in enumerate(cond_keys):
        cond = conditions.get(key, {})
        completed  = cond.get("completed", 0)
        total      = cond.get("total", gpg)
        cw         = cond.get("chess_wins", 0)
        xw         = cond.get("xiangqi_wins", 0)
        dr         = cond.get("draws", 0)
        plies      = cond.get("plies", [])
        term_reasons = cond.get("termination_reasons", {})
        cond_status = cond.get("status", "pending")

        tag = {"pending": "PENDING", "running": ">> RUNNING", "done": "DONE"}.get(cond_status, "")

        # ── Col 0: Win/Draw Donut ──
        ax_donut = fig.add_subplot(gs[idx, 0])
        ax_donut.set_facecolor(PANEL)
        for sp in ax_donut.spines.values():
            sp.set_color(BORDER)
        ax_donut.set_title(f"{key}\n{tag}", color=TEXT if cond_status != "pending" else DIM,
                           fontsize=12, fontweight="bold", fontfamily="monospace", pad=10)

        if completed > 0:
            sizes  = [cw, xw, dr]
            colors = [CHESS_C, XQ_C, DRAW_C]
            labels = [f"Chess {cw}", f"Xiangqi {xw}", f"Draw {dr}"]
            nz = [(s, c, l) for s, c, l in zip(sizes, colors, labels) if s > 0]
            if nz:
                sz, cl, lb = zip(*nz)
                wedges, txt, atxt = ax_donut.pie(
                    sz, colors=cl, labels=lb, autopct=lambda p: f"{p:.0f}%",
                    startangle=90, pctdistance=0.75,
                    wedgeprops={"edgecolor": BG, "linewidth": 2},
                    textprops={"color": TEXT, "fontsize": 9, "fontfamily": "monospace"},
                )
                for a in atxt:
                    a.set_color("white"); a.set_fontweight("bold"); a.set_fontsize(10)
                ax_donut.add_patch(plt.Circle((0, 0), 0.50, fc=PANEL))
                ax_donut.text(0, 0, f"{completed}/{total}", ha="center", va="center",
                             color=TEXT, fontsize=14, fontweight="bold", fontfamily="monospace")
        else:
            ax_donut.text(0.5, 0.5, "Waiting...", transform=ax_donut.transAxes,
                         ha="center", va="center", color=DIM, fontsize=14, fontfamily="monospace")
            ax_donut.set_xlim(-1, 1); ax_donut.set_ylim(-1, 1)
        ax_donut.set_aspect("equal")

        # ── Col 1: Termination Reason Donut (THE KEY NEW CHART) ──
        ax_term = fig.add_subplot(gs[idx, 1])
        ax_term.set_facecolor(PANEL)
        for sp in ax_term.spines.values():
            sp.set_color(BORDER)
        ax_term.set_title("Termination Reasons", color=TEXT,
                          fontsize=11, fontweight="bold", fontfamily="monospace", pad=10)

        if term_reasons and completed > 0:
            # Sort by count descending
            sorted_reasons = sorted(term_reasons.items(), key=lambda x: x[1], reverse=True)
            t_labels = []
            t_sizes = []
            t_colors = []
            for reason, count in sorted_reasons:
                short = TERM_SHORT.get(reason, reason[:15])
                emoji = TERM_EMOJI.get(reason, "❓")
                t_labels.append(f"{emoji} {short} ({count})")
                t_sizes.append(count)
                t_colors.append(TERM_COLORS.get(reason, DIM))

            wedges, txt, atxt = ax_term.pie(
                t_sizes, colors=t_colors, labels=t_labels,
                autopct=lambda p: f"{p:.0f}%",
                startangle=90, pctdistance=0.78,
                wedgeprops={"edgecolor": BG, "linewidth": 2},
                textprops={"color": TEXT, "fontsize": 8, "fontfamily": "monospace"},
            )
            for a in atxt:
                a.set_color("white"); a.set_fontweight("bold"); a.set_fontsize(9)
            ax_term.add_patch(plt.Circle((0, 0), 0.45, fc=PANEL))
            ax_term.text(0, 0, "WHY?", ha="center", va="center",
                        color=DIM, fontsize=12, fontweight="bold", fontfamily="monospace")
        else:
            ax_term.text(0.5, 0.5, "—", transform=ax_term.transAxes,
                        ha="center", va="center", color=DIM, fontsize=20)
            ax_term.set_xlim(-1, 1); ax_term.set_ylim(-1, 1)
        ax_term.set_aspect("equal")

        # ── Col 2: Ply Distribution Histogram ──
        ax_hist = fig.add_subplot(gs[idx, 2])
        ax_hist.set_facecolor(PANEL)
        for sp in ax_hist.spines.values():
            sp.set_color(BORDER)
        ax_hist.tick_params(colors=DIM, labelsize=9)
        ax_hist.set_title("Game Length Distribution", color=TEXT, fontsize=11,
                         fontweight="bold", fontfamily="monospace", pad=8)

        if plies and completed > 0:
            reasons_list = cond.get("reasons", [])
            if reasons_list:
                # Color bars by termination reason
                ply_by_reason = {}
                for p, r in zip(plies, reasons_list):
                    ply_by_reason.setdefault(r, []).append(p)

                all_plies = np.array(plies)
                bins = np.linspace(0, max(all_plies) + 5, min(25, max(all_plies) // 3 + 1))

                bottom = np.zeros(len(bins) - 1)
                for reason in sorted(ply_by_reason.keys()):
                    rp = np.array(ply_by_reason[reason])
                    hist_vals, _ = np.histogram(rp, bins=bins)
                    color = TERM_COLORS.get(reason, DIM)
                    short = TERM_SHORT.get(reason, reason[:12])
                    ax_hist.bar(bins[:-1], hist_vals, width=np.diff(bins) * 0.9,
                               bottom=bottom, color=color, edgecolor=BG,
                               linewidth=0.5, alpha=0.85, label=short, align="edge")
                    bottom += hist_vals
                ax_hist.legend(loc="upper right", fontsize=7, facecolor=PANEL,
                              edgecolor=BORDER, labelcolor=TEXT, framealpha=0.9)
            else:
                ax_hist.hist(plies, bins=20, color=DRAW_C, edgecolor=BG, alpha=0.85)

            avg_ply = sum(plies) / len(plies)
            ax_hist.axvline(x=avg_ply, color=GREEN, linestyle="--", linewidth=1.5, alpha=0.8)
            ax_hist.text(avg_ply + 1, ax_hist.get_ylim()[1] * 0.9,
                        f"avg={avg_ply:.0f}", color=GREEN, fontsize=9,
                        fontfamily="monospace", fontweight="bold")
            ax_hist.set_xlabel("Plies", color=DIM, fontsize=10, fontfamily="monospace")
            ax_hist.set_ylabel("Count", color=DIM, fontsize=10, fontfamily="monospace")
        else:
            ax_hist.text(0.5, 0.5, "—", transform=ax_hist.transAxes,
                        ha="center", va="center", color=DIM, fontsize=20)

        # ── Col 3: Stats Card ──
        ax_stats = fig.add_subplot(gs[idx, 3])
        ax_stats.set_facecolor(PANEL)
        for sp in ax_stats.spines.values():
            sp.set_color(BORDER)
        ax_stats.set_xticks([]); ax_stats.set_yticks([])
        ax_stats.set_xlim(0, 1); ax_stats.set_ylim(0, 1)

        if completed > 0:
            cw_r, cw_lo, cw_hi = wilson(cw, completed)
            xw_r, xw_lo, xw_hi = wilson(xw, completed)
            avg_ply = sum(plies) / len(plies) if plies else 0

            lines = [
                (0.92, f"Chess WR", CHESS_C),
                (0.86, f"  {cw_r:.0%}  [{cw_lo:.0%}–{cw_hi:.0%}]", TEXT),
                (0.76, f"Xiangqi WR", XQ_C),
                (0.70, f"  {xw_r:.0%}  [{xw_lo:.0%}–{xw_hi:.0%}]", TEXT),
                (0.60, f"Draw Rate", DRAW_C),
                (0.54, f"  {dr}/{completed} ({dr/completed:.0%})", TEXT),
            ]

            # Add termination breakdown
            y_pos = 0.42
            if term_reasons:
                lines.append((y_pos, "── Termination ──", DIM))
                y_pos -= 0.07
                for reason, count in sorted(term_reasons.items(), key=lambda x: x[1], reverse=True):
                    short = TERM_SHORT.get(reason, reason[:15])
                    color = TERM_COLORS.get(reason, DIM)
                    pct = count / completed * 100
                    lines.append((y_pos, f"  {short}: {count} ({pct:.0f}%)", color))
                    y_pos -= 0.06

            lines.append((max(y_pos, 0.04), f"Avg Ply: {avg_ply:.1f}  |  {completed}/{total}", DIM))

            for y, txt, col in lines:
                if y >= 0:
                    ax_stats.text(0.05, y, txt, color=col, fontsize=9,
                                 fontfamily="monospace",
                                 fontweight="bold" if col not in (DIM, TEXT) else "normal",
                                 transform=ax_stats.transAxes)
        else:
            ax_stats.text(0.5, 0.5, "—", ha="center", va="center",
                         color=DIM, fontsize=20, transform=ax_stats.transAxes)

    # Legend
    patches = [
        mpatches.Patch(color=CHESS_C, label="Chess wins"),
        mpatches.Patch(color=XQ_C, label="Xiangqi wins"),
        mpatches.Patch(color=DRAW_C, label="Draws"),
        mpatches.Patch(color=TERM_COLORS["Threefold repetition"], label="3-fold Repetition"),
        mpatches.Patch(color=TERM_COLORS["Max plies reached"], label="Move Limit"),
        mpatches.Patch(color=TERM_COLORS["Checkmate"], label="Checkmate"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=6, fontsize=10,
              facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, framealpha=0.9)

    plt.savefig(out, dpi=130, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", type=int, default=0,
                        help="Seconds between refreshes (0 = one-shot)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Custom tag for files (default: ab_tournament_d{depth})")
    parser.add_argument("--depth", type=int, default=2,
                        help="AB depth to monitor (used for default tag)")
    args = parser.parse_args()

    tag = args.tag or f"ab_tournament_d{args.depth}"
    progress_file = Path(f"runs/{tag}_progress.json")
    dashboard_file = Path(f"runs/{tag}_dashboard.png")

    while True:
        if not progress_file.exists():
            print(f"⏳ Waiting for {progress_file} ...")
        else:
            with open(progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            render(data, str(dashboard_file))
            st = data.get("status", "?")
            conds = data.get("conditions", {})
            done = sum(1 for c in conds.values() if c.get("status") == "done")
            total_conds = len(conds)

            # Summary of termination reasons
            all_reasons = {}
            for c in conds.values():
                for reason, count in c.get("termination_reasons", {}).items():
                    all_reasons[reason] = all_reasons.get(reason, 0) + count

            reason_str = " | ".join(
                f"{TERM_SHORT.get(r, r[:10])}:{c}"
                for r, c in sorted(all_reasons.items(), key=lambda x: x[1], reverse=True)
            )
            print(f"[Dashboard] -> {dashboard_file}  "
                  f"[{done}/{total_conds} conditions, {st}]  {reason_str}", flush=True)
            if st == "done" or args.watch <= 0:
                break
        if args.watch <= 0:
            break
        time.sleep(args.watch)
    print("Done.")


if __name__ == "__main__":
    main()
