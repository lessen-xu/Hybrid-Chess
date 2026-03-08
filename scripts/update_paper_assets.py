# -*- coding: utf-8 -*-
"""Generate updated paper assets: figures, CSVs, gate summary for 8-agent multi-seed results."""
import sys, io, os, csv, json
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PAPER = r"paper"
os.makedirs(PAPER, exist_ok=True)

LABELS = ["S0_E", "S0_M", "S0_L", "S100_E", "S100_L", "S200_E", "S200_M", "S200_L"]
FULL_LABELS = [
    "AZ-S0-Early", "AZ-S0-Mid", "AZ-S0-Late",
    "AZ-S100-Early", "AZ-S100-Late",
    "AZ-S200-Early", "AZ-S200-Mid", "AZ-S200-Late",
]

# ── Load raw matrices ────────────────────────────────────────
def load_csv(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append([float(x) for x in row[1:]])
    return np.array(rows)

sym = load_csv(r'runs/multi_seed_egta/tournament/payoff_matrix.csv')
role = load_csv(r'runs/multi_seed_egta/tournament/payoff_matrix_role_separated.csv')
N = len(LABELS)

# ── Save relabeled CSVs ─────────────────────────────────────
def save_matrix_csv(matrix, labels, path, header_prefix=""):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([header_prefix] + labels)
        for i, label in enumerate(labels):
            w.writerow([label] + [f"{matrix[i,j]:.4f}" for j in range(len(labels))])
    print(f"  Saved: {path}")

save_matrix_csv(sym, FULL_LABELS, os.path.join(PAPER, "M_v4_8agent_symmetric.csv"))
save_matrix_csv(role, 
    [f"{l}@Chess" for l in FULL_LABELS],
    os.path.join(PAPER, "M_v4_8agent_role_separated.csv"),
    header_prefix="Chess \\ Xiangqi")

# ── Gate summary CSV ─────────────────────────────────────────
gate_data = [
    ("AZ-S0-Early",   "seed0",  "iter2",  "88%",  "PASS"),
    ("AZ-S0-Mid",     "seed0",  "iter9",  "88%",  "PASS"),
    ("AZ-S0-Late",    "seed0",  "iter19", "100%", "PASS"),
    ("AZ-S100-Early", "seed100","iter2",  "100%", "PASS"),
    ("AZ-S100-Mid",   "seed100","iter9",  "75%",  "FAIL"),
    ("AZ-S100-Late",  "seed100","iter19", "100%", "PASS"),
    ("AZ-S200-Early", "seed200","iter2",  "100%", "PASS"),
    ("AZ-S200-Mid",   "seed200","iter9",  "100%", "PASS"),
    ("AZ-S200-Late",  "seed200","iter19", "100%", "PASS"),
]
gate_path = os.path.join(PAPER, "gate_multiseed_v1.csv")
with open(gate_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Agent", "Seed", "Iteration", "Conversion", "Gate"])
    w.writerows(gate_data)
print(f"  Saved: {gate_path}")

# ── Side masking table ───────────────────────────────────────
masking_path = os.path.join(PAPER, "table_side_masking_8agent.csv")
with open(masking_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Pair", "Symmetric", "Chess_Score", "Xiangqi_Score", "Masking_Gap"])
    for i in range(N):
        for j in range(i+1, N):
            chess_ij = role[i,j]
            chess_ji = role[j,i]
            xiangqi_ij = 1.0 - chess_ji
            gap = abs(chess_ij - xiangqi_ij)
            if gap > 0.01:
                w.writerow([
                    f"{FULL_LABELS[i]} vs {FULL_LABELS[j]}",
                    f"{sym[i,j]:.3f}",
                    f"{chess_ij:.3f}",
                    f"{xiangqi_ij:.3f}",
                    f"{gap:.3f}",
                ])
print(f"  Saved: {masking_path}")

# ── Heatmaps with correct labels ────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_heatmap(matrix, labels, save_path, title, cmap='RdYlGn'):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect='equal')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = matrix[i,j]
            color = 'white' if v < 0.3 or v > 0.7 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8, color=color, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

plot_heatmap(sym, FULL_LABELS,
    os.path.join(PAPER, "fig_8agent_symmetric.png"),
    "8-Agent Symmetric Payoff Matrix (V4, 3 Seeds × Early/Mid/Late)")

plot_heatmap(role, FULL_LABELS,
    os.path.join(PAPER, "fig_8agent_role_separated.png"),
    "8-Agent Role-Separated Matrix (Chess=Row, Xiangqi=Col)")

# ── Gate comparison figure ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
all_agents = [
    ("RANDOM", 40, "baseline"), ("GREEDY", 38, "baseline"),
    ("MCTS-100", 8, "baseline"), ("MCTS-400", 72, "baseline"),
    ("AB-D4", 0, "ab"), ("AB-D6", 0, "ab"), ("AB-D8", 0, "ab"), ("AB-D10", 0, "ab"),
    ("AZ-S0-E", 88, "az"), ("AZ-S0-M", 88, "az"), ("AZ-S0-L", 100, "az"),
    ("AZ-S100-E", 100, "az"), ("AZ-S100-M", 75, "az_fail"), ("AZ-S100-L", 100, "az"),
    ("AZ-S200-E", 100, "az"), ("AZ-S200-M", 100, "az"), ("AZ-S200-L", 100, "az"),
]
colors = {"baseline": "#6c757d", "ab": "#dc3545", "az": "#198754", "az_fail": "#fd7e14"}
x = range(len(all_agents))
bars = ax.bar(x, [a[1] for a in all_agents],
    color=[colors[a[2]] for a in all_agents], edgecolor='white', linewidth=0.5)
ax.axhline(y=80, color='#333', linestyle='--', linewidth=1.5, label='Gate threshold (80%)')
ax.set_xticks(x)
ax.set_xticklabels([a[0] for a in all_agents], rotation=55, ha='right', fontsize=7.5)
ax.set_ylabel('Conversion Rate (%)', fontsize=11)
ax.set_title('Endgame Gate Results: All Agent Families', fontsize=13, fontweight='bold')
ax.set_ylim(0, 110)
ax.legend(fontsize=9)
plt.tight_layout()
fig_gate = os.path.join(PAPER, "fig_gate_all_agents.png")
plt.savefig(fig_gate, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_gate}")

# ── Paper pool JSON ──────────────────────────────────────────
pool = {
    "description": "8-agent gate-pass pool from 3 independent AZ training seeds",
    "gate_threshold": 0.80,
    "agents": [
        {"label": label, "seed": seed, "iteration": it, "conversion": conv}
        for label, seed, it, conv, gate in gate_data if gate == "PASS"
    ],
    "filtered_out": [
        {"label": "AZ-S100-Mid", "seed": "seed100", "iteration": "iter9",
         "conversion": "75%", "reason": "below 80% gate threshold"}
    ],
    "total_candidates": 9,
    "pass_count": 8,
    "fail_count": 1,
}
pool_path = os.path.join(PAPER, "paper_pool_v3.json")
with open(pool_path, "w", encoding="utf-8") as f:
    json.dump(pool, f, indent=2)
print(f"  Saved: {pool_path}")

print("\n  All paper assets updated!")
