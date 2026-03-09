# -*- coding: utf-8 -*-
"""2×2 Ablation Analysis: Ungated/Gated × Symmetric/Role-Separated.

Reads the 11-agent ungated tournament output, extracts 4 pipeline matrices,
runs α-Rank + cycle detection on each, and produces comparison figures + CSVs.

Pipelines:
  A = Ungated + Symmetric      (all 11 agents, averaged payoff)
  B = Ungated + Role-Separated  (all 11 agents, Chess-role matrix)
  C = Gated + Symmetric         (8 AZ agents subset, averaged)
  D = Gated + Role-Separated    (8 AZ agents subset, Chess-role)
"""
import sys, io, os, csv, json
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── Agent labels ─────────────────────────────────────────────
# Raw labels from tournament (checkpoints collide → need relabeling)
RAW_LABELS = [
    "RANDOM", "AB_D2",
    "ckpt_iter2", "ckpt_iter9", "ckpt_iter19",   # seed 0
    "ckpt_iter2", "ckpt_iter9", "ckpt_iter19",   # seed 100
    "ckpt_iter2", "ckpt_iter9", "ckpt_iter19",   # seed 200
]

FULL_LABELS = [
    "RANDOM", "AB_D2",
    "AZ-S0-E", "AZ-S0-M", "AZ-S0-L",
    "AZ-S100-E", "AZ-S100-M", "AZ-S100-L",
    "AZ-S200-E", "AZ-S200-M", "AZ-S200-L",
]

# Gate results: indices of agents that PASS the gate (≥80%)
# S100-M (index 6) fails at 75%
GATED_INDICES = [2, 3, 4, 5, 7, 8, 9, 10]  # all AZ except S100-M
GATED_LABELS = [FULL_LABELS[i] for i in GATED_INDICES]

OUTDIR = os.path.join("paper", "data")
FIGDIR = os.path.join("paper", "figures")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

# ── Load matrices ────────────────────────────────────────────
def load_csv(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            rows.append([float(x) for x in row[1:]])
    return np.array(rows)

SRC = "runs/ablation_ungated"
M_sym_full = load_csv(os.path.join(SRC, "payoff_matrix.csv"))
M_role_full = load_csv(os.path.join(SRC, "payoff_matrix_role_separated.csv"))

N = len(FULL_LABELS)
assert M_sym_full.shape == (N, N), f"Expected ({N},{N}), got {M_sym_full.shape}"

# ── Build 4 pipeline matrices ────────────────────────────────
# A: Ungated + Symmetric
M_A = M_sym_full.copy()
L_A = FULL_LABELS[:]

# B: Ungated + Role-Separated
M_B = M_role_full.copy()
L_B = FULL_LABELS[:]

# C: Gated + Symmetric (extract submatrix)
M_C = M_sym_full[np.ix_(GATED_INDICES, GATED_INDICES)]
L_C = GATED_LABELS[:]

# D: Gated + Role-Separated (extract submatrix)
M_D = M_role_full[np.ix_(GATED_INDICES, GATED_INDICES)]
L_D = GATED_LABELS[:]

pipelines = {
    "A_ungated_sym": (M_A, L_A, "A: Ungated + Symmetric"),
    "B_ungated_role": (M_B, L_B, "B: Ungated + Role-Separated"),
    "C_gated_sym": (M_C, L_C, "C: Gated + Symmetric"),
    "D_gated_role": (M_D, L_D, "D: Gated + Role-Separated"),
}

# ── α-Rank ───────────────────────────────────────────────────
from scripts.analyze_topology import compute_alpha_rank, find_dominance_cycles

results = {}
for key, (M, labels, title) in pipelines.items():
    pi = compute_alpha_rank(M, alpha=10.0)
    edges, cycles = find_dominance_cycles(M, labels, threshold=0.60)
    
    ranked = sorted(zip(labels, pi), key=lambda x: -x[1])
    support = [name for name, p in ranked if p > 0.01]
    top_agent = ranked[0][0]
    top_prob = float(ranked[0][1])
    avg_scores = [float(M[i].mean()) for i in range(len(labels))]
    
    # Count intransitive triples
    n_agents = len(labels)
    triples = 0
    for i in range(n_agents):
        for j in range(i+1, n_agents):
            for k in range(j+1, n_agents):
                a, b, c = M[i,j], M[j,k], M[i,k]
                if (a > 0.5 and b > 0.5 and c < 0.5) or \
                   (a < 0.5 and b < 0.5 and c > 0.5):
                    triples += 1
    
    results[key] = {
        "title": title,
        "n_agents": n_agents,
        "alpha_rank": {name: float(p) for name, p in zip(labels, pi)},
        "support": support,
        "support_size": len(support),
        "top_agent": top_agent,
        "top_prob": top_prob,
        "n_edges": len(edges),
        "n_cycles": len(cycles),
        "cycles": cycles[:10],  # top 10
        "intransitive_triples": triples,
        "avg_scores": {name: round(s, 4) for name, s in zip(labels, avg_scores)},
    }
    
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Agents: {n_agents}")
    print(f"  α-Rank support (>1%): {support}")
    print(f"  Top: {top_agent} ({top_prob:.1%})")
    print(f"  Dominance edges: {len(edges)}, Cycles: {len(cycles)}")
    print(f"  Intransitive triples: {triples}")
    for name, prob in ranked:
        bar = "#" * int(prob * 40)
        print(f"    {name:>12s}  {prob:.4f}  {bar}")

# ── Key comparisons ──────────────────────────────────────────
print(f"\n{'='*60}")
print("  KEY ABLATION FINDINGS")
print(f"{'='*60}")

# Phenomenon 1: Draw Wall (A vs D)
print("\n  [1] DRAW WALL (Pipeline A vs D):")
ab_row_A = M_A[1, :]  # AB_D2 is index 1
ab_avg = float(np.mean(ab_row_A))
ab_near_05 = int(np.sum(np.abs(ab_row_A - 0.5) < 0.05))
print(f"    AB_D2 row in Ungated (A): mean={ab_avg:.3f}, entries near 0.5: {ab_near_05}/{N}")
print(f"    AB_D2 α-Rank mass (A): {results['A_ungated_sym']['alpha_rank'].get('AB_D2', 0):.4f}")
print(f"    After gating (D): AB_D2 removed → draw wall eliminated")
print(f"    α-Rank support A: {results['A_ungated_sym']['support_size']} agents")
print(f"    α-Rank support D: {results['D_gated_role']['support_size']} agents")

# Phenomenon 2: Topology Collapse (C vs D)
print("\n  [2] TOPOLOGY COLLAPSE (Pipeline C vs D):")
print(f"    Gated+Symmetric (C): {results['C_gated_sym']['intransitive_triples']} intransitive triples")
print(f"    Gated+Role-Sep  (D): {results['D_gated_role']['intransitive_triples']} intransitive triples")
print(f"    Cycles in C: {results['C_gated_sym']['n_cycles']}")
print(f"    Cycles in D: {results['D_gated_role']['n_cycles']}")

# Masking gap analysis on gated subset
n_gated = len(GATED_INDICES)
gaps = []
extreme_pairs = []
for i in range(n_gated):
    for j in range(i+1, n_gated):
        chess_score = M_D[i, j]
        sym_score = M_C[i, j]
        gap = abs(chess_score - (1.0 - M_D[j, i]))
        if gap > 0.01:
            gaps.append(gap)
            if gap >= 0.99:
                extreme_pairs.append((GATED_LABELS[i], GATED_LABELS[j], chess_score, 1.0-M_D[j,i]))

print(f"\n    Masking gap (C vs D): {len(gaps)} pairs with gap > 0.01")
if gaps:
    print(f"    Mean gap: {np.mean(gaps):.3f}")
    print(f"    Max gap: {np.max(gaps):.3f}")
    print(f"    Extreme pairs (gap ≥ 1.0): {len(extreme_pairs)}")
    for a, b, cs, xs in extreme_pairs[:5]:
        print(f"      {a} vs {b}: Chess={cs:.1f}, Xiangqi={xs:.1f}")

# ── Save results JSON ────────────────────────────────────────
results_path = os.path.join(OUTDIR, "ablation_2x2_results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n  Results saved: {results_path}")

# ── Save relabeled CSVs ──────────────────────────────────────
def save_matrix(M, labels, path, header=""):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([header] + labels)
        for i, label in enumerate(labels):
            w.writerow([label] + [f"{M[i,j]:.4f}" for j in range(len(labels))])
    print(f"  Saved: {path}")

save_matrix(M_A, L_A, os.path.join(OUTDIR, "ablation_A_ungated_sym.csv"))
save_matrix(M_B, L_B, os.path.join(OUTDIR, "ablation_B_ungated_role.csv"), "Chess\\Xiangqi")
save_matrix(M_C, L_C, os.path.join(OUTDIR, "ablation_C_gated_sym.csv"))
save_matrix(M_D, L_D, os.path.join(OUTDIR, "ablation_D_gated_role.csv"), "Chess\\Xiangqi")

# ── Generate figures ─────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_heatmap_ax(ax, M, labels, title, cmap='RdYlGn'):
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect='equal')
    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    for i in range(n):
        for j in range(n):
            v = M[i,j]
            color = 'white' if v < 0.3 or v > 0.7 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5, 
                    color=color, fontweight='bold')
    ax.set_title(title, fontsize=9, fontweight='bold', pad=6)
    return im

# Figure 1: 4-panel heatmap comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
for ax, (key, (M, labels, title)) in zip(axes.flat, pipelines.items()):
    im = plot_heatmap_ax(ax, M, labels, title)
fig.colorbar(im, ax=axes, shrink=0.6, label='Win Rate')
fig.suptitle('2×2 Ablation: Ungated/Gated × Symmetric/Role-Separated', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = os.path.join(FIGDIR, "fig_ablation_4panel.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path}")

# Figure 2: α-Rank bar chart comparison (4 panels)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, (key, (M, labels, title)) in zip(axes.flat, pipelines.items()):
    ar = results[key]["alpha_rank"]
    names = list(ar.keys())
    probs = list(ar.values())
    colors = ['#dc3545' if n in ('RANDOM', 'AB_D2') else '#198754' for n in names]
    bars = ax.barh(range(len(names)), probs, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('α-Rank probability', fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
fig.suptitle('α-Rank Stationary Distribution — 2×2 Ablation', 
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path2 = os.path.join(FIGDIR, "fig_ablation_alpha_rank.png")
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path2}")

# Figure 3: Draw wall spotlight — AB_D2 row vs gated absence
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Left: Ungated symmetric — highlight AB_D2 row
ax = axes[0]
im = ax.imshow(M_A, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
n = len(L_A)
for i in range(n):
    for j in range(n):
        v = M_A[i,j]
        color = 'white' if v < 0.3 or v > 0.7 else 'black'
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5.5, 
                color=color, fontweight='bold')
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(L_A, rotation=45, ha='right', fontsize=6.5)
ax.set_yticklabels(L_A, fontsize=6.5)
# Highlight AB_D2 row
rect = plt.Rectangle((-0.5, 0.5), n, 1, linewidth=3, edgecolor='red', facecolor='none')
ax.add_patch(rect)
ax.set_title('A: Ungated + Symmetric\n(red box = AB_D2 "draw wall")', fontsize=10, fontweight='bold')

# Right: Gated + Role-Separated
ax = axes[1]
im = ax.imshow(M_D, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
ng = len(L_D)
for i in range(ng):
    for j in range(ng):
        v = M_D[i,j]
        color = 'white' if v < 0.3 or v > 0.7 else 'black'
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6, 
                color=color, fontweight='bold')
ax.set_xticks(range(ng)); ax.set_yticks(range(ng))
ax.set_xticklabels(L_D, rotation=45, ha='right', fontsize=7)
ax.set_yticklabels(L_D, fontsize=7)
ax.set_title('D: Gated + Role-Separated\n(draw wall removed, structure revealed)', fontsize=10, fontweight='bold')

fig.suptitle('Draw Wall Effect: Before vs After Gating', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
fig_path3 = os.path.join(FIGDIR, "fig_ablation_draw_wall.png")
plt.savefig(fig_path3, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fig_path3}")

print("\n  All ablation analysis complete!")
