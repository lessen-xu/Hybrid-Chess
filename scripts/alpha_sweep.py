# -*- coding: utf-8 -*-
"""Phase 4 Action 1: α-Rank sweep + directed response graphs.

Scans α ∈ {10, 50, 100, 500, 1000} on all 4 pipeline matrices,
generates phase transition plot and response graph diagrams.
"""
import sys, io, os, csv, json
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.analyze_topology import compute_alpha_rank

OUTDIR = os.path.join("paper", "figures")
DATADIR = os.path.join("paper", "data")
os.makedirs(OUTDIR, exist_ok=True)

# ── Labels ───────────────────────────────────────────────────
FULL_LABELS = [
    "RANDOM", "AB_D2",
    "AZ-S0-E", "AZ-S0-M", "AZ-S0-L",
    "AZ-S100-E", "AZ-S100-M", "AZ-S100-L",
    "AZ-S200-E", "AZ-S200-M", "AZ-S200-L",
]
GATED_INDICES = [2, 3, 4, 5, 7, 8, 9, 10]
GATED_LABELS = [FULL_LABELS[i] for i in GATED_INDICES]

# ── Load matrices ────────────────────────────────────────────
def load_csv(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append([float(x) for x in row[1:]])
    return np.array(rows)

SRC = "runs/ablation_ungated"
M_sym_full = load_csv(os.path.join(SRC, "payoff_matrix.csv"))
M_role_full = load_csv(os.path.join(SRC, "payoff_matrix_role_separated.csv"))

pipelines = {
    "A": ("Ungated+Sym", M_sym_full, FULL_LABELS),
    "B": ("Ungated+Role", M_role_full, FULL_LABELS),
    "C": ("Gated+Sym", M_sym_full[np.ix_(GATED_INDICES, GATED_INDICES)], GATED_LABELS),
    "D": ("Gated+Role", M_role_full[np.ix_(GATED_INDICES, GATED_INDICES)], GATED_LABELS),
}

# ═══════════════════════════════════════════════════════════════
# Part 1: α sweep
# ═══════════════════════════════════════════════════════════════
ALPHAS = [1, 5, 10, 25, 50, 100, 250, 500, 1000]

print("=" * 60)
print("  α-Rank Sweep")
print("=" * 60)

sweep_results = {}
for key, (title, M, labels) in pipelines.items():
    sweep_results[key] = {}
    for alpha in ALPHAS:
        pi = compute_alpha_rank(M, alpha=alpha)
        ranked = sorted(zip(labels, pi), key=lambda x: -x[1])
        top_name, top_prob = ranked[0]
        support = [n for n, p in ranked if p > 0.05]
        sweep_results[key][alpha] = {
            "dist": {n: float(p) for n, p in zip(labels, pi)},
            "top": top_name,
            "top_prob": float(top_prob),
            "support": support,
            "entropy": float(-np.sum(pi * np.log(pi + 1e-12))),
        }
    
    print(f"\n  Pipeline {key} ({title}):")
    for alpha in ALPHAS:
        r = sweep_results[key][alpha]
        print(f"    α={alpha:>5d}: top={r['top']:>12s} ({r['top_prob']:.3f}) "
              f"support={len(r['support'])} entropy={r['entropy']:.3f}")

# Save sweep results
with open(os.path.join(DATADIR, "alpha_sweep_results.json"), "w", encoding="utf-8") as f:
    json.dump(sweep_results, f, indent=2)

# ── Phase transition plot ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Top-agent probability vs α
ax = axes[0]
colors = {"A": "#dc3545", "B": "#fd7e14", "C": "#198754", "D": "#0d6efd"}
for key in ["A", "D", "C", "B"]:
    title = pipelines[key][0]
    top_probs = [sweep_results[key][a]["top_prob"] for a in ALPHAS]
    ax.plot(ALPHAS, top_probs, 'o-', label=f"{key}: {title}", 
            color=colors[key], linewidth=2, markersize=5)
ax.axhline(y=1/11, color='gray', linestyle='--', alpha=0.5, label='Uniform (1/11)')
ax.axhline(y=1/8, color='gray', linestyle=':', alpha=0.5, label='Uniform (1/8)')
ax.set_xlabel("α (Selection Pressure)", fontsize=11)
ax.set_ylabel("Top Agent Probability Mass", fontsize=11)
ax.set_title("Phase Transition: α-Rank Concentration", fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Right: Entropy vs α
ax = axes[1]
for key in ["A", "D", "C", "B"]:
    title = pipelines[key][0]
    entropies = [sweep_results[key][a]["entropy"] for a in ALPHAS]
    ax.plot(ALPHAS, entropies, 'o-', label=f"{key}: {title}",
            color=colors[key], linewidth=2, markersize=5)
ax.set_xlabel("α (Selection Pressure)", fontsize=11)
ax.set_ylabel("Distribution Entropy", fontsize=11)
ax.set_title("Entropy Collapse Under Selection Pressure", fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "fig_alpha_sweep.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: {os.path.join(OUTDIR, 'fig_alpha_sweep.png')}")

# ═══════════════════════════════════════════════════════════════
# Part 2: Directed Response Graphs
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Directed Response Graphs")
print("=" * 60)

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("  WARNING: networkx not installed, using matplotlib-only fallback")

EDGE_THRESHOLD = 0.55

def draw_response_graph(ax, M, labels, title, threshold=EDGE_THRESHOLD):
    """Draw directed response graph on given axes."""
    n = len(labels)
    
    if HAS_NX:
        G = nx.DiGraph()
        G.add_nodes_from(labels)
        
        for i in range(n):
            for j in range(n):
                if i != j and M[i, j] > threshold:
                    G.add_edge(labels[i], labels[j], weight=M[i, j])
        
        # Color nodes
        node_colors = []
        for name in labels:
            if name in ("RANDOM", "AB_D2"):
                node_colors.append("#dc3545")  # red for incompetent
            else:
                node_colors.append("#198754")  # green for AZ
        
        # Layout
        pos = nx.spring_layout(G, seed=42, k=2.0, iterations=100)
        
        # Draw edges with varying width
        edge_widths = [G[u][v]['weight'] * 3 for u, v in G.edges()]
        edge_colors = []
        for u, v in G.edges():
            if u in ("RANDOM", "AB_D2") or v in ("RANDOM", "AB_D2"):
                edge_colors.append("#dc354580")  # semi-transparent red
            else:
                edge_colors.append("#19875480")  # semi-transparent green
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=600, edgecolors='black', linewidths=1.5)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_weight='bold')
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                               width=edge_widths, arrows=True, arrowsize=12,
                               connectionstyle='arc3,rad=0.1', alpha=0.7)
        
        ax.set_title(f"{title}\n({len(G.edges())} edges, threshold={threshold})",
                     fontsize=10, fontweight='bold')
        
        # Count bidirectional edges (draw-like)
        bidir = sum(1 for i in range(n) for j in range(i+1, n) 
                    if M[i,j] > threshold and M[j,i] > threshold)
        
        print(f"  {title}: {len(G.edges())} edges, {bidir} bidirectional")
    else:
        # Fallback: adjacency matrix visualization
        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and M[i, j] > threshold:
                    adj[i, j] = M[i, j]
        ax.imshow(adj, cmap='Greens', vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(title, fontsize=10, fontweight='bold')

# Generate response graph comparison (A vs D)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

draw_response_graph(axes[0], pipelines["A"][1], pipelines["A"][2],
                    "A: Ungated + Symmetric")
draw_response_graph(axes[1], pipelines["D"][1], pipelines["D"][2],
                    "D: Gated + Role-Separated")

fig.suptitle("Directed Response Graphs: Draw Sink vs Clean Topology",
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUTDIR, "fig_response_graphs.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {os.path.join(OUTDIR, 'fig_response_graphs.png')}")

# Also generate 4-panel response graph
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
for ax, (key, (title, M, labels)) in zip(axes.flat, pipelines.items()):
    draw_response_graph(ax, M, labels, f"{key}: {title}")

fig.suptitle("Directed Response Graphs — 2×2 Ablation",
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUTDIR, "fig_response_graphs_4panel.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {os.path.join(OUTDIR, 'fig_response_graphs_4panel.png')}")

# ═══════════════════════════════════════════════════════════════
# Part 3: Best α-Rank comparison at optimal α
# ═══════════════════════════════════════════════════════════════
# Find α where A and D differ most (by KL divergence or entropy diff)
best_alpha = 100  # default
best_diff = 0
for alpha in ALPHAS:
    ent_A = sweep_results["A"][alpha]["entropy"]
    ent_D = sweep_results["D"][alpha]["entropy"]
    diff = abs(ent_A - ent_D)
    if diff > best_diff:
        best_diff = diff
        best_alpha = alpha

print(f"\n  Best α for A vs D differentiation: α={best_alpha} (entropy diff={best_diff:.3f})")

# Generate α-Rank bar charts at best α
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, (key, (title, M, labels)) in zip(axes.flat, pipelines.items()):
    pi = compute_alpha_rank(M, alpha=best_alpha)
    ranked = sorted(zip(labels, pi), key=lambda x: -x[1])
    names = [n for n, _ in ranked]
    probs = [p for _, p in ranked]
    colors_bar = ['#dc3545' if n in ('RANDOM', 'AB_D2') else '#198754' for n in names]
    ax.barh(range(len(names)), probs, color=colors_bar, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel(f'α-Rank probability (α={best_alpha})', fontsize=8)
    ax.set_title(f"{key}: {title}", fontsize=9, fontweight='bold')
    ax.set_xlim(0, max(probs) * 1.15 if max(probs) > 0.2 else 1.05)
    ax.invert_yaxis()
fig.suptitle(f'α-Rank Stationary Distribution (α={best_alpha})',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUTDIR, "fig_alpha_rank_optimal.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {os.path.join(OUTDIR, 'fig_alpha_rank_optimal.png')}")

print("\n  All topology rescue analysis complete!")
