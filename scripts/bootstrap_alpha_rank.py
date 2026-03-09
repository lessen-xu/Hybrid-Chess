# -*- coding: utf-8 -*-
"""Phase 5: Bootstrap confidence intervals for α-Rank + publication-quality figures.

Parses raw game logs, resamples, computes α-Rank CIs, and regenerates
all figures with diverging colormap and improved styling.
"""
import sys, io, os, csv, json, re, glob
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.analyze_topology import compute_alpha_rank

np.random.seed(42)

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
N_FULL = len(FULL_LABELS)
GATED_INDICES = [2, 3, 4, 5, 7, 8, 9, 10]
GATED_LABELS = [FULL_LABELS[i] for i in GATED_INDICES]

# ── Parse raw game logs ──────────────────────────────────────
def parse_pair_log(path):
    """Parse a pair log file -> list of per-game outcomes (1=win_i, 0.5=draw, 0=loss_i)."""
    outcomes = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'WIN_i' in line:
                outcomes.append(1.0)
            elif 'WIN_j' in line:
                outcomes.append(0.0)
            elif 'DRAW' in line and '/50]' in line:
                outcomes.append(0.5)
    return outcomes

SRC = "runs/ablation_ungated/live"
# Build raw outcomes matrix: raw_games[i][j] = list of per-game scores for i vs j
raw_games = [[[] for _ in range(N_FULL)] for _ in range(N_FULL)]

for logfile in glob.glob(os.path.join(SRC, "pair_*.log")):
    fname = os.path.basename(logfile)
    match = re.match(r'pair_(\d+)_(\d+)\.log', fname)
    if match:
        i, j = int(match.group(1)), int(match.group(2))
        outcomes = parse_pair_log(logfile)
        raw_games[i][j] = outcomes
        raw_games[j][i] = [1.0 - x for x in outcomes]  # mirror

# Fill diagonal with 0.5
for i in range(N_FULL):
    raw_games[i][i] = [0.5] * 50

# Build base matrices
def build_matrix(games, n):
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if games[i][j]:
                M[i, j] = np.mean(games[i][j])
            else:
                M[i, j] = 0.5
    return M

M_base = build_matrix(raw_games, N_FULL)

# Also load role-separated matrix
def load_csv(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append([float(x) for x in row[1:]])
    return np.array(rows)

M_role = load_csv(os.path.join("runs/ablation_ungated", "payoff_matrix_role_separated.csv"))

print(f"Loaded {sum(1 for i in range(N_FULL) for j in range(N_FULL) if raw_games[i][j])} pair records")
print(f"Base matrix shape: {M_base.shape}")

pipelines = {
    "A": ("Ungated+Sym", M_base, FULL_LABELS, N_FULL),
    "B": ("Ungated+Role", M_role, FULL_LABELS, N_FULL),
    "C": ("Gated+Sym", M_base[np.ix_(GATED_INDICES, GATED_INDICES)], GATED_LABELS, len(GATED_INDICES)),
    "D": ("Gated+Role", M_role[np.ix_(GATED_INDICES, GATED_INDICES)], GATED_LABELS, len(GATED_INDICES)),
}

# ═══════════════════════════════════════════════════════════════
# Bootstrap α-Rank
# ═══════════════════════════════════════════════════════════════
N_BOOT = 1000
ALPHA_BOOT = 100

print(f"\nBootstrap α-Rank (α={ALPHA_BOOT}, {N_BOOT} resamples)...")

def bootstrap_matrix(games, n, indices=None):
    """Resample each pair's games and build a new matrix."""
    M = np.zeros((n, n))
    for ii in range(n):
        for jj in range(n):
            i = indices[ii] if indices else ii
            j = indices[jj] if indices else jj
            g = games[i][j]
            if g and len(g) > 0:
                boot = np.random.choice(g, size=len(g), replace=True)
                M[ii, jj] = np.mean(boot)
            else:
                M[ii, jj] = 0.5
    return M

boot_results = {}
for key in ["A", "C", "D"]:
    title, M_orig, labels, n = pipelines[key]
    indices = None if key == "A" else GATED_INDICES
    
    boot_dists = np.zeros((N_BOOT, n))
    for b in range(N_BOOT):
        M_boot = bootstrap_matrix(raw_games, n, indices)
        if key in ("B", "D"):
            # For role-separated, we can't easily bootstrap without raw role data
            # Use binomial resampling from aggregated rates instead
            M_boot = np.zeros((n, n))
            M_orig_local = pipelines[key][1]
            for ii in range(n):
                for jj in range(n):
                    if ii == jj:
                        M_boot[ii, jj] = 0.5
                    else:
                        # Binomial resample: N=50, p=original rate
                        wins = np.random.binomial(50, M_orig_local[ii, jj])
                        M_boot[ii, jj] = wins / 50.0
        pi = compute_alpha_rank(M_boot, alpha=ALPHA_BOOT)
        boot_dists[b] = pi
    
    means = np.mean(boot_dists, axis=0)
    ci_lo = np.percentile(boot_dists, 2.5, axis=0)
    ci_hi = np.percentile(boot_dists, 97.5, axis=0)
    
    boot_results[key] = {
        "labels": labels,
        "means": means,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "all_dists": boot_dists,
    }
    
    print(f"\n  Pipeline {key} ({title}):")
    ranked = sorted(zip(labels, means, ci_lo, ci_hi), key=lambda x: -x[1])
    for name, m, lo, hi in ranked:
        print(f"    {name:>12s}: {m:.4f} [{lo:.4f}, {hi:.4f}]")

# Save bootstrap results
boot_json = {}
for key, data in boot_results.items():
    boot_json[key] = {
        name: {"mean": float(m), "ci_lo": float(lo), "ci_hi": float(hi)}
        for name, m, lo, hi in zip(data["labels"], data["means"], data["ci_lo"], data["ci_hi"])
    }
with open(os.path.join(DATADIR, "bootstrap_alpha_rank.json"), "w", encoding="utf-8") as f:
    json.dump(boot_json, f, indent=2)

# ═══════════════════════════════════════════════════════════════
# Bootstrap α sweep (CI shading on phase transition plot)
# ═══════════════════════════════════════════════════════════════
ALPHAS = [1, 5, 10, 25, 50, 100, 250, 500, 1000]

print(f"\nBootstrap α sweep ({N_BOOT} resamples per α)...")

boot_sweep = {}
for key in ["A", "D"]:
    title, M_orig, labels, n = pipelines[key]
    indices = None if key == "A" else GATED_INDICES
    
    # Pre-generate bootstrap matrices
    boot_matrices = []
    for b in range(N_BOOT):
        M_b = bootstrap_matrix(raw_games, n, indices)
        boot_matrices.append(M_b)
    
    boot_sweep[key] = {}
    for alpha in ALPHAS:
        top_probs = []
        for M_b in boot_matrices:
            pi = compute_alpha_rank(M_b, alpha=alpha)
            top_probs.append(float(np.max(pi)))
        boot_sweep[key][alpha] = {
            "mean": float(np.mean(top_probs)),
            "ci_lo": float(np.percentile(top_probs, 2.5)),
            "ci_hi": float(np.percentile(top_probs, 97.5)),
        }
    print(f"  Pipeline {key} done")

# ═══════════════════════════════════════════════════════════════
# Regenerate figures: publication quality
# ═══════════════════════════════════════════════════════════════
print("\nGenerating publication-quality figures...")

# ── Figure 1: α sweep with CI shading ────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
colors = {"A": "#dc3545", "D": "#0d6efd"}
for key, color in colors.items():
    title = pipelines[key][0]
    means = [boot_sweep[key][a]["mean"] for a in ALPHAS]
    ci_lo = [boot_sweep[key][a]["ci_lo"] for a in ALPHAS]
    ci_hi = [boot_sweep[key][a]["ci_hi"] for a in ALPHAS]
    ax.plot(ALPHAS, means, 'o-', label=f"{key}: {title}", color=color, linewidth=2.5, markersize=6)
    ax.fill_between(ALPHAS, ci_lo, ci_hi, color=color, alpha=0.15)

n_full = len(FULL_LABELS)
n_gated = len(GATED_LABELS)
ax.axhline(y=1/n_full, color='gray', linestyle='--', alpha=0.5, label=f'Uniform (1/{n_full})')
ax.axhline(y=1/n_gated, color='gray', linestyle=':', alpha=0.5, label=f'Uniform (1/{n_gated})')
ax.set_xlabel(r"$\alpha$ (Selection Pressure)", fontsize=12)
ax.set_ylabel("Top Agent Probability Mass", fontsize=12)
ax.set_title(r"Phase Transition: $\alpha$-Rank Concentration (95% Bootstrap CI)", fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTDIR, "fig_alpha_sweep.png"), dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: fig_alpha_sweep.png")

# ── Figure 2: α-Rank bar chart with error bars ───────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, key in zip(axes, ["A", "C", "D"]):
    data = boot_results[key]
    title = pipelines[key][0]
    ranked = sorted(zip(data["labels"], data["means"], data["ci_lo"], data["ci_hi"]), key=lambda x: -x[1])
    names = [n for n, _, _, _ in ranked]
    means = [m for _, m, _, _ in ranked]
    errors_lo = [m - lo for _, m, lo, _ in ranked]
    errors_hi = [hi - m for _, _, _, hi in ranked]
    bar_colors = ['#dc3545' if n in ('RANDOM', 'AB_D2') else '#198754' for n in names]
    
    bars = ax.barh(range(len(names)), means, xerr=[errors_lo, errors_hi],
                   color=bar_colors, edgecolor='white', linewidth=0.5,
                   capsize=3, error_kw={'linewidth': 1.2, 'color': '#333'})
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(f'α-Rank probability (α={ALPHA_BOOT})', fontsize=9)
    ax.set_title(f'{key}: {title}', fontsize=10, fontweight='bold')
    ax.set_xlim(0, max(means) * 1.3 if max(means) > 0.2 else 0.3)
    ax.invert_yaxis()

fig.suptitle(f'α-Rank with 95% Bootstrap CI (α={ALPHA_BOOT}, {N_BOOT} resamples)',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUTDIR, "fig_alpha_rank_bootstrap.png"), dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: fig_alpha_rank_bootstrap.png")

# ── Figure 3: 4-panel heatmap with RdBu diverging colormap ───
def plot_heatmap_rdbu(ax, M, labels, title):
    im = ax.imshow(M, cmap='RdBu_r', vmin=0, vmax=1, aspect='equal')
    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    for i in range(n):
        for j in range(n):
            v = M[i,j]
            color = 'white' if (v < 0.2 or v > 0.8) else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5,
                    color=color, fontweight='bold')
    ax.set_title(title, fontsize=9, fontweight='bold', pad=6)
    return im

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
for ax, (key, (title, M, labels, n)) in zip(axes.flat, pipelines.items()):
    im = plot_heatmap_rdbu(ax, M, labels, f"{key}: {title}")
cbar = fig.colorbar(im, ax=axes, shrink=0.6, label='Win Rate (0.5 = white)')
fig.suptitle('2×2 Ablation: Ungated/Gated × Symmetric/Role-Separated',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 0.92, 0.96])
fig.savefig(os.path.join(OUTDIR, "fig_ablation_4panel.png"), dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: fig_ablation_4panel.png")

# ── Figure 4: Draw Wall spotlight with RdBu ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
# Left: Ungated symmetric
ax = axes[0]
M_A = pipelines["A"][1]
L_A = pipelines["A"][2]
im = ax.imshow(M_A, cmap='RdBu_r', vmin=0, vmax=1, aspect='equal')
n = len(L_A)
for i in range(n):
    for j in range(n):
        v = M_A[i,j]
        color = 'white' if (v < 0.2 or v > 0.8) else 'black'
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5.5,
                color=color, fontweight='bold')
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(L_A, rotation=45, ha='right', fontsize=6.5)
ax.set_yticklabels(L_A, fontsize=6.5)
rect = plt.Rectangle((-0.5, 0.5), n, 1, linewidth=3, edgecolor='#ff6600', facecolor='none', linestyle='--')
ax.add_patch(rect)
# Highlight AZ-S0-L column (index 4) as the False King
rect2 = plt.Rectangle((3.5, -0.5), 1, n, linewidth=3, edgecolor='gold', facecolor='none', linestyle='--')
ax.add_patch(rect2)
ax.set_title('A: Ungated + Symmetric\n(orange = AB_D2 draw wall, gold = False King AZ-S0-L)',
             fontsize=9, fontweight='bold')

# Right: Gated + Role-Separated
ax = axes[1]
M_D = pipelines["D"][1]
L_D = pipelines["D"][2]
im = ax.imshow(M_D, cmap='RdBu_r', vmin=0, vmax=1, aspect='equal')
ng = len(L_D)
for i in range(ng):
    for j in range(ng):
        v = M_D[i,j]
        color = 'white' if (v < 0.2 or v > 0.8) else 'black'
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6,
                color=color, fontweight='bold')
ax.set_xticks(range(ng)); ax.set_yticks(range(ng))
ax.set_xticklabels(L_D, rotation=45, ha='right', fontsize=7)
ax.set_yticklabels(L_D, fontsize=7)
ax.set_title('D: Gated + Role-Separated\n(clean red/blue structure, no false hierarchy)',
             fontsize=9, fontweight='bold')

fig.suptitle('Draw Wall Effect: Before vs After Gating (Diverging Colormap)',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUTDIR, "fig_ablation_draw_wall.png"), dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: fig_ablation_draw_wall.png")

# ── Figure 5: Improved response graph ────────────────────────
import networkx as nx

EDGE_THRESHOLD = 0.55

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, (key, kw) in zip(axes, [("A", {"title": "A: Ungated+Sym\n(Spurious Hierarchy)"}),
                                  ("D", {"title": "D: Gated+Role\n(Clean Non-Transitive)"})]):
    title_str = kw["title"]
    M = pipelines[key][1]
    labels = pipelines[key][2]
    n = pipelines[key][3]
    
    G = nx.DiGraph()
    G.add_nodes_from(labels)
    
    for i in range(n):
        for j in range(n):
            if i != j and M[i, j] > EDGE_THRESHOLD:
                G.add_edge(labels[i], labels[j], weight=M[i, j])
    
    # Node colors 
    node_colors = ['#dc3545' if name in ("RANDOM", "AB_D2") else '#198754' for name in labels]
    node_sizes = [900 if name == "AZ-S0-L" else 600 for name in labels]
    
    pos = nx.kamada_kawai_layout(G)
    
    # Classify edges: spurious (involving incompetent) vs genuine
    spurious_edges = [(u, v) for u, v in G.edges() if u in ("RANDOM", "AB_D2") or v in ("RANDOM", "AB_D2")]
    genuine_edges = [(u, v) for u, v in G.edges() if u not in ("RANDOM", "AB_D2") and v not in ("RANDOM", "AB_D2")]
    
    # Draw genuine edges
    if genuine_edges:
        genuine_widths = [G[u][v]['weight'] * 3 for u, v in genuine_edges]
        nx.draw_networkx_edges(G, pos, edgelist=genuine_edges, ax=ax,
                               edge_color='#198754', width=genuine_widths,
                               arrows=True, arrowsize=12, connectionstyle='arc3,rad=0.1', alpha=0.7)
    
    # Draw spurious edges (red dashed)
    if spurious_edges:
        spurious_widths = [G[u][v]['weight'] * 2 for u, v in spurious_edges]
        nx.draw_networkx_edges(G, pos, edgelist=spurious_edges, ax=ax,
                               edge_color='#dc3545', width=spurious_widths,
                               arrows=True, arrowsize=10, connectionstyle='arc3,rad=0.15',
                               alpha=0.5, style='dashed')
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=6, font_weight='bold')
    
    n_spurious = len(spurious_edges)
    n_genuine = len(genuine_edges)
    ax.set_title(f"{title_str}\n({n_genuine} genuine + {n_spurious} spurious edges)",
                 fontsize=10, fontweight='bold')

fig.suptitle("Directed Response Graphs: Spurious Hierarchy vs Clean Topology",
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUTDIR, "fig_response_graphs.png"), dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved: fig_response_graphs.png")

print("\n  All Phase 5 figures complete!")
