# -*- coding: utf-8 -*-
"""Non-parametric bootstrap from N=200 atomic game records.

Reads game_records.csv (produced by ultimate_tournament.py), performs
true non-parametric bootstrap (resampling actual game records, not
parametric Binomial), and regenerates all α-Rank figures with CIs.

Usage:
    python -u -m scripts.bootstrap_nonparametric
"""
import sys, io, os, csv, json
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.analyze_topology import compute_alpha_rank

np.random.seed(42)

OUTDIR = "paper/figures"
DATADIR = "paper/data"
CSV_PATH = "runs/ultimate_n200/game_records.csv"
N_BOOT = 1000
ALPHA_BOOT = 100


def load_game_records(csv_path):
    """Load atomic game records and build per-pair record lists.
    
    Returns:
        labels: sorted list of unique agent names
        records: dict[(chess_agent, xiangqi_agent)] -> list of winner values
    """
    records = {}
    agents = set()
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ca = row["chess_agent"]
            xa = row["xiangqi_agent"]
            winner = int(row["winner"])
            agents.add(ca)
            agents.add(xa)
            
            key = (ca, xa)
            if key not in records:
                records[key] = []
            records[key].append(winner)
    
    labels = sorted(agents)
    return labels, records


def build_role_separated_matrix(labels, records):
    """Build role-separated matrix: M[i,j] = win rate of i as Chess vs j as Xiangqi."""
    n = len(labels)
    M = np.full((n, n), 0.5)
    idx = {name: i for i, name in enumerate(labels)}
    
    for (ca, xa), winners in records.items():
        i, j = idx[ca], idx[xa]
        # winner=1 means chess wins, winner=-1 means xiangqi wins, 0=draw
        scores = [(1.0 if w == 1 else (0.0 if w == -1 else 0.5)) for w in winners]
        M[i, j] = np.mean(scores)
    
    return M


def bootstrap_role_matrix(labels, records, rng):
    """Non-parametric bootstrap: resample actual game records per pair."""
    n = len(labels)
    M = np.full((n, n), 0.5)
    idx = {name: i for i, name in enumerate(labels)}
    
    for (ca, xa), winners in records.items():
        i, j = idx[ca], idx[xa]
        boot_winners = rng.choice(winners, size=len(winners), replace=True)
        scores = [(1.0 if w == 1 else (0.0 if w == -1 else 0.5)) for w in boot_winners]
        M[i, j] = np.mean(scores)
    
    return M


def main():
    if not os.path.exists(CSV_PATH):
        print(f"  ERROR: {CSV_PATH} not found. Run ultimate_tournament.py first.")
        return
    
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(DATADIR, exist_ok=True)
    
    print("Loading game records...")
    labels, records = load_game_records(CSV_PATH)
    n = len(labels)
    
    total_games = sum(len(v) for v in records.values())
    print(f"  {n} agents, {len(records)} directed pairs, {total_games} total games")
    print(f"  Agents: {labels}")
    
    # Build base matrix
    M_base = build_role_separated_matrix(labels, records)
    print(f"\nBase role-separated matrix:")
    for i, name in enumerate(labels):
        row = " ".join(f"{M_base[i,j]:.3f}" for j in range(n))
        print(f"  {name:>12s}: {row}")
    
    # ── Bootstrap α-Rank ──────────────────────────────────────
    rng = np.random.RandomState(42)
    ALPHAS = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
    
    print(f"\nNon-parametric bootstrap (N={N_BOOT})...")
    
    # Bootstrap at fixed α
    boot_dists = np.zeros((N_BOOT, n))
    for b in range(N_BOOT):
        M_b = bootstrap_role_matrix(labels, records, rng)
        pi = compute_alpha_rank(M_b, alpha=ALPHA_BOOT)
        boot_dists[b] = pi
    
    means = np.mean(boot_dists, axis=0)
    ci_lo = np.percentile(boot_dists, 2.5, axis=0)
    ci_hi = np.percentile(boot_dists, 97.5, axis=0)
    
    print(f"\n  α-Rank (α={ALPHA_BOOT}) with 95% CI:")
    ranked = sorted(zip(labels, means, ci_lo, ci_hi), key=lambda x: -x[1])
    for name, m, lo, hi in ranked:
        print(f"    {name:>12s}: {m:.4f} [{lo:.4f}, {hi:.4f}]")
    
    # Bootstrap α sweep
    print(f"\n  α sweep with bootstrap CIs...")
    sweep_data = {}
    for alpha in ALPHAS:
        top_probs = []
        for b in range(N_BOOT):
            M_b = bootstrap_role_matrix(labels, records, rng)
            pi = compute_alpha_rank(M_b, alpha=alpha)
            top_probs.append(float(np.max(pi)))
        sweep_data[alpha] = {
            "mean": float(np.mean(top_probs)),
            "ci_lo": float(np.percentile(top_probs, 2.5)),
            "ci_hi": float(np.percentile(top_probs, 97.5)),
        }
        print(f"    α={alpha:>5d}: max_prob={sweep_data[alpha]['mean']:.4f} "
              f"[{sweep_data[alpha]['ci_lo']:.4f}, {sweep_data[alpha]['ci_hi']:.4f}]")
    
    # ── Save results ──────────────────────────────────────────
    results = {
        "n_agents": n,
        "n_games_total": total_games,
        "n_games_per_directed_pair": total_games // max(len(records), 1),
        "bootstrap_n": N_BOOT,
        "method": "non-parametric (direct game record resampling)",
        "alpha_rank": {
            name: {"mean": float(m), "ci_lo": float(lo), "ci_hi": float(hi)}
            for name, m, lo, hi in zip(labels, means, ci_lo, ci_hi)
        },
        "alpha_sweep": {str(k): v for k, v in sweep_data.items()},
    }
    with open(os.path.join(DATADIR, "bootstrap_n200_nonparametric.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # ── Regenerate figures ────────────────────────────────────
    print("\nGenerating figures...")
    
    # Bar chart with CIs
    fig, ax = plt.subplots(figsize=(8, 5))
    ranked_data = sorted(zip(labels, means, ci_lo, ci_hi), key=lambda x: -x[1])
    names = [nm for nm, _, _, _ in ranked_data]
    bar_means = [m for _, m, _, _ in ranked_data]
    err_lo = [max(0, m - lo) for _, m, lo, _ in ranked_data]
    err_hi = [max(0, hi - m) for _, _, _, hi in ranked_data]
    
    bars = ax.barh(range(len(names)), bar_means,
                   xerr=[err_lo, err_hi],
                   color='#198754', edgecolor='white',
                   capsize=3, error_kw={'linewidth': 1.2, 'color': '#333'})
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(r'$\alpha$-Rank probability ($\alpha$={})'.format(ALPHA_BOOT), fontsize=10)
    ax.set_title('N=200 Non-Parametric Bootstrap\n(Gated Pool, {} resamples)'.format(N_BOOT),
                 fontsize=12, fontweight='bold')
    ax.axvline(x=1/n, color='gray', linestyle='--', alpha=0.5, label='Uniform (1/{})'.format(n))
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_alpha_rank_n200.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_alpha_rank_n200.png")
    
    # Heatmap with RdBu
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(M_base, cmap='RdBu_r', vmin=0, vmax=1, aspect='equal')
    for i in range(n):
        for j in range(n):
            v = M_base[i, j]
            color = 'white' if (v < 0.2 or v > 0.8) else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Xiangqi Agent", fontsize=10)
    ax.set_ylabel("Chess Agent", fontsize=10)
    fig.colorbar(im, label='Win Rate (Chess agent)')
    ax.set_title('N=200 Gated Role-Separated Payoff Matrix', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_heatmap_n200.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_heatmap_n200.png")
    
    # DRR computation
    decisive = sum(1 for i in range(n) for j in range(n)
                   if i != j and abs(M_base[i,j] - 0.5) > 0.25)
    total_off = n * (n - 1)
    drr = decisive / total_off
    print(f"\n  DRR(N=200) = {drr:.3f} ({drr*100:.1f}%)")
    
    # SIPD - need symmetric matrix to compare
    sipd_vals = []
    for i in range(n):
        for j in range(i+1, n):
            p_chess = M_base[i, j]
            p_reverse = M_base[j, i]
            delta = abs(p_chess - (1.0 - p_reverse))
            sipd_vals.append(delta)
    mean_sipd = np.mean(sipd_vals) if sipd_vals else 0
    print(f"  SIPD(N=200) = {mean_sipd:.3f}")
    
    print("\n  Non-parametric bootstrap analysis complete!")


if __name__ == "__main__":
    main()
