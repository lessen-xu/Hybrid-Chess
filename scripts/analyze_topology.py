# -*- coding: utf-8 -*-
"""Topology analysis for EGTA payoff matrices.

Computes:
  1. Alpha-Rank stationary distribution (Omidshafiei et al. 2019)
  2. Dominance graph + cycle detection (Wenner et al. 2023)

Usage:
    python -m scripts.analyze_topology --payoff runs/egta_pilot/payoff_matrix.csv
"""

from __future__ import annotations
import argparse
import csv
import json
import sys

import numpy as np
import scipy.linalg


def load_payoff_csv(path: str):
    """Load a payoff_matrix.csv → (agent_names, matrix)."""
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        agents = header[1:]
        rows = []
        for row in reader:
            if row[0]:
                rows.append([float(x) for x in row[1:]])
    return agents, np.array(rows)


def compute_alpha_rank(
    payoff: np.ndarray, alpha: float = 10.0
) -> np.ndarray:
    """
    Single-population Alpha-Rank stationary distribution.

    payoff[i, j] = win rate of agent i against agent j.
    alpha = selection pressure (inverse Fermi temperature).
    Returns: stationary distribution pi over agents.
    """
    K = payoff.shape[0]
    C = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            if i != j:
                payoff_diff = payoff[j, i] - payoff[i, j]
                C[i, j] = (1.0 / (K - 1)) * (
                    1.0 / (1.0 + np.exp(-alpha * payoff_diff))
                )

    for i in range(K):
        C[i, i] = 1.0 - np.sum(C[i, :])

    eigenvalues, eigenvectors = scipy.linalg.eig(C, left=True, right=False)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = np.maximum(pi, 0)
    total = np.sum(pi)
    if total < 1e-12:
        return np.ones(K) / K  # fallback to uniform
    return pi / total


def find_dominance_cycles(
    payoff: np.ndarray,
    agents: list[str],
    threshold: float = 0.60,
) -> tuple[list[tuple[str, str, float]], list[list[str]]]:
    """
    Build directed dominance graph and find cycles.

    Edge (i → j) exists if payoff[i, j] >= threshold.
    Returns: (edges, cycles).
    """
    edges = []
    K = len(agents)
    for i in range(K):
        for j in range(K):
            if i != j and payoff[i, j] >= threshold:
                edges.append((agents[i], agents[j], float(payoff[i, j])))

    # Find cycles via DFS (no networkx dependency)
    adj: dict[str, list[str]] = {a: [] for a in agents}
    for src, dst, _ in edges:
        adj[src].append(dst)

    cycles: list[list[str]] = []

    def _dfs(node: str, path: list[str], visited: set[str]):
        for nb in adj[node]:
            if nb == path[0]:
                cycles.append(path + [nb])
            elif nb not in visited:
                visited.add(nb)
                _dfs(nb, path + [nb], visited)
                visited.discard(nb)

    for start in agents:
        _dfs(start, [start], {start})

    # Deduplicate cycles (canonical form = rotation with min element first)
    unique = set()
    deduped = []
    for cyc in cycles:
        ring = cyc[:-1]  # remove repeated start
        min_idx = ring.index(min(ring))
        canon = tuple(ring[min_idx:] + ring[:min_idx])
        if canon not in unique:
            unique.add(canon)
            deduped.append(list(canon))

    return edges, deduped


def main():
    parser = argparse.ArgumentParser(description="EGTA Topology Analysis")
    parser.add_argument("--payoff", required=True, help="Path to payoff_matrix.csv")
    parser.add_argument("--alpha", type=float, default=10.0, help="Alpha-Rank selection pressure")
    parser.add_argument("--threshold", type=float, default=0.60, help="Dominance threshold for cycle detection")
    parser.add_argument("--json", default=None, help="Optional: save results to JSON")
    args = parser.parse_args()

    agents, M = load_payoff_csv(args.payoff)
    K = len(agents)

    # Alpha-Rank
    pi = compute_alpha_rank(M, alpha=args.alpha)
    print(f"Alpha-Rank (alpha={args.alpha}):")
    ranked = sorted(zip(agents, pi), key=lambda x: -x[1])
    for name, prob in ranked:
        bar = "#" * int(prob * 50)
        print(f"  {name:>18s}  {prob:.4f}  {bar}")

    top_agent = ranked[0][0]
    top_prob = ranked[0][1]
    support_ar = [name for name, p in ranked if p > 0.01]
    print(f"\n  Dominant: {top_agent} ({top_prob:.1%})")
    print(f"  Support (>1%): {support_ar}")

    # Dominance cycles
    edges, cycles = find_dominance_cycles(M, agents, threshold=args.threshold)
    print(f"\nDominance Graph (threshold={args.threshold}):")
    print(f"  Edges: {len(edges)}")
    print(f"  Cycles: {len(cycles)}")
    for cyc in cycles:
        print(f"    {'→'.join(cyc)}→{cyc[0]}")

    # Summary
    print(f"\n{'='*60}")
    if len(cycles) == 0:
        print(f"  TOPOLOGY: TRANSITIVE (0 cycles, alpha-rank support={len(support_ar)})")
    else:
        print(f"  TOPOLOGY: NON-TRANSITIVE ({len(cycles)} cycles, alpha-rank support={len(support_ar)})")
    print(f"{'='*60}")

    # Optional JSON output
    if args.json:
        result = {
            "alpha": args.alpha,
            "threshold": args.threshold,
            "alpha_rank": {name: float(p) for name, p in zip(agents, pi)},
            "alpha_rank_support": support_ar,
            "dominance_edges": len(edges),
            "cycles": cycles,
            "num_cycles": len(cycles),
            "topology": "TRANSITIVE" if len(cycles) == 0 else "NON-TRANSITIVE",
        }
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
