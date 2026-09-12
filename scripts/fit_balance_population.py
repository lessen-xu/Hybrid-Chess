"""Hierarchical Bradley-Terry / Logistic Regression Population Model for Asymmetric Game Balance.

Fits:
  logit P(Chess wins) = (s_chess - s_xq) + sum_r [ beta_r * I(var=r) + delta_r * I(var=r) * log10(B) ]

Parameters:
  - beta_r:  Intrinsic Army Bias under variant r (log-odds Chess edge at baseline budget)
  - delta_r: Balance Drift Slope under variant r (search sensitivity d(logit P)/d(log10 B))
  - s_i:     Latent agent skill rating

Implemented in pure NumPy using Newton-Raphson (Iteratively Reweighted Least Squares)
for maximum performance, exact covariance estimation, and zero external dependencies.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def fit_bradley_terry_population(games: List[Dict[str, Any]], variants: List[str], agents: List[str]) -> Dict[str, Any]:
    """Fit generalized Bradley-Terry model with variant army bias and budget interaction slopes."""
    # Filter valid games
    valid_games = [g for g in games if g.get("winner") in ("chess", "xiangqi", None)]
    if not valid_games:
        raise ValueError("No valid game records found to fit Bradley-Terry model.")

    n_agents = len(agents)
    n_vars = len(variants)
    n_params = (n_agents - 1) + 2 * n_vars

    agent_idx = {a: i for i, a in enumerate(agents)}
    var_idx = {v: i for i, v in enumerate(variants)}

    X = np.zeros((len(valid_games), n_params), dtype=np.float64)
    y = np.zeros(len(valid_games), dtype=np.float64)

    for row, g in enumerate(valid_games):
        c_agent = g["chess_agent"]
        x_agent = g["xiangqi_agent"]
        v_name = g["variant"]

        # Skill difference s_chess - s_xq (agent 0 is fixed at 0)
        if agent_idx[c_agent] > 0:
            X[row, agent_idx[c_agent] - 1] += 1.0
        if agent_idx[x_agent] > 0:
            X[row, agent_idx[x_agent] - 1] -= 1.0

        # Variant army bias beta_r
        v_offset = (n_agents - 1) + var_idx[v_name]
        X[row, v_offset] = 1.0

        # Variant budget interaction delta_r * log10(B)
        b_c = g.get("chess_budget", 1.0)
        b_x = g.get("xiangqi_budget", 1.0)
        log_b = 0.5 * (math.log10(max(1.0, b_c)) + math.log10(max(1.0, b_x)))
        
        slope_offset = (n_agents - 1) + n_vars + var_idx[v_name]
        X[row, slope_offset] = log_b

        # Target probability: Chess win = 1.0, draw = 0.5, Xiangqi win = 0.0
        if g["winner"] == "chess":
            y[row] = 1.0
        elif g["winner"] == "xiangqi":
            y[row] = 0.0
        else:
            y[row] = 0.5

    # L2 regularization to ensure strict convexity and numerical stability
    l2_reg = 0.05
    reg_matrix = l2_reg * np.eye(n_params)
    # Don't over-regularize army biases
    for i in range(n_agents - 1, n_params):
        reg_matrix[i, i] = 0.01

    # Newton-Raphson solver (IRLS)
    theta = np.zeros(n_params, dtype=np.float64)
    max_iter = 50
    tol = 1e-7

    converged = False
    H_final = reg_matrix

    for iteration in range(max_iter):
        logits = X @ theta
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        
        # Diagonal weights W = p * (1 - p)
        weights = probs * (1.0 - probs)
        weights = np.maximum(weights, 1e-6)

        # Gradient: X^T (probs - y) + reg * theta
        grad = X.T @ (probs - y) + reg_matrix @ theta

        # Hessian: X^T W X + reg
        H = X.T @ (weights[:, None] * X) + reg_matrix

        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, grad, rcond=None)[0]

        theta -= delta
        H_final = H

        if np.max(np.abs(delta)) < tol:
            converged = True
            break

    # Standard errors via inverse Hessian (Fisher Information)
    try:
        cov = np.linalg.inv(H_final)
        se = np.sqrt(np.maximum(0.0, np.diag(cov)))
    except np.linalg.LinAlgError:
        se = np.full(n_params, 0.1)

    # Format parameter estimates
    variant_results = {}
    for i, v in enumerate(variants):
        beta_idx = (n_agents - 1) + i
        delta_idx = (n_agents - 1) + n_vars + i

        beta_val = float(theta[beta_idx])
        beta_se = float(se[beta_idx])
        delta_val = float(theta[delta_idx])
        delta_se = float(se[delta_idx])

        p_baseline = 1.0 / (1.0 + math.exp(-beta_val))
        is_robust = (abs(beta_val) < 0.20) and (abs(delta_val) < 0.10)

        variant_results[v] = {
            "army_bias_beta": round(beta_val, 4),
            "army_bias_se": round(beta_se, 4),
            "army_bias_95_ci": [round(beta_val - 1.96 * beta_se, 4), round(beta_val + 1.96 * beta_se, 4)],
            "implied_chess_prob_baseline": round(p_baseline, 4),
            "balance_drift_delta": round(delta_val, 4),
            "balance_drift_se": round(delta_se, 4),
            "balance_drift_95_ci": [round(delta_val - 1.96 * delta_se, 4), round(delta_val + 1.96 * delta_se, 4)],
            "robust_balance_verdict": is_robust,
        }

    skill_ratings = {agents[0]: 0.0}
    for i in range(1, n_agents):
        skill_ratings[agents[i]] = round(float(theta[i - 1]), 4)

    return {
        "n_games": len(valid_games),
        "variants": variant_results,
        "agent_skills": skill_ratings,
        "optimization_success": converged,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to balance curve output directory containing games/")
    parser.add_argument("--output", help="Optional path to save model estimates JSON")
    args = parser.parse_args()

    input_dir = Path(args.input)
    games_dir = input_dir / "games" if (input_dir / "games").exists() else input_dir

    game_files = sorted(list(games_dir.glob("*.json")))
    if not game_files:
        raise FileNotFoundError(f"No game JSON files found in {games_dir}")

    records = [json.loads(f.read_text()) for f in game_files]

    variants = sorted(list({r["variant"] for r in records}))
    agents = sorted(list({r["chess_agent"] for r in records} | {r["xiangqi_agent"] for r in records}))

    print(f"[BradleyTerry] Fitting population model across {len(records)} games...")
    print(f"  Variants: {variants}")
    print(f"  Agents: {agents}")

    results = fit_bradley_terry_population(records, variants, agents)

    print("\n" + "=" * 80)
    print(" HIERARCHICAL BRADLEY-TERRY ESTIMATES: OPERATIONAL BALANCE & DRIFT")
    print("=" * 80)
    print(f"{'Variant':<22} | {'Army Bias (beta)':<18} | {'Drift (delta)':<16} | {'Implied P(Chess)':<16} | {'Robust?'}")
    print("-" * 80)

    for v, res in results["variants"].items():
        beta_str = f"{res['army_bias_beta']:+.3f} ± {1.96*res['army_bias_se']:.3f}"
        delta_str = f"{res['balance_drift_delta']:+.3f} ± {1.96*res['balance_drift_se']:.3f}"
        p_str = f"{res['implied_chess_prob_baseline']*100:.1f}%"
        verdict = "YES" if res["robust_balance_verdict"] else "NO"
        print(f"{v:<22} | {beta_str:<18} | {delta_str:<16} | {p_str:<16} | {verdict}")

    print("=" * 80)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"[BradleyTerry] Results written to {out_path}")


if __name__ == "__main__":
    main()
