# -*- coding: utf-8 -*-
"""EGTA Dual-Matrix Ablation — Round-Robin Payoff Matrix & Nash Equilibrium.

Runs two independent 7x7 round-robin tournaments under different rule variants
to contrast transitivity (V4 extra_cannon) vs non-transitivity (V3 no_queen).

Each universe pits N agents (Random, AlphaBeta depths, AZ checkpoints) against
each other in all-pairs side-swapping matches, builds the NxN empirical payoff
matrix, computes mixed-strategy Nash Equilibrium via LP, and generates a heatmap.

Usage:
  # Run both universes (default)
  python -m scripts.egta_tournament --outdir runs/egta

  # Run only V4 universe
  python -m scripts.egta_tournament --preset v4 --outdir runs/egta

  # Run only V3 universe
  python -m scripts.egta_tournament --preset v3 --outdir runs/egta

  # Custom agents (quick smoke test)
  python -m scripts.egta_tournament --preset custom \
      --agents "random,ab_d2,runs/az_grand_run_v4/ckpt_iter19.pt" \
      --ablation extra_cannon --games-per-pair 2 --simulations 50 \
      --outdir runs/egta_smoke
"""

from __future__ import annotations

import scripts._fix_encoding  # noqa: F401
import argparse
import csv
import json
import os
import time
from itertools import combinations
from pathlib import Path
from typing import List

import numpy as np

# ====================================================================
# Reuse agent factory from eval_arena
# ====================================================================
from scripts.eval_arena import _create_agent, _agent_label, play_arena_game

import hybrid.core.config as cfg


# ====================================================================
# Strategy Pools — two parallel universes
# ====================================================================

# ── Frozen 9-Agent Pool (DO NOT MODIFY — paper-critical) ──
V4_AGENTS = [
    "random",
    "greedy",
    "pure_mcts_100",
    "ab_d1",
    "ab_d2",
    "ab_d4",
    "runs/az_grand_run_v4/ckpt_iter2.pt",   # AZ-Early
    "runs/az_grand_run_v4/ckpt_iter9.pt",   # AZ-Mid
    "runs/az_grand_run_v4/ckpt_iter19.pt",  # AZ-Best
]

V3_AGENTS = [
    "random",
    "greedy",
    "pure_mcts_100",
    "ab_d1",
    "ab_d2",
    "ab_d4",
    "runs/az_no_queen_run/ckpt_iter1.pt",   # AZ-Early
    "runs/az_no_queen_run/ckpt_iter6.pt",   # AZ-Mid  (only AB breakthrough: 10W)
    "runs/az_no_queen_run/ckpt_iter9.pt",   # AZ-Best (final)
]

PRESETS = {
    "v4": [("V4_extra_cannon", "extra_cannon", V4_AGENTS)],
    "v3": [("V3_no_queen",     "no_queen",     V3_AGENTS)],
    "both": [
        ("V4_extra_cannon", "extra_cannon", V4_AGENTS),
        ("V3_no_queen",     "no_queen",     V3_AGENTS),
    ],
}


# ====================================================================
# Nash Equilibrium (LP Minimax for two-player zero-sum)
# ====================================================================

def compute_nash_equilibrium(payoff_matrix: np.ndarray):
    """Compute mixed-strategy Nash Equilibrium via Linear Programming.

    payoff_matrix: NxN matrix of row-player win rates (0-1).
    Returns (nash_distribution, game_value) or (None, None) on failure.
    """
    from scipy.optimize import linprog

    A = payoff_matrix - 0.5   # centre around 0 for zero-sum LP

    num_strategies = A.shape[0]

    # Objective: maximise game value v  <=>  minimise -v
    c = np.zeros(num_strategies + 1)
    c[-1] = -1.0

    # Constraints: for each opponent pure strategy j,
    #   sum_i  x_i * A[i,j]  >=  v
    #   <=>  -sum_i  x_i * A[i,j] + v  <=  0
    A_ub = np.hstack([-A.T, np.ones((num_strategies, 1))])
    b_ub = np.zeros(num_strategies)

    # Probability simplex: sum(x) = 1
    A_eq = np.ones((1, num_strategies + 1))
    A_eq[0, -1] = 0.0
    b_eq = np.array([1.0])

    bounds = [(0, None)] * num_strategies + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')

    if res.success:
        nash_dist = res.x[:-1].copy()
        nash_dist[nash_dist < 1e-5] = 0       # clip floating-point dust
        total = nash_dist.sum()
        if total > 0:
            nash_dist /= total
        game_value = res.x[-1]
        return nash_dist, game_value
    return None, None


# ====================================================================
# Heatmap
# ====================================================================

def plot_payoff_heatmap(matrix: np.ndarray, labels: List[str],
                        save_path: str, title: str = "") -> None:
    """Plot and save an annotated heatmap of the payoff matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="equal")

    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            colour = "white" if abs(val - 0.5) > 0.25 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=colour)

    ax.set_xlabel("Column player (opponent)", fontsize=11)
    ax.set_ylabel("Row player", fontsize=11)
    ax.set_title(title or "EGTA Payoff Matrix - Row Player Win Rate",
                 fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Win Rate")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Heatmap saved: {save_path}")


# ====================================================================
# Progress I/O
# ====================================================================

def _write_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _load_progress(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ====================================================================
# Single-universe tournament
# ====================================================================

def _init_worker(ablation: str):
    """Per-worker process initializer: apply ablation config."""
    from hybrid.rl.az_runner import _apply_ablation
    _apply_ablation(ablation)


def _run_pair(args_tuple):
    """Worker function: play all games for one pair, return results.

    Runs in a subprocess — each call creates its own agents.
    Writes per-game progress to live/pair_{i}_{j}.log for monitoring.
    """
    (i, j, label_i, label_j, spec_i, spec_j,
     games_per_pair, simulations, use_cpp, seed, pair_idx, outdir) = args_tuple

    # Per-game log file
    import os, time as _time
    live_dir = os.path.join(outdir, "live")
    os.makedirs(live_dir, exist_ok=True)
    pair_log_path = os.path.join(live_dir, f"pair_{i}_{j}.log")
    pair_log = open(pair_log_path, "w", encoding="utf-8")
    pair_log.write(f"{label_i} vs {label_j} ({games_per_pair} games)\n")
    pair_log.flush()

    games_per_half = games_per_pair // 2
    pair_scores_ij = []
    pair_scores_ji = []
    game_count = 0

    for half_label, i_is_chess in [("i=Chess", True), ("i=Xiangqi", False)]:
        for gi in range(games_per_half):
            game_seed = seed + pair_idx * 10000 + gi + (0 if i_is_chess else 5000)

            agent_i = _create_agent(spec_i, simulations, game_seed, use_cpp)
            agent_j = _create_agent(spec_j, simulations, game_seed + 500, use_cpp)

            if i_is_chess:
                agent_chess, agent_xiangqi = agent_i, agent_j
            else:
                agent_chess, agent_xiangqi = agent_j, agent_i

            t0 = _time.time()
            result = play_arena_game(agent_chess, agent_xiangqi,
                                     seed=game_seed, use_cpp=use_cpp)
            elapsed = _time.time() - t0

            if result["winner_side"] is None:
                score_i = 0.5
            elif (result["winner_side"] == "chess" and i_is_chess) or \
                 (result["winner_side"] == "xiangqi" and not i_is_chess):
                score_i = 1.0
            else:
                score_i = 0.0

            pair_scores_ij.append(score_i)
            pair_scores_ji.append(1.0 - score_i)
            game_count += 1

            outcome = "WIN_i" if score_i == 1.0 else ("DRAW" if score_i == 0.5 else "WIN_j")
            avg_i = sum(pair_scores_ij) / len(pair_scores_ij)
            pair_log.write(
                f"  [{game_count}/{games_per_pair}] {half_label} "
                f"{outcome} ({result['plies']}ply, {result['reason']}, "
                f"{elapsed:.1f}s) running={avg_i:.3f}\n")
            pair_log.flush()

    avg_ij = sum(pair_scores_ij) / max(len(pair_scores_ij), 1)
    avg_ji = sum(pair_scores_ji) / max(len(pair_scores_ji), 1)
    pair_log.write(f"DONE: {avg_ij:.3f} / {avg_ji:.3f}\n")
    pair_log.close()

    return {
        "i": i, "j": j,
        "label_i": label_i, "label_j": label_j,
        "scores_ij": pair_scores_ij,
        "scores_ji": pair_scores_ji,
        "avg_ij": avg_ij,
        "avg_ji": avg_ji,
    }


def run_tournament(
    agent_specs: List[str],
    games_per_pair: int,
    simulations: int,
    ablation: str,
    use_cpp: bool,
    seed: int,
    outdir: str,
    universe_label: str = "",
    workers: int = 1,
) -> dict:
    """Run round-robin tournament and compute Nash Equilibrium."""

    from hybrid.rl.az_runner import _apply_ablation
    _apply_ablation(ablation)

    os.makedirs(outdir, exist_ok=True)
    progress_path = os.path.join(outdir, "tournament_progress.json")

    n = len(agent_specs)
    labels = [_agent_label(s) for s in agent_specs]

    # Load existing progress for resume
    existing = _load_progress(progress_path)
    completed_pairs: dict = {}
    if existing and existing.get("agent_specs") == agent_specs:
        completed_pairs = existing.get("completed_pairs", {})
        print(f"  Resuming: {len(completed_pairs)} pairs already done.")

    # Results storage: M[i][j] = list of per-game scores (1/0.5/0)
    scores = [[[] for _ in range(n)] for _ in range(n)]

    # Reload completed pair scores
    for key, data in completed_pairs.items():
        i_idx, j_idx = map(int, key.split(","))
        scores[i_idx][j_idx] = data["scores_ij"]
        scores[j_idx][i_idx] = data["scores_ji"]

    all_pairs = list(combinations(range(n), 2))
    total_pairs = len(all_pairs)
    pending_pairs = [(i, j) for i, j in all_pairs if f"{i},{j}" not in completed_pairs]

    tag = universe_label or ablation
    print(f"\n{'='*60}")
    print(f"  EGTA Tournament: {tag}")
    print(f"  Agents: {labels}")
    print(f"  Pairs: {total_pairs} ({len(pending_pairs)} pending) | "
          f"Games/pair: {games_per_pair} | Sims: {simulations}")
    print(f"  Ablation: {ablation} | C++: {use_cpp} | Workers: {workers}")
    print(f"{'='*60}\n")

    progress = {
        "status": "running",
        "universe": universe_label,
        "agent_specs": agent_specs,
        "labels": labels,
        "games_per_pair": games_per_pair,
        "simulations": simulations,
        "ablation": ablation,
        "total_pairs": total_pairs,
        "completed_pairs": completed_pairs,
        "workers": workers,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if pending_pairs:
        # Build work items
        work_items = []
        for i, j in pending_pairs:
            pair_idx = all_pairs.index((i, j)) + 1
            work_items.append((
                i, j, labels[i], labels[j], agent_specs[i], agent_specs[j],
                games_per_pair, simulations, use_cpp, seed, pair_idx, outdir,
            ))

        # Open log file for incremental writing
        log_path = os.path.join(outdir, "tournament.log")
        log_file = open(log_path, "a", encoding="utf-8")

        def _save_pair_result(pair_result):
            """Save one completed pair: update JSON + write to log."""
            i_idx = pair_result["i"]
            j_idx = pair_result["j"]
            pair_key = f"{i_idx},{j_idx}"
            scores[i_idx][j_idx] = pair_result["scores_ij"]
            scores[j_idx][i_idx] = pair_result["scores_ji"]
            completed_pairs[pair_key] = pair_result

            # Incremental JSON save (checkpoint)
            progress["completed_pairs"] = completed_pairs
            progress["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _write_json(progress_path, progress)

            # Write to log + stdout
            n_done = len(completed_pairs)
            line = (f"  [{n_done}/{total_pairs}] "
                    f"{pair_result['label_i']} vs {pair_result['label_j']}: "
                    f"{pair_result['avg_ij']:.3f} / {pair_result['avg_ji']:.3f}"
                    f"  ({len(pair_result['scores_ij'])} games)")
            print(line, flush=True)
            log_file.write(line + "\n")
            log_file.flush()

        if workers <= 1:
            for item in work_items:
                _save_pair_result(_run_pair(item))
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            print(f"  Launching {workers} worker processes...\n", flush=True)
            log_file.write(f"  Launching {workers} worker processes...\n")
            log_file.flush()
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(ablation,),
            ) as pool:
                futures = {pool.submit(_run_pair, item): item for item in work_items}
                for future in as_completed(futures):
                    _save_pair_result(future.result())

        log_file.close()

    # -- Build payoff matrix --
    matrix = np.full((n, n), 0.5)  # diagonal = 0.5
    for i in range(n):
        for j in range(n):
            if i != j and scores[i][j]:
                matrix[i, j] = sum(scores[i][j]) / len(scores[i][j])

    # Save CSV
    csv_path = os.path.join(outdir, "payoff_matrix.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for i in range(n):
            writer.writerow([labels[i]] + [f"{matrix[i,j]:.4f}" for j in range(n)])
    print(f"\n  Payoff matrix saved: {csv_path}")

    # -- Heatmap --
    heatmap_path = os.path.join(outdir, "payoff_heatmap.png")
    plot_payoff_heatmap(matrix, labels, heatmap_path,
                        title=f"EGTA Payoff Matrix - {tag}")

    # -- Nash Equilibrium --
    nash_dist, game_value = compute_nash_equilibrium(matrix)

    nash_result = {}
    if nash_dist is not None:
        support = [labels[i] for i in range(n) if nash_dist[i] > 1e-4]
        nash_result = {
            "universe": tag,
            "distribution": {labels[i]: round(float(nash_dist[i]), 4)
                             for i in range(n)},
            "support": support,
            "support_size": len(support),
            "game_value": round(float(game_value), 6),
            "interpretation": (
                "TRANSITIVE: Nash concentrates on one agent"
                if len(support) <= 1 else
                "NON-TRANSITIVE: Nash mixes multiple agents (cyclic exploitation)"
            ),
        }

        print(f"\n{'='*60}")
        print(f"  NASH EQUILIBRIUM — {tag}")
        print(f"{'='*60}")
        for lbl, prob in nash_result["distribution"].items():
            bar = "#" * int(prob * 40)
            print(f"  {lbl:>20s}: {prob:.4f}  {bar}")
        print(f"\n  Support size: {nash_result['support_size']}  {nash_result['support']}")
        print(f"  Game value: {nash_result['game_value']:.6f}")
        print(f"  => {nash_result['interpretation']}")
        print(f"{'='*60}\n")
    else:
        nash_result = {"universe": tag, "error": "LP solver failed"}
        print("  WARNING: Nash equilibrium computation failed.")

    nash_path = os.path.join(outdir, "nash_equilibrium.json")
    _write_json(nash_path, nash_result)
    print(f"  Nash result saved: {nash_path}")

    # -- Zero-sum validation --
    max_err = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            err = abs(matrix[i, j] + matrix[j, i] - 1.0)
            max_err = max(max_err, err)
    print(f"  Zero-sum max deviation: {max_err:.6f}")

    # Finalise progress
    progress["status"] = "done"
    progress["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(progress_path, progress)

    return {
        "universe": tag,
        "matrix": matrix.tolist(),
        "labels": labels,
        "nash": nash_result,
    }


# ====================================================================
# Cross-universe comparison
# ====================================================================

def print_comparison(results: List[dict]) -> None:
    """Print side-by-side Nash comparison table."""
    if len(results) < 2:
        return

    print(f"\n{'='*60}")
    print(f"  DUAL-MATRIX COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<30s}", end="")
    for r in results:
        print(f"  {r['universe']:>15s}", end="")
    print()
    print(f"  {'-'*30}", end="")
    for _ in results:
        print(f"  {'-'*15}", end="")
    print()

    # Game value
    print(f"  {'Game value (v)':<30s}", end="")
    for r in results:
        v = r["nash"].get("game_value", "N/A")
        print(f"  {v:>15}", end="")
    print()

    # Support size
    print(f"  {'Nash support size':<30s}", end="")
    for r in results:
        ss = r["nash"].get("support_size", "N/A")
        print(f"  {ss:>15}", end="")
    print()

    # Interpretation
    print(f"  {'Interpretation':<30s}", end="")
    for r in results:
        interp = r["nash"].get("interpretation", "N/A")
        short = "TRANSITIVE" if "TRANSITIVE:" in interp and "NON" not in interp else "NON-TRANSITIVE"
        print(f"  {short:>15}", end="")
    print()

    # Nash distribution
    print(f"\n  Nash distributions:")
    for r in results:
        print(f"    {r['universe']}:")
        dist = r["nash"].get("distribution", {})
        for lbl, prob in dist.items():
            if prob > 0:
                bar = "#" * int(prob * 30)
                print(f"      {lbl:>20s}: {prob:.4f}  {bar}")

    print(f"{'='*60}\n")


# ====================================================================
# CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EGTA Dual-Matrix Ablation Tournament & Nash Equilibrium"
    )
    parser.add_argument("--preset", type=str, default="both",
                        choices=["v3", "v4", "both", "custom"],
                        help="Which universe(s) to run. Default: both")
    parser.add_argument("--agents", type=str, default=None,
                        help="Comma-separated agent specs (only for --preset custom)")
    parser.add_argument("--ablation", type=str, default="extra_cannon",
                        choices=["none", "extra_cannon", "no_queen"],
                        help="Rule variant (only for --preset custom). Default: extra_cannon")
    parser.add_argument("--games-per-pair", type=int, default=100,
                        help="Total games per pair (split evenly across sides). Default: 100")
    parser.add_argument("--simulations", type=int, default=400,
                        help="MCTS simulations for AZ agents. Default: 400")
    parser.add_argument("--use-cpp", action="store_true", default=True,
                        help="Use C++ game engine (default: True)")
    parser.add_argument("--no-cpp", action="store_true", default=False,
                        help="Disable C++ game engine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="runs/egta")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes per universe. Default: 1")
    args = parser.parse_args()

    if args.no_cpp:
        args.use_cpp = False

    # Build list of (label, ablation, agents) universes to run
    if args.preset == "custom":
        if not args.agents:
            parser.error("--agents required when --preset=custom")
        agent_specs = [s.strip() for s in args.agents.split(",")]
        universes = [("custom", args.ablation, agent_specs)]
    else:
        universes = PRESETS[args.preset]

    all_results = []
    for label, ablation, agents in universes:
        universe_outdir = os.path.join(args.outdir, label) if len(universes) > 1 else args.outdir
        result = run_tournament(
            agent_specs=agents,
            games_per_pair=args.games_per_pair,
            simulations=args.simulations,
            ablation=ablation,
            use_cpp=args.use_cpp,
            seed=args.seed,
            outdir=universe_outdir,
            universe_label=label,
            workers=args.workers,
        )
        all_results.append(result)

    # Cross-universe comparison
    if len(all_results) >= 2:
        print_comparison(all_results)

        # Save combined comparison
        comparison_path = os.path.join(args.outdir, "dual_matrix_comparison.json")
        comparison = {
            "universes": [r["universe"] for r in all_results],
            "nash_results": [r["nash"] for r in all_results],
        }
        _write_json(comparison_path, comparison)
        print(f"  Comparison saved: {comparison_path}")


if __name__ == "__main__":
    main()
