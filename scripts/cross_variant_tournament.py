"""Cross-variant round-robin tournament.

Take best_model.pt from each rule-variant training run and have them
play against each other under a common ruleset (default rules).
Uses multiprocessing for speed.
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass


# ── Agent pool ──────────────────────────────────────────────
VARIANTS = {
    "Default":     "runs/rq4_az_default_v2/best_model.pt",
    "Q_only":      "runs/rq4_az_noq_only/best_model.pt",
    "X_only":      "runs/rq4_az_xqqueen_only/best_model.pt",
    "PK":          "runs/rq4_az_palace_knight_v2/best_model.pt",
    "PK_noPromo":  "runs/rq4_az_pk_nopromo/best_model.pt",
    "PK_xqQueen":  "runs/rq4_az_pk_xqqueen/best_model.pt",
    "noQ_noPromo": "runs/rq4_az_nq_nopromo/best_model.pt",
    "noQ_PK":      "runs/rq4_az_nq_pk/best_model.pt",
    "noQ_ALL":     "runs/rq4_az_nq_allrules_v2/best_model.pt",
}


def _play_one(args: tuple) -> dict:
    """Worker: play one game between two AZ models."""
    name_a, path_a, name_b, path_b, game_idx, a_is_chess, sims, seed = args

    import sys, os
    # Ensure hybrid package is importable in spawned workers
    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if proj not in sys.path:
        sys.path.insert(0, proj)

    import torch
    from hybrid.core.env import HybridChessEnv
    from hybrid.core.types import Side
    from hybrid.rl.az_runner import build_net_from_checkpoint
    from hybrid.agents.alphazero_stub import (
        AlphaZeroMiniAgent, MCTSConfig, TorchPolicyValueModel,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _make_agent(path, s):
        net = build_net_from_checkpoint(path, device=device)
        model = TorchPolicyValueModel(net, device=device)
        return AlphaZeroMiniAgent(
            model=model,
            cfg=MCTSConfig(simulations=sims, dirichlet_eps=0.0),
            seed=s,
            use_cpp=True,
        )

    agent_a = _make_agent(path_a, seed)
    agent_b = _make_agent(path_b, seed + 500)

    # Default rules for tournament
    env = HybridChessEnv(use_cpp=True)
    state = env.reset()

    if a_is_chess:
        agents = {Side.CHESS: agent_a, Side.XIANGQI: agent_b}
    else:
        agents = {Side.CHESS: agent_b, Side.XIANGQI: agent_a}

    while True:
        legal = env.legal_moves()
        if not legal:
            break
        mv = agents[state.side_to_move].select_move(state, legal)
        state, _, done, info = env.step(mv)
        if done:
            break

    # Result from A's perspective
    if info.winner is None:
        outcome = "draw"
    elif (info.winner == Side.CHESS and a_is_chess) or \
         (info.winner == Side.XIANGQI and not a_is_chess):
        outcome = "win_a"
    else:
        outcome = "win_b"

    return {
        "name_a": name_a, "name_b": name_b,
        "a_is_chess": a_is_chess,
        "outcome": outcome,
        "plies": state.ply,
        "game_idx": game_idx,
    }


def run_tournament(games_per_half: int = 25, sims: int = 50, 
                   workers: int = 4, seed: int = 42,
                   outdir: str = "runs/cross_variant_tournament"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Verify all models exist
    for name, path in VARIANTS.items():
        assert Path(path).exists(), f"Missing: {path} ({name})"

    pairs = list(itertools.combinations(VARIANTS.keys(), 2))
    total_games = len(pairs) * 2 * games_per_half
    print(f"Tournament: {len(VARIANTS)} agents, {len(pairs)} pairs, "
          f"{games_per_half} games/half = {total_games} total games")
    print(f"Workers: {workers}, Sims: {sims}\n")

    # Build task list
    tasks = []
    for name_a, name_b in pairs:
        path_a, path_b = VARIANTS[name_a], VARIANTS[name_b]
        for half, a_is_chess in [(0, True), (1, False)]:
            for gi in range(games_per_half):
                game_seed = seed + hash((name_a, name_b, half, gi)) % 10000
                tasks.append((
                    name_a, path_a, name_b, path_b,
                    gi + half * games_per_half,
                    a_is_chess, sims, game_seed,
                ))

    # Execute
    results = []
    t0 = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_play_one, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            completed += 1
            if completed % 20 == 0 or completed == total_games:
                elapsed = time.time() - t0
                eta = elapsed / completed * (total_games - completed)
                print(f"  [{completed}/{total_games}] "
                      f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

    total_time = time.time() - t0
    print(f"\nDone in {total_time:.0f}s ({total_time/60:.1f} min)")

    # Build payoff matrix
    names = list(VARIANTS.keys())
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}
    wins = [[0]*n for _ in range(n)]
    draws = [[0]*n for _ in range(n)]
    total = [[0]*n for _ in range(n)]

    for r in results:
        i, j = idx[r["name_a"]], idx[r["name_b"]]
        total[i][j] += 1
        total[j][i] += 1
        if r["outcome"] == "win_a":
            wins[i][j] += 1
        elif r["outcome"] == "win_b":
            wins[j][i] += 1
        else:
            draws[i][j] += 1
            draws[j][i] += 1

    # Score matrix (win=1, draw=0.5, loss=0)
    score = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                score[i][j] = 0.5
            elif total[i][j] > 0:
                score[i][j] = (wins[i][j] + 0.5 * draws[i][j]) / total[i][j]

    # Print matrix
    print(f"\n{'':18s}", end="")
    for name in names:
        print(f" {name:>10s}", end="")
    print(f" {'AVG':>7s}")
    print("-" * (18 + 11*n + 8))
    for i, name in enumerate(names):
        print(f"{name:18s}", end="")
        row_scores = []
        for j in range(n):
            print(f" {score[i][j]:10.3f}", end="")
            if i != j:
                row_scores.append(score[i][j])
        avg = sum(row_scores) / max(len(row_scores), 1)
        print(f" {avg:7.3f}")

    # Save results
    with open(outdir / "game_records.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)

    with open(outdir / "payoff_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, name in enumerate(names):
            w.writerow([name] + [f"{score[i][j]:.3f}" for j in range(n)])

    # Summary
    avg_scores = {}
    for i, name in enumerate(names):
        s = [score[i][j] for j in range(n) if i != j]
        avg_scores[name] = sum(s) / len(s)

    ranking = sorted(avg_scores.items(), key=lambda x: -x[1])
    print(f"\n{'Rank':>4s}  {'Agent':18s}  {'Avg Score':>9s}")
    print("-" * 35)
    for rank, (name, sc) in enumerate(ranking, 1):
        print(f"{rank:4d}  {name:18s}  {sc:9.3f}")

    summary = {
        "n_agents": n,
        "n_pairs": len(pairs),
        "games_per_half": games_per_half,
        "total_games": total_games,
        "simulations": sims,
        "total_time_s": round(total_time, 1),
        "ranking": [{"agent": name, "avg_score": round(sc, 4)} 
                    for name, sc in ranking],
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to: {outdir}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=25, help="Games per half (total=2*games per pair)")
    p.add_argument("--sims", type=int, default=50, help="MCTS simulations")
    p.add_argument("--workers", type=int, default=4, help="Parallel workers")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="runs/cross_variant_tournament")
    args = p.parse_args()
    run_tournament(args.games, args.sims, args.workers, args.seed, args.outdir)
