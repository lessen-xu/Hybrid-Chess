# -*- coding: utf-8 -*-
"""Evaluate champion checkpoints from Grand Run V2 against baselines.

Loads specific iteration checkpoints (the "golden nodes") and pits them
against AlphaBeta-d1 and Random with high simulation budgets (400/800).

Usage:
  python -m scripts.eval_champions --run-dir runs/az_grand_run_v2 \
      --iters 11 16 --eval-simulations 400 --games 20 --num-workers 8

Results are printed to stdout AND saved to
  <run-dir>/champion_eval_results.txt

Progress is streamed to:
  <run-dir>/champion_eval_progress.json
"""

from __future__ import annotations

import scripts._fix_encoding  # noqa: F401
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ====================================================================
# Statistics
# ====================================================================

def wilson_ci(wins: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion (95% CI)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


# ====================================================================
# Worker — reports per-game results via queue
# ====================================================================

def _eval_worker(
    worker_id: int,
    games: int,
    model_ckpt_path: str,
    opponent_type: str,
    simulations: int,
    seed: int,
    ablation: str,
    swap_sides: bool,
    game_offset: int,
    total_games: int,
    result_queue: mp.Queue,
    use_cpp: bool = False,
) -> None:
    """Worker: load model, create agent, play games, report EACH game via queue."""
    import torch
    from hybrid.rl.az_runner import _apply_ablation
    from hybrid.rl.az_network import PolicyValueNet
    from hybrid.agents.alphazero_stub import (
        AlphaZeroMiniAgent, MCTSConfig, TorchPolicyValueModel,
    )
    from hybrid.agents.random_agent import RandomAgent
    from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig
    from hybrid.rl.az_eval import play_one_game
    from hybrid.core.types import Side

    _apply_ablation(ablation)

    net = PolicyValueNet()
    ckpt = torch.load(model_ckpt_path, map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["model"])
    net.eval()
    model = TorchPolicyValueModel(net, device="cpu")

    az_agent = AlphaZeroMiniAgent(
        model=model,
        cfg=MCTSConfig(simulations=simulations, dirichlet_eps=0.0),
        seed=seed,
        use_cpp=use_cpp,
    )

    if opponent_type == "random":
        opponent = RandomAgent(seed=seed + 999)
    elif opponent_type == "ab_d1":
        opponent = AlphaBetaAgent(SearchConfig(depth=1))
    else:
        raise ValueError(f"Unknown opponent: {opponent_type}")

    for local_i in range(games):
        global_i = game_offset + local_i

        if swap_sides and global_i >= total_games // 2:
            agent_chess = opponent
            agent_xiangqi = az_agent
            az_is_chess = False
        else:
            agent_chess = az_agent
            agent_xiangqi = opponent
            az_is_chess = True

        t0 = time.time()
        winner, plies, _ = play_one_game(
            agent_chess, agent_xiangqi, seed=seed + local_i,
            use_cpp=use_cpp,
        )
        elapsed = time.time() - t0

        if winner is None:
            outcome = "draw"
        elif (winner == Side.CHESS and az_is_chess) or \
             (winner == Side.XIANGQI and not az_is_chess):
            outcome = "win"
        else:
            outcome = "loss"

        # Report each game individually
        result_queue.put({
            "type": "game",
            "worker_id": worker_id,
            "global_index": global_i,
            "outcome": outcome,
            "plies": plies,
            "elapsed": round(elapsed, 1),
        })

    # Signal worker done
    result_queue.put({"type": "done", "worker_id": worker_id})


# ====================================================================
# Orchestration with live progress
# ====================================================================

def run_eval(
    ckpt_path: str,
    iteration: int,
    opponent_type: str,
    games: int,
    simulations: int,
    num_workers: int,
    ablation: str = "extra_cannon",
    seed: int = 42,
    progress_path: str | None = None,
    all_progress: dict | None = None,
    use_cpp: bool = False,
) -> dict:
    """Run parallel evaluation with live progress updates."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    # split games across workers
    workers_games = []
    base = games // num_workers
    rem = games % num_workers
    for i in range(num_workers):
        workers_games.append(base + (1 if i < rem else 0))

    offsets = []
    acc = 0
    for g in workers_games:
        offsets.append(acc)
        acc += g

    active_workers = 0
    procs = []
    for i in range(num_workers):
        if workers_games[i] == 0:
            continue
        p = ctx.Process(
            target=_eval_worker,
            args=(
                i, workers_games[i], ckpt_path, opponent_type,
                simulations, seed + i * 1000, ablation,
                True,          # swap_sides
                offsets[i],
                games,
                result_queue,
                use_cpp,
            ),
        )
        p.start()
        procs.append(p)
        active_workers += 1

    # Collect results with live progress
    matchup_key = f"iter{iteration}_vs_{opponent_type}_{simulations}sims"
    game_results = []
    done_workers = 0

    while done_workers < active_workers:
        msg = result_queue.get()
        if msg["type"] == "done":
            done_workers += 1
            continue

        game_results.append(msg)

        # Update live progress
        wins = sum(1 for r in game_results if r["outcome"] == "win")
        draws = sum(1 for r in game_results if r["outcome"] == "draw")
        losses = sum(1 for r in game_results if r["outcome"] == "loss")
        completed = len(game_results)

        if all_progress is not None and progress_path:
            all_progress["matchups"][matchup_key] = {
                "iteration": iteration,
                "opponent": opponent_type,
                "simulations": simulations,
                "completed": completed,
                "total": games,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "status": "running",
                "last_update": time.strftime("%H:%M:%S"),
            }
            all_progress["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _write_progress(progress_path, all_progress)

            # Live console output
            print(f"    [{completed}/{games}] {wins}W/{draws}D/{losses}L  "
                  f"(game {msg['global_index']+1}: {msg['outcome']}, "
                  f"{msg['plies']} ply, {msg['elapsed']}s)", flush=True)

    for p in procs:
        p.join()

    # Final summary
    wins = sum(1 for r in game_results if r["outcome"] == "win")
    draws = sum(1 for r in game_results if r["outcome"] == "draw")
    losses = sum(1 for r in game_results if r["outcome"] == "loss")
    total_plies = sum(r["plies"] for r in game_results)
    n = len(game_results)
    avg_plies = total_plies / n if n > 0 else 0

    wr, wr_lo, wr_hi = wilson_ci(wins, n)
    score = (wins + 0.5 * draws) / n if n > 0 else 0

    result = {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "games": n,
        "win_rate": wr,
        "win_rate_ci": (wr_lo, wr_hi),
        "score": score,
        "avg_plies": avg_plies,
    }

    # Mark matchup complete in progress
    if all_progress is not None and progress_path:
        all_progress["matchups"][matchup_key]["status"] = "done"
        all_progress["matchups"][matchup_key].update(result)
        _write_progress(progress_path, all_progress)

    return result


def _write_progress(path: str, data: dict):
    """Atomically write progress JSON."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def fmt_result(label: str, res: dict, sims: int) -> str:
    """Pretty-format a result dict."""
    lines = []
    lines.append(f"  {label} ({sims} sims)")
    lines.append(f"    W/D/L: {res['wins']}W / {res['draws']}D / {res['losses']}L  "
                 f"({res['games']} games)")
    lines.append(f"    Win rate: {res['win_rate']:.1%}  "
                 f"95% CI [{res['win_rate_ci'][0]:.1%}, {res['win_rate_ci'][1]:.1%}]")
    lines.append(f"    Score: {res['score']:.3f}")
    lines.append(f"    Avg plies: {res['avg_plies']:.1f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate champion checkpoints from a training run."
    )
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to the training run directory")
    parser.add_argument("--iters", type=int, nargs="+", default=[11, 16],
                        help="Iteration numbers to evaluate (default: 11 16)")
    parser.add_argument("--eval-simulations", type=int, nargs="+", default=[400],
                        help="MCTS simulation budgets to test (default: 400)")
    parser.add_argument("--games", type=int, default=20,
                        help="Number of games per matchup (default: 20)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Parallel workers (default: 8)")
    parser.add_argument("--ablation", type=str, default="extra_cannon")
    parser.add_argument("--use-cpp", action="store_true", default=False,
                        help="Use C++ engine for MCTS")
    parser.add_argument("--opponents", type=str, nargs="+",
                        default=["ab_d1", "random"],
                        help="Opponents to test against (default: ab_d1 random)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: run directory not found: {run_dir}")
        sys.exit(1)

    # Progress file
    progress_path = str(run_dir / "champion_eval_progress.json")
    all_progress = {
        "status": "running",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "iterations": args.iters,
            "simulations": args.eval_simulations,
            "games": args.games,
            "opponents": args.opponents,
        },
        "matchups": {},
        "plan": [],
    }

    # Build plan of all matchups
    for it in args.iters:
        for sims in args.eval_simulations:
            for opp in args.opponents:
                all_progress["plan"].append({
                    "iteration": it,
                    "opponent": opp,
                    "simulations": sims,
                    "key": f"iter{it}_vs_{opp}_{sims}sims",
                })
    _write_progress(progress_path, all_progress)

    output_lines: List[str] = []

    def log(msg: str = ""):
        print(msg, flush=True)
        output_lines.append(msg)

    log("=" * 70)
    log("  Champion Evaluation - Hybrid Chess AlphaZero")
    log("=" * 70)
    log(f"  Run dir:      {run_dir}")
    log(f"  Iterations:   {args.iters}")
    log(f"  Sim budgets:  {args.eval_simulations}")
    log(f"  Games/match:  {args.games}")
    log(f"  Workers:      {args.num_workers}")
    log(f"  Opponents:    {args.opponents}")
    log(f"  Ablation:     {args.ablation}")
    log("=" * 70)

    opponent_labels = {"ab_d1": "AlphaBeta-d1", "random": "Random"}

    for it in args.iters:
        ckpt = run_dir / f"ckpt_iter{it}.pt"
        if not ckpt.exists():
            log(f"\n[!] Checkpoint not found: {ckpt} -- skipping.")
            continue

        log(f"\n{'─' * 70}")
        log(f"  Iteration {it}  ({ckpt.name})")
        log(f"{'─' * 70}")

        for sims in args.eval_simulations:
            for opp in args.opponents:
                opp_label = opponent_labels.get(opp, opp)
                log(f"\n  > vs {opp_label} @ {sims} sims ...")
                t0 = time.time()
                res = run_eval(
                    ckpt_path=str(ckpt),
                    iteration=it,
                    opponent_type=opp,
                    games=args.games,
                    simulations=sims,
                    num_workers=args.num_workers,
                    ablation=args.ablation,
                    progress_path=progress_path,
                    all_progress=all_progress,
                    use_cpp=args.use_cpp,
                )
                elapsed = time.time() - t0
                log(fmt_result(f"vs {opp_label}", res, sims))
                log(f"    Time: {elapsed:.1f}s")

    all_progress["status"] = "done"
    all_progress["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_progress(progress_path, all_progress)

    log(f"\n{'=' * 70}")
    log("  ✅ Evaluation complete.")
    log(f"{'=' * 70}")

    # Save results
    out_path = run_dir / "champion_eval_results.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    log(f"\n  📄 Results saved to: {out_path}")


if __name__ == "__main__":
    main()
