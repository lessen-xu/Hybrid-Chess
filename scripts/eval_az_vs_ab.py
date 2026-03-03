# -*- coding: utf-8 -*-
"""AZ vs AB-d2 showdown with termination tracking.

Loads an AZ checkpoint and plays against AlphaBeta-d2 with:
- Side-swapping (first half AZ=Chess, second half AZ=Xiangqi)
- Random opening plies for diversity
- Termination reason tracking
- Progress JSON for live dashboard

Usage:
  python -m scripts.eval_az_vs_ab \
      --ckpt runs/az_grand_run_v2/ckpt_iter16.pt \
      --eval-simulations 800 --games 40 --max-ply 400 \
      --ablation no_queen --random-plies 4 \
      --tag az_vs_ab_showdown
"""

from __future__ import annotations

import scripts._fix_encoding  # noqa: F401
import argparse
import json
import os
import random
import time

import torch
from tqdm import trange

from hybrid.core.env import HybridChessEnv
from hybrid.core.types import Side
from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig
from hybrid.agents.alphazero_stub import (
    AlphaZeroMiniAgent, MCTSConfig, TorchPolicyValueModel,
)
from hybrid.rl.az_network import PolicyValueNet
import hybrid.core.config as cfg


ABLATION_PRESETS = {
    "none":         {},
    "no_queen":     {"ABLATION_NO_QUEEN": True},
    "extra_cannon": {"ABLATION_EXTRA_CANNON": True},
}


def apply_ablation(name: str):
    cfg.ABLATION_NO_QUEEN = False
    cfg.ABLATION_NO_QUEEN_PROMOTION = False
    cfg.ABLATION_EXTRA_CANNON = False
    cfg.ABLATION_REMOVE_EXTRA_PAWN = False
    preset = ABLATION_PRESETS.get(name)
    if preset is None:
        raise ValueError(f"Unknown ablation: {name!r}")
    for key, val in preset.items():
        setattr(cfg, key, val)


def _write(path: str, data: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def play_one_game(
    az_agent,
    ab_agent,
    az_is_chess: bool,
    max_ply: int,
    random_plies: int,
    seed: int,
    use_cpp: bool = False,
):
    """Play one game with random opening + AZ vs AB. Returns (winner, plies, reason, az_outcome)."""
    rng = random.Random(seed)
    env = HybridChessEnv(max_plies=max_ply, use_cpp=use_cpp)
    state = env.reset()

    agents = {}
    if az_is_chess:
        agents[Side.CHESS] = az_agent
        agents[Side.XIANGQI] = ab_agent
    else:
        agents[Side.CHESS] = ab_agent
        agents[Side.XIANGQI] = az_agent

    # Random opening phase
    for _ in range(random_plies):
        legal = env.legal_moves()
        if not legal:
            break
        mv = rng.choice(legal)
        state, _, done, info = env.step(mv)
        if done:
            az_side = Side.CHESS if az_is_chess else Side.XIANGQI
            if info.winner is None:
                outcome = "draw"
            elif info.winner == az_side:
                outcome = "win"
            else:
                outcome = "loss"
            return info.winner, state.ply, info.reason, outcome

    # Main game
    while True:
        legal = env.legal_moves()
        if not legal:
            break
        agent = agents[state.side_to_move]
        mv = agent.select_move(state, legal)
        state, _, done, info = env.step(mv)
        if done:
            break

    az_side = Side.CHESS if az_is_chess else Side.XIANGQI
    if info.winner is None:
        outcome = "draw"
    elif info.winner == az_side:
        outcome = "win"
    else:
        outcome = "loss"

    return info.winner, state.ply, info.reason, outcome


def main():
    parser = argparse.ArgumentParser(description="AZ vs AB-d2 showdown")
    parser.add_argument("--ckpt", type=str, required=True, help="AZ checkpoint path")
    parser.add_argument("--eval-simulations", type=int, default=800)
    parser.add_argument("--ab-depth", type=int, default=2)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--max-ply", type=int, default=400)
    parser.add_argument("--random-plies", type=int, default=4)
    parser.add_argument("--ablation", type=str, default="no_queen",
                        choices=list(ABLATION_PRESETS.keys()))
    parser.add_argument("--tag", type=str, default="az_vs_ab_showdown")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cpp", action="store_true", default=False,
                        help="Use C++ engine for MCTS")
    args = parser.parse_args()

    apply_ablation(args.ablation)

    # Load AZ model
    print(f"Loading checkpoint: {args.ckpt}")
    net = PolicyValueNet()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["model"])
    net.eval()
    model = TorchPolicyValueModel(net, device="cpu")

    az_agent = AlphaZeroMiniAgent(
        model=model,
        cfg=MCTSConfig(simulations=args.eval_simulations, dirichlet_eps=0.0),
        seed=args.seed,
        use_cpp=args.use_cpp,
    )

    ab_agent = AlphaBetaAgent(SearchConfig(depth=args.ab_depth))

    # Progress file
    prog_path = os.path.join("runs", f"{args.tag}_progress.json")
    progress = {
        "status": "running",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "ckpt": args.ckpt,
            "eval_simulations": args.eval_simulations,
            "ab_depth": args.ab_depth,
            "games": args.games,
            "max_ply": args.max_ply,
            "random_plies": args.random_plies,
            "ablation": args.ablation,
        },
        "conditions": {},
    }

    # Single condition entry (for compatibility with monitor_tournament dashboard)
    cond_label = f"AZ-iter16 ({args.eval_simulations}sims) vs AB-d{args.ab_depth}"
    cond = {
        "label": cond_label,
        "total": args.games,
        "completed": 0,
        "chess_wins": 0,
        "xiangqi_wins": 0,
        "draws": 0,
        "az_wins": 0,
        "az_losses": 0,
        "az_draws": 0,
        "plies": [],
        "reasons": [],
        "termination_reasons": {},
        "az_outcomes": [],
        "per_side": {
            "az_as_chess": {"wins": 0, "draws": 0, "losses": 0, "games": 0},
            "az_as_xiangqi": {"wins": 0, "draws": 0, "losses": 0, "games": 0},
        },
        "status": "running",
        "start_time": time.strftime("%H:%M:%S"),
    }
    progress["conditions"][cond_label] = cond
    _write(prog_path, progress)

    half = args.games // 2
    print(f"\n{'='*70}")
    print(f"  AZ Iter 16 vs AB-d{args.ab_depth}  |  {args.eval_simulations} sims  |"
          f"  {args.ablation}  |  max_ply={args.max_ply}")
    print(f"  Games: {args.games} (first {half} AZ=Chess, last {args.games - half} AZ=Xiangqi)")
    print(f"  Random opening plies: {args.random_plies}")
    print(f"{'='*70}\n")

    for gi in trange(args.games, desc="AZ vs AB"):
        az_is_chess = gi < half
        side_label = "Chess" if az_is_chess else "Xiangqi"

        winner, plies, reason, az_outcome = play_one_game(
            az_agent=az_agent,
            ab_agent=ab_agent,
            az_is_chess=az_is_chess,
            max_ply=args.max_ply,
            random_plies=args.random_plies,
            seed=args.seed + gi * 31337,
            use_cpp=args.use_cpp,
        )

        # Update stats
        if winner == Side.CHESS:
            cond["chess_wins"] += 1
        elif winner == Side.XIANGQI:
            cond["xiangqi_wins"] += 1
        else:
            cond["draws"] += 1

        if az_outcome == "win":
            cond["az_wins"] += 1
        elif az_outcome == "loss":
            cond["az_losses"] += 1
        else:
            cond["az_draws"] += 1

        # Per-side tracking
        side_key = "az_as_chess" if az_is_chess else "az_as_xiangqi"
        cond["per_side"][side_key]["games"] += 1
        if az_outcome == "win":
            cond["per_side"][side_key]["wins"] += 1
        elif az_outcome == "loss":
            cond["per_side"][side_key]["losses"] += 1
        else:
            cond["per_side"][side_key]["draws"] += 1

        cond["plies"].append(plies)
        cond["reasons"].append(reason)
        cond["az_outcomes"].append(az_outcome)
        cond["termination_reasons"][reason] = cond["termination_reasons"].get(reason, 0) + 1
        cond["completed"] = gi + 1
        cond["last_update"] = time.strftime("%H:%M:%S")
        progress["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write(prog_path, progress)

        print(f"  [{gi+1}/{args.games}] AZ={side_label} -> {az_outcome} "
              f"({plies} ply, {reason})  "
              f"W:{cond['az_wins']} D:{cond['az_draws']} L:{cond['az_losses']}",
              flush=True)

    cond["status"] = "done"
    cond["end_time"] = time.strftime("%H:%M:%S")
    cond["avg_ply"] = sum(cond["plies"]) / len(cond["plies"]) if cond["plies"] else 0
    progress["status"] = "done"
    progress["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write(prog_path, progress)

    # Final summary
    n = args.games
    w, d, l = cond["az_wins"], cond["az_draws"], cond["az_losses"]
    score = (w + 0.5 * d) / n if n > 0 else 0
    print(f"\n{'='*70}")
    print(f"  FINAL: AZ {w}W / {d}D / {l}L  (score={score:.3f})")
    print(f"  Avg ply: {cond['avg_ply']:.1f}")
    print(f"  Termination: {json.dumps(cond['termination_reasons'])}")
    ps = cond["per_side"]
    ac = ps["az_as_chess"]
    ax = ps["az_as_xiangqi"]
    print(f"  AZ as Chess:   {ac['wins']}W/{ac['draws']}D/{ac['losses']}L ({ac['games']} games)")
    print(f"  AZ as Xiangqi: {ax['wins']}W/{ax['draws']}D/{ax['losses']}L ({ax['games']} games)")
    print(f"{'='*70}")
    print(f"Progress: {prog_path}")


if __name__ == "__main__":
    main()
