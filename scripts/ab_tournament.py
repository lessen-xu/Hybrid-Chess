# -*- coding: utf-8 -*-
"""Noisy AlphaBeta tournament across rule variants.

Injects random opening moves to break deterministic mirrors, then lets
AlphaBeta play out the rest. Tests three rule conditions to quantify
the Queen's impact on game balance.

Usage:
  python -m scripts.ab_tournament              # default: depth 2
  python -m scripts.ab_tournament --depth 1    # faster, depth 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

from tqdm import trange

from hybrid.core.env import HybridChessEnv
from hybrid.core.types import Side
from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig
import hybrid.core.config as cfg


RANDOM_PLIES = 4  # random opening moves to break symmetry


def _progress_path(depth: int) -> str:
    return os.path.join("runs", f"ab_tournament_d{depth}_progress.json")


def _write(path: str, data: dict):
    os.makedirs("runs", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def play_noisy_ab_game(depth: int = 2, random_plies: int = 4, seed: int = 0):
    """Play one game: random opening -> AB vs AB. Returns (winner, plies, reason)."""
    rng = random.Random(seed)
    env = HybridChessEnv()
    state = env.reset()
    ab_chess = AlphaBetaAgent(SearchConfig(depth=depth))
    ab_xq    = AlphaBetaAgent(SearchConfig(depth=depth))

    # Random opening phase
    for _ in range(random_plies):
        legal = env.legal_moves()
        if not legal:
            break
        mv = rng.choice(legal)
        state, _, done, info = env.step(mv)
        if done:
            return info.winner, state.ply, info.reason

    # AlphaBeta takeover
    while True:
        legal = env.legal_moves()
        if not legal:
            break
        agent = ab_chess if state.side_to_move == Side.CHESS else ab_xq
        mv = agent.select_move(state, legal)
        state, _, done, info = env.step(mv)
        if done:
            break

    return info.winner, state.ply, info.reason


CONDITIONS = [
    ("Vanilla (baseline)",    {"no_queen": False, "extra_cannon": False}),
    ("Extra Cannon (V2 env)", {"no_queen": False, "extra_cannon": True}),
    ("No Queen (fair env)",   {"no_queen": True,  "extra_cannon": False}),
]


CONDITION_MAP = {
    "vanilla":      "Vanilla (baseline)",
    "extra_cannon": "Extra Cannon (V2 env)",
    "no_queen":     "No Queen (fair env)",
}


def main():
    parser = argparse.ArgumentParser(description="AB vs AB rule balance tournament")
    parser.add_argument("--depth", type=int, default=2, help="AB search depth (default: 2)")
    parser.add_argument("--games", type=int, default=100, help="Games per condition (default: 100)")
    parser.add_argument("--condition", type=str, default=None,
                        choices=list(CONDITION_MAP.keys()),
                        help="Run only this condition (default: all three)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Custom tag for output files (default: ab_tournament_d{depth})")
    args = parser.parse_args()

    depth = args.depth
    games = args.games

    tag = args.tag or f"ab_tournament_d{depth}"
    prog_path = os.path.join("runs", f"{tag}_progress.json")

    # Filter conditions if --condition is specified
    if args.condition:
        target_label = CONDITION_MAP[args.condition]
        run_conditions = [(n, a) for n, a in CONDITIONS if n == target_label]
    else:
        run_conditions = CONDITIONS

    progress = {
        "status": "running",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
        "depth": depth,
        "random_plies": RANDOM_PLIES,
        "games_per_condition": games,
        "conditions": {},
    }
    _write(prog_path, progress)

    for name, ablation in run_conditions:
        # Apply ablation
        cfg.ABLATION_NO_QUEEN = ablation.get("no_queen", False)
        cfg.ABLATION_EXTRA_CANNON = ablation.get("extra_cannon", False)
        cfg.ABLATION_NO_QUEEN_PROMOTION = False
        cfg.ABLATION_REMOVE_EXTRA_PAWN = False

        cond = {
            "label": name,
            "total": games,
            "completed": 0,
            "chess_wins": 0,
            "xiangqi_wins": 0,
            "draws": 0,
            "plies": [],
            "reasons": [],
            "termination_reasons": {},
            "status": "running",
            "start_time": time.strftime("%H:%M:%S"),
        }
        progress["conditions"][name] = cond
        _write(prog_path, progress)

        print(f"\n{'='*60}")
        print(f"  {name}  (AB-d{depth}, {RANDOM_PLIES} random opening plies)")
        print(f"{'='*60}")

        for gi in trange(games, desc=name):
            winner, plies, reason = play_noisy_ab_game(
                depth=depth, random_plies=RANDOM_PLIES, seed=gi * 31337
            )

            if winner == Side.CHESS:
                cond["chess_wins"] += 1
                tag = "chess_win"
            elif winner == Side.XIANGQI:
                cond["xiangqi_wins"] += 1
                tag = "xiangqi_win"
            else:
                cond["draws"] += 1
                tag = "draw"

            cond["plies"].append(plies)
            cond["reasons"].append(reason)
            cond["termination_reasons"][reason] = cond["termination_reasons"].get(reason, 0) + 1
            cond["completed"] = gi + 1
            cond["last_update"] = time.strftime("%H:%M:%S")
            progress["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _write(prog_path, progress)

            print(f"  [{gi+1}/{games}] {tag} ({plies} ply, {reason})  "
                  f"C:{cond['chess_wins']} X:{cond['xiangqi_wins']} D:{cond['draws']}",
                  flush=True)

        cond["status"] = "done"
        cond["end_time"] = time.strftime("%H:%M:%S")
        cond["avg_ply"] = sum(cond["plies"]) / len(cond["plies"]) if cond["plies"] else 0
        _write(prog_path, progress)

        cw, xw, dr = cond["chess_wins"], cond["xiangqi_wins"], cond["draws"]
        print(f"\n  Result: Chess {cw}/{games} ({cw/games:.0%})  "
              f"Xiangqi {xw}/{games} ({xw/games:.0%})  Draw {dr}/{games} ({dr/games:.0%})")

    progress["status"] = "done"
    progress["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write(prog_path, progress)
    print(f"\n✅ Tournament complete. Progress: {prog_path}")


if __name__ == "__main__":
    main()
