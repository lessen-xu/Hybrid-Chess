# -*- coding: utf-8 -*-
"""Side-switching evaluation arena.

Evaluates Model A vs Model B with forced side-swapping to neutralize
asymmetric game bias. Critical for non-symmetric games where faction
identity (Chess vs Xiangqi) is a core variable.

Protocol:
  Upper half: A=Chess, B=Xiangqi  ->  play N games
  Lower half: A=Xiangqi, B=Chess  ->  play N games
  Total: 2N games, WinRate_A = wins_A / 2N

Usage:
  # AZ checkpoint vs AlphaBeta-d1
  python -m scripts.eval_arena \
      --model-a runs/az_grand_run_v4/ckpt_iter19.pt \
      --model-b ab_d1 \
      --games 10 --simulations 200 --ablation extra_cannon --use-cpp

  # Random vs AB-d1 baseline
  python -m scripts.eval_arena --model-a random --model-b ab_d1 --games 20

  # Two AZ checkpoints head-to-head
  python -m scripts.eval_arena \
      --model-a runs/experiment_vanilla/ckpt_iter19.pt \
      --model-b runs/experiment_no_queen/ckpt_iter19.pt \
      --games 10 --simulations 200 --use-cpp
"""

from __future__ import annotations

import scripts._fix_encoding  # noqa: F401
import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

from hybrid.core.env import HybridChessEnv
from hybrid.core.types import Side, Move
from hybrid.core.rules import generate_legal_moves
from hybrid.agents.base import Agent


# ====================================================================
# Agent factory
# ====================================================================

BASELINE_AGENTS = {"random", "ab_d1", "ab_d2", "ab_d4"}


class _CppABAgent(Agent):
    """AlphaBeta agent backed by C++ best_move() — single call per move."""
    name = "alphabeta_cpp"

    def __init__(self, depth: int):
        self.depth = depth
        from hybrid.core.env import _ensure_cpp_maps
        _ensure_cpp_maps()

    def select_move(self, state, legal_moves):
        from hybrid.core.env import _sync_to_cpp, _PY_TO_CPP_SIDE, _cpp_to_py_move
        from hybrid.cpp_engine import hybrid_cpp_engine as eng
        from hybrid.core.config import MAX_PLIES

        cpp_board = _sync_to_cpp(state.board)
        cpp_side = _PY_TO_CPP_SIDE[state.side_to_move]
        r = eng.best_move(cpp_board, cpp_side, self.depth,
                          state.repetition, state.ply, MAX_PLIES)
        return _cpp_to_py_move(r.best_move)

def _create_agent(
    spec: str,
    simulations: int = 200,
    seed: int = 0,
    use_cpp: bool = False,
) -> Agent:
    """Create an agent from a specification string.

    Spec is either a baseline name ('random', 'ab_d1', 'ab_d2')
    or a path to an AZ checkpoint .pt file.
    """
    if spec == "random":
        from hybrid.agents.random_agent import RandomAgent
        return RandomAgent(seed=seed)
    # AB agents: use C++ best_move when use_cpp=True
    ab_depths = {"ab_d1": 1, "ab_d2": 2, "ab_d4": 4}
    if spec in ab_depths:
        depth = ab_depths[spec]
        if use_cpp:
            return _CppABAgent(depth)
        from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig
        return AlphaBetaAgent(SearchConfig(depth=depth))

    # AZ checkpoint
    import torch
    from hybrid.rl.az_network import PolicyValueNet
    from hybrid.agents.alphazero_stub import (
        AlphaZeroMiniAgent, MCTSConfig, TorchPolicyValueModel,
    )
    net = PolicyValueNet()
    ckpt = torch.load(spec, map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["model"])
    net.eval()
    model = TorchPolicyValueModel(net, device="cuda")
    return AlphaZeroMiniAgent(
        model=model,
        cfg=MCTSConfig(simulations=simulations, dirichlet_eps=0.0),
        seed=seed,
        use_cpp=use_cpp,
    )


def _agent_label(spec: str) -> str:
    """Human-readable label for an agent spec."""
    if spec in BASELINE_AGENTS:
        return spec.upper()
    return Path(spec).stem


# ====================================================================
# Single game with telemetry
# ====================================================================

def play_arena_game(
    agent_chess: Agent,
    agent_xiangqi: Agent,
    seed: int = 0,
    use_cpp: bool = False,
) -> dict:
    """Play one game, collecting telemetry.

    Returns dict with: winner_side, plies, reason, legal_move_counts.
    """
    env = HybridChessEnv(use_cpp=use_cpp)
    state = env.reset()
    agents = {Side.CHESS: agent_chess, Side.XIANGQI: agent_xiangqi}
    legal_move_counts: List[int] = []

    while True:
        legal = env.legal_moves()
        if not legal:
            break
        legal_move_counts.append(len(legal))
        agent = agents[state.side_to_move]
        mv = agent.select_move(state, legal)
        state, _, done, info = env.step(mv)
        if done:
            break

    winner_side = None
    if info.winner == Side.CHESS:
        winner_side = "chess"
    elif info.winner == Side.XIANGQI:
        winner_side = "xiangqi"

    reason = info.reason if hasattr(info, "reason") else ""

    return {
        "winner_side": winner_side,
        "plies": state.ply,
        "reason": reason,
        "legal_move_counts": legal_move_counts,
    }


# ====================================================================
# Arena orchestration
# ====================================================================

def run_arena(
    agent_a_spec: str,
    agent_b_spec: str,
    games_per_half: int,
    simulations: int = 200,
    ablation: str = "none",
    seed: int = 42,
    use_cpp: bool = False,
) -> dict:
    """Run paired side-switching evaluation.

    Returns full results dict with per-game detail and aggregate stats.
    """
    from hybrid.rl.az_runner import _apply_ablation
    _apply_ablation(ablation)

    label_a = _agent_label(agent_a_spec)
    label_b = _agent_label(agent_b_spec)
    total_games = 2 * games_per_half

    print(f"\n  Arena: {label_a} vs {label_b}")
    print(f"  {games_per_half} games/half = {total_games} total")
    print(f"  Ablation: {ablation} | Sims: {simulations} | C++: {use_cpp}\n")

    all_games: List[dict] = []
    wins_a = draws = wins_b = 0

    for half_label, a_is_chess in [("Upper (A=Chess)", True),
                                    ("Lower (A=Xiangqi)", False)]:
        print(f"  --- {half_label} ---")
        for gi in range(games_per_half):
            # Create fresh agents each game for deterministic seeding
            game_seed = seed + gi + (0 if a_is_chess else games_per_half * 100)
            agent_a = _create_agent(agent_a_spec, simulations, game_seed, use_cpp)
            agent_b = _create_agent(agent_b_spec, simulations, game_seed + 500, use_cpp)

            if a_is_chess:
                agent_chess, agent_xiangqi = agent_a, agent_b
            else:
                agent_chess, agent_xiangqi = agent_b, agent_a

            t0 = time.time()
            result = play_arena_game(
                agent_chess, agent_xiangqi,
                seed=game_seed, use_cpp=use_cpp,
            )
            elapsed = time.time() - t0

            # Determine outcome from A's perspective
            if result["winner_side"] is None:
                outcome_a = "draw"
                draws += 1
            elif (result["winner_side"] == "chess" and a_is_chess) or \
                 (result["winner_side"] == "xiangqi" and not a_is_chess):
                outcome_a = "win"
                wins_a += 1
            else:
                outcome_a = "loss"
                wins_b += 1

            game_record = {
                "game_index": len(all_games),
                "a_is_chess": a_is_chess,
                "a_side": "chess" if a_is_chess else "xiangqi",
                "b_side": "xiangqi" if a_is_chess else "chess",
                "winner_side": result["winner_side"],
                "outcome_a": outcome_a,
                "plies": result["plies"],
                "reason": result["reason"],
                "legal_move_counts": result["legal_move_counts"],
                "elapsed": round(elapsed, 1),
            }
            all_games.append(game_record)

            global_idx = len(all_games)
            print(f"    [{global_idx}/{total_games}] "
                  f"{outcome_a} ({result['plies']} ply, "
                  f"{result['reason']}, {elapsed:.1f}s)", flush=True)

    # Aggregate statistics
    n = len(all_games)
    win_rate_a = wins_a / max(n, 1)
    score_a = (wins_a + 0.5 * draws) / max(n, 1)

    # Branching factor by side
    chess_legals, xiangqi_legals = [], []
    for g in all_games:
        for ply_i, count in enumerate(g["legal_move_counts"]):
            if ply_i % 2 == 0:
                chess_legals.append(count)
            else:
                xiangqi_legals.append(count)

    avg_legal_chess = sum(chess_legals) / max(len(chess_legals), 1)
    avg_legal_xiangqi = sum(xiangqi_legals) / max(len(xiangqi_legals), 1)

    # Per-side breakdown for A
    a_as_chess_wins = sum(1 for g in all_games if g["a_is_chess"] and g["outcome_a"] == "win")
    a_as_chess_draws = sum(1 for g in all_games if g["a_is_chess"] and g["outcome_a"] == "draw")
    a_as_chess_losses = sum(1 for g in all_games if g["a_is_chess"] and g["outcome_a"] == "loss")
    a_as_xiangqi_wins = sum(1 for g in all_games if not g["a_is_chess"] and g["outcome_a"] == "win")
    a_as_xiangqi_draws = sum(1 for g in all_games if not g["a_is_chess"] and g["outcome_a"] == "draw")
    a_as_xiangqi_losses = sum(1 for g in all_games if not g["a_is_chess"] and g["outcome_a"] == "loss")

    avg_plies = sum(g["plies"] for g in all_games) / max(n, 1)

    summary = {
        "player_a": label_a,
        "player_b": label_b,
        "ablation": ablation,
        "simulations": simulations,
        "games_per_half": games_per_half,
        "total_games": n,
        "wins_a": wins_a,
        "draws": draws,
        "wins_b": wins_b,
        "win_rate_a": round(win_rate_a, 4),
        "score_a": round(score_a, 4),
        "avg_plies": round(avg_plies, 1),
        "a_as_chess": {"W": a_as_chess_wins, "D": a_as_chess_draws, "L": a_as_chess_losses},
        "a_as_xiangqi": {"W": a_as_xiangqi_wins, "D": a_as_xiangqi_draws, "L": a_as_xiangqi_losses},
        "avg_legal_chess": round(avg_legal_chess, 2),
        "avg_legal_xiangqi": round(avg_legal_xiangqi, 2),
    }

    print(f"\n{'='*60}")
    print(f"  RESULT: {label_a} vs {label_b}")
    print(f"{'='*60}")
    print(f"  Total: {wins_a}W / {draws}D / {wins_b}L  "
          f"(WinRate_A={win_rate_a:.1%}, Score={score_a:.3f})")
    print(f"  A as Chess:   {a_as_chess_wins}W/{a_as_chess_draws}D/{a_as_chess_losses}L")
    print(f"  A as Xiangqi: {a_as_xiangqi_wins}W/{a_as_xiangqi_draws}D/{a_as_xiangqi_losses}L")
    print(f"  Avg plies: {avg_plies:.1f}")
    print(f"  Branching: Chess={avg_legal_chess:.1f}, Xiangqi={avg_legal_xiangqi:.1f}")
    print(f"{'='*60}\n")

    return {"summary": summary, "games": all_games}


# ====================================================================
# CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Side-switching evaluation arena for asymmetric chess."
    )
    parser.add_argument("--model-a", type=str, required=True,
                        help="Player A: checkpoint path or baseline (random/ab_d1/ab_d2)")
    parser.add_argument("--model-b", type=str, required=True,
                        help="Player B: checkpoint path or baseline (random/ab_d1/ab_d2)")
    parser.add_argument("--games", type=int, default=10,
                        help="Games per half (total = 2*games). Default: 10")
    parser.add_argument("--simulations", type=int, default=200,
                        help="MCTS simulations for AZ agents. Default: 200")
    parser.add_argument("--ablation", type=str, default="none",
                        choices=["none", "extra_cannon", "no_queen"],
                        help="Rule variant. Default: none")
    parser.add_argument("--use-cpp", action="store_true", default=False,
                        help="Use C++ game engine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory for results JSON (default: runs/arena/)")
    args = parser.parse_args()

    results = run_arena(
        agent_a_spec=args.model_a,
        agent_b_spec=args.model_b,
        games_per_half=args.games,
        simulations=args.simulations,
        ablation=args.ablation,
        seed=args.seed,
        use_cpp=args.use_cpp,
    )

    # Save results
    outdir = Path(args.outdir) if args.outdir else Path("runs/arena")
    outdir.mkdir(parents=True, exist_ok=True)

    label_a = _agent_label(args.model_a)
    label_b = _agent_label(args.model_b)
    tag = f"{label_a}_vs_{label_b}_{args.ablation}"
    out_path = outdir / f"arena_{tag}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
