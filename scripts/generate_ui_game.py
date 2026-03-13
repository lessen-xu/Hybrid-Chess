#!/usr/bin/env python3
"""Generate a game replay JSON file for the Web UI.

Plays a single game between two agents, recording every board state
in ASCII format + move notation.  Output is directly loadable by
ui/index.html via the "Open JSON" button.

Usage:
  python -m scripts.generate_ui_game --chess ab_d1 --xiangqi random --out ui/demo_game.json
  python -m scripts.generate_ui_game --chess random --xiangqi ab_d1 --out ui/demo_game.json
"""

from __future__ import annotations
import scripts._fix_encoding  # noqa: F401
import argparse
import json
import time
from pathlib import Path

from hybrid.core.env import HybridChessEnv
from hybrid.core.types import Side, PieceKind
from hybrid.core.render import render_board
from hybrid.agents.base import Agent


# ── Agent factory (lightweight, no C++ dependency) ──

def create_agent(spec: str, seed: int = 0) -> Agent:
    if spec == "random":
        from hybrid.agents.random_agent import RandomAgent
        return RandomAgent(seed=seed)
    if spec == "greedy":
        from hybrid.agents.greedy_agent import GreedyAgent
        return GreedyAgent(seed=seed)

    import re
    ab = re.match(r"ab_d(\d+)", spec)
    if ab:
        depth = int(ab.group(1))
        from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig
        return AlphaBetaAgent(SearchConfig(depth=depth))

    raise ValueError(f"Unknown agent spec: {spec!r}  (supported: random, greedy, ab_d1, ab_d2, ab_d4)")


# ── Move formatting ──

_COL_NAMES = "abcdefghi"

def format_move(mv) -> str:
    """Format move as 'e2-e4' style notation."""
    base = f"{_COL_NAMES[mv.fx]}{mv.fy + 1}-{_COL_NAMES[mv.tx]}{mv.ty + 1}"
    if mv.promotion is not None:
        promo_map = {
            PieceKind.QUEEN: "Q", PieceKind.ROOK: "R",
            PieceKind.BISHOP: "B", PieceKind.KNIGHT: "N",
        }
        base += "=" + promo_map.get(mv.promotion, "?")
    return base


def main():
    parser = argparse.ArgumentParser(description="Generate a UI-compatible game replay JSON.")
    parser.add_argument("--chess", type=str, default="ab_d1",
                        help="Agent for Chess side (bottom). Default: ab_d1")
    parser.add_argument("--xiangqi", type=str, default="random",
                        help="Agent for Xiangqi side (top). Default: random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="ui/demo_game.json",
                        help="Output JSON path. Default: ui/demo_game.json")
    args = parser.parse_args()

    print(f"  Chess (bottom):  {args.chess}")
    print(f"  Xiangqi (top):   {args.xiangqi}")
    print(f"  Seed:            {args.seed}")
    print()

    # Create agents
    agent_chess = create_agent(args.chess, seed=args.seed)
    agent_xiangqi = create_agent(args.xiangqi, seed=args.seed + 1000)
    agents = {Side.CHESS: agent_chess, Side.XIANGQI: agent_xiangqi}

    # Play game
    env = HybridChessEnv()
    state = env.reset()

    states_ascii = [render_board(state.board)]
    moves = []
    info = None

    t0 = time.time()
    print("  Playing...", flush=True)

    while True:
        legal = env.legal_moves()
        if not legal:
            break

        agent = agents[state.side_to_move]
        mv = agent.select_move(state, legal)
        moves.append(format_move(mv))

        state, reward, done, info = env.step(mv)
        states_ascii.append(render_board(state.board))

        # Progress
        if state.ply % 20 == 0:
            print(f"    ply {state.ply}...", flush=True)

        if done:
            break

    elapsed = time.time() - t0

    # Determine result string
    if info is None or info.winner is None:
        result = "Draw"
    elif info.winner == Side.CHESS:
        result = f"Chess wins ({args.chess})"
    else:
        result = f"Xiangqi wins ({args.xiangqi})"

    reason = info.reason if info and hasattr(info, "reason") else "unknown"

    game_data = {
        "result": result,
        "moves": moves,
        "states_ascii": states_ascii,
        "meta": {
            "chess_agent": args.chess,
            "xiangqi_agent": args.xiangqi,
            "plies": state.ply,
            "reason": reason,
            "seed": args.seed,
            "elapsed": round(elapsed, 2),
        }
    }

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(game_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"  Result:   {result}")
    print(f"  Plies:    {state.ply}")
    print(f"  Reason:   {reason}")
    print(f"  Time:     {elapsed:.1f}s")
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
