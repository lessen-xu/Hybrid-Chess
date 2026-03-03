# -*- coding: utf-8 -*-
"""Pure MCTS sanity check (Silent-Bug Diagnosis Step 3).

Uses a "dummy model" (uniform policy, material-based value) with MCTS
to verify the tree search logic itself is correct.
Pure MCTS should crush RandomAgent even with modest simulations.

Usage:
  python -m scripts.mcts_sanity_check [--sims 50] [--games 3]
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Tuple

from hybrid.core.env import HybridChessEnv, GameState
from hybrid.core.types import Move, Side, PieceKind
from hybrid.agents.alphazero_stub import (
    AlphaZeroMiniAgent,
    MCTSConfig,
    PolicyValueModel,
)
from hybrid.agents.random_agent import RandomAgent

# Helper: flushing print
def fprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# ====================================================================
# Dummy model: uniform policy + material value
# ====================================================================

PIECE_VALUES = {
    PieceKind.PAWN: 1.0, PieceKind.KNIGHT: 3.0, PieceKind.BISHOP: 3.0,
    PieceKind.ROOK: 5.0, PieceKind.QUEEN: 9.0, PieceKind.KING: 0.0,
    PieceKind.SOLDIER: 1.0, PieceKind.HORSE: 3.0, PieceKind.ELEPHANT: 2.0,
    PieceKind.ADVISOR: 2.0, PieceKind.CHARIOT: 5.0, PieceKind.CANNON: 4.5,
    PieceKind.GENERAL: 0.0,
}


def _material_value(state: GameState) -> float:
    """Return value in [-1, 1] from side_to_move's perspective via material count."""
    import math
    diff = 0.0
    for x, y, piece in state.board.iter_pieces():
        val = PIECE_VALUES.get(piece.kind, 0.0)
        if piece.side == state.side_to_move:
            diff += val
        else:
            diff -= val
    return math.tanh(diff / 20.0)


class DummyModel(PolicyValueModel):
    """Uniform policy + material-based value. No neural network."""

    def predict(
        self, state: GameState, legal_moves: List[Move]
    ) -> Tuple[Dict[Move, float], float]:
        n = len(legal_moves)
        if n == 0:
            return {}, 0.0
        policy = {mv: 1.0 / n for mv in legal_moves}
        value = _material_value(state)
        return policy, value


# ====================================================================
# Match runner
# ====================================================================

def play_match(
    mcts_side: Side,
    num_games: int = 3,
    simulations: int = 100,
    max_ply: int = 150,
    seed: int = 42,
) -> dict:
    """Play MCTS (dummy model) vs Random. Returns win/draw/loss stats."""
    import math
    model = DummyModel()
    mcts_agent = AlphaZeroMiniAgent(
        model=model,
        cfg=MCTSConfig(simulations=simulations, dirichlet_eps=0.0),
        seed=seed,
    )
    random_agent = RandomAgent(seed=seed + 1000)

    agents = {
        mcts_side: mcts_agent,
        mcts_side.opponent(): random_agent,
    }

    win, draw, lose = 0, 0, 0
    total_ply = 0
    mat_diffs = []

    for game_i in range(num_games):
        env = HybridChessEnv(max_plies=max_ply)
        state = env.reset()
        t0 = time.time()

        while True:
            legal = env.legal_moves()
            if len(legal) == 0:
                break
            agent = agents[state.side_to_move]
            mv = agent.select_move(state, legal)
            state, reward, done, info = env.step(mv)
            if done:
                break

        elapsed = time.time() - t0
        total_ply += state.ply

        # Compute material diff from MCTS side's perspective
        diff = 0.0
        for x, y, piece in state.board.iter_pieces():
            val = PIECE_VALUES.get(piece.kind, 0.0)
            if piece.side == mcts_side:
                diff += val
            else:
                diff -= val
        mat_diffs.append(diff)

        if info.winner == mcts_side:
            win += 1
            result_char = "W"
        elif info.winner == mcts_side.opponent():
            lose += 1
            result_char = "L"
        else:
            draw += 1
            result_char = "D"

        fprint(f"  game {game_i+1:>2}/{num_games}  "
               f"result={result_char}  ply={state.ply:>3}  "
               f"mat_diff={diff:+.0f}  "
               f"reason={info.reason}  time={elapsed:.1f}s")

    avg_mat = sum(mat_diffs) / len(mat_diffs) if mat_diffs else 0
    return {
        "win": win, "draw": draw, "lose": lose, "total": num_games,
        "avg_ply": total_ply / num_games,
        "avg_mat_diff": avg_mat,
    }


# ====================================================================
# Main
# ====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=50)
    parser.add_argument("--games", type=int, default=3)
    args = parser.parse_args()

    sims = args.sims
    n_games = args.games

    fprint("=" * 70)
    fprint("  PURE MCTS SANITY CHECK  (Silent-Bug Diagnosis Step 3)")
    fprint("=" * 70)
    fprint()
    fprint("Model: DummyModel (uniform policy, material-based value)")
    fprint(f"MCTS simulations: {sims},  max_ply: 150")
    fprint()

    # --- MCTS plays Chess side ---
    fprint("-" * 70)
    fprint(f"Match 1: MCTS=Chess vs Random=Xiangqi  ({n_games} games)")
    fprint("-" * 70)
    t0 = time.time()
    stats_chess = play_match(mcts_side=Side.CHESS, num_games=n_games, simulations=sims)
    t_chess = time.time() - t0
    wr_chess = stats_chess["win"] / stats_chess["total"] * 100
    fprint(f"\n  W={stats_chess['win']} D={stats_chess['draw']} L={stats_chess['lose']}  "
           f"win_rate={wr_chess:.0f}%  avg_ply={stats_chess['avg_ply']:.0f}  "
           f"avg_mat_diff={stats_chess['avg_mat_diff']:+.1f}  "
           f"time={t_chess:.1f}s")

    # --- MCTS plays Xiangqi side ---
    fprint()
    fprint("-" * 70)
    fprint(f"Match 2: MCTS=Xiangqi vs Random=Chess  ({n_games} games)")
    fprint("-" * 70)
    t0 = time.time()
    stats_xiangqi = play_match(mcts_side=Side.XIANGQI, num_games=n_games, simulations=sims)
    t_xiangqi = time.time() - t0
    wr_xiangqi = stats_xiangqi["win"] / stats_xiangqi["total"] * 100
    fprint(f"\n  W={stats_xiangqi['win']} D={stats_xiangqi['draw']} L={stats_xiangqi['lose']}  "
           f"win_rate={wr_xiangqi:.0f}%  avg_ply={stats_xiangqi['avg_ply']:.0f}  "
           f"avg_mat_diff={stats_xiangqi['avg_mat_diff']:+.1f}  "
           f"time={t_xiangqi:.1f}s")

    # --- Verdict ---
    fprint()
    fprint("=" * 70)
    fprint("  VERDICT")
    fprint("=" * 70)

    combined_wr = (stats_chess["win"] + stats_xiangqi["win"]) / (
        stats_chess["total"] + stats_xiangqi["total"]
    ) * 100

    checks = {
        f"MCTS as Chess never loses:        {stats_chess['lose']}L  (expect 0)":
            stats_chess["lose"] == 0,
        f"MCTS as Chess material advantage: {stats_chess['avg_mat_diff']:+.1f}  (expect > 0)":
            stats_chess["avg_mat_diff"] > 0,
        f"Combined loss rate:               {stats_chess['lose']+stats_xiangqi['lose']}/{stats_chess['total']+stats_xiangqi['total']}  (expect <= 30%)":
            (stats_chess["lose"] + stats_xiangqi["lose"]) / (stats_chess["total"] + stats_xiangqi["total"]) <= 0.34,
    }

    all_pass = True
    for desc, ok in checks.items():
        icon = "[PASS]" if ok else "[FAIL]"
        fprint(f"  {icon} {desc}")
        if not ok:
            all_pass = False

    fprint()
    if all_pass:
        fprint("  [PASS] Pure MCTS logic is correct. UCB + backup work properly.")
    else:
        fprint("  [FAIL] MCTS cannot beat RandomAgent!")
        fprint("         Check: _backup sign flip, _select_child UCB formula,")
        fprint("         terminal value assignment in MCTS evaluation.")

    fprint("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
