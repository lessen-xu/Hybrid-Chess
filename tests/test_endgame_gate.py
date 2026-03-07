# -*- coding: utf-8 -*-
"""Endgame conversion gate tests for AB_D4 (C++ engine).

Hand-crafted positions with clear material advantage.
Gate criteria:
  - ≥ 80% conversion rate per role
  - ≤ 20% max-plies/repetition draws
"""

import pytest
from hybrid.core.board import Board
from hybrid.core.types import Piece, Side, PieceKind
from hybrid.core.env import HybridChessEnv, GameState

# Use C++ AB agent for speed
from scripts.eval_arena import _CppABAgent, play_arena_game
from hybrid.agents.base import Agent

DEPTH = 4
N_TRIALS = 10
CONVERSION_THRESHOLD = 0.80
MAX_DRAW_THRESHOLD = 0.20


def _board(*pieces):
    """Create a board from (x, y, kind, side) tuples."""
    b = Board.empty()
    for x, y, kind, side in pieces:
        b.set(x, y, Piece(kind, side))
    return b


def _play_from_position(board, side_to_move, max_ply=200):
    """Play from a given position with C++ AB_D4 on both sides until terminal.

    Returns: (winner: Side|None, reason: str, plies: int)
    """
    agent = _CppABAgent(depth=DEPTH)
    env = HybridChessEnv(max_plies=max_ply, use_cpp=True)
    env.reset()
    # Properly re-init with custom board via the env's C++ sync method
    env.state = GameState(board=board.clone(), side_to_move=side_to_move,
                          ply=0, repetition={})
    env._init_cpp_state(board, side_to_move)

    state = env.state
    while True:
        legal = env.legal_moves()
        if not legal:
            break
        mv = agent.select_move(state, legal)
        state, _, done, info = env.step(mv)
        if done:
            break

    winner = None
    if info.winner == Side.CHESS:
        winner = Side.CHESS
    elif info.winner == Side.XIANGQI:
        winner = Side.XIANGQI

    return winner, info.reason, state.ply


# ══════════════════════════════════════════════════════════════
# Chess-side advantage positions (Chess should win)
# ══════════════════════════════════════════════════════════════

CHESS_ADVANTAGE_POSITIONS = [
    {
        "name": "KQ vs General",
        "board": _board(
            (4, 0, PieceKind.KING, Side.CHESS),
            (2, 5, PieceKind.QUEEN, Side.CHESS),
            (4, 9, PieceKind.GENERAL, Side.XIANGQI),
        ),
        "side": Side.CHESS,
        "expected_winner": Side.CHESS,
    },
    {
        "name": "KRR vs General",
        "board": _board(
            (4, 0, PieceKind.KING, Side.CHESS),
            (0, 4, PieceKind.ROOK, Side.CHESS),
            (8, 4, PieceKind.ROOK, Side.CHESS),
            (4, 9, PieceKind.GENERAL, Side.XIANGQI),
        ),
        "side": Side.CHESS,
        "expected_winner": Side.CHESS,
    },
    {
        "name": "KR+B vs General",
        "board": _board(
            (4, 0, PieceKind.KING, Side.CHESS),
            (0, 5, PieceKind.ROOK, Side.CHESS),
            (6, 3, PieceKind.BISHOP, Side.CHESS),
            (4, 9, PieceKind.GENERAL, Side.XIANGQI),
        ),
        "side": Side.CHESS,
        "expected_winner": Side.CHESS,
    },
    {
        "name": "KQ+N vs General+Advisor",
        "board": _board(
            (4, 0, PieceKind.KING, Side.CHESS),
            (3, 5, PieceKind.QUEEN, Side.CHESS),
            (6, 4, PieceKind.KNIGHT, Side.CHESS),
            (4, 9, PieceKind.GENERAL, Side.XIANGQI),
            (3, 8, PieceKind.ADVISOR, Side.XIANGQI),
        ),
        "side": Side.CHESS,
        "expected_winner": Side.CHESS,
    },
    {
        "name": "KRR+N vs General+Chariot",
        "board": _board(
            (4, 0, PieceKind.KING, Side.CHESS),
            (0, 3, PieceKind.ROOK, Side.CHESS),
            (8, 3, PieceKind.ROOK, Side.CHESS),
            (6, 2, PieceKind.KNIGHT, Side.CHESS),
            (4, 9, PieceKind.GENERAL, Side.XIANGQI),
            (0, 9, PieceKind.CHARIOT, Side.XIANGQI),
        ),
        "side": Side.CHESS,
        "expected_winner": Side.CHESS,
    },
    {
        "name": "KQ+R vs General+Horse",
        "board": _board(
            (4, 0, PieceKind.KING, Side.CHESS),
            (3, 4, PieceKind.QUEEN, Side.CHESS),
            (7, 3, PieceKind.ROOK, Side.CHESS),
            (4, 9, PieceKind.GENERAL, Side.XIANGQI),
            (2, 8, PieceKind.HORSE, Side.XIANGQI),
        ),
        "side": Side.CHESS,
        "expected_winner": Side.CHESS,
    },
]


# ══════════════════════════════════════════════════════════════
# Xiangqi-side advantage positions (Xiangqi should win)
# ══════════════════════════════════════════════════════════════

XIANGQI_ADVANTAGE_POSITIONS = [
    {
        "name": "General+2Chariots vs King",
        "board": _board(
            (4, 0, PieceKind.KING, Side.CHESS),
            (4, 9, PieceKind.GENERAL, Side.XIANGQI),
            (0, 5, PieceKind.CHARIOT, Side.XIANGQI),
            (8, 5, PieceKind.CHARIOT, Side.XIANGQI),
        ),
        "side": Side.XIANGQI,
        "expected_winner": Side.XIANGQI,
    },
    {
        "name": "General+Chariot+Cannon vs King",
        "board": _board(
            (4, 0, PieceKind.KING, Side.CHESS),
            (4, 9, PieceKind.GENERAL, Side.XIANGQI),
            (0, 4, PieceKind.CHARIOT, Side.XIANGQI),
            (4, 6, PieceKind.CANNON, Side.XIANGQI),
        ),
        "side": Side.XIANGQI,
        "expected_winner": Side.XIANGQI,
    },
    {
        "name": "General+2Chariots+Horse vs King+Pawn",
        "board": _board(
            (4, 0, PieceKind.KING, Side.CHESS),
            (3, 1, PieceKind.PAWN, Side.CHESS),
            (4, 9, PieceKind.GENERAL, Side.XIANGQI),
            (0, 5, PieceKind.CHARIOT, Side.XIANGQI),
            (8, 5, PieceKind.CHARIOT, Side.XIANGQI),
            (6, 7, PieceKind.HORSE, Side.XIANGQI),
        ),
        "side": Side.XIANGQI,
        "expected_winner": Side.XIANGQI,
    },
    {
        "name": "General+2Chariots+Cannon vs King+Bishop",
        "board": _board(
            (4, 0, PieceKind.KING, Side.CHESS),
            (2, 2, PieceKind.BISHOP, Side.CHESS),
            (4, 9, PieceKind.GENERAL, Side.XIANGQI),
            (0, 4, PieceKind.CHARIOT, Side.XIANGQI),
            (8, 4, PieceKind.CHARIOT, Side.XIANGQI),
            (4, 7, PieceKind.CANNON, Side.XIANGQI),
        ),
        "side": Side.XIANGQI,
        "expected_winner": Side.XIANGQI,
    },
    {
        "name": "General+Chariot+2Horses vs King",
        "board": _board(
            (4, 0, PieceKind.KING, Side.CHESS),
            (4, 9, PieceKind.GENERAL, Side.XIANGQI),
            (0, 5, PieceKind.CHARIOT, Side.XIANGQI),
            (2, 7, PieceKind.HORSE, Side.XIANGQI),
            (6, 7, PieceKind.HORSE, Side.XIANGQI),
        ),
        "side": Side.XIANGQI,
        "expected_winner": Side.XIANGQI,
    },
    {
        "name": "General+2Chariots vs King+Knight",
        "board": _board(
            (4, 0, PieceKind.KING, Side.CHESS),
            (6, 2, PieceKind.KNIGHT, Side.CHESS),
            (4, 9, PieceKind.GENERAL, Side.XIANGQI),
            (0, 5, PieceKind.CHARIOT, Side.XIANGQI),
            (8, 5, PieceKind.CHARIOT, Side.XIANGQI),
        ),
        "side": Side.XIANGQI,
        "expected_winner": Side.XIANGQI,
    },
]


# ══════════════════════════════════════════════════════════════
# Aggregate tests
# ══════════════════════════════════════════════════════════════

def _run_single(args):
    """Worker: play one position once. Returns (pos_idx, winner, reason, plies)."""
    pos_idx, board_pieces, side, max_ply = args
    # Reconstruct board in subprocess
    b = Board.empty()
    for x, y, kind, side_p in board_pieces:
        b.set(x, y, Piece(kind, side_p))
    winner, reason, plies = _play_from_position(b, side, max_ply)
    return pos_idx, winner, reason, plies


def _run_gate(positions):
    """Run N_TRIALS per position IN PARALLEL, return (conversion_rate, draw_rate, details)."""
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Serialize board as piece tuples for subprocess pickling
    work_items = []
    for pos_idx, pos in enumerate(positions):
        board_pieces = [(x, y, p.kind, p.side) for x, y, p in pos["board"].iter_pieces()]
        for trial in range(N_TRIALS):
            work_items.append((pos_idx, board_pieces, pos["side"], 200))

    # Run all trials in parallel
    results_by_pos = {i: [] for i in range(len(positions))}
    n_workers = min(os.cpu_count() or 4, len(work_items))

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_single, item): item for item in work_items}
        for future in as_completed(futures):
            pos_idx, winner, reason, plies = future.result()
            results_by_pos[pos_idx].append((winner, reason, plies))

    total_wins = 0
    total_draws = 0
    total = 0
    details = []

    for pos_idx, pos in enumerate(positions):
        wins = sum(1 for w, _, _ in results_by_pos[pos_idx] if w == pos["expected_winner"])
        draws = sum(1 for w, _, _ in results_by_pos[pos_idx] if w is None)
        total_wins += wins
        total_draws += draws
        total += len(results_by_pos[pos_idx])
        details.append({
            "name": pos["name"],
            "wins": wins,
            "draws": N_TRIALS - wins,
            "conversion": wins / N_TRIALS,
        })

    return total_wins / total, total_draws / total, details


class TestEndgameGate:
    """AB_D4 must convert ≥80% of clear-advantage endgames."""

    @pytest.mark.slow
    def test_chess_side_conversion(self):
        """Chess-side advantage: AB_D4 should win ≥80% as Chess."""
        conv_rate, draw_rate, details = _run_gate(CHESS_ADVANTAGE_POSITIONS)

        print(f"\n{'='*60}")
        print(f"  Chess-side advantage conversion: {conv_rate:.1%}")
        print(f"  Draw rate: {draw_rate:.1%}")
        for d in details:
            print(f"    {d['name']}: {d['conversion']:.0%} ({d['wins']}/{N_TRIALS})")
        print(f"{'='*60}")

        assert conv_rate >= CONVERSION_THRESHOLD, \
            f"Chess conversion {conv_rate:.1%} < {CONVERSION_THRESHOLD:.0%} gate"
        assert draw_rate <= MAX_DRAW_THRESHOLD, \
            f"Draw rate {draw_rate:.1%} > {MAX_DRAW_THRESHOLD:.0%} gate"

    @pytest.mark.slow
    def test_xiangqi_side_conversion(self):
        """Xiangqi-side advantage: AB_D4 should win ≥80% as Xiangqi."""
        conv_rate, draw_rate, details = _run_gate(XIANGQI_ADVANTAGE_POSITIONS)

        print(f"\n{'='*60}")
        print(f"  Xiangqi-side advantage conversion: {conv_rate:.1%}")
        print(f"  Draw rate: {draw_rate:.1%}")
        for d in details:
            print(f"    {d['name']}: {d['conversion']:.0%} ({d['wins']}/{N_TRIALS})")
        print(f"{'='*60}")

        assert conv_rate >= CONVERSION_THRESHOLD, \
            f"Xiangqi conversion {conv_rate:.1%} < {CONVERSION_THRESHOLD:.0%} gate"
        assert draw_rate <= MAX_DRAW_THRESHOLD, \
            f"Draw rate {draw_rate:.1%} > {MAX_DRAW_THRESHOLD:.0%} gate"
