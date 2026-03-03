# -*- coding: utf-8 -*-
"""
Dual-engine differential fuzzing: Python engine vs C++ engine.

Runs random games through both engines simultaneously, comparing
generate_legal_moves output at every position.

Usage:
    python tests/fuzz_dual_engine.py [--games 5000] [--verbose]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

# ── Python engine ──
from hybrid.core.board import Board as PyBoard, initial_board
from hybrid.core.types import Side as PySide, PieceKind as PyPieceKind, Move as PyMove
from hybrid.core.rules import (
    generate_legal_moves as py_gen_legal,
    apply_move as py_apply_move,
    terminal_info as py_terminal_info,
    board_hash as py_board_hash,
)
from hybrid.core.config import MAX_PLIES

# ── C++ engine ──
from hybrid.cpp_engine import (
    Side as CppSide,
    PieceKind as CppPieceKind,
    Piece as CppPiece,
    Move as CppMove,
    Board as CppBoard,
    generate_legal_moves as cpp_gen_legal,
    apply_move as cpp_apply_move,
    terminal_info as cpp_terminal_info,
)


# ═══════════════════════════════════════════════════════════════
# Type mapping helpers
# ═══════════════════════════════════════════════════════════════

_PY_TO_CPP_SIDE = {PySide.CHESS: CppSide.CHESS, PySide.XIANGQI: CppSide.XIANGQI}

_PY_TO_CPP_KIND = {
    PyPieceKind.KING: CppPieceKind.KING,
    PyPieceKind.QUEEN: CppPieceKind.QUEEN,
    PyPieceKind.ROOK: CppPieceKind.ROOK,
    PyPieceKind.BISHOP: CppPieceKind.BISHOP,
    PyPieceKind.KNIGHT: CppPieceKind.KNIGHT,
    PyPieceKind.PAWN: CppPieceKind.PAWN,
    PyPieceKind.GENERAL: CppPieceKind.GENERAL,
    PyPieceKind.ADVISOR: CppPieceKind.ADVISOR,
    PyPieceKind.ELEPHANT: CppPieceKind.ELEPHANT,
    PyPieceKind.HORSE: CppPieceKind.HORSE,
    PyPieceKind.CHARIOT: CppPieceKind.CHARIOT,
    PyPieceKind.CANNON: CppPieceKind.CANNON,
    PyPieceKind.SOLDIER: CppPieceKind.SOLDIER,
}

# Reverse mapping for promotion comparison
_CPP_KIND_TO_STR = {
    CppPieceKind.QUEEN: "QUEEN", CppPieceKind.ROOK: "ROOK",
    CppPieceKind.BISHOP: "BISHOP", CppPieceKind.KNIGHT: "KNIGHT",
    CppPieceKind.NONE: None,
}


def sync_board(py_board: PyBoard) -> CppBoard:
    """Create a C++ Board that mirrors the Python Board exactly."""
    cpp_board = CppBoard.empty()
    for x, y, p in py_board.iter_pieces():
        cpp_board.set(x, y, CppPiece(_PY_TO_CPP_KIND[p.kind], _PY_TO_CPP_SIDE[p.side]))
    return cpp_board


def move_to_tuple(m) -> Tuple:
    """Normalize any Move (Python or C++) to a comparable tuple."""
    # Python Move: m.fx, m.fy, m.tx, m.ty, m.promotion (PieceKind or None)
    # C++ Move:    m.fx, m.fy, m.tx, m.ty, m.promotion (PieceKind enum)
    promo = getattr(m, 'promotion', None)
    if promo is not None:
        # Python side: PieceKind enum or None
        if isinstance(promo, PyPieceKind):
            promo = promo.name  # e.g. "QUEEN"
        elif isinstance(promo, CppPieceKind):
            promo = _CPP_KIND_TO_STR.get(promo, str(promo))
        # else: already a string or None
    return (m.fx, m.fy, m.tx, m.ty, promo)


def convert_py_move_to_cpp(py_move: PyMove) -> CppMove:
    """Convert a Python Move to a C++ Move."""
    promo = CppPieceKind.NONE
    if py_move.promotion is not None:
        promo = _PY_TO_CPP_KIND[py_move.promotion]
    return CppMove(py_move.fx, py_move.fy, py_move.tx, py_move.ty, promo)


def board_ascii(py_board: PyBoard) -> str:
    """Simple ASCII rendering of a board for debug output."""
    kind_char = {
        PyPieceKind.KING: 'K', PyPieceKind.QUEEN: 'Q', PyPieceKind.ROOK: 'R',
        PyPieceKind.BISHOP: 'B', PyPieceKind.KNIGHT: 'N', PyPieceKind.PAWN: 'P',
        PyPieceKind.GENERAL: 'G', PyPieceKind.ADVISOR: 'A', PyPieceKind.ELEPHANT: 'E',
        PyPieceKind.HORSE: 'H', PyPieceKind.CHARIOT: 'C', PyPieceKind.CANNON: 'O',
        PyPieceKind.SOLDIER: 'S',
    }
    lines = []
    for y in range(9, -1, -1):
        row = []
        for x in range(9):
            p = py_board.get(x, y)
            if p is None:
                row.append('. ')
            else:
                c = kind_char.get(p.kind, '?')
                prefix = 'c' if p.side == PySide.CHESS else 'x'
                row.append(f"{prefix}{c}")
        lines.append(f"  {y} {'|'.join(row)}")
    lines.append("    " + " ".join(f"{x} " for x in range(9)))
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Core fuzzing loop
# ═══════════════════════════════════════════════════════════════

def fuzz_one_game(game_id: int, verbose: bool) -> Tuple[int, List[str]]:
    """Run one random game, return (plies, list_of_errors)."""
    random.seed(game_id)
    errors: List[str] = []

    # Initialize boards
    py_board = initial_board()
    cpp_board = sync_board(py_board)

    side = PySide.CHESS
    cpp_side = CppSide.CHESS
    ply = 0

    # Repetition tables (separate per engine, implicitly tests hash consistency)
    py_rep: Dict[str, int] = {}
    cpp_rep: Dict[str, int] = {}

    while ply < MAX_PLIES:
        # ── 1) Compare legal moves ──
        py_moves = py_gen_legal(py_board, side)
        cpp_moves = cpp_gen_legal(cpp_board, cpp_side)

        py_set = sorted(set(move_to_tuple(m) for m in py_moves))
        cpp_set = sorted(set(move_to_tuple(m) for m in cpp_moves))

        if py_set != cpp_set:
            py_only = sorted(set(py_set) - set(cpp_set))
            cpp_only = sorted(set(cpp_set) - set(py_set))
            err = (
                f"Game {game_id}, ply {ply}, side={side.name}: MOVE MISMATCH\n"
                f"  Python-only ({len(py_only)}): {py_only[:10]}{'...' if len(py_only) > 10 else ''}\n"
                f"  C++-only ({len(cpp_only)}): {cpp_only[:10]}{'...' if len(cpp_only) > 10 else ''}\n"
                f"  Board:\n{board_ascii(py_board)}"
            )
            errors.append(err)
            break  # Can't continue if moves differ

        # ── 2) Compare terminal info ──
        py_hash = py_board_hash(py_board, side)
        cpp_hash = cpp_board.board_hash(cpp_side)
        py_rep[py_hash] = py_rep.get(py_hash, 0) + 1
        cpp_rep[cpp_hash] = cpp_rep.get(cpp_hash, 0) + 1

        py_info = py_terminal_info(py_board, side, py_rep, ply, MAX_PLIES)
        cpp_info = cpp_terminal_info(cpp_board, cpp_side, cpp_rep, ply, MAX_PLIES)

        if py_info.status != cpp_info.status:
            err = (
                f"Game {game_id}, ply {ply}: TERMINAL MISMATCH\n"
                f"  Python: status={py_info.status}, reason={py_info.reason}\n"
                f"  C++:    status={cpp_info.status}, reason={cpp_info.reason}\n"
                f"  Board:\n{board_ascii(py_board)}"
            )
            errors.append(err)
            break

        # ── 3) Check if game is over ──
        if py_info.status != "ongoing":
            break

        if len(py_moves) == 0:
            break

        # ── 4) Pick a random move and apply to both engines ──
        py_move = random.choice(py_moves)
        cpp_move = convert_py_move_to_cpp(py_move)

        py_board = py_apply_move(py_board, py_move)
        cpp_board = cpp_apply_move(cpp_board, cpp_move)

        # Switch sides
        side = side.opponent()
        cpp_side = CppSide.XIANGQI if cpp_side == CppSide.CHESS else CppSide.CHESS
        ply += 1

    if verbose:
        status = "ERR" if errors else "OK"
        print(f"  Game {game_id:5d}: {ply:4d} plies, {status}")

    return ply, errors


def main():
    parser = argparse.ArgumentParser(description="Dual-engine differential fuzzing")
    parser.add_argument("--games", type=int, default=5000, help="Number of games")
    parser.add_argument("--verbose", action="store_true", help="Print per-game status")
    parser.add_argument("--game", type=int, default=None, help="Run a single specific game (for debugging)")
    args = parser.parse_args()

    if args.game is not None:
        # Debug mode: run a single game
        print(f"Running game {args.game} only...")
        plies, errors = fuzz_one_game(args.game, verbose=True)
        if errors:
            for e in errors:
                print(f"\n{e}")
            sys.exit(1)
        else:
            print(f"Game {args.game}: {plies} plies, OK")
            sys.exit(0)

    print(f"=== Dual-Engine Differential Fuzzing ===")
    print(f"Games: {args.games}")
    print()

    total_plies = 0
    total_errors: List[str] = []
    start = time.time()

    for game_id in range(args.games):
        plies, errors = fuzz_one_game(game_id, verbose=args.verbose)
        total_plies += plies
        total_errors.extend(errors)

        # Progress every 500 games
        if not args.verbose and (game_id + 1) % 500 == 0:
            elapsed = time.time() - start
            rate = (game_id + 1) / elapsed
            print(f"  {game_id + 1:5d}/{args.games} games | {total_plies:,} positions | "
                  f"{len(total_errors)} mismatches | {rate:.0f} games/s")

    elapsed = time.time() - start

    print()
    print("=== SUMMARY ===")
    print(f"Games:                   {args.games:,}")
    print(f"Total positions checked: {total_plies:,}")
    print(f"Mismatches:              {len(total_errors)}")
    print(f"Time:                    {elapsed:.1f}s ({args.games / elapsed:.0f} games/s)")
    print()

    if total_errors:
        print(f"=== ERRORS ({len(total_errors)}) ===")
        for i, e in enumerate(total_errors):
            print(f"\n--- Error {i+1} ---")
            print(e)
        sys.exit(1)
    else:
        print("ALL CLEAR")
        sys.exit(0)


if __name__ == "__main__":
    main()
