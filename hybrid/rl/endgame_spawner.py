# -*- coding: utf-8 -*-
"""Endgame position generator for curriculum learning.

Generates sparse endgame boards where one side has a decisive advantage
and can mate within ~5–15 moves. Used to bootstrap the value network
with dense +1 / -1 labels.

Templates:
  Chess-advantage (Chess should mate):
    1. K + Q  vs  bare General
    2. K + R  vs  bare General
    3. K + 2R vs  bare General
  Xiangqi-advantage (Xiangqi should mate):
    4. General + 2 Chariots       vs  bare King
    5. General + Chariot + Horse   vs  bare King
    6. General + Chariot + Cannon + Soldier(screen)  vs  bare King
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import random

from hybrid.core.board import Board
from hybrid.core.types import Side, PieceKind, Piece
from hybrid.core.config import BOARD_W, BOARD_H
from hybrid.core.rules import generate_legal_moves, _find_royal


# ====================================================================
# Placement helpers
# ====================================================================

def _random_square(rng: random.Random) -> Tuple[int, int]:
    """Random square on the 9×10 board."""
    return rng.randint(0, BOARD_W - 1), rng.randint(0, BOARD_H - 1)


def _random_palace_xiangqi(rng: random.Random) -> Tuple[int, int]:
    """Random square inside the Xiangqi palace (x∈[3,5], y∈[7,9])."""
    return rng.randint(3, 5), rng.randint(7, 9)


def _random_chess_king_zone(rng: random.Random) -> Tuple[int, int]:
    """Random square in the bottom 3 rows for the Chess King."""
    return rng.randint(0, BOARD_W - 1), rng.randint(0, 2)


def _occupied_squares(board: Board) -> set:
    """Return set of (x, y) with a piece."""
    return {(x, y) for x, y, _ in board.iter_pieces()}


def _place_random(
    board: Board,
    piece: Piece,
    occupied: set,
    rng: random.Random,
    max_attempts: int = 200,
) -> Tuple[int, int]:
    """Place a piece on a random empty square. Returns (x, y)."""
    for _ in range(max_attempts):
        x, y = _random_square(rng)
        if (x, y) not in occupied:
            board.set(x, y, piece)
            occupied.add((x, y))
            return x, y
    raise RuntimeError("Could not find an empty square for piece placement")


# ====================================================================
# Templates
# ====================================================================

def _template_chess_kq_vs_general(rng: random.Random) -> Board:
    """Chess K + Q vs Xiangqi bare General."""
    b = Board.empty()
    occ: set = set()
    # Place Xiangqi General in its palace
    gx, gy = _random_palace_xiangqi(rng)
    b.set(gx, gy, Piece(PieceKind.GENERAL, Side.XIANGQI))
    occ.add((gx, gy))
    # Place Chess King in bottom zone
    for _ in range(200):
        kx, ky = _random_chess_king_zone(rng)
        if (kx, ky) not in occ:
            b.set(kx, ky, Piece(PieceKind.KING, Side.CHESS))
            occ.add((kx, ky))
            break
    # Place Queen anywhere
    _place_random(b, Piece(PieceKind.QUEEN, Side.CHESS), occ, rng)
    return b


def _template_chess_kr_vs_general(rng: random.Random) -> Board:
    """Chess K + R vs Xiangqi bare General."""
    b = Board.empty()
    occ: set = set()
    gx, gy = _random_palace_xiangqi(rng)
    b.set(gx, gy, Piece(PieceKind.GENERAL, Side.XIANGQI))
    occ.add((gx, gy))
    for _ in range(200):
        kx, ky = _random_chess_king_zone(rng)
        if (kx, ky) not in occ:
            b.set(kx, ky, Piece(PieceKind.KING, Side.CHESS))
            occ.add((kx, ky))
            break
    _place_random(b, Piece(PieceKind.ROOK, Side.CHESS), occ, rng)
    return b


def _template_chess_k2r_vs_general(rng: random.Random) -> Board:
    """Chess K + 2R vs Xiangqi bare General."""
    b = Board.empty()
    occ: set = set()
    gx, gy = _random_palace_xiangqi(rng)
    b.set(gx, gy, Piece(PieceKind.GENERAL, Side.XIANGQI))
    occ.add((gx, gy))
    for _ in range(200):
        kx, ky = _random_chess_king_zone(rng)
        if (kx, ky) not in occ:
            b.set(kx, ky, Piece(PieceKind.KING, Side.CHESS))
            occ.add((kx, ky))
            break
    _place_random(b, Piece(PieceKind.ROOK, Side.CHESS), occ, rng)
    _place_random(b, Piece(PieceKind.ROOK, Side.CHESS), occ, rng)
    return b


def _template_xiangqi_2chariots_vs_king(rng: random.Random) -> Board:
    """Xiangqi General + 2 Chariots vs Chess bare King."""
    b = Board.empty()
    occ: set = set()
    gx, gy = _random_palace_xiangqi(rng)
    b.set(gx, gy, Piece(PieceKind.GENERAL, Side.XIANGQI))
    occ.add((gx, gy))
    for _ in range(200):
        kx, ky = _random_chess_king_zone(rng)
        if (kx, ky) not in occ:
            b.set(kx, ky, Piece(PieceKind.KING, Side.CHESS))
            occ.add((kx, ky))
            break
    _place_random(b, Piece(PieceKind.CHARIOT, Side.XIANGQI), occ, rng)
    _place_random(b, Piece(PieceKind.CHARIOT, Side.XIANGQI), occ, rng)
    return b


def _template_xiangqi_chariot_horse_vs_king(rng: random.Random) -> Board:
    """Xiangqi General + Chariot + Horse vs Chess bare King."""
    b = Board.empty()
    occ: set = set()
    gx, gy = _random_palace_xiangqi(rng)
    b.set(gx, gy, Piece(PieceKind.GENERAL, Side.XIANGQI))
    occ.add((gx, gy))
    for _ in range(200):
        kx, ky = _random_chess_king_zone(rng)
        if (kx, ky) not in occ:
            b.set(kx, ky, Piece(PieceKind.KING, Side.CHESS))
            occ.add((kx, ky))
            break
    _place_random(b, Piece(PieceKind.CHARIOT, Side.XIANGQI), occ, rng)
    _place_random(b, Piece(PieceKind.HORSE, Side.XIANGQI), occ, rng)
    return b


def _template_xiangqi_chariot_cannon_soldier_vs_king(rng: random.Random) -> Board:
    """Xiangqi General + Chariot + Cannon + Soldier vs Chess bare King."""
    b = Board.empty()
    occ: set = set()
    gx, gy = _random_palace_xiangqi(rng)
    b.set(gx, gy, Piece(PieceKind.GENERAL, Side.XIANGQI))
    occ.add((gx, gy))
    for _ in range(200):
        kx, ky = _random_chess_king_zone(rng)
        if (kx, ky) not in occ:
            b.set(kx, ky, Piece(PieceKind.KING, Side.CHESS))
            occ.add((kx, ky))
            break
    _place_random(b, Piece(PieceKind.CHARIOT, Side.XIANGQI), occ, rng)
    _place_random(b, Piece(PieceKind.CANNON, Side.XIANGQI), occ, rng)
    _place_random(b, Piece(PieceKind.SOLDIER, Side.XIANGQI), occ, rng)
    return b


# All templates and which side has the advantage (moves first)
_TEMPLATES = [
    (_template_chess_kq_vs_general, Side.CHESS),
    (_template_chess_kr_vs_general, Side.CHESS),
    (_template_chess_k2r_vs_general, Side.CHESS),
    (_template_xiangqi_2chariots_vs_king, Side.XIANGQI),
    (_template_xiangqi_chariot_horse_vs_king, Side.XIANGQI),
    (_template_xiangqi_chariot_cannon_soldier_vs_king, Side.XIANGQI),
]


# ====================================================================
# Public API
# ====================================================================

def generate_endgame_board(
    rng: Optional[random.Random] = None,
    max_retries: int = 50,
) -> Tuple[Board, Side]:
    """Generate one random endgame board.

    Returns (board, side_to_move) where side_to_move is the advantaged side.
    Validates that:
      - Both royal pieces exist
      - The side-to-move has at least one legal move
      - The side-to-move is NOT already in a position where the opponent has
        no legal moves (to avoid trivial 0-step games)
    """
    if rng is None:
        rng = random.Random()

    for _ in range(max_retries):
        template_fn, advantage_side = rng.choice(_TEMPLATES)
        board = template_fn(rng)

        # Validate: both royals exist
        if _find_royal(board, Side.CHESS) is None:
            continue
        if _find_royal(board, Side.XIANGQI) is None:
            continue

        # Validate: side-to-move has legal moves
        legal = generate_legal_moves(board, advantage_side)
        if len(legal) == 0:
            continue

        # Validate: opponent also has legal moves (avoid instant game-over)
        opponent_legal = generate_legal_moves(board, advantage_side.opponent())
        if len(opponent_legal) == 0:
            continue

        return board, advantage_side

    raise RuntimeError(
        f"Could not generate a valid endgame position after {max_retries} retries"
    )
