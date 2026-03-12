# -*- coding: utf-8 -*-
"""FEN-like notation parser/serializer for Hybrid Chess.

Format: ``<rank9>/<rank8>/.../<rank0> <side>``

Each rank is described left-to-right (file 0→8). Pieces use single
uppercase letters for Chess and lowercase for Xiangqi:

  Chess:   K=King  Q=Queen  R=Rook  B=Bishop  N=Knight  P=Pawn
  Xiangqi: g=General a=Advisor e=Elephant h=Horse c=Chariot n=Cannon s=Soldier

Digits 1-9 represent consecutive empty squares. Side is ``c`` (Chess)
or ``x`` (Xiangqi).

Example (default starting position ranks 0 and 9)::

    RNBQKBNR./......... c
"""

from __future__ import annotations
from typing import Tuple

from .board import Board
from .config import BOARD_W, BOARD_H
from .types import Side, PieceKind, Piece


# ---- Piece ↔ Character mapping ----

_PIECE_TO_CHAR = {
    (PieceKind.KING, Side.CHESS): 'K',
    (PieceKind.QUEEN, Side.CHESS): 'Q',
    (PieceKind.ROOK, Side.CHESS): 'R',
    (PieceKind.BISHOP, Side.CHESS): 'B',
    (PieceKind.KNIGHT, Side.CHESS): 'N',
    (PieceKind.PAWN, Side.CHESS): 'P',
    (PieceKind.GENERAL, Side.XIANGQI): 'g',
    (PieceKind.ADVISOR, Side.XIANGQI): 'a',
    (PieceKind.ELEPHANT, Side.XIANGQI): 'e',
    (PieceKind.HORSE, Side.XIANGQI): 'h',
    (PieceKind.CHARIOT, Side.XIANGQI): 'c',
    (PieceKind.CANNON, Side.XIANGQI): 'n',
    (PieceKind.SOLDIER, Side.XIANGQI): 's',
}
_CHAR_TO_PIECE = {v: k for k, v in _PIECE_TO_CHAR.items()}


def board_to_fen(board: Board, side_to_move: Side) -> str:
    """Serialize a board position to FEN-like string.

    Ranks are listed top-to-bottom (y=9 → y=0) to match visual layout.
    """
    ranks = []
    for y in range(BOARD_H - 1, -1, -1):
        rank_str = ""
        empty = 0
        for x in range(BOARD_W):
            p = board.get(x, y)
            if p is None:
                empty += 1
            else:
                if empty > 0:
                    rank_str += str(empty)
                    empty = 0
                rank_str += _PIECE_TO_CHAR[(p.kind, p.side)]
        if empty > 0:
            rank_str += str(empty)
        ranks.append(rank_str)

    side_char = "c" if side_to_move == Side.CHESS else "x"
    return "/".join(ranks) + " " + side_char


def parse_fen(fen: str) -> Tuple[Board, Side]:
    """Parse a FEN-like string into a Board and side-to-move.

    Raises ValueError on malformed input.
    """
    parts = fen.strip().split()
    if len(parts) != 2:
        raise ValueError(f"FEN must have 2 parts (ranks + side), got {len(parts)}")

    rank_strs = parts[0].split("/")
    if len(rank_strs) != BOARD_H:
        raise ValueError(f"FEN must have {BOARD_H} ranks, got {len(rank_strs)}")

    side_char = parts[1].lower()
    if side_char == "c":
        side = Side.CHESS
    elif side_char == "x":
        side = Side.XIANGQI
    else:
        raise ValueError(f"Unknown side '{parts[1]}', expected 'c' or 'x'")

    board = Board.empty()
    for rank_idx, rank_str in enumerate(rank_strs):
        y = BOARD_H - 1 - rank_idx  # top rank first
        x = 0
        for ch in rank_str:
            if ch.isdigit():
                x += int(ch)
            elif ch in _CHAR_TO_PIECE:
                kind, p_side = _CHAR_TO_PIECE[ch]
                board.set(x, y, Piece(kind, p_side))
                x += 1
            else:
                raise ValueError(f"Unknown piece char '{ch}' in rank {rank_idx}")
        if x != BOARD_W:
            raise ValueError(
                f"Rank {rank_idx} has {x} squares, expected {BOARD_W}"
            )

    return board, side
