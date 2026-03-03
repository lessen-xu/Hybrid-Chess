# -*- coding: utf-8 -*-
"""ASCII board renderer. Chess pieces = uppercase, Xiangqi = lowercase."""

from __future__ import annotations
from typing import Optional
from .board import Board
from .types import Piece, Side, PieceKind
from .config import BOARD_W, BOARD_H


_CHESS_MAP = {
    PieceKind.KING: "K",
    PieceKind.QUEEN: "Q",
    PieceKind.ROOK: "R",
    PieceKind.BISHOP: "B",
    PieceKind.KNIGHT: "N",
    PieceKind.PAWN: "P",
}

_XIANGQI_MAP = {
    PieceKind.GENERAL: "g",
    PieceKind.ADVISOR: "a",
    PieceKind.ELEPHANT: "e",
    PieceKind.HORSE: "h",
    PieceKind.CHARIOT: "c",
    PieceKind.CANNON: "n",   # 'n' because 'c' is taken by chariot
    PieceKind.SOLDIER: "s",
}


def piece_to_char(p: Piece) -> str:
    if p.side == Side.CHESS:
        return _CHESS_MAP[p.kind]
    else:
        return _XIANGQI_MAP[p.kind]


def render_board(board: Board) -> str:
    lines = []
    for y in reversed(range(BOARD_H)):
        row = []
        for x in range(BOARD_W):
            p = board.get(x, y)
            row.append(piece_to_char(p) if p else ".")
        lines.append(f"{y+1:>2} " + " ".join(row))
    lines.append("    " + " ".join(list("abcdefghi")))
    return "\n".join(lines)
