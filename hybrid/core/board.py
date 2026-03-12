# -*- coding: utf-8 -*-
"""Board data structure: 9x10 grid stored as grid[y][x]."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Iterable, Tuple

from .config import BOARD_W, BOARD_H
from .types import Piece, Side, PieceKind


@dataclass
class Board:
    """The board object."""

    grid: List[List[Optional[Piece]]]

    @staticmethod
    def empty() -> "Board":
        return Board([[None for _ in range(BOARD_W)] for _ in range(BOARD_H)])

    def clone(self) -> "Board":
        # Piece is frozen; shallow copy suffices.
        return Board([row[:] for row in self.grid])

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < BOARD_W and 0 <= y < BOARD_H

    def get(self, x: int, y: int) -> Optional[Piece]:
        if not self.in_bounds(x, y):
            return None
        return self.grid[y][x]

    def set(self, x: int, y: int, piece: Optional[Piece]) -> None:
        assert self.in_bounds(x, y)
        self.grid[y][x] = piece

    def move_piece(self, fx: int, fy: int, tx: int, ty: int) -> Optional[Piece]:
        """Move a piece and return the captured piece (if any).

        Only performs board-level movement; legality is checked by rules.
        """
        p = self.get(fx, fy)
        assert p is not None
        captured = self.get(tx, ty)
        self.set(tx, ty, p)
        self.set(fx, fy, None)
        return captured

    def iter_pieces(self) -> Iterable[Tuple[int, int, Piece]]:
        """Iterate over all pieces on the board."""
        for y in range(BOARD_H):
            for x in range(BOARD_W):
                p = self.grid[y][x]
                if p is not None:
                    yield x, y, p


def initial_board(variant: "VariantConfig | None" = None) -> Board:
    """Create the initial hybrid-chess layout.

    y=0/1: Chess back rank and pawn rank.
    y=9: Xiangqi back rank; y=7: cannons; y=6: soldiers.

    Args:
        variant: Game variant configuration. If None, reads from legacy
                 global flags in config.py (backwards compatible).
    """
    if variant is not None:
        # ---- New path: read from VariantConfig ----
        v = variant
    else:
        # ---- Legacy path: read from global flags ----
        from .config import (CHESS_EXTRA_PAWN_ON_I_FILE,
                              ABLATION_NO_QUEEN, ABLATION_EXTRA_CANNON,
                              ABLATION_REMOVE_EXTRA_PAWN,
                              ABLATION_CHESS_NO_BISHOP, ABLATION_XIANGQI_EXTRA_SOLDIER,
                              ABLATION_CHESS_ONE_ROOK)
        from .config import VariantConfig
        v = VariantConfig(
            extra_pawn_i_file=CHESS_EXTRA_PAWN_ON_I_FILE,
            no_queen=ABLATION_NO_QUEEN,
            no_bishop=ABLATION_CHESS_NO_BISHOP,
            one_rook=ABLATION_CHESS_ONE_ROOK,
            remove_extra_pawn=ABLATION_REMOVE_EXTRA_PAWN,
            extra_cannon=ABLATION_EXTRA_CANNON,
            extra_soldier=ABLATION_XIANGQI_EXTRA_SOLDIER,
        )

    b = Board.empty()

    # --- Chess side (bottom) ---
    chess_back = [
        PieceKind.ROOK,
        PieceKind.KNIGHT,
        None if v.no_bishop else PieceKind.BISHOP,
        None if v.no_queen else PieceKind.QUEEN,
        PieceKind.KING,
        PieceKind.BISHOP,
        PieceKind.KNIGHT,
        None if v.one_rook else PieceKind.ROOK,
        None,  # 9th file empty by default
    ]
    for x, kind in enumerate(chess_back):
        if kind is not None:
            b.set(x, 0, Piece(kind, Side.CHESS))

    # Pawn rank: 8 pawns + optional 9th
    for x in range(8):
        b.set(x, 1, Piece(PieceKind.PAWN, Side.CHESS))
    extra_pawn = v.extra_pawn_i_file and not v.remove_extra_pawn
    if extra_pawn:
        b.set(8, 1, Piece(PieceKind.PAWN, Side.CHESS))

    # --- Xiangqi side (top) ---
    xiangqi_back = [
        PieceKind.CHARIOT,
        PieceKind.HORSE,
        PieceKind.ELEPHANT,
        PieceKind.ADVISOR,
        PieceKind.GENERAL,
        PieceKind.ADVISOR,
        PieceKind.ELEPHANT,
        PieceKind.HORSE,
        PieceKind.CHARIOT,
    ]
    for x, kind in enumerate(xiangqi_back):
        b.set(x, 9, Piece(kind, Side.XIANGQI))

    b.set(1, 7, Piece(PieceKind.CANNON, Side.XIANGQI))
    b.set(7, 7, Piece(PieceKind.CANNON, Side.XIANGQI))
    if v.extra_cannon:
        b.set(4, 7, Piece(PieceKind.CANNON, Side.XIANGQI))

    for x in [0, 2, 4, 6, 8]:
        b.set(x, 6, Piece(PieceKind.SOLDIER, Side.XIANGQI))
    if v.extra_soldier:
        b.set(4, 5, Piece(PieceKind.SOLDIER, Side.XIANGQI))

    return b

