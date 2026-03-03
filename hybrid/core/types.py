# -*- coding: utf-8 -*-
"""Core type definitions for the hybrid chess engine."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple


class Side(Enum):
    """Side: Chess vs Xiangqi."""

    CHESS = auto()
    XIANGQI = auto()

    def opponent(self) -> "Side":
        return Side.XIANGQI if self == Side.CHESS else Side.CHESS


class PieceKind(Enum):
    """Unified piece-kind enum for both Chess and Xiangqi pieces."""

    # --- Chess side ---
    KING = auto()
    QUEEN = auto()
    ROOK = auto()
    BISHOP = auto()
    KNIGHT = auto()
    PAWN = auto()

    # --- Xiangqi side ---
    GENERAL = auto()
    ADVISOR = auto()
    ELEPHANT = auto()
    HORSE = auto()
    CHARIOT = auto()
    CANNON = auto()
    SOLDIER = auto()


@dataclass(frozen=True)
class Piece:
    """A piece = (kind, side)."""

    kind: PieceKind
    side: Side


@dataclass(frozen=True)
class Move:
    """A single move (action).

    promotion: Only for Chess Pawn promotion at y=9.
    """

    fx: int
    fy: int
    tx: int
    ty: int
    promotion: Optional[PieceKind] = None

    def from_sq(self) -> Tuple[int, int]:
        return (self.fx, self.fy)

    def to_sq(self) -> Tuple[int, int]:
        return (self.tx, self.ty)
