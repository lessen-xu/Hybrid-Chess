# -*- coding: utf-8 -*-
"""Hand-crafted evaluation function for the AlphaBeta baseline.

Returns a score from the perspective of `perspective_side`:
  > 0  means favorable for that side,
  < 0  means unfavorable.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

from hybrid.core.types import Side, PieceKind
from hybrid.core.rules import generate_legal_moves, is_in_check
from hybrid.core.env import GameState


# Piece values (rough baseline; TD/RL will learn better weights)
PIECE_VALUES: Dict[PieceKind, float] = {
    # Chess
    PieceKind.KING: 0.0,      # King value handled by terminal conditions
    PieceKind.QUEEN: 9.0,
    PieceKind.ROOK: 5.0,
    PieceKind.BISHOP: 3.0,
    PieceKind.KNIGHT: 3.0,
    PieceKind.PAWN: 1.0,
    # Xiangqi
    PieceKind.GENERAL: 0.0,
    PieceKind.ADVISOR: 2.0,
    PieceKind.ELEPHANT: 2.0,
    PieceKind.HORSE: 4.0,
    PieceKind.CHARIOT: 9.0,
    PieceKind.CANNON: 5.0,
    PieceKind.SOLDIER: 1.0,
}


@dataclass
class EvalWeights:
    mobility: float = 0.05   # Mobility weight (much smaller than material)
    check_bonus: float = 0.3 # Small bonus for giving check


def material_score(state: GameState, perspective: Side) -> float:
    """Material difference: perspective's material minus opponent's."""
    s = 0.0
    for _, _, p in state.board.iter_pieces():
        v = PIECE_VALUES[p.kind]
        s += v if p.side == perspective else -v
    return s


def mobility_score(state: GameState, perspective: Side) -> float:
    """Mobility difference: number of legal moves (perspective minus opponent)."""
    my_moves = len(generate_legal_moves(state.board, perspective))
    op_moves = len(generate_legal_moves(state.board, perspective.opponent()))
    return float(my_moves - op_moves)


def evaluate(state: GameState, perspective: Side, w: EvalWeights = EvalWeights()) -> float:
    """Combined evaluation: material + mobility + check bonus."""
    val = 0.0
    val += material_score(state, perspective)
    val += w.mobility * mobility_score(state, perspective)

    if is_in_check(state.board, perspective.opponent()):
        val += w.check_bonus
    if is_in_check(state.board, perspective):
        val -= w.check_bonus

    return val
