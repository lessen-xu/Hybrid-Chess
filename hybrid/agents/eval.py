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
    """Combined evaluation: material + mobility + check bonus + endgame heuristics V2."""
    mat = material_score(state, perspective)
    mob = w.mobility * mobility_score(state, perspective)

    # ── Endgame heuristics V2 (mirror C++ evaluate_leaf) ──
    winning_big = mat > 5.0

    # Check bonus — amplified when winning big
    effective_check_bonus = 5.0 if winning_big else w.check_bonus
    chk = 0.0
    opp_side = perspective.opponent()
    opp_in_check = is_in_check(state.board, opp_side)
    if opp_in_check:
        chk += effective_check_bonus
    if is_in_check(state.board, perspective):
        chk -= effective_check_bonus

    endgame_bonus = 0.0
    if winning_big:
        # Material amplification (3×)
        mat *= 3.0

        # Locate royals
        opp_royal = None
        my_royal = None
        royal_kind_opp = PieceKind.KING if opp_side == Side.CHESS else PieceKind.GENERAL
        royal_kind_my = PieceKind.KING if perspective == Side.CHESS else PieceKind.GENERAL

        opp_piece_count = 0
        my_pieces = []
        for x, y, p in state.board.iter_pieces():
            if p.side == opp_side:
                opp_piece_count += 1
                if p.kind == royal_kind_opp:
                    opp_royal = (x, y)
            else:
                my_pieces.append((x, y))
                if p.kind == royal_kind_my:
                    my_royal = (x, y)

        if opp_royal is not None:
            ekx, eky = opp_royal
            # King confinement
            dx = abs(ekx - 4.0)
            dy = abs(eky - 4.5)
            endgame_bonus += 1.0 * (dx + dy)

            # Approach bonus: all our pieces close to enemy king
            for px, py in my_pieces:
                dist = max(abs(px - ekx), abs(py - eky))  # Chebyshev
                endgame_bonus += 0.8 * (10.0 - dist)

            # Own king proximity
            if my_royal is not None:
                mkx, mky = my_royal
                king_dist = max(abs(mkx - ekx), abs(mky - eky))
                endgame_bonus += 0.5 * (10.0 - king_dist)

        # Mobility squeeze
        if opp_piece_count <= 6:
            opp_moves = len(generate_legal_moves(state.board, opp_side))
            endgame_bonus += 0.3 * (30.0 - opp_moves)

            # Anti-stalemate
            if opp_moves <= 1 and not opp_in_check:
                endgame_bonus -= 8.0

    return mat + mob + chk + endgame_bonus


