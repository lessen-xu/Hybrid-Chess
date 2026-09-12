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
    PieceKind.XQ_QUEEN: 9.0,
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
    """Material + mobility + check bonus + an endgame conversion set of bonuses.

    Guarantees strict zero-sum antisymmetry:
        evaluate(state, Side.CHESS) == -evaluate(state, Side.XIANGQI)

    Endgame heuristics only activate when total board pieces are sufficiently
    depleted (not at the opening, even if raw material scores differ) and one side
    holds a decisive material advantage (> 5.0).
    The previous anti-stalemate penalty (-8.0) is removed because under Hybrid Chess
    and Xiangqi rules, driving the opponent into stalemate is a win for the attacker.
    """
    mat = material_score(state, perspective)
    mob = w.mobility * mobility_score(state, perspective) if w.mobility != 0.0 else 0.0

    # Count pieces and locate royals for both armies
    chess_pieces = []
    xiangqi_pieces = []
    chess_royal = None
    xiangqi_royal = None

    for x, y, p in state.board.iter_pieces():
        if p.side == Side.CHESS:
            chess_pieces.append((x, y))
            if p.kind == PieceKind.KING:
                chess_royal = (x, y)
        else:
            xiangqi_pieces.append((x, y))
            if p.kind == PieceKind.GENERAL:
                xiangqi_royal = (x, y)

    total_pieces = len(chess_pieces) + len(xiangqi_pieces)
    mat_chess = material_score(state, Side.CHESS)
    abs_mat = abs(mat_chess)

    if mat_chess >= 0.0:
        attacker_side = Side.CHESS
        defender_side = Side.XIANGQI
        attacker_pieces = chess_pieces
        defender_pieces = xiangqi_pieces
        attacker_royal = chess_royal
        defender_royal = xiangqi_royal
    else:
        attacker_side = Side.XIANGQI
        defender_side = Side.CHESS
        attacker_pieces = xiangqi_pieces
        defender_pieces = chess_pieces
        attacker_royal = xiangqi_royal
        defender_royal = chess_royal

    defender_count = len(defender_pieces)

    # Endgame heuristics only fire when pieces are actually depleted,
    # preventing false triggers at ply 0 opening where material sums differ.
    is_endgame = (total_pieces <= 14) or (defender_count <= 6)
    winning_big = is_endgame and (abs_mat > 5.0)

    effective_check_bonus = 5.0 if winning_big else w.check_bonus
    chk = 0.0
    opp_side = perspective.opponent()
    if is_in_check(state.board, opp_side):
        chk += effective_check_bonus
    if is_in_check(state.board, perspective):
        chk -= effective_check_bonus

    conversion_bonus = 0.0
    if winning_big:
        # (1) Amplify material so converting is prioritized over shuffling.
        # Since base mat already includes 1.0 * abs_mat for the attacker,
        # adding 2.0 * abs_mat makes the effective material contribution 3.0 * abs_mat.
        conversion_bonus += 2.0 * abs_mat

        if defender_royal is not None:
            ekx, eky = defender_royal
            # (2) King confinement: reward pushing the defender royal toward the board's edge
            dx = abs(ekx - 4.0)
            dy = abs(eky - 4.5)
            conversion_bonus += 1.0 * (dx + dy)

            # (3) Piece approach: reward bringing attacker pieces close to defender royal
            for px, py in attacker_pieces:
                dist = max(abs(px - ekx), abs(py - eky))
                conversion_bonus += 0.8 * (10.0 - dist)

            # (4) Own king proximity: reward cooperation in mating positions
            if attacker_royal is not None:
                mkx, mky = attacker_royal
                king_dist = max(abs(mkx - ekx), abs(mky - eky))
                conversion_bonus += 0.5 * (10.0 - king_dist)

        if defender_count <= 6:
            # (5) Mobility squeeze: reward positions where the defender has few legal moves
            defender_moves = len(generate_legal_moves(state.board, defender_side))
            conversion_bonus += 0.3 * (30.0 - defender_moves)
            # Anti-stalemate penalty is intentionally omitted: stalemate is a win for the attacker.

    if perspective == attacker_side:
        return mat + mob + chk + conversion_bonus
    else:
        return mat + mob + chk - conversion_bonus



