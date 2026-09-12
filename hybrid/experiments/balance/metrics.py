"""Tier 0 static balance metrics.

Zero-cost structural and geometric properties computed directly from the
VariantConfig and initial board state without playing games:
- Initial legal moves per side (opening mobility)
- Royal mobility / escape squares
- Knight/Horse leg block obstruction ratio
- Material asymmetry
- Long-range piece projection capacity
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any

from hybrid.core.types import Side, PieceKind
from hybrid.core.board import initial_board, Board
from hybrid.core.config import VariantConfig
from hybrid.core.rules import generate_legal_moves, _active_variant, is_in_check
import hybrid.core.rules as rules_module


@dataclass
class Tier0Metrics:
    """Zero-cost structural metrics for a rule variant."""
    variant_name: str
    chess_initial_moves: int
    xiangqi_initial_moves: int
    mobility_ratio: float  # chess_moves / max(1, xiangqi_moves)
    chess_royal_escapes: int
    xiangqi_royal_escapes: int
    knight_block_obstruction_ratio: float  # fraction of knight moves blocked by initial pieces
    chess_material_pawn_units: float
    xiangqi_material_pawn_units: float
    material_ratio: float  # chess / xiangqi
    palace_restricted_royals: int  # 0, 1, or 2 royals confined to palace

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Approximate standard heuristic pawn-unit values for initial material comparison
PIECE_VALUES = {
    PieceKind.PAWN: 1.0,
    PieceKind.KNIGHT: 3.0,
    PieceKind.BISHOP: 3.0,
    PieceKind.ROOK: 5.0,
    PieceKind.QUEEN: 9.0,
    PieceKind.KING: 0.0,
    # Xiangqi
    PieceKind.SOLDIER: 1.0,
    PieceKind.ADVISOR: 2.0,
    PieceKind.ELEPHANT: 2.0,
    PieceKind.HORSE: 4.0,
    PieceKind.CHARIOT: 9.0,
    PieceKind.CANNON: 4.5,
    PieceKind.GENERAL: 0.0,
    PieceKind.XQ_QUEEN: 9.0,
}


def compute_tier0_metrics(variant: VariantConfig, name: str = "") -> Tier0Metrics:
    """Compute Tier 0 static metrics for a given VariantConfig."""
    old_active = rules_module._active_variant
    try:
        rules_module._active_variant = variant
        b = initial_board(variant)

        # 1. Initial legal moves per side
        chess_moves = len(generate_legal_moves(b, Side.CHESS))
        xq_moves = len(generate_legal_moves(b, Side.XIANGQI))
        mobility_ratio = float(chess_moves) / max(1.0, float(xq_moves))

        # 2. Royal mobility in initial position
        chess_royal_escapes = 0
        xq_royal_escapes = 0
        for m in generate_legal_moves(b, Side.CHESS):
            p = b.get(m.fx, m.fy)
            if p and p.kind == PieceKind.KING:
                chess_royal_escapes += 1
        for m in generate_legal_moves(b, Side.XIANGQI):
            p = b.get(m.fx, m.fy)
            if p and p.kind == PieceKind.GENERAL:
                xq_royal_escapes += 1

        # 3. Knight block obstruction ratio:
        # Check all knights/horses on board. Compare unblocked moves (8 per knight) to actual pseudo-moves.
        total_potential_knight_moves = 0
        actual_knight_moves = 0
        for x, y, p in b.iter_pieces():
            if p.kind in (PieceKind.KNIGHT, PieceKind.HORSE):
                total_potential_knight_moves += 8
                # Count moves generated for this piece
                moves = [m for m in generate_legal_moves(b, p.side) if m.fx == x and m.fy == y]
                actual_knight_moves += len(moves)

        if total_potential_knight_moves > 0:
            obstruction_ratio = 1.0 - (actual_knight_moves / total_potential_knight_moves)
        else:
            obstruction_ratio = 0.0

        # 4. Material sum in pawn units
        chess_mat = sum(PIECE_VALUES.get(p.kind, 0.0) for _, _, p in b.iter_pieces() if p.side == Side.CHESS)
        xq_mat = sum(PIECE_VALUES.get(p.kind, 0.0) for _, _, p in b.iter_pieces() if p.side == Side.XIANGQI)
        material_ratio = chess_mat / max(0.1, xq_mat)

        # 5. Palace restriction count
        palace_royals = 1  # Xiangqi General is always palace restricted
        if variant.chess_palace:
            palace_royals += 1

        return Tier0Metrics(
            variant_name=name or "custom",
            chess_initial_moves=chess_moves,
            xiangqi_initial_moves=xq_moves,
            mobility_ratio=round(mobility_ratio, 4),
            chess_royal_escapes=chess_royal_escapes,
            xiangqi_royal_escapes=xq_royal_escapes,
            knight_block_obstruction_ratio=round(obstruction_ratio, 4),
            chess_material_pawn_units=round(chess_mat, 2),
            xiangqi_material_pawn_units=round(xq_mat, 2),
            material_ratio=round(material_ratio, 4),
            palace_restricted_royals=palace_royals,
        )
    finally:
        rules_module._active_variant = old_active
