# -*- coding: utf-8 -*-
"""Feature extraction for TD-learning: V_theta(s) = theta · phi(s)."""

from __future__ import annotations
from typing import Dict, List
import numpy as np

from hybrid.core.env import GameState
from hybrid.core.types import Side, PieceKind
from hybrid.core.rules import generate_legal_moves


# Feature order: one count-diff feature per piece kind (excl. KING/GENERAL)
PIECE_FEATURES: List[PieceKind] = [
    # Chess
    PieceKind.QUEEN, PieceKind.ROOK, PieceKind.BISHOP, PieceKind.KNIGHT, PieceKind.PAWN,
    # Xiangqi
    PieceKind.ADVISOR, PieceKind.ELEPHANT, PieceKind.HORSE, PieceKind.CHARIOT, PieceKind.CANNON, PieceKind.SOLDIER,
]


def extract_features(state: GameState, perspective: Side) -> np.ndarray:
    """Extract feature vector phi(s) from perspective's viewpoint. Returns shape (D,)."""
    counts = {k: 0 for k in PIECE_FEATURES}
    for _, _, p in state.board.iter_pieces():
        if p.kind in counts:
            counts[p.kind] += 1 if p.side == perspective else -1

    feat = [float(counts[k]) / 10.0 for k in PIECE_FEATURES]

    # Mobility difference (normalized)
    my_moves = len(generate_legal_moves(state.board, perspective))
    op_moves = len(generate_legal_moves(state.board, perspective.opponent()))
    feat.append(float(my_moves - op_moves) / 50.0)

    return np.array(feat, dtype=np.float32)


def feature_dim() -> int:
    return len(PIECE_FEATURES) + 1
