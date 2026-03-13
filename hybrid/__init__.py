"""Hybrid Chess — Asymmetric board game engine.

International Chess vs Chinese Chess on a shared 9×10 board,
with a full AlphaZero RL training pipeline.

Quick start:
    from hybrid import HybridChessEnv, Side, Move

    env = HybridChessEnv()
    state = env.reset()
    moves = env.legal_moves()
    state, reward, done, info = env.step(moves[0])
"""

__version__ = "0.1.0"

from hybrid.core.env import HybridChessEnv, GameState
from hybrid.core.types import Side, Move, PieceKind, Piece

__all__ = [
    "__version__",
    "HybridChessEnv",
    "GameState",
    "Side",
    "Move",
    "PieceKind",
    "Piece",
]
