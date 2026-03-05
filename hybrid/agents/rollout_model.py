# -*- coding: utf-8 -*-
"""Random-rollout policy-value model for Pure MCTS.

Implements the PolicyValueModel interface WITHOUT any neural network:
  - Policy: uniform distribution over legal moves.
  - Value:  random rollout (up to `rollout_steps` moves), then material-based
            truncation evaluation mapped to [-1, 1] via tanh.

Uses the C++ engine for rollout moves (~20x faster than Python rules).
Plugs directly into AlphaZeroMiniAgent as a drop-in replacement for
TorchPolicyValueModel, enabling controlled ablation of the NN prior.
"""

from __future__ import annotations
import math
import random
from typing import Dict, List, Tuple

from hybrid.agents.alphazero_stub import PolicyValueModel
from hybrid.core.env import GameState
from hybrid.core.types import Move, Side, PieceKind

# Piece values for truncation evaluation (mirrors hybrid.agents.eval.PIECE_VALUES)
_PIECE_VALUES = {
    PieceKind.KING: 0.0,
    PieceKind.QUEEN: 9.0,
    PieceKind.ROOK: 5.0,
    PieceKind.BISHOP: 3.0,
    PieceKind.KNIGHT: 3.0,
    PieceKind.PAWN: 1.0,
    PieceKind.GENERAL: 0.0,
    PieceKind.ADVISOR: 2.0,
    PieceKind.ELEPHANT: 2.0,
    PieceKind.HORSE: 4.0,
    PieceKind.CHARIOT: 9.0,
    PieceKind.CANNON: 5.0,
    PieceKind.SOLDIER: 1.0,
}


class RolloutModel(PolicyValueModel):
    """Uniform policy + random-rollout value estimation (C++ accelerated)."""

    def __init__(self, rollout_steps: int = 50, value_scale: float = 20.0,
                 seed: int = 0):
        self.rollout_steps = rollout_steps
        self.value_scale = value_scale
        self.rng = random.Random(seed)

        # Lazy-import C++ engine
        self._cpp = None
        self._cpp_chess_side = None
        self._cpp_xiangqi_side = None

    def _ensure_cpp(self):
        """Lazy-load the C++ engine module and type maps."""
        if self._cpp is not None:
            return
        from hybrid.core.env import _ensure_cpp_maps, _sync_to_cpp, _PY_TO_CPP_SIDE
        _ensure_cpp_maps()
        # Re-read module-level globals after initialization
        from hybrid.core.env import _cpp_module, _PY_TO_CPP_SIDE as side_map
        from hybrid.core.env import _CPP_TO_PY_KIND as kind_map
        self._cpp = _cpp_module             # SimpleNamespace: gen_legal, apply_move, ...
        self._sync_to_cpp = _sync_to_cpp
        self._side_map = side_map
        self._kind_map = kind_map
        self._cpp_chess_side = side_map[Side.CHESS]
        self._cpp_xiangqi_side = side_map[Side.XIANGQI]

    def predict(
        self, state: GameState, legal_moves: List[Move],
    ) -> Tuple[Dict[Move, float], float]:
        """Return (uniform_policy, rollout_value)."""
        if not legal_moves:
            return {}, 0.0

        # 1. Uniform policy
        prob = 1.0 / len(legal_moves)
        policy = {mv: prob for mv in legal_moves}

        # 2. Random rollout using C++ engine
        self._ensure_cpp()
        cpp = self._cpp
        root_side = state.side_to_move
        cpp_root_side = self._side_map[root_side]

        # Convert Python board to C++ board
        cpp_board = self._sync_to_cpp(state.board)
        cpp_side = self._side_map[state.side_to_move]

        for _ in range(self.rollout_steps):
            # Check if either royal is dead (captures end the game)
            if not cpp_board.has_royal(self._cpp_chess_side):
                value = -1.0 if root_side == Side.CHESS else 1.0
                return policy, value
            if not cpp_board.has_royal(self._cpp_xiangqi_side):
                value = 1.0 if root_side == Side.CHESS else -1.0
                return policy, value

            # Generate legal moves in C++
            cpp_moves = cpp.gen_legal(cpp_board, cpp_side)
            if not cpp_moves:
                # No legal moves = stalemate (draw)
                return policy, 0.0

            # Random selection
            idx = self.rng.randrange(len(cpp_moves))
            cpp_mv = cpp_moves[idx]

            # Apply move (returns new board)
            cpp_board = cpp.apply_move(cpp_board, cpp_mv)
            # Flip side
            if cpp_side == self._cpp_chess_side:
                cpp_side = self._cpp_xiangqi_side
            else:
                cpp_side = self._cpp_chess_side

        # 3. Truncation: material evaluation mapped to [-1, 1]
        mat = 0.0
        for triple in cpp_board.iter_pieces():
            x, y, piece = triple
            # Map C++ PieceKind → Python PieceKind via kind_map
            py_kind = self._kind_map.get(piece.kind)
            if py_kind is None:
                continue
            v = _PIECE_VALUES.get(py_kind, 0.0)
            if piece.side == cpp_root_side:
                mat += v
            else:
                mat -= v

        value = math.tanh(mat / self.value_scale)
        return policy, value
