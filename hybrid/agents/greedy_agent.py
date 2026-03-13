"""Greedy agent baseline: 1-ply capture maximizer.

Always picks the legal move that captures the highest-value enemy piece.
Ties are broken randomly. If no capture is available, selects a random
legal move.  Represents an "extremely short-sighted but rational" archetype.
"""

from __future__ import annotations
import random
from typing import List

from .base import Agent
from .eval import PIECE_VALUES
from hybrid.core.env import GameState
from hybrid.core.types import Move


class GreedyAgent(Agent):
    name = "greedy"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def select_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        best_moves: List[Move] = []
        max_val = 0.0

        for mv in legal_moves:
            target = state.board.get(mv.tx, mv.ty)
            if target is not None and target.side != state.side_to_move:
                val = PIECE_VALUES.get(target.kind, 0.0)
            else:
                val = 0.0

            if val > max_val:
                max_val = val
                best_moves = [mv]
            elif val == max_val:
                best_moves.append(mv)

        # No capture available → full random
        if max_val == 0.0:
            return self.rng.choice(legal_moves)
        return self.rng.choice(best_moves)
