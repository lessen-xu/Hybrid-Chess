"""Random agent baseline: uniform random move selection."""

from __future__ import annotations
import random
from typing import List

from .base import Agent
from hybrid.core.env import GameState
from hybrid.core.types import Move


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def select_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        return self.rng.choice(legal_moves)
