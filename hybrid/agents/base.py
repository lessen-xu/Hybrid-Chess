# -*- coding: utf-8 -*-
"""Agent base class."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

from hybrid.core.env import GameState
from hybrid.core.types import Move


class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def select_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        raise NotImplementedError
