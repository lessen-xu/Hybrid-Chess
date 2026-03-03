# -*- coding: utf-8 -*-
"""TD-learning agent: Alpha-Beta search with a learned linear value function at leaf nodes."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

from .base import Agent
from hybrid.core.env import GameState
from hybrid.core.types import Move, Side
from hybrid.core.rules import apply_move, generate_legal_moves, terminal_info, TerminalStatus
from hybrid.core.config import MAX_PLIES

from hybrid.rl.td_learning import LinearValueFunction, TDConfig
from hybrid.rl.features import feature_dim


@dataclass
class TDSearchConfig:
    depth: int = 3


class TDAgent(Agent):
    name = "td"

    def __init__(self, value_fn: LinearValueFunction, cfg: TDSearchConfig = TDSearchConfig()):
        self.vf = value_fn
        self.cfg = cfg

    def select_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        side = state.side_to_move
        best_mv = legal_moves[0]
        best_val = -1e18
        alpha, beta = -1e18, 1e18

        for mv in legal_moves:
            nb = apply_move(state.board, mv)
            child = GameState(board=nb, side_to_move=side.opponent(), ply=state.ply+1, repetition=state.repetition)
            v = -self._negamax(child, self.cfg.depth - 1, -beta, -alpha)
            if v > best_val:
                best_val = v
                best_mv = mv
            alpha = max(alpha, v)

        return best_mv

    def _negamax(self, state: GameState, depth: int, alpha: float, beta: float) -> float:
        info = terminal_info(state.board, state.side_to_move, state.repetition, state.ply, MAX_PLIES)
        if info.status != TerminalStatus.ONGOING:
            if info.status == TerminalStatus.DRAW:
                return 0.0
            return 1e6 if info.winner == state.side_to_move else -1e6

        if depth <= 0:
            return self.vf.value(state, state.side_to_move)

        moves = generate_legal_moves(state.board, state.side_to_move)
        if not moves:
            return self.vf.value(state, state.side_to_move)

        best = -1e18
        for mv in moves:
            nb = apply_move(state.board, mv)
            child = GameState(board=nb, side_to_move=state.side_to_move.opponent(), ply=state.ply+1, repetition=state.repetition)
            v = -self._negamax(child, depth - 1, -beta, -alpha)
            best = max(best, v)
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return best


def new_default_td_value_function() -> LinearValueFunction:
    """Create a zero-initialized linear value function."""
    return LinearValueFunction(feature_dim())
