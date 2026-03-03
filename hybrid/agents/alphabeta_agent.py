# -*- coding: utf-8 -*-
"""Alpha-Beta search agent (Negamax variant) with hand-crafted evaluation."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .base import Agent
from .eval import evaluate, EvalWeights
from hybrid.core.env import GameState
from hybrid.core.types import Move, Side
from hybrid.core.rules import apply_move, generate_legal_moves, is_in_check, terminal_info, TerminalStatus
from hybrid.core.config import MAX_PLIES


@dataclass
class SearchConfig:
    depth: int = 3
    eval_weights: EvalWeights = field(default_factory=EvalWeights)


class AlphaBetaAgent(Agent):
    name = "alphabeta"

    def __init__(self, cfg: SearchConfig = SearchConfig()):
        self.cfg = cfg

    def select_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        side = state.side_to_move
        best_mv = legal_moves[0]
        best_val = -1e18

        # Move ordering: captures and checks first
        ordered = sorted(legal_moves, key=lambda m: self._move_order_key(state, m), reverse=True)

        alpha, beta = -1e18, 1e18
        for mv in ordered:
            nb = apply_move(state.board, mv)
            child = GameState(board=nb, side_to_move=side.opponent(), ply=state.ply+1, repetition=state.repetition)
            v = -self._negamax(child, self.cfg.depth - 1, -beta, -alpha, side.opponent())
            if v > best_val:
                best_val = v
                best_mv = mv
            alpha = max(alpha, v)

        return best_mv

    def _negamax(self, state: GameState, depth: int, alpha: float, beta: float, perspective: Side) -> float:
        """Negamax: returns value from perspective's point of view."""
        info = terminal_info(state.board, state.side_to_move, state.repetition, state.ply, MAX_PLIES)
        if info.status != TerminalStatus.ONGOING:
            if info.status == TerminalStatus.DRAW:
                return 0.0
            return 1e6 if info.winner == perspective else -1e6

        if depth <= 0:
            return evaluate(state, perspective, self.cfg.eval_weights)

        moves = generate_legal_moves(state.board, state.side_to_move)
        if not moves:
            return evaluate(state, perspective, self.cfg.eval_weights)

        ordered = sorted(moves, key=lambda m: self._move_order_key(state, m), reverse=True)

        best = -1e18
        for mv in ordered:
            nb = apply_move(state.board, mv)
            child = GameState(board=nb, side_to_move=state.side_to_move.opponent(), ply=state.ply+1, repetition=state.repetition)
            v = -self._negamax(child, depth - 1, -beta, -alpha, perspective)
            best = max(best, v)
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return best

    def _move_order_key(self, state: GameState, mv: Move) -> float:
        """Heuristic key for move ordering: captures + checks."""
        b = state.board
        t = b.get(mv.tx, mv.ty)
        capture_bonus = 10.0 if t is not None else 0.0
        nb = apply_move(b, mv)
        check_bonus = 2.0 if is_in_check(nb, state.side_to_move.opponent()) else 0.0
        return capture_bonus + check_bonus
