"""Alpha-beta negamax with optional bounded search for local web play."""
from __future__ import annotations
from dataclasses import dataclass, field
from time import perf_counter
from typing import List, Optional
from .base import Agent
from .eval import evaluate, EvalWeights
from hybrid.core.env import GameState
from hybrid.core.types import Move, Side
from hybrid.core.rules import (apply_move, generate_legal_moves, is_in_check,
                               terminal_info, TerminalStatus, board_hash)
from hybrid.core.config import MAX_PLIES, ENABLE_THREEFOLD_REPETITION_DRAW


@dataclass
class SearchConfig:
    depth: int = 3
    eval_weights: EvalWeights = field(default_factory=EvalWeights)
    time_limit_seconds: Optional[float] = None
    max_plies: int = MAX_PLIES

    def __post_init__(self):
        if self.depth < 1:
            raise ValueError("depth must be positive")
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")


class _SearchTimeout(Exception):
    pass


class AlphaBetaAgent(Agent):
    name = "alphabeta"

    def __init__(self, cfg: Optional[SearchConfig] = None):
        self.cfg = cfg or SearchConfig()
        self._deadline = None
        self.last_completed_depth = 0

    def _check_time(self):
        if self._deadline is not None and perf_counter() >= self._deadline:
            raise _SearchTimeout

    def _child(self, state: GameState, mv: Move) -> GameState:
        board = apply_move(state.board, mv)
        side = state.side_to_move.opponent()
        repetition = dict(state.repetition)
        if ENABLE_THREEFOLD_REPETITION_DRAW:
            key = board_hash(board, side)
            repetition[key] = repetition.get(key, 0) + 1
        return GameState(board, side, state.ply + 1, repetition, state.variant, state.max_plies)

    def select_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        if not legal_moves:
            raise ValueError("No legal moves")
        best_move = legal_moves[0]
        self.last_completed_depth = 0
        limit = self.cfg.time_limit_seconds
        self._deadline = perf_counter() + limit if limit is not None else None
        depths = range(1, self.cfg.depth + 1) if limit is not None else [self.cfg.depth]
        try:
            ordered = self._ordered_moves(state, legal_moves)
            for depth in depths:
                alpha, beta = -1e18, 1e18
                candidate, best_value = ordered[0], -1e18
                for mv in ordered:
                    self._check_time()
                    child = self._child(state, mv)
                    value = -self._negamax(child, depth - 1, -beta, -alpha,
                                           child.side_to_move)
                    self._check_time()
                    if value > best_value:
                        candidate, best_value = mv, value
                    alpha = max(alpha, value)
                best_move = candidate
                self.last_completed_depth = depth
                ordered = [candidate] + [m for m in ordered if m != candidate]
        except _SearchTimeout:
            pass
        finally:
            self._deadline = None
        return best_move

    def _negamax(self, state: GameState, depth: int, alpha: float,
                 beta: float, perspective: Side) -> float:
        """Each node is valued for its side to move; each edge flips sign."""
        self._check_time()
        info = terminal_info(state.board, state.side_to_move, state.repetition,
                             state.ply, self.cfg.max_plies)
        if info.status != TerminalStatus.ONGOING:
            if info.status == TerminalStatus.DRAW:
                return 0.0
            score = 1e6 - state.ply
            return score if info.winner == perspective else -score
        if depth <= 0:
            value = evaluate(state, perspective, self.cfg.eval_weights)
            self._check_time()
            return value
        best = -1e18
        moves = generate_legal_moves(state.board, state.side_to_move)
        for mv in self._ordered_moves(state, moves):
            child = self._child(state, mv)
            value = -self._negamax(child, depth - 1, -beta, -alpha,
                                   perspective.opponent())
            best = max(best, value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best

    def _ordered_moves(self, state, moves):
        return sorted(moves, key=lambda m: self._move_order_key(state, m), reverse=True)

    def _move_order_key(self, state: GameState, mv: Move) -> float:
        self._check_time()
        capture = 10.0 if state.board.get(mv.tx, mv.ty) is not None else 0.0
        board = apply_move(state.board, mv)
        return capture + (2.0 if is_in_check(board, state.side_to_move.opponent()) else 0.0)
