# -*- coding: utf-8 -*-
"""Alpha-Beta agent backed by C++ engine for fast search.

Same logic as AlphaBetaAgent but uses the C++ rules engine for
generate_legal_moves, apply_move, is_in_check, and terminal_info
inside the search tree.  Only enters/exits Python at the top level.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from .base import Agent
from .eval import PIECE_VALUES, EvalWeights
from hybrid.core.env import GameState, _ensure_cpp_maps
from hybrid.core.types import Move, Side, PieceKind
from hybrid.core.config import MAX_PLIES
from hybrid.core.rules import TerminalStatus


@dataclass
class SearchConfig:
    depth: int = 3
    eval_weights: EvalWeights = field(default_factory=EvalWeights)


class AlphaBetaCppAgent(Agent):
    """Alpha-Beta agent using C++ engine for all internal search."""
    name = "alphabeta_cpp"

    def __init__(self, cfg: SearchConfig = SearchConfig()):
        self.cfg = cfg
        # Must call _ensure_cpp_maps() FIRST, then access the globals
        _ensure_cpp_maps()
        import hybrid.core.env as _env
        self._m = _env._cpp_module
        self._side_map = _env._PY_TO_CPP_SIDE
        self._sync_to_cpp = _env._sync_to_cpp
        self._cpp_to_py_move = _env._cpp_to_py_move
        self._kind_values = {}  # C++ PieceKind -> float
        for py_kind, val in PIECE_VALUES.items():
            if py_kind in _env._PY_TO_CPP_KIND:
                self._kind_values[_env._PY_TO_CPP_KIND[py_kind]] = val

    def select_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        m = self._m
        cpp_board = self._sync_to_cpp(state.board)
        cpp_side = self._side_map[state.side_to_move]
        perspective = cpp_side

        # Generate moves in C++ space
        cpp_moves = m.gen_legal(cpp_board, cpp_side)
        if not cpp_moves:
            return legal_moves[0]

        # Move ordering
        ordered = sorted(cpp_moves,
                         key=lambda mv: self._move_order_key_cpp(cpp_board, cpp_side, mv),
                         reverse=True)

        best_mv = ordered[0]
        best_val = -1e18
        alpha, beta = -1e18, 1e18

        opp_side = m.CppSide.XIANGQI if cpp_side == m.CppSide.CHESS else m.CppSide.CHESS

        for mv in ordered:
            child_board = m.apply_move(cpp_board, mv)
            v = -self._negamax_cpp(child_board, opp_side,
                                   self.cfg.depth - 1, -beta, -alpha,
                                   perspective, state.ply + 1, state.repetition)
            if v > best_val:
                best_val = v
                best_mv = mv
            alpha = max(alpha, v)

        return self._cpp_to_py_move(best_mv)

    def _negamax_cpp(self, board, side, depth: int,
                     alpha: float, beta: float,
                     perspective, ply: int, repetition: dict) -> float:
        m = self._m

        info = m.terminal_info(board, side, repetition, ply, MAX_PLIES)

        if info.status != TerminalStatus.ONGOING:
            if info.status == TerminalStatus.DRAW:
                return 0.0
            # Winner is int in C++ (1=CHESS, 2=XIANGQI)
            winner_side = m.CppSide.CHESS if info.winner == 1 else m.CppSide.XIANGQI
            return 1e6 if winner_side == perspective else -1e6

        if depth <= 0:
            return self._evaluate_cpp(board, side, perspective)

        moves = m.gen_legal(board, side)
        if not moves:
            return self._evaluate_cpp(board, side, perspective)

        ordered = sorted(moves,
                         key=lambda mv: self._move_order_key_cpp(board, side, mv),
                         reverse=True)

        opp = m.CppSide.XIANGQI if side == m.CppSide.CHESS else m.CppSide.CHESS

        best = -1e18
        for mv in ordered:
            child = m.apply_move(board, mv)
            v = -self._negamax_cpp(child, opp, depth - 1, -beta, -alpha,
                                   perspective, ply + 1, repetition)
            best = max(best, v)
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return best

    def _evaluate_cpp(self, board, side, perspective) -> float:
        """Evaluation using C++ board: material + mobility + check."""
        m = self._m
        w = self.cfg.eval_weights

        # Material
        mat = 0.0
        for x, y, p in board.iter_pieces():
            v = self._kind_values.get(p.kind, 0.0)
            mat += v if p.side == perspective else -v

        # Mobility
        opp = m.CppSide.XIANGQI if perspective == m.CppSide.CHESS else m.CppSide.CHESS
        my_moves = len(m.gen_legal(board, perspective))
        op_moves = len(m.gen_legal(board, opp))
        mob = float(my_moves - op_moves)

        # Check bonus
        check_val = 0.0
        if m.is_in_check(board, opp):
            check_val += w.check_bonus
        if m.is_in_check(board, perspective):
            check_val -= w.check_bonus

        return mat + w.mobility * mob + check_val

    def _move_order_key_cpp(self, board, side, mv) -> float:
        m = self._m
        cap = board.get(mv.tx, mv.ty)
        capture_bonus = 10.0 if cap is not None else 0.0

        child = m.apply_move(board, mv)
        opp = m.CppSide.XIANGQI if side == m.CppSide.CHESS else m.CppSide.CHESS
        check_bonus = 2.0 if m.is_in_check(child, opp) else 0.0

        return capture_bonus + check_bonus
