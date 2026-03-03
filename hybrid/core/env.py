# -*- coding: utf-8 -*-
"""Minimal game environment (gym-like, no gym dependency)."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .types import Side, PieceKind, Piece, Move
from .board import Board, initial_board
from .rules import apply_move, generate_legal_moves, terminal_info, GameInfo, TerminalStatus, board_hash
from .config import MAX_PLIES, ENABLE_THREEFOLD_REPETITION_DRAW, BOARD_W, BOARD_H


@dataclass
class GameState:
    board: Board
    side_to_move: Side
    ply: int = 0  # half-move count
    repetition: Dict[str, int] = field(default_factory=dict)

    def clone(self) -> "GameState":
        return GameState(
            board=self.board.clone(),
            side_to_move=self.side_to_move,
            ply=self.ply,
            repetition=dict(self.repetition),
        )


# ═══════════════════════════════════════════════════════════════
# C++ engine helpers (lazy-imported only when use_cpp=True)
# ═══════════════════════════════════════════════════════════════

_PY_TO_CPP_SIDE = None
_PY_TO_CPP_KIND = None
_CPP_TO_PY_KIND = None
_cpp_module = None


def _ensure_cpp_maps():
    """Lazy-initialize the Python ↔ C++ type mappings."""
    global _PY_TO_CPP_SIDE, _PY_TO_CPP_KIND, _CPP_TO_PY_KIND, _cpp_module
    if _cpp_module is not None:
        return

    from types import SimpleNamespace
    from hybrid.cpp_engine import (
        Side as CppSide,
        PieceKind as CppPieceKind,
        Piece as CppPiece,
        Move as CppMove,
        Board as CppBoard,
        generate_legal_moves as cpp_gen_legal,
        apply_move as cpp_apply,
        terminal_info as cpp_term_info,
        is_in_check as cpp_is_in_check,
    )

    _PY_TO_CPP_SIDE = {Side.CHESS: CppSide.CHESS, Side.XIANGQI: CppSide.XIANGQI}

    _PY_TO_CPP_KIND = {
        PieceKind.KING: CppPieceKind.KING, PieceKind.QUEEN: CppPieceKind.QUEEN,
        PieceKind.ROOK: CppPieceKind.ROOK, PieceKind.BISHOP: CppPieceKind.BISHOP,
        PieceKind.KNIGHT: CppPieceKind.KNIGHT, PieceKind.PAWN: CppPieceKind.PAWN,
        PieceKind.GENERAL: CppPieceKind.GENERAL, PieceKind.ADVISOR: CppPieceKind.ADVISOR,
        PieceKind.ELEPHANT: CppPieceKind.ELEPHANT, PieceKind.HORSE: CppPieceKind.HORSE,
        PieceKind.CHARIOT: CppPieceKind.CHARIOT, PieceKind.CANNON: CppPieceKind.CANNON,
        PieceKind.SOLDIER: CppPieceKind.SOLDIER,
    }

    _CPP_TO_PY_KIND = {v: k for k, v in _PY_TO_CPP_KIND.items()}

    _cpp_module = SimpleNamespace(
        CppSide=CppSide,
        CppPieceKind=CppPieceKind,
        CppPiece=CppPiece,
        CppMove=CppMove,
        CppBoard=CppBoard,
        gen_legal=cpp_gen_legal,
        apply_move=cpp_apply,
        terminal_info=cpp_term_info,
        is_in_check=cpp_is_in_check,
    )


def _sync_to_cpp(py_board: Board):
    """Create a C++ Board from a Python Board."""
    _ensure_cpp_maps()
    cpp_board = _cpp_module.CppBoard.empty()
    for x, y, p in py_board.iter_pieces():
        cpp_board.set(x, y, _cpp_module.CppPiece(
            _PY_TO_CPP_KIND[p.kind], _PY_TO_CPP_SIDE[p.side]
        ))
    return cpp_board


def _sync_to_py(cpp_board) -> Board:
    """Create a Python Board from a C++ Board."""
    py_board = Board.empty()
    for x, y, p in cpp_board.iter_pieces():
        py_board.set(x, y, Piece(_CPP_TO_PY_KIND[p.kind],
                                  Side.CHESS if p.side == _cpp_module.CppSide.CHESS else Side.XIANGQI))
    return py_board


def _cpp_to_py_move(cm) -> Move:
    """Convert a C++ Move to a Python Move."""
    promo = None
    if cm.promotion != _cpp_module.CppPieceKind.NONE:
        promo = _CPP_TO_PY_KIND[cm.promotion]
    return Move(cm.fx, cm.fy, cm.tx, cm.ty, promo)


def _py_to_cpp_move(pm: Move):
    """Convert a Python Move to a C++ Move."""
    _ensure_cpp_maps()
    promo = _cpp_module.CppPieceKind.NONE
    if pm.promotion is not None:
        promo = _PY_TO_CPP_KIND[pm.promotion]
    return _cpp_module.CppMove(pm.fx, pm.fy, pm.tx, pm.ty, promo)


class HybridChessEnv:
    """Hybrid chess game environment."""

    def __init__(self, max_plies: int = MAX_PLIES, use_cpp: bool = False):
        if max_plies <= 0:
            raise ValueError("max_plies must be > 0")
        self.max_plies = max_plies
        self.use_cpp = use_cpp
        self.state: Optional[GameState] = None
        self._cpp_board = None
        self._cpp_side = None

        if use_cpp:
            _ensure_cpp_maps()

    def set_max_plies(self, max_plies: int) -> None:
        """Update the move limit (for self-play compute budgeting)."""
        if max_plies <= 0:
            raise ValueError("max_plies must be > 0")
        self.max_plies = max_plies

    def _init_cpp_state(self, py_board: Board, side: Side) -> None:
        """Sync C++ board from Python board."""
        self._cpp_board = _sync_to_cpp(py_board)
        self._cpp_side = _PY_TO_CPP_SIDE[side]

    def reset(self) -> GameState:
        b = initial_board()
        s = GameState(board=b, side_to_move=Side.CHESS, ply=0, repetition={})
        if ENABLE_THREEFOLD_REPETITION_DRAW:
            key = board_hash(s.board, s.side_to_move)
            s.repetition[key] = s.repetition.get(key, 0) + 1
        self.state = s

        if self.use_cpp:
            self._init_cpp_state(b, Side.CHESS)

        return s.clone()

    def reset_from_board(self, board: Board, side_to_move: Side) -> GameState:
        """Reset to a custom board position (for endgame curriculum learning)."""
        s = GameState(board=board.clone(), side_to_move=side_to_move, ply=0, repetition={})
        if ENABLE_THREEFOLD_REPETITION_DRAW:
            key = board_hash(s.board, s.side_to_move)
            s.repetition[key] = s.repetition.get(key, 0) + 1
        self.state = s

        if self.use_cpp:
            self._init_cpp_state(board, side_to_move)

        return s.clone()

    def legal_moves(self) -> List[Move]:
        assert self.state is not None
        if self.use_cpp:
            cpp_moves = _cpp_module.gen_legal(self._cpp_board, self._cpp_side)
            return [_cpp_to_py_move(cm) for cm in cpp_moves]
        return generate_legal_moves(self.state.board, self.state.side_to_move)

    def step(self, mv: Move) -> Tuple[GameState, float, bool, GameInfo]:
        """Execute one move.

        Returns (next_state, reward, done, info).
        Reward is from the *moving* side's perspective (+1 win, -1 loss, 0 draw/ongoing).
        """
        assert self.state is not None
        s = self.state

        if self.use_cpp:
            return self._step_cpp(mv)

        # ── Python path (unchanged) ──
        legal = self.legal_moves()
        if mv not in legal:
            raise ValueError(f"Illegal move: {mv} for side {s.side_to_move}")

        nb = apply_move(s.board, mv)
        next_side = s.side_to_move.opponent()
        next_state = GameState(board=nb, side_to_move=next_side, ply=s.ply + 1, repetition=dict(s.repetition))

        if ENABLE_THREEFOLD_REPETITION_DRAW:
            key = board_hash(next_state.board, next_state.side_to_move)
            next_state.repetition[key] = next_state.repetition.get(key, 0) + 1

        info = terminal_info(
            next_state.board,
            next_state.side_to_move,
            next_state.repetition,
            next_state.ply,
            self.max_plies,
        )
        done = info.status != TerminalStatus.ONGOING

        reward = 0.0
        if done:
            if info.status == TerminalStatus.DRAW:
                reward = 0.0
            else:
                winner = info.winner
                reward = 1.0 if winner == s.side_to_move else -1.0

        self.state = next_state
        return next_state.clone(), reward, done, info

    def _step_cpp(self, mv: Move) -> Tuple[GameState, float, bool, GameInfo]:
        """Execute one move using the C++ engine."""
        s = self.state
        moving_side = s.side_to_move

        # Apply move on C++ board
        cpp_mv = _py_to_cpp_move(mv)
        self._cpp_board = _cpp_module.apply_move(self._cpp_board, cpp_mv)

        # Advance side
        next_py_side = moving_side.opponent()
        self._cpp_side = _PY_TO_CPP_SIDE[next_py_side]

        # Sync Python board from C++ (needed for state encoding)
        py_board = _sync_to_py(self._cpp_board)

        # Build next GameState
        next_state = GameState(
            board=py_board,
            side_to_move=next_py_side,
            ply=s.ply + 1,
            repetition=dict(s.repetition),
        )

        # Repetition tracking using C++ hash
        if ENABLE_THREEFOLD_REPETITION_DRAW:
            key = self._cpp_board.board_hash(self._cpp_side)
            next_state.repetition[key] = next_state.repetition.get(key, 0) + 1

        # Terminal detection via C++
        cpp_info = _cpp_module.terminal_info(
            self._cpp_board, self._cpp_side,
            next_state.repetition, next_state.ply, self.max_plies,
        )

        # Convert C++ GameInfo → Python GameInfo
        status = cpp_info.status  # string, same values
        done = status != TerminalStatus.ONGOING

        winner = None
        if cpp_info.winner == 1:
            winner = Side.CHESS
        elif cpp_info.winner == 2:
            winner = Side.XIANGQI

        info = GameInfo(status=status, winner=winner, reason=cpp_info.reason)

        reward = 0.0
        if done:
            if status == TerminalStatus.DRAW:
                reward = 0.0
            else:
                reward = 1.0 if winner == moving_side else -1.0

        self.state = next_state
        return next_state.clone(), reward, done, info
