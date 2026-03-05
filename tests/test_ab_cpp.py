# -*- coding: utf-8 -*-
"""Tests for the C++ alpha-beta search engine (best_move)."""
import pytest
from hybrid.cpp_engine import hybrid_cpp_engine as eng


def _initial_board():
    """Set up a standard initial board via The Python board, then sync to C++."""
    from hybrid.core.board import initial_board
    from hybrid.core.env import _ensure_cpp_maps, _sync_to_cpp
    _ensure_cpp_maps()
    py_board = initial_board()
    return _sync_to_cpp(py_board)


class TestBestMoveLegality:
    """best_move must return a legal move at any depth."""

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_initial_board(self, depth):
        board = _initial_board()
        legal = eng.generate_legal_moves(board, eng.Side.CHESS)
        r = eng.best_move(board, eng.Side.CHESS, depth, {}, 0, 400)
        # Result move must be one of the legal moves
        assert any(
            r.best_move.fx == m.fx and r.best_move.fy == m.fy and
            r.best_move.tx == m.tx and r.best_move.ty == m.ty and
            r.best_move.promotion == m.promotion
            for m in legal
        ), f"best_move returned illegal move at depth {depth}"

    @pytest.mark.parametrize("depth", [1, 2])
    def test_xiangqi_side(self, depth):
        """Xiangqi also gets a legal move."""
        board = _initial_board()
        legal = eng.generate_legal_moves(board, eng.Side.XIANGQI)
        r = eng.best_move(board, eng.Side.XIANGQI, depth, {}, 0, 400)
        assert any(
            r.best_move.fx == m.fx and r.best_move.fy == m.fy and
            r.best_move.tx == m.tx and r.best_move.ty == m.ty and
            r.best_move.promotion == m.promotion
            for m in legal
        )


class TestBestMoveProperties:
    """Nodes, determinism, score sanity."""

    def test_nodes_positive(self):
        board = _initial_board()
        r = eng.best_move(board, eng.Side.CHESS, 2, {}, 0, 400)
        assert r.nodes > 0, "search must visit at least one node"

    def test_nodes_increase_with_depth(self):
        board = _initial_board()
        r1 = eng.best_move(board, eng.Side.CHESS, 1, {}, 0, 400)
        r2 = eng.best_move(board, eng.Side.CHESS, 2, {}, 0, 400)
        assert r2.nodes > r1.nodes, "deeper search should visit more nodes"

    def test_deterministic(self):
        """Same input → same output."""
        board = _initial_board()
        r1 = eng.best_move(board, eng.Side.CHESS, 2, {}, 0, 400)
        r2 = eng.best_move(board, eng.Side.CHESS, 2, {}, 0, 400)
        assert r1.best_move == r2.best_move
        assert r1.score == r2.score
        assert r1.nodes == r2.nodes

    def test_score_finite(self):
        board = _initial_board()
        r = eng.best_move(board, eng.Side.CHESS, 2, {}, 0, 400)
        assert abs(r.score) < 1e7, "score should be finite for initial position"


class TestBestMoveWithRepetition:
    """Repetition table is correctly handled."""

    def test_with_nonempty_repetition(self):
        """Passing a repetition table should not crash."""
        board = _initial_board()
        key = board.board_hash(eng.Side.CHESS)
        rep = {key: 1}
        r = eng.best_move(board, eng.Side.CHESS, 1, rep, 0, 400)
        assert r.nodes > 0


class TestNoMutation:
    """make/unmake must not leak state through the Python API."""

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_best_move_no_board_mutation(self, depth):
        """Board hash must be identical before and after best_move."""
        board = _initial_board()
        h_chess = board.board_hash(eng.Side.CHESS)
        h_xiangqi = board.board_hash(eng.Side.XIANGQI)
        eng.best_move(board, eng.Side.CHESS, depth, {}, 0, 400)
        assert board.board_hash(eng.Side.CHESS) == h_chess
        assert board.board_hash(eng.Side.XIANGQI) == h_xiangqi

    def test_generate_legal_moves_no_mutation(self):
        """generate_legal_moves must not change the board."""
        board = _initial_board()
        h = board.board_hash(eng.Side.CHESS)
        eng.generate_legal_moves(board, eng.Side.CHESS)
        assert board.board_hash(eng.Side.CHESS) == h
        eng.generate_legal_moves(board, eng.Side.XIANGQI)
        assert board.board_hash(eng.Side.CHESS) == h

