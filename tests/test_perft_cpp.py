"""Perft regression tests — frozen node counts to catch rule changes.

These values are specific to the Hybrid Chess ruleset of this repository.
Any rule change (movegen, check detection, make/unmake) that alters the
legal move tree will cause a perft mismatch here.
"""

import pytest
from hybrid.cpp_engine import hybrid_cpp_engine as eng


def _initial_board():
    from hybrid.core.board import initial_board
    from hybrid.core.env import _ensure_cpp_maps, _sync_to_cpp
    _ensure_cpp_maps()
    return _sync_to_cpp(initial_board())


def _midgame_board():
    """Deterministic playout: seed=42, 10 plies from initial."""
    import random
    rng = random.Random(42)
    board = _initial_board()
    stm = eng.Side.CHESS
    for _ in range(10):
        legal = eng.generate_legal_moves(board, stm)
        mv = rng.choice(legal)
        board = eng.apply_move(board, mv)
        stm = eng.opponent(stm)
    return board, stm


class TestPerftInitialChess:
    """Perft from initial position, CHESS to move."""

    EXPECTED = {1: 23, 2: 1048, 3: 26311}

    @pytest.mark.parametrize("depth,expected", list(EXPECTED.items()))
    def test_perft(self, depth, expected):
        board = _initial_board()
        assert eng.perft_nodes(board, eng.Side.CHESS, depth) == expected


class TestPerftInitialXiangqi:
    """Perft from initial position, XIANGQI to move."""

    EXPECTED = {1: 46, 2: 1052, 3: 45840}

    @pytest.mark.parametrize("depth,expected", list(EXPECTED.items()))
    def test_perft(self, depth, expected):
        board = _initial_board()
        assert eng.perft_nodes(board, eng.Side.XIANGQI, depth) == expected


class TestPerftMidgame:
    """Perft from a deterministic mid-game position (seed=42, 10 plies)."""

    EXPECTED = {1: 26, 2: 1266, 3: 33477}

    @pytest.mark.parametrize("depth,expected", list(EXPECTED.items()))
    def test_perft(self, depth, expected):
        board, stm = _midgame_board()
        assert stm == eng.Side.CHESS
        assert eng.perft_nodes(board, stm, depth) == expected


class TestPerftDepthZero:
    """Depth 0 always returns 1."""

    def test_depth_zero(self):
        board = _initial_board()
        assert eng.perft_nodes(board, eng.Side.CHESS, 0) == 1
        assert eng.perft_nodes(board, eng.Side.XIANGQI, 0) == 1
