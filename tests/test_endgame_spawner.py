# -*- coding: utf-8 -*-
"""Tests for the endgame position spawner and env.reset_from_board."""

import random
import pytest

from hybrid.core.board import Board
from hybrid.core.env import HybridChessEnv, GameState
from hybrid.core.types import Side, PieceKind
from hybrid.core.rules import generate_legal_moves, _find_royal
from hybrid.rl.endgame_spawner import generate_endgame_board


class TestEndgameSpawner:
    """Test endgame board generation."""

    def test_generates_valid_boards(self):
        """100 generated boards should all have both royals and legal moves."""
        rng = random.Random(42)
        for _ in range(100):
            board, side = generate_endgame_board(rng)
            # Both royals exist
            assert _find_royal(board, Side.CHESS) is not None
            assert _find_royal(board, Side.XIANGQI) is not None
            # Side-to-move has legal moves
            legal = generate_legal_moves(board, side)
            assert len(legal) > 0
            # Opponent also has legal moves (not instant game-over)
            opp_legal = generate_legal_moves(board, side.opponent())
            assert len(opp_legal) > 0

    def test_side_to_move_is_valid(self):
        """Side-to-move should be either CHESS or XIANGQI."""
        rng = random.Random(123)
        for _ in range(50):
            _, side = generate_endgame_board(rng)
            assert side in (Side.CHESS, Side.XIANGQI)

    def test_boards_are_sparse(self):
        """Endgame boards should have few pieces (typically 3-5)."""
        rng = random.Random(999)
        for _ in range(50):
            board, _ = generate_endgame_board(rng)
            piece_count = sum(1 for _ in board.iter_pieces())
            assert 2 <= piece_count <= 6, f"Expected 2-6 pieces, got {piece_count}"

    def test_deterministic_with_same_seed(self):
        """Same seed should produce same boards."""
        boards_a = [generate_endgame_board(random.Random(0)) for _ in range(5)]
        boards_b = [generate_endgame_board(random.Random(0)) for _ in range(5)]
        for (ba, sa), (bb, sb) in zip(boards_a, boards_b):
            assert sa == sb
            for x, y, pa in ba.iter_pieces():
                pb = bb.get(x, y)
                assert pb is not None
                assert pa.kind == pb.kind and pa.side == pb.side


class TestResetFromBoard:
    """Test HybridChessEnv.reset_from_board."""

    def test_reset_from_board_basic(self):
        """reset_from_board should set the state correctly."""
        env = HybridChessEnv(max_plies=40)
        rng = random.Random(42)
        board, side = generate_endgame_board(rng)
        state = env.reset_from_board(board, side)
        assert state.side_to_move == side
        assert state.ply == 0
        # Should be able to get legal moves
        legal = env.legal_moves()
        assert len(legal) > 0

    def test_can_play_from_endgame(self):
        """Should be able to play a full game from an endgame position."""
        env = HybridChessEnv(max_plies=40)
        rng = random.Random(42)
        board, side = generate_endgame_board(rng)
        state = env.reset_from_board(board, side)

        # Play random moves until done
        for _ in range(40):
            legal = env.legal_moves()
            if len(legal) == 0:
                break
            mv = rng.choice(legal)
            state, reward, done, info = env.step(mv)
            if done:
                break
