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


# ═══════════════════════════════════════════════════════════════
# Zobrist 128-bit hashing tests
# ═══════════════════════════════════════════════════════════════

class TestZobristCorrectness:
    """Incremental Zobrist key must match full-board recompute."""

    def test_initial_board_matches_recompute(self):
        board = _initial_board()
        for side in [eng.Side.CHESS, eng.Side.XIANGQI]:
            inc = board.zobrist_key_hex(side)
            rec = board.zobrist_key_hex_recompute(side)
            assert inc == rec, f"mismatch for {side}: {inc} vs {rec}"
            assert len(inc) == 32, f"expected 32 hex chars, got {len(inc)}"

    def test_after_apply_move(self):
        board = _initial_board()
        moves = eng.generate_legal_moves(board, eng.Side.CHESS)
        assert len(moves) > 0
        b2 = eng.apply_move(board, moves[0])
        for side in [eng.Side.CHESS, eng.Side.XIANGQI]:
            inc = b2.zobrist_key_hex(side)
            rec = b2.zobrist_key_hex_recompute(side)
            assert inc == rec, f"after move, mismatch for {side}"

    def test_different_sides_different_keys(self):
        """CHESS and XIANGQI keys should differ (due to side-to-move toggle)."""
        board = _initial_board()
        kc = board.zobrist_key_hex(eng.Side.CHESS)
        kx = board.zobrist_key_hex(eng.Side.XIANGQI)
        assert kc != kx, "same key for both sides — side toggle missing"


class TestZobristInvariance:
    """best_move must not mutate Zobrist key."""

    @pytest.mark.parametrize("depth", [1, 2])
    def test_zobrist_unchanged_after_best_move(self, depth):
        board = _initial_board()
        z0_c = board.zobrist_key_hex(eng.Side.CHESS)
        z0_x = board.zobrist_key_hex(eng.Side.XIANGQI)
        eng.best_move(board, eng.Side.CHESS, depth, {}, 0, 400)
        assert board.zobrist_key_hex(eng.Side.CHESS) == z0_c
        assert board.zobrist_key_hex(eng.Side.XIANGQI) == z0_x


class TestZobristRepetitionFastPath:
    """best_move with Zobrist-keyed (32-hex) repetition table must work."""

    def test_zobrist_rep_table_accepted(self):
        """Pass a 32-hex Zobrist key as rep table — should not crash."""
        board = _initial_board()
        zkey = board.zobrist_key_hex(eng.Side.CHESS)
        assert len(zkey) == 32
        rep = {zkey: 1}
        r = eng.best_move(board, eng.Side.CHESS, 1, rep, 0, 400)
        assert r.nodes > 0

    def test_zobrist_deterministic(self):
        """Same input with Zobrist rep table → same output."""
        board = _initial_board()
        zkey = board.zobrist_key_hex(eng.Side.CHESS)
        rep = {zkey: 1}
        r1 = eng.best_move(board, eng.Side.CHESS, 2, rep, 0, 400)
        r2 = eng.best_move(board, eng.Side.CHESS, 2, rep, 0, 400)
        assert r1.best_move == r2.best_move
        assert r1.score == r2.score
        assert r1.nodes == r2.nodes

    def test_empty_rep_uses_zobrist_path(self):
        """Empty rep table defaults to Zobrist fast path — must be deterministic."""
        board = _initial_board()
        r1 = eng.best_move(board, eng.Side.CHESS, 2, {}, 0, 400)
        r2 = eng.best_move(board, eng.Side.CHESS, 2, {}, 0, 400)
        assert r1.best_move == r2.best_move
        assert r1.score == r2.score
        assert r1.nodes == r2.nodes


class TestLegacySHA1Repetition:
    """best_move with SHA1-keyed (40-hex) repetition table must still work."""

    def test_sha1_rep_table(self):
        board = _initial_board()
        sha_key = board.board_hash(eng.Side.CHESS)
        assert len(sha_key) == 40
        rep = {sha_key: 1}
        r = eng.best_move(board, eng.Side.CHESS, 1, rep, 0, 400)
        assert r.nodes > 0

    def test_sha1_deterministic(self):
        board = _initial_board()
        sha_key = board.board_hash(eng.Side.CHESS)
        rep = {sha_key: 1}
        r1 = eng.best_move(board, eng.Side.CHESS, 2, rep, 0, 400)
        r2 = eng.best_move(board, eng.Side.CHESS, 2, rep, 0, 400)
        assert r1.best_move == r2.best_move
        assert r1.score == r2.score
        assert r1.nodes == r2.nodes


class TestRoyalCacheCorrectness:
    """royal_square must match royal_square_recompute for both sides."""

    def test_initial_board(self):
        board = _initial_board()
        for side in [eng.Side.CHESS, eng.Side.XIANGQI]:
            assert board.royal_square(side) == board.royal_square_recompute(side)
            assert board.has_royal(side) is True

    def test_after_set_piece(self):
        """After setting pieces manually, cache must remain correct."""
        board = eng.Board.empty()
        board.set(4, 0, eng.Piece(eng.PieceKind.KING, eng.Side.CHESS))
        board.set(4, 9, eng.Piece(eng.PieceKind.GENERAL, eng.Side.XIANGQI))
        assert board.royal_square(eng.Side.CHESS) == board.royal_square_recompute(eng.Side.CHESS)
        assert board.royal_square(eng.Side.XIANGQI) == board.royal_square_recompute(eng.Side.XIANGQI)


class TestRoyalCacheInvariance:
    """best_move must not mutate the board's royal cache."""

    def test_best_move_preserves_cache(self):
        board = _initial_board()
        rc0 = board.royal_square(eng.Side.CHESS)
        rx0 = board.royal_square(eng.Side.XIANGQI)
        zk0 = board.zobrist_key_hex(eng.Side.CHESS)
        eng.best_move(board, eng.Side.CHESS, 3, {}, 0, 400)
        assert board.royal_square(eng.Side.CHESS) == rc0
        assert board.royal_square(eng.Side.XIANGQI) == rx0
        assert board.zobrist_key_hex(eng.Side.CHESS) == zk0


class TestRoyalCacheApplyMove:
    """apply_move must produce a board with correct royal cache."""

    def test_apply_move_cache(self):
        board = _initial_board()
        legal = eng.generate_legal_moves(board, eng.Side.CHESS)
        assert len(legal) > 0
        b2 = eng.apply_move(board, legal[0])
        for side in [eng.Side.CHESS, eng.Side.XIANGQI]:
            assert b2.royal_square(side) == b2.royal_square_recompute(side)


class TestAttackEquivalence:
    """is_square_attacked_fast must exactly match is_square_attacked_slow."""

    def test_random_playout_equivalence(self):
        """Deterministic 40-ply playout, full-board attack check each step."""
        import random
        rng = random.Random(42)
        board = _initial_board()
        stm = eng.Side.CHESS
        for ply in range(40):
            # Check all squares for both sides
            for by_side in [eng.Side.CHESS, eng.Side.XIANGQI]:
                for y in range(10):
                    for x in range(9):
                        slow = eng.is_square_attacked_slow(board, x, y, by_side)
                        fast = eng.is_square_attacked_fast(board, x, y, by_side)
                        assert slow == fast, (
                            f"ply={ply} sq=({x},{y}) by={by_side} slow={slow} fast={fast}"
                        )
            legal = eng.generate_legal_moves(board, stm)
            if not legal:
                break
            mv = rng.choice(legal)
            board = eng.apply_move(board, mv)
            stm = eng.opponent(stm)
