"""Tests for balance primitives, causal de-confounding flags, and balance lab components."""

import pytest
from hybrid.core.types import Side, PieceKind, Piece, Move
from hybrid.core.board import Board, initial_board
from hybrid.core.config import VariantConfig
from hybrid.core.env import HybridChessEnv, GameState
from hybrid.core.rules import (
    terminal_info,
    generate_legal_moves,
    is_in_check,
    TerminalStatus,
    _active_variant,
)
import hybrid.core.rules as rules_module
from hybrid.experiments.balance.metrics import compute_tier0_metrics, Tier0Metrics
from hybrid.experiments.balance.design import (
    generate_screening_matrix,
    point_to_variant,
    get_standard_reference_points,
    DesignPoint,
)
from hybrid.experiments.balance.screening import (
    run_paired_tournament,
    TournamentConfig,
    TournamentResult,
)
from hybrid.experiments.balance.analysis import (
    estimate_causal_effects,
    fit_bradley_terry,
)


class TestFirstSideTempo:
    """Tests for initial tempo de-confounding (first_side)."""

    def test_first_side_chess_default(self):
        env = HybridChessEnv(variant=VariantConfig(first_side="chess"))
        state = env.reset()
        assert state.side_to_move == Side.CHESS
        assert state.ply == 0
        # Legal moves on ply 0 must be Chess moves
        moves = env.legal_moves()
        for m in moves:
            p = state.board.get(m.fx, m.fy)
            assert p is not None
            assert p.side == Side.CHESS

    def test_first_side_xiangqi(self):
        env = HybridChessEnv(variant=VariantConfig(first_side="xiangqi"))
        state = env.reset()
        assert state.side_to_move == Side.XIANGQI
        assert state.ply == 0
        # Legal moves on ply 0 must be Xiangqi moves
        moves = env.legal_moves()
        assert len(moves) > 0
        for m in moves:
            p = state.board.get(m.fx, m.fy)
            assert p is not None
            assert p.side == Side.XIANGQI


class TestStalemateRules:
    """Tests for stalemate rule de-coupling ('loss' vs 'draw')."""

    def _setup_stalemate_board(self) -> Board:
        """Create a simple stalemate position: Chess king trapped in corner with no legal moves, not in check."""
        b = Board.empty()
        # Chess king in corner (0, 0)
        b.set(0, 0, Piece(PieceKind.KING, Side.CHESS))
        # Xiangqi pieces confining the king without delivering check:
        # e.g., Xiangqi Chariot on (1, 2) attacks column 1 and row 2
        # Xiangqi Chariot on (2, 1) attacks column 2 and row 1
        # Neither directly attacks (0, 0), but (0, 1), (1, 0), (1, 1) are covered.
        b.set(1, 9, Piece(PieceKind.CHARIOT, Side.XIANGQI))  # attacks (1, 0), (1, 1) etc.
        b.set(8, 1, Piece(PieceKind.CHARIOT, Side.XIANGQI))  # attacks (0, 1), (1, 1) etc.
        # General placed far away in its palace
        b.set(4, 9, Piece(PieceKind.GENERAL, Side.XIANGQI))
        return b

    def test_stalemate_loss_default(self):
        b = self._setup_stalemate_board()
        var = VariantConfig(stalemate_rule="loss")
        rules_module._active_variant = var
        try:
            assert not is_in_check(b, Side.CHESS)
            assert len(generate_legal_moves(b, Side.CHESS)) == 0

            info = terminal_info(b, Side.CHESS, {}, 0, 200)
            assert info.status == TerminalStatus.XIANGQI_WIN
            assert info.winner == Side.XIANGQI
            assert "Stalemate" in info.reason
        finally:
            rules_module._active_variant = None

    def test_stalemate_draw_fide(self):
        b = self._setup_stalemate_board()
        var = VariantConfig(stalemate_rule="draw")
        rules_module._active_variant = var
        try:
            assert not is_in_check(b, Side.CHESS)
            assert len(generate_legal_moves(b, Side.CHESS)) == 0

            info = terminal_info(b, Side.CHESS, {}, 0, 200)
            assert info.status == TerminalStatus.DRAW
            assert info.winner is None
            assert info.reason == "Stalemate (draw)"
        finally:
            rules_module._active_variant = None


class TestChessMirror:
    """Tests for horizontal mirroring of the Chess army."""

    def test_chess_mirror_layout(self):
        var_standard = VariantConfig(chess_mirror=False)
        b_std = initial_board(var_standard)
        # Standard: (0, 0) is Rook, (4, 0) is King, (8, 0) is empty
        assert b_std.get(0, 0).kind == PieceKind.ROOK
        assert b_std.get(3, 0).kind == PieceKind.QUEEN
        assert b_std.get(4, 0).kind == PieceKind.KING
        assert b_std.get(8, 0) is None

        var_mirrored = VariantConfig(chess_mirror=True)
        b_mir = initial_board(var_mirrored)
        # Mirrored: (0, 0) is empty, (8, 0) is Rook, King is at (4, 0), Queen at (5, 0)
        assert b_mir.get(0, 0) is None
        assert b_mir.get(8, 0).kind == PieceKind.ROOK
        assert b_mir.get(5, 0).kind == PieceKind.QUEEN
        assert b_mir.get(4, 0).kind == PieceKind.KING


class TestAtomicRuleIndependence:
    """Test that palace and knight blocking operate independently."""

    def test_palace_without_knight_block(self):
        var = VariantConfig(chess_palace=True, knight_block=False)
        rules_module._active_variant = var
        try:
            b = Board.empty()
            # King at boundary of palace (5, 2)
            b.set(5, 2, Piece(PieceKind.KING, Side.CHESS))
            b.set(4, 9, Piece(PieceKind.GENERAL, Side.XIANGQI))
            # Knight with an adjacent obstacle
            b.set(2, 4, Piece(PieceKind.KNIGHT, Side.CHESS))
            b.set(2, 5, Piece(PieceKind.PAWN, Side.CHESS))  # obstacle directly in front

            moves = generate_legal_moves(b, Side.CHESS)
            # King cannot move outside palace (e.g. to (6, 2) or (5, 3))
            king_moves = [m for m in moves if m.fx == 5 and m.fy == 2]
            for km in king_moves:
                assert 3 <= km.tx <= 5 and 0 <= km.ty <= 2

            # Knight CAN jump over (2, 5) to (1, 6) or (3, 6) because knight_block is False
            knight_moves = [m for m in moves if m.fx == 2 and m.fy == 4]
            targets = {(m.tx, m.ty) for m in knight_moves}
            assert (1, 6) in targets or (3, 6) in targets
        finally:
            rules_module._active_variant = None

    def test_knight_block_without_palace(self):
        var = VariantConfig(chess_palace=False, knight_block=True)
        rules_module._active_variant = var
        try:
            b = Board.empty()
            # King outside palace (6, 5)
            b.set(6, 5, Piece(PieceKind.KING, Side.CHESS))
            b.set(4, 9, Piece(PieceKind.GENERAL, Side.XIANGQI))
            # Knight with leg blocked
            b.set(2, 4, Piece(PieceKind.KNIGHT, Side.CHESS))
            b.set(2, 5, Piece(PieceKind.PAWN, Side.CHESS))  # leg blocked in +y direction

            moves = generate_legal_moves(b, Side.CHESS)
            # King can move outside palace freely
            king_moves = [m for m in moves if m.fx == 6 and m.fy == 5]
            assert len(king_moves) > 0

            # Knight CANNOT jump to (1, 6) or (3, 6) because leg at (2, 5) is blocked
            knight_moves = [m for m in moves if m.fx == 2 and m.fy == 4]
            targets = {(m.tx, m.ty) for m in knight_moves}
            assert (1, 6) not in targets
            assert (3, 6) not in targets
        finally:
            rules_module._active_variant = None


class TestBalanceLabPipeline:
    """Integration test of metrics, design matrix, screening, and analysis."""

    def test_tier0_metrics(self):
        metrics = compute_tier0_metrics(VariantConfig(), name="baseline")
        assert isinstance(metrics, Tier0Metrics)
        assert metrics.chess_initial_moves == 23
        assert metrics.xiangqi_initial_moves == 46
        assert metrics.palace_restricted_royals == 1

    def test_screening_design_matrix(self):
        points = generate_screening_matrix(32)
        assert len(points) == 32
        # Check that base factor first_side has exactly 16 ones and 16 zeros
        first_sides = [p.factors["first_side"] for p in points]
        assert sum(first_sides) == 16
        # Check generator factor xq_queen also has 16 ones and 16 zeros
        xq_queens = [p.factors["xq_queen"] for p in points]
        assert sum(xq_queens) == 16

    def test_screening_tournament_and_causal_analysis(self):
        # Run a micro-tournament of 2 paired games across 2 configurations
        cfg = TournamentConfig(num_pairs=1, max_plies=20, agent_a="greedy", agent_b="random")
        p1 = DesignPoint(0, "std", {"first_side": 0, "chess_palace": 0}, VariantConfig())
        p2 = DesignPoint(1, "palace", {"first_side": 0, "chess_palace": 1}, VariantConfig(chess_palace=True))

        res1 = run_paired_tournament(p1.variant, cfg, variant_name=p1.name, factors=p1.factors)
        res2 = run_paired_tournament(p2.variant, cfg, variant_name=p2.name, factors=p2.factors)

        assert res1.total_games == 2
        assert res2.total_games == 2

        report = estimate_causal_effects([res1, res2], interaction_pairs=[])
        assert report.num_variants_analyzed == 2
        assert len(report.main_effects) >= 2
        md = report.summary_table()
        assert "Rule Balance Causal Attribution Report" in md
