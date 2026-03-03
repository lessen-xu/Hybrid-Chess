# -*- coding: utf-8 -*-
"""Tests for resign mechanism, draw adjudication, and diagnostics."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from hybrid.core.types import Side, PieceKind
from hybrid.core.config import MAX_PLIES
from hybrid.rl.az_selfplay import (
    GameRecord,
    SelfPlayConfig,
    compute_material_diff,
    material_diff_to_value,
    self_play_game,
)


# ====================================================================
# A) GameRecord defaults
# ====================================================================

class TestGameRecord:
    def test_default_values(self):
        r = GameRecord()
        assert r.result == "draw"
        assert r.termination_reason == ""
        assert r.ply_count == 0
        assert r.material_diff == 0.0
        assert r.resigned is False
        assert r.resign_side is None
        assert r.rootv_min == 0.0
        assert r.rootv_p05 == 0.0
        assert r.low_rootv_steps == 0
        assert r.rootv_steps == 0

    def test_custom_values(self):
        r = GameRecord(
            result="chess_win",
            termination_reason="Resign",
            ply_count=53,
            material_diff=4.5,
            resigned=True,
            resign_side="xiangqi",
            rootv_min=-0.9,
            rootv_p05=-0.8,
            low_rootv_steps=3,
            rootv_steps=40,
        )
        assert r.result == "chess_win"
        assert r.resigned is True
        assert r.resign_side == "xiangqi"
        assert r.rootv_min == -0.9
        assert r.rootv_steps == 40


# ====================================================================
# B) SelfPlayConfig defaults
# ====================================================================

class TestSelfPlayConfig:
    def test_defaults(self):
        cfg = SelfPlayConfig()
        assert cfg.max_ply == MAX_PLIES
        assert cfg.move_limit_value_mode == "penalty"
        assert cfg.move_limit_value_scale == 4.0
        assert cfg.resign_enabled is True
        assert cfg.resign_threshold == -0.95
        assert cfg.resign_min_ply == 40
        assert cfg.resign_patience == 3
        assert cfg.draw_adjudicate_enabled is True
        assert cfg.draw_adjudicate_min_ply == 60
        assert cfg.draw_adjudicate_patience == 15
        assert cfg.draw_adjudicate_value_abs_thr == 0.08
        assert cfg.simulations == 50
        assert cfg.temperature == 1.0

    def test_resign_disabled(self):
        cfg = SelfPlayConfig(resign_enabled=False)
        assert cfg.resign_enabled is False


# ====================================================================
# C) compute_material_diff
# ====================================================================

class TestMaterialDiff:
    def test_initial_board(self):
        """Initial board material diff should be small but non-zero (Chess has Queen)."""
        from hybrid.core.board import initial_board
        board = initial_board()
        diff = compute_material_diff(board)
        assert isinstance(diff, float)
        # Initial material is roughly balanced, |diff| < 20
        assert abs(diff) < 20.0

    def test_empty_board_mock(self):
        """Empty board diff == 0."""
        mock_board = MagicMock()
        mock_board.iter_pieces.return_value = []
        diff = compute_material_diff(mock_board)
        assert diff == 0.0


class TestMoveLimitValue:
    def test_material_diff_to_value_modes(self):
        assert material_diff_to_value(9.0, mode="zero") == 0.0
        assert material_diff_to_value(9.0, mode="hard") == 1.0
        assert material_diff_to_value(-0.1, mode="hard") == -1.0
        assert material_diff_to_value(0.0, mode="hard") == 0.0

        v_pos = material_diff_to_value(4.0, mode="soft", scale=4.0)
        v_neg = material_diff_to_value(-4.0, mode="soft", scale=4.0)
        assert 0.0 < v_pos < 1.0
        assert -1.0 < v_neg < 0.0
        assert v_neg == pytest.approx(-v_pos, abs=1e-6)


# ====================================================================
# D) Resign mechanism integration tests
# ====================================================================

class TestResignMechanism:
    def _make_mock_agent(self, values_sequence):
        """Create a mock agent whose select_move_with_pi returns values from a sequence."""
        agent = MagicMock()
        call_counter = [0]

        def mock_select_move_with_pi(state, legal_moves, temperature=1.0, add_noise=True):
            idx = min(call_counter[0], len(values_sequence) - 1)
            val = values_sequence[idx]
            call_counter[0] += 1

            mv = legal_moves[0]
            pi = {m: 1.0 / len(legal_moves) for m in legal_moves}
            return mv, pi, val

        agent.select_move_with_pi = mock_select_move_with_pi
        return agent

    def test_resign_triggers_after_patience(self):
        """Consecutive low values exceeding patience should trigger resign."""
        from hybrid.core.env import HybridChessEnv

        # First 40 steps normal (resign_min_ply=40), then consistently low
        values = [0.0] * 40 + [-0.99] * 10

        agent = self._make_mock_agent(values)
        env = HybridChessEnv()

        cfg = SelfPlayConfig(
            resign_enabled=True,
            resign_threshold=-0.95,
            resign_min_ply=40,
            resign_patience=3,
            simulations=10,
        )

        examples, record = self_play_game(env, agent, cfg)

        assert record.resigned is True
        assert record.termination_reason == "Resign"
        assert record.resign_side is not None
        assert record.result in ("chess_win", "xiangqi_win")
        assert record.rootv_steps > 0
        assert record.rootv_min <= record.rootv_p05

        # z should be correctly assigned after resign (not all zero)
        z_values = [ex.z for ex in examples]
        assert any(z != 0.0 for z in z_values), \
            "z values should not be all 0 after resign"
        assert any(z == 1.0 for z in z_values), "Winner side should have z=1"
        assert any(z == -1.0 for z in z_values), "Loser side should have z=-1"

    def test_resign_disabled(self):
        """resign_enabled=False should never trigger resign."""
        from hybrid.core.env import HybridChessEnv

        values = [-0.99] * 500

        agent = self._make_mock_agent(values)
        env = HybridChessEnv()

        cfg = SelfPlayConfig(
            resign_enabled=False,
            resign_threshold=-0.95,
            resign_min_ply=40,
            resign_patience=3,
            simulations=10,
        )

        examples, record = self_play_game(env, agent, cfg)

        assert record.resigned is False
        assert record.termination_reason != "Resign"
        assert record.rootv_steps > 0

    def test_resign_not_before_min_ply(self):
        """Resign should not trigger before min_ply."""
        from hybrid.core.env import HybridChessEnv

        values = [-0.99] * 300

        agent = self._make_mock_agent(values)
        env = HybridChessEnv()

        cfg = SelfPlayConfig(
            resign_enabled=True,
            resign_threshold=-0.95,
            resign_min_ply=200,  # Higher than typical game length
            resign_patience=3,
            simulations=10,
        )

        examples, record = self_play_game(env, agent, cfg)

        # Game should end by other means before reaching min_ply=200
        if record.ply_count < 200:
            assert record.resigned is False

    def test_draw_adjudication_triggers_on_stable_near_draw(self):
        """Should adjudicate draw when root values are consistently near zero."""
        from hybrid.core.env import HybridChessEnv

        agent = self._make_mock_agent([0.0] * 20)
        env = HybridChessEnv()

        cfg = SelfPlayConfig(
            resign_enabled=False,
            draw_adjudicate_enabled=True,
            draw_adjudicate_min_ply=2,
            draw_adjudicate_patience=2,
            draw_adjudicate_value_abs_thr=0.05,
            simulations=1,
        )

        examples, record = self_play_game(env, agent, cfg)
        assert record.termination_reason == "Adjudicated draw"
        assert record.result == "draw"
        assert record.resigned is False
        assert record.ply_count < cfg.max_ply
        assert all(ex.z == 0.0 for ex in examples)

    def test_draw_adjudication_disabled(self):
        """When disabled, adjudicated draw should not occur."""
        from hybrid.core.env import HybridChessEnv

        agent = self._make_mock_agent([0.0] * 20)
        env = HybridChessEnv()

        cfg = SelfPlayConfig(
            resign_enabled=False,
            draw_adjudicate_enabled=False,
            max_ply=8,
            simulations=1,
        )
        _, record = self_play_game(env, agent, cfg)
        assert record.termination_reason != "Adjudicated draw"

    def test_move_limit_uses_cfg_max_ply_with_hard_value(self):
        """Self-play max_ply is configurable; move-limit can produce non-zero z."""
        from hybrid.core.env import HybridChessEnv

        agent = self._make_mock_agent([0.0] * 20)
        env = HybridChessEnv(max_plies=MAX_PLIES)

        cfg = SelfPlayConfig(
            resign_enabled=False,
            max_ply=2,
            move_limit_value_mode="hard",
            simulations=1,
        )
        examples, record = self_play_game(env, agent, cfg)

        assert record.termination_reason == "Max plies reached"
        assert record.ply_count == 2
        assert record.resigned is False
        assert len(examples) == 2

        if record.material_diff > 0:
            expected = {Side.CHESS: 1.0, Side.XIANGQI: -1.0}
        elif record.material_diff < 0:
            expected = {Side.CHESS: -1.0, Side.XIANGQI: 1.0}
        else:
            expected = {Side.CHESS: 0.0, Side.XIANGQI: 0.0}

        for ex in examples:
            assert ex.z == expected[ex.side_to_move]

    def test_move_limit_zero_mode_keeps_draw_targets(self):
        """Compatibility: move-limit with zero mode produces all-zero z."""
        from hybrid.core.env import HybridChessEnv

        agent = self._make_mock_agent([0.0] * 20)
        env = HybridChessEnv(max_plies=MAX_PLIES)

        cfg = SelfPlayConfig(
            resign_enabled=False,
            max_ply=2,
            move_limit_value_mode="zero",
            simulations=1,
        )
        examples, record = self_play_game(env, agent, cfg)

        assert record.termination_reason == "Max plies reached"
        assert len(examples) == 2
        assert all(ex.z == 0.0 for ex in examples)


# ====================================================================
# E) Runner diagnostics aggregation
# ====================================================================

class TestDiagnosticAggregation:
    def test_aggregate_empty(self):
        from hybrid.rl.az_runner import _aggregate_game_records
        result = _aggregate_game_records([])
        assert result == {}

    def test_aggregate_basic(self):
        from hybrid.rl.az_runner import _aggregate_game_records

        records = [
            GameRecord(result="chess_win", termination_reason="Checkmate",
                       ply_count=60, material_diff=5.0),
            GameRecord(result="draw", termination_reason="Max plies reached",
                       ply_count=400, material_diff=-1.0),
            GameRecord(result="draw", termination_reason="Adjudicated draw",
                       ply_count=150, material_diff=0.0),
            GameRecord(result="draw", termination_reason="Threefold repetition",
                       ply_count=120, material_diff=0.0),
            GameRecord(result="xiangqi_win", termination_reason="Resign",
                       ply_count=80, material_diff=-3.0, resigned=True,
                       resign_side="chess"),
        ]
        stats = _aggregate_game_records(records)

        assert stats["sp_games"] == 5
        assert stats["sp_decisive"] == 2
        assert stats["sp_draw_move_limit"] == 1
        assert stats["sp_draw_threefold"] == 1
        assert stats["sp_draw_adjudicated"] == 1
        assert stats["sp_draw_stalemate"] == 0
        assert stats["sp_resign_count"] == 1
        assert stats["sp_adjudicate_avg_ply"] == 150.0
        assert stats["sp_avg_ply"] == (60 + 400 + 150 + 120 + 80) / 5
        assert stats["sp_rootv_min_mean"] == 0.0
        assert stats["sp_low_rootv_steps_sum"] == 0
        assert stats["sp_low_rootv_steps_rate"] == 0.0


# ====================================================================
# F) score_ci tests
# ====================================================================

class TestScoreCI:
    def test_all_draws_tight_ci(self):
        """All draws: score_ci should give a tight CI around 0.5, not [0,1]."""
        from hybrid.rl.az_eval import score_ci
        mean, lo, hi = score_ci(0, 80, 0)
        assert abs(mean - 0.5) < 1e-6
        # 80 draws → CI should be narrow (< 0.25 wide)
        assert hi - lo < 0.25
        assert lo > 0.35
        assert hi < 0.65

    def test_all_wins(self):
        from hybrid.rl.az_eval import score_ci
        mean, lo, hi = score_ci(20, 0, 0)
        assert abs(mean - 1.0) < 1e-6
        assert lo > 0.8

    def test_mixed(self):
        from hybrid.rl.az_eval import score_ci
        mean, lo, hi = score_ci(10, 5, 5)
        # score = (10 + 2.5) / 20 = 0.625
        assert abs(mean - 0.625) < 1e-6
        assert lo < mean < hi

    def test_n_zero(self):
        from hybrid.rl.az_eval import score_ci
        mean, lo, hi = score_ci(0, 0, 0)
        assert mean == 0.5
        assert lo == 0.0
        assert hi == 1.0

    def test_wilson_vs_score_on_all_draws(self):
        """wilson_ci returns [0,1] on all draws; score_ci should be tighter."""
        from hybrid.rl.az_eval import wilson_ci, score_ci
        w_mean, w_lo, w_hi = wilson_ci(0, 0)
        assert w_lo == 0.0 and w_hi == 1.0  # Completely uncertain

        s_mean, s_lo, s_hi = score_ci(0, 80, 0)
        assert (s_hi - s_lo) < (w_hi - w_lo)  # score CI is tighter
