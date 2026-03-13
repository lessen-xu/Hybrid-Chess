"""Env-level comparison: Python backend vs C++ backend.

Runs 100 random games through HybridChessEnv with both use_cpp=False and
use_cpp=True, asserting identical legal_moves, reward, done, and terminal
info at every ply.
"""

from __future__ import annotations
import random
import pytest
from hybrid.core.env import HybridChessEnv
from hybrid.core.types import Move


def _move_key(m: Move):
    promo = m.promotion.name if m.promotion is not None else ""
    return (m.fx, m.fy, m.tx, m.ty, promo)


class TestEnvCppVsPython:
    """Run random games through both backends and compare every step."""

    @pytest.mark.parametrize("game_id", range(100))
    def test_env_game(self, game_id: int):
        py_env = HybridChessEnv(use_cpp=False)
        cpp_env = HybridChessEnv(use_cpp=True)

        random.seed(game_id)
        py_env.reset()
        random.seed(game_id)
        cpp_env.reset()

        for ply in range(400):
            py_moves = sorted(py_env.legal_moves(), key=_move_key)
            cpp_moves = sorted(cpp_env.legal_moves(), key=_move_key)

            py_set = [_move_key(m) for m in py_moves]
            cpp_set = [_move_key(m) for m in cpp_moves]

            assert py_set == cpp_set, (
                f"Game {game_id}, ply {ply}: move mismatch\n"
                f"  py_only={set(py_set) - set(cpp_set)}\n"
                f"  cpp_only={set(cpp_set) - set(py_set)}"
            )

            if len(py_moves) == 0:
                break

            move = random.choice(py_moves)

            py_state, py_reward, py_done, py_info = py_env.step(move)
            cpp_state, cpp_reward, cpp_done, cpp_info = cpp_env.step(move)

            assert py_reward == cpp_reward, (
                f"Game {game_id}, ply {ply}: reward mismatch "
                f"py={py_reward} cpp={cpp_reward}"
            )
            assert py_done == cpp_done, (
                f"Game {game_id}, ply {ply}: done mismatch "
                f"py={py_done} cpp={cpp_done}"
            )
            assert py_info.status == cpp_info.status, (
                f"Game {game_id}, ply {ply}: status mismatch "
                f"py={py_info.status} cpp={cpp_info.status}"
            )

            if py_done:
                break


class TestEnvCppBasic:
    """Quick sanity checks for use_cpp mode."""

    def test_reset_returns_valid_state(self):
        env = HybridChessEnv(use_cpp=True)
        state = env.reset()
        assert state is not None
        assert state.ply == 0

    def test_legal_moves_nonempty_at_start(self):
        env = HybridChessEnv(use_cpp=True)
        env.reset()
        moves = env.legal_moves()
        assert len(moves) > 0

    def test_step_returns_correct_types(self):
        env = HybridChessEnv(use_cpp=True)
        env.reset()
        move = env.legal_moves()[0]
        state, reward, done, info = env.step(move)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert hasattr(info, 'status')
        assert hasattr(info, 'reason')
