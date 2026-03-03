# -*- coding: utf-8 -*-
"""Basic unit tests: board initialization, move generation, turn switching."""

from hybrid.core.env import HybridChessEnv
from hybrid.core.types import Side
from hybrid.core.board import initial_board


def test_initial_board_has_kings():
    b = initial_board()
    found_chess_king = False
    found_xq_general = False
    for _, _, p in b.iter_pieces():
        if p.side == Side.CHESS and p.kind.name == "KING":
            found_chess_king = True
        if p.side == Side.XIANGQI and p.kind.name == "GENERAL":
            found_xq_general = True
    assert found_chess_king and found_xq_general


def test_env_step_switch_turn():
    env = HybridChessEnv()
    s = env.reset()
    legal = env.legal_moves()
    assert len(legal) > 0
    mv = legal[0]
    s2, r, done, info = env.step(mv)
    assert s2.side_to_move != s.side_to_move
