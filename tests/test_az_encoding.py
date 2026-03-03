# -*- coding: utf-8 -*-
"""AlphaZero encoding layer unit tests.

Verifies:
  1. encode_state output shape is correct
  2. Piece channel nonzero count matches actual piece count
  3. move_to_plane produces valid plane indices for all legal moves
  4. extract_policy_logits output length matches legal_moves count
"""

import torch

from hybrid.core.env import HybridChessEnv, GameState
from hybrid.core.board import initial_board
from hybrid.core.types import Side
from hybrid.core.rules import generate_legal_moves
from hybrid.rl.az_encoding import (
    encode_state,
    move_to_plane,
    extract_policy_logits,
    NUM_STATE_CHANNELS,
    TOTAL_POLICY_PLANES,
)
from hybrid.core.config import BOARD_H, BOARD_W


def test_encode_state_shape():
    """encode_state output shape should be (14, 10, 9)."""
    env = HybridChessEnv()
    state = env.reset()
    tensor = encode_state(state)
    assert tensor.shape == (NUM_STATE_CHANNELS, BOARD_H, BOARD_W), \
        f"Expected shape ({NUM_STATE_CHANNELS}, {BOARD_H}, {BOARD_W}), got {tensor.shape}"


def test_encode_state_nonzero_count():
    """Piece channels (0-12) nonzero count should equal total pieces on board.

    Channel 13 is the side-to-move indicator and is excluded from piece counting.
    """
    board = initial_board()
    state = GameState(board=board, side_to_move=Side.CHESS)
    tensor = encode_state(state)

    piece_count = sum(1 for _ in board.iter_pieces())
    nonzero = (tensor[:13] != 0).sum().item()

    assert nonzero == piece_count, \
        f"Expected {piece_count} nonzero positions, got {nonzero}"


def test_encode_state_side_to_move_channel():
    """Side-to-move channel: all 1s when Chess moves, all 0s when Xiangqi moves."""
    board = initial_board()

    state_chess = GameState(board=board, side_to_move=Side.CHESS)
    t_chess = encode_state(state_chess)
    assert t_chess[13].sum().item() == BOARD_H * BOARD_W, \
        "Side-to-move channel should be all 1s when Chess moves"

    state_xq = GameState(board=board, side_to_move=Side.XIANGQI)
    t_xq = encode_state(state_xq)
    assert t_xq[13].sum().item() == 0, \
        "Side-to-move channel should be all 0s when Xiangqi moves"


def test_move_to_plane_legal_range():
    """All initial legal moves should produce plane indices in [0, 92)."""
    board = initial_board()
    legal = generate_legal_moves(board, Side.CHESS)
    assert len(legal) > 0, "Initial position should have legal moves"

    for mv in legal:
        plane_idx, fy, fx = move_to_plane(mv)
        assert 0 <= plane_idx < TOTAL_POLICY_PLANES, \
            f"plane_idx={plane_idx} out of range [0, {TOTAL_POLICY_PLANES}), move={mv}"
        assert 0 <= fy < BOARD_H, f"fy={fy} out of board range"
        assert 0 <= fx < BOARD_W, f"fx={fx} out of board range"


def test_move_to_plane_xiangqi_legal():
    """Xiangqi legal moves should all map correctly."""
    board = initial_board()
    legal = generate_legal_moves(board, Side.XIANGQI)
    assert len(legal) > 0

    for mv in legal:
        plane_idx, fy, fx = move_to_plane(mv)
        assert 0 <= plane_idx < TOTAL_POLICY_PLANES, \
            f"plane_idx={plane_idx} out of range, move={mv}"


def test_extract_policy_logits_shape():
    """extract_policy_logits should return logits with length == len(legal_moves)."""
    board = initial_board()
    legal = generate_legal_moves(board, Side.CHESS)

    policy_planes = torch.randn(TOTAL_POLICY_PLANES, BOARD_H, BOARD_W)
    logits = extract_policy_logits(policy_planes, legal)

    assert logits.shape == (len(legal),), \
        f"Expected shape ({len(legal)},), got {logits.shape}"


def test_extract_policy_logits_empty():
    """Empty legal_moves should return empty tensor without error."""
    policy_planes = torch.randn(TOTAL_POLICY_PLANES, BOARD_H, BOARD_W)
    logits = extract_policy_logits(policy_planes, [])
    assert logits.shape == (0,)
