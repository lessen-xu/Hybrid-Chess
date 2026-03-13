"""AlphaZero state and action encoding.

State encoding (14 channels):
  - Channels 0-12: one binary plane per PieceKind (piece presence at (y,x)).
    PieceKind implicitly encodes side (Chess kinds != Xiangqi kinds).
  - Channel 13: side-to-move (all 1s = Chess, all 0s = Xiangqi).

Action encoding (92 move-planes):
  - Planes 0-71: sliding moves (8 directions × 9 distances).
  - Planes 72-79: knight/horse jumps (8 L-shaped deltas).
  - Planes 80-91: pawn promotions (3 dx × 4 promo kinds).
"""

from __future__ import annotations
from typing import List, Tuple, Dict

import numpy as np
import torch

from hybrid.core.env import GameState
from hybrid.core.types import Move, PieceKind, Side
from hybrid.core.config import BOARD_W, BOARD_H
# State encoding

PIECE_CHANNELS: Dict[PieceKind, int] = {
    PieceKind.KING:     0,
    PieceKind.QUEEN:    1,
    PieceKind.ROOK:     2,
    PieceKind.BISHOP:   3,
    PieceKind.KNIGHT:   4,
    PieceKind.PAWN:     5,
    PieceKind.GENERAL:  6,
    PieceKind.ADVISOR:  7,
    PieceKind.ELEPHANT: 8,
    PieceKind.HORSE:    9,
    PieceKind.CHARIOT:  10,
    PieceKind.CANNON:   11,
    PieceKind.SOLDIER:  12,
}

NUM_PIECE_CHANNELS = 13
SIDE_TO_MOVE_CHANNEL = 13
NUM_STATE_CHANNELS = 14


def encode_state_cpu_legacy(state: GameState) -> torch.Tensor:
    """Encode GameState as (C, H, W) = (14, 10, 9) float32 tensor.

    Original CPU implementation kept as reference/baseline.
    """
    tensor = torch.zeros(NUM_STATE_CHANNELS, BOARD_H, BOARD_W, dtype=torch.float32)

    for x, y, piece in state.board.iter_pieces():
        ch = PIECE_CHANNELS[piece.kind]
        tensor[ch, y, x] = 1.0

    if state.side_to_move == Side.CHESS:
        tensor[SIDE_TO_MOVE_CHANNEL, :, :] = 1.0

    return tensor


def encode_state(state: GameState) -> torch.Tensor:
    """Encode GameState as (C, H, W) = (14, 10, 9) float32 tensor.

    Thin wrapper over encode_state_cpu_legacy for backward compatibility.
    """
    return encode_state_cpu_legacy(state)


def board_to_piece_ids(board) -> np.ndarray:
    """Convert a Board to a compact (10, 9) int8 grid of piece channel IDs.

    Returns:
        (BOARD_H, BOARD_W) int8 array.  Values 0–12 map to PIECE_CHANNELS;
        -1 = empty square.
    """
    ids = np.full((BOARD_H, BOARD_W), -1, dtype=np.int8)
    for x, y, piece in board.iter_pieces():
        ids[y, x] = PIECE_CHANNELS[piece.kind]
    return ids


def encode_batch_gpu(
    piece_ids: torch.Tensor,
    sides: torch.Tensor,
    device: torch.device,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """GPU batch one-hot encoding of board states.

    Args:
        piece_ids: (B, 10, 9) int8/long — channel IDs 0–12, -1 = empty.
        sides:     (B,) int8/long — 0 = Xiangqi's turn, 1 = Chess's turn.
        device:    target device (cuda / cpu).
        out:       optional pre-allocated (B, 14, 10, 9) float32 tensor.
                   When provided, writes in-place (zero-allocation hot path).
                   When None, allocates a fresh tensor.

    Returns:
        (B, 14, 10, 9) float32 tensor equivalent to stacking
        encode_state_cpu_legacy for each board.
    """
    B, H, W = piece_ids.shape

    if out is None:
        out = torch.zeros((B, NUM_STATE_CHANNELS, H, W), device=device, dtype=torch.float32)
    else:
        out.zero_()  # Critical: clear previous batch's residual data

    # Mask of occupied squares
    mask = piece_ids >= 0  # (B, H, W)

    # Clamp -1→0 to prevent scatter_ index error
    valid_ids = piece_ids.clamp(min=0).unsqueeze(1).long()  # (B, 1, H, W)

    # Scatter one-hot along channel dim
    out.scatter_(dim=1, index=valid_ids, src=mask.unsqueeze(1).float())

    # Side-to-move channel (index 13): 1.0 when Chess moves
    out[:, SIDE_TO_MOVE_CHANNEL, :, :] = sides.view(B, 1, 1).expand(B, H, W).float()

    return out
# Action encoding

# Sliding planes (0..71): 8 directions × 9 distances
_SLIDE_DIRECTIONS = [
    (0, 1),    # 0: N
    (1, 1),    # 1: NE
    (1, 0),    # 2: E
    (1, -1),   # 3: SE
    (0, -1),   # 4: S
    (-1, -1),  # 5: SW
    (-1, 0),   # 6: W
    (-1, 1),   # 7: NW
]

_DIR_LOOKUP: Dict[Tuple[int, int], int] = {d: i for i, d in enumerate(_SLIDE_DIRECTIONS)}

NUM_SLIDE_PLANES = 72
SLIDE_PLANE_OFFSET = 0

# Knight planes (72..79): 8 L-shaped deltas
_KNIGHT_DELTAS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]
_KNIGHT_LOOKUP: Dict[Tuple[int, int], int] = {d: i for i, d in enumerate(_KNIGHT_DELTAS)}

NUM_KNIGHT_PLANES = 8
KNIGHT_PLANE_OFFSET = NUM_SLIDE_PLANES  # 72

# Promotion planes (80..91): dx in {-1,0,+1} × promo in {Q,R,B,N}
_PROMO_KINDS = [PieceKind.QUEEN, PieceKind.ROOK, PieceKind.BISHOP, PieceKind.KNIGHT]
_PROMO_DX_VALUES = [-1, 0, 1]

_PROMO_LOOKUP: Dict[Tuple[int, PieceKind], int] = {}
for _di, _dx in enumerate(_PROMO_DX_VALUES):
    for _pi, _pk in enumerate(_PROMO_KINDS):
        _PROMO_LOOKUP[(_dx, _pk)] = _di * len(_PROMO_KINDS) + _pi

NUM_PROMO_PLANES = 12
PROMO_PLANE_OFFSET = KNIGHT_PLANE_OFFSET + NUM_KNIGHT_PLANES  # 80

TOTAL_POLICY_PLANES = NUM_SLIDE_PLANES + NUM_KNIGHT_PLANES + NUM_PROMO_PLANES  # 92
# Action encoding functions

def move_to_plane(mv: Move) -> Tuple[int, int, int]:
    """Map a Move to (plane_idx, fy, fx). Policy output is indexed by origin square."""
    dx = mv.tx - mv.fx
    dy = mv.ty - mv.fy

    # Promotion move
    if mv.promotion is not None:
        key = (dx, mv.promotion)
        assert key in _PROMO_LOOKUP, f"Invalid promotion: dx={dx}, promo={mv.promotion}"
        plane_idx = PROMO_PLANE_OFFSET + _PROMO_LOOKUP[key]
        return (plane_idx, mv.fy, mv.fx)

    # Knight/horse move
    delta = (dx, dy)
    if delta in _KNIGHT_LOOKUP:
        plane_idx = KNIGHT_PLANE_OFFSET + _KNIGHT_LOOKUP[delta]
        return (plane_idx, mv.fy, mv.fx)

    # Sliding move (orthogonal or diagonal)
    if dx == 0 and dy == 0:
        raise ValueError(f"Zero-displacement move: {mv}")

    if dx == 0:
        dist = abs(dy)
        dx_unit, dy_unit = 0, (1 if dy > 0 else -1)
    elif dy == 0:
        dist = abs(dx)
        dx_unit, dy_unit = (1 if dx > 0 else -1), 0
    elif abs(dx) == abs(dy):
        dist = abs(dx)
        dx_unit = 1 if dx > 0 else -1
        dy_unit = 1 if dy > 0 else -1
    else:
        raise ValueError(
            f"Move {mv} is neither sliding nor knight (dx={dx}, dy={dy})."
        )

    dir_idx = _DIR_LOOKUP[(dx_unit, dy_unit)]
    assert 1 <= dist <= 9, f"Slide distance {dist} out of range [1, 9]"
    plane_idx = SLIDE_PLANE_OFFSET + dir_idx * 9 + (dist - 1)
    return (plane_idx, mv.fy, mv.fx)


def extract_policy_logits(
    policy_planes: torch.Tensor,
    legal_moves: List[Move],
) -> torch.Tensor:
    """Extract logits for legal moves from policy planes (92, 10, 9). Returns shape (N,)."""
    if len(legal_moves) == 0:
        return torch.zeros(0, dtype=torch.float32)

    logits = torch.empty(len(legal_moves), dtype=torch.float32)
    for i, mv in enumerate(legal_moves):
        plane_idx, fy, fx = move_to_plane(mv)
        logits[i] = policy_planes[plane_idx, fy, fx]
    return logits
