"""Gymnasium-compatible environment wrapper for Hybrid Chess.

Usage:
    import gymnasium as gym
    import hybrid.gym_env  # registers the env

    env = gym.make("HybridChess-v0")
    obs, info = env.reset()
    action = env.action_space.sample()  # random (likely illegal)
    legal = info["legal_actions"]       # list of valid action indices
    obs, reward, terminated, truncated, info = env.step(legal[0])
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError(
        "gymnasium is required for hybrid.gym_env. "
        "Install with: pip install hybrid-chess[gym]"
    )

from hybrid.core.env import HybridChessEnv, GameState
from hybrid.core.types import Side, Move, PieceKind
from hybrid.core.config import BOARD_W, BOARD_H

# ── Action encoding (mirrors az_encoding.py) ──
# We reuse the same 92×10×9 = 8280 flat action space as the AlphaZero pipeline.

_SLIDE_DIRECTIONS = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
]
_DIR_LOOKUP = {d: i for i, d in enumerate(_SLIDE_DIRECTIONS)}

_KNIGHT_DELTAS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]
_KNIGHT_LOOKUP = {d: i for i, d in enumerate(_KNIGHT_DELTAS)}

_PROMO_KINDS = [PieceKind.QUEEN, PieceKind.ROOK, PieceKind.BISHOP, PieceKind.KNIGHT]
_PROMO_DX_VALUES = [-1, 0, 1]
_PROMO_LOOKUP = {}
for _di, _dx in enumerate(_PROMO_DX_VALUES):
    for _pi, _pk in enumerate(_PROMO_KINDS):
        _PROMO_LOOKUP[(_dx, _pk)] = _di * len(_PROMO_KINDS) + _pi

TOTAL_POLICY_PLANES = 92  # 72 slide + 8 knight + 12 promo
TOTAL_ACTIONS = TOTAL_POLICY_PLANES * BOARD_H * BOARD_W  # 92 * 10 * 9 = 8280

# Piece channel map (same as az_encoding.py)
_PIECE_CHANNELS = {
    PieceKind.KING: 0, PieceKind.QUEEN: 1, PieceKind.ROOK: 2,
    PieceKind.BISHOP: 3, PieceKind.KNIGHT: 4, PieceKind.PAWN: 5,
    PieceKind.GENERAL: 6, PieceKind.ADVISOR: 7, PieceKind.ELEPHANT: 8,
    PieceKind.HORSE: 9, PieceKind.CHARIOT: 10, PieceKind.CANNON: 11,
    PieceKind.SOLDIER: 12,
}
NUM_STATE_CHANNELS = 14


def _move_to_action(mv: Move) -> int:
    """Convert a Move to a flat action index in [0, 8280)."""
    dx = mv.tx - mv.fx
    dy = mv.ty - mv.fy

    if mv.promotion is not None:
        plane = 80 + _PROMO_LOOKUP[(dx, mv.promotion)]
    elif (dx, dy) in _KNIGHT_LOOKUP:
        plane = 72 + _KNIGHT_LOOKUP[(dx, dy)]
    else:
        # Sliding move
        if dx == 0:
            dist, dx_u, dy_u = abs(dy), 0, (1 if dy > 0 else -1)
        elif dy == 0:
            dist, dx_u, dy_u = abs(dx), (1 if dx > 0 else -1), 0
        else:
            dist = abs(dx)
            dx_u = 1 if dx > 0 else -1
            dy_u = 1 if dy > 0 else -1
        plane = _DIR_LOOKUP[(dx_u, dy_u)] * 9 + (dist - 1)

    return plane * (BOARD_H * BOARD_W) + mv.fy * BOARD_W + mv.fx


def _action_to_move(action: int) -> Move:
    """Convert a flat action index back to a Move (without full validation)."""
    plane = action // (BOARD_H * BOARD_W)
    rem = action % (BOARD_H * BOARD_W)
    fy = rem // BOARD_W
    fx = rem % BOARD_W

    if plane < 72:
        # Sliding
        dir_idx = plane // 9
        dist = (plane % 9) + 1
        dx_u, dy_u = _SLIDE_DIRECTIONS[dir_idx]
        tx, ty = fx + dx_u * dist, fy + dy_u * dist
        return Move(fx, fy, tx, ty, None)
    elif plane < 80:
        # Knight
        ki = plane - 72
        dx, dy = _KNIGHT_DELTAS[ki]
        return Move(fx, fy, fx + dx, fy + dy, None)
    else:
        # Promotion
        pi = plane - 80
        dx_i = pi // 4
        pk_i = pi % 4
        dx = _PROMO_DX_VALUES[dx_i]
        pk = _PROMO_KINDS[pk_i]
        return Move(fx, fy, fx + dx, fy + 1, pk)


def _encode_obs(state: GameState) -> np.ndarray:
    """Encode GameState as (14, 10, 9) float32 numpy array."""
    obs = np.zeros((NUM_STATE_CHANNELS, BOARD_H, BOARD_W), dtype=np.float32)
    for x, y, piece in state.board.iter_pieces():
        ch = _PIECE_CHANNELS[piece.kind]
        obs[ch, y, x] = 1.0
    if state.side_to_move == Side.CHESS:
        obs[13, :, :] = 1.0
    return obs


class HybridChessGymEnv(gym.Env):
    """Gymnasium wrapper for Hybrid Chess.

    Observation: (14, 10, 9) float32 binary planes.
    Action: Discrete(8280) — flat index into 92×10×9 policy space.
    Reward: +1 win, -1 loss, 0 draw/ongoing, from mover's perspective.

    The info dict always contains 'legal_actions' (list of valid action indices).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, max_plies: int = 400, use_cpp: bool = False,
                 render_mode: Optional[str] = None):
        super().__init__()
        self.env = HybridChessEnv(max_plies=max_plies, use_cpp=use_cpp)
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(NUM_STATE_CHANNELS, BOARD_H, BOARD_W),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)

        # Cache for move ↔ action mapping
        self._legal_moves: List[Move] = []
        self._legal_actions: List[int] = []
        self._action_to_move_map: Dict[int, Move] = {}

    def _update_legal_cache(self):
        self._legal_moves = self.env.legal_moves()
        self._action_to_move_map = {}
        self._legal_actions = []
        for mv in self._legal_moves:
            a = _move_to_action(mv)
            self._legal_actions.append(a)
            self._action_to_move_map[a] = mv

    def _make_info(self) -> Dict[str, Any]:
        return {
            "legal_actions": list(self._legal_actions),
            "side_to_move": self.env.state.side_to_move.name.lower(),
            "ply": self.env.state.ply,
        }

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        state = self.env.reset()
        self._update_legal_cache()
        return _encode_obs(state), self._make_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if action not in self._action_to_move_map:
            raise ValueError(
                f"Illegal action {action}. Legal actions: {self._legal_actions}"
            )

        mv = self._action_to_move_map[action]
        state, reward, done, info = self.env.step(mv)
        self._update_legal_cache()

        terminated = done
        truncated = False  # max-ply draws are terminal, not truncation
        gym_info = self._make_info()
        gym_info["game_info"] = {
            "status": info.status,
            "winner": info.winner.name.lower() if info.winner else None,
            "reason": info.reason,
        }

        return _encode_obs(state), reward, terminated, truncated, gym_info

    def legal_actions(self) -> List[int]:
        """Return list of legal action indices."""
        return list(self._legal_actions)

    def render(self) -> Optional[str]:
        if self.render_mode == "ansi":
            from hybrid.core.render import render_board
            return render_board(self.env.state.board)
        return None


# ── Register with Gymnasium ──
gym.register(
    id="HybridChess-v0",
    entry_point="hybrid.gym_env:HybridChessGymEnv",
    kwargs={"max_plies": 400, "use_cpp": False},
)
