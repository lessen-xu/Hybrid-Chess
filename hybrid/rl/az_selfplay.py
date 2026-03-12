# -*- coding: utf-8 -*-
"""AlphaZero self-play data generation.

Produces training samples (state, π, z) where:
- state: encoded board position
- π: MCTS visit distribution (sparse format: indices + probs)
- z: terminal outcome from the sample's side-to-move perspective

Includes resign and draw adjudication mechanisms for self-play only.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from hybrid.core.env import HybridChessEnv, GameState
from hybrid.core.types import Move, Side, PieceKind
from hybrid.rl.az_encoding import (
    encode_state, move_to_plane,
    TOTAL_POLICY_PLANES, NUM_STATE_CHANNELS,
)
from hybrid.core.config import BOARD_H, BOARD_W, MAX_PLIES


# Action space: 92 planes × 10 rows × 9 cols = 8,280
ACTION_SPACE_SIZE = TOTAL_POLICY_PLANES * BOARD_H * BOARD_W  # 8280

# ====================================================================
# Material values (for move-limit truncation)
# ====================================================================
PIECE_VALUES = {
    PieceKind.PAWN: 1.0, PieceKind.KNIGHT: 3.0, PieceKind.BISHOP: 3.0,
    PieceKind.ROOK: 5.0, PieceKind.QUEEN: 9.0, PieceKind.KING: 0.0,
    PieceKind.SOLDIER: 1.0, PieceKind.HORSE: 3.0, PieceKind.ELEPHANT: 2.0,
    PieceKind.ADVISOR: 2.0, PieceKind.CHARIOT: 5.0, PieceKind.CANNON: 4.5,
    PieceKind.GENERAL: 0.0,
}


def compute_material_diff(board) -> float:
    """Material difference: Chess total - Xiangqi total. Positive = Chess advantage."""
    diff = 0.0
    for x, y, piece in board.iter_pieces():
        val = PIECE_VALUES.get(piece.kind, 0.0)
        if piece.side == Side.CHESS:
            diff += val
        else:
            diff -= val
    return diff


def material_diff_to_value(
    material_diff: float,
    mode: str = "penalty",
    scale: float = 4.0,
) -> float:
    """Convert material diff to training value (for move-limit truncation only)."""
    if mode == "zero":
        return 0.0
    if mode == "penalty":
        # Small negative penalty: punish the model for not finishing the game.
        # Both sides see z = -0.1, discouraging stalling / material hoarding.
        return -0.1
    if mode == "hard":
        if material_diff > 0:
            return 1.0
        if material_diff < 0:
            return -1.0
        return 0.0
    if mode == "soft":
        safe_scale = max(float(scale), 1e-6)
        return float(np.tanh(material_diff / safe_scale))
    raise ValueError(f"Unknown move_limit value mode: {mode!r}")


def summarize_root_values(
    root_values: List[float],
    low_threshold: float,
) -> Tuple[float, float, int, int]:
    """Summarize root value trajectory for diagnostics."""
    if not root_values:
        return 0.0, 0.0, 0, 0
    arr = np.asarray(root_values, dtype=np.float32)
    return (
        float(arr.min()),
        float(np.percentile(arr, 5)),
        int((arr <= low_threshold).sum()),
        int(arr.size),
    )


def move_to_action_index(mv: Move) -> int:
    """Map Move to flat action index: plane_idx * 90 + fy * 9 + fx."""
    plane_idx, fy, fx = move_to_plane(mv)
    return plane_idx * (BOARD_H * BOARD_W) + fy * BOARD_W + fx


@dataclass
class Example:
    """One training sample (state, sparse π, z)."""
    state: np.ndarray       # uint8, (C, 10, 9)
    pi_indices: np.ndarray  # uint16, (L,) — flat action indices
    pi_probs: np.ndarray    # float32, (L,) — MCTS visit probabilities
    side_to_move: Side
    z: float = 0.0          # terminal outcome, assigned after game ends


@dataclass
class GameRecord:
    """Per-game metadata for diagnostics."""
    result: str = "draw"
    termination_reason: str = ""
    ply_count: int = 0
    material_diff: float = 0.0
    resigned: bool = False
    resign_side: Optional[str] = None
    rootv_min: float = 0.0
    rootv_p05: float = 0.0
    low_rootv_steps: int = 0
    rootv_steps: int = 0
    # --- Telemetry for evaluation protocol ---
    winner_side: Optional[str] = None       # "chess" / "xiangqi" / None (draw)
    legal_move_counts: List[int] = field(default_factory=list)  # per-ply |legal_moves|


@dataclass
class SelfPlayConfig:
    """Self-play configuration."""
    temperature: float = 1.0
    temp_cutoff_ply: int = 30     # Use T=1.0 for first N plies, then T→0
    simulations: int = 50
    max_ply: int = MAX_PLIES
    move_limit_value_mode: str = "penalty"
    move_limit_value_scale: float = 4.0
    # Resign
    resign_enabled: bool = True
    resign_threshold: float = -0.95
    resign_min_ply: int = 40
    resign_patience: int = 3
    # Draw adjudication (self-play only): consecutive |root_value| <= thr → draw
    draw_adjudicate_enabled: bool = True
    draw_adjudicate_min_ply: int = 60
    draw_adjudicate_patience: int = 15
    draw_adjudicate_value_abs_thr: float = 0.08
    # Reward shaping (optional). Called as reward_shaper(state, move, next_state).
    # Return value is added to z during backfill. Default: None (pure AlphaZero).
    reward_shaper: Optional["Callable"] = None


def self_play_game(
    env: HybridChessEnv,
    agent,  # AlphaZeroMiniAgent
    cfg: SelfPlayConfig = SelfPlayConfig(),
    initial_state: Optional['GameState'] = None,
) -> Tuple[List[Example], GameRecord]:
    """Run one self-play game. Returns (examples, record).

    If initial_state is provided (board, side_to_move), start from that
    position instead of the standard opening (for endgame curriculum).
    """
    if hasattr(env, "set_max_plies"):
        env.set_max_plies(cfg.max_ply)
    elif hasattr(env, "max_plies"):
        env.max_plies = cfg.max_ply

    if initial_state is not None:
        state = env.reset_from_board(initial_state.board, initial_state.side_to_move)
    else:
        state = env.reset()
    examples: List[Example] = []
    info = None
    legal_move_counts: List[int] = []

    resign_counter = 0
    draw_adjudicate_counter = 0
    root_values: List[float] = []

    while True:
        legal_moves = env.legal_moves()
        if len(legal_moves) == 0:
            break
        legal_move_counts.append(len(legal_moves))

        current_ply = state.ply
        temp = cfg.temperature if current_ply < cfg.temp_cutoff_ply else 0.0

        chosen_move, pi_dict, root_value = agent.select_move_with_pi(
            state, legal_moves, temperature=temp
        )
        root_values.append(float(root_value))
        current_material_diff = compute_material_diff(state.board)

        # Record training sample (sparse π)
        state_np = encode_state(state).numpy().astype(np.uint8)
        indices = []
        probs = []
        for mv, prob in pi_dict.items():
            if prob > 0:
                indices.append(move_to_action_index(mv))
                probs.append(prob)

        example = Example(
            state=state_np,
            pi_indices=np.array(indices, dtype=np.uint16),
            pi_probs=np.array(probs, dtype=np.float32),
            side_to_move=state.side_to_move,
            z=0.0,
        )
        examples.append(example)

        # Resign check
        if cfg.resign_enabled and current_ply >= cfg.resign_min_ply:
            if root_value <= cfg.resign_threshold:
                resign_counter += 1
            else:
                resign_counter = 0

            if resign_counter >= cfg.resign_patience:
                loser = state.side_to_move
                winner = loser.opponent()
                rootv_min, rootv_p05, low_rootv_steps, rootv_steps = summarize_root_values(
                    root_values, cfg.resign_threshold
                )
                for ex in examples:
                    ex.z = 1.0 if ex.side_to_move == winner else -1.0
                _winner_side = "chess" if winner == Side.CHESS else "xiangqi"
                record = GameRecord(
                    result="chess_win" if winner == Side.CHESS else "xiangqi_win",
                    termination_reason="Resign",
                    ply_count=state.ply,
                    material_diff=current_material_diff,
                    resigned=True,
                    resign_side="chess" if loser == Side.CHESS else "xiangqi",
                    rootv_min=rootv_min, rootv_p05=rootv_p05,
                    low_rootv_steps=low_rootv_steps, rootv_steps=rootv_steps,
                    winner_side=_winner_side,
                    legal_move_counts=legal_move_counts,
                )
                return examples, record

        # Draw adjudication check
        if cfg.draw_adjudicate_enabled and current_ply >= cfg.draw_adjudicate_min_ply:
            if abs(root_value) <= cfg.draw_adjudicate_value_abs_thr:
                draw_adjudicate_counter += 1
            else:
                draw_adjudicate_counter = 0

            if draw_adjudicate_counter >= cfg.draw_adjudicate_patience:
                rootv_min, rootv_p05, low_rootv_steps, rootv_steps = summarize_root_values(
                    root_values, cfg.resign_threshold
                )
                for ex in examples:
                    ex.z = 0.0
                record = GameRecord(
                    result="draw",
                    termination_reason="Adjudicated draw",
                    ply_count=state.ply,
                    material_diff=current_material_diff,
                    resigned=False,
                    rootv_min=rootv_min, rootv_p05=rootv_p05,
                    low_rootv_steps=low_rootv_steps, rootv_steps=rootv_steps,
                    winner_side=None,
                    legal_move_counts=legal_move_counts,
                )
                return examples, record

        state, reward, done, info = env.step(chosen_move)
        if done:
            break

    # Assign z based on terminal result
    winner: Optional[Side] = (
        info.winner if (info is not None and hasattr(info, "winner")) else None
    )
    reason = info.reason if (info is not None and hasattr(info, "reason")) else ""
    material_diff = compute_material_diff(state.board)
    rootv_min, rootv_p05, low_rootv_steps, rootv_steps = summarize_root_values(
        root_values, cfg.resign_threshold
    )
    move_limit_value: Optional[float] = None
    if winner is None and reason == "Max plies reached":
        move_limit_value = material_diff_to_value(
            material_diff, mode=cfg.move_limit_value_mode, scale=cfg.move_limit_value_scale,
        )

    for ex in examples:
        if winner is None:
            if move_limit_value is None:
                ex.z = 0.0
            elif ex.side_to_move == Side.CHESS:
                ex.z = move_limit_value
            else:
                ex.z = -move_limit_value
        elif ex.side_to_move == winner:
            ex.z = 1.0
        else:
            ex.z = -1.0

    result_str = "draw"
    winner_side_str: Optional[str] = None
    if winner == Side.CHESS:
        result_str = "chess_win"
        winner_side_str = "chess"
    elif winner == Side.XIANGQI:
        result_str = "xiangqi_win"
        winner_side_str = "xiangqi"

    record = GameRecord(
        result=result_str, termination_reason=reason,
        ply_count=state.ply, material_diff=material_diff, resigned=False,
        rootv_min=rootv_min, rootv_p05=rootv_p05,
        low_rootv_steps=low_rootv_steps, rootv_steps=rootv_steps,
        winner_side=winner_side_str,
        legal_move_counts=legal_move_counts,
    )

    return examples, record
