"""RemotePolicyValueModel: adapts InferenceClient to the PolicyValueModel protocol.

Allows AlphaZeroMiniAgent and self_play_game() to use remote GPU inference
without any code changes. Dirichlet noise is still applied by the agent.

Workers send compact board IDs (10×9 int8) + side byte, NOT full encoded
tensors (14×10×9 float).  Encoding is done server-side on GPU.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import torch

from hybrid.agents.alphazero_stub import PolicyValueModel
from hybrid.core.env import GameState
from hybrid.core.types import Move, Side
from hybrid.rl.az_encoding import board_to_piece_ids
from hybrid.rl.az_selfplay import move_to_action_index
from hybrid.rl.az_inference_server import InferenceClient


class RemotePolicyValueModel(PolicyValueModel):
    """PolicyValueModel backed by a remote InferenceServer (GPU micro-batching)."""

    def __init__(self, client: InferenceClient):
        self.client = client
        self.ipc_wait_s = 0.0
        self.predict_count = 0

    def predict(
        self, state: GameState, legal_moves: List[Move]
    ) -> Tuple[Dict[Move, float], float]:
        if len(legal_moves) == 0:
            return {}, 0.0

        # Compact board representation — ~14× smaller than full encoding
        board_ids = board_to_piece_ids(state.board)  # (10, 9) int8
        side = np.int8(1 if state.side_to_move == Side.CHESS else 0)

        # Build flat action indices for legal moves
        action_indices = np.array(
            [move_to_action_index(mv) for mv in legal_moves],
            dtype=np.uint16,
        )

        # Remote forward pass (encoding happens server-side on GPU)
        import time as _time
        t0 = _time.perf_counter()
        logits_np, value = self.client.predict_raw(board_ids, side, action_indices)
        self.ipc_wait_s += (_time.perf_counter() - t0)
        self.predict_count += 1

        # Numerically stable softmax
        logits_t = torch.from_numpy(logits_np).float()
        logits_t = logits_t - logits_t.max()
        probs = torch.softmax(logits_t, dim=0)

        policy_dict = {mv: probs[i].item() for i, mv in enumerate(legal_moves)}
        return policy_dict, value

    def predict_batch(
        self, leaf_data: List[Tuple[GameState, List[Move]]]
    ) -> List[Tuple[Dict[Move, float], float]]:
        """Batch-predict K leaves in one IPC round-trip.

        Args:
            leaf_data: list of (state, legal_moves) tuples.

        Returns:
            list of (policy_dict, value) tuples, one per leaf.
        """
        K = len(leaf_data)
        if K == 0:
            return []
        if K == 1:
            return [self.predict(leaf_data[0][0], leaf_data[0][1])]

        # Pack K states into stacked arrays
        board_ids_list = []
        sides_list = []
        action_indices_list = []
        all_legal_moves = []

        for state, legal_moves in leaf_data:
            board_ids_list.append(board_to_piece_ids(state.board))
            sides_list.append(1 if state.side_to_move == Side.CHESS else 0)
            action_indices = np.array(
                [move_to_action_index(mv) for mv in legal_moves],
                dtype=np.uint16,
            )
            action_indices_list.append(action_indices)
            all_legal_moves.append(legal_moves)

        board_ids_stack = np.stack(board_ids_list)            # (K, 10, 9)
        sides_stack = np.array(sides_list, dtype=np.int8)     # (K,)

        # One IPC round-trip for K states
        import time as _time
        t0 = _time.perf_counter()
        logits_list, values = self.client.predict_batch_raw(
            board_ids_stack, sides_stack, action_indices_list,
        )
        self.ipc_wait_s += (_time.perf_counter() - t0)
        self.predict_count += K

        # Unpack K results into (policy_dict, value) tuples
        results = []
        for k in range(K):
            logits_t = torch.from_numpy(logits_list[k]).float()
            logits_t = logits_t - logits_t.max()
            probs = torch.softmax(logits_t, dim=0)
            legal_moves = all_legal_moves[k]
            policy_dict = {mv: probs[i].item() for i, mv in enumerate(legal_moves)}
            results.append((policy_dict, float(values[k])))

        return results
