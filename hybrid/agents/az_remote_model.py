# -*- coding: utf-8 -*-
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
        logits_np, value = self.client.predict_raw(board_ids, side, action_indices)

        # Numerically stable softmax
        logits_t = torch.from_numpy(logits_np).float()
        logits_t = logits_t - logits_t.max()
        probs = torch.softmax(logits_t, dim=0)

        policy_dict = {mv: probs[i].item() for i, mv in enumerate(legal_moves)}
        return policy_dict, value
