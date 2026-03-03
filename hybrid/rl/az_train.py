# -*- coding: utf-8 -*-
"""AlphaZero-Mini training logic.

Loss = policy_loss (cross-entropy with MCTS π) + value_loss (MSE with terminal z).
L2 regularization is handled by optimizer weight_decay.
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .az_replay import ReplayBuffer
from .az_selfplay import ACTION_SPACE_SIZE
from .az_network import PolicyValueNet
from hybrid.core.config import BOARD_H, BOARD_W


def train_one_epoch(
    net: PolicyValueNet,
    buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 256,
    max_steps: int = 0,
    grad_clip: float = 1.0,
) -> dict:
    """Train one epoch over the replay buffer.

    Returns dict with 'policy_loss', 'value_loss', 'total_loss', 'steps'.
    """
    net.train()
    rng = np.random.default_rng()

    if max_steps <= 0:
        max_steps = max(1, len(buffer) // batch_size)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_loss_sum = 0.0

    for step_i in range(max_steps):
        states_np, pi_indices_list, pi_probs_list, z_np = buffer.sample_batch(
            batch_size, rng
        )

        B = len(z_np)

        states_t = torch.from_numpy(states_np).to(device)
        z_t = torch.from_numpy(z_np).to(device)

        policy_logits, value = net(states_t)

        # Policy loss: masked cross-entropy over legal moves
        policy_flat = policy_logits.view(B, -1)  # (B, 8280)

        policy_losses = []
        for i in range(B):
            indices = torch.from_numpy(
                pi_indices_list[i].astype(np.int64)
            ).to(device)
            target_probs = torch.from_numpy(
                pi_probs_list[i]
            ).to(device)

            if len(indices) == 0:
                continue

            legal_logits = policy_flat[i, indices]
            log_probs = F.log_softmax(legal_logits, dim=0)
            ce = -(target_probs * log_probs).sum()
            policy_losses.append(ce)

        if policy_losses:
            policy_loss = torch.stack(policy_losses).mean()
        else:
            policy_loss = torch.tensor(0.0, device=device)

        # Value loss: MSE
        value_loss = F.mse_loss(value.squeeze(-1), z_t)

        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_loss_sum += loss.item()

    steps = max_steps
    return {
        "policy_loss": total_policy_loss / steps if steps > 0 else 0,
        "value_loss": total_value_loss / steps if steps > 0 else 0,
        "total_loss": total_loss_sum / steps if steps > 0 else 0,
        "steps": steps,
    }
