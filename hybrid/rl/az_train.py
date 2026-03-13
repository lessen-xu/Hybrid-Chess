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

        # Policy loss: vectorized masked cross-entropy (zero Python loops)
        policy_flat = policy_logits.view(B, -1)  # (B, 8280)

        # Pad variable-length indices and probs to (B, max_L)
        max_L = max(len(idx) for idx in pi_indices_list)
        padded_idx = torch.zeros((B, max_L), dtype=torch.long, device=device)
        padded_probs = torch.zeros((B, max_L), dtype=torch.float32, device=device)
        mask = torch.zeros((B, max_L), dtype=torch.bool, device=device)

        for i in range(B):
            L = len(pi_indices_list[i])
            if L > 0:
                padded_idx[i, :L] = torch.as_tensor(
                    pi_indices_list[i].astype(np.int64), device=device
                )
                padded_probs[i, :L] = torch.as_tensor(
                    pi_probs_list[i], dtype=torch.float32, device=device
                )
                mask[i, :L] = True

        # Gather all legal logits at once → (B, max_L)
        gathered = policy_flat.gather(1, padded_idx)
        # Mask padding to -inf so log_softmax ignores them
        gathered = gathered.masked_fill(~mask, float('-inf'))
        log_probs = F.log_softmax(gathered, dim=1)
        # Kill 0 * -inf = NaN trap: zero out padding positions (out-of-place for autograd)
        log_probs = log_probs.masked_fill(~mask, 0.0)

        # Vectorized cross-entropy
        policy_loss = -(padded_probs * log_probs).sum(dim=1).mean()

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
