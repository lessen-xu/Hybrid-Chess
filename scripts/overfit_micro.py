# -*- coding: utf-8 -*-
"""Micro-dataset overfitting test (Silent-Bug Diagnosis Step 1).

Proves that PolicyValueNet + loss function can memorize 20 hand-crafted
samples.  No MCTS, no self-play, no ReplayBuffer — just raw network training.

Usage:
  python -m scripts.overfit_micro
"""

from __future__ import annotations

import scripts._fix_encoding  # noqa: F401
import sys
import numpy as np
import torch
import torch.nn.functional as F

from hybrid.core.env import HybridChessEnv
from hybrid.core.types import Side
from hybrid.rl.az_encoding import (
    encode_state, move_to_plane,
    TOTAL_POLICY_PLANES, NUM_STATE_CHANNELS,
)
from hybrid.rl.az_network import PolicyValueNet
from hybrid.core.config import BOARD_H, BOARD_W

ACTION_SPACE_SIZE = TOTAL_POLICY_PLANES * BOARD_H * BOARD_W  # 8280


# ====================================================================
# 1. Generate 20 synthetic training samples from random play
# ====================================================================

def _move_to_flat_index(mv) -> int:
    plane_idx, fy, fx = move_to_plane(mv)
    return plane_idx * (BOARD_H * BOARD_W) + fy * BOARD_W + fx


def generate_micro_dataset(n_samples: int = 20, seed: int = 42):
    """Play random moves to reach diverse board states, then assign labels.

    Returns:
        states:     float32 (N, 14, 10, 9)
        pi_indices: list of int arrays — flat action indices of legal moves
        pi_targets: list of float32 arrays — one-hot over legal moves
        z_targets:  float32 (N,) — alternating +1 / -1
    """
    rng = np.random.default_rng(seed)
    env = HybridChessEnv()

    states = []
    pi_indices_list = []
    pi_targets_list = []
    z_targets = []

    collected = 0
    while collected < n_samples:
        state = env.reset()
        # Play a random number of moves (5-15) to get a diverse state
        n_moves = rng.integers(5, 16)
        for _ in range(n_moves):
            legal = env.legal_moves()
            if len(legal) == 0:
                break
            mv = legal[rng.integers(len(legal))]
            state, _, done, _ = env.step(mv)
            if done:
                break

        legal = env.legal_moves()
        if len(legal) == 0:
            continue  # game ended, skip

        # Encode state
        state_tensor = encode_state(state).numpy()  # float32 (14, 10, 9)

        # Build sparse one-hot policy target: pick one legal move
        chosen_idx = rng.integers(len(legal))
        indices = np.array([_move_to_flat_index(mv) for mv in legal], dtype=np.int64)
        target_probs = np.zeros(len(legal), dtype=np.float32)
        target_probs[chosen_idx] = 1.0

        # Value target: alternate +1 / -1
        z = 1.0 if collected % 2 == 0 else -1.0

        states.append(state_tensor)
        pi_indices_list.append(indices)
        pi_targets_list.append(target_probs)
        z_targets.append(z)
        collected += 1

    states_np = np.stack(states).astype(np.float32)
    z_np = np.array(z_targets, dtype=np.float32)
    return states_np, pi_indices_list, pi_targets_list, z_np


# ====================================================================
# 2. Training loop with inline loss (same math as az_train.py)
# ====================================================================

def train_overfit(
    net: PolicyValueNet,
    states_t: torch.Tensor,       # (N, 14, 10, 9)
    pi_indices_list,               # list of int64 arrays
    pi_targets_list,               # list of float32 arrays
    z_t: torch.Tensor,            # (N,)
    device: torch.device,
    epochs: int = 2000,
    lr: float = 1e-3,
    log_every: int = 50,
):
    """Full-batch training loop. Returns final stats dict."""
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    N = states_t.size(0)

    # Pre-convert targets to device tensors
    pi_idx_tensors = [torch.from_numpy(idx).to(device) for idx in pi_indices_list]
    pi_tgt_tensors = [torch.from_numpy(tgt).to(device) for tgt in pi_targets_list]

    history = []

    for epoch in range(1, epochs + 1):
        net.train()
        policy_logits, value = net(states_t)  # (N, 92, 10, 9), (N, 1)

        # Flatten policy logits: (N, 8280)
        policy_flat = policy_logits.view(N, -1)

        # Policy loss: masked cross-entropy (same as az_train.py)
        policy_losses = []
        policy_correct = 0
        for i in range(N):
            indices = pi_idx_tensors[i]
            target_probs = pi_tgt_tensors[i]

            legal_logits = policy_flat[i, indices]     # (L,)
            log_probs = F.log_softmax(legal_logits, dim=0)
            ce = -(target_probs * log_probs).sum()
            policy_losses.append(ce)

            # Accuracy: does argmax match the target?
            pred_idx = legal_logits.argmax().item()
            true_idx = target_probs.argmax().item()
            if pred_idx == true_idx:
                policy_correct += 1

        policy_loss = torch.stack(policy_losses).mean()

        # Value loss: MSE
        value_loss = F.mse_loss(value.squeeze(-1), z_t)

        # Value accuracy: sign match
        value_correct = int((value.squeeze(-1).sign() == z_t.sign()).sum().item())

        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stats = {
            "epoch": epoch,
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": loss.item(),
            "policy_acc": policy_correct / N,
            "value_acc": value_correct / N,
        }
        history.append(stats)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"  epoch {epoch:>5d}  "
                f"p_loss={stats['policy_loss']:.6f}  "
                f"v_loss={stats['value_loss']:.6f}  "
                f"total={stats['total_loss']:.6f}  "
                f"p_acc={stats['policy_acc']*100:.0f}%  "
                f"v_acc={stats['value_acc']*100:.0f}%"
            )

    return history


# ====================================================================
# 3. Main: generate data → train → verdict
# ====================================================================

def main():
    print("=" * 70)
    print("  MICRO-DATASET OVERFITTING TEST  (Silent-Bug Diagnosis Step 1)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Generate data
    print("\n[1/3] Generating 20 synthetic samples from random play ...")
    states_np, pi_indices, pi_targets, z_np = generate_micro_dataset(n_samples=20)
    states_t = torch.from_numpy(states_np).to(device)
    z_t = torch.from_numpy(z_np).to(device)
    print(f"  states: {states_np.shape}, z: {z_np.shape}")
    print(f"  legal moves per sample: {[len(idx) for idx in pi_indices]}")

    # Init network
    print("\n[2/3] Initializing fresh PolicyValueNet ...")
    net = PolicyValueNet()
    net = net.to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  Parameters: {n_params:,}")

    # Train
    print("\n[3/3] Training for 2000 epochs (full-batch, no shuffle) ...\n")
    history = train_overfit(
        net, states_t, pi_indices, pi_targets, z_t,
        device=device, epochs=2000, lr=1e-3, log_every=50,
    )

    # Verdict
    final = history[-1]
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    checks = {
        "Policy accuracy = 100%": final["policy_acc"] >= 1.0,
        "Value accuracy  = 100%": final["value_acc"] >= 1.0,
        "Policy loss     < 0.05": final["policy_loss"] < 0.05,
        "Value loss      < 0.01": final["value_loss"] < 0.01,
    }

    all_pass = True
    for desc, ok in checks.items():
        icon = "[PASS]" if ok else "[FAIL]"
        print(f"  {icon} {desc}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("  [PASS] Network + loss function can learn. No fatal DL bug.")
    else:
        print("  [FAIL] Network cannot memorize 20 samples!")
        print("         Check: cross-entropy dimensions, value head tanh,")
        print("         feature flatten order, batch norm, gradient flow.")

    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
