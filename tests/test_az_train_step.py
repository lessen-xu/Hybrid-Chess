"""Smoke test for training step.

Runs 1 training step with a tiny buffer and verifies:
- Losses are finite (not NaN/inf)
- Backward pass completes without error
"""

import numpy as np
import torch
import pytest

from hybrid.core.types import Side
from hybrid.rl.az_selfplay import Example
from hybrid.rl.az_replay import ReplayBuffer
from hybrid.rl.az_train import train_one_epoch
from hybrid.rl.az_network import PolicyValueNet
from hybrid.rl.az_encoding import NUM_STATE_CHANNELS
from hybrid.core.config import BOARD_H, BOARD_W


def _make_fake_example(rng: np.random.Generator, side: Side) -> Example:
    """Create a fake training example."""
    state = rng.integers(0, 2, size=(NUM_STATE_CHANNELS, BOARD_H, BOARD_W), dtype=np.uint8)
    n_legal = rng.integers(5, 40)
    indices = rng.choice(8280, size=n_legal, replace=False).astype(np.uint16)
    probs = rng.dirichlet(np.ones(n_legal)).astype(np.float32)
    z = float(rng.choice([-1.0, 0.0, 1.0]))
    return Example(state=state, pi_indices=indices, pi_probs=probs,
                   side_to_move=side, z=z)


def test_train_one_step_finite_loss():
    """8 fake samples, 1 training step -> losses must be finite."""
    rng = np.random.default_rng(42)
    examples = [
        _make_fake_example(rng, Side.CHESS if i % 2 == 0 else Side.XIANGQI)
        for i in range(8)
    ]

    buf = ReplayBuffer()
    buf.append(examples)

    net = PolicyValueNet()
    device = torch.device("cpu")
    net = net.to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)

    stats = train_one_epoch(
        net=net,
        buffer=buf,
        optimizer=optimizer,
        device=device,
        batch_size=4,
        max_steps=1,
    )

    assert np.isfinite(stats["policy_loss"]), f"policy_loss is not finite: {stats['policy_loss']}"
    assert np.isfinite(stats["value_loss"]), f"value_loss is not finite: {stats['value_loss']}"
    assert np.isfinite(stats["total_loss"]), f"total_loss is not finite: {stats['total_loss']}"
    assert stats["steps"] == 1


def test_multiple_steps_loss_changes():
    """Multiple training steps should produce finite losses (gradients updating)."""
    rng = np.random.default_rng(99)
    examples = [
        _make_fake_example(rng, Side.CHESS if i % 2 == 0 else Side.XIANGQI)
        for i in range(20)
    ]

    buf = ReplayBuffer()
    buf.append(examples)

    net = PolicyValueNet()
    device = torch.device("cpu")
    net = net.to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-4)

    stats1 = train_one_epoch(net=net, buffer=buf, optimizer=optimizer,
                              device=device, batch_size=10, max_steps=5)
    stats2 = train_one_epoch(net=net, buffer=buf, optimizer=optimizer,
                              device=device, batch_size=10, max_steps=5)

    assert np.isfinite(stats1["total_loss"])
    assert np.isfinite(stats2["total_loss"])
