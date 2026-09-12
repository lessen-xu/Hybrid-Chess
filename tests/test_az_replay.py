"""ReplayBuffer save/load round-trip tests.

Ensures serialization and deserialization produce identical data.
"""

import os
import tempfile
import numpy as np
import pytest

from hybrid.core.types import Side
from hybrid.rl.az_selfplay import Example
from hybrid.rl.az_replay import ReplayBuffer
from hybrid.rl.az_encoding import NUM_STATE_CHANNELS
from hybrid.core.config import BOARD_H, BOARD_W


def _make_fake_example(rng: np.random.Generator, side: Side) -> Example:
    """Create a fake training example for testing."""
    state = rng.integers(0, 2, size=(NUM_STATE_CHANNELS, BOARD_H, BOARD_W), dtype=np.uint8)
    n_legal = rng.integers(5, 40)
    indices = rng.choice(8280, size=n_legal, replace=False).astype(np.uint16)
    probs = rng.dirichlet(np.ones(n_legal)).astype(np.float32)
    z = float(rng.choice([-1.0, 0.0, 1.0]))
    return Example(state=state, pi_indices=indices, pi_probs=probs,
                   side_to_move=side, z=z)


def test_save_load_roundtrip():
    """save_npz -> load_npz should preserve all fields exactly."""
    rng = np.random.default_rng(42)
    examples = [
        _make_fake_example(rng, Side.CHESS),
        _make_fake_example(rng, Side.XIANGQI),
        _make_fake_example(rng, Side.CHESS),
    ]

    buf = ReplayBuffer()
    buf.append(examples)
    assert len(buf) == 3

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_replay.npz")
        buf.save_npz(path)
        loaded = ReplayBuffer.load_npz(path)

    assert len(loaded) == 3
    for orig, load in zip(buf.examples, loaded.examples):
        np.testing.assert_array_equal(orig.state, load.state)
        np.testing.assert_array_equal(orig.pi_indices, load.pi_indices)
        np.testing.assert_array_almost_equal(orig.pi_probs, load.pi_probs, decimal=5)
        assert orig.side_to_move == load.side_to_move
        assert orig.z == load.z


def test_empty_buffer_save_load():
    """Empty buffer save/load should not crash."""
    buf = ReplayBuffer()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "empty.npz")
        buf.save_npz(path)
        loaded = ReplayBuffer.load_npz(path)
    assert len(loaded) == 0


def test_sample_batch():
    """sample_batch returns correct shapes and types."""
    rng = np.random.default_rng(123)
    examples = [_make_fake_example(rng, Side.CHESS) for _ in range(10)]

    buf = ReplayBuffer()
    buf.append(examples)

    states, pi_idx_list, pi_probs_list, z = buf.sample_batch(5, rng)

    assert states.shape == (5, NUM_STATE_CHANNELS, BOARD_H, BOARD_W)
    assert states.dtype == np.float32
    assert len(pi_idx_list) == 5
    assert len(pi_probs_list) == 5
    assert z.shape == (5,)
    assert z.dtype == np.float32


def test_balanced_buffer_uniform_distribution():
    """BalancedBuffer samples uniformly across strata despite extreme skew in sample counts."""
    from hybrid.rl.az_replay import BalancedBuffer

    rng = np.random.default_rng(42)
    # Heavily skewed: 500 Chess wins (z=1), 500 Chess losses (z=-1),
    # but only 5 Xiangqi wins (z=1), 5 Xiangqi losses (z=-1), 10 draws (z=0)
    examples = []
    for _ in range(500):
        examples.append(Example(np.zeros((NUM_STATE_CHANNELS, BOARD_H, BOARD_W), dtype=np.uint8),
                                np.array([0], dtype=np.uint16), np.array([1.0], dtype=np.float32),
                                Side.CHESS, 1.0))
        examples.append(Example(np.zeros((NUM_STATE_CHANNELS, BOARD_H, BOARD_W), dtype=np.uint8),
                                np.array([0], dtype=np.uint16), np.array([1.0], dtype=np.float32),
                                Side.CHESS, -1.0))
    for _ in range(5):
        examples.append(Example(np.zeros((NUM_STATE_CHANNELS, BOARD_H, BOARD_W), dtype=np.uint8),
                                np.array([0], dtype=np.uint16), np.array([1.0], dtype=np.float32),
                                Side.XIANGQI, 1.0))
        examples.append(Example(np.zeros((NUM_STATE_CHANNELS, BOARD_H, BOARD_W), dtype=np.uint8),
                                np.array([0], dtype=np.uint16), np.array([1.0], dtype=np.float32),
                                Side.XIANGQI, -1.0))
    for _ in range(10):
        examples.append(Example(np.zeros((NUM_STATE_CHANNELS, BOARD_H, BOARD_W), dtype=np.uint8),
                                np.array([0], dtype=np.uint16), np.array([1.0], dtype=np.float32),
                                Side.CHESS, 0.0))
        examples.append(Example(np.zeros((NUM_STATE_CHANNELS, BOARD_H, BOARD_W), dtype=np.uint8),
                                np.array([0], dtype=np.uint16), np.array([1.0], dtype=np.float32),
                                Side.XIANGQI, 0.0))

    buf = BalancedBuffer()
    buf.append(examples)

    # Sample a large batch: each of the 6 strata should get roughly 1/6 (~100 out of 600)
    batch_states, _, _, batch_z = buf.sample_batch(600, rng=np.random.default_rng(99))
    assert len(batch_z) == 600
    for z_val in (-1.0, 0.0, 1.0):
        count = sum(batch_z == z_val)
        # Each outcome appears across 2 sides (Chess + XQ) = 2/6 = 1/3 of draws (~200)
        assert 140 <= count <= 260, f"Outcome {z_val} count {count} deviates from expected ~200"

