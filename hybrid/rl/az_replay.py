"""Replay buffer for AlphaZero self-play training samples.

Storage format: npz with sparse π stored as offsets + flat arrays (CSR-like).
"""

from __future__ import annotations
from typing import List, Optional
import os

import numpy as np

from .az_selfplay import Example
from hybrid.core.types import Side
from hybrid.rl.az_encoding import NUM_STATE_CHANNELS
from hybrid.core.config import BOARD_H, BOARD_W


class ReplayBuffer:
    """Experience replay buffer with npz persistence."""

    def __init__(self, max_size: int = 100_000):
        self.max_size = max_size
        self.examples: List[Example] = []

    def __len__(self) -> int:
        return len(self.examples)

    def append(self, examples: List[Example]) -> None:
        """Add examples from one game. Evicts oldest if over max_size."""
        self.examples.extend(examples)
        if len(self.examples) > self.max_size:
            self.examples = self.examples[-self.max_size:]

    def sample_batch(self, batch_size: int, rng: Optional[np.random.Generator] = None):
        """Sample a random batch. Returns (states, pi_indices_list, pi_probs_list, z)."""
        if rng is None:
            rng = np.random.default_rng()

        n = len(self.examples)
        if batch_size >= n:
            idxs = np.arange(n)
        else:
            idxs = rng.choice(n, size=batch_size, replace=False)

        states = np.stack([self.examples[i].state for i in idxs]).astype(np.float32)
        pi_indices_list = [self.examples[i].pi_indices for i in idxs]
        pi_probs_list = [self.examples[i].pi_probs for i in idxs]
        z = np.array([self.examples[i].z for i in idxs], dtype=np.float32)

        return states, pi_indices_list, pi_probs_list, z

    def save_npz(self, path: str) -> None:
        """Save buffer to npz. Sparse π uses offsets + flat arrays."""
        n = len(self.examples)
        if n == 0:
            np.savez_compressed(
                path,
                states=np.zeros((0, NUM_STATE_CHANNELS, BOARD_H, BOARD_W), dtype=np.uint8),
                offsets=np.zeros(1, dtype=np.int32),
                indices_flat=np.zeros(0, dtype=np.uint16),
                probs_flat=np.zeros(0, dtype=np.float32),
                z=np.zeros(0, dtype=np.float32),
                sides=np.zeros(0, dtype=np.uint8),
            )
            return

        states = np.stack([ex.state for ex in self.examples])
        z = np.array([ex.z for ex in self.examples], dtype=np.float32)
        sides = np.array([0 if ex.side_to_move == Side.CHESS else 1
                          for ex in self.examples], dtype=np.uint8)

        offsets = np.zeros(n + 1, dtype=np.int32)
        all_indices = []
        all_probs = []
        for i, ex in enumerate(self.examples):
            offsets[i + 1] = offsets[i] + len(ex.pi_indices)
            all_indices.append(ex.pi_indices)
            all_probs.append(ex.pi_probs)

        indices_flat = np.concatenate(all_indices) if all_indices else np.zeros(0, dtype=np.uint16)
        probs_flat = np.concatenate(all_probs) if all_probs else np.zeros(0, dtype=np.float32)

        np.savez_compressed(
            path,
            states=states,
            offsets=offsets,
            indices_flat=indices_flat,
            probs_flat=probs_flat,
            z=z,
            sides=sides,
        )

    @staticmethod
    def load_npz(path: str) -> "ReplayBuffer":
        """Load buffer from npz file."""
        data = np.load(path)
        states = data["states"]
        offsets = data["offsets"]
        indices_flat = data["indices_flat"]
        probs_flat = data["probs_flat"]
        z = data["z"]
        sides = data["sides"]

        buf = ReplayBuffer()
        n = len(states)
        for i in range(n):
            start, end = offsets[i], offsets[i + 1]
            ex = Example(
                state=states[i],
                pi_indices=indices_flat[start:end].copy(),
                pi_probs=probs_flat[start:end].copy(),
                side_to_move=Side.CHESS if sides[i] == 0 else Side.XIANGQI,
                z=float(z[i]),
            )
            buf.examples.append(ex)

        return buf
