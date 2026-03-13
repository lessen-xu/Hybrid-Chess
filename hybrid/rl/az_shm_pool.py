"""Shared memory pool for zero-copy IPC between workers and inference server.

Workers write board state into their dedicated slot, send a tiny (wid, K)
signal via Queue, then wait on an mp.Event. Server reads from the pool,
runs GPU inference, writes results back, and sets the event to wake the worker.

No pickle serialization of tensors crosses the Queue — only 8-byte tuples.
"""

from __future__ import annotations

import multiprocessing as mp
import torch

# Pool dimensions
MAX_WORKERS = 32   # physical CPU core ceiling
MAX_LEAVES = 16    # K ceiling (leaf_batch_size)
BOARD_H, BOARD_W = 10, 9
POLICY_FLAT_SIZE = 92 * BOARD_H * BOARD_W  # 8280


class SharedMemoryPool:
    """Cross-process shared memory tensors for zero-copy inference IPC."""

    def __init__(self, max_workers: int = MAX_WORKERS, max_leaves: int = MAX_LEAVES):
        self.max_workers = max_workers
        self.max_leaves = max_leaves

        # ── Forward buffers (Worker writes, Server reads) ──
        self.boards = torch.zeros(
            max_workers, max_leaves, BOARD_H, BOARD_W, dtype=torch.int8
        ).share_memory_()
        self.sides = torch.zeros(
            max_workers, max_leaves, dtype=torch.int8
        ).share_memory_()

        # ── Backward buffers (Server writes, Worker reads) ──
        self.policies = torch.zeros(
            max_workers, max_leaves, POLICY_FLAT_SIZE, dtype=torch.float32
        ).share_memory_()
        self.values = torch.zeros(
            max_workers, max_leaves, dtype=torch.float32
        ).share_memory_()

        # ── Per-worker wake events ──
        self.events = [mp.Event() for _ in range(max_workers)]
