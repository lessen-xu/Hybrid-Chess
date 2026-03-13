"""GPU batch inference server with zero-copy shared memory IPC.

Architecture:
  - SharedMemoryPool holds cross-process tensors (boards, sides, policies, values)
  - Workers write board state into pool slots, send tiny (wid, K) via Queue
  - Server reads from pool, runs GPU inference, writes results back, wakes workers
  - No pickle serialization of tensors — only 8-byte tuples cross the Queue
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp
import queue
import time

import numpy as np
import torch

from hybrid.rl.az_network import PolicyValueNet
from hybrid.rl.az_encoding import encode_batch_gpu, NUM_STATE_CHANNELS
from hybrid.core.config import BOARD_H, BOARD_W
from hybrid.rl.az_shm_pool import SharedMemoryPool


_STOP_SENTINEL = "STOP"
# Inference server

class InferenceServer:
    """GPU batch inference server with shared memory IPC.

    Model is loaded inside run() (after spawn). Greedy batching strategy:
    block for first request, then drain up to max_batch_size within timeout_ms.
    """

    def __init__(
        self,
        model_ckpt_path: str,
        request_queue: mp.Queue,
        pool: SharedMemoryPool,
        stop_event: mp.Event,
        max_batch_size: int = 32,
        timeout_ms: float = 5.0,
        device: str = "cuda",
        stats_queue: Optional[mp.Queue] = None,
    ):
        self.model_ckpt_path = model_ckpt_path
        self.request_queue = request_queue
        self.pool = pool
        self.stop_event = stop_event
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.device = device
        self.stats_queue = stats_queue

    def run(self) -> None:
        """Server main loop."""
        dev = torch.device(self.device)
        from hybrid.rl.az_runner import build_net_from_checkpoint
        net = build_net_from_checkpoint(self.model_ckpt_path, device=str(dev))
        net.to(dev)

        use_amp = dev.type == 'cuda'
        B_max = self.max_batch_size

        # --- TF32 for Ampere+ GPUs ---
        if dev.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # NOTE: torch.compile disabled — hangs on Windows (Triton/Inductor issues).
        # Static batching (always full B_max forward) already eliminates shape-change overhead.

        # --- Pre-allocate pinned CPU buffers (DMA-ready) ---
        pinned_ids = torch.zeros((B_max, BOARD_H, BOARD_W), dtype=torch.int8)
        pinned_sides = torch.zeros((B_max,), dtype=torch.int8)
        if dev.type == 'cuda':
            pinned_ids = pinned_ids.pin_memory()
            pinned_sides = pinned_sides.pin_memory()

        # --- Pre-allocate GPU-resident buffers ---
        gpu_ids = torch.zeros((B_max, BOARD_H, BOARD_W), dtype=torch.int8, device=dev)
        gpu_sides = torch.zeros((B_max,), dtype=torch.int8, device=dev)
        gpu_states = torch.zeros((B_max, NUM_STATE_CHANNELS, BOARD_H, BOARD_W),
                                 dtype=torch.float32, device=dev)

        # --- Warmup: force graph capture before first real request ---
        if dev.type == 'cuda':
            print("[Server] Warming up (3 rounds)...")
            with torch.no_grad(), torch.autocast(
                device_type=dev.type, enabled=True, dtype=torch.float16
            ):
                for _ in range(3):
                    _ = net(gpu_states)
                    torch.cuda.synchronize(dev)
            print("[Server] Warmup complete. Ready.")

        total_batches = 0
        total_states = 0
        batch_sizes = []
        queue_wait_time = 0.0
        gpu_compute_time = 0.0

        while not self.stop_event.is_set():
            t0 = time.perf_counter()
            signals = self._collect_batch()
            t1 = time.perf_counter()
            queue_wait_time += (t1 - t0)

            if not signals:
                continue

            self._process_batch(net, dev, signals, use_amp,
                                pinned_ids, pinned_sides,
                                gpu_ids, gpu_sides, gpu_states)
            if dev.type == 'cuda':
                torch.cuda.synchronize(dev)
            gpu_compute_time += (time.perf_counter() - t1)

            batch_state_count = sum(K for _, K in signals)
            total_batches += 1
            total_states += batch_state_count
            batch_sizes.append(batch_state_count)

        # Drain remaining requests to prevent worker deadlock
        self._drain_remaining(net, dev, use_amp,
                              pinned_ids, pinned_sides,
                              gpu_ids, gpu_sides, gpu_states)

        if self.stats_queue is not None and total_batches > 0:
            self.stats_queue.put({
                "inference_batches": total_batches,
                "inference_states": total_states,
                "avg_batch_size": round(total_states / total_batches, 2),
                "max_batch_size_seen": max(batch_sizes) if batch_sizes else 0,
                "queue_wait_s": round(queue_wait_time, 3),
                "gpu_compute_s": round(gpu_compute_time, 3),
                "avg_batch_fill_pct": round(
                    100 * (total_states / total_batches) / B_max, 1
                ),
            })

    def _collect_batch(self) -> List[Tuple[int, int]]:
        """Collect batch signals: block for first, then greedily drain.

        Returns list of (worker_id, K) tuples.
        """
        signals: List[Tuple[int, int]] = []
        total_states = 0

        try:
            first = self.request_queue.get(timeout=self.timeout_ms / 1000.0)
            if first == _STOP_SENTINEL:
                self.stop_event.set()
                return []
            signals.append(first)
            total_states += first[1]
        except queue.Empty:
            return []

        deadline = time.monotonic() + self.timeout_ms / 1000.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                sig = self.request_queue.get(timeout=max(0.0001, remaining))
                if sig == _STOP_SENTINEL:
                    self.stop_event.set()
                    break
                wid, K = sig
                if total_states + K > self.max_batch_size:
                    # Put it back — doesn't fit in this batch
                    self.request_queue.put(sig)
                    break
                signals.append(sig)
                total_states += K
            except queue.Empty:
                break

        return signals

    def _process_batch(
        self, net: PolicyValueNet, dev: torch.device,
        signals: List[Tuple[int, int]], use_amp: bool,
        pinned_ids: torch.Tensor, pinned_sides: torch.Tensor,
        gpu_ids: torch.Tensor, gpu_sides: torch.Tensor,
        gpu_states: torch.Tensor,
    ) -> None:
        """Run batch forward using shared memory (zero-copy hot path).

        Reads boards/sides from pool, runs GPU inference, writes full policy
        flat + values back to pool, then wakes workers via events.
        """
        pool = self.pool

        # 1. Read from shared memory into pinned buffers
        offset = 0
        job_info = []  # (wid, K, start_offset)
        for wid, K in signals:
            pinned_ids[offset:offset+K].copy_(pool.boards[wid, :K])
            pinned_sides[offset:offset+K].copy_(pool.sides[wid, :K])
            job_info.append((wid, K, offset))
            offset += K

        total_B = offset
        if total_B == 0:
            return

        # 2. Async DMA transfer to GPU
        gpu_ids[:total_B].copy_(pinned_ids[:total_B], non_blocking=True)
        gpu_sides[:total_B].copy_(pinned_sides[:total_B], non_blocking=True)

        # 3. In-place GPU feature encoding
        encode_batch_gpu(gpu_ids[:total_B], gpu_sides[:total_B], dev, out=gpu_states[:total_B])

        # 4. Static-batch AMP forward pass (always full B_max for CUDA Graphs)
        B_max = gpu_states.shape[0]
        with torch.no_grad(), torch.autocast(
            device_type=dev.type, enabled=use_amp, dtype=torch.float16
        ):
            policy_logits_full, values_full = net(gpu_states)

        # 5. Slice valid results, cast to FP32, transfer to CPU
        policy_flat_cpu = policy_logits_full[:total_B].float().reshape(total_B, -1).cpu()
        values_cpu = values_full[:total_B].float().squeeze(-1).cpu()

        # 6. Write results back to shared memory and wake workers
        for wid, K, start_idx in job_info:
            pool.policies[wid, :K].copy_(policy_flat_cpu[start_idx:start_idx+K])
            pool.values[wid, :K].copy_(values_cpu[start_idx:start_idx+K])
            pool.events[wid].set()

    def _drain_remaining(
        self, net: PolicyValueNet, dev: torch.device, use_amp: bool,
        pinned_ids: torch.Tensor, pinned_sides: torch.Tensor,
        gpu_ids: torch.Tensor, gpu_sides: torch.Tensor,
        gpu_states: torch.Tensor,
    ) -> None:
        """Process remaining requests before shutdown."""
        remaining: List[Tuple[int, int]] = []
        while True:
            try:
                sig = self.request_queue.get_nowait()
                if sig != _STOP_SENTINEL:
                    remaining.append(sig)
            except queue.Empty:
                break
        if remaining:
            self._process_batch(net, dev, remaining, use_amp,
                                pinned_ids, pinned_sides,
                                gpu_ids, gpu_sides, gpu_states)
# Process entry point

def inference_server_process(
    model_ckpt_path: str,
    request_queue: mp.Queue,
    pool: SharedMemoryPool,
    stop_event: mp.Event,
    max_batch_size: int = 32,
    timeout_ms: float = 5.0,
    device: str = "cuda",
    stats_queue: Optional[mp.Queue] = None,
) -> None:
    """Entry point for inference server subprocess."""
    server = InferenceServer(
        model_ckpt_path=model_ckpt_path,
        request_queue=request_queue,
        pool=pool,
        stop_event=stop_event,
        max_batch_size=max_batch_size,
        timeout_ms=timeout_ms,
        device=device,
        stats_queue=stats_queue,
    )
    server.run()
# Inference client (used in worker processes)

class InferenceClient:
    """Client that writes to shared memory and signals the server."""

    def __init__(
        self,
        worker_id: int,
        request_queue: mp.Queue,
        pool: SharedMemoryPool,
        track_latency: bool = False,
    ):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.pool = pool
        self.track_latency = track_latency
        self.latencies_ms: List[float] = []

    def predict_raw(
        self,
        board_ids: np.ndarray,
        side: np.int8,
        legal_action_indices: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Single-state inference via shared memory.

        Returns:
            (logits, value) — logits shape (L,) float32.
        """
        wid = self.worker_id
        pool = self.pool

        # Write to shared memory
        pool.boards[wid, 0].copy_(torch.from_numpy(board_ids))
        pool.sides[wid, 0] = side

        # Signal server (8-byte tuple)
        pool.events[wid].clear()
        t0 = time.perf_counter()
        self.request_queue.put((wid, 1))
        pool.events[wid].wait()

        if self.track_latency:
            self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        # Read results and slice locally
        flat_policy = pool.policies[wid, 0].numpy()
        indices = legal_action_indices.astype(np.int64)
        logits = flat_policy[indices].copy().astype(np.float32)
        value = float(pool.values[wid, 0].item())

        return logits, value

    def predict_batch_raw(
        self,
        board_ids_stack: np.ndarray,
        sides_stack: np.ndarray,
        action_indices_list: list,
    ) -> Tuple[list, np.ndarray]:
        """K-batched inference via shared memory.

        Returns:
            (logits_list, values) — K arrays + (K,) float32.
        """
        wid = self.worker_id
        pool = self.pool
        K = len(sides_stack)

        # Write to shared memory
        pool.boards[wid, :K].copy_(torch.from_numpy(board_ids_stack))
        pool.sides[wid, :K].copy_(torch.from_numpy(sides_stack))

        # Signal server
        pool.events[wid].clear()
        t0 = time.perf_counter()
        self.request_queue.put((wid, K))
        pool.events[wid].wait()

        if self.track_latency:
            self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        # Read results and slice locally
        logits_list = []
        pol_view = pool.policies[wid].numpy()
        for k in range(K):
            indices = action_indices_list[k].astype(np.int64)
            logits_list.append(pol_view[k][indices].copy().astype(np.float32))

        values = pool.values[wid, :K].numpy().copy()

        return logits_list, values
