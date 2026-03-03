# -*- coding: utf-8 -*-
"""GPU batch inference server + client for parallel self-play.

InferenceServer (separate process, holds GPU model):
  - Collects requests from workers via request_queue
  - Batches up to max_batch_size or timeout_ms, runs batch forward
  - Routes results to per-worker response_queues

InferenceClient (in worker processes):
  - predict_raw(state_u8, legal_action_indices) → (logits, value)
  - Blocks waiting for response, matched by req_id

All queue data uses numpy/Python native types (no torch tensors across processes).
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


# ====================================================================
# Data protocol (must be picklable)
# ====================================================================

@dataclass
class InferenceRequest:
    """Worker → Server inference request.

    Supports K-batched states (leaf batching):
      board_ids:  (K, 10, 9) int8
      sides:      (K,) int8
      action_indices_list: list of K arrays, each (L_k,) uint16
      batch_count: K (number of states in this request)
    """
    req_id: int
    worker_id: int
    board_ids: np.ndarray               # (K, 10, 9), int8
    sides: np.ndarray                   # (K,), int8
    action_indices_list: list           # list of K np.ndarray, each (L_k,) uint16
    batch_count: int = 1               # K: number of states


@dataclass
class InferenceResponse:
    """Server → Worker inference response."""
    req_id: int
    worker_id: int
    logits_list: list     # list of K np.ndarray, each (L_k,) float32
    values: np.ndarray    # (K,) float32


_STOP_SENTINEL = "STOP"


# ====================================================================
# Inference server
# ====================================================================

class InferenceServer:
    """GPU batch inference server.

    Model is loaded inside run() (after spawn). Greedy batching strategy:
    block for first request, then drain up to max_batch_size within timeout_ms.
    """

    def __init__(
        self,
        model_ckpt_path: str,
        request_queue: mp.Queue,
        response_queues: Dict[int, mp.Queue],
        stop_event: mp.Event,
        max_batch_size: int = 32,
        timeout_ms: float = 5.0,
        device: str = "cuda",
        stats_queue: Optional[mp.Queue] = None,
    ):
        self.model_ckpt_path = model_ckpt_path
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.stop_event = stop_event
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.device = device
        self.stats_queue = stats_queue

    def run(self) -> None:
        """Server main loop."""
        dev = torch.device(self.device)
        net = PolicyValueNet()
        ckpt = torch.load(self.model_ckpt_path, map_location=dev, weights_only=True)
        net.load_state_dict(ckpt["model"])
        net.to(dev)
        net.eval()

        use_amp = dev.type == 'cuda'
        B_max = self.max_batch_size

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

        total_batches = 0
        total_states = 0
        batch_sizes = []
        queue_wait_time = 0.0
        gpu_compute_time = 0.0

        while not self.stop_event.is_set():
            t0 = time.perf_counter()
            batch = self._collect_batch()
            t1 = time.perf_counter()
            queue_wait_time += (t1 - t0)

            if not batch:
                continue

            self._process_batch(net, dev, batch, use_amp,
                                pinned_ids, pinned_sides,
                                gpu_ids, gpu_sides, gpu_states)
            if dev.type == 'cuda':
                torch.cuda.synchronize(dev)
            gpu_compute_time += (time.perf_counter() - t1)

            batch_state_count = sum(req.batch_count for req in batch)
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

    def _collect_batch(self) -> List[InferenceRequest]:
        """Collect a batch: block for first request, then greedily drain."""
        requests: List[InferenceRequest] = []

        try:
            first = self.request_queue.get(timeout=self.timeout_ms / 1000.0)
            if first == _STOP_SENTINEL:
                self.stop_event.set()
                return []
            requests.append(first)
        except queue.Empty:
            return []

        deadline = time.monotonic() + self.timeout_ms / 1000.0
        while len(requests) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                req = self.request_queue.get(timeout=max(0.0001, remaining))
                if req == _STOP_SENTINEL:
                    self.stop_event.set()
                    break
                requests.append(req)
            except queue.Empty:
                break

        return requests

    def _process_batch(
        self, net: PolicyValueNet, dev: torch.device,
        batch: List[InferenceRequest], use_amp: bool,
        pinned_ids: torch.Tensor, pinned_sides: torch.Tensor,
        gpu_ids: torch.Tensor, gpu_sides: torch.Tensor,
        gpu_states: torch.Tensor,
    ) -> None:
        """Run batch forward and dispatch results (zero-allocation hot path).

        Unflattens K-batched worker requests into one big GPU batch.
        """
        # 1. Flatten all worker requests into contiguous pinned buffers
        offset = 0
        req_sizes = []  # (req_index, K) for re-splitting
        for req in batch:
            K = req.batch_count
            req_sizes.append(K)
            pinned_ids[offset:offset+K].copy_(torch.from_numpy(req.board_ids))
            pinned_sides[offset:offset+K].copy_(torch.from_numpy(req.sides))
            offset += K

        total_B = offset  # True GPU batch size
        if total_B == 0:
            return

        # 2. Async DMA transfer to GPU
        gpu_ids[:total_B].copy_(pinned_ids[:total_B], non_blocking=True)
        gpu_sides[:total_B].copy_(pinned_sides[:total_B], non_blocking=True)

        # 3. In-place GPU feature encoding
        encode_batch_gpu(gpu_ids[:total_B], gpu_sides[:total_B], dev, out=gpu_states[:total_B])

        # 4. AMP forward pass
        with torch.no_grad(), torch.autocast(
            device_type=dev.type, enabled=use_amp, dtype=torch.float16
        ):
            policy_logits, values = net(gpu_states[:total_B])

        # 5. Cast back to FP32 and transfer to CPU
        policy_flat = policy_logits.float().reshape(total_B, -1).cpu().numpy()
        values_np = values.float().squeeze(-1).cpu().numpy()

        # 6. Re-split results by worker request
        offset = 0
        for req, K in zip(batch, req_sizes):
            logits_list = []
            for k in range(K):
                idx = offset + k
                indices = req.action_indices_list[k].astype(np.int64)
                logits_list.append(policy_flat[idx][indices].astype(np.float32))

            resp = InferenceResponse(
                req_id=req.req_id,
                worker_id=req.worker_id,
                logits_list=logits_list,
                values=values_np[offset:offset+K].copy(),
            )
            self.response_queues[req.worker_id].put(resp)
            offset += K

    def _drain_remaining(
        self, net: PolicyValueNet, dev: torch.device, use_amp: bool,
        pinned_ids: torch.Tensor, pinned_sides: torch.Tensor,
        gpu_ids: torch.Tensor, gpu_sides: torch.Tensor,
        gpu_states: torch.Tensor,
    ) -> None:
        """Process remaining requests before shutdown."""
        remaining: List[InferenceRequest] = []
        while True:
            try:
                req = self.request_queue.get_nowait()
                if req != _STOP_SENTINEL:
                    remaining.append(req)
            except queue.Empty:
                break
        if remaining:
            self._process_batch(net, dev, remaining, use_amp,
                                pinned_ids, pinned_sides,
                                gpu_ids, gpu_sides, gpu_states)


# ====================================================================
# Process entry point
# ====================================================================

def inference_server_process(
    model_ckpt_path: str,
    request_queue: mp.Queue,
    response_queues: Dict[int, mp.Queue],
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
        response_queues=response_queues,
        stop_event=stop_event,
        max_batch_size=max_batch_size,
        timeout_ms=timeout_ms,
        device=device,
        stats_queue=stats_queue,
    )
    server.run()


# ====================================================================
# Inference client (used in worker processes)
# ====================================================================

class InferenceClient:
    """Client that sends leaf evaluation requests to InferenceServer and blocks for results."""

    def __init__(
        self,
        worker_id: int,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        track_latency: bool = False,
    ):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._next_req_id = 0
        self.track_latency = track_latency
        self.latencies_ms: List[float] = []

    def predict_raw(
        self,
        board_ids: np.ndarray,
        side: np.int8,
        legal_action_indices: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Send single-state inference request and block for result.

        Args:
            board_ids: (10, 9) int8
            side: int8
            legal_action_indices: (L,) uint16

        Returns:
            (logits, value) — logits shape (L,) float32, value is scalar.
        """
        req_id = self._next_req_id
        self._next_req_id += 1

        req = InferenceRequest(
            req_id=req_id,
            worker_id=self.worker_id,
            board_ids=board_ids[np.newaxis],           # (1, 10, 9)
            sides=np.array([side], dtype=np.int8),     # (1,)
            action_indices_list=[legal_action_indices], # [array]
            batch_count=1,
        )

        t0 = time.perf_counter()
        self.request_queue.put(req)

        resp: InferenceResponse = self.response_queue.get()
        if self.track_latency:
            self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        assert resp.req_id == req_id, f"req_id mismatch: expected {req_id}, got {resp.req_id}"
        assert resp.worker_id == self.worker_id

        return resp.logits_list[0], float(resp.values[0])

    def predict_batch_raw(
        self,
        board_ids_stack: np.ndarray,
        sides_stack: np.ndarray,
        action_indices_list: list,
    ) -> Tuple[list, np.ndarray]:
        """Send K-batched inference request and block for K results.

        Args:
            board_ids_stack: (K, 10, 9) int8
            sides_stack: (K,) int8
            action_indices_list: list of K arrays, each (L_k,) uint16

        Returns:
            (logits_list, values) — logits_list: K arrays, values: (K,) float32.
        """
        req_id = self._next_req_id
        self._next_req_id += 1
        K = len(sides_stack)

        req = InferenceRequest(
            req_id=req_id,
            worker_id=self.worker_id,
            board_ids=board_ids_stack,
            sides=sides_stack,
            action_indices_list=action_indices_list,
            batch_count=K,
        )

        t0 = time.perf_counter()
        self.request_queue.put(req)

        resp: InferenceResponse = self.response_queue.get()
        if self.track_latency:
            self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        assert resp.req_id == req_id
        assert resp.worker_id == self.worker_id

        return resp.logits_list, resp.values
