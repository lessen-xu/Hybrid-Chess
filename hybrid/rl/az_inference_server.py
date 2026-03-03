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
from hybrid.rl.az_encoding import encode_batch_gpu


# ====================================================================
# Data protocol (must be picklable)
# ====================================================================

@dataclass
class InferenceRequest:
    """Worker → Server inference request."""
    req_id: int
    worker_id: int
    board_ids: np.ndarray               # (10, 9), int8 — piece channel IDs, -1=empty
    side: np.int8                       # 1=Chess, 0=Xiangqi
    legal_action_indices: np.ndarray    # (L,), uint16


@dataclass
class InferenceResponse:
    """Server → Worker inference response."""
    req_id: int
    worker_id: int
    logits: np.ndarray   # (L,), float32 — legal move logits only
    value: float         # value head output, in [-1, 1]


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

        total_batches = 0
        total_requests = 0
        batch_sizes = []

        while not self.stop_event.is_set():
            batch = self._collect_batch()
            if not batch:
                continue
            self._process_batch(net, dev, batch)
            total_batches += 1
            total_requests += len(batch)
            batch_sizes.append(len(batch))

        # Drain remaining requests to prevent worker deadlock
        self._drain_remaining(net, dev)

        if self.stats_queue is not None and total_batches > 0:
            self.stats_queue.put({
                "inference_batches": total_batches,
                "inference_requests": total_requests,
                "avg_batch_size": round(total_requests / total_batches, 2),
                "max_batch_size_seen": max(batch_sizes) if batch_sizes else 0,
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
        self, net: PolicyValueNet, dev: torch.device, batch: List[InferenceRequest],
    ) -> None:
        """Run batch forward and dispatch results."""
        B = len(batch)

        # Stack compact board IDs and sides from all requests
        ids_np = np.stack([req.board_ids for req in batch])  # (B, 10, 9)
        sides_np = np.array([req.side for req in batch], dtype=np.int8)  # (B,)

        # GPU batch encoding: (B, 10, 9) int8 → (B, 14, 10, 9) float32
        ids_t = torch.from_numpy(ids_np).to(dev)
        sides_t = torch.from_numpy(sides_np).to(dev)
        states_t = encode_batch_gpu(ids_t, sides_t, dev)

        with torch.no_grad():
            policy_logits, values = net(states_t)

        policy_flat = policy_logits.reshape(B, -1).cpu().numpy()
        values_np = values.squeeze(-1).cpu().numpy()

        for i, req in enumerate(batch):
            indices = req.legal_action_indices.astype(np.int64)
            legal_logits = policy_flat[i][indices]

            resp = InferenceResponse(
                req_id=req.req_id,
                worker_id=req.worker_id,
                logits=legal_logits.astype(np.float32),
                value=float(values_np[i]),
            )
            self.response_queues[req.worker_id].put(resp)

    def _drain_remaining(self, net: PolicyValueNet, dev: torch.device) -> None:
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
            self._process_batch(net, dev, remaining)


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
        """Send inference request and block for result.

        Args:
            board_ids: (10, 9) int8 — piece channel IDs, -1=empty
            side: int8 — 1=Chess, 0=Xiangqi
            legal_action_indices: (L,) uint16

        Returns:
            (logits, value) — logits shape (L,) float32, value is scalar.
        """
        req_id = self._next_req_id
        self._next_req_id += 1

        req = InferenceRequest(
            req_id=req_id,
            worker_id=self.worker_id,
            board_ids=board_ids,
            side=side,
            legal_action_indices=legal_action_indices,
        )

        t0 = time.perf_counter()
        self.request_queue.put(req)

        resp: InferenceResponse = self.response_queue.get()
        if self.track_latency:
            self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        assert resp.req_id == req_id, f"req_id mismatch: expected {req_id}, got {resp.req_id}"
        assert resp.worker_id == self.worker_id

        return resp.logits, resp.value
