# -*- coding: utf-8 -*-
"""Tests for batched inference server (CPU mode, no GPU required).

Verifies:
  - Server process starts, handles requests, and returns correct responses
  - Logits shape, dtype, and value range are correct
  - Clean server shutdown
"""

import multiprocessing as mp
import numpy as np
import torch
import pytest

from hybrid.rl.az_network import PolicyValueNet
from hybrid.rl.az_inference_server import (
    InferenceClient,
    inference_server_process,
)


@pytest.fixture
def cpu_model_ckpt(tmp_path):
    """Save a randomly initialized PolicyValueNet checkpoint."""
    net = PolicyValueNet()
    path = str(tmp_path / "test_model.pt")
    torch.save({"model": net.state_dict()}, path)
    return path


def test_inference_server_cpu(cpu_model_ckpt):
    """Start CPU inference server, send 3 requests, verify responses."""
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    request_queue = mp.Queue()
    response_queues = {0: mp.Queue()}
    stop_event = mp.Event()

    server = mp.Process(
        target=inference_server_process,
        args=(
            cpu_model_ckpt,
            request_queue,
            response_queues,
            stop_event,
        ),
        kwargs={"device": "cpu", "max_batch_size": 8, "timeout_ms": 50.0},
        daemon=True,
    )
    server.start()

    client = InferenceClient(
        worker_id=0,
        request_queue=request_queue,
        response_queue=response_queues[0],
    )

    legal_sizes = [4, 10, 1]
    for i, L in enumerate(legal_sizes):
        board_ids = np.random.randint(-1, 13, size=(10, 9)).astype(np.int8)
        side = np.int8(i % 2)
        legal_indices = np.random.choice(8280, size=L, replace=False).astype(np.uint16)

        logits, value = client.predict_raw(board_ids, side, legal_indices)

        assert logits.shape == (L,), f"Request {i}: expected logits shape ({L},), got {logits.shape}"
        assert logits.dtype == np.float32, f"Request {i}: expected dtype float32, got {logits.dtype}"
        assert np.all(np.isfinite(logits)), f"Request {i}: logits contain NaN/inf"

        assert isinstance(value, float), f"Request {i}: value should be float"
        assert np.isfinite(value), f"Request {i}: value is not finite"
        assert -1.0 <= value <= 1.0, f"Request {i}: value={value} outside [-1,1]"

    stop_event.set()
    server.join(timeout=10.0)
    if server.is_alive():
        server.terminate()
        server.join(timeout=5.0)
    assert not server.is_alive(), "Server process did not exit cleanly"


def test_inference_server_multi_worker(cpu_model_ckpt):
    """Test multiple workers using the server (2 workers, 2 requests each)."""
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    request_queue = mp.Queue()
    response_queues = {0: mp.Queue(), 1: mp.Queue()}
    stop_event = mp.Event()

    server = mp.Process(
        target=inference_server_process,
        args=(
            cpu_model_ckpt,
            request_queue,
            response_queues,
            stop_event,
        ),
        kwargs={"device": "cpu", "max_batch_size": 8, "timeout_ms": 50.0},
        daemon=True,
    )
    server.start()

    clients = {
        wid: InferenceClient(wid, request_queue, response_queues[wid])
        for wid in range(2)
    }

    for req_i in range(2):
        for wid in range(2):
            board_ids = np.random.randint(-1, 13, size=(10, 9)).astype(np.int8)
            side = np.int8(req_i % 2)
            legal_indices = np.array([0, 42, 100], dtype=np.uint16)
            logits, value = clients[wid].predict_raw(board_ids, side, legal_indices)

            assert logits.shape == (3,)
            assert np.isfinite(value)

    stop_event.set()
    server.join(timeout=10.0)
    if server.is_alive():
        server.terminate()
        server.join(timeout=5.0)

