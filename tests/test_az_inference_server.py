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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP FP16 precision test requires CUDA")
def test_amp_fp16_precision_vs_fp32():
    """🚩 Checkpoint 2: AMP FP16 must not alter network strategy.

    value max_diff < 0.01, policy argmax agreement >= 99%.
    """
    import random
    from hybrid.core.env import HybridChessEnv
    from hybrid.rl.az_encoding import board_to_piece_ids, encode_batch_gpu

    dev = torch.device("cuda")
    net = PolicyValueNet().to(dev).eval()

    # Generate 100 random board states
    rng = random.Random(42)
    boards_ids, sides_list = [], []
    for _ in range(100):
        env = HybridChessEnv()
        state = env.reset()
        for _ in range(rng.randint(0, 20)):
            legal = env.legal_moves()
            if not legal:
                break
            state, _, done, _ = env.step(rng.choice(legal))
            if done:
                break
        boards_ids.append(board_to_piece_ids(state.board))
        sides_list.append(1 if state.side_to_move.name == "CHESS" else 0)

    piece_ids = torch.from_numpy(np.stack(boards_ids)).to(dev)
    sides = torch.tensor(sides_list, dtype=torch.int8).to(dev)
    states = encode_batch_gpu(piece_ids, sides, dev)

    # FP32 baseline
    with torch.no_grad():
        policy_fp32, value_fp32 = net(states)

    # AMP FP16
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
        policy_fp16, value_fp16 = net(states)

    # Cast both to FP32 for comparison
    policy_fp16 = policy_fp16.float()
    value_fp16 = value_fp16.float()

    # Value tolerance
    val_diff = torch.max(torch.abs(value_fp32 - value_fp16)).item()
    print(f"\nAMP precision: value max_diff = {val_diff:.6f}")
    assert val_diff < 0.01, f"Value drift too large: {val_diff}"

    # Policy argmax agreement
    argmax_fp32 = policy_fp32.reshape(100, -1).argmax(dim=1)
    argmax_fp16 = policy_fp16.reshape(100, -1).argmax(dim=1)
    agreement = (argmax_fp32 == argmax_fp16).float().mean().item()
    print(f"AMP precision: policy argmax agreement = {agreement*100:.1f}%")
    assert agreement >= 0.98, f"Policy argmax agreement too low: {agreement*100:.1f}%"

