"""Real CUDA integration tests; skipped on CPU compute allocations."""
import multiprocessing as mp
import time

import numpy as np
import pytest
import torch

from hybrid.core.env import HybridChessEnv
from hybrid.rl.az_encoding import board_to_piece_ids
from hybrid.rl.az_inference_server import InferenceClient, inference_server_process
from hybrid.rl.az_shm_pool import SharedMemoryPool
from hybrid.rl.az_selfplay import move_to_action_index
from hybrid.rl.general_model import new_model, model_payload, encode_general, context_features
from hybrid.web_variants import parse_variant

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA compute allocation required")


def test_rule_context_survives_gpu_ipc(tmp_path):
    mp.set_start_method("spawn", force=True)
    torch.set_num_threads(1)
    torch.manual_seed(74)
    net = new_model(96, 4).eval()
    path = tmp_path / "network.pt"
    torch.save(model_payload(net), path)
    states = []
    moves = []
    for flags in ({}, {"chess_palace": True, "knight_block": True}, {"flying_general": False, "xq_queen": True}):
        env = HybridChessEnv(variant=parse_variant(flags))
        states.append(env.reset())
        moves.append(env.legal_moves())
    states[1].ply = 111
    reference = net.cuda()(torch.stack([encode_general(s) for s in states]).cuda())
    pool, requests, stop = SharedMemoryPool(1, 8), mp.Queue(), mp.Event()
    process = mp.Process(target=inference_server_process, args=(str(path), requests, pool, stop, 8, 2., "cuda"))
    process.start()
    try:
        client = InferenceClient(0, requests, pool)
        indices = [np.asarray([move_to_action_index(m) for m in legal], dtype=np.uint16) for legal in moves]
        logits, values = client.predict_batch_raw(np.stack([board_to_piece_ids(s.board) for s in states]),
            np.ones(len(states), dtype=np.int8), indices, np.stack([context_features(s) for s in states]))
        for i, actions in enumerate(indices):
            expected = reference[0][i].flatten().detach().cpu().numpy()[actions.astype(np.int64)]
            assert np.allclose(expected, logits[i], atol=.003, rtol=.01)
        assert np.allclose(reference[1].flatten().detach().cpu().numpy(), values, atol=.003, rtol=.01)
    finally:
        stop.set()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)
        requests.close()
    assert process.exitcode == 0
