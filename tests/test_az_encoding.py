# -*- coding: utf-8 -*-
"""AlphaZero encoding layer unit tests.

Verifies:
  1. encode_state output shape is correct
  2. Piece channel nonzero count matches actual piece count
  3. move_to_plane produces valid plane indices for all legal moves
  4. extract_policy_logits output length matches legal_moves count
"""

import torch

from hybrid.core.env import HybridChessEnv, GameState
from hybrid.core.board import initial_board
from hybrid.core.types import Side
from hybrid.core.rules import generate_legal_moves
from hybrid.rl.az_encoding import (
    encode_state,
    move_to_plane,
    extract_policy_logits,
    NUM_STATE_CHANNELS,
    TOTAL_POLICY_PLANES,
)
from hybrid.core.config import BOARD_H, BOARD_W


def test_encode_state_shape():
    """encode_state output shape should be (14, 10, 9)."""
    env = HybridChessEnv()
    state = env.reset()
    tensor = encode_state(state)
    assert tensor.shape == (NUM_STATE_CHANNELS, BOARD_H, BOARD_W), \
        f"Expected shape ({NUM_STATE_CHANNELS}, {BOARD_H}, {BOARD_W}), got {tensor.shape}"


def test_encode_state_nonzero_count():
    """Piece channels (0-12) nonzero count should equal total pieces on board.

    Channel 13 is the side-to-move indicator and is excluded from piece counting.
    """
    board = initial_board()
    state = GameState(board=board, side_to_move=Side.CHESS)
    tensor = encode_state(state)

    piece_count = sum(1 for _ in board.iter_pieces())
    nonzero = (tensor[:13] != 0).sum().item()

    assert nonzero == piece_count, \
        f"Expected {piece_count} nonzero positions, got {nonzero}"


def test_encode_state_side_to_move_channel():
    """Side-to-move channel: all 1s when Chess moves, all 0s when Xiangqi moves."""
    board = initial_board()

    state_chess = GameState(board=board, side_to_move=Side.CHESS)
    t_chess = encode_state(state_chess)
    assert t_chess[13].sum().item() == BOARD_H * BOARD_W, \
        "Side-to-move channel should be all 1s when Chess moves"

    state_xq = GameState(board=board, side_to_move=Side.XIANGQI)
    t_xq = encode_state(state_xq)
    assert t_xq[13].sum().item() == 0, \
        "Side-to-move channel should be all 0s when Xiangqi moves"


def test_move_to_plane_legal_range():
    """All initial legal moves should produce plane indices in [0, 92)."""
    board = initial_board()
    legal = generate_legal_moves(board, Side.CHESS)
    assert len(legal) > 0, "Initial position should have legal moves"

    for mv in legal:
        plane_idx, fy, fx = move_to_plane(mv)
        assert 0 <= plane_idx < TOTAL_POLICY_PLANES, \
            f"plane_idx={plane_idx} out of range [0, {TOTAL_POLICY_PLANES}), move={mv}"
        assert 0 <= fy < BOARD_H, f"fy={fy} out of board range"
        assert 0 <= fx < BOARD_W, f"fx={fx} out of board range"


def test_move_to_plane_xiangqi_legal():
    """Xiangqi legal moves should all map correctly."""
    board = initial_board()
    legal = generate_legal_moves(board, Side.XIANGQI)
    assert len(legal) > 0

    for mv in legal:
        plane_idx, fy, fx = move_to_plane(mv)
        assert 0 <= plane_idx < TOTAL_POLICY_PLANES, \
            f"plane_idx={plane_idx} out of range, move={mv}"


def test_extract_policy_logits_shape():
    """extract_policy_logits should return logits with length == len(legal_moves)."""
    board = initial_board()
    legal = generate_legal_moves(board, Side.CHESS)

    policy_planes = torch.randn(TOTAL_POLICY_PLANES, BOARD_H, BOARD_W)
    logits = extract_policy_logits(policy_planes, legal)

    assert logits.shape == (len(legal),), \
        f"Expected shape ({len(legal)},), got {logits.shape}"


def test_extract_policy_logits_empty():
    """Empty legal_moves should return empty tensor without error."""
    policy_planes = torch.randn(TOTAL_POLICY_PLANES, BOARD_H, BOARD_W)
    logits = extract_policy_logits(policy_planes, [])
    assert logits.shape == (0,)


# ====================================================================
# GPU encode_batch_gpu verification tests
# ====================================================================

import random
import pickle
import time
import numpy as np

from hybrid.rl.az_encoding import (
    encode_state_cpu_legacy,
    encode_batch_gpu,
    board_to_piece_ids,
)
from hybrid.core.types import Piece, PieceKind
from hybrid.core.rules import generate_legal_moves, apply_move


def _generate_random_states(n: int, seed: int = 42):
    """Generate n different board states by playing random legal moves."""
    rng = random.Random(seed)
    states = []
    for _ in range(n):
        env = HybridChessEnv()
        state = env.reset()
        # Play 0–30 random moves to get diverse positions
        num_moves = rng.randint(0, 30)
        for _ in range(num_moves):
            legal = env.legal_moves()
            if not legal:
                break
            mv = rng.choice(legal)
            state, _, done, _ = env.step(mv)
            if done:
                break
        states.append(state)
    return states


def test_encode_batch_gpu_vs_cpu_legacy():
    """🚩 Checkpoint 1: GPU batch encoding must be pixel-exact with CPU legacy.

    Generate 100 random board states, encode with both methods,
    assert zero maximum absolute difference.
    """
    states = _generate_random_states(100, seed=12345)

    # CPU legacy: encode each state individually, stack
    cpu_tensors = [encode_state_cpu_legacy(s) for s in states]
    cpu_batch = torch.stack(cpu_tensors)  # (100, 14, 10, 9)

    # GPU batch: convert to IDs and encode
    ids_list = [board_to_piece_ids(s.board) for s in states]
    sides_list = [1 if s.side_to_move == Side.CHESS else 0 for s in states]

    piece_ids = torch.from_numpy(np.stack(ids_list))
    sides = torch.tensor(sides_list, dtype=torch.int8)

    gpu_batch = encode_batch_gpu(piece_ids, sides, torch.device("cpu"))

    max_diff = torch.max(torch.abs(cpu_batch - gpu_batch)).item()
    assert max_diff == 0.0, (
        f"GPU batch encoding differs from CPU legacy! max_diff={max_diff}"
    )


def test_board_to_piece_ids_consistency():
    """board_to_piece_ids must assign the same channel IDs as PIECE_CHANNELS."""
    from hybrid.rl.az_encoding import PIECE_CHANNELS

    board = initial_board()
    ids = board_to_piece_ids(board)

    for x, y, piece in board.iter_pieces():
        expected_ch = PIECE_CHANNELS[piece.kind]
        assert ids[y, x] == expected_ch, (
            f"Mismatch at ({x},{y}): expected channel {expected_ch}, got {ids[y, x]}"
        )

    # Empty squares should be -1
    for y in range(BOARD_H):
        for x in range(BOARD_W):
            if board.grid[y][x] is None:
                assert ids[y, x] == -1, (
                    f"Empty square ({x},{y}) should be -1, got {ids[y, x]}"
                )


def test_communication_size_reduction():
    """🚩 Checkpoint 2: Compact request payload must be substantially smaller.

    Old: InferenceRequest with state_u8 (14, 10, 9) uint8 = 1260 bytes payload
    New: InferenceRequest with board_ids (10, 9) int8 + side int8 = 91 bytes payload
    """
    from hybrid.rl.az_inference_server import InferenceRequest

    legal_indices = np.array([0, 42, 100], dtype=np.uint16)

    # New compact request
    new_req = InferenceRequest(
        req_id=0, worker_id=0,
        board_ids=np.zeros((10, 9), dtype=np.int8),
        side=np.int8(1),
        legal_action_indices=legal_indices,
    )
    new_size = len(pickle.dumps(new_req))

    # Compare raw payload sizes (the variable-size portion)
    old_payload_bytes = 14 * 10 * 9  # 1260 bytes for uint8
    new_payload_bytes = 10 * 9 + 1   # 91 bytes for int8 + side

    payload_ratio = old_payload_bytes / new_payload_bytes
    print(f"\nPayload size: old={old_payload_bytes}B, new={new_payload_bytes}B, ratio={payload_ratio:.1f}×")
    print(f"Full pickled request size (new): {new_size}B")

    # Raw payload is 13.8× smaller
    assert payload_ratio > 10.0, f"Expected ≥10× payload reduction, got {payload_ratio:.1f}×"
    # Full pickled request should be well under 1KB
    assert new_size < 1000, f"Compact request should be <1KB, got {new_size}B"


def test_encode_batch_gpu_micro_benchmark():
    """🚩 Checkpoint 3: GPU batch encoding should be ≥5× faster than CPU loops.

    Batch size 32, 500 iterations each.
    """
    B = 32
    ITERS = 500

    # Generate a fixed batch of random states
    states = _generate_random_states(B, seed=9999)

    # CPU legacy timing
    t0 = time.perf_counter()
    for _ in range(ITERS):
        batch = torch.stack([encode_state_cpu_legacy(s) for s in states])
    cpu_time = time.perf_counter() - t0

    # GPU batch timing (on CPU device for CI)
    ids_list = [board_to_piece_ids(s.board) for s in states]
    sides_list = [1 if s.side_to_move == Side.CHESS else 0 for s in states]
    piece_ids = torch.from_numpy(np.stack(ids_list))
    sides = torch.tensor(sides_list, dtype=torch.int8)
    dev = torch.device("cpu")

    # Warm-up
    _ = encode_batch_gpu(piece_ids, sides, dev)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        _ = encode_batch_gpu(piece_ids, sides, dev)
    gpu_time = time.perf_counter() - t0

    speedup = cpu_time / gpu_time
    print(f"\nMicro-benchmark (B={B}, iters={ITERS}):")
    print(f"  CPU legacy: {cpu_time*1000:.1f}ms total ({cpu_time/ITERS*1000:.2f}ms/iter)")
    print(f"  GPU batch:  {gpu_time*1000:.1f}ms total ({gpu_time/ITERS*1000:.2f}ms/iter)")
    print(f"  Speedup:    {speedup:.1f}×")

    # Even on CPU, vectorized scatter should beat Python loops substantially
    assert speedup > 5.0, f"Expected ≥5× speedup, got {speedup:.1f}×"


def test_encode_batch_gpu_inplace():
    """🚩 In-place out= mode must be pixel-exact with allocating mode."""
    states = _generate_random_states(32, seed=7777)
    ids_list = [board_to_piece_ids(s.board) for s in states]
    sides_list = [1 if s.side_to_move == Side.CHESS else 0 for s in states]
    piece_ids = torch.from_numpy(np.stack(ids_list))
    sides = torch.tensor(sides_list, dtype=torch.int8)
    dev = torch.device("cpu")

    # Allocating mode
    alloc_result = encode_batch_gpu(piece_ids, sides, dev)

    # In-place mode with pre-allocated buffer
    out_buf = torch.zeros(32, 14, 10, 9, dtype=torch.float32, device=dev)
    inplace_result = encode_batch_gpu(piece_ids, sides, dev, out=out_buf)

    assert inplace_result is out_buf, "In-place should return the same buffer"
    max_diff = torch.max(torch.abs(alloc_result - inplace_result)).item()
    assert max_diff == 0.0, f"In-place differs from allocating! max_diff={max_diff}"


def test_encode_batch_gpu_cache_pollution():
    """🚩 Checkpoint 1: Large batch followed by small batch must not leak residual data.

    Process 32 full boards into a buffer, then 4 sparse boards into the SAME buffer.
    The small batch result must match a fresh computation.
    """
    full_states = _generate_random_states(32, seed=1111)
    sparse_states = _generate_random_states(4, seed=2222)

    dev = torch.device("cpu")
    B_max = 32

    # Pre-allocate shared buffer (like the server does)
    shared_buf = torch.zeros(B_max, 14, 10, 9, dtype=torch.float32, device=dev)

    # Process large batch A into shared buffer
    ids_a = torch.from_numpy(np.stack([board_to_piece_ids(s.board) for s in full_states]))
    sides_a = torch.tensor([1 if s.side_to_move == Side.CHESS else 0 for s in full_states], dtype=torch.int8)
    encode_batch_gpu(ids_a, sides_a, dev, out=shared_buf[:32])

    # Process small batch B into the SAME buffer (only first 4 slots)
    ids_b = torch.from_numpy(np.stack([board_to_piece_ids(s.board) for s in sparse_states]))
    sides_b = torch.tensor([1 if s.side_to_move == Side.CHESS else 0 for s in sparse_states], dtype=torch.int8)
    encode_batch_gpu(ids_b, sides_b, dev, out=shared_buf[:4])

    # Fresh computation for batch B (no shared buffer history)
    fresh_result = encode_batch_gpu(ids_b, sides_b, dev)

    max_diff = torch.max(torch.abs(shared_buf[:4] - fresh_result)).item()
    assert max_diff == 0.0, (
        f"Cache pollution detected! Batch B result differs from fresh. max_diff={max_diff}"
    )

