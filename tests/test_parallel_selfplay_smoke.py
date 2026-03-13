"""Parallel self-play smoke test (Mode A, CPU only, no GPU/server).

Verifies:
  - 2 workers x 1 game x 5 sims completes
  - Worker npz files are produced with samples > 0
  - Statistics are reasonable
"""

import numpy as np
import torch
import pytest
from pathlib import Path

from hybrid.rl.az_network import PolicyValueNet
from hybrid.rl.az_selfplay import SelfPlayConfig
from hybrid.rl.az_replay import ReplayBuffer
from hybrid.agents.alphazero_stub import MCTSConfig
from hybrid.rl.az_selfplay_parallel import generate_selfplay_parallel


@pytest.fixture
def cpu_model_ckpt(tmp_path):
    """Save a random PolicyValueNet to a temporary checkpoint."""
    net = PolicyValueNet()
    path = str(tmp_path / "test_model.pt")
    torch.save({"model": net.state_dict()}, path)
    return path


def test_parallel_selfplay_mode_a(cpu_model_ckpt, tmp_path):
    """Mode A (no server): 2 workers x 1 game x 5 sims."""
    out_dir = str(tmp_path / "selfplay_out")

    stats = generate_selfplay_parallel(
        num_workers=2,
        games_per_worker=1,
        selfplay_cfg=SelfPlayConfig(
            temperature=1.0,
            temp_cutoff_ply=5,
            simulations=5,
        ),
        mcts_cfg=MCTSConfig(
            simulations=5,
            dirichlet_alpha=0.3,
            dirichlet_eps=0.25,
        ),
        model_ckpt_path=cpu_model_ckpt,
        out_dir=out_dir,
        seed=42,
        ablation="extra_cannon",
        use_inference_server=False,
    )

    assert stats["total_games"] == 2, f"Expected 2 games, got {stats['total_games']}"
    assert stats["total_samples"] > 0, "Expected samples > 0"
    assert stats["elapsed_seconds"] > 0, "Expected elapsed > 0"
    assert stats["samples_per_sec"] > 0, "Expected samples_per_sec > 0"

    out_path = Path(out_dir)
    for wid in range(2):
        npz_file = out_path / f"worker_{wid}.npz"
        assert npz_file.exists(), f"Missing {npz_file}"
        with np.load(str(npz_file)) as data:
            assert "states" in data, "npz missing 'states' field"
            assert len(data["states"]) > 0, f"Worker {wid} has 0 samples"

    # Verify loadable into ReplayBuffer
    main_buffer = ReplayBuffer()
    for wid in range(2):
        npz_file = str(out_path / f"worker_{wid}.npz")
        worker_buf = ReplayBuffer.load_npz(npz_file)
        main_buffer.append(worker_buf.examples)

    assert len(main_buffer) == stats["total_samples"], (
        f"Buffer samples {len(main_buffer)} != stats {stats['total_samples']}"
    )


def test_parallel_selfplay_games_list(cpu_model_ckpt, tmp_path):
    """Supports per-worker game counts; total_games sums correctly."""
    out_dir = str(tmp_path / "selfplay_out_list")
    stats = generate_selfplay_parallel(
        num_workers=2,
        games_per_worker=[1, 0],
        selfplay_cfg=SelfPlayConfig(
            temperature=1.0,
            temp_cutoff_ply=5,
            simulations=5,
        ),
        mcts_cfg=MCTSConfig(
            simulations=5,
            dirichlet_alpha=0.3,
            dirichlet_eps=0.25,
        ),
        model_ckpt_path=cpu_model_ckpt,
        out_dir=out_dir,
        seed=7,
        ablation="extra_cannon",
        use_inference_server=False,
    )
    assert stats["total_games"] == 1
    assert stats["total_samples"] > 0
