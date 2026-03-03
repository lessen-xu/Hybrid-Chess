# -*- coding: utf-8 -*-
"""Multi-process parallel self-play data generation.

Mode A (CPU): each worker loads model locally for CPU inference.
Mode B (GPU): workers send requests to a shared InferenceServer for batched GPU inference.
"""

from __future__ import annotations
from typing import Dict, Optional, Sequence, Union
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch

from hybrid.rl.az_selfplay import self_play_game, SelfPlayConfig
from hybrid.rl.az_replay import ReplayBuffer
from hybrid.rl.az_network import PolicyValueNet
from hybrid.agents.alphazero_stub import (
    AlphaZeroMiniAgent,
    MCTSConfig,
    TorchPolicyValueModel,
)
from hybrid.core.env import HybridChessEnv


# ====================================================================
# Worker function
# ====================================================================

def selfplay_worker(
    worker_id: int,
    num_games: int,
    selfplay_cfg: SelfPlayConfig,
    mcts_cfg: MCTSConfig,
    model_ckpt_path: str,
    out_npz_path: str,
    seed: int,
    ablation: str = "extra_cannon",
    request_queue: Optional[mp.Queue] = None,
    response_queue: Optional[mp.Queue] = None,
    track_latency: bool = False,
    endgame_ratio: float = 0.0,
    use_cpp: bool = False,
) -> None:
    """Worker entry point: run num_games self-play games, save to out_npz_path."""
    from hybrid.rl.az_runner import _apply_ablation
    _apply_ablation(ablation)

    # Build model (Mode B with inference server, or Mode A with local CPU)
    client = None
    if request_queue is not None and response_queue is not None:
        from hybrid.rl.az_inference_server import InferenceClient
        from hybrid.agents.az_remote_model import RemotePolicyValueModel
        client = InferenceClient(
            worker_id, request_queue, response_queue,
            track_latency=track_latency,
        )
        model = RemotePolicyValueModel(client)
    else:
        net = PolicyValueNet()
        ckpt = torch.load(model_ckpt_path, map_location="cpu", weights_only=True)
        net.load_state_dict(ckpt["model"])
        net.eval()
        model = TorchPolicyValueModel(net, device="cpu")

    agent = AlphaZeroMiniAgent(model=model, cfg=mcts_cfg, seed=seed, use_cpp=use_cpp)
    env = HybridChessEnv(use_cpp=use_cpp)

    all_examples = []
    all_records = []
    import random as _random
    endgame_rng = _random.Random(seed + 9999)
    for game_i in range(num_games):
        initial_state = None
        if endgame_ratio > 0 and endgame_rng.random() < endgame_ratio:
            from hybrid.rl.endgame_spawner import generate_endgame_board
            from hybrid.core.env import GameState
            eg_board, eg_side = generate_endgame_board(endgame_rng)
            initial_state = GameState(
                board=eg_board, side_to_move=eg_side, ply=0, repetition={}
            )
        examples, record = self_play_game(
            env, agent, selfplay_cfg, initial_state=initial_state,
        )
        all_examples.extend(examples)
        all_records.append(record)

    buf = ReplayBuffer()
    buf.examples = all_examples
    buf.save_npz(out_npz_path)

    # Save GameRecords as JSON for diagnostics
    import json
    from dataclasses import asdict
    records_path = out_npz_path.replace(".npz", "_records.json")
    with open(records_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_records], f)

    # Save latency stats (benchmark mode)
    if client is not None and track_latency and client.latencies_ms:
        latency_path = out_npz_path.replace(".npz", "_latency.npy")
        np.save(latency_path, np.array(client.latencies_ms, dtype=np.float64))


# ====================================================================
# Main-process orchestration
# ====================================================================

def generate_selfplay_parallel(
    num_workers: int,
    games_per_worker: Union[int, Sequence[int]],
    selfplay_cfg: SelfPlayConfig,
    mcts_cfg: MCTSConfig,
    model_ckpt_path: str,
    out_dir: str,
    seed: int,
    ablation: str = "extra_cannon",
    use_inference_server: bool = False,
    inference_batch_size: int = 32,
    inference_timeout_ms: float = 5.0,
    inference_device: str = "cuda",
    track_latency: bool = False,
    endgame_ratio: float = 0.0,
    use_cpp: bool = False,
) -> Dict[str, float]:
    """Orchestrate multi-process parallel self-play.

    Returns dict with elapsed_seconds, total_games, total_samples, samples_per_sec, etc.
    """
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()

    worker_npz_paths = [
        str(out_path / f"worker_{wid}.npz") for wid in range(num_workers)
    ]

    if isinstance(games_per_worker, int):
        per_worker_games = [games_per_worker] * num_workers
    else:
        per_worker_games = [int(x) for x in games_per_worker]
        if len(per_worker_games) != num_workers:
            raise ValueError(
                f"games_per_worker length {len(per_worker_games)} != num_workers {num_workers}"
            )
    if any(g < 0 for g in per_worker_games):
        raise ValueError("games_per_worker must be non-negative")

    # Optional: start inference server (Mode B)
    server_proc = None
    stop_event = None
    request_queue = None
    response_queues = None
    stats_queue = None

    if use_inference_server:
        from hybrid.rl.az_inference_server import inference_server_process

        request_queue = mp.Queue()
        response_queues = {wid: mp.Queue() for wid in range(num_workers)}
        stop_event = mp.Event()
        stats_queue = mp.Queue()

        server_proc = mp.Process(
            target=inference_server_process,
            args=(
                model_ckpt_path,
                request_queue,
                response_queues,
                stop_event,
                inference_batch_size,
                inference_timeout_ms,
                inference_device,
                stats_queue,
            ),
            daemon=True,
        )
        server_proc.start()

    # Start worker processes
    workers = []
    for wid in range(num_workers):
        kwargs = dict(
            worker_id=wid,
            num_games=per_worker_games[wid],
            selfplay_cfg=selfplay_cfg,
            mcts_cfg=mcts_cfg,
            model_ckpt_path=model_ckpt_path,
            out_npz_path=worker_npz_paths[wid],
            seed=seed + wid * 10000,
            ablation=ablation,
            track_latency=track_latency and use_inference_server,
            endgame_ratio=endgame_ratio,
            use_cpp=use_cpp,
        )
        if use_inference_server:
            kwargs["request_queue"] = request_queue
            kwargs["response_queue"] = response_queues[wid]

        p = mp.Process(target=selfplay_worker, kwargs=kwargs)
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    for i, p in enumerate(workers):
        if p.exitcode != 0:
            raise RuntimeError(f"selfplay_worker {i} exited with code {p.exitcode}")

    # Shut down inference server
    server_stats = {}
    if server_proc is not None:
        stop_event.set()
        server_proc.join(timeout=10.0)
        if server_proc.is_alive():
            server_proc.terminate()
            server_proc.join(timeout=5.0)
        import queue as _queue
        try:
            server_stats = stats_queue.get_nowait()
        except _queue.Empty:
            pass

    elapsed = time.monotonic() - start_time

    total_samples = 0
    for npz_path in worker_npz_paths:
        if os.path.exists(npz_path):
            with np.load(npz_path) as data:
                total_samples += len(data["states"])

    total_games = sum(per_worker_games)
    result = {
        "elapsed_seconds": round(elapsed, 2),
        "total_games": total_games,
        "total_samples": total_samples,
        "samples_per_sec": round(total_samples / max(elapsed, 0.001), 1),
    }
    result.update(server_stats)

    # Collect client latency stats
    if track_latency and use_inference_server:
        all_latencies = []
        for wid in range(num_workers):
            lat_path = str(out_path / f"worker_{wid}_latency.npy")
            if os.path.exists(lat_path):
                lat = np.load(lat_path)
                all_latencies.append(lat)
        if all_latencies:
            merged = np.concatenate(all_latencies)
            result["latency_p50_ms"] = round(float(np.percentile(merged, 50)), 2)
            result["latency_p95_ms"] = round(float(np.percentile(merged, 95)), 2)
            result["latency_mean_ms"] = round(float(np.mean(merged)), 2)
            result["latency_count"] = len(merged)

    return result
