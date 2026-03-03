#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Profile the multi-worker + GPU server inference pipeline.

Spawns 1 InferenceServer + N workers, each running a short fixed workload
(2 moves × configurable simulations). Reports:
  - Throughput (states/sec)
  - Server: avg batch size, GPU duty cycle, queue wait
  - Worker: MCTS CPU time vs IPC wait time

Usage:
  python scripts/profile_server_path.py --workers 4 --sims 200
  python scripts/profile_server_path.py --workers 8 --sims 200 --batch-size 128
"""

import scripts._fix_encoding  # noqa: F401

import argparse
import multiprocessing as mp
import os
import queue
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


def profile_worker(
    worker_id: int,
    model_ckpt_path: str,
    request_queue: mp.Queue,
    pool,
    result_queue: mp.Queue,
    num_plies: int,
    simulations: int,
    use_cpp: bool,
):
    """Worker: play num_plies moves with MCTS, report timing."""
    from hybrid.rl.az_inference_server import InferenceClient
    from hybrid.agents.az_remote_model import RemotePolicyValueModel
    from hybrid.agents.alphazero_stub import AlphaZeroMiniAgent, MCTSConfig
    from hybrid.core.env import HybridChessEnv

    client = InferenceClient(worker_id, request_queue, pool)
    model = RemotePolicyValueModel(client)
    agent = AlphaZeroMiniAgent(
        model=model,
        cfg=MCTSConfig(simulations=simulations, dirichlet_eps=0.25),
        seed=42 + worker_id,
        use_cpp=use_cpp,
    )
    env = HybridChessEnv(use_cpp=use_cpp)

    t_wall_start = time.perf_counter()
    state = env.reset()

    for ply_i in range(num_plies):
        legal = env.legal_moves()
        if not legal:
            state = env.reset()
            legal = env.legal_moves()
        move = agent.select_move(state, legal)
        state, _, done, _ = env.step(move)
        if done:
            state = env.reset()

    t_wall = time.perf_counter() - t_wall_start

    result_queue.put({
        "worker_id": worker_id,
        "wall_s": round(t_wall, 3),
        "ipc_wait_s": round(model.ipc_wait_s, 3),
        "predict_count": model.predict_count,
        "mcts_cpu_s": round(t_wall - model.ipc_wait_s, 3),
    })


def main():
    parser = argparse.ArgumentParser(description="Profile server inference pipeline")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations per move")
    parser.add_argument("--plies", type=int, default=2, help="Moves per worker")
    parser.add_argument("--batch-size", type=int, default=128, help="Server max batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Server device")
    parser.add_argument("--use-cpp", action="store_true", help="Use C++ engine")
    args = parser.parse_args()

    N = args.workers
    print(f"=== Server-Path Profile ===")
    print(f"Workers={N}, sims={args.sims}, plies={args.plies}, "
          f"batch_max={args.batch_size}, device={args.device}, use_cpp={args.use_cpp}")
    print()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Create a temporary model checkpoint
    from hybrid.rl.az_network import PolicyValueNet
    net = PolicyValueNet()
    ckpt_path = os.path.join(tempfile.gettempdir(), "profile_model.pt")
    torch.save({"model": net.state_dict()}, ckpt_path)

    # Shared memory pool + queues
    from hybrid.rl.az_shm_pool import SharedMemoryPool
    pool = SharedMemoryPool(max_workers=N)
    request_queue = mp.Queue()
    stop_event = mp.Event()
    stats_queue = mp.Queue()
    result_queue = mp.Queue()

    # Start server
    from hybrid.rl.az_inference_server import inference_server_process
    server = mp.Process(
        target=inference_server_process,
        args=(ckpt_path, request_queue, pool, stop_event,
              args.batch_size, 5.0, args.device, stats_queue),
        daemon=True,
    )
    server.start()

    # Start workers
    t_total_start = time.perf_counter()
    workers = []
    for wid in range(N):
        p = mp.Process(
            target=profile_worker,
            args=(wid, ckpt_path, request_queue, pool,
                  result_queue, args.plies, args.sims, args.use_cpp),
        )
        p.start()
        workers.append(p)

    # Wait for workers
    for p in workers:
        p.join(timeout=300)
    t_total = time.perf_counter() - t_total_start

    # Collect worker results
    worker_results = []
    while not result_queue.empty():
        worker_results.append(result_queue.get_nowait())
    worker_results.sort(key=lambda r: r["worker_id"])

    # Shut down server
    stop_event.set()
    server.join(timeout=10)
    if server.is_alive():
        server.terminate()
        server.join(timeout=5)

    # Collect server stats
    server_stats = {}
    try:
        server_stats = stats_queue.get_nowait()
    except queue.Empty:
        pass

    # ── Report ──
    total_predicts = sum(r["predict_count"] for r in worker_results)
    print(f"{'='*60}")
    print(f"  RESULTS  (wall={t_total:.1f}s)")
    print(f"{'='*60}")
    print()

    # Throughput
    throughput = total_predicts / t_total if t_total > 0 else 0
    print(f"Throughput:  {throughput:.0f} states/sec  "
          f"({total_predicts} total predictions)")
    print()

    # Server stats
    if server_stats:
        batches = server_stats.get("inference_batches", 0)
        avg_bs = server_stats.get("avg_batch_size", 0)
        max_bs = server_stats.get("max_batch_size_seen", 0)
        fill = server_stats.get("avg_batch_fill_pct", 0)
        qw = server_stats.get("queue_wait_s", 0)
        gc = server_stats.get("gpu_compute_s", 0)
        server_total = qw + gc
        gpu_duty = 100 * gc / server_total if server_total > 0 else 0
        print(f"Server:")
        print(f"  Batches:        {batches}")
        print(f"  Avg batch size: {avg_bs:.1f} / {args.batch_size}  ({fill:.0f}% fill)")
        print(f"  Max batch size: {max_bs}")
        print(f"  GPU compute:    {gc:.2f}s  ({gpu_duty:.0f}% duty)")
        print(f"  Queue wait:     {qw:.2f}s  ({100-gpu_duty:.0f}% idle)")
    else:
        print("Server: no stats collected")
    print()

    # Worker stats
    if worker_results:
        walls = [r["wall_s"] for r in worker_results]
        ipcs = [r["ipc_wait_s"] for r in worker_results]
        cpus = [r["mcts_cpu_s"] for r in worker_results]
        preds = [r["predict_count"] for r in worker_results]

        mean_wall = np.mean(walls)
        mean_ipc = np.mean(ipcs)
        mean_cpu = np.mean(cpus)
        ipc_pct = 100 * mean_ipc / mean_wall if mean_wall > 0 else 0
        cpu_pct = 100 * mean_cpu / mean_wall if mean_wall > 0 else 0

        print(f"Workers (mean of {N}):")
        print(f"  Wall time:   {mean_wall:.2f}s")
        print(f"  IPC wait:    {mean_ipc:.2f}s  ({ipc_pct:.0f}%)")
        print(f"  MCTS CPU:    {mean_cpu:.2f}s  ({cpu_pct:.0f}%)")
        print(f"  Predictions: {np.mean(preds):.0f} per worker")
        print()

        print(f"Per-worker detail:")
        print(f"  {'WID':>4}  {'wall':>7}  {'ipc':>7}  {'cpu':>7}  {'preds':>6}  {'ipc%':>5}")
        for r in worker_results:
            ipc_p = 100 * r["ipc_wait_s"] / r["wall_s"] if r["wall_s"] > 0 else 0
            print(f"  {r['worker_id']:>4}  {r['wall_s']:>7.2f}  "
                  f"{r['ipc_wait_s']:>7.2f}  {r['mcts_cpu_s']:>7.2f}  "
                  f"{r['predict_count']:>6}  {ipc_p:>4.0f}%")

    # Cleanup
    try:
        os.remove(ckpt_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
