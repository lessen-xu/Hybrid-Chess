"""Complete, independently seeded games with atomic per-game persistence."""
from __future__ import annotations

from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import queue
import random
import time
import traceback

import numpy as np
import torch

from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig
from hybrid.agents.alphazero_stub import AlphaZeroMiniAgent, MCTSConfig, TorchPolicyValueModel
from hybrid.core.env import HybridChessEnv
from hybrid.core.render import render_board
from hybrid.core.rules import terminal_info, TerminalStatus
from hybrid.core.types import Side
from hybrid.rl.az_selfplay import Example, move_to_action_index
from hybrid.rl.general_model import encode_general, sample_variant, load_general_model, ENCODING_VERSION
from hybrid.rl.run_store import atomic_write, sha256
from hybrid.web_variants import parse_variant


def save_game(path, examples, metadata):
    offsets = np.cumsum([0] + [len(ex.pi_indices) for ex in examples], dtype=np.int32)
    atomic_write(path, lambda f: np.savez_compressed(
        f, encoding_version=ENCODING_VERSION, metadata=json.dumps(metadata),
        states=np.stack([ex.state for ex in examples]).astype(np.float16),
        offsets=offsets, indices=np.concatenate([ex.pi_indices for ex in examples]),
        probs=np.concatenate([ex.pi_probs for ex in examples]),
        sides=np.array([int(ex.side_to_move == Side.CHESS) for ex in examples], dtype=np.uint8),
        values=np.array([ex.z for ex in examples], dtype=np.float32),
    ))


def load_game(path):
    with np.load(path, allow_pickle=False) as data:
        if int(data["encoding_version"]) != ENCODING_VERSION or data["states"].shape[1] != 29:
            raise ValueError("Incompatible game encoding")
        meta = json.loads(str(data["metadata"]))
        examples = []
        for i, state in enumerate(data["states"]):
            start, end = data["offsets"][i:i+2]
            examples.append(Example(state.copy(), data["indices"][start:end].copy(),
                data["probs"][start:end].copy(), Side.CHESS if data["sides"][i] else Side.XIANGQI,
                float(data["values"][i])))
    return examples, meta


def play_training_game(job, model=None, cancelled=None):
    """Return None for interrupted games; never label a wall-time cutoff as a draw."""
    seed, game_id = job["seed"], job["game_id"]
    rng = random.Random(seed * 1_000_003 + game_id)
    target_var = job.get("target_variant")
    if target_var:
        variant = parse_variant(target_var)
        family = target_var if isinstance(target_var, str) else "golden"
    else:
        variant, family = sample_variant(game_id, seed)
    max_plies = job.get("max_plies", 400)
    env = HybridChessEnv(variant=variant, use_cpp=job.get("use_cpp", False), max_plies=max_plies)
    state = env.reset()
    teacher = job["mode"] == "teacher"
    if teacher:
        agent = AlphaBetaAgent(SearchConfig(depth=2, time_limit_seconds=job.get("teacher_seconds", .25),
                                            max_plies=max_plies))
    else:
        agent = AlphaZeroMiniAgent(model, MCTSConfig(simulations=job["simulations"],
            discount_factor=1., max_plies=max_plies, leaf_batch_size=8), seed=rng.randrange(2**32),
            use_cpp=job.get("use_cpp", False))
    examples, moves = [], []
    boards = [render_board(state.board)] if job.get("record", False) else []
    started = time.monotonic()
    while True:
        if time.time() >= job["deadline"] or (cancelled is not None and cancelled.is_set()):
            return None
        env._set_active_variant()
        info = terminal_info(state.board, state.side_to_move, state.repetition, state.ply, max_plies)
        if info.status != TerminalStatus.ONGOING:
            break
        legal = env.legal_moves()
        if teacher:
            recommended = agent.select_move(state, legal)
            pi = {m: float(m == recommended) for m in legal}
            chosen = rng.choice(legal) if state.ply < 12 and rng.random() < .2 else recommended
        else:
            chosen, pi, _ = agent.select_move_with_pi(state, legal,
                temperature=1. if state.ply < 20 else 0., add_noise=True)
        examples.append(Example(encode_general(state).numpy().astype(np.float16),
            np.array([move_to_action_index(m) for m in legal], dtype=np.uint16),
            np.array([pi.get(m, 0.) for m in legal], dtype=np.float32), state.side_to_move))
        moves.append(f"{'abcdefghi'[chosen.fx]}{chosen.fy+1}-{'abcdefghi'[chosen.tx]}{chosen.ty+1}" +
                     ("=" + {"QUEEN":"Q", "ROOK":"R", "BISHOP":"B", "KNIGHT":"N"}[chosen.promotion.name]
                      if chosen.promotion else ""))
        state, _, done, info = env.step(chosen)
        if boards:
            boards.append(render_board(state.board))
        if done:
            break
    if not examples:
        raise RuntimeError("Empty game from initial position")
    for example in examples:
        example.z = 0. if info.winner is None else (1. if info.winner == example.side_to_move else -1.)
    metadata = {"game_id": game_id, "seed": seed, "mode": job["mode"], "family": family,
        "variant": variant.to_dict(), "model_sha256": job.get("model_sha256"),
        "winner": info.winner.name.lower() if info.winner else None, "reason": info.reason,
        "plies": state.ply, "samples": len(examples), "seconds": time.monotonic()-started,
        "moves": moves, "states_ascii": boards, "result": info.winner.name.lower()+"_win" if info.winner else "draw"}
    save_game(job["path"], examples, metadata)
    return metadata


def _worker(wid, tasks, results, cancelled, model_path, requests, pool):
    torch.set_num_threads(1)
    model = None
    if model_path:
        if requests is not None:
            from hybrid.rl.az_inference_server import InferenceClient
            from hybrid.agents.az_remote_model import RemotePolicyValueModel
            model = RemotePolicyValueModel(InferenceClient(wid, requests, pool))
        else:
            model = TorchPolicyValueModel(load_general_model(model_path), "cpu")
    while not cancelled.is_set():
        job = tasks.get()
        if job is None:
            return
        try:
            result = play_training_game(job, model, cancelled)
            results.put((job["game_id"], result, None))
        except Exception:
            results.put((job["game_id"], None, traceback.format_exc()))
            return


def collect_games(jobs, workers, model_path=None, device="cpu", on_game=None, should_stop=lambda: False):
    """Bounded worker group, shared GPU inference, durable completed-game recovery."""
    pending = []
    for job in jobs:
        path = Path(job["path"])
        if path.exists():
            _, meta = load_game(path)
            if any(meta.get(key) != job.get(key) for key in ("game_id", "seed", "mode", "model_sha256")):
                raise ValueError(f"Game identity mismatch: {path}")
            if on_game:
                on_game(meta)
        else:
            pending.append(job)
    if not pending or should_stop():
        return
    workers = min(workers, len(pending))
    tasks, results, cancelled = mp.Queue(), mp.Queue(), mp.Event()
    server, requests, pool, stop_server = None, None, None, None
    processes = []
    try:
        if model_path and device == "cuda":
            from hybrid.rl.az_shm_pool import SharedMemoryPool
            from hybrid.rl.az_inference_server import inference_server_process
            requests, stop_server = mp.Queue(), mp.Event()
            pool = SharedMemoryPool(workers, 8)
            server = mp.Process(target=inference_server_process,
                args=(str(model_path), requests, pool, stop_server, 64, 2., device))
            server.start()
        for wid in range(workers):
            process = mp.Process(target=_worker, args=(wid, tasks, results, cancelled, model_path, requests, pool))
            process.start()
            processes.append(process)
        for job in pending:
            tasks.put(job)
        for _ in processes:
            tasks.put(None)
        completed = 0
        while completed < len(pending):
            if should_stop() or time.time() >= pending[0]["deadline"]:
                cancelled.set()
                break
            if server is not None and server.exitcode is not None:
                raise RuntimeError(f"Inference server exited: {server.exitcode}")
            try:
                game_id, meta, error = results.get(timeout=1.)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in processes):
                    raise RuntimeError("Game worker exited unexpectedly")
                if all(p.exitcode is not None for p in processes):
                    raise RuntimeError("Workers exited before returning all games")
                continue
            if error:
                raise RuntimeError(f"Game {game_id}: {error}")
            completed += 1
            if meta and on_game:
                on_game(meta)
    finally:
        cancelled.set()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        if server is not None:
            stop_server.set()
            server.join(timeout=5)
            if server.is_alive():
                server.terminate()
                server.join(timeout=5)
        for q in (tasks, results, requests):
            if q is not None:
                q.cancel_join_thread()
                q.close()
