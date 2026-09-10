"""Paired CPU evaluation of a frozen model, with resumable per-game results."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import time

import numpy as np
import torch

from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig
from hybrid.agents.alphazero_stub import AlphaZeroMiniAgent, MCTSConfig, TorchPolicyValueModel
from hybrid.agents.greedy_agent import GreedyAgent
from hybrid.agents.random_agent import RandomAgent
from hybrid.core.env import HybridChessEnv
from hybrid.core.render import render_board
from hybrid.core.rules import terminal_info, TerminalStatus
from hybrid.core.types import Side
from hybrid.rl.general_model import load_general_model
from hybrid.rl.general_train import require_compute, StopControl
from hybrid.rl.run_store import RunStore, run_lock, write_json, sha256
from hybrid.web_variants import PRESETS, parse_variant

_model = None


def _init_worker(path):
    global _model
    torch.set_num_threads(1)
    _model = TorchPolicyValueModel(load_general_model(path), "cpu")


def play_eval(job):
    env = HybridChessEnv(variant=parse_variant(job["variant"]))
    state = env.reset()
    neural_side = Side.CHESS if job["side"] == "chess" else Side.XIANGQI
    neural = AlphaZeroMiniAgent(_model, MCTSConfig(simulations=100000, dirichlet_eps=0.,
        discount_factor=1., time_limit_seconds=job["seconds"]), seed=job["seed"])
    opponent = {"random": lambda: RandomAgent(seed=job["seed"]), "greedy": GreedyAgent,
                "ab_fast": lambda: AlphaBetaAgent(SearchConfig(depth=1, time_limit_seconds=job["seconds"]))}[job["opponent"]]()
    opening_rng = random.Random(job["seed"])
    times, moves, boards = [], [], [render_board(state.board)] if job["record"] else []
    info = None
    while True:
        if time.time() >= job["deadline"]:
            return None
        legal = env.legal_moves()
        if not legal:
            info = terminal_info(state.board, state.side_to_move, state.repetition, state.ply, 400)
            break
        if state.ply < 4:
            move = opening_rng.choice(legal)
        else:
            agent = neural if state.side_to_move == neural_side else opponent
            started = time.perf_counter()
            move = agent.select_move(state, legal)
            if state.side_to_move == neural_side:
                times.append(time.perf_counter()-started)
        moves.append(f"{'abcdefghi'[move.fx]}{move.fy+1}-{'abcdefghi'[move.tx]}{move.ty+1}" +
            ("="+{"QUEEN":"Q", "ROOK":"R", "BISHOP":"B", "KNIGHT":"N"}[move.promotion.name] if move.promotion else ""))
        state, _, done, info = env.step(move)
        if boards:
            boards.append(render_board(state.board))
        if done:
            break
    score = .5 if info.winner is None else float(info.winner == neural_side)
    result = {k: job[k] for k in ("id", "preset", "variant", "side", "opponent", "seed", "seconds", "model_sha256")}
    result.update(score=score, winner=info.winner.name.lower() if info.winner else None,
        result=info.winner.name.lower()+"_win" if info.winner else "draw", reason=info.reason, plies=state.ply,
        move_seconds=times, moves=moves, states_ascii=boards)
    write_json(job["path"], result)
    return result


def summarize(records, expected, variants=PRESETS):
    groups = []
    for preset in variants:
        for opponent in ("random", "greedy", "ab_fast"):
            rows = [r for r in records if r["preset"] == preset["id"] and r["opponent"] == opponent]
            if not rows:
                continue
            pairs = {}
            for row in rows:
                pairs.setdefault(row["seed"], []).append(row["score"])
            pair_scores = [sum(pair)/2 for pair in pairs.values() if len(pair) == 2]
            score = float(np.mean([r["score"] for r in rows]))
            radius = math.sqrt(math.log(40)/(2*len(pair_scores))) if pair_scores else 1.
            pair_mean = float(np.mean(pair_scores)) if pair_scores else score
            times = [seconds for row in rows for seconds in row["move_seconds"]]
            groups.append({"preset": preset["id"], "opponent": opponent, "games": len(rows),
                "wins": sum(r["score"] == 1 for r in rows), "draws": sum(r["score"] == .5 for r in rows),
                "losses": sum(r["score"] == 0 for r in rows), "score": score,
                "score_95_ci": [max(0., pair_mean-radius), min(1., pair_mean+radius)],
                "complete_pairs": len(pair_scores), "ci_method": "Hoeffding bound over opening-pair means",
                "chess_score": float(np.mean([r["score"] for r in rows if r["side"] == "chess"])) if any(r["side"] == "chess" for r in rows) else None,
                "xiangqi_score": float(np.mean([r["score"] for r in rows if r["side"] == "xiangqi"])) if any(r["side"] == "xiangqi" for r in rows) else None,
                "move_seconds_p50": float(np.median(times)) if times else None,
                "move_seconds_p95": float(np.quantile(times, .95)) if times else None})
    return {"expected_games": expected, "completed_games": len(records), "complete": len(records) == expected,
            "groups": groups, "protocol": "CPU, one thread per agent; four seeded opening plies; armies swapped per opening."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--games", type=int, default=20, help="Even number per preset/opponent")
    parser.add_argument("--seconds", type=float, default=1.)
    parser.add_argument("--wall-seconds", type=float, default=7000)
    parser.add_argument("--variants", help="Optional JSON list of {id, variant} evaluation configurations")
    args = parser.parse_args()
    require_compute()
    mp.set_start_method("spawn", force=True)
    if args.games <= 0 or args.games % 2:
        raise ValueError("Games must be a positive even number")
    if args.workers <= 0 or not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("Workers and per-move seconds must be positive")
    if not math.isfinite(args.wall_seconds) or args.wall_seconds <= 0:
        raise ValueError("Wall seconds must be positive")
    root = Path(args.output)
    stop = StopControl(args.wall_seconds)
    model_hash = sha256(args.model)
    variants = json.loads(Path(args.variants).read_text()) if args.variants else PRESETS
    variants = [{"id": v["id"], "variant": parse_variant(v["variant"]).to_dict()} for v in variants]
    identity = {"model_sha256": model_hash, "games": args.games, "seconds": args.seconds,
                "protocol": 2, "variants": variants}
    with run_lock(root):
        RunStore(root, identity)
        jobs = []
        for preset_index, preset in enumerate(variants):
            for opponent in ("random", "greedy", "ab_fast"):
                for i in range(args.games):
                    game_id = f"{preset['id']}-{opponent}-{i:02d}"
                    jobs.append({"id": game_id, "preset": preset["id"], "variant": preset["variant"],
                        "side": "chess" if i % 2 == 0 else "xiangqi", "opponent": opponent,
                        "seed": 876000+100*preset_index+i//2, "seconds": args.seconds,
                        "model_sha256": model_hash, "deadline": stop.deadline,
                        "record": i < 2, "path": str(root / "games" / (game_id+".json"))})
        records, pending = [], []
        for job in jobs:
            if Path(job["path"]).exists():
                record = json.loads(Path(job["path"]).read_text())
                if record["model_sha256"] != model_hash:
                    raise ValueError("Evaluation game model mismatch")
                records.append(record)
            else:
                pending.append(job)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(args.model,)) as pool:
            futures = [pool.submit(play_eval, job) for job in pending]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    records.append(result)
                    print(json.dumps({"game": result["id"], "score": result["score"], "plies": result["plies"]}), flush=True)
                    write_json(root / "summary.json", summarize(records, len(jobs), variants))
                if stop():
                    for future in futures:
                        future.cancel()
                    break
        write_json(root / "summary.json", summarize(records, len(jobs), variants))


if __name__ == "__main__":
    main()
