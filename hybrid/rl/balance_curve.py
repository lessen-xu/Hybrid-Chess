"""Robust Balance Curve Engine across Agent Families, Search Horizons, and Rule Variants.

Evaluates operational balance A_k(r) = P(Chess wins | k, r) - P(XQ wins | k, r)
across a 2D grid of rule variants and computational search budgets:
  - Variants: none (baseline), golden_palace_draw (golden), pk_xq_queen (heuristic)
  - Agents: Random, Greedy, AlphaBeta(d=1,2,4), Pure MCTS(32,128,512), NN-MCTS(64,256,1024)
  - Metrics: Army advantage, decisiveness, plies distribution, checks, captures, and termination breakdown.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig
from hybrid.agents.alphazero_stub import AlphaZeroMiniAgent, MCTSConfig, TorchPolicyValueModel
from hybrid.agents.greedy_agent import GreedyAgent
from hybrid.agents.random_agent import RandomAgent
from hybrid.agents.rollout_model import RolloutModel
from hybrid.core.env import HybridChessEnv
from hybrid.core.rules import is_in_check, terminal_info, TerminalStatus
from hybrid.core.types import Move, Side, PieceKind
from hybrid.rl.general_model import load_general_model
from hybrid.rl.general_train import StopControl
from hybrid.rl.run_store import RunStore, run_lock, write_json, sha256
from hybrid.web_variants import parse_variant

# Agent Budget Mapping (log10 of approximate search states evaluated per move)
AGENT_BUDGETS: Dict[str, float] = {
    "random": 0.0,
    "greedy": 1.0,
    "ab_d1": 1.4,        # ~25 moves evaluated
    "ab_d2": 2.6,        # ~400 nodes
    "ab_d4": 4.5,        # ~30,000 nodes
    "pure_mcts_32": 1.5,
    "pure_mcts_128": 2.1,
    "pure_mcts_512": 2.7,
    "nn_mcts_64": 1.8,
    "nn_mcts_256": 2.4,
    "nn_mcts_1024": 3.0,
}

_GLOBAL_MODEL = None


def _init_eval_worker(model_path: Optional[str]):
    global _GLOBAL_MODEL
    torch.set_num_threads(1)
    if model_path and Path(model_path).exists():
        _GLOBAL_MODEL = TorchPolicyValueModel(load_general_model(model_path), "cpu")


def create_agent(name: str, seed: int):
    """Instantiate agent by name with deterministic seed."""
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "greedy":
        return GreedyAgent()
    if name == "ab_d1":
        return AlphaBetaAgent(SearchConfig(depth=1))
    if name == "ab_d2":
        return AlphaBetaAgent(SearchConfig(depth=2))
    if name == "ab_d4":
        return AlphaBetaAgent(SearchConfig(depth=4))
    if name.startswith("pure_mcts_"):
        sims = int(name.split("_")[-1])
        model = RolloutModel(seed=seed)
        return AlphaZeroMiniAgent(model, MCTSConfig(simulations=sims, dirichlet_eps=0.0, discount_factor=1.0), seed=seed)
    if name.startswith("nn_mcts_"):
        sims = int(name.split("_")[-1])
        if _GLOBAL_MODEL is None:
            raise RuntimeError(f"NN agent '{name}' requested but neural model was not loaded")
        return AlphaZeroMiniAgent(_GLOBAL_MODEL, MCTSConfig(simulations=sims, dirichlet_eps=0.0, discount_factor=1.0), seed=seed)
    raise ValueError(f"Unknown agent identifier: {name}")


def play_balance_game(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Execute a single instrumented balance match."""
    seed = job["seed"]
    max_plies = job.get("max_plies", 400)
    deadline = job.get("deadline", float("inf"))
    variant_cfg = parse_variant(job["variant"])

    env = HybridChessEnv(variant=variant_cfg, max_plies=max_plies)
    state = env.reset()

    agent_chess = create_agent(job["chess_agent"], seed=seed * 1009 + 1)
    agent_xq = create_agent(job["xiangqi_agent"], seed=seed * 1009 + 2)
    opening_rng = random.Random(seed)

    chess_checks = 0
    xq_checks = 0
    captures = 0
    moves_history = []
    times = []

    info = None
    started = time.perf_counter()

    while True:
        if time.time() >= deadline:
            return None

        legal = env.legal_moves()
        if not legal:
            info = terminal_info(state.board, state.side_to_move, state.repetition, state.ply, max_plies)
            break

        # 4-ply random opening book for paired openings
        if state.ply < 4:
            move = opening_rng.choice(legal)
        else:
            agent = agent_chess if state.side_to_move == Side.CHESS else agent_xq
            t0 = time.perf_counter()
            move = agent.select_move(state, legal)
            times.append(time.perf_counter() - t0)

        # Track captures (destination square occupied before move)
        if state.board.get(move.tx, move.ty) is not None:
            captures += 1

        moves_history.append(
            f"{'abcdefghi'[move.fx]}{move.fy+1}-{'abcdefghi'[move.tx]}{move.ty+1}"
            + ("=" + {"QUEEN": "Q", "ROOK": "R", "BISHOP": "B", "KNIGHT": "N"}[move.promotion.name]
               if move.promotion else "")
        )

        state, _, done, info = env.step(move)

        # Track checks delivered by the side that just moved
        if not done:
            if state.side_to_move == Side.XIANGQI and is_in_check(state.board, Side.XIANGQI):
                chess_checks += 1
            elif state.side_to_move == Side.CHESS and is_in_check(state.board, Side.CHESS):
                xq_checks += 1

        if done:
            break

    elapsed = time.perf_counter() - started
    winner_str = info.winner.name.lower() if info.winner else None
    result_str = (info.winner.name.lower() + "_win") if info.winner else "draw"

    record = {
        "id": job["id"],
        "variant": job["variant"],
        "chess_agent": job["chess_agent"],
        "xiangqi_agent": job["xiangqi_agent"],
        "seed": seed,
        "is_symmetric": job["chess_agent"] == job["xiangqi_agent"],
        "winner": winner_str,
        "result": result_str,
        "reason": info.reason,
        "plies": state.ply,
        "seconds": elapsed,
        "chess_checks": chess_checks,
        "xiangqi_checks": xq_checks,
        "captures": captures,
        "chess_budget": AGENT_BUDGETS.get(job["chess_agent"], 1.0),
        "xiangqi_budget": AGENT_BUDGETS.get(job["xiangqi_agent"], 1.0),
    }

    if job.get("path"):
        write_json(Path(job["path"]), record)

    return record


def summarize_balance_records(records: List[Dict[str, Any]], expected_games: int) -> Dict[str, Any]:
    """Aggregate game results into empirical balance curves and game quality descriptors."""
    variants = sorted(list({r["variant"] for r in records}))
    summary_by_variant: Dict[str, Any] = {}

    for v in variants:
        v_rows = [r for r in records if r["variant"] == v]
        
        # 1. Symmetric match analysis (k vs k)
        sym_rows = [r for r in v_rows if r["is_symmetric"]]
        agents = sorted(list({r["chess_agent"] for r in sym_rows}))
        
        balance_curve = []
        for agent in agents:
            a_rows = [r for r in sym_rows if r["chess_agent"] == agent]
            total = len(a_rows)
            if total == 0:
                continue
            c_wins = sum(r["winner"] == "chess" for r in a_rows)
            x_wins = sum(r["winner"] == "xiangqi" for r in a_rows)
            draws = sum(r["winner"] is None for r in a_rows)
            
            c_score = (c_wins + 0.5 * draws) / total
            x_score = (x_wins + 0.5 * draws) / total
            army_adv = c_score - x_score
            
            # Confidence interval via normal approximation
            se = math.sqrt(c_score * (1 - c_score) / total) if total > 1 else 0.5
            
            balance_curve.append({
                "agent": agent,
                "log_budget": AGENT_BUDGETS.get(agent, 1.0),
                "games": total,
                "chess_wins": c_wins,
                "xiangqi_wins": x_wins,
                "draws": draws,
                "chess_score": round(c_score, 4),
                "xiangqi_score": round(x_score, 4),
                "army_advantage": round(army_adv, 4),
                "army_adv_95_ci": [round(max(-1.0, army_adv - 1.96 * se), 4),
                                  round(min(1.0, army_adv + 1.96 * se), 4)],
            })
            
        balance_curve.sort(key=lambda x: x["log_budget"])
        
        # 2. Game Quality Descriptors
        plies = [r["plies"] for r in v_rows]
        draw_count = sum(r["winner"] is None for r in v_rows)
        total_v = len(v_rows)
        
        termination_counts = {}
        for r in v_rows:
            reason = r.get("reason", "Unknown")
            termination_counts[reason] = termination_counts.get(reason, 0) + 1
            
        quality = {
            "total_games": total_v,
            "decisiveness": round(1.0 - (draw_count / total_v), 4) if total_v else 0.0,
            "draw_rate": round(draw_count / total_v, 4) if total_v else 0.0,
            "plies_median": float(np.median(plies)) if plies else 0.0,
            "plies_mean": round(float(np.mean(plies)), 1) if plies else 0.0,
            "plies_p90": float(np.percentile(plies, 90)) if plies else 0.0,
            "plies_iqr": float(np.percentile(plies, 75) - np.percentile(plies, 25)) if plies else 0.0,
            "avg_checks_chess": round(float(np.mean([r["chess_checks"] for r in v_rows])), 2) if v_rows else 0.0,
            "avg_checks_xiangqi": round(float(np.mean([r["xiangqi_checks"] for r in v_rows])), 2) if v_rows else 0.0,
            "avg_captures": round(float(np.mean([r["captures"] for r in v_rows])), 2) if v_rows else 0.0,
            "termination_distribution": {k: round(v / total_v, 4) for k, v in termination_counts.items()} if total_v else {},
        }
        
        summary_by_variant[v] = {
            "balance_curve": balance_curve,
            "game_quality": quality,
        }

    return {
        "expected_games": expected_games,
        "completed_games": len(records),
        "complete": len(records) == expected_games,
        "summary_by_variant": summary_by_variant,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Directory to store game records and summary")
    parser.add_argument("--variants", default="none,golden_palace_draw,pk_xq_queen",
                        help="Comma-separated variant IDs to evaluate")
    parser.add_argument("--agents", default="random,greedy,ab_d1,ab_d2,pure_mcts_32,pure_mcts_128",
                        help="Comma-separated list of agent IDs to include in the balance curve")
    parser.add_argument("--model", help="Optional path to general AI weights candidate.pt for NN agents")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeded paired games per cell")
    parser.add_argument("--workers", type=int, default=14, help="Parallel worker processes")
    parser.add_argument("--wall-seconds", type=float, default=7200, help="Max execution time in seconds")
    parser.add_argument("--tournament", action="store_true",
                        help="Also evaluate cross-agent tournament matches against reference baselines")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    root = Path(args.output)
    games_dir = root / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    stop = StopControl(args.wall_seconds)

    variant_list = [v.strip() for v in args.variants.split(",") if v.strip()]
    agent_list = [a.strip() for a in args.agents.split(",") if a.strip()]

    # Filter out NN agents if model is not provided
    if not args.model or not Path(args.model).exists():
        nn_agents = [a for a in agent_list if a.startswith("nn_")]
        if nn_agents:
            print(f"[BalanceCurve] No model provided; omitting NN agents: {nn_agents}")
            agent_list = [a for a in agent_list if not a.startswith("nn_")]

    model_hash = sha256(args.model) if (args.model and Path(args.model).exists()) else "none"

    identity = {
        "variants": variant_list,
        "agents": agent_list,
        "seeds": args.seeds,
        "model_sha256": model_hash,
        "tournament": args.tournament,
    }

    with run_lock(root):
        RunStore(root, identity)
        jobs = []
        
        # Build match jobs
        base_seed = 20260913
        for v in variant_list:
            # 1. Symmetric Self-Play (Agent k vs Agent k) across all seeds
            for agent in agent_list:
                for s in range(args.seeds):
                    # In symmetric play, 2 games per seed (opening plies reversed / distinct seeds)
                    for rep in range(2):
                        seed_val = base_seed + 1000 * s + rep
                        game_id = f"sym_{v}_{agent}_{s:02d}_{rep}"
                        jobs.append({
                            "id": game_id,
                            "variant": v,
                            "chess_agent": agent,
                            "xiangqi_agent": agent,
                            "seed": seed_val,
                            "deadline": stop.deadline,
                            "path": str(games_dir / f"{game_id}.json"),
                        })

            # 2. If tournament requested, cross against anchor agents (greedy, ab_d1)
            if args.tournament:
                anchors = [a for a in ("greedy", "ab_d1") if a in agent_list]
                for agent in agent_list:
                    for anchor in anchors:
                        if agent == anchor:
                            continue
                        for s in range(args.seeds):
                            seed_val = base_seed + 50000 + 1000 * s
                            # Paired opening: Agent as Chess vs Anchor as XQ
                            id1 = f"tour_{v}_{agent}_as_chess_vs_{anchor}_{s:02d}"
                            jobs.append({
                                "id": id1,
                                "variant": v,
                                "chess_agent": agent,
                                "xiangqi_agent": anchor,
                                "seed": seed_val,
                                "deadline": stop.deadline,
                                "path": str(games_dir / f"{id1}.json"),
                            })
                            # Paired opening: Anchor as Chess vs Agent as XQ
                            id2 = f"tour_{v}_{anchor}_as_chess_vs_{agent}_{s:02d}"
                            jobs.append({
                                "id": id2,
                                "variant": v,
                                "chess_agent": anchor,
                                "xiangqi_agent": agent,
                                "seed": seed_val,
                                "deadline": stop.deadline,
                                "path": str(games_dir / f"{id2}.json"),
                            })

        # Filter already completed games
        records = []
        pending = []
        for j in jobs:
            path = Path(j["path"])
            if path.exists():
                records.append(json.loads(path.read_text()))
            else:
                pending.append(j)

        print(f"[BalanceCurve] Total matches: {len(jobs)} (Completed: {len(records)}, Pending: {len(pending)})")

        if pending and not stop():
            workers = min(args.workers, len(pending))
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_eval_worker,
                                     initargs=(args.model,)) as pool:
                futures = [pool.submit(play_balance_game, job) for job in pending]
                for idx, future in enumerate(as_completed(futures), 1):
                    res = future.result()
                    if res:
                        records.append(res)
                        if idx % 10 == 0 or idx == len(pending):
                            summary = summarize_balance_records(records, len(jobs))
                            write_json(root / "summary.json", summary)
                            print(json.dumps({
                                "progress": f"{len(records)}/{len(jobs)}",
                                "last_game": res["id"],
                                "winner": res["winner"],
                                "plies": res["plies"],
                            }), flush=True)
                    if stop():
                        for f in futures:
                            f.cancel()
                        break

        summary = summarize_balance_records(records, len(jobs))
        write_json(root / "summary.json", summary)
        print(f"[BalanceCurve] Evaluation complete. Saved summary to {root / 'summary.json'}")


if __name__ == "__main__":
    main()
