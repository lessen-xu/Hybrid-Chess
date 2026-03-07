# -*- coding: utf-8 -*-
"""Fast AZ-only tournament with dual inference servers.

For pairs of AZ checkpoints, launches 2 inference servers (one per model)
and plays N games in PARALLEL via mp.Process workers using RemotePolicyValueModel.

Usage:
  python -m scripts.egta_tournament_fast \
      --agents "runs/az_grand_run_v4/ckpt_iter2.pt,runs/az_grand_run_v4/ckpt_iter9.pt,runs/az_grand_run_v4/ckpt_iter19.pt" \
      --ablation extra_cannon --games-per-pair 100 --simulations 400 \
      --outdir runs/egta_v4_gated --workers 8
"""

from __future__ import annotations

import scripts._fix_encoding  # noqa: F401
import argparse
import csv
import json
import os
import time
import multiprocessing as mp
from itertools import combinations
from pathlib import Path
from typing import List

import numpy as np

from scripts.eval_arena import _agent_label, play_arena_game, BASELINE_AGENTS
from scripts.egta_tournament import (
    compute_nash_equilibrium, plot_payoff_heatmap,
    _write_json, _load_progress,
)


# ====================================================================
# Game worker process — plays a batch of games using inference servers
# ====================================================================

def _game_worker(
    worker_id: int,
    game_assignments: list,  # list of (game_idx, chess_ckpt, xiangqi_ckpt, is_first_half, game_seed)
    rq_a,           # Request queue for server A
    rq_b,           # Request queue for server B
    pool_a,         # SharedMemoryPool for server A
    pool_b,         # SharedMemoryPool for server B
    ckpt_a: str,    # Which checkpoint server A serves
    ckpt_b: str,    # Which checkpoint server B serves
    simulations: int,
    use_cpp: bool,
    ablation: str,
    result_queue,   # mp.Queue to send results back
):
    """Worker process: plays assigned games using inference servers."""
    from hybrid.rl.az_runner import _apply_ablation
    _apply_ablation(ablation)

    from hybrid.rl.az_inference_server import InferenceClient
    from hybrid.agents.az_remote_model import RemotePolicyValueModel
    from hybrid.agents.alphazero_stub import AlphaZeroMiniAgent, MCTSConfig
    import time as _time

    # Create clients connected to both servers (using this worker's ID slot)
    client_a = InferenceClient(worker_id, rq_a, pool_a)
    client_b = InferenceClient(worker_id, rq_b, pool_b)
    model_a = RemotePolicyValueModel(client_a)
    model_b = RemotePolicyValueModel(client_b)

    for game_idx, chess_ckpt, xiangqi_ckpt, is_first_half, game_seed in game_assignments:
        # Pick the right model for each role
        chess_model = model_a if chess_ckpt == ckpt_a else model_b
        xiangqi_model = model_a if xiangqi_ckpt == ckpt_a else model_b

        agent_chess = AlphaZeroMiniAgent(
            model=chess_model,
            cfg=MCTSConfig(simulations=simulations, dirichlet_eps=0.0),
            seed=game_seed,
            use_cpp=use_cpp,
        )
        agent_xiangqi = AlphaZeroMiniAgent(
            model=xiangqi_model,
            cfg=MCTSConfig(simulations=simulations, dirichlet_eps=0.0),
            seed=game_seed + 500,
            use_cpp=use_cpp,
        )

        t0 = _time.time()
        result = play_arena_game(agent_chess, agent_xiangqi,
                                 seed=game_seed, use_cpp=use_cpp)
        elapsed = _time.time() - t0

        if result["winner_side"] is None:
            chess_score = 0.5
        elif result["winner_side"] == "chess":
            chess_score = 1.0
        else:
            chess_score = 0.0

        result_queue.put({
            "game_idx": game_idx,
            "is_first_half": is_first_half,
            "chess_score": chess_score,
            "plies": result["plies"],
            "reason": result["reason"],
            "elapsed": elapsed,
            "worker_id": worker_id,
        })


def _run_pair_fast(
    spec_i: str, spec_j: str,
    label_i: str, label_j: str,
    i: int, j: int,
    games_per_pair: int,
    simulations: int,
    use_cpp: bool,
    seed: int,
    pair_idx: int,
    outdir: str,
    ablation: str,
    n_workers: int,
) -> dict:
    """Run all games for one pair using dual inference servers + parallel workers."""
    from hybrid.rl.az_inference_server import inference_server_process
    from hybrid.rl.az_shm_pool import SharedMemoryPool

    live_dir = os.path.join(outdir, "live")
    os.makedirs(live_dir, exist_ok=True)
    pair_log_path = os.path.join(live_dir, f"pair_{i}_{j}.log")

    games_per_half = games_per_pair // 2

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Launch two inference servers — one per checkpoint
    pool_a = SharedMemoryPool(max_workers=n_workers)
    pool_b = SharedMemoryPool(max_workers=n_workers)
    rq_a = mp.Queue()
    rq_b = mp.Queue()
    stop_a = mp.Event()
    stop_b = mp.Event()
    result_queue = mp.Queue()

    srv_a = mp.Process(
        target=inference_server_process,
        args=(spec_i, rq_a, pool_a, stop_a, 64, 5.0, "cuda"),
        daemon=True,
    )
    srv_b = mp.Process(
        target=inference_server_process,
        args=(spec_j, rq_b, pool_b, stop_b, 64, 5.0, "cuda"),
        daemon=True,
    )
    srv_a.start()
    srv_b.start()
    print(f"    Servers started (PIDs {srv_a.pid}, {srv_b.pid})")

    # Wait for servers to load models
    time.sleep(2.0)

    # Build game assignments and distribute across workers
    all_games = []
    # Half 1: i=Chess, j=Xiangqi
    for gi in range(games_per_half):
        game_seed = seed + pair_idx * 10000 + gi
        all_games.append((gi, spec_i, spec_j, True, game_seed))
    # Half 2: j=Chess, i=Xiangqi
    for gi in range(games_per_half):
        game_seed = seed + pair_idx * 10000 + gi + 5000
        all_games.append((games_per_half + gi, spec_j, spec_i, False, game_seed))

    # Distribute games across workers (round-robin)
    worker_games = [[] for _ in range(n_workers)]
    for idx, game in enumerate(all_games):
        worker_games[idx % n_workers].append(game)

    # Launch worker processes
    workers = []
    for wid in range(n_workers):
        if not worker_games[wid]:
            continue
        p = mp.Process(
            target=_game_worker,
            args=(
                wid, worker_games[wid],
                rq_a, rq_b, pool_a, pool_b,
                spec_i, spec_j,
                simulations, use_cpp, ablation, result_queue,
            ),
        )
        p.start()
        workers.append(p)

    print(f"    {len(workers)} game workers launched for {len(all_games)} games")

    # Collect results
    pair_log = open(pair_log_path, "w", encoding="utf-8")
    pair_log.write(f"{label_i} vs {label_j} ({games_per_pair} games, FAST parallel server)\n")
    pair_log.flush()

    results = []
    for count in range(len(all_games)):
        r = result_queue.get()
        results.append(r)

        half = "i=Chess" if r["is_first_half"] else "i=Xiangqi"
        cs = r["chess_score"]
        if r["is_first_half"]:
            score_i = cs
        else:
            score_i = 1.0 - cs
        outcome = "WIN_i" if score_i == 1.0 else ("DRAW" if score_i == 0.5 else "WIN_j")

        pair_log.write(
            f"  [{count+1}/{len(all_games)}] {half} "
            f"{outcome} ({r['plies']}ply, {r['reason']}, "
            f"{r['elapsed']:.1f}s) w{r['worker_id']}\n")
        pair_log.flush()

        if (count + 1) % 10 == 0 or count == 0:
            print(f"      [{count+1}/{len(all_games)}] last: {outcome} "
                  f"({r['plies']}ply, {r['elapsed']:.1f}s)", flush=True)

    # Wait for workers to finish
    for p in workers:
        p.join(timeout=10.0)

    # Shutdown servers
    stop_a.set()
    stop_b.set()
    srv_a.join(timeout=5.0)
    srv_b.join(timeout=5.0)
    if srv_a.is_alive():
        srv_a.terminate()
    if srv_b.is_alive():
        srv_b.terminate()

    pair_log.close()

    # Aggregate results
    scores_i_as_chess = []
    scores_j_as_chess = []
    pair_scores_ij = []
    pair_scores_ji = []

    for r in sorted(results, key=lambda x: x["game_idx"]):
        cs = r["chess_score"]
        if r["is_first_half"]:
            scores_i_as_chess.append(cs)
            score_i = cs
        else:
            scores_j_as_chess.append(cs)
            score_i = 1.0 - cs
        pair_scores_ij.append(score_i)
        pair_scores_ji.append(1.0 - score_i)

    avg_ij = sum(pair_scores_ij) / max(len(pair_scores_ij), 1)
    avg_ji = sum(pair_scores_ji) / max(len(pair_scores_ji), 1)

    return {
        "i": i, "j": j,
        "label_i": label_i, "label_j": label_j,
        "scores_ij": pair_scores_ij,
        "scores_ji": pair_scores_ji,
        "scores_i_as_chess": scores_i_as_chess,
        "scores_j_as_chess": scores_j_as_chess,
        "avg_ij": avg_ij,
        "avg_ji": avg_ji,
    }


# ====================================================================
# Tournament orchestration
# ====================================================================

def run_tournament_fast(
    agent_specs: List[str],
    games_per_pair: int,
    simulations: int,
    ablation: str,
    use_cpp: bool,
    seed: int,
    outdir: str,
    universe_label: str = "",
    workers: int = 8,
) -> dict:
    """Run tournament with inference server acceleration for AZ agents."""
    from hybrid.rl.az_runner import _apply_ablation
    _apply_ablation(ablation)

    os.makedirs(outdir, exist_ok=True)
    progress_path = os.path.join(outdir, "tournament_progress.json")

    n = len(agent_specs)
    labels = [_agent_label(s) for s in agent_specs]

    existing = _load_progress(progress_path)
    completed_pairs: dict = {}
    if existing and existing.get("agent_specs") == agent_specs:
        completed_pairs = existing.get("completed_pairs", {})
        print(f"  Resuming: {len(completed_pairs)} pairs already done.")

    scores = [[[] for _ in range(n)] for _ in range(n)]
    for key, data in completed_pairs.items():
        i_idx, j_idx = map(int, key.split(","))
        scores[i_idx][j_idx] = data["scores_ij"]
        scores[j_idx][i_idx] = data["scores_ji"]

    all_pairs = list(combinations(range(n), 2))
    total_pairs = len(all_pairs)
    pending_pairs = [(i, j) for i, j in all_pairs if f"{i},{j}" not in completed_pairs]

    tag = universe_label or ablation
    print(f"\n{'='*60}")
    print(f"  EGTA Tournament (FAST): {tag}")
    print(f"  Agents: {labels}")
    print(f"  Pairs: {total_pairs} ({len(pending_pairs)} pending) | "
          f"Games/pair: {games_per_pair} | Sims: {simulations}")
    print(f"  Ablation: {ablation} | Workers: {workers} | Server: DUAL")
    print(f"{'='*60}\n")

    progress = {
        "status": "running",
        "universe": universe_label,
        "agent_specs": agent_specs,
        "labels": labels,
        "games_per_pair": games_per_pair,
        "simulations": simulations,
        "ablation": ablation,
        "total_pairs": total_pairs,
        "completed_pairs": completed_pairs,
        "workers": workers,
        "mode": "fast_dual_server",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    log_path = os.path.join(outdir, "tournament.log")
    log_file = open(log_path, "a", encoding="utf-8")

    for pi, (i, j) in enumerate(pending_pairs, 1):
        pair_idx = all_pairs.index((i, j)) + 1
        t0 = time.time()
        print(f"\n  [{pi}/{len(pending_pairs)}] {labels[i]} vs {labels[j]}...", flush=True)

        pair_result = _run_pair_fast(
            spec_i=agent_specs[i], spec_j=agent_specs[j],
            label_i=labels[i], label_j=labels[j],
            i=i, j=j,
            games_per_pair=games_per_pair,
            simulations=simulations,
            use_cpp=use_cpp,
            seed=seed,
            pair_idx=pair_idx,
            outdir=outdir,
            ablation=ablation,
            n_workers=workers,
        )
        pair_elapsed = time.time() - t0

        pair_key = f"{i},{j}"
        scores[i][j] = pair_result["scores_ij"]
        scores[j][i] = pair_result["scores_ji"]
        completed_pairs[pair_key] = pair_result

        progress["completed_pairs"] = completed_pairs
        progress["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write_json(progress_path, progress)

        line = (f"  [{len(completed_pairs)}/{total_pairs}] "
                f"{pair_result['label_i']} vs {pair_result['label_j']}: "
                f"{pair_result['avg_ij']:.3f} / {pair_result['avg_ji']:.3f}"
                f"  ({len(pair_result['scores_ij'])} games, {pair_elapsed:.0f}s)")
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log_file.close()

    # Build matrices
    matrix_sym = np.full((n, n), 0.5)
    matrix_chess = np.full((n, n), 0.5)

    for ii in range(n):
        for jj in range(n):
            if ii != jj and scores[ii][jj]:
                matrix_sym[ii, jj] = sum(scores[ii][jj]) / len(scores[ii][jj])

    for ii in range(n):
        for jj in range(n):
            if ii != jj:
                key = f"{min(ii,jj)},{max(ii,jj)}"
                pd = completed_pairs.get(key)
                if pd:
                    pi_idx, pj_idx = pd["i"], pd["j"]
                    ic = pd.get("scores_i_as_chess", [])
                    jc = pd.get("scores_j_as_chess", [])
                    if ii == pi_idx and jj == pj_idx and ic:
                        matrix_chess[ii, jj] = sum(ic) / len(ic)
                    elif ii == pj_idx and jj == pi_idx and jc:
                        matrix_chess[ii, jj] = sum(jc) / len(jc)
                    elif not ic and not jc:
                        matrix_chess[ii, jj] = matrix_sym[ii, jj]

    # Save CSVs
    csv_sym = os.path.join(outdir, "payoff_matrix.csv")
    with open(csv_sym, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for ii in range(n):
            w.writerow([labels[ii]] + [f"{matrix_sym[ii,jj]:.4f}" for jj in range(n)])

    csv_role = os.path.join(outdir, "payoff_matrix_role_separated.csv")
    with open(csv_role, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Row=Chess / Col=Xiangqi"] + [f"{l}@Xiangqi" for l in labels])
        for ii in range(n):
            w.writerow([f"{labels[ii]}@Chess"] + [f"{matrix_chess[ii,jj]:.4f}" for jj in range(n)])

    # Heatmaps
    plot_payoff_heatmap(matrix_sym, labels,
                        os.path.join(outdir, "payoff_heatmap.png"),
                        title=f"EGTA Payoff (Symmetric) - {tag}")
    plot_payoff_heatmap(matrix_chess, labels,
                        os.path.join(outdir, "payoff_heatmap_role_separated.png"),
                        title=f"EGTA Payoff (Chess vs Xiangqi) - {tag}")

    # Nash
    nash_dist, gv = compute_nash_equilibrium(matrix_sym)
    nr = {}
    if nash_dist is not None:
        support = [labels[ii] for ii in range(n) if nash_dist[ii] > 1e-4]
        nr = {
            "universe": tag,
            "distribution": {labels[ii]: round(float(nash_dist[ii]), 4) for ii in range(n)},
            "support": support, "support_size": len(support),
            "game_value": round(float(gv), 6),
            "interpretation": ("TRANSITIVE" if len(support) <= 1 else "NON-TRANSITIVE"),
        }
        print(f"\n  Nash: support={support}, value={gv:.4f}")
    _write_json(os.path.join(outdir, "nash_equilibrium.json"), nr)

    progress["status"] = "done"
    progress["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(progress_path, progress)

    print(f"\n  Payoff matrix: {csv_sym}")
    print(f"  Role-separated: {csv_role}")

    return {"universe": tag, "matrix": matrix_sym.tolist(),
            "matrix_role_separated": matrix_chess.tolist(),
            "labels": labels, "nash": nr}


def main():
    parser = argparse.ArgumentParser(
        description="Fast EGTA Tournament with inference server acceleration"
    )
    parser.add_argument("--agents", type=str, required=True)
    parser.add_argument("--ablation", type=str, default="extra_cannon",
                        choices=["none", "extra_cannon", "no_queen"])
    parser.add_argument("--games-per-pair", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="runs/egta_fast")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    run_tournament_fast(
        agent_specs=[s.strip() for s in args.agents.split(",")],
        games_per_pair=args.games_per_pair,
        simulations=args.simulations,
        ablation=args.ablation,
        use_cpp=True,
        seed=args.seed,
        outdir=args.outdir,
        universe_label="V4_extra_cannon_gated",
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
