# -*- coding: utf-8 -*-
"""N=200 Ultimate Tournament: Gated AZ agents only.

Runs 200 games per pair (100 as Chess + 100 as Xiangqi) for the 8 gated
AZ agents. Logs every game as an atomic CSV record for non-parametric
bootstrap. Designed for background overnight execution.

Usage:
    python -u -m scripts.ultimate_tournament 2>&1 | tee runs/ultimate_n200/console.log
"""
import sys, io, os, csv, time, json, argparse
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from itertools import combinations
from scripts.eval_arena import play_arena_game, _agent_label

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

# 8 gated agents (gate ≥ 80%): all AZ except S100-Mid (iter 9)
GATED_SPECS = [
    "runs/az_grand_run_v4/ckpt_iter2.pt",    # AZ-S0-E
    "runs/az_grand_run_v4/ckpt_iter9.pt",     # AZ-S0-M
    "runs/az_grand_run_v4/ckpt_iter19.pt",    # AZ-S0-L
    "runs/az_v4_seed100/ckpt_iter2.pt",       # AZ-S100-E
    # SKIP: runs/az_v4_seed100/ckpt_iter9.pt  # AZ-S100-M (FAILS GATE @ 75%)
    "runs/az_v4_seed100/ckpt_iter19.pt",      # AZ-S100-L
    "runs/az_v4_seed200/ckpt_iter2.pt",       # AZ-S200-E
    "runs/az_v4_seed200/ckpt_iter9.pt",       # AZ-S200-M
    "runs/az_v4_seed200/ckpt_iter19.pt",      # AZ-S200-L
]

GAMES_PER_PAIR = 200       # 100 as Chess + 100 as Xiangqi
SIMULATIONS = 400          # MCTS simulations per move
ABLATION = "extra_cannon"  # V4 rule variant
SEED = 12345
N_WORKERS = 4              # Reduced from 8 to save CPU/memory for overnight
OUTDIR = "runs/ultimate_n200"

# ═══════════════════════════════════════════════════════════════
# Atomic game CSV logger
# ═══════════════════════════════════════════════════════════════

CSV_HEADER = [
    "game_id", "pair_id", "chess_agent", "xiangqi_agent",
    "winner",  # 1=chess_agent_wins, 0=draw, -1=xiangqi_agent_wins
    "plies", "termination_reason", "elapsed_sec",
]

def init_csv(path):
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_HEADER)

def append_csv(path, row):
    """Append a single game record."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


# ═══════════════════════════════════════════════════════════════
# Sequential pair runner (uses inference servers for speed)
# ═══════════════════════════════════════════════════════════════

def run_pair_with_servers(
    spec_i, spec_j, label_i, label_j,
    pair_id, games_per_pair, simulations, ablation, seed, outdir, n_workers,
    csv_path, progress_callback,
):
    """Run all games for one pair using dual inference servers."""
    import multiprocessing as mp
    from hybrid.rl.az_inference_server import inference_server_process
    from hybrid.rl.az_shm_pool import SharedMemoryPool

    games_per_half = games_per_pair // 2

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

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
    time.sleep(2.0)

    # Build game assignments
    all_games = []
    for gi in range(games_per_half):
        game_seed = seed + pair_id * 10000 + gi
        all_games.append((gi, spec_i, spec_j, True, game_seed))
    for gi in range(games_per_half):
        game_seed = seed + pair_id * 10000 + gi + 5000
        all_games.append((games_per_half + gi, spec_j, spec_i, False, game_seed))

    # Distribute across workers
    from scripts.egta_tournament_fast import _game_worker
    worker_games = [[] for _ in range(n_workers)]
    for idx, game in enumerate(all_games):
        worker_games[idx % n_workers].append(game)

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
                simulations, True, ablation, result_queue,
            ),
        )
        p.start()
        workers.append(p)

    # Collect results and write CSV
    game_records = []
    global_game_id = 0

    for count in range(len(all_games)):
        r = result_queue.get()

        if r["is_first_half"]:
            chess_agent = label_i
            xiangqi_agent = label_j
            cs = r["chess_score"]
        else:
            chess_agent = label_j
            xiangqi_agent = label_i
            cs = r["chess_score"]

        # Encode winner: 1=chess wins, -1=xiangqi wins, 0=draw
        if cs == 1.0:
            winner = 1
        elif cs == 0.0:
            winner = -1
        else:
            winner = 0

        row = [
            f"{pair_id}_{r['game_idx']:03d}",
            pair_id,
            chess_agent,
            xiangqi_agent,
            winner,
            r["plies"],
            r["reason"],
            f"{r['elapsed']:.1f}",
        ]
        append_csv(csv_path, row)
        game_records.append(r)

        if (count + 1) % 20 == 0 or count == 0:
            progress_callback(count + 1, len(all_games), r)

    # Shutdown
    for p in workers:
        p.join(timeout=10.0)
    stop_a.set()
    stop_b.set()
    srv_a.join(timeout=5.0)
    srv_b.join(timeout=5.0)
    if srv_a.is_alive():
        srv_a.terminate()
    if srv_b.is_alive():
        srv_b.terminate()

    return game_records


# ═══════════════════════════════════════════════════════════════
# Tournament orchestration
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    csv_path = os.path.join(OUTDIR, "game_records.csv")
    progress_path = os.path.join(OUTDIR, "tournament_progress.json")
    log_path = os.path.join(OUTDIR, "tournament.log")

    from hybrid.rl.az_runner import _apply_ablation
    _apply_ablation(ABLATION)

    labels = [_agent_label(s) for s in GATED_SPECS]
    n = len(GATED_SPECS)
    all_pairs = list(combinations(range(n), 2))

    # Resume support
    completed_pairs = set()
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            prog = json.load(f)
        if prog.get("agent_specs") == GATED_SPECS:
            completed_pairs = set(prog.get("completed_pair_ids", []))
            print(f"  Resuming: {len(completed_pairs)}/{len(all_pairs)} pairs done.")

    init_csv(csv_path)

    print(f"\n{'='*70}")
    print(f"  N=200 ULTIMATE TOURNAMENT (Gated AZ Only)")
    print(f"  Agents: {labels}")
    print(f"  Pairs: {len(all_pairs)} | Games/pair: {GAMES_PER_PAIR}")
    print(f"  Simulations: {SIMULATIONS} | Workers: {N_WORKERS}")
    print(f"  Output: {OUTDIR}")
    print(f"{'='*70}\n")

    log = open(log_path, "a", encoding="utf-8")
    log.write(f"\n--- Tournament started: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    log.flush()

    t_start = time.time()

    for pi, (i, j) in enumerate(all_pairs):
        pair_id = pi
        pair_key = f"{i},{j}"

        if pair_key in completed_pairs:
            continue

        t0 = time.time()
        msg = f"  [{len(completed_pairs)+1}/{len(all_pairs)}] {labels[i]} vs {labels[j]}..."
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

        def progress_cb(done, total, r):
            outcome = "W" if r["chess_score"] == 1.0 else ("L" if r["chess_score"] == 0.0 else "D")
            elapsed_total = time.time() - t_start
            print(f"      [{done}/{total}] {outcome} ({r['plies']}ply, {r['elapsed']:.1f}s) "
                  f"total={elapsed_total/3600:.1f}h", flush=True)

        run_pair_with_servers(
            spec_i=GATED_SPECS[i], spec_j=GATED_SPECS[j],
            label_i=labels[i], label_j=labels[j],
            pair_id=pair_id,
            games_per_pair=GAMES_PER_PAIR,
            simulations=SIMULATIONS,
            ablation=ABLATION,
            seed=SEED,
            outdir=OUTDIR,
            n_workers=N_WORKERS,
            csv_path=csv_path,
            progress_callback=progress_cb,
        )

        elapsed = time.time() - t0
        completed_pairs.add(pair_key)

        # Save progress
        prog = {
            "status": "running",
            "agent_specs": GATED_SPECS,
            "labels": labels,
            "games_per_pair": GAMES_PER_PAIR,
            "simulations": SIMULATIONS,
            "completed_pair_ids": list(completed_pairs),
            "completed_count": len(completed_pairs),
            "total_pairs": len(all_pairs),
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(prog, f, indent=2)

        result_msg = (f"    Done: {labels[i]} vs {labels[j]} ({elapsed:.0f}s, "
                      f"{len(completed_pairs)}/{len(all_pairs)} pairs complete)")
        print(result_msg, flush=True)
        log.write(result_msg + "\n")
        log.flush()

    total_time = time.time() - t_start
    final_msg = f"\n  TOURNAMENT COMPLETE: {len(all_pairs)} pairs, {total_time/3600:.1f}h total"
    print(final_msg, flush=True)
    log.write(final_msg + "\n")

    # Update progress
    prog["status"] = "done"
    prog["total_time_hours"] = round(total_time / 3600, 2)
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(prog, f, indent=2)

    log.close()
    print(f"\n  Game records: {csv_path}")
    print(f"  Progress: {progress_path}")


if __name__ == "__main__":
    main()
