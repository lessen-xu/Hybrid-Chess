"""Cross-variant round-robin tournament across the 9 trained AlphaZero
variants under default Hybrid Chess rules.

Reads best_model.pt from each runs/rq4_az_*/ directory, plays a
round-robin, and writes the payoff matrix + W/D/L counts + Wilson 95%
confidence intervals to runs/cross_variant_tournament/.

  * Per-game seed uses hashlib.sha256 instead of Python's randomized
    hash(), so reruns are byte-identical across processes/sessions.
  * Outputs: wdl_matrix.csv (raw W/D/L counts) and pairwise_ci.csv
    (Wilson score 95% CI per ordered pair).
  * Default --games is 50 (= 100 games/pair = 3600 total games);
    pair with --seed-offset to extend an earlier run.

Usage::

    python -m scripts.cross_variant_tournament \\
        --games 50 --sims 50 --workers 6 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass


# ── Agent pool ──────────────────────────────────────────────
# noQ / xqQueen are the bare single-rule variants (no Queen, replace-with-
# xqQueen). PK is palace+knight-block. _* suffixes are combos.
VARIANTS = {
    "Default":     "runs/rq4_az_default/best_model.pt",
    "noQ":         "runs/rq4_az_noq_only/best_model.pt",
    "xqQueen":     "runs/rq4_az_xqqueen_only/best_model.pt",
    "PK":          "runs/rq4_az_palace_knight/best_model.pt",
    "PK_noPromo":  "runs/rq4_az_pk_nopromo/best_model.pt",
    "PK_xqQueen":  "runs/rq4_az_pk_xqqueen/best_model.pt",
    "noQ_noPromo": "runs/rq4_az_nq_nopromo/best_model.pt",
    "noQ_PK":      "runs/rq4_az_nq_pk/best_model.pt",
    "noQ_ALL":     "runs/rq4_az_nq_allrules/best_model.pt",
}


def stable_seed(base_seed: int, *parts) -> int:
    """Deterministic seed across processes/sessions (replaces Python's
    randomized hash())."""
    s = "|".join(str(p) for p in parts)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return base_seed + int(h[:8], 16) % 100000


def wilson_ci(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion (handles n=0)."""
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


_MODEL_CACHE: dict = {}  # worker-local: maps checkpoint path → TorchPolicyValueModel


def _get_or_load_model(path: str, device: str):
    """Worker-local cache. With ProcessPoolExecutor reusing workers, each
    worker loads each of the 9 checkpoints from disk only once, instead of
    rebuilding the net for every game (saves 7200 → ~9 torch.load calls
    per worker across the full tournament).
    """
    from hybrid.rl.az_runner import build_net_from_checkpoint
    from hybrid.agents.alphazero_stub import TorchPolicyValueModel
    if path not in _MODEL_CACHE:
        net = build_net_from_checkpoint(path, device=device)
        _MODEL_CACHE[path] = TorchPolicyValueModel(net, device=device)
    return _MODEL_CACHE[path]


def _play_one(args: tuple) -> dict:
    """Worker: play one game between two AZ models under default rules.

    Action selection uses temperature-sampled visit counts (not argmax) so
    that games with the same (pair, color) but different seeds genuinely
    diverge — without this, eps=0 + argmax produces identical replicas.
    """
    (name_a, path_a, name_b, path_b, game_idx, a_is_chess,
     sims, seed, temperature) = args

    # Ensure hybrid package is importable in spawned workers
    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if proj not in sys.path:
        sys.path.insert(0, proj)

    import torch
    from hybrid.core.env import HybridChessEnv
    from hybrid.core.types import Side
    from hybrid.core.config import DEFAULT_VARIANT
    from hybrid.agents.alphazero_stub import (
        AlphaZeroMiniAgent, MCTSConfig,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _make_agent(path, s):
        model = _get_or_load_model(path, device)
        return AlphaZeroMiniAgent(
            model=model,
            cfg=MCTSConfig(simulations=sims, dirichlet_eps=0.0),
            seed=s,
            use_cpp=True,
        )

    agent_a = _make_agent(path_a, seed)
    agent_b = _make_agent(path_b, seed + 500)

    env = HybridChessEnv(use_cpp=True, variant=DEFAULT_VARIANT)
    state = env.reset()

    if a_is_chess:
        agents = {Side.CHESS: agent_a, Side.XIANGQI: agent_b}
    else:
        agents = {Side.CHESS: agent_b, Side.XIANGQI: agent_a}

    info = None
    while True:
        legal = env.legal_moves()
        if not legal:
            break
        mv, _, _ = agents[state.side_to_move].select_move_with_pi(
            state, legal, temperature=temperature, add_noise=False,
        )
        state, _, done, info = env.step(mv)
        if done:
            break

    if info is None or info.winner is None:
        outcome = "draw"
        reason = info.reason if info is not None else ""
    elif (info.winner == Side.CHESS and a_is_chess) or \
         (info.winner == Side.XIANGQI and not a_is_chess):
        outcome = "win_a"
        reason = info.reason
    else:
        outcome = "win_b"
        reason = info.reason

    return {
        "name_a": name_a, "name_b": name_b,
        "a_is_chess": a_is_chess,
        "outcome": outcome,
        "reason": reason,
        "plies": state.ply,
        "game_idx": game_idx,
        "seed": seed,
    }


def run_tournament(games_per_half: int, sims: int, workers: int, seed: int,
                   outdir: Path, temperature: float = 0.5,
                   seed_offset: int = 0) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    missing = [(name, p) for name, p in VARIANTS.items() if not Path(p).exists()]
    if missing:
        for name, p in missing:
            print(f"  MISSING: {p}  ({name})", file=sys.stderr)
        raise SystemExit("Missing checkpoint(s); abort.")

    pairs = list(itertools.combinations(VARIANTS.keys(), 2))
    total_games = len(pairs) * 2 * games_per_half
    print(f"Tournament: {len(VARIANTS)} agents, {len(pairs)} pairs, "
          f"{games_per_half} games/half = {total_games} total games")
    print(f"Workers: {workers}, Sims: {sims}, Seed: {seed}, "
          f"Temperature: {temperature}\n")

    tasks = []
    for name_a, name_b in pairs:
        path_a, path_b = VARIANTS[name_a], VARIANTS[name_b]
        for half, a_is_chess in [(0, True), (1, False)]:
            for gi in range(games_per_half):
                # seed_offset shifts the gi index in stable_seed so a follow-up
                # run can extend an earlier tournament without re-playing the
                # same games. Game_idx label is also shifted so the merged
                # records have unique (pair, half, game_idx) triples.
                gi_eff = gi + seed_offset
                game_seed = stable_seed(seed, name_a, name_b, half, gi_eff)
                tasks.append((
                    name_a, path_a, name_b, path_b,
                    gi_eff + half * (games_per_half + seed_offset),
                    a_is_chess, sims, game_seed, temperature,
                ))

    results: list[dict] = []
    t0 = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_play_one, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            completed += 1
            if completed % 50 == 0 or completed == total_games:
                elapsed = time.time() - t0
                eta = elapsed / completed * (total_games - completed)
                print(f"  [{completed}/{total_games}] "
                      f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

    total_time = time.time() - t0
    print(f"\nDone in {total_time:.0f}s ({total_time/60:.1f} min)")

    # ── Build matrices ────────────────────────────────────────
    names = list(VARIANTS.keys())
    n = len(names)
    idx = {name: i for i, name in enumerate(names)}
    wins = [[0]*n for _ in range(n)]
    draws = [[0]*n for _ in range(n)]
    total = [[0]*n for _ in range(n)]
    for r in results:
        i, j = idx[r["name_a"]], idx[r["name_b"]]
        total[i][j] += 1
        total[j][i] += 1
        if r["outcome"] == "win_a":
            wins[i][j] += 1
        elif r["outcome"] == "win_b":
            wins[j][i] += 1
        else:
            draws[i][j] += 1
            draws[j][i] += 1

    score = [[0.5 if i == j else
              ((wins[i][j] + 0.5 * draws[i][j]) / total[i][j]
               if total[i][j] > 0 else 0.0)
              for j in range(n)] for i in range(n)]

    # ── Console report ────────────────────────────────────────
    print(f"\n{'':18s}", end="")
    for name in names:
        print(f" {name:>10s}", end="")
    print(f" {'AVG':>7s}")
    print("-" * (18 + 11*n + 8))
    for i, name in enumerate(names):
        print(f"{name:18s}", end="")
        row_scores = []
        for j in range(n):
            print(f" {score[i][j]:10.3f}", end="")
            if i != j:
                row_scores.append(score[i][j])
        avg = sum(row_scores) / max(len(row_scores), 1)
        print(f" {avg:7.3f}")

    avg_scores = {
        name: sum(score[i][j] for j in range(n) if i != j) / (n - 1)
        for i, name in enumerate(names)
    }
    ranking = sorted(avg_scores.items(), key=lambda x: -x[1])
    print(f"\n{'Rank':>4s}  {'Agent':18s}  {'Avg Score':>9s}")
    print("-" * 35)
    for rank, (name, sc) in enumerate(ranking, 1):
        print(f"{rank:4d}  {name:18s}  {sc:9.3f}")

    # ── Persist ───────────────────────────────────────────────
    with open(outdir / "game_records.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)

    with open(outdir / "payoff_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, name in enumerate(names):
            w.writerow([name] + [f"{score[i][j]:.3f}" for j in range(n)])

    with open(outdir / "wdl_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_agent", "col_agent", "wins_row", "draws", "wins_col", "n"])
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i == j:
                    continue
                w.writerow([a, b, wins[i][j], draws[i][j], wins[j][i], total[i][j]])

    with open(outdir / "pairwise_ci.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_agent", "col_agent", "score_row", "ci_low",
                    "ci_high", "wins_row", "draws", "wins_col", "n"])
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i == j:
                    continue
                W, D, L = wins[i][j], draws[i][j], wins[j][i]
                N = total[i][j]
                # Score as a binomial-like proportion with draws counted as 0.5;
                # CI is computed by treating "score points" / "max points" with
                # n trials. This is an approximation but adequate for a rank
                # ordering sanity check.
                eff_wins = W + 0.5 * D
                s = eff_wins / N if N > 0 else 0.0
                lo, hi = wilson_ci(eff_wins, N)
                w.writerow([a, b, f"{s:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                            W, D, L, N])

    summary = {
        "source": "runs/",
        "n_agents": n,
        "n_pairs": len(pairs),
        "games_per_half": games_per_half,
        "games_per_pair": 2 * games_per_half,
        "total_games": total_games,
        "simulations": sims,
        "seed": seed,
        "total_time_s": round(total_time, 1),
        "temperature": temperature,
        "ranking": [{"agent": name, "avg_score": round(sc, 4)}
                    for name, sc in ranking],
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {outdir}/")
    print(f"  game_records.json  payoff_matrix.csv  wdl_matrix.csv  "
          f"pairwise_ci.csv  summary.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=50,
                   help="games per half (total games per pair = 2 * games)")
    p.add_argument("--sims", type=int, default=50, help="MCTS simulations")
    p.add_argument("--workers", type=int, default=6, help="parallel workers")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.5,
                   help="action-selection temperature; visit counts ^ (1/T). "
                        "T>0 guarantees inter-game divergence even at fixed model+state.")
    p.add_argument("--outdir", type=str,
                   default="runs/cross_variant_tournament")
    p.add_argument("--seed-offset", type=int, default=0,
                   help="Shifts the per-game seed index so a follow-up run "
                        "extends an earlier tournament with disjoint games. "
                        "Use --seed-offset 50 to skip the first 50 gi values "
                        "already played at --games 50.")
    args = p.parse_args()
    run_tournament(args.games, args.sims, args.workers, args.seed,
                   Path(args.outdir), temperature=args.temperature,
                   seed_offset=args.seed_offset)
