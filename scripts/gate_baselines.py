# -*- coding: utf-8 -*-
"""Sprint 6 Step 2: Gate baseline agents on Tier-A oracle.

Runs RANDOM, GREEDY, PURE_MCTS_100, PURE_MCTS_400 through the same
Tier-A gate protocol used for AZ checkpoints.

Usage:
  python -m scripts.gate_baselines --outdir paper --workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scripts.gate_az_checkpoints import load_oracle, _KIND_MAP, _SIDE_MAP

from hybrid.core.board import Board
from hybrid.core.types import Piece, Side


def _baseline_gate_worker(args):
    """Worker: play one position with one baseline agent."""
    import time as _time
    from scripts.diagnose_endgame import play_from_position
    from scripts.eval_arena import _create_agent

    agent_spec, pos_id, pos_name, board_pieces, side_str, expected_str, trial, simulations = args

    b = Board.empty()
    side = _SIDE_MAP[side_str]
    expected = _SIDE_MAP[expected_str]
    for x, y, kind_str, s_str in board_pieces:
        b.set(x, y, Piece(_KIND_MAP[kind_str], _SIDE_MAP[s_str]))

    use_cpp = True
    agent = _create_agent(agent_spec, simulations=simulations, seed=42 + trial, use_cpp=use_cpp)

    t0 = _time.time()
    winner, reason, plies, _ = play_from_position(b, side, agent, max_ply=200)
    elapsed = _time.time() - t0

    won = winner == expected
    # Classify termination
    reason_lower = reason.lower() if reason else ""
    if "checkmate" in reason_lower or "stalemate" in reason_lower:
        term_type = "mate_or_stalemate"
    elif "repetition" in reason_lower:
        term_type = "repetition"
    elif "capture" in reason_lower:
        term_type = "royal_capture"
    elif plies >= 200:
        term_type = "max_ply"
    else:
        term_type = "other"

    return {
        "agent": agent_spec,
        "pos_id": pos_id,
        "pos_name": pos_name,
        "trial": trial,
        "won": won,
        "plies": plies,
        "reason": reason,
        "term_type": term_type,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Gate baseline agents on Tier-A oracle")
    parser.add_argument("--oracle", default="paper/data/tier_a_oracle.json")
    parser.add_argument("--outdir", default="paper")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    baselines = [
        ("random", 0),
        ("greedy", 0),
        ("pure_mcts_100", 100),
        ("pure_mcts_400", 400),
    ]

    positions = load_oracle(args.oracle)
    os.makedirs(args.outdir, exist_ok=True)

    # Build work items
    work_items = []
    for agent_spec, sims in baselines:
        for pos in positions:
            for t in range(args.trials):
                work_items.append((
                    agent_spec,
                    pos["id"], pos["name"],
                    pos["board_serializable"],
                    "CHESS" if pos["side"] == Side.CHESS else "XIANGQI",
                    "CHESS" if pos["expected"] == Side.CHESS else "XIANGQI",
                    t, sims,
                ))

    n = len(work_items)
    print(f"\n  Gate Baselines: {len(baselines)} agents × {len(positions)} positions × {args.trials} trials = {n} games")
    print(f"  Workers: {args.workers}\n")

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_baseline_gate_worker, item): item for item in work_items}
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            results.append(r)
            status = "WIN" if r["won"] else "FAIL"
            print(f"  [{i:3d}/{n}] {r['agent']:16s} {r['pos_name']:30s} t{r['trial']} "
                  f"{status:4s} plies={r['plies']:3d} {r['term_type']:15s} {r['elapsed']:.1f}s",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\n  Total: {elapsed:.1f}s\n")

    # Aggregate per agent
    summary = []
    for agent_spec, _ in baselines:
        agent_results = [r for r in results if r["agent"] == agent_spec]
        n_total = len(agent_results)
        n_won = sum(1 for r in agent_results if r["won"])
        n_rep = sum(1 for r in agent_results if r["term_type"] == "repetition")
        plies_won = [r["plies"] for r in agent_results if r["won"]]
        median_win = sorted(plies_won)[len(plies_won) // 2] if plies_won else None

        term_counts = Counter(r["term_type"] for r in agent_results)
        term_str = "; ".join(f"{k}={v}" for k, v in sorted(term_counts.items()))

        conv = n_won / n_total if n_total else 0.0
        rep = n_rep / n_total if n_total else 0.0
        passed = conv >= 0.80

        summary.append({
            "agent": agent_spec,
            "total_games": n_total,
            "wins": n_won,
            "conversion_rate": round(conv, 4),
            "repetition_rate": round(rep, 4),
            "median_win_ply": median_win,
            "termination_breakdown": term_str,
            "pass": passed,
        })

    # Print table
    print(f"  {'Agent':20s} {'Conv':>6s} {'Rep':>6s} {'MedPly':>7s} {'Gate':>6s}  Termination")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*7} {'-'*6}  {'-'*30}")
    for s in summary:
        med = f"{s['median_win_ply']:>5d}" if s["median_win_ply"] is not None else "   --"
        gate = " PASS" if s["pass"] else " FAIL"
        print(f"  {s['agent']:20s} {s['conversion_rate']:>5.0%} {s['repetition_rate']:>5.0%} "
              f"{med:>7s} {gate:>6s}  {s['termination_breakdown']}")

    # Save CSV
    csv_path = os.path.join(args.outdir, "gate_baselines_v1.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "agent", "total_games", "wins", "conversion_rate",
            "repetition_rate", "median_win_ply", "termination_breakdown", "pass",
        ])
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n  Saved: {csv_path}")


if __name__ == "__main__":
    main()
