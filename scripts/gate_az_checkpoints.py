# -*- coding: utf-8 -*-
"""Sprint 5 Step 3: Gate AZ checkpoints on Tier-A oracle endgames.

Sweeps V4 AZ checkpoints through the Readiness Gate:
  - 8 Tier-A positions × 5 trials = 40 games per checkpoint
  - Records: conversion rate, repetition rate, median win length
  - Outputs pass/fail (�?0% conversion = PASS)
  - Parallel via ProcessPoolExecutor

Usage:
  # Full sweep (all V4 checkpoints)
  python -m scripts.gate_az_checkpoints --outdir paper --workers 8

  # Single checkpoint (smoke test)
  python -m scripts.gate_az_checkpoints \
      --checkpoint runs/az_grand_run_v4/ckpt_iter19.pt \
      --trials 1 --outdir paper
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

from hybrid.core.board import Board
from hybrid.core.types import Piece, Side, PieceKind


# ══════════════════════════════════════════════════════════════
# Load Tier-A positions from oracle JSON
# ══════════════════════════════════════════════════════════════

_KIND_MAP = {
    "KING": PieceKind.KING,
    "QUEEN": PieceKind.QUEEN,
    "ROOK": PieceKind.ROOK,
    "BISHOP": PieceKind.BISHOP,
    "KNIGHT": PieceKind.KNIGHT,
    "PAWN": PieceKind.PAWN,
    "GENERAL": PieceKind.GENERAL,
    "ADVISOR": PieceKind.ADVISOR,
    "ELEPHANT": PieceKind.ELEPHANT,
    "HORSE": PieceKind.HORSE,
    "CHARIOT": PieceKind.CHARIOT,
    "CANNON": PieceKind.CANNON,
    "SOLDIER": PieceKind.SOLDIER,
}

_SIDE_MAP = {
    "CHESS": Side.CHESS,
    "XIANGQI": Side.XIANGQI,
}


def load_oracle(path: str = "paper/data/tier_a_oracle.json") -> list:
    """Load and parse oracle positions from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    positions = []
    for entry in raw:
        # Build board
        b = Board.empty()
        pieces_serializable = []
        for x, y, kind_str, side_str in entry["board"]:
            kind = _KIND_MAP[kind_str]
            side = _SIDE_MAP[side_str]
            b.set(x, y, Piece(kind, side))
            pieces_serializable.append((x, y, kind_str, side_str))

        positions.append({
            "id": entry["id"],
            "name": entry["name"],
            "board": b,
            "board_serializable": pieces_serializable,
            "side": _SIDE_MAP[entry["side_to_move"]],
            "expected": _SIDE_MAP[entry["expected_result"]],
        })
    return positions


# ══════════════════════════════════════════════════════════════
# Worker: play one (checkpoint × position × trial)
# ══════════════════════════════════════════════════════════════

def _gate_worker(args):
    """Worker: play one position with one checkpoint. Returns result dict."""
    import time as _time
    from scripts.diagnose_endgame import play_from_position
    from scripts.eval_arena import _create_agent

    ckpt_path, pos_id, pos_name, board_pieces, side_str, expected_str, trial = args

    # Reconstruct board
    b = Board.empty()
    side = _SIDE_MAP[side_str]
    expected = _SIDE_MAP[expected_str]
    for x, y, kind_str, s_str in board_pieces:
        b.set(x, y, Piece(_KIND_MAP[kind_str], _SIDE_MAP[s_str]))

    agent = _create_agent(ckpt_path, simulations=400, seed=42 + trial, use_cpp=True)
    t0 = _time.time()
    winner, reason, plies, _ = play_from_position(b, side, agent, max_ply=200)
    elapsed = _time.time() - t0

    won = winner == expected
    is_repetition = "repetition" in reason.lower() if reason else False

    return {
        "checkpoint": ckpt_path,
        "pos_id": pos_id,
        "pos_name": pos_name,
        "trial": trial,
        "won": won,
        "plies": plies,
        "reason": reason,
        "is_repetition": is_repetition,
        "elapsed": elapsed,
    }


# ══════════════════════════════════════════════════════════════
# Main gate sweep
# ══════════════════════════════════════════════════════════════

def run_gate_sweep(
    checkpoints: List[str],
    positions: list,
    trials: int = 5,
    workers: int = 1,
    outdir: str = "paper",
) -> List[Dict]:
    """Run gate sweep: all checkpoints × all positions × trials."""

    os.makedirs(outdir, exist_ok=True)

    # Build work items
    work_items = []
    for ckpt in checkpoints:
        for pos in positions:
            for t in range(trials):
                work_items.append((
                    ckpt,
                    pos["id"],
                    pos["name"],
                    pos["board_serializable"],
                    "CHESS" if pos["side"] == Side.CHESS else "XIANGQI",
                    "CHESS" if pos["expected"] == Side.CHESS else "XIANGQI",
                    t,
                ))

    n_total = len(work_items)
    n_workers = min(workers, n_total)
    print(f"\n{'='*70}")
    print(f"  AZ CHECKPOINT GATE SWEEP")
    print(f"  Checkpoints: {len(checkpoints)} | Positions: {len(positions)} | "
          f"Trials: {trials} | Total jobs: {n_total}")
    print(f"  Workers: {n_workers}")
    print(f"{'='*70}\n")

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_gate_worker, item): item for item in work_items}
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results.append(r)
            status = "WIN" if r["won"] else "FAIL"
            ckpt_label = Path(r["checkpoint"]).stem
            print(f"  [{i:3d}/{n_total}] {ckpt_label:16s} {r['pos_name']:30s} "
                  f"t{r['trial']} {status:4s}  plies={r['plies']:3d}  "
                  f"{r['elapsed']:.1f}s", flush=True)

    elapsed_total = time.time() - t0
    print(f"\n  Total time: {elapsed_total:.1f}s\n")

    return results


def aggregate_results(results: List[Dict], checkpoints: List[str]) -> List[Dict]:
    """Aggregate per-checkpoint gate results."""
    summary = []
    for ckpt in checkpoints:
        ckpt_results = [r for r in results if r["checkpoint"] == ckpt]
        n_total = len(ckpt_results)
        n_won = sum(1 for r in ckpt_results if r["won"])
        n_rep = sum(1 for r in ckpt_results if r["is_repetition"])
        plies_won = [r["plies"] for r in ckpt_results if r["won"]]
        median_win_len = sorted(plies_won)[len(plies_won) // 2] if plies_won else None

        conv_rate = n_won / n_total if n_total > 0 else 0.0
        rep_rate = n_rep / n_total if n_total > 0 else 0.0
        passed = conv_rate >= 0.80

        label = Path(ckpt).stem
        summary.append({
            "checkpoint": ckpt,
            "label": label,
            "total_games": n_total,
            "wins": n_won,
            "conversion_rate": round(conv_rate, 4),
            "repetition_rate": round(rep_rate, 4),
            "median_win_length": median_win_len,
            "pass": passed,
        })

    return summary


def save_gate_csv(summary: List[Dict], outdir: str) -> str:
    """Save gate results to CSV."""
    csv_path = os.path.join(outdir, "az_gate_v4.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "checkpoint", "label", "total_games", "wins",
            "conversion_rate", "repetition_rate", "median_win_length", "pass",
        ])
        writer.writeheader()
        writer.writerows(summary)
    return csv_path


def print_gate_summary(summary: List[Dict]):
    """Print formatted gate results."""
    print(f"\n{'='*80}")
    print(f"  AZ GATE RESULTS (V4 extra_cannon)")
    print(f"{'='*80}")
    print(f"  {'Checkpoint':20s} {'Conv':>6s} {'Rep':>6s} {'MedPly':>7s} {'Gate':>6s}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*7} {'-'*6}")

    passed_agents = []
    for s in summary:
        med = f"{s['median_win_length']:>5d}" if s['median_win_length'] is not None else "   --"
        gate = " PASS" if s["pass"] else " FAIL"
        print(f"  {s['label']:20s} {s['conversion_rate']:>5.0%} {s['repetition_rate']:>5.0%} "
              f"{med:>7s} {gate:>6s}")
        if s["pass"]:
            passed_agents.append(s["label"])

    print(f"\n  Gate-pass agents ({len(passed_agents)}): {passed_agents}")
    print(f"{'='*80}\n")

    return passed_agents


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Sprint 5: AZ Checkpoint Gate Sweep")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single checkpoint to test (default: all V4)")
    parser.add_argument("--run-dir", type=str, default="runs/az_grand_run_v4",
                        help="Directory with AZ checkpoints. Default: runs/az_grand_run_v4")
    parser.add_argument("--oracle", type=str, default="paper/data/tier_a_oracle.json",
                        help="Path to oracle JSON. Default: paper/data/tier_a_oracle.json")
    parser.add_argument("--trials", type=int, default=5,
                        help="Trials per position per checkpoint. Default: 5")
    parser.add_argument("--positions", type=int, default=None,
                        help="Limit to first N positions (for smoke testing)")
    parser.add_argument("--outdir", type=str, default="paper",
                        help="Output directory. Default: paper")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers. Default: 1")
    args = parser.parse_args()

    # Determine checkpoints
    if args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        run_dir = Path(args.run_dir)
        checkpoints = sorted(
            [str(p) for p in run_dir.glob("ckpt_iter*.pt")],
            key=lambda p: int(Path(p).stem.split("iter")[1])
        )
        if not checkpoints:
            print(f"  ERROR: No checkpoints found in {run_dir}")
            return

    print(f"  Checkpoints ({len(checkpoints)}):")
    for c in checkpoints:
        print(f"    {c}")

    # Load positions
    positions = load_oracle(args.oracle)
    if args.positions:
        positions = positions[:args.positions]
    print(f"\n  Positions ({len(positions)}):")
    for p in positions:
        print(f"    {p['id']}: {p['name']}")

    # Run gate sweep
    results = run_gate_sweep(
        checkpoints=checkpoints,
        positions=positions,
        trials=args.trials,
        workers=args.workers,
        outdir=args.outdir,
    )

    # Aggregate and save
    summary = aggregate_results(results, checkpoints)
    csv_path = save_gate_csv(summary, args.outdir)
    print(f"  CSV saved: {csv_path}")

    passed_agents = print_gate_summary(summary)

    # Save paper_pool_v2
    pool_path = os.path.join(args.outdir, "paper_pool_v2.json")
    pool_data = {
        "description": "Gate-pass AZ agents for clean EGTA meta-game",
        "gate_threshold": 0.80,
        "agents": [s["checkpoint"] for s in summary if s["pass"]],
        "labels": passed_agents,
        "total_checkpoints_tested": len(checkpoints),
        "pass_count": len(passed_agents),
    }
    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump(pool_data, f, indent=2)
    print(f"  Paper pool saved: {pool_path}")


if __name__ == "__main__":
    main()
