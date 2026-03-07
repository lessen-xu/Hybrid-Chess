# -*- coding: utf-8 -*-
"""Sprint 4 Step 2: Endgame diagnostic — AB_D4 vs AZ-best on Tier A positions.

Determines whether 0% conversion is AB-specific or environment-wide.
Also supports --depth-sweep for Step 3 profiling.

Usage:
  python -m scripts.diagnose_endgame                    # AB_D4 vs AZ comparison
  python -m scripts.diagnose_endgame --depth-sweep      # D4/D6/D8/D10 sweep
  python -m scripts.diagnose_endgame --agent ab_d4      # Single agent
"""

from __future__ import annotations

import argparse
import time
from typing import List, Tuple, Optional

from hybrid.core.board import Board
from hybrid.core.types import Piece, Side, PieceKind, Move
from hybrid.core.env import HybridChessEnv, GameState
from hybrid.core.rules import generate_legal_moves, board_hash

from scripts.eval_arena import _create_agent, play_arena_game


# ══════════════════════════════════════════════════════════════
# Tier A positions — canonical, provably won endings
# ══════════════════════════════════════════════════════════════

def _board(*pieces):
    b = Board.empty()
    for x, y, kind, side in pieces:
        b.set(x, y, Piece(kind, side))
    return b


TIER_A = [
    # Chess-side advantage
    {
        "name": "KQ vs General",
        "board": _board(
            (4, 0, PieceKind.KING, Side.CHESS),
            (2, 5, PieceKind.QUEEN, Side.CHESS),
            (4, 9, PieceKind.GENERAL, Side.XIANGQI),
        ),
        "side": Side.CHESS,
        "expected": Side.CHESS,
    },
    {
        "name": "KR vs General",
        "board": _board(
            (4, 0, PieceKind.KING, Side.CHESS),
            (0, 4, PieceKind.ROOK, Side.CHESS),
            (4, 9, PieceKind.GENERAL, Side.XIANGQI),
        ),
        "side": Side.CHESS,
        "expected": Side.CHESS,
    },
    {
        "name": "KRR vs General",
        "board": _board(
            (4, 0, PieceKind.KING, Side.CHESS),
            (0, 4, PieceKind.ROOK, Side.CHESS),
            (8, 4, PieceKind.ROOK, Side.CHESS),
            (4, 9, PieceKind.GENERAL, Side.XIANGQI),
        ),
        "side": Side.CHESS,
        "expected": Side.CHESS,
    },
    # Xiangqi-side advantage
    {
        "name": "General+2Chariots vs King",
        "board": _board(
            (4, 0, PieceKind.KING, Side.CHESS),
            (4, 9, PieceKind.GENERAL, Side.XIANGQI),
            (0, 5, PieceKind.CHARIOT, Side.XIANGQI),
            (8, 5, PieceKind.CHARIOT, Side.XIANGQI),
        ),
        "side": Side.XIANGQI,
        "expected": Side.XIANGQI,
    },
    {
        "name": "General+Chariot+Cannon vs King",
        "board": _board(
            (4, 0, PieceKind.KING, Side.CHESS),
            (4, 9, PieceKind.GENERAL, Side.XIANGQI),
            (0, 4, PieceKind.CHARIOT, Side.XIANGQI),
            (4, 6, PieceKind.CANNON, Side.XIANGQI),
        ),
        "side": Side.XIANGQI,
        "expected": Side.XIANGQI,
    },
]


# ══════════════════════════════════════════════════════════════
# Game runner
# ══════════════════════════════════════════════════════════════

def play_from_position(board, side_to_move, agent, max_ply=200, verbose=False):
    """Play from custom position. Returns (winner, reason, plies, move_log)."""
    env = HybridChessEnv(max_plies=max_ply, use_cpp=True)
    env.reset()
    env.state = GameState(board=board.clone(), side_to_move=side_to_move,
                          ply=0, repetition={})
    env._init_cpp_state(board, side_to_move)

    state = env.state
    move_log = []

    while True:
        legal = env.legal_moves()
        if not legal:
            break
        mv = agent.select_move(state, legal)
        opp_legal_before = len(generate_legal_moves(state.board, state.side_to_move.opponent()))
        state, _, done, info = env.step(mv)
        move_log.append({
            "ply": state.ply,
            "move": f"({mv.fx},{mv.fy})->({mv.tx},{mv.ty})",
            "opp_moves_after": len(generate_legal_moves(state.board, state.side_to_move)) if not done else 0,
        })
        if verbose and state.ply <= 20:
            print(f"    Ply {state.ply}: {move_log[-1]['move']}  opp_moves={move_log[-1]['opp_moves_after']}")
        if done:
            break

    winner = info.winner if hasattr(info, 'winner') else None
    reason = info.reason if hasattr(info, 'reason') else ""
    return winner, reason, state.ply, move_log


# ══════════════════════════════════════════════════════════════
# Step 2: AB_D4 vs AZ-best comparison
# ══════════════════════════════════════════════════════════════

def run_comparison():
    """Run AB_D4 and AZ-best on all Tier A positions."""
    print("=" * 70)
    print("  STEP 2: Endgame Diagnostic — AB_D4 vs AZ-best")
    print("=" * 70)

    agents = {
        "AB_D4": _create_agent("ab_d4", simulations=400, seed=42, use_cpp=True),
        "AZ-best": _create_agent("runs/az_grand_run_v4/ckpt_iter19.pt",
                                  simulations=400, seed=42, use_cpp=True),
    }

    results = {}
    for agent_name, agent in agents.items():
        print(f"\n--- {agent_name} ---")
        results[agent_name] = []
        for pos in TIER_A:
            wins = 0
            draws = 0
            plies_list = []
            for trial in range(5):
                winner, reason, plies, _ = play_from_position(
                    pos["board"], pos["side"], agent, max_ply=200
                )
                if winner == pos["expected"]:
                    wins += 1
                elif winner is None:
                    draws += 1
                plies_list.append(plies)

            conv = wins / 5
            avg_ply = sum(plies_list) / len(plies_list)
            results[agent_name].append({
                "name": pos["name"], "conv": conv, "draws": draws,
                "avg_ply": avg_ply,
            })
            status = "PASS" if conv >= 0.8 else "FAIL"
            print(f"  {pos['name']:35s}: {conv:.0%} ({wins}/5)  "
                  f"avg_ply={avg_ply:.0f}  [{status}]")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  {'Position':35s} {'AB_D4':>10s} {'AZ-best':>10s}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}")
    for i, pos in enumerate(TIER_A):
        ab = results["AB_D4"][i]
        az = results["AZ-best"][i]
        print(f"  {pos['name']:35s} {ab['conv']:>9.0%} {az['conv']:>9.0%}")
    print(f"{'=' * 70}")


# ══════════════════════════════════════════════════════════════
# Step 3: Depth sweep (PARALLEL)
# ══════════════════════════════════════════════════════════════

def _sweep_worker(args):
    """Worker: play one position at one depth. Returns result dict."""
    import time as _time
    from scripts.eval_arena import _CppABAgent

    pos_idx, pos_name, board_pieces, side, expected_side, depth = args

    # Reconstruct board
    b = Board.empty()
    for x, y, kind, s in board_pieces:
        b.set(x, y, Piece(kind, s))

    agent = _CppABAgent(depth=depth)
    t0 = _time.time()
    winner, reason, plies, log = play_from_position(b, side, agent, max_ply=200)
    elapsed = _time.time() - t0
    won = "WIN" if winner == expected_side else ("DRAW" if winner is None else "LOSS")

    return {
        "pos_idx": pos_idx,
        "pos_name": pos_name,
        "depth": depth,
        "result": won,
        "plies": plies,
        "reason": reason,
        "elapsed": elapsed,
    }


def run_depth_sweep():
    """Run AB at D4/D6/D8/D10 on Tier A — ALL in parallel."""
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print("=" * 70, flush=True)
    print("  STEP 3: Depth Sweep — D4 / D6 / D8 / D10 (PARALLEL)", flush=True)
    print("=" * 70, flush=True)

    depths = [4, 6, 8, 10]

    # Build work items: serialize boards for subprocess pickling
    work_items = []
    for pos_idx, pos in enumerate(TIER_A):
        board_pieces = [(x, y, p.kind, p.side) for x, y, p in pos["board"].iter_pieces()]
        for d in depths:
            work_items.append((
                pos_idx, pos["name"], board_pieces, pos["side"], pos["expected"], d
            ))

    n_total = len(work_items)
    n_workers = min(os.cpu_count() or 4, n_total)
    print(f"  {n_total} jobs on {n_workers} workers\n", flush=True)

    # Run in parallel
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_sweep_worker, item): item for item in work_items}
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results.append(r)
            print(f"  [{i:2d}/{n_total}] {r['pos_name']:35s} D{r['depth']:2d}: "
                  f"{r['result']:4s}  plies={r['plies']:3d}  "
                  f"reason={r['reason']:20s}  time={r['elapsed']:.1f}s",
                  flush=True)

    # Summary table grouped by position
    print(f"\n{'=' * 70}")
    print(f"  {'Position':35s} {'D4':>8s} {'D6':>8s} {'D8':>8s} {'D10':>8s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for pos_idx, pos in enumerate(TIER_A):
        row = {}
        for r in results:
            if r["pos_idx"] == pos_idx:
                row[r["depth"]] = r
        cells = []
        for d in depths:
            if d in row:
                r = row[d]
                cells.append(f"{r['result']:>4s}/{r['plies']:>3d}")
            else:
                cells.append("   -   ")
        print(f"  {pos['name']:35s} {'  '.join(cells)}")

    # Classification
    print(f"\n  Classification:")
    for pos_idx, pos in enumerate(TIER_A):
        row = {r["depth"]: r for r in results if r["pos_idx"] == pos_idx}
        d4 = row.get(4, {}).get("result", "?")
        d6 = row.get(6, {}).get("result", "?")
        d8 = row.get(8, {}).get("result", "?")
        d10 = row.get(10, {}).get("result", "?")

        if d4 == "WIN":
            label = "ALREADY_SOLVED"
        elif any(row.get(d, {}).get("result") == "WIN" for d in [6, 8, 10]):
            label = "HORIZON (deeper search solves it)"
        elif all(row.get(d, {}).get("result") == "DRAW" for d in depths):
            label = "OBJECTIVE (eval function is wrong)"
        else:
            label = "MIXED"
        print(f"    {pos['name']:35s} -> {label}")

    print(f"{'=' * 70}", flush=True)


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Sprint 4 Endgame Diagnostic")
    parser.add_argument("--depth-sweep", action="store_true",
                        help="Run D4/D6/D8/D10 sweep (Step 3)")
    parser.add_argument("--agent", type=str, default=None,
                        help="Run single agent only (e.g. ab_d4)")
    args = parser.parse_args()

    if args.depth_sweep:
        run_depth_sweep()
    else:
        run_comparison()


if __name__ == "__main__":
    main()
