# -*- coding: utf-8 -*-
"""Phase 4 Action 2: Gate-test deep AB agents (D4/D6) on oracle endgames.

Tests whether deep AB can pass the gate when given enough search depth,
proving the gate doesn't discriminate by algorithmic family.
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from concurrent.futures import ProcessPoolExecutor
from hybrid.rl.az_runner import _apply_ablation
from hybrid.core.types import Side
from hybrid.core.board import Board, Piece, PieceKind

_KIND_MAP = {
    "KING": PieceKind.KING, "QUEEN": PieceKind.QUEEN, "ROOK": PieceKind.ROOK,
    "BISHOP": PieceKind.BISHOP, "KNIGHT": PieceKind.KNIGHT, "PAWN": PieceKind.PAWN,
    "GENERAL": PieceKind.GENERAL, "ADVISOR": PieceKind.ADVISOR,
    "ELEPHANT": PieceKind.ELEPHANT, "HORSE": PieceKind.HORSE,
    "CHARIOT": PieceKind.CHARIOT, "CANNON": PieceKind.CANNON, "SOLDIER": PieceKind.SOLDIER,
}
_SIDE_MAP = {"CHESS": Side.CHESS, "XIANGQI": Side.XIANGQI}

ORACLE_PATH = "paper/data/tier_a_oracle.json"
AGENTS = [
    ("AB_D1", "ab_d1"),
    ("AB_D2", "ab_d2"),
    ("AB_D4", "ab_d4"),
]
TRIALS = 5
GATE_THRESHOLD = 0.80


def _gate_worker(args):
    agent_spec, pos_id, board_pieces, side_str, expected_str, trial = args
    _apply_ablation("extra_cannon")

    from scripts.diagnose_endgame import play_from_position
    from scripts.eval_arena import _create_agent

    b = Board.empty()
    side = _SIDE_MAP[side_str]
    expected = _SIDE_MAP[expected_str]
    for x, y, kind_str, s_str in board_pieces:
        b.set(x, y, Piece(_KIND_MAP[kind_str], _SIDE_MAP[s_str]))

    agent = _create_agent(agent_spec, simulations=400, seed=42 + trial, use_cpp=True)
    
    import time as _time
    t0 = _time.time()
    winner, reason, plies, _ = play_from_position(b, side, agent, max_ply=200)
    elapsed = _time.time() - t0
    
    won = winner == expected
    return {
        "agent": agent_spec, "pos_id": pos_id, "trial": trial,
        "won": won, "plies": plies, "reason": reason, "elapsed": round(elapsed, 1),
    }


def main():
    with open(ORACLE_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    positions = []
    for entry in raw:
        pieces = [(x, y, k, s) for x, y, k, s in entry["board"]]
        positions.append({
            "id": entry["id"], "pieces": pieces,
            "side": entry["side_to_move"], "expected": entry["expected_result"],
        })

    print(f"Gate testing {len(AGENTS)} AB agents on {len(positions)} positions x {TRIALS} trials")
    print(f"Threshold: {GATE_THRESHOLD*100:.0f}%\n")

    work_items = []
    for label, spec in AGENTS:
        for pos in positions:
            for trial in range(TRIALS):
                work_items.append((
                    spec, pos["id"], pos["pieces"],
                    pos["side"], pos["expected"], trial,
                ))

    print(f"Total jobs: {len(work_items)}, running with 4 workers...\n")

    all_results = list(map(_gate_worker, work_items))  # sequential to avoid GPU issues

    # Aggregate
    summary = []
    for label, spec in AGENTS:
        agent_results = [r for r in all_results if r["agent"] == spec]
        wins = sum(1 for r in agent_results if r["won"])
        total = len(agent_results)
        rate = wins / total if total > 0 else 0
        passed = rate >= GATE_THRESHOLD
        avg_time = sum(r["elapsed"] for r in agent_results) / total if total > 0 else 0
        status = "PASS" if passed else "FAIL"
        print(f"  {label:>8s}: {rate*100:5.1f}% ({wins}/{total}) [{status}] avg={avg_time:.1f}s/game")
        
        # Detail by position
        for pos in positions:
            pos_results = [r for r in agent_results if r["pos_id"] == pos["id"]]
            pos_wins = sum(1 for r in pos_results if r["won"])
            pos_total = len(pos_results)
            if pos_total > 0:
                reasons = set(r["reason"] for r in pos_results)
                print(f"    {pos['id']:>20s}: {pos_wins}/{pos_total} ({', '.join(reasons)})")
        
        summary.append({
            "label": label, "agent": spec,
            "total": total, "wins": wins,
            "conversion_rate": round(rate, 4), "pass": passed,
            "avg_time": round(avg_time, 1),
        })

    out_path = os.path.join("paper", "data", "gate_deep_ab.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
