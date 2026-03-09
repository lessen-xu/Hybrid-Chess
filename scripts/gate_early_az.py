# -*- coding: utf-8 -*-
"""Phase 3: Gate-test AZ early checkpoints (iter 0/1) from all seeds."""
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
CHECKPOINTS = [
    ("AZ-S0-iter0",   "runs/az_grand_run_v4/ckpt_iter0.pt"),
    ("AZ-S0-iter1",   "runs/az_grand_run_v4/ckpt_iter1.pt"),
    ("AZ-S100-iter0", "runs/az_v4_seed100/ckpt_iter0.pt"),
    ("AZ-S100-iter1", "runs/az_v4_seed100/ckpt_iter1.pt"),
    ("AZ-S200-iter0", "runs/az_v4_seed200/ckpt_iter0.pt"),
    ("AZ-S200-iter1", "runs/az_v4_seed200/ckpt_iter1.pt"),
]
TRIALS = 5
GATE_THRESHOLD = 0.80


def _gate_worker(args):
    ckpt_path, pos_id, board_pieces, side_str, expected_str, trial = args
    _apply_ablation("extra_cannon")

    from scripts.diagnose_endgame import play_from_position
    from scripts.eval_arena import _create_agent

    b = Board.empty()
    side = _SIDE_MAP[side_str]
    expected = _SIDE_MAP[expected_str]
    for x, y, kind_str, s_str in board_pieces:
        b.set(x, y, Piece(_KIND_MAP[kind_str], _SIDE_MAP[s_str]))

    agent = _create_agent(ckpt_path, simulations=400, seed=42 + trial, use_cpp=True)
    winner, reason, plies, _ = play_from_position(b, side, agent, max_ply=200)
    won = winner == expected
    return {"checkpoint": ckpt_path, "pos_id": pos_id, "trial": trial, "won": won, "plies": plies}


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

    print(f"Gate testing {len(CHECKPOINTS)} early AZ checkpoints on {len(positions)} positions x {TRIALS} trials")
    print(f"Threshold: {GATE_THRESHOLD*100:.0f}%\n")

    # Build all work items
    work_items = []
    for label, ckpt_path in CHECKPOINTS:
        for pos in positions:
            for trial in range(TRIALS):
                work_items.append((
                    ckpt_path, pos["id"], pos["pieces"],
                    pos["side"], pos["expected"], trial,
                ))

    print(f"Total jobs: {len(work_items)}, running with 8 workers...\n")

    all_results = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        all_results = list(pool.map(_gate_worker, work_items))

    # Aggregate per checkpoint
    summary = []
    for label, ckpt_path in CHECKPOINTS:
        ckpt_results = [r for r in all_results if r["checkpoint"] == ckpt_path]
        wins = sum(1 for r in ckpt_results if r["won"])
        total = len(ckpt_results)
        rate = wins / total if total > 0 else 0
        passed = rate >= GATE_THRESHOLD
        gate_str = "PASS" if passed else "FAIL"
        print(f"  {label:>16s}: {rate*100:5.1f}% ({wins}/{total}) [{gate_str}]")
        summary.append({
            "label": label, "checkpoint": ckpt_path,
            "total": total, "wins": wins,
            "conversion_rate": round(rate, 4), "pass": passed,
        })

    # Save
    out_path = os.path.join("paper", "data", "gate_early_az.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
