# -*- coding: utf-8 -*-
"""Sprint 2.5 Task 3: Sanity regression test.

Runs 3 matchups (20 games each = 10 per side) to verify that the
evaluation fix resolves the draw disease:

1. AB_D2 vs RANDOM   — expected score ≥ 0.95
2. GREEDY vs RANDOM   — expected score ≥ 0.70
3. AB_D4 vs AB_D1     — expected score ≥ 0.80
"""

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_arena import _create_agent, play_arena_game
from hybrid.core.types import Side


MATCHUPS = [
    ("ab_d2",  "random",  0.95, "AB_D2 vs RANDOM"),
    ("greedy", "random",  0.70, "GREEDY vs RANDOM"),
    ("ab_d4",  "ab_d1",   0.80, "AB_D4 vs AB_D1"),
]

GAMES_PER_HALF = 10  # 10 per side = 20 total


def run_matchup(spec_a: str, spec_b: str, label: str) -> dict:
    """Run one matchup: 10 games with A=Chess, 10 with A=Xiangqi."""
    scores_a = []
    all_games = []

    for half_label, a_is_chess in [("A=Chess", True), ("A=Xiangqi", False)]:
        for gi in range(GAMES_PER_HALF):
            seed = gi + (0 if a_is_chess else 5000)
            agent_a = _create_agent(spec_a, simulations=200, seed=seed, use_cpp=True)
            agent_b = _create_agent(spec_b, simulations=200, seed=seed + 500, use_cpp=True)

            if a_is_chess:
                chess_agent, xiangqi_agent = agent_a, agent_b
            else:
                chess_agent, xiangqi_agent = agent_b, agent_a

            t0 = time.time()
            result = play_arena_game(chess_agent, xiangqi_agent, seed=seed, use_cpp=True)
            elapsed = time.time() - t0

            if result["winner_side"] is None:
                score_a = 0.5
            elif (result["winner_side"] == "chess" and a_is_chess) or \
                 (result["winner_side"] == "xiangqi" and not a_is_chess):
                score_a = 1.0
            else:
                score_a = 0.0

            scores_a.append(score_a)
            outcome = "WIN_A" if score_a == 1.0 else ("DRAW" if score_a == 0.5 else "WIN_B")
            game_info = {
                "half": half_label,
                "outcome": outcome,
                "plies": result["plies"],
                "reason": result["reason"],
                "elapsed": elapsed,
            }
            all_games.append(game_info)

            running_avg = sum(scores_a) / len(scores_a)
            print(f"  [{len(scores_a)}/{GAMES_PER_HALF * 2}] {half_label} "
                  f"{outcome} ({result['plies']}ply, {result['reason']}, "
                  f"{elapsed:.1f}s) running={running_avg:.3f}")

    avg_a = sum(scores_a) / len(scores_a)
    return {
        "avg_a": avg_a,
        "games": all_games,
        "total": len(scores_a),
    }


def main():
    print("=" * 70)
    print("Sprint 2.5 — Sanity Regression Test")
    print("=" * 70)

    results = {}
    all_pass = True

    for spec_a, spec_b, threshold, label in MATCHUPS:
        print(f"\n{'─' * 60}")
        print(f"  {label}  (threshold: {threshold:.2f})")
        print(f"{'─' * 60}")

        res = run_matchup(spec_a, spec_b, label)
        results[label] = res

        status = "✓ PASS" if res["avg_a"] >= threshold else "✗ FAIL"
        if res["avg_a"] < threshold:
            all_pass = False
        print(f"\n  Result: {res['avg_a']:.3f}  {status}  (threshold: {threshold:.2f})")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    for spec_a, spec_b, threshold, label in MATCHUPS:
        res = results[label]
        status = "PASS" if res["avg_a"] >= threshold else "FAIL"
        print(f"  {label:25s}  {res['avg_a']:.3f}  (>= {threshold:.2f})  {status}")

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print("=" * 70)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
