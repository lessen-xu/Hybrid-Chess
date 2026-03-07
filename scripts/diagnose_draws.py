# -*- coding: utf-8 -*-
"""Sprint 2.5 Task 1: Diagnose the AB draw disease.

Parses pair log files from the N=100 tournament to confirm that
AB_D4 vs RANDOM games are almost entirely draws due to max-ply limit.
"""

import re
import sys
from pathlib import Path
from collections import Counter

# Regex to parse one game line from pair log
# Example: "  [1/100] i=Chess DRAW (400ply, Max plies reached, 62.7s) running=0.500"
GAME_RE = re.compile(
    r'\[(\d+)/(\d+)\]\s+i=(\w+)\s+(WIN_i|WIN_j|DRAW)\s+'
    r'\((\d+)ply,\s*([^,]+),\s*([\d.]+)s\)\s+running=([\d.]+)'
)


def analyze_pair_log(path: Path) -> dict:
    """Parse a single pair log file and return statistics."""
    header = None
    games = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if header is None and 'games)' in line:
                header = line
                continue
            if line.startswith('DONE:'):
                continue
            m = GAME_RE.search(line)
            if m:
                games.append({
                    'game_num': int(m.group(1)),
                    'total': int(m.group(2)),
                    'side_config': m.group(3),
                    'outcome': m.group(4),
                    'plies': int(m.group(5)),
                    'reason': m.group(6).strip(),
                    'elapsed': float(m.group(7)),
                })

    if not games:
        return {'error': f'No games found in {path}'}

    total = len(games)
    wins_i = sum(1 for g in games if g['outcome'] == 'WIN_i')
    wins_j = sum(1 for g in games if g['outcome'] == 'WIN_j')
    draws = sum(1 for g in games if g['outcome'] == 'DRAW')

    avg_ply = sum(g['plies'] for g in games) / total
    max_ply_games = sum(1 for g in games if g['plies'] >= 400)

    reason_counts = Counter(g['reason'] for g in games)

    return {
        'header': header,
        'total_games': total,
        'wins_i': wins_i,
        'wins_j': wins_j,
        'draws': draws,
        'draw_pct': draws / total * 100,
        'avg_ply': avg_ply,
        'max_ply_games': max_ply_games,
        'max_ply_pct': max_ply_games / total * 100,
        'reason_distribution': dict(reason_counts),
    }


def main():
    base = Path(__file__).resolve().parent.parent

    # Analyze key pair logs from V4 tournament
    pairs_to_check = {
        'RANDOM vs AB_D4': base / 'runs' / 'egta_v4_n100' / 'live' / 'pair_0_5.log',
        'RANDOM vs AB_D2': base / 'runs' / 'egta_v4_n100' / 'live' / 'pair_0_6.log',
    }

    # Also try to find RANDOM vs AB_D1 in V3 if available
    v3_live = base / 'runs' / 'egta_v3_n100' / 'live'
    if v3_live.exists():
        for f in v3_live.glob('pair_*.log'):
            with open(f, 'r') as fh:
                first = fh.readline().strip()
            if 'RANDOM' in first and 'AB_D1' in first:
                pairs_to_check['RANDOM vs AB_D1 (V3)'] = f
            if 'RANDOM' in first and 'AB_D4' in first:
                pairs_to_check['RANDOM vs AB_D4 (V3)'] = f

    print("=" * 70)
    print("Sprint 2.5 — Draw Disease Diagnostic Report")
    print("=" * 70)

    for label, path in pairs_to_check.items():
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"  File: {path}")
        print(f"{'─' * 60}")

        if not path.exists():
            print(f"  ⚠ File not found, skipping.")
            continue

        stats = analyze_pair_log(path)
        if 'error' in stats:
            print(f"  ⚠ {stats['error']}")
            continue

        print(f"  Header:     {stats['header']}")
        print(f"  Total:      {stats['total_games']} games")
        print(f"  Wins (i):   {stats['wins_i']}")
        print(f"  Wins (j):   {stats['wins_j']}")
        print(f"  Draws:      {stats['draws']}  ({stats['draw_pct']:.1f}%)")
        print(f"  Avg ply:    {stats['avg_ply']:.1f}")
        print(f"  Hit 400 ply: {stats['max_ply_games']}  ({stats['max_ply_pct']:.1f}%)")
        print(f"\n  Termination reasons:")
        for reason, count in sorted(stats['reason_distribution'].items(),
                                     key=lambda x: -x[1]):
            pct = count / stats['total_games'] * 100
            print(f"    {reason:40s} {count:4d}  ({pct:.1f}%)")

    print(f"\n{'=' * 70}")
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    print("If avg_ply ≈ 400 and draws ≈ 100%, the draw disease is confirmed:")
    print("AB agents eat all pieces but cannot deliver checkmate.")
    print("The evaluation function has no checkmate incentive.")
    print("=" * 70)


if __name__ == '__main__':
    main()
