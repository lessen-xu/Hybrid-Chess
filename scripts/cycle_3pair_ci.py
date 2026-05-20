"""Tighten the 3-cycle CI by running 500 games per pair on the three
variants involved in the cycle observed in cross_variant_tournament:
PK, xqQueen, PK_xqQueen.

Reuses `_play_one` and `run_tournament` from cross_variant_tournament
to keep all per-game logic byte-identical with the main tournament. Same
seed=42 means the first 50 games/half are exactly the same as the main
tournament's; the remaining 200 games/half are fresh draws.

Usage::

    python -m scripts.cycle_3pair_ci --games 250 --workers 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Patch VARIANTS in the parent script to only the 3 cycle agents, then
# delegate to run_tournament so the combinations(...) call produces
# exactly the 3 cycle pairs.
from scripts import cross_variant_tournament as _tour

CYCLE_VARIANTS = {
    "PK":         "runs/rq4_az_palace_knight/best_model.pt",
    "xqQueen":    "runs/rq4_az_xqqueen_only/best_model.pt",
    "PK_xqQueen": "runs/rq4_az_pk_xqqueen/best_model.pt",
}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=250,
                   help="games per half (total per pair = 2*games; default 250 = 500/pair)")
    p.add_argument("--sims", type=int, default=50, help="MCTS simulations")
    p.add_argument("--workers", type=int, default=12, help="parallel workers")
    p.add_argument("--seed", type=int, default=42,
                   help="same as main tournament: first 50/half will reproduce exactly")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--outdir", type=str,
                   default="runs/cycle_3pair_ci")
    args = p.parse_args()

    _tour.VARIANTS = CYCLE_VARIANTS
    _tour.run_tournament(
        games_per_half=args.games,
        sims=args.sims,
        workers=args.workers,
        seed=args.seed,
        outdir=Path(args.outdir),
        temperature=args.temperature,
    )
