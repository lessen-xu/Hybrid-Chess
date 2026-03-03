# -*- coding: utf-8 -*-
"""Run a batch of games and report win rates for Random / AlphaBeta / TD agents."""

from __future__ import annotations
import argparse
import json
import math

from tqdm import trange

from hybrid.core.env import HybridChessEnv
from hybrid.core.types import Side
from hybrid.agents.random_agent import RandomAgent
from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig
from hybrid.agents.td_agent import TDAgent, TDSearchConfig
from hybrid.rl.td_learning import LinearValueFunction
from hybrid.rl.features import feature_dim


ABLATION_PRESETS = {
    "none":            {},
    "no_queen":        {"ABLATION_NO_QUEEN": True},
    "no_queen_promo":  {"ABLATION_NO_QUEEN_PROMOTION": True},
    "extra_cannon":    {"ABLATION_EXTRA_CANNON": True},
    "remove_pawn":     {"ABLATION_REMOVE_EXTRA_PAWN": True},
}


def apply_ablation(name: str):
    """Set config flags for a named ablation preset."""
    import hybrid.core.config as cfg
    # Reset all ablation flags first
    cfg.ABLATION_NO_QUEEN = False
    cfg.ABLATION_NO_QUEEN_PROMOTION = False
    cfg.ABLATION_EXTRA_CANNON = False
    cfg.ABLATION_REMOVE_EXTRA_PAWN = False
    preset = ABLATION_PRESETS.get(name)
    if preset is None:
        raise ValueError(f"Unknown ablation: {name!r}. Choose from {list(ABLATION_PRESETS)}")
    for key, val in preset.items():
        setattr(cfg, key, val)


def build_agent(name: str, depth: int = 3, td_weights: str = ""):
    name = name.lower()
    if name == "random":
        return RandomAgent(seed=0)
    if name == "alphabeta":
        return AlphaBetaAgent(SearchConfig(depth=depth))
    if name == "td":
        if td_weights:
            with open(td_weights, "r", encoding="utf-8") as f:
                vf = LinearValueFunction.from_dict(json.load(f))
        else:
            vf = LinearValueFunction(feature_dim())
        return TDAgent(vf, TDSearchConfig(depth=depth))
    raise ValueError(f"unknown agent: {name}")


def wilson_ci(wins: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion (95% CI by default)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--p1", type=str, default="alphabeta", help="agent for CHESS side")
    parser.add_argument("--p2", type=str, default="random", help="agent for XIANGQI side")
    parser.add_argument("--depth", type=int, default=3, help="search depth for alphabeta/td agents")
    parser.add_argument("--ablation", type=str, default="none",
                        choices=list(ABLATION_PRESETS),
                        help="ablation preset to apply")
    parser.add_argument("--td-weights", type=str, default="",
                        help="path to trained TD value function json (for td agent)")
    args = parser.parse_args()

    apply_ablation(args.ablation)

    a_chess = build_agent(args.p1, depth=args.depth, td_weights=args.td_weights)
    a_xq = build_agent(args.p2, depth=args.depth, td_weights=args.td_weights)

    env = HybridChessEnv()
    stats = {"chess_win": 0, "xiangqi_win": 0, "draw": 0}
    total_plies = []

    for _ in trange(args.games):
        s = env.reset()
        done = False
        while not done:
            legal = env.legal_moves()
            agent = a_chess if s.side_to_move == Side.CHESS else a_xq
            mv = agent.select_move(s, legal)
            s, r, done, info = env.step(mv)
        stats[info.status] = stats.get(info.status, 0) + 1
        total_plies.append(s.ply)

    n = args.games
    cw = stats["chess_win"]
    xw = stats["xiangqi_win"]
    dr = stats["draw"]
    avg_ply = sum(total_plies) / len(total_plies) if total_plies else 0

    cw_rate, cw_lo, cw_hi = wilson_ci(cw, n)
    xw_rate, xw_lo, xw_hi = wilson_ci(xw, n)
    dr_rate, dr_lo, dr_hi = wilson_ci(dr, n)

    ablation_label = args.ablation if args.ablation != "none" else "baseline"
    print()
    print("=" * 60)
    print(f"  Ablation          : {ablation_label}")
    print(f"  CHESS  side agent : {a_chess.name}  (depth={args.depth})")
    print(f"  XIANGQI side agent: {a_xq.name}  (depth={args.depth})")
    print(f"  Games played      : {n}")
    print(f"  Avg plies/game    : {avg_ply:.1f}")
    print("-" * 60)
    print(f"  Chess win   : {cw:>4d} / {n}  = {cw_rate:.1%}  95%CI [{cw_lo:.1%}, {cw_hi:.1%}]")
    print(f"  Xiangqi win : {xw:>4d} / {n}  = {xw_rate:.1%}  95%CI [{xw_lo:.1%}, {xw_hi:.1%}]")
    print(f"  Draw        : {dr:>4d} / {n}  = {dr_rate:.1%}  95%CI [{dr_lo:.1%}, {dr_hi:.1%}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
