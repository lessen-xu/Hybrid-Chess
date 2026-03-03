# -*- coding: utf-8 -*-
"""TD(0) self-play training demo.

Simple training loop: both sides share a TDAgent. Each game records a trajectory,
then applies TD(0) updates based on the terminal outcome.
"""

from __future__ import annotations
import argparse
import json
import random
from tqdm import trange

from hybrid.core.env import HybridChessEnv
from hybrid.core.types import Side
from hybrid.agents.td_agent import TDAgent, new_default_td_value_function, TDSearchConfig
from hybrid.rl.td_learning import TDConfig
from scripts.play_match import ABLATION_PRESETS, apply_ablation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="epsilon-greedy exploration rate during training")
    parser.add_argument("--save", type=str, default="", help="path to save value function json")
    parser.add_argument("--ablation", type=str, default="none",
                        choices=list(ABLATION_PRESETS),
                        help="ablation preset to apply")
    args = parser.parse_args()

    apply_ablation(args.ablation)

    rng = random.Random(42)
    env = HybridChessEnv()
    vf = new_default_td_value_function()
    agent = TDAgent(vf, TDSearchConfig(depth=args.depth))

    td_cfg = TDConfig(alpha=args.alpha)

    stats = {"chess_win": 0, "xiangqi_win": 0, "draw": 0}

    for ep in trange(args.episodes):
        s = env.reset()
        done = False
        trajectory = []
        while not done:
            trajectory.append(s)
            legal = env.legal_moves()
            if rng.random() < args.epsilon:
                mv = rng.choice(legal)
            else:
                mv = agent.select_move(s, legal)
            s, r, done, info = env.step(mv)

        stats[info.status] = stats.get(info.status, 0) + 1

        # Terminal outcome from Chess perspective: +1 win, -1 loss, 0 draw
        if info.status == "draw":
            outcome_for_chess = 0.0
        else:
            outcome_for_chess = 1.0 if info.winner == Side.CHESS else -1.0

        # Convert outcome to last mover's perspective for TD update
        last_side = trajectory[-1].side_to_move
        final = outcome_for_chess if last_side == Side.CHESS else -outcome_for_chess
        vf.update_td0(trajectory, final, td_cfg)

        if (ep + 1) % 50 == 0:
            from hybrid.rl.features import PIECE_FEATURES
            names = [pk.name for pk in PIECE_FEATURES] + ["MOBILITY"]
            print()
            total = stats["chess_win"] + stats["xiangqi_win"] + stats["draw"]
            dr = stats["draw"] / total * 100 if total else 0
            print(f"[ep {ep+1}] stats={stats}  draw_rate={dr:.1f}%")
            print("  Learned weights:")
            for nm, w in zip(names, vf.theta):
                print(f"    {nm:12s} = {w:+.4f}")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(vf.as_dict(), f, ensure_ascii=False, indent=2)
        print(f"Saved value function to {args.save}")


if __name__ == "__main__":
    main()
