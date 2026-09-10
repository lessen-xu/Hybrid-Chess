"""Measure 64-simulation play throughput on a compute allocation (no fitting)."""
import argparse
import json
import time

import torch

from hybrid.agents.alphazero_stub import AlphaZeroMiniAgent, MCTSConfig, TorchPolicyValueModel
from hybrid.core.env import HybridChessEnv
from hybrid.rl.general_model import new_model
from hybrid.rl.general_train import require_compute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--moves", type=int, default=32)
    args = parser.parse_args()
    require_compute()
    torch.set_num_threads(1)
    env = HybridChessEnv(use_cpp=True)
    state = env.reset()
    agent = AlphaZeroMiniAgent(TorchPolicyValueModel(new_model(), args.device),
        MCTSConfig(simulations=64, discount_factor=1., dirichlet_eps=0.), use_cpp=True)
    started = time.perf_counter()
    completed = 0
    for _ in range(args.moves):
        legal = env.legal_moves()
        move = agent.select_move(state, legal)
        state, _, done, _ = env.step(move)
        completed += 1
        if done:
            state = env.reset()
    elapsed = time.perf_counter()-started
    print(json.dumps({"device": args.device, "moves": completed, "seconds": elapsed,
                      "moves_per_second": completed/elapsed, "simulations": 64}), flush=True)


if __name__ == "__main__":
    main()
