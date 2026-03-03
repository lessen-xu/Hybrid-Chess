#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Profile a single self-play game to measure time distribution.

Patches timing hooks into key functions to measure:
- legal_moves (env-level, per-ply)
- MCTS internal: generate_legal_moves + terminal_info (per-simulation, Python rules)
- NN inference (encode_state + forward)
- encode_state (outer self-play loop)
- env.step (per-ply)
"""

import scripts._fix_encoding  # noqa: F401
import time
import sys
import numpy as np
import torch

from hybrid.core.env import HybridChessEnv
from hybrid.core.types import Side
from hybrid.rl.az_selfplay import self_play_game, SelfPlayConfig
from hybrid.rl.az_network import PolicyValueNet
from hybrid.agents.alphazero_stub import (
    AlphaZeroMiniAgent, MCTSConfig, TorchPolicyValueModel,
)


def main():
    use_cpp = "--use-cpp" in sys.argv
    sims = 50
    print(f"Profile: use_cpp={use_cpp}, simulations={sims}")

    # Build fresh model
    net = PolicyValueNet()
    net.eval()
    model = TorchPolicyValueModel(net, device="cpu")
    agent = AlphaZeroMiniAgent(
        model=model,
        cfg=MCTSConfig(simulations=sims, dirichlet_eps=0.25),
        seed=42,
        use_cpp=use_cpp,
    )
    env = HybridChessEnv(max_plies=150, use_cpp=use_cpp)
    sp_cfg = SelfPlayConfig(
        simulations=sims,
        max_ply=150,
        resign_enabled=False,
        draw_adjudicate_enabled=True,
        draw_adjudicate_patience=15,
    )

    # ── Timing accumulators ──
    timers = {
        "mcts_gen_legal": 0.0,   # generate_legal_moves inside MCTS sims
        "mcts_terminal": 0.0,    # terminal_info inside MCTS sims
        "nn_predict": 0.0,       # model.predict (encode + forward + extract)
        "nn_encode": 0.0,        # encode_state inside predict
        "nn_forward": 0.0,       # net forward pass inside predict
        "env_legal": 0.0,        # env.legal_moves (outer loop)
        "env_step": 0.0,         # env.step (outer loop)
        "outer_encode": 0.0,     # encode_state in self_play_game
    }
    call_counts = {k: 0 for k in timers}

    # ── Monkey-patch MCTS internals ──
    import hybrid.agents.alphazero_stub as _stub
    import hybrid.core.rules as _rules
    import hybrid.rl.az_encoding as _enc
    import hybrid.rl.az_selfplay as _sp

    # Patch generate_legal_moves used in MCTS
    _orig_gen = _rules.generate_legal_moves
    def _timed_gen(*a, **kw):
        t0 = time.perf_counter()
        r = _orig_gen(*a, **kw)
        timers["mcts_gen_legal"] += time.perf_counter() - t0
        call_counts["mcts_gen_legal"] += 1
        return r
    _stub.generate_legal_moves = _timed_gen

    # Patch terminal_info used in MCTS
    _orig_term = _rules.terminal_info
    def _timed_term(*a, **kw):
        t0 = time.perf_counter()
        r = _orig_term(*a, **kw)
        timers["mcts_terminal"] += time.perf_counter() - t0
        call_counts["mcts_terminal"] += 1
        return r
    _stub.terminal_info = _timed_term

    # Patch model.predict
    _orig_predict = model.predict
    def _timed_predict(*a, **kw):
        t0 = time.perf_counter()
        r = _orig_predict(*a, **kw)
        timers["nn_predict"] += time.perf_counter() - t0
        call_counts["nn_predict"] += 1
        return r
    model.predict = _timed_predict

    # Patch encode_state inside predict for finer measurement
    _orig_encode = _enc.encode_state
    def _timed_encode(*a, **kw):
        t0 = time.perf_counter()
        r = _orig_encode(*a, **kw)
        timers["nn_encode"] += time.perf_counter() - t0
        call_counts["nn_encode"] += 1
        return r
    _enc.encode_state = _timed_encode
    # Also patch in the selfplay module
    _sp.encode_state = _timed_encode

    # Patch env.legal_moves
    _orig_legal = env.legal_moves
    def _timed_legal(*a, **kw):
        t0 = time.perf_counter()
        r = _orig_legal(*a, **kw)
        timers["env_legal"] += time.perf_counter() - t0
        call_counts["env_legal"] += 1
        return r
    env.legal_moves = _timed_legal

    # Patch env.step
    _orig_step = env.step
    def _timed_step(*a, **kw):
        t0 = time.perf_counter()
        r = _orig_step(*a, **kw)
        timers["env_step"] += time.perf_counter() - t0
        call_counts["env_step"] += 1
        return r
    env.step = _timed_step

    # ── Run one game ──
    print("Running 1 self-play game ...")
    t_game_start = time.perf_counter()
    examples, record = self_play_game(env, agent, sp_cfg)
    t_game_total = time.perf_counter() - t_game_start

    plies = record.ply_count
    print(f"\nGame: {plies} plies, {len(examples)} samples, "
          f"result={record.result}, reason={record.termination_reason}")
    print(f"Total wall time: {t_game_total:.2f}s\n")

    # ── Report ──
    print(f"{'Component':<25} {'Time (s)':>10} {'Calls':>8} {'% Total':>8}")
    print("-" * 53)
    for key in ["mcts_gen_legal", "mcts_terminal", "nn_predict", "nn_encode",
                "env_legal", "env_step"]:
        t = timers[key]
        c = call_counts[key]
        pct = 100 * t / t_game_total if t_game_total > 0 else 0
        print(f"{key:<25} {t:>10.3f} {c:>8} {pct:>7.1f}%")

    # Derived
    nn_other = timers["nn_predict"] - timers["nn_encode"]
    accounted = (timers["mcts_gen_legal"] + timers["mcts_terminal"] +
                 timers["nn_predict"] + timers["env_legal"] + timers["env_step"])
    overhead = t_game_total - accounted

    print(f"\n{'--- Derived ---':<25}")
    print(f"{'nn_other (fwd+extract)':<25} {nn_other:>10.3f} {'':>8} "
          f"{100*nn_other/t_game_total:>7.1f}%")
    print(f"{'overhead (tree/noise/..)':<25} {overhead:>10.3f} {'':>8} "
          f"{100*overhead/t_game_total:>7.1f}%")
    print(f"{'TOTAL accounted':<25} {accounted:>10.3f} {'':>8} "
          f"{100*accounted/t_game_total:>7.1f}%")


if __name__ == "__main__":
    main()
