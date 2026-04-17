"""RQ4 Side Balance Spectrum: test Chess vs Xiangqi balance across agent skill levels.

For each variant (default, no_queen, extra_cannon), we have identical agents
play against themselves and record which SIDE wins. This removes all agent-skill
confounds and measures pure game-structural bias.

Usage:
    python scripts/rq4_side_balance.py
"""

import json
import time
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class SideResult:
    chess_wins: int = 0
    xiangqi_wins: int = 0
    draws: int = 0
    total_plies: int = 0

    @property
    def games(self):
        return self.chess_wins + self.xiangqi_wins + self.draws

    @property
    def chess_pct(self):
        return self.chess_wins / self.games * 100 if self.games else 0

    @property
    def xiangqi_pct(self):
        return self.xiangqi_wins / self.games * 100 if self.games else 0

    @property
    def draw_pct(self):
        return self.draws / self.games * 100 if self.games else 0

    @property
    def avg_ply(self):
        return self.total_plies / self.games if self.games else 0


def play_side_balance_batch(
    agent_type: str,
    agent_kwargs: dict,
    variant: str,
    num_games: int,
    seed: int,
    use_cpp: bool = True,
) -> SideResult:
    """Play num_games of agent-vs-self, return side balance."""
    from hybrid.core.config import VariantConfig
    from hybrid.core.env import HybridChessEnv
    from hybrid.core.types import Side

    # Build variant config directly (not via global state, which doesn't survive spawn)
    _VARIANT_MAP = {
        "none": VariantConfig(),
        "no_queen": VariantConfig(no_queen=True),
        "extra_cannon": VariantConfig(extra_cannon=True),
    }
    vcfg = _VARIANT_MAP.get(variant, VariantConfig())

    use_cpp_ab = agent_type.startswith("ab") and use_cpp
    cpp_engine = None
    if use_cpp_ab:
        try:
            from hybrid.cpp_engine import best_move as cpp_best_move, Side as CppSide
            cpp_engine = True
        except ImportError:
            use_cpp_ab = False

    # Build agent(s) — fresh copies per game for determinism
    def make_agent(s):
        if agent_type == "random":
            from hybrid.agents.random_agent import RandomAgent
            return RandomAgent(seed=s)
        elif agent_type == "greedy":
            from hybrid.agents.greedy_agent import GreedyAgent
            return GreedyAgent()
        elif agent_type.startswith("ab") and not use_cpp_ab:
            from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig
            depth = agent_kwargs.get("depth", 2)
            return AlphaBetaAgent(cfg=SearchConfig(depth=depth))
        elif agent_type.startswith("ab") and use_cpp_ab:
            return None  # will use C++ best_move directly
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    result = SideResult()
    env = HybridChessEnv(use_cpp=use_cpp, max_plies=200, variant=vcfg)
    ab_depth = agent_kwargs.get("depth", 2)

    for i in range(num_games):
        state = env.reset()

        if use_cpp_ab:
            # Full C++ game loop using best_move
            while True:
                cpp_board = env._cpp_board
                side_cpp = CppSide.CHESS if state.side_to_move == Side.CHESS else CppSide.XIANGQI
                rep_table = dict(state.repetition)
                sr = cpp_best_move(cpp_board, side_cpp, ab_depth, rep_table, state.ply, 200)
                if sr.best_move is None:
                    break
                from hybrid.core.types import Move as PyMove
                py_mv = PyMove(sr.best_move.fx, sr.best_move.fy, sr.best_move.tx, sr.best_move.ty)
                state, reward, done, info = env.step(py_mv)
                if done:
                    break
        else:
            agent_chess = make_agent(seed + i * 2)
            agent_xq = make_agent(seed + i * 2 + 1)
            agents = {Side.CHESS: agent_chess, Side.XIANGQI: agent_xq}

            while True:
                legal = env.legal_moves()
                if len(legal) == 0:
                    break
                agent = agents[state.side_to_move]
                mv = agent.select_move(state, legal)
                state, reward, done, info = env.step(mv)
                if done:
                    break

        result.total_plies += state.ply
        if info.winner == Side.CHESS:
            result.chess_wins += 1
        elif info.winner == Side.XIANGQI:
            result.xiangqi_wins += 1
        else:
            result.draws += 1

    return result


def run_experiment_parallel(
    agent_type: str,
    agent_kwargs: dict,
    variant: str,
    total_games: int,
    num_workers: int = 4,
    seed: int = 0,
) -> SideResult:
    """Run side balance experiment with multiprocessing."""
    games_per_worker = total_games // num_workers
    remainder = total_games % num_workers

    futures = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for w in range(num_workers):
            n = games_per_worker + (1 if w < remainder else 0)
            if n == 0:
                continue
            f = pool.submit(
                play_side_balance_batch,
                agent_type=agent_type,
                agent_kwargs=agent_kwargs,
                variant=variant,
                num_games=n,
                seed=seed + w * 10000,
                use_cpp=True,
            )
            futures.append(f)

    merged = SideResult()
    for f in as_completed(futures):
        r = f.result()
        merged.chess_wins += r.chess_wins
        merged.xiangqi_wins += r.xiangqi_wins
        merged.draws += r.draws
        merged.total_plies += r.total_plies

    return merged


def main():
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    VARIANTS = ["none", "no_queen", "extra_cannon"]
    VARIANT_LABELS = {"none": "Default", "no_queen": "No-Queen", "extra_cannon": "Extra-Cannon (V4)"}

    # Agent configs: (type, kwargs, total_games, num_workers)
    AGENTS = [
        ("random",  {},             500, 6),
        ("greedy",  {},             200, 6),
        ("ab",      {"depth": 2},   60,  4),
    ]
    AGENT_LABELS = {
        ("random", 0): "Random",
        ("greedy", 0): "Greedy",
        ("ab", 2): "AB D2",
    }

    results = {}
    outdir = Path("runs/rq4_side_balance")
    outdir.mkdir(parents=True, exist_ok=True)

    total_exps = len(VARIANTS) * len(AGENTS)
    exp_i = 0

    print("=" * 70)
    print("  RQ4 Side Balance Spectrum Experiment")
    print("=" * 70)
    print()

    for variant in VARIANTS:
        label_v = VARIANT_LABELS[variant]
        print(f"--- Variant: {label_v} ({variant}) ---")

        for agent_type, agent_kwargs, total_games, num_workers in AGENTS:
            depth = agent_kwargs.get("depth", 0)
            label_a = AGENT_LABELS.get((agent_type, depth), agent_type)
            exp_i += 1

            print(f"  [{exp_i}/{total_exps}] {label_a} vs {label_a}, "
                  f"N={total_games}, {num_workers} workers ... ", end="", flush=True)

            t0 = time.time()
            r = run_experiment_parallel(
                agent_type=agent_type,
                agent_kwargs=agent_kwargs,
                variant=variant,
                total_games=total_games,
                num_workers=num_workers,
                seed=42,
            )
            elapsed = time.time() - t0

            key = f"{variant}__{label_a}"
            results[key] = {
                "variant": variant,
                "variant_label": label_v,
                "agent": label_a,
                "chess_wins": r.chess_wins,
                "xiangqi_wins": r.xiangqi_wins,
                "draws": r.draws,
                "games": r.games,
                "chess_pct": round(r.chess_pct, 1),
                "xiangqi_pct": round(r.xiangqi_pct, 1),
                "draw_pct": round(r.draw_pct, 1),
                "avg_ply": round(r.avg_ply, 1),
                "elapsed_s": round(elapsed, 1),
            }

            print(f"Chess {r.chess_wins}({r.chess_pct:.0f}%)  "
                  f"XQ {r.xiangqi_wins}({r.xiangqi_pct:.0f}%)  "
                  f"Draw {r.draws}({r.draw_pct:.0f}%)  "
                  f"[{elapsed:.0f}s]")

        print()

    # Save results
    out_path = outdir / "side_balance_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved: {out_path}")

    # Print summary table
    print()
    print("=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Agent':<12} {'Variant':<18} {'Chess%':>7} {'XQ%':>7} {'Draw%':>7} {'N':>5} {'AvgPly':>7}")
    print("-" * 70)
    for key, r in results.items():
        print(f"{r['agent']:<12} {r['variant_label']:<18} "
              f"{r['chess_pct']:>6.1f}% {r['xiangqi_pct']:>6.1f}% "
              f"{r['draw_pct']:>6.1f}% {r['games']:>5} {r['avg_ply']:>7.1f}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
