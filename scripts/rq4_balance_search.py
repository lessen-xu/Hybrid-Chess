"""RQ4 Balance Search: find the rule variant closest to 50/50 side balance.

Uses Greedy + Random agents as fast probes across many variant combinations.
max_plies=1000 to minimize draw masking.

Usage:
    python scripts/rq4_balance_search.py
"""

import json
import time
import sys
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

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
    def balance_score(self):
        """0 = perfect balance, 1 = completely one-sided."""
        decisive = self.chess_wins + self.xiangqi_wins
        if decisive == 0:
            return None  # all draws, can't measure
        return abs(self.chess_wins - self.xiangqi_wins) / decisive

    @property
    def avg_ply(self):
        return self.total_plies / self.games if self.games else 0


def play_batch(variant_dict, agent_type, num_games, seed, max_plies=1000):
    """Play games with given variant config. Runs in subprocess."""
    from hybrid.core.config import VariantConfig
    from hybrid.core.env import HybridChessEnv
    from hybrid.core.types import Side

    vcfg = VariantConfig(**variant_dict)
    env = HybridChessEnv(use_cpp=True, max_plies=max_plies, variant=vcfg)

    if agent_type == "random":
        from hybrid.agents.random_agent import RandomAgent
    elif agent_type == "greedy":
        from hybrid.agents.greedy_agent import GreedyAgent

    result = SideResult()
    for i in range(num_games):
        state = env.reset()

        if agent_type == "random":
            a_chess = RandomAgent(seed=seed + i * 2)
            a_xq = RandomAgent(seed=seed + i * 2 + 1)
        else:
            a_chess = GreedyAgent()
            a_xq = GreedyAgent()

        agents = {Side.CHESS: a_chess, Side.XIANGQI: a_xq}

        while True:
            legal = env.legal_moves()
            if len(legal) == 0:
                break
            mv = agents[state.side_to_move].select_move(state, legal)
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


def run_parallel(variant_dict, agent_type, total_games, num_workers=6,
                 seed=42, max_plies=1000):
    """Run games in parallel."""
    gpw = total_games // num_workers
    rem = total_games % num_workers

    futures = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for w in range(num_workers):
            n = gpw + (1 if w < rem else 0)
            if n == 0:
                continue
            futures.append(pool.submit(
                play_batch, variant_dict, agent_type, n,
                seed + w * 10000, max_plies
            ))

    merged = SideResult()
    for f in as_completed(futures):
        r = f.result()
        merged.chess_wins += r.chess_wins
        merged.xiangqi_wins += r.xiangqi_wins
        merged.draws += r.draws
        merged.total_plies += r.total_plies
    return merged


# ── Variant definitions ──

# Individual rule tweaks (name, VariantConfig kwargs)
SINGLE_TWEAKS = {
    "no_queen":          {"no_queen": True},
    "no_bishop":         {"no_bishop": True},
    "one_rook":          {"one_rook": True},
    "remove_pawn":       {"remove_extra_pawn": True},
    "no_promo":          {"no_queen_promotion": True},
    "extra_cannon":      {"extra_cannon": True},
    "extra_soldier":     {"extra_soldier": True},
    "no_flying_general": {"flying_general": False},
}

def build_variants():
    """Build a list of (name, variant_dict) to test."""
    variants = [("default", {})]

    # All singles
    for name, kw in SINGLE_TWEAKS.items():
        variants.append((name, kw))

    # Promising combinations (Chess nerfs + Xiangqi buffs)
    combos = [
        # Double nerfs
        ("no_queen+extra_cannon", {**SINGLE_TWEAKS["no_queen"], **SINGLE_TWEAKS["extra_cannon"]}),
        ("no_queen+extra_soldier", {**SINGLE_TWEAKS["no_queen"], **SINGLE_TWEAKS["extra_soldier"]}),
        ("no_queen+no_promo", {**SINGLE_TWEAKS["no_queen"], **SINGLE_TWEAKS["no_promo"]}),
        ("no_bishop+extra_cannon", {**SINGLE_TWEAKS["no_bishop"], **SINGLE_TWEAKS["extra_cannon"]}),
        ("one_rook+extra_cannon", {**SINGLE_TWEAKS["one_rook"], **SINGLE_TWEAKS["extra_cannon"]}),
        ("one_rook+extra_soldier", {**SINGLE_TWEAKS["one_rook"], **SINGLE_TWEAKS["extra_soldier"]}),
        # Triple combos
        ("no_queen+extra_cannon+extra_soldier",
         {**SINGLE_TWEAKS["no_queen"], **SINGLE_TWEAKS["extra_cannon"],
          **SINGLE_TWEAKS["extra_soldier"]}),
        ("no_queen+no_bishop+extra_cannon",
         {**SINGLE_TWEAKS["no_queen"], **SINGLE_TWEAKS["no_bishop"],
          **SINGLE_TWEAKS["extra_cannon"]}),
        ("no_queen+one_rook+extra_cannon",
         {**SINGLE_TWEAKS["no_queen"], **SINGLE_TWEAKS["one_rook"],
          **SINGLE_TWEAKS["extra_cannon"]}),
        # Heavy nerf
        ("no_queen+no_bishop+one_rook",
         {**SINGLE_TWEAKS["no_queen"], **SINGLE_TWEAKS["no_bishop"],
          **SINGLE_TWEAKS["one_rook"]}),
        ("no_queen+no_bishop+extra_cannon+extra_soldier",
         {**SINGLE_TWEAKS["no_queen"], **SINGLE_TWEAKS["no_bishop"],
          **SINGLE_TWEAKS["extra_cannon"], **SINGLE_TWEAKS["extra_soldier"]}),
    ]
    variants.extend(combos)
    return variants


def main():
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    variants = build_variants()
    outdir = Path("runs/rq4_balance_search")
    outdir.mkdir(parents=True, exist_ok=True)

    GAMES_RANDOM = 300
    GAMES_GREEDY = 200
    MAX_PLIES = 1000

    results = []
    total = len(variants)

    print("=" * 72)
    print("  RQ4 Balance Search — Finding the Balanced Hybrid Chess Variant")
    print(f"  {total} variants, Random(N={GAMES_RANDOM}) + Greedy(N={GAMES_GREEDY})")
    print(f"  max_plies={MAX_PLIES} (minimal draw masking)")
    print("=" * 72)
    print()

    for idx, (name, vdict) in enumerate(variants):
        print(f"  [{idx+1}/{total}] {name}")

        # Random probe
        t0 = time.time()
        r_rand = run_parallel(vdict, "random", GAMES_RANDOM, 6, 42, MAX_PLIES)
        t_rand = time.time() - t0

        # Greedy probe
        t0 = time.time()
        r_grdy = run_parallel(vdict, "greedy", GAMES_GREEDY, 6, 42, MAX_PLIES)
        t_grdy = time.time() - t0

        entry = {
            "name": name,
            "variant": vdict,
            "random": {
                "chess": r_rand.chess_wins, "xq": r_rand.xiangqi_wins,
                "draw": r_rand.draws, "chess_pct": round(r_rand.chess_pct, 1),
                "xq_pct": round(r_rand.xiangqi_pct, 1),
                "draw_pct": round(r_rand.draw_pct, 1),
                "balance": round(r_rand.balance_score, 3) if r_rand.balance_score is not None else None,
                "avg_ply": round(r_rand.avg_ply, 1),
            },
            "greedy": {
                "chess": r_grdy.chess_wins, "xq": r_grdy.xiangqi_wins,
                "draw": r_grdy.draws, "chess_pct": round(r_grdy.chess_pct, 1),
                "xq_pct": round(r_grdy.xiangqi_pct, 1),
                "draw_pct": round(r_grdy.draw_pct, 1),
                "balance": round(r_grdy.balance_score, 3) if r_grdy.balance_score is not None else None,
                "avg_ply": round(r_grdy.avg_ply, 1),
            },
            "time_s": round(t_rand + t_grdy, 1),
        }
        results.append(entry)

        # Quick inline display
        rb = r_rand.balance_score
        gb = r_grdy.balance_score
        rb_s = f"{rb:.2f}" if rb is not None else "N/A"
        gb_s = f"{gb:.2f}" if gb is not None else "N/A"
        print(f"    Random: C={r_rand.chess_wins} X={r_rand.xiangqi_wins} "
              f"D={r_rand.draws}  bal={rb_s}  [{t_rand:.0f}s]")
        print(f"    Greedy: C={r_grdy.chess_wins} X={r_grdy.xiangqi_wins} "
              f"D={r_grdy.draws}  bal={gb_s}  [{t_grdy:.0f}s]")
        print()

    # Save
    out_path = outdir / "balance_search_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Ranking ──
    print("=" * 72)
    print("  BALANCE RANKING (lower = more balanced)")
    print("=" * 72)
    print()

    # Rank by combined balance (Random + Greedy)
    def combined_balance(e):
        rb = e["random"]["balance"]
        gb = e["greedy"]["balance"]
        if rb is None and gb is None:
            return 999  # all draws = unmeasurable
        if rb is None:
            return gb
        if gb is None:
            return rb
        return (rb + gb) / 2

    ranked = sorted(results, key=combined_balance)

    print(f"  {'Rank':<5} {'Variant':<40} {'Rand_Bal':>8} {'Grdy_Bal':>8} "
          f"{'R_C%':>5} {'R_X%':>5} {'G_C%':>5} {'G_X%':>5}")
    print("  " + "-" * 80)
    for i, e in enumerate(ranked):
        rb = e["random"]["balance"]
        gb = e["greedy"]["balance"]
        rb_s = f"{rb:.3f}" if rb is not None else "N/A"
        gb_s = f"{gb:.3f}" if gb is not None else "N/A"
        print(f"  {i+1:<5} {e['name']:<40} {rb_s:>8} {gb_s:>8} "
              f"{e['random']['chess_pct']:>5.1f} {e['random']['xq_pct']:>5.1f} "
              f"{e['greedy']['chess_pct']:>5.1f} {e['greedy']['xq_pct']:>5.1f}")

    print()
    print(f"  Results saved: {out_path}")
    print("  Done!")


if __name__ == "__main__":
    main()
