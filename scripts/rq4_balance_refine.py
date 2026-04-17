"""RQ4 Balance Refinement: fine-grained search around the balance flip point.

Round 1 found the flip between:
  - no_queen+no_bishop (Chess dominant)
  - no_queen+no_bishop+one_rook (XQ dominant)

This script tests many intermediate combos to find the closest to 50/50.
"""
import json, time, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

@dataclass
class SR:
    cw: int = 0; xw: int = 0; dr: int = 0; plies: int = 0
    @property
    def games(self): return self.cw + self.xw + self.dr
    @property
    def c_pct(self): return self.cw/self.games*100 if self.games else 0
    @property
    def x_pct(self): return self.xw/self.games*100 if self.games else 0
    @property
    def d_pct(self): return self.dr/self.games*100 if self.games else 0
    @property
    def bal(self):
        d = self.cw + self.xw
        if d == 0: return None
        return abs(self.cw - self.xw) / d
    @property
    def direction(self):
        if self.cw > self.xw: return "Chess"
        elif self.xw > self.cw: return "XQ"
        else: return "Even"

def play_batch(vdict, agent, n, seed, max_plies=1000):
    from hybrid.core.config import VariantConfig
    from hybrid.core.env import HybridChessEnv
    from hybrid.core.types import Side
    vcfg = VariantConfig(**vdict)
    env = HybridChessEnv(use_cpp=True, max_plies=max_plies, variant=vcfg)
    if agent == "random":
        from hybrid.agents.random_agent import RandomAgent
    else:
        from hybrid.agents.greedy_agent import GreedyAgent
    r = SR()
    for i in range(n):
        state = env.reset()
        if agent == "random":
            ac = RandomAgent(seed=seed+i*2); ax = RandomAgent(seed=seed+i*2+1)
        else:
            ac = GreedyAgent(); ax = GreedyAgent()
        ags = {Side.CHESS: ac, Side.XIANGQI: ax}
        while True:
            legal = env.legal_moves()
            if not legal: break
            mv = ags[state.side_to_move].select_move(state, legal)
            state, _, done, info = env.step(mv)
            if done: break
        r.plies += state.ply
        if info.winner == Side.CHESS: r.cw += 1
        elif info.winner == Side.XIANGQI: r.xw += 1
        else: r.dr += 1
    return r

def run_par(vdict, agent, total, workers=6, seed=42, max_plies=1000):
    gpw = total // workers; rem = total % workers
    futs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for w in range(workers):
            n = gpw + (1 if w < rem else 0)
            if n: futs.append(pool.submit(play_batch, vdict, agent, n, seed+w*10000, max_plies))
    m = SR()
    for f in futs:
        r = f.result(); m.cw+=r.cw; m.xw+=r.xw; m.dr+=r.dr; m.plies+=r.plies
    return m

def main():
    import multiprocessing as mp
    try: mp.set_start_method("spawn", force=True)
    except: pass

    # Fine-grained variants around the flip point
    variants = [
        # Baselines
        ("default", {}),
        ("no_queen", {"no_queen": True}),

        # Two-nerf combos (untested)
        ("no_queen+no_bishop", {"no_queen": True, "no_bishop": True}),
        ("no_queen+one_rook", {"no_queen": True, "one_rook": True}),
        ("no_queen+remove_pawn", {"no_queen": True, "remove_extra_pawn": True}),
        ("no_queen+no_promo", {"no_queen": True, "no_queen_promotion": True}),

        # Two-nerf + XQ buff (approaching balance?)
        ("no_queen+no_bishop+extra_soldier", {"no_queen": True, "no_bishop": True, "extra_soldier": True}),
        ("no_queen+no_bishop+extra_cannon", {"no_queen": True, "no_bishop": True, "extra_cannon": True}),
        ("no_queen+one_rook+extra_soldier", {"no_queen": True, "one_rook": True, "extra_soldier": True}),

        # The flip point
        ("no_queen+no_bishop+one_rook", {"no_queen": True, "no_bishop": True, "one_rook": True}),

        # Flip point + XQ buffs (over-correction?)
        ("no_queen+no_bishop+one_rook+extra_cannon",
         {"no_queen": True, "no_bishop": True, "one_rook": True, "extra_cannon": True}),
        ("no_queen+no_bishop+one_rook+extra_soldier",
         {"no_queen": True, "no_bishop": True, "one_rook": True, "extra_soldier": True}),

        # Mild combos: remove_pawn instead of one_rook
        ("no_queen+no_bishop+remove_pawn",
         {"no_queen": True, "no_bishop": True, "remove_extra_pawn": True}),
        ("no_queen+no_bishop+no_promo",
         {"no_queen": True, "no_bishop": True, "no_queen_promotion": True}),
        ("no_queen+no_bishop+remove_pawn+extra_cannon",
         {"no_queen": True, "no_bishop": True, "remove_extra_pawn": True, "extra_cannon": True}),

        # no_promo combos
        ("no_queen+no_bishop+one_rook+no_promo",
         {"no_queen": True, "no_bishop": True, "one_rook": True, "no_queen_promotion": True}),

        # Flying general variants
        ("no_queen+no_bishop+no_flying",
         {"no_queen": True, "no_bishop": True, "flying_general": False}),

        # Kitchen sink
        ("no_queen+no_bishop+one_rook+extra_cannon+extra_soldier",
         {"no_queen": True, "no_bishop": True, "one_rook": True,
          "extra_cannon": True, "extra_soldier": True}),
    ]

    N_RAND = 300
    N_GRDY = 300
    MAX_PLY = 1000
    outdir = Path("runs/rq4_balance_refine")
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(variants)
    print("=" * 75)
    print(f"  RQ4 Balance Refinement — {total} variants")
    print(f"  Random(N={N_RAND}) + Greedy(N={N_GRDY}), max_plies={MAX_PLY}")
    print("=" * 75)

    for idx, (name, vd) in enumerate(variants):
        t0 = time.time()
        rr = run_par(vd, "random", N_RAND, 6, 42, MAX_PLY)
        rg = run_par(vd, "greedy", N_GRDY, 6, 42, MAX_PLY)
        dt = time.time() - t0

        rb = rr.bal; gb = rg.bal
        rb_s = f"{rb:.3f}" if rb is not None else "N/A"
        gb_s = f"{gb:.3f}" if gb is not None else "N/A"
        rd = rr.direction; gd = rg.direction

        print(f"  [{idx+1}/{total}] {name}")
        print(f"    Rand: C={rr.cw:3d} X={rr.xw:3d} D={rr.dr:3d}  "
              f"bal={rb_s} dir={rd:5s}  "
              f"Grdy: C={rg.cw:3d} X={rg.xw:3d} D={rg.dr:3d}  "
              f"bal={gb_s} dir={gd:5s}  [{dt:.0f}s]")

        results.append({
            "name": name, "variant": vd,
            "random": {"chess": rr.cw, "xq": rr.xw, "draw": rr.dr,
                       "c_pct": round(rr.c_pct,1), "x_pct": round(rr.x_pct,1),
                       "d_pct": round(rr.d_pct,1), "balance": rb, "dir": rd},
            "greedy": {"chess": rg.cw, "xq": rg.xw, "draw": rg.dr,
                       "c_pct": round(rg.c_pct,1), "x_pct": round(rg.x_pct,1),
                       "d_pct": round(rg.d_pct,1), "balance": gb, "dir": gd},
            "time_s": round(dt,1),
        })

    # Save
    op = outdir / "refine_results.json"
    with open(op, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Final ranking
    print()
    print("=" * 75)
    print("  BALANCE RANKING (combined Random+Greedy, lower=better)")
    print("=" * 75)
    def score(e):
        rb = e["random"]["balance"]; gb = e["greedy"]["balance"]
        if rb is None and gb is None: return 999
        if rb is None: return gb
        if gb is None: return rb
        return (rb + gb) / 2
    ranked = sorted(results, key=score)
    for i, e in enumerate(ranked):
        rb = e["random"]["balance"]; gb = e["greedy"]["balance"]
        rs = f"{rb:.3f}" if rb is not None else "N/A"
        gs = f"{gb:.3f}" if gb is not None else "N/A"
        print(f"  {i+1:>2}. {e['name']:<48s} R={rs:>5} ({e['random']['dir']:>5s})  "
              f"G={gs:>5} ({e['greedy']['dir']:>5s})")
    print(f"\n  Saved: {op}")

if __name__ == "__main__":
    main()
