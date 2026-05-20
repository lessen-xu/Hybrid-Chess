"""RQ4 AB depth-2 rule-reform scan.

  * Accepts CLI args (workers / games / depth / outdir / max_plies).
  * Default workers = 8 (AB workers do no NN inference, so they fit
    comfortably on an 8-physical-core box).
  * Default outdir is runs/rq4_rule_reform_ab.

Usage::

    python -m scripts.rq4_rule_reform_ab
    python -m scripts.rq4_rule_reform_ab --workers 8 --n-games 40
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


MAT_VAL = {
    "KING": 0, "QUEEN": 9, "ROOK": 5, "BISHOP": 3, "KNIGHT": 3, "PAWN": 1,
    "GENERAL": 0, "CHARIOT": 5, "CANNON": 3, "HORSE": 3,
    "ELEPHANT": 2, "ADVISOR": 2, "SOLDIER": 1,
}

# (name, variant_dict) — covers single-flag baselines + Queen-removal combos.
VARIANTS = [
    ("default",                 {}),
    ("no_promo",                {"no_promotion": True}),
    ("palace",                  {"chess_palace": True}),
    ("knight_blk",              {"knight_block": True}),
    ("no_promo+palace",         {"no_promotion": True, "chess_palace": True}),
    ("no_promo+knight_blk",     {"no_promotion": True, "knight_block": True}),
    ("palace+knight_blk",       {"chess_palace": True, "knight_block": True}),
    ("ALL_RULES",               {"no_promotion": True, "chess_palace": True, "knight_block": True}),
    ("no_queen",                {"no_queen": True}),
    ("no_queen+no_promo",       {"no_queen": True, "no_promotion": True}),
    ("no_queen+palace",         {"no_queen": True, "chess_palace": True}),
    ("no_queen+knight_blk",     {"no_queen": True, "knight_block": True}),
    ("no_queen+ALL_RULES",      {"no_queen": True, "no_promotion": True, "chess_palace": True, "knight_block": True}),
    ("nq+ec",                   {"no_queen": True, "extra_cannon": True}),
    ("nq+ec+no_promo",          {"no_queen": True, "extra_cannon": True, "no_promotion": True}),
    ("nq+ec+palace",            {"no_queen": True, "extra_cannon": True, "chess_palace": True}),
    ("nq+ec+ALL_RULES",         {"no_queen": True, "extra_cannon": True, "no_promotion": True, "chess_palace": True, "knight_block": True}),
    ("nq+nb",                   {"no_queen": True, "no_bishop": True}),
    ("nq+nb+no_promo",          {"no_queen": True, "no_bishop": True, "no_promotion": True}),
    ("nq+nb+palace",            {"no_queen": True, "no_bishop": True, "chess_palace": True}),
    ("nq+nb+knight_blk",        {"no_queen": True, "no_bishop": True, "knight_block": True}),
    ("nq+nb+ALL_RULES",         {"no_queen": True, "no_bishop": True, "no_promotion": True, "chess_palace": True, "knight_block": True}),
    ("nq+nb+es+ALL_RULES",      {"no_queen": True, "no_bishop": True, "extra_soldier": True,
                                  "no_promotion": True, "chess_palace": True, "knight_block": True}),
]


def play_ab_batch(variant_dict, depth, num_games, seed, max_plies=150):
    from hybrid.core.config import VariantConfig
    from hybrid.core.env import HybridChessEnv
    from hybrid.core.types import Side, Move as PyMove
    from hybrid.cpp_engine import best_move as cpp_best_move, Side as CppSide

    vcfg = VariantConfig(**variant_dict)
    env = HybridChessEnv(use_cpp=True, max_plies=max_plies, variant=vcfg)
    results = []

    for _ in range(num_games):
        state = env.reset()
        info = type("obj", (object,), {"winner": None, "reason": ""})()
        while True:
            cpp_board = env._cpp_board
            side_cpp = CppSide.CHESS if state.side_to_move == Side.CHESS else CppSide.XIANGQI
            rep_table = dict(state.repetition)
            sr = cpp_best_move(cpp_board, side_cpp, depth, rep_table, state.ply, max_plies)
            if sr.best_move is None:
                break
            py_mv = PyMove(sr.best_move.fx, sr.best_move.fy, sr.best_move.tx, sr.best_move.ty)
            state, _, done, info = env.step(py_mv)
            if done:
                break

        if info.winner == Side.CHESS:
            actual = "Chess"
        elif info.winner == Side.XIANGQI:
            actual = "XQ"
        else:
            actual = "Draw"

        chess_mat = xq_mat = 0
        for _x, _y, p in state.board.iter_pieces():
            val = MAT_VAL.get(p.kind.name, 0)
            if p.side == Side.CHESS:
                chess_mat += val
            else:
                xq_mat += val
        mat_diff = chess_mat - xq_mat
        mat_winner = "Chess" if mat_diff > 0 else ("XQ" if mat_diff < 0 else "Even")

        results.append({
            "actual": actual, "ply": state.ply,
            "mat_diff": mat_diff, "mat_winner": mat_winner,
            "reason": getattr(info, "reason", ""),
        })
    return results


def summarize(games):
    n = len(games)
    if n == 0:
        return {"n": 0}
    cw = sum(1 for g in games if g["actual"] == "Chess")
    xw = sum(1 for g in games if g["actual"] == "XQ")
    dr = sum(1 for g in games if g["actual"] == "Draw")
    avg_ply = sum(g["ply"] for g in games) / n
    draws = [g for g in games if g["actual"] == "Draw"]
    mtb_c = sum(1 for g in draws if g["mat_winner"] == "Chess") if draws else 0
    mtb_x = sum(1 for g in draws if g["mat_winner"] == "XQ") if draws else 0
    mtb_e = sum(1 for g in draws if g["mat_winner"] == "Even") if draws else 0
    avg_md = sum(g["mat_diff"] for g in draws) / len(draws) if draws else 0
    adj_c, adj_x = cw + mtb_c, xw + mtb_x
    total_dec = adj_c + adj_x
    signed = (adj_c - adj_x) / total_dec if total_dec > 0 else 0.0
    return {
        "n": n, "chess_wins": cw, "xq_wins": xw, "draws": dr,
        "avg_ply": round(avg_ply, 1),
        "mtb_chess": mtb_c, "mtb_xq": mtb_x, "mtb_even": mtb_e,
        "avg_mat_diff": round(avg_md, 2),
        "adj_chess": adj_c, "adj_xq": adj_x,
        "signed_balance": round(signed, 4),
    }


def run_parallel(variant_dict, depth, total, workers, max_plies, timeout=600):
    gpw = total // workers
    rem = total % workers
    futs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for w in range(workers):
            n = gpw + (1 if w < rem else 0)
            if n:
                futs.append(pool.submit(
                    play_ab_batch, variant_dict, depth, n,
                    42 + w * 10000, max_plies,
                ))
    all_games = []
    for f in futs:
        try:
            all_games.extend(f.result(timeout=timeout))
        except Exception as e:
            print(f"    [WARNING] Worker failed: {e}")
    return all_games


def main():
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--n-games", type=int, default=40)
    ap.add_argument("--max-plies", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--outdir", type=str,
                    default="runs/rq4_rule_reform_ab")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logpath = outdir / "progress.log"
    jsonpath = outdir / "results.json"
    logf = open(logpath, "w", encoding="utf-8", buffering=1)

    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    total_variants = len(VARIANTS)
    log("=" * 80)
    log(f"  RQ4: RULE REFORM AB D{args.depth} Balance Test")
    log(f"  {total_variants} variants x {args.n_games} games, "
        f"max_plies={args.max_plies}, {args.workers} workers")
    log(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Outdir: {outdir}")
    log("=" * 80)
    log("")

    all_results = {}
    t_total = time.time()

    for vi, (vname, vdict) in enumerate(VARIANTS):
        log(f"[{vi+1}/{total_variants}] {vname}")
        try:
            t0 = time.time()
            games = run_parallel(
                vdict, args.depth, args.n_games, args.workers, args.max_plies,
            )
            dt = time.time() - t0
            if not games:
                log("  SKIPPED (no results)")
                log("")
                continue
            s = summarize(games)
            log(f"  Time: {dt:.1f}s  AvgPly: {s['avg_ply']}  Games: {s['n']}/{args.n_games}")
            log(f"  Result: Chess={s['chess_wins']}  XQ={s['xq_wins']}  Draw={s['draws']}")
            if s['draws'] > 0:
                log(f"  Tiebreak: C={s['mtb_chess']}  X={s['mtb_xq']}  E={s['mtb_even']}  "
                    f"avg_matdiff={s['avg_mat_diff']:+.2f}")
            log(f"  BALANCE: signed={s['signed_balance']:+.4f}  "
                f"adj_C={s['adj_chess']}  adj_X={s['adj_xq']}")
            elapsed = time.time() - t_total
            eta = elapsed / (vi + 1) * (total_variants - vi - 1)
            log(f"  Elapsed: {elapsed:.0f}s  ETA: {eta:.0f}s (~{eta/60:.0f}min)")
            log("")
            all_results[vname] = {
                "variant_dict": vdict, "summary": s, "elapsed_s": round(dt, 1),
            }
        except Exception as e:
            log(f"  ERROR: {e}")
            log("")

    # Final ranking
    log("=" * 80)
    log("  FINAL RANKING (by |avg_mat_diff|, closest to 0 = best)")
    log("=" * 80)
    log("")
    log(f"  {'Rk':<4} {'Variant':<35} {'matdiff':>8} {'signed':>8} "
        f"{'C':>3} {'X':>3} {'D':>3} {'mtbC':>4} {'mtbX':>4} {'mtbE':>4} {'ply':>5}")
    log(f"  {'-'*90}")

    ranked = sorted(
        all_results.items(),
        key=lambda x: abs(x[1]["summary"].get("avg_mat_diff", 99)),
    )
    for rank, (vname, vdata) in enumerate(ranked, 1):
        s = vdata["summary"]
        marker = " ***" if abs(s.get("avg_mat_diff", 99)) <= 3 else ""
        log(f"  {rank:<4} {vname:<35} {s['avg_mat_diff']:>+8.2f} "
            f"{s['signed_balance']:>+8.4f} "
            f"{s['chess_wins']:>3} {s['xq_wins']:>3} {s['draws']:>3} "
            f"{s['mtb_chess']:>4} {s['mtb_xq']:>4} {s['mtb_even']:>4} "
            f"{s['avg_ply']:>5}{marker}")

    log("")
    log(f"  Total: {time.time() - t_total:.0f}s")
    if ranked:
        log(f"  Best: {ranked[0][0]} (matdiff={ranked[0][1]['summary']['avg_mat_diff']:+.2f})")
    log("")

    with open(jsonpath, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"  Saved: {jsonpath}")
    logf.close()


if __name__ == "__main__":
    main()
