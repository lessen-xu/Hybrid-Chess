"""RQ4 Comprehensive AB D2 Balance Ablation.

Tests ALL candidate balance variants with AB D2 (C++ accelerated).
Provides rich diagnostics per variant: decisive results, material tiebreak,
draw classification, and per-game detail.

Log file updates in real-time (flush after every variant).
"""
import json, time, sys, os, collections
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Material values
MAT_VAL = {
    'KING': 0, 'QUEEN': 9, 'ROOK': 5, 'BISHOP': 3, 'KNIGHT': 3, 'PAWN': 1,
    'GENERAL': 0, 'CHARIOT': 5, 'CANNON': 3, 'HORSE': 3,
    'ELEPHANT': 2, 'ADVISOR': 2, 'SOLDIER': 1,
}

# ── ALL variants from balance_search + balance_refine (union, deduplicated) ──
VARIANTS = [
    # --- Baselines ---
    ("default",                                     {}),
    ("extra_cannon",                                {"extra_cannon": True}),
    ("extra_soldier",                               {"extra_soldier": True}),

    # --- Single Chess nerfs ---
    ("no_queen",                                    {"no_queen": True}),
    ("no_bishop",                                   {"no_bishop": True}),
    ("one_rook",                                    {"one_rook": True}),
    ("remove_pawn",                                 {"remove_extra_pawn": True}),
    ("no_promo",                                    {"no_queen_promotion": True}),
    ("no_flying_general",                           {"flying_general": False}),

    # --- Queen + single XQ buff ---
    ("no_queen+extra_cannon",                       {"no_queen": True, "extra_cannon": True}),
    ("no_queen+extra_soldier",                      {"no_queen": True, "extra_soldier": True}),
    ("no_queen+no_promo",                           {"no_queen": True, "no_queen_promotion": True}),

    # --- Bishop/Rook + XQ buff ---
    ("no_bishop+extra_cannon",                      {"no_bishop": True, "extra_cannon": True}),
    ("one_rook+extra_cannon",                       {"one_rook": True, "extra_cannon": True}),
    ("one_rook+extra_soldier",                      {"one_rook": True, "extra_soldier": True}),

    # --- Two Chess nerfs ---
    ("no_queen+no_bishop",                          {"no_queen": True, "no_bishop": True}),
    ("no_queen+one_rook",                           {"no_queen": True, "one_rook": True}),
    ("no_queen+remove_pawn",                        {"no_queen": True, "remove_extra_pawn": True}),

    # --- Two nerfs + XQ buff ---
    ("no_queen+extra_cannon+extra_soldier",         {"no_queen": True, "extra_cannon": True, "extra_soldier": True}),
    ("no_queen+no_bishop+extra_cannon",             {"no_queen": True, "no_bishop": True, "extra_cannon": True}),
    ("no_queen+no_bishop+extra_soldier",            {"no_queen": True, "no_bishop": True, "extra_soldier": True}),
    ("no_queen+one_rook+extra_cannon",              {"no_queen": True, "one_rook": True, "extra_cannon": True}),
    ("no_queen+one_rook+extra_soldier",             {"no_queen": True, "one_rook": True, "extra_soldier": True}),

    # --- Deep nerfs ---
    ("no_queen+no_bishop+one_rook",                 {"no_queen": True, "no_bishop": True, "one_rook": True}),
    ("no_queen+no_bishop+remove_pawn",              {"no_queen": True, "no_bishop": True, "remove_extra_pawn": True}),
    ("no_queen+no_bishop+no_promo",                 {"no_queen": True, "no_bishop": True, "no_queen_promotion": True}),
    ("no_queen+no_bishop+no_flying",                {"no_queen": True, "no_bishop": True, "flying_general": False}),

    # --- Deep nerfs + XQ buff ---
    ("no_queen+no_bishop+one_rook+extra_cannon",    {"no_queen": True, "no_bishop": True, "one_rook": True, "extra_cannon": True}),
    ("no_queen+no_bishop+one_rook+extra_soldier",   {"no_queen": True, "no_bishop": True, "one_rook": True, "extra_soldier": True}),
    ("no_queen+no_bishop+remove_pawn+extra_cannon", {"no_queen": True, "no_bishop": True, "remove_extra_pawn": True, "extra_cannon": True}),
    ("no_queen+no_bishop+one_rook+no_promo",        {"no_queen": True, "no_bishop": True, "one_rook": True, "no_queen_promotion": True}),
    ("no_queen+no_bishop+extra_cannon+extra_soldier", {"no_queen": True, "no_bishop": True, "extra_cannon": True, "extra_soldier": True}),

    # --- Extreme ---
    ("no_queen+no_bishop+one_rook+extra_cannon+extra_soldier",
     {"no_queen": True, "no_bishop": True, "one_rook": True, "extra_cannon": True, "extra_soldier": True}),
]


def play_ab_batch(variant_dict, depth, num_games, seed, max_plies=200):
    """Pure C++ AB vs AB games with per-game diagnostics."""
    from hybrid.core.config import VariantConfig
    from hybrid.core.env import HybridChessEnv
    from hybrid.core.types import Side, Move as PyMove
    from hybrid.cpp_engine import best_move as cpp_best_move, Side as CppSide

    vcfg = VariantConfig(**variant_dict)
    env = HybridChessEnv(use_cpp=True, max_plies=max_plies, variant=vcfg)
    results = []

    for i in range(num_games):
        state = env.reset()
        while True:
            cpp_board = env._cpp_board
            side_cpp = CppSide.CHESS if state.side_to_move == Side.CHESS else CppSide.XIANGQI
            rep_table = dict(state.repetition)
            sr = cpp_best_move(cpp_board, side_cpp, depth, rep_table, state.ply, max_plies)
            if sr.best_move is None:
                break
            py_mv = PyMove(sr.best_move.fx, sr.best_move.fy, sr.best_move.tx, sr.best_move.ty)
            state, reward, done, info = env.step(py_mv)
            if done:
                break

        # Winner
        if info.winner == Side.CHESS:
            actual = "Chess"
        elif info.winner == Side.XIANGQI:
            actual = "XQ"
        else:
            actual = "Draw"

        # Material
        chess_mat = xq_mat = 0
        for x, y, p in state.board.iter_pieces():
            val = MAT_VAL.get(p.kind.name, 0)
            if p.side == Side.CHESS:
                chess_mat += val
            else:
                xq_mat += val
        mat_diff = chess_mat - xq_mat
        mat_winner = "Chess" if mat_diff > 0 else ("XQ" if mat_diff < 0 else "Even")

        results.append({
            "actual": actual,
            "ply": state.ply,
            "mat_diff": mat_diff,
            "mat_winner": mat_winner,
            "reason": getattr(info, 'reason', ''),
        })
    return results


def run_parallel(variant_dict, depth, total, workers=6, max_plies=150, timeout=600):
    gpw = total // workers
    rem = total % workers
    futs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for w in range(workers):
            n = gpw + (1 if w < rem else 0)
            if n:
                futs.append(pool.submit(
                    play_ab_batch, variant_dict, depth, n, 42 + w * 10000, max_plies
                ))
    all_games = []
    for f in futs:
        try:
            all_games.extend(f.result(timeout=timeout))
        except Exception as e:
            print(f"    [WARNING] Worker failed/timeout: {e}")
    return all_games


def summarize(games):
    """Return summary dict for a set of games."""
    n = len(games)
    cw = sum(1 for g in games if g["actual"] == "Chess")
    xw = sum(1 for g in games if g["actual"] == "XQ")
    dr = sum(1 for g in games if g["actual"] == "Draw")
    avg_ply = sum(g["ply"] for g in games) / n if n else 0

    draws = [g for g in games if g["actual"] == "Draw"]
    mtb_c = sum(1 for g in draws if g["mat_winner"] == "Chess") if draws else 0
    mtb_x = sum(1 for g in draws if g["mat_winner"] == "XQ") if draws else 0
    mtb_e = sum(1 for g in draws if g["mat_winner"] == "Even") if draws else 0
    avg_md = sum(g["mat_diff"] for g in draws) / len(draws) if draws else 0

    # Adjusted results
    adj_c = cw + mtb_c
    adj_x = xw + mtb_x
    total_dec = adj_c + adj_x
    if total_dec > 0:
        adj_balance = abs(adj_c - adj_x) / total_dec
        adj_dominant = "Chess" if adj_c > adj_x else ("XQ" if adj_x > adj_c else "Even")
    else:
        adj_balance = 0.0
        adj_dominant = "Even"

    # Signed balance: positive = Chess advantage, negative = XQ advantage
    if total_dec > 0:
        signed_balance = (adj_c - adj_x) / total_dec
    else:
        signed_balance = 0.0

    return {
        "n": n,
        "chess_wins": cw, "xq_wins": xw, "draws": dr,
        "avg_ply": round(avg_ply, 1),
        "mtb_chess": mtb_c, "mtb_xq": mtb_x, "mtb_even": mtb_e,
        "avg_mat_diff": round(avg_md, 2),
        "adj_chess": adj_c, "adj_xq": adj_x,
        "adj_balance": round(adj_balance, 4),
        "adj_dominant": adj_dominant,
        "signed_balance": round(signed_balance, 4),
    }


def main():
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except:
        pass

    DEPTH = 2
    N_GAMES = 40
    MAX_PLIES = 150
    WORKERS = 6
    VARIANT_TIMEOUT = 600  # seconds per variant before skip

    outdir = Path("runs/rq4_ab_full_ablation")
    outdir.mkdir(parents=True, exist_ok=True)
    logpath = outdir / "progress.log"
    jsonpath = outdir / "ab_ablation_results.json"

    # Open log for real-time writing
    logf = open(logpath, "w", encoding="utf-8", buffering=1)  # line-buffered

    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    total_variants = len(VARIANTS)
    log("=" * 80)
    log(f"  RQ4: COMPREHENSIVE AB D{DEPTH} Balance Ablation")
    log(f"  {total_variants} variants x {N_GAMES} games, max_plies={MAX_PLIES}, {WORKERS} workers")
    log(f"  All C++ accelerated: best_move() + HybridChessEnv(use_cpp=True)")
    log(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 80)
    log("")

    all_results = {}
    t_total = time.time()

    for vi, (vname, vdict) in enumerate(VARIANTS):
        log(f"[{vi+1}/{total_variants}] {vname}")
        try:
            t0 = time.time()
            games = run_parallel(vdict, DEPTH, N_GAMES, WORKERS, MAX_PLIES, timeout=VARIANT_TIMEOUT)
            dt = time.time() - t0

            if not games:
                log(f"  SKIPPED (no games completed within timeout)")
                log("")
                continue

            s = summarize(games)

            log(f"  Time: {dt:.1f}s  AvgPly: {s['avg_ply']}  Games: {s['n']}/{N_GAMES}")
            log(f"  Result: Chess={s['chess_wins']}  XQ={s['xq_wins']}  Draw={s['draws']}")
            if s['draws'] > 0:
                log(f"  Material tiebreak: Chess={s['mtb_chess']}  XQ={s['mtb_xq']}  Even={s['mtb_even']}  avg_mat_diff={s['avg_mat_diff']:+.2f}")
            log(f"  ADJUSTED: Chess={s['adj_chess']}  XQ={s['adj_xq']}  balance={s['adj_balance']:.4f} ({s['adj_dominant']})  signed={s['signed_balance']:+.4f}")
            elapsed_total = time.time() - t_total
            remaining = elapsed_total / (vi + 1) * (total_variants - vi - 1)
            log(f"  Elapsed: {elapsed_total:.0f}s  ETA: {remaining:.0f}s (~{remaining/60:.0f}min)")
            log("")

            all_results[vname] = {
                "variant_dict": vdict,
                "games": games,
                "summary": s,
                "elapsed_s": round(dt, 1),
            }
        except Exception as e:
            log(f"  ERROR: {e}")
            log("")

    # Final ranking by |signed_balance| (closest to 0 = most balanced)
    log("=" * 80)
    log("  FINAL RANKING (sorted by |signed_balance|, closest to 0 = best)")
    log("=" * 80)
    log("")
    log(f"  {'Rank':<5} {'Variant':<52} {'signed':>8} {'matdiff':>8} {'C':>3} {'X':>3} {'D':>3} {'mtbC':>4} {'mtbX':>4} {'mtbE':>4} {'ply':>5}")
    log(f"  {'-'*100}")

    ranked = sorted(all_results.items(), key=lambda x: abs(x[1]["summary"]["signed_balance"]))
    for rank, (vname, vdata) in enumerate(ranked, 1):
        s = vdata["summary"]
        log(f"  {rank:<5} {vname:<52} {s['signed_balance']:>+8.4f} {s['avg_mat_diff']:>+8.2f} "
            f"{s['chess_wins']:>3} {s['xq_wins']:>3} {s['draws']:>3} "
            f"{s['mtb_chess']:>4} {s['mtb_xq']:>4} {s['mtb_even']:>4} {s['avg_ply']:>5}")

    log("")
    log(f"  Total time: {time.time() - t_total:.0f}s")
    log(f"  Best balanced: {ranked[0][0]} (signed_balance={ranked[0][1]['summary']['signed_balance']:+.4f})")
    log("")

    # Save JSON (without per-game data for clean summary; full data separate)
    summary_json = {}
    for vname, vdata in all_results.items():
        summary_json[vname] = {
            "variant_dict": vdata["variant_dict"],
            "summary": vdata["summary"],
            "elapsed_s": vdata["elapsed_s"],
        }

    with open(jsonpath, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)
    log(f"  Summary saved: {jsonpath}")

    # Also save full per-game data
    fullpath = outdir / "ab_ablation_full.json"
    with open(fullpath, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"  Full data saved: {fullpath}")

    logf.close()


if __name__ == "__main__":
    main()
