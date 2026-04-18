"""RQ4: AB D2 side balance on nqnbs (no_queen+no_bishop+extra_soldier).

ALL computation in C++:
  - Board / move gen / terminal / hash: C++ via HybridChessEnv(use_cpp=True)
  - AB search: C++ best_move() via hybrid.cpp_engine
  - Parallelism: ProcessPoolExecutor

Also includes material tiebreak analysis for draw games.
"""
import json, time, sys, collections
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Material values for tiebreak
MAT_VAL = {
    'KING': 0, 'QUEEN': 9, 'ROOK': 5, 'BISHOP': 3, 'KNIGHT': 3, 'PAWN': 1,
    'GENERAL': 0, 'CHARIOT': 5, 'CANNON': 3, 'HORSE': 3,
    'ELEPHANT': 2, 'ADVISOR': 2, 'SOLDIER': 1,
}


def play_ab_batch(variant_dict, depth, num_games, seed, max_plies=200):
    """Play AB vs AB games using PURE C++ search. Returns per-game results."""
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

        # Determine winner
        if info.winner == Side.CHESS:
            actual = "Chess"
        elif info.winner == Side.XIANGQI:
            actual = "XQ"
        else:
            actual = "Draw"

        # Material tiebreak for draws
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
            "chess_mat": chess_mat,
            "xq_mat": xq_mat,
            "mat_diff": mat_diff,
            "mat_winner": mat_winner,
            "reason": info.reason if hasattr(info, 'reason') else "",
        })

    return results


def run_parallel(variant_dict, depth, total, workers=6, max_plies=200):
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
        all_games.extend(f.result())
    return all_games


def analyze(games, label):
    n = len(games)
    cw = sum(1 for g in games if g["actual"] == "Chess")
    xw = sum(1 for g in games if g["actual"] == "XQ")
    dr = sum(1 for g in games if g["actual"] == "Draw")
    avg_ply = sum(g["ply"] for g in games) / n if n else 0

    print(f"  {label}")
    print(f"    N={n}  Chess={cw}  XQ={xw}  Draw={dr}  AvgPly={avg_ply:.0f}")

    if dr > 0:
        draws = [g for g in games if g["actual"] == "Draw"]
        mtb_c = sum(1 for g in draws if g["mat_winner"] == "Chess")
        mtb_x = sum(1 for g in draws if g["mat_winner"] == "XQ")
        mtb_e = sum(1 for g in draws if g["mat_winner"] == "Even")
        avg_md = sum(g["mat_diff"] for g in draws) / len(draws)
        print(f"    Draw tiebreak: Chess={mtb_c}  XQ={mtb_x}  Even={mtb_e}  avg_mat_diff={avg_md:+.2f}")

        # Adjusted balance (decisive + tiebreak)
        adj_c = cw + mtb_c
        adj_x = xw + mtb_x
        total_dec = adj_c + adj_x
        if total_dec > 0:
            bal = abs(adj_c - adj_x) / total_dec
            dom = "Chess" if adj_c > adj_x else "XQ"
            print(f"    ADJUSTED: Chess={adj_c}  XQ={adj_x}  Even={mtb_e}  balance={bal:.3f} ({dom})")

    print()


def main():
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except:
        pass

    VARIANTS = [
        ("default", {}),
        ("no_queen+no_bishop+extra_soldier",
         {"no_queen": True, "no_bishop": True, "extra_soldier": True}),
        ("no_queen+no_bishop+one_rook+extra_cannon",
         {"no_queen": True, "no_bishop": True, "one_rook": True, "extra_cannon": True}),
        ("extra_cannon", {"extra_cannon": True}),
    ]

    DEPTH = 2
    N_GAMES = 60
    MAX_PLIES = 200
    WORKERS = 6

    outdir = Path("runs/rq4_nqnbs_ab_test")
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  RQ4: AB D{DEPTH} Side Balance (C++ accelerated)")
    print(f"  N={N_GAMES}, max_plies={MAX_PLIES}, workers={WORKERS}")
    print("=" * 72)
    print()

    all_results = {}
    for vname, vdict in VARIANTS:
        t0 = time.time()
        games = run_parallel(vdict, DEPTH, N_GAMES, WORKERS, MAX_PLIES)
        dt = time.time() - t0
        label = f"AB D{DEPTH} - {vname} ({dt:.0f}s)"
        analyze(games, label)
        all_results[vname] = {
            "games": games,
            "elapsed_s": round(dt, 1),
        }

    # Save
    op = outdir / "ab_balance_results.json"
    with open(op, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved: {op}")


if __name__ == "__main__":
    main()
