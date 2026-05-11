"""RQ4 AB depth-2 rule-reform scan, fixed_v1 rerun.

Same logic as scripts/rq4_rule_reform_ab.py, but:
  * Accepts CLI args (workers / games / depth / outdir / max_plies).
  * Default workers raised from 6 to 8 (AB workers do no NN inference,
    so they fit comfortably on an 8-physical-core box).
  * Default outdir is runs/fixed_v1/rq4_rule_reform_ab so the result
    sits next to the rest of the fixed_v1 retrain artifacts.

Usage::

    python -m scripts.rq4_rule_reform_ab_fixed_v1
    python -m scripts.rq4_rule_reform_ab_fixed_v1 --workers 8 --n-games 40
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.rq4_rule_reform_ab import (  # reuse logic
    VARIANTS, play_ab_batch, summarize,
)


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
                    default="runs/fixed_v1/rq4_rule_reform_ab")
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
    log(f"  RQ4: RULE REFORM AB D{args.depth} Balance Test (fixed_v1)")
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
