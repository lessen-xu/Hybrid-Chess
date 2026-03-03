"""Analyze game records and metrics for summary enrichment."""
import json, os, csv

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def analyze_game_records(game_dir):
    """Analyze all JSON game records in a directory."""
    if not os.path.isdir(game_dir):
        return
    for f in sorted(os.listdir(game_dir)):
        if not f.endswith(".json"):
            continue
        path = os.path.join(game_dir, f)
        with open(path, "r") as fp:
            data = json.load(fp)
        meta = data.get("meta", {})
        moves = data.get("moves", [])
        states = data.get("states_ascii", [])
        # Count pieces in final state
        final_state = states[-1] if states else ""
        chess_upper = sum(final_state.count(c) for c in "RNBQKP")
        xiangqi_lower = sum(final_state.count(c) for c in "cheagsn")
        print(f"  {f}: result={data['result']}, plies={meta.get('plies')}, "
              f"reason={meta.get('reason')}, a_is_chess={meta.get('a_is_chess')}, "
              f"chess_pieces={chess_upper}, xiangqi_pieces={xiangqi_lower}")


def analyze_metrics(csv_path, label):
    """Analyze a metrics CSV."""
    if not os.path.exists(csv_path):
        print(f"  [not found: {csv_path}]")
        return
    with open(csv_path, "r") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    print(f"  Total iterations: {len(rows)}")
    for row in rows:
        i = row["iter"]
        tl = row.get("total_loss", "?")
        vl = row.get("value_loss", "?")
        rw = row.get("eval_random_w", "?")
        rd = row.get("eval_random_d", "?")
        rl = row.get("eval_random_l", "?")
        aw = row.get("eval_ab_w", "?")
        ad = row.get("eval_ab_d", "?")
        al_ = row.get("eval_ab_l", "?")
        gate = row.get("gate_accepted", row.get("gate", "?"))
        # selfplay diagnostics if present
        sp_decisive = row.get("sp_decisive", "")
        sp_avg_ply = row.get("sp_avg_ply", "")
        sp_draw_ml = row.get("sp_draw_move_limit", "")
        sp_draw_adj = row.get("sp_draw_adjudicated", "")
        sp_mat = row.get("sp_avg_mat_diff", "")
        extra = ""
        if sp_decisive:
            extra = f" | sp: decisive={sp_decisive}, avg_ply={sp_avg_ply}, draw_ml={sp_draw_ml}, adj={sp_draw_adj}, mat={sp_mat}"
        print(f"  iter{i}: loss={tl}, val={vl}, "
              f"vs_rand={rw}W/{rd}D/{rl}L, vs_ab={aw}W/{ad}D/{al_}L, "
              f"gate={gate}{extra}")


def main():
    # 1. Phase 13 smoke test game records
    print("=" * 60)
    print("Phase 13 Smoke Test - Game Records")
    print("=" * 60)
    analyze_game_records(os.path.join(BASE, "runs", "phase13_smoke2", "game_records"))

    print()
    print("Phase 13 Smoke Test - Metrics")
    analyze_metrics(os.path.join(BASE, "runs", "phase13_smoke2", "metrics.csv"), "phase13")

    # 2. local_midscale_run
    print()
    print("=" * 60)
    print("local_midscale_run (15 iterations, no truncation)")
    print("=" * 60)
    analyze_metrics(os.path.join(BASE, "results", "az_runs", "local_midscale_run", "metrics.csv"), "midscale")

    # 3. local_gating_ci_run
    print()
    print("=" * 60)
    print("local_gating_ci_run (20 iterations, CI gating)")
    print("=" * 60)
    analyze_metrics(os.path.join(BASE, "results", "az_runs", "local_gating_ci_run", "metrics.csv"), "ci_run")

    # 4. Phase 10 resign tests
    for run in ["phase10_resign_off", "phase10_resign_on"]:
        print()
        print("=" * 60)
        print(f"{run}")
        print("=" * 60)
        analyze_metrics(os.path.join(BASE, "runs", run, "metrics.csv"), run)

    # 5. Summary stats
    print()
    print("=" * 60)
    print("SUMMARY OBSERVATIONS")
    print("=" * 60)

    # Read midscale metrics for trend
    csv_path = os.path.join(BASE, "results", "az_runs", "local_midscale_run", "metrics.csv")
    with open(csv_path, "r") as fp:
        rows = list(csv.DictReader(fp))
    losses = [float(r["total_loss"]) for r in rows]
    rand_wins = [int(r["eval_random_w"]) for r in rows]
    rand_draws = [int(r["eval_random_d"]) for r in rows]
    rand_losses = [int(r["eval_random_l"]) for r in rows]
    ab_wins = [int(r["eval_ab_w"]) for r in rows]
    ab_draws = [int(r["eval_ab_d"]) for r in rows]

    print(f"  Loss trend: {losses[0]:.3f} -> {losses[-1]:.3f} (delta={losses[-1]-losses[0]:.3f})")
    print(f"  vs Random total: {sum(rand_wins)}W / {sum(rand_draws)}D / {sum(rand_losses)}L")
    print(f"  vs AB total: {sum(ab_wins)}W / {sum(ab_draws)}D")
    print(f"  AB win appeared at iter: {[i for i,w in enumerate(ab_wins) if w > 0]}")
    print(f"  Best vs Random win rate (single iter): {max(rand_wins)}/{max(rand_wins)+min(rand_losses)+min(rand_draws)}")

    # CI run
    csv_path2 = os.path.join(BASE, "results", "az_runs", "local_gating_ci_run", "metrics.csv")
    with open(csv_path2, "r") as fp:
        rows2 = list(csv.DictReader(fp))
    losses2 = [float(r["total_loss"]) for r in rows2]
    ab_wins2 = [int(r["eval_ab_w"]) for r in rows2]
    gates = [r.get("gate_accepted", r.get("gate", "")) for r in rows2]
    accepted = sum(1 for g in gates if g in ("True", "Y"))
    print(f"\n  CI run loss trend: {losses2[0]:.3f} -> {losses2[-1]:.3f}")
    print(f"  CI run gates accepted: {accepted}/{len(gates)}")
    print(f"  CI run AB wins: {[i for i,w in enumerate(ab_wins2) if w > 0]}")


if __name__ == "__main__":
    main()
