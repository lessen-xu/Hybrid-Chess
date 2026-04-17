"""RQ4 Draw Diagnosis: Are draws genuine equilibria or meaningless shuffling?

For each game, tracks:
- Material balance over time
- Last capture ply (how long since any piece was taken)
- Position repetition rate (unique positions / total plies)
- Material tiebreak winner (who has more material at game end)

Uses max_plies=3000 to give games maximum room to resolve.
"""
import json, time, sys, collections
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Material values (standard relative values)
MATERIAL_VALUES = {
    # Chess pieces
    'KING': 0, 'QUEEN': 9, 'ROOK': 5, 'BISHOP': 3, 'KNIGHT': 3, 'PAWN': 1,
    # Xiangqi pieces
    'GENERAL': 0, 'CHARIOT': 5, 'CANNON': 3, 'HORSE': 3,
    'ELEPHANT': 2, 'ADVISOR': 2, 'SOLDIER': 1,
}

def count_material(board, Side):
    """Count material for each side."""
    chess_mat = 0
    xq_mat = 0
    for x, y, p in board.iter_pieces():
        val = MATERIAL_VALUES.get(p.kind.name, 0)
        if p.side == Side.CHESS:
            chess_mat += val
        else:
            xq_mat += val
    return chess_mat, xq_mat


def play_diagnostic_batch(variant_dict, agent_type, num_games, seed, max_plies=3000):
    """Play games with rich per-game diagnostics."""
    from hybrid.core.config import VariantConfig
    from hybrid.core.env import HybridChessEnv
    from hybrid.core.types import Side

    vcfg = VariantConfig(**variant_dict)
    env = HybridChessEnv(use_cpp=True, max_plies=max_plies, variant=vcfg)

    if agent_type == "random":
        from hybrid.agents.random_agent import RandomAgent
    else:
        from hybrid.agents.greedy_agent import GreedyAgent

    game_results = []

    for i in range(num_games):
        state = env.reset()
        if agent_type == "random":
            ac = RandomAgent(seed=seed + i * 2)
            ax = RandomAgent(seed=seed + i * 2 + 1)
        else:
            ac = GreedyAgent()
            ax = GreedyAgent()
        agents = {Side.CHESS: ac, Side.XIANGQI: ax}

        # Track diagnostics
        last_capture_ply = 0
        piece_count_prev = sum(1 for _ in state.board.iter_pieces())
        positions_seen = collections.Counter()
        material_snapshots = []  # (ply, chess_mat, xq_mat) every 50 plies

        while True:
            # Track positions
            try:
                pos_key = env._cpp_board.board_hash(
                    env._cpp_side if env.use_cpp else 0
                )
            except:
                pos_key = state.ply  # fallback
            positions_seen[pos_key] += 1

            # Material snapshot every 100 plies
            if state.ply % 100 == 0:
                cm, xm = count_material(state.board, Side)
                material_snapshots.append((state.ply, cm, xm))

            legal = env.legal_moves()
            if not legal:
                break
            mv = agents[state.side_to_move].select_move(state, legal)
            state, _, done, info = env.step(mv)

            # Detect captures
            piece_count_now = sum(1 for _ in state.board.iter_pieces())
            if piece_count_now < piece_count_prev:
                last_capture_ply = state.ply
            piece_count_prev = piece_count_now

            if done:
                break

        # Final material
        cm_final, xm_final = count_material(state.board, Side)
        mat_diff = cm_final - xm_final  # positive = Chess advantage

        # Material tiebreak winner
        if mat_diff > 0:
            mat_winner = "Chess"
        elif mat_diff < 0:
            mat_winner = "XQ"
        else:
            mat_winner = "Even"

        # Actual game result
        if info.winner == Side.CHESS:
            actual = "Chess"
        elif info.winner == Side.XIANGQI:
            actual = "XQ"
        else:
            actual = "Draw"

        # Shuffle detection: moves since last capture
        moves_without_capture = state.ply - last_capture_ply
        unique_positions = len(positions_seen)
        repetition_rate = 1.0 - (unique_positions / max(state.ply, 1))

        game_results.append({
            "actual": actual,
            "ply": state.ply,
            "chess_mat": cm_final,
            "xq_mat": xm_final,
            "mat_diff": mat_diff,
            "mat_winner": mat_winner,
            "last_capture_ply": last_capture_ply,
            "moves_no_capture": moves_without_capture,
            "unique_positions": unique_positions,
            "repetition_rate": round(repetition_rate, 3),
            "material_curve": material_snapshots,
        })

    return game_results


def run_parallel(variant_dict, agent_type, total, workers=6, seed=42, max_plies=3000):
    gpw = total // workers
    rem = total % workers
    futs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for w in range(workers):
            n = gpw + (1 if w < rem else 0)
            if n:
                futs.append(pool.submit(
                    play_diagnostic_batch, variant_dict, agent_type,
                    n, seed + w * 10000, max_plies
                ))
    all_games = []
    for f in futs:
        all_games.extend(f.result())
    return all_games


def analyze(games, label):
    """Print diagnostic summary for a set of games."""
    n = len(games)
    decisive = [g for g in games if g["actual"] != "Draw"]
    draws = [g for g in games if g["actual"] == "Draw"]

    chess_wins = sum(1 for g in games if g["actual"] == "Chess")
    xq_wins = sum(1 for g in games if g["actual"] == "XQ")

    print(f"  {label}")
    print(f"    Games: {n}  Chess={chess_wins}  XQ={xq_wins}  Draw={len(draws)}")

    if draws:
        # Material tiebreak for draws
        mtb_chess = sum(1 for g in draws if g["mat_winner"] == "Chess")
        mtb_xq = sum(1 for g in draws if g["mat_winner"] == "XQ")
        mtb_even = sum(1 for g in draws if g["mat_winner"] == "Even")
        print(f"    Draw material tiebreak: Chess={mtb_chess}  XQ={mtb_xq}  Even={mtb_even}")

        # Adjusted results (decisive + material tiebreak)
        adj_chess = chess_wins + mtb_chess
        adj_xq = xq_wins + mtb_xq
        adj_even = mtb_even
        total_dec = adj_chess + adj_xq
        if total_dec > 0:
            balance = abs(adj_chess - adj_xq) / total_dec
            dominant = "Chess" if adj_chess > adj_xq else "XQ"
            print(f"    ADJUSTED balance: Chess={adj_chess}  XQ={adj_xq}  Even={adj_even}  "
                  f"bal={balance:.3f} ({dominant})")

        # Shuffle analysis
        avg_no_cap = sum(g["moves_no_capture"] for g in draws) / len(draws)
        avg_rep = sum(g["repetition_rate"] for g in draws) / len(draws)
        avg_ply = sum(g["ply"] for g in draws) / len(draws)
        max_ply = max(g["ply"] for g in draws)

        print(f"    Draw diagnostics:")
        print(f"      Avg ply: {avg_ply:.0f}  Max ply: {max_ply}")
        print(f"      Avg moves without capture: {avg_no_cap:.0f}")
        print(f"      Avg repetition rate: {avg_rep:.1%}")

        # Categorize draws
        shuffle_draws = sum(1 for g in draws if g["moves_no_capture"] > 200)
        timeout_draws = sum(1 for g in draws if g["ply"] >= 2990)
        rep_draws = sum(1 for g in draws if g["repetition_rate"] > 0.5)
        print(f"      Shuffle draws (>200 moves no capture): {shuffle_draws}/{len(draws)}")
        print(f"      Timeout draws (hit max_plies): {timeout_draws}/{len(draws)}")
        print(f"      High-repetition draws (>50%% repeated): {rep_draws}/{len(draws)}")

    print()


def main():
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except:
        pass

    VARIANTS = [
        ("default", {}),
        ("no_queen+no_bishop+one_rook+extra_cannon",
         {"no_queen": True, "no_bishop": True, "one_rook": True, "extra_cannon": True}),
        ("no_queen+no_bishop+one_rook",
         {"no_queen": True, "no_bishop": True, "one_rook": True}),
        ("no_queen+no_bishop+extra_soldier",
         {"no_queen": True, "no_bishop": True, "extra_soldier": True}),
    ]

    MAX_PLIES = 3000
    N_RANDOM = 200
    N_GREEDY = 200

    outdir = Path("runs/rq4_draw_diagnosis")
    outdir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    print("=" * 72)
    print("  RQ4 Draw Diagnosis — Are draws equilibria or shuffling?")
    print(f"  max_plies={MAX_PLIES}, Random(N={N_RANDOM}), Greedy(N={N_GREEDY})")
    print("=" * 72)
    print()

    for vname, vdict in VARIANTS:
        print(f"--- {vname} ---")

        for agent in ["random", "greedy"]:
            n = N_RANDOM if agent == "random" else N_GREEDY
            t0 = time.time()
            games = run_parallel(vdict, agent, n, 6, 42, MAX_PLIES)
            dt = time.time() - t0
            label = f"{agent.capitalize()} (N={n}, {dt:.0f}s)"
            analyze(games, label)

            key = f"{vname}__{agent}"
            all_results[key] = {
                "variant": vname,
                "agent": agent,
                "games": games,
                "elapsed_s": round(dt, 1),
            }

    # Save (without material curves to keep file manageable)
    save_data = {}
    for k, v in all_results.items():
        save_games = []
        for g in v["games"]:
            sg = dict(g)
            sg.pop("material_curve", None)  # too verbose
            save_games.append(sg)
        save_data[k] = {"variant": v["variant"], "agent": v["agent"],
                        "games": save_games, "elapsed_s": v["elapsed_s"]}

    op = outdir / "draw_diagnosis_results.json"
    with open(op, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {op}")


if __name__ == "__main__":
    main()
