"""Hybrid Chess — Unified CLI entry point.

Usage:
    python -m hybrid server  [--port 8000]
    python -m hybrid train   [--iterations 20 --games 100 ...]
    python -m hybrid eval    [--model best_model.pt --vs ab_d2 ...]
"""

from __future__ import annotations
import argparse
import sys


def cmd_server(args):
    """Launch the game server."""
    from hybrid.server import main as server_main
    # Rebuild sys.argv for server's own argparse
    argv = ["hybrid.server"]
    if args.port:
        argv += ["--port", str(args.port)]
    if args.host:
        argv += ["--host", args.host]
    if args.no_browser:
        argv += ["--no-browser"]
    sys.argv = argv
    server_main()


def cmd_train(args):
    """Run AlphaZero iterative training."""
    from hybrid.rl.az_runner import run_iterations, AZIterConfig
    from pathlib import Path

    cfg = AZIterConfig(
        iterations=args.iterations,
        selfplay_games_per_iter=args.games,
        simulations=args.simulations,
        device=args.device,
        use_cpp=args.use_cpp,
        num_workers=args.workers,
        ablation=args.ablation,
        lr=args.lr,
        batch_size=args.batch_size,
        # Network architecture
        res_blocks=args.res_blocks,
        channels=args.channels,
        # Inference server
        use_inference_server=args.use_inference_server,
        inference_batch_size=args.inference_batch_size,
        # Evaluation
        eval_games=args.eval_games,
        eval_simulations=args.eval_simulations,
        gating_simulations=args.gating_simulations,
        # Curriculum / endgame
        curriculum_schedule=args.curriculum,
        endgame_ratio=args.endgame_ratio,
        # Training
        buffer_capacity_states=args.buffer_capacity,
        train_epochs=args.train_epochs,
        seed=args.seed,
    )
    outdir = Path(args.output)
    run_iterations(cfg, outdir)


def cmd_eval(args):
    """Evaluate an agent against a baseline."""
    from hybrid.rl.az_eval import play_match, make_eval_az_agent
    from hybrid.agents.random_agent import RandomAgent
    from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig
    from hybrid.core.env import HybridChessEnv

    # Build AZ agent from checkpoint
    if args.model:
        agent_a = make_eval_az_agent(args.model, simulations=args.simulations,
                                      device=args.device)
        label_a = f"AZ({args.model})"
    else:
        agent_a = AlphaBetaAgent(cfg=SearchConfig(depth=2))
        label_a = "AB(d=2)"

    # Build opponent
    if args.vs == "random":
        agent_b = RandomAgent()
        label_b = "Random"
    elif args.vs.startswith("ab_d"):
        d = int(args.vs.split("d")[1])
        agent_b = AlphaBetaAgent(cfg=SearchConfig(depth=d))
        label_b = f"AB(d={d})"
    else:
        agent_b = RandomAgent()
        label_b = "Random"

    env = HybridChessEnv(max_plies=400)
    stats = play_match(env, agent_a, agent_b, games=args.games)
    print(f"\n  {label_a} vs {label_b}  ({args.games} games)")
    print(f"  W={stats.win_a}  D={stats.draw}  L={stats.win_b}  "
          f"avg_plies={stats.avg_plies:.1f}")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="hybrid-chess",
        description="Hybrid Chess — International Chess vs Chinese Chess",
    )
    parser.add_argument("--version", action="version",
                        version="%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── server ──
    p_server = sub.add_parser("server", help="Launch web UI game server")
    p_server.add_argument("--port", type=int, default=8000)
    p_server.add_argument("--host", type=str, default="127.0.0.1")
    p_server.add_argument("--no-browser", action="store_true")

    # ── train ──
    p_train = sub.add_parser("train", help="Run AlphaZero training")
    p_train.add_argument("--iterations", type=int, default=10)
    p_train.add_argument("--games", type=int, default=50,
                         help="Self-play games per iteration")
    p_train.add_argument("--simulations", type=int, default=50,
                         help="MCTS simulations per move (self-play)")
    p_train.add_argument("--device", default="auto",
                         help="cpu / cuda / auto")
    p_train.add_argument("--use-cpp", action="store_true",
                         help="Use C++ engine for faster move gen")
    p_train.add_argument("--workers", type=int, default=1,
                         help="Parallel self-play workers")
    p_train.add_argument("--ablation", default="none",
                         help="Rule variant(s), comma-separated: "
                              "none, no_queen, xq_queen, chess_palace, "
                              "knight_block, no_promotion, extra_cannon, "
                              "no_bishop, extra_soldier, one_rook, "
                              "no_flying_general, remove_pawn, no_queen_promo")
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--output", default="runs/az_run",
                         help="Output directory for checkpoints and logs")
    # Network architecture
    p_train.add_argument("--res-blocks", type=int, default=3,
                         help="Number of residual blocks (default: 3)")
    p_train.add_argument("--channels", type=int, default=64,
                         help="Conv channel width (default: 64)")
    # Inference server (GPU batching)
    p_train.add_argument("--use-inference-server", action="store_true",
                         help="Enable GPU batch inference server for parallel self-play")
    p_train.add_argument("--inference-batch-size", type=int, default=32,
                         help="Max batch size for inference server")
    # Evaluation
    p_train.add_argument("--eval-games", type=int, default=20,
                         help="Games per evaluation round")
    p_train.add_argument("--eval-simulations", type=int, default=200,
                         help="MCTS sims for evaluation (decoupled from self-play)")
    p_train.add_argument("--gating-simulations", type=int, default=20,
                         help="MCTS sims for gating matches")
    # Curriculum / endgame
    p_train.add_argument("--curriculum", default="none",
                         choices=["none", "3phase", "3phase_v2"],
                         help="Curriculum schedule")
    p_train.add_argument("--endgame-ratio", type=float, default=0.0,
                         help="Fraction of self-play starting from endgame")
    # Training
    p_train.add_argument("--buffer-capacity", type=int, default=50_000,
                         help="Replay buffer capacity (states)")
    p_train.add_argument("--train-epochs", type=int, default=1,
                         help="Training epochs per iteration")
    p_train.add_argument("--seed", type=int, default=0)

    # ── eval ──
    p_eval = sub.add_parser("eval", help="Evaluate agent vs baseline")
    p_eval.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint (.pt)")
    p_eval.add_argument("--vs", default="random",
                        help="Opponent: random, ab_d1, ab_d2, ab_d4")
    p_eval.add_argument("--games", type=int, default=20)
    p_eval.add_argument("--simulations", type=int, default=100)
    p_eval.add_argument("--device", default="auto")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "server": cmd_server,
        "train": cmd_train,
        "eval": cmd_eval,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
