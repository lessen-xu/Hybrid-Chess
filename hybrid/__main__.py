# -*- coding: utf-8 -*-
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
                         help="MCTS simulations per move")
    p_train.add_argument("--device", default="auto",
                         help="cpu / cuda / auto")
    p_train.add_argument("--use-cpp", action="store_true",
                         help="Use C++ engine for faster move gen")
    p_train.add_argument("--workers", type=int, default=1,
                         help="Parallel self-play workers")
    p_train.add_argument("--ablation", default="extra_cannon",
                         help="Rule variant: none, extra_cannon, no_queen")
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--output", default="runs/az_run",
                         help="Output directory for checkpoints and logs")

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
