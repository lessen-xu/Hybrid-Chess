# -*- coding: utf-8 -*-
"""AlphaZero-Mini iterative training CLI.

Usage:
  # Quick smoke test (~minutes)
  python -m scripts.train_az_iter --iterations 3 --selfplay-games-per-iter 5 \
      --simulations 20 --train-epochs 1 --eval-games 6

  # Full training (~hours)
  python -m scripts.train_az_iter --iterations 20 --selfplay-games-per-iter 50 \
      --simulations 50 --train-epochs 3 --eval-games 20

Output:
  results/az_runs/iter_YYYYMMDD_HHMMSS/
    config.json, metrics.csv, best_model.pt, ckpt_iter*.pt, replay_iter*.npz
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from hybrid.core.config import MAX_PLIES
from hybrid.rl.az_runner import AZIterConfig, run_iterations


def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero-Mini Iterative Training Runner"
    )
    parser.add_argument("--iterations", type=int, default=10,
                        help="number of iterations (default: 10)")
    parser.add_argument("--selfplay-games-per-iter", type=int, default=50,
                        help="self-play games per iteration (default: 50)")
    parser.add_argument("--simulations", type=int, default=50,
                        help="MCTS simulations per move (default: 50)")
    parser.add_argument("--selfplay-max-ply", type=int, default=150,
                        help="self-play max ply limit (default: 150)")
    parser.add_argument("--selfplay-move-limit-value-mode", type=str, default="penalty",
                        choices=["zero", "hard", "soft", "penalty"],
                        help="value assignment at move limit: zero/hard/soft/penalty (default: penalty)")
    parser.add_argument("--selfplay-move-limit-value-scale", type=float, default=4.0,
                        help="tanh scale for soft mode (default: 4.0)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="training batch size (default: 256)")
    parser.add_argument("--train-epochs", type=int, default=1,
                        help="training epochs per iteration (default: 1)")
    parser.add_argument("--buffer-capacity", type=int, default=50_000,
                        help="replay buffer max samples (default: 50000)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="gradient clipping threshold (default: 1.0)")
    parser.add_argument("--eval-games", type=int, default=20,
                        help="evaluation games per round (default: 20)")
    parser.add_argument("--eval-simulations", type=int, default=200,
                        help="MCTS sims for eval/recording (decoupled from "
                             "self-play --simulations). Higher = stronger play but "
                             "slower eval. (default: 200)")
    parser.add_argument("--eval-interval", type=int, default=5,
                        help="eval every N iters (0=every iter, default: 5)")
    # Gating
    parser.add_argument("--gating-threshold", type=float, default=0.55,
                        help="gating win-rate threshold (default: 0.55)")
    parser.add_argument("--gating-min-games", type=int, default=10,
                        help="gating min games (default: 10)")
    parser.add_argument("--gating-max-games", type=int, default=80,
                        help="gating max games (default: 80)")
    parser.add_argument("--gating-step-games", type=int, default=10,
                        help="gating batch increment (default: 10)")
    parser.add_argument("--gating-confidence", type=float, default=0.95,
                        help="Wilson CI confidence level (default: 0.95)")
    parser.add_argument("--gating-simulations", type=int, default=20,
                        help="gating MCTS sims (default: 20)")
    parser.add_argument("--gating-use-score", type=int, default=1,
                        help="1=score_ci(W,D,L), 0=wilson_ci(W,L) (default: 1)")
    # Resign
    parser.add_argument("--resign-enabled", type=int, default=1,
                        help="enable resign (1=on, 0=off, default: 1)")
    parser.add_argument("--resign-threshold", type=float, default=-0.95,
                        help="resign threshold (default: -0.95)")
    parser.add_argument("--resign-min-ply", type=int, default=40,
                        help="resign min ply (default: 40)")
    parser.add_argument("--resign-patience", type=int, default=3,
                        help="resign patience (default: 3)")
    parser.add_argument("--draw-adjudicate-enabled", type=int, default=1,
                        help="enable draw adjudication (1=on, 0=off, default: 1)")
    parser.add_argument("--draw-adjudicate-min-ply", type=int, default=60,
                        help="draw adjudication min ply (default: 60)")
    parser.add_argument("--draw-adjudicate-patience", type=int, default=15,
                        help="draw adjudication patience (default: 15)")
    parser.add_argument("--draw-adjudicate-value-abs-thr", type=float, default=0.08,
                        help="draw adjudication |root_value| threshold (default: 0.08)")
    parser.add_argument("--temp-cutoff", type=int, default=20,
                        help="T=1.0 for first N plies, then T=0 (default: 20)")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3,
                        help="Dirichlet noise alpha (default: 0.3)")
    parser.add_argument("--dirichlet-eps", type=float, default=0.25,
                        help="Dirichlet noise mixing ratio (default: 0.25)")
    parser.add_argument("--device", type=str, default="auto",
                        help="device: auto/cuda/cpu (default: auto)")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed (default: 0)")
    parser.add_argument("--ablation", type=str, default="extra_cannon",
                        help="ablation experiment name (default: extra_cannon)")
    parser.add_argument("--outdir", type=str, default="",
                        help="output directory (default: results/az_runs/iter_YYYYMMDD_HHMMSS/)")
    # Parallel self-play
    parser.add_argument("--num-workers", type=int, default=1,
                        help="self-play worker processes (1=sequential, default: 1)")
    parser.add_argument("--use-inference-server", action="store_true",
                        help="enable GPU inference server (Mode B)")
    parser.add_argument("--inference-batch-size", type=int, default=32,
                        help="inference server max batch size (default: 32)")
    parser.add_argument("--inference-timeout-ms", type=float, default=5.0,
                        help="inference server batch timeout ms (default: 5.0)")
    parser.add_argument("--inference-device", type=str, default="auto",
                        help="inference server device: auto/cuda/cpu (default: auto)")
    # Endgame curriculum
    parser.add_argument("--endgame-ratio", type=float, default=0.0,
                        help="fraction of self-play games starting from endgame positions "
                             "(0.0=none, 1.0=all, default: 0.0)")
    # Curriculum annealing
    parser.add_argument("--curriculum-schedule", type=str, default="none",
                        choices=["none", "3phase", "3phase_v2"],
                        help="curriculum annealing: none, 3phase, or 3phase_v2 "
                             "(v2: endgame anchor 80%%→40%%→15%%, gating always off). "
                             "Default: none")
    parser.add_argument("--disable-gating", type=int, default=0,
                        help="1=unconditionally accept new model, skip gating "
                             "(default: 0)")
    parser.add_argument("--use-cpp", action="store_true",
                        help="use C++ engine for game logic (legal moves, apply, terminal)")
    args = parser.parse_args()

    cfg = AZIterConfig(
        iterations=args.iterations,
        selfplay_games_per_iter=args.selfplay_games_per_iter,
        simulations=args.simulations,
        selfplay_max_ply=args.selfplay_max_ply,
        selfplay_move_limit_value_mode=args.selfplay_move_limit_value_mode,
        selfplay_move_limit_value_scale=args.selfplay_move_limit_value_scale,
        batch_size=args.batch_size,
        train_epochs=args.train_epochs,
        buffer_capacity_states=args.buffer_capacity,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_games=args.eval_games,
        eval_simulations=args.eval_simulations,
        eval_interval=args.eval_interval,
        gating_threshold=args.gating_threshold,
        gating_min_games=args.gating_min_games,
        gating_max_games=args.gating_max_games,
        gating_step_games=args.gating_step_games,
        gating_confidence=args.gating_confidence,
        gating_simulations=args.gating_simulations,
        gating_use_score=bool(args.gating_use_score),
        resign_enabled=bool(args.resign_enabled),
        resign_threshold=args.resign_threshold,
        resign_min_ply=args.resign_min_ply,
        resign_patience=args.resign_patience,
        draw_adjudicate_enabled=bool(args.draw_adjudicate_enabled),
        draw_adjudicate_min_ply=args.draw_adjudicate_min_ply,
        draw_adjudicate_patience=args.draw_adjudicate_patience,
        draw_adjudicate_value_abs_thr=args.draw_adjudicate_value_abs_thr,
        temp_cutoff=args.temp_cutoff,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_eps=args.dirichlet_eps,
        device=args.device,
        seed=args.seed,
        ablation=args.ablation,
        num_workers=args.num_workers,
        use_inference_server=args.use_inference_server,
        inference_batch_size=args.inference_batch_size,
        inference_timeout_ms=args.inference_timeout_ms,
        inference_device=args.inference_device,
        endgame_ratio=args.endgame_ratio,
        curriculum_schedule=args.curriculum_schedule,
        disable_gating=bool(args.disable_gating),
        use_cpp=args.use_cpp,
    )

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path(f"results/az_runs/iter_{ts}")

    run_iterations(cfg, outdir)


if __name__ == "__main__":
    main()
