"""AlphaZero-Mini iterative training runner.

Each iteration:
  1. Self-play (Dirichlet noise + temperature) → generate training data
  2. Train (policy cross-entropy + value MSE on replay buffer)
  3. Gating (candidate vs best, adaptive Wilson/score CI) → update best model
  4. Evaluate (vs Random / vs AB depth=1) → measure absolute strength
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import csv
import json
import os
import time

import numpy as np
import torch

from hybrid.core.env import HybridChessEnv
from hybrid.core.types import Side
from hybrid.core.config import MAX_PLIES
from hybrid.agents.alphazero_stub import (
    AlphaZeroMiniAgent,
    MCTSConfig,
    TorchPolicyValueModel,
)
from hybrid.agents.random_agent import RandomAgent
from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig
from hybrid.rl.az_network import PolicyValueNet
from hybrid.rl.az_selfplay import self_play_game, SelfPlayConfig, GameRecord
from hybrid.rl.az_replay import ReplayBuffer
from hybrid.rl.az_train import train_one_epoch
from hybrid.rl.az_eval import play_match, make_eval_az_agent, MatchStats, wilson_ci, score_ci
from hybrid.rl.endgame_spawner import generate_endgame_board
# Configuration

@dataclass
class AZIterConfig:
    """Full configuration for iterative AlphaZero training."""
    # Iteration control
    iterations: int = 10
    selfplay_games_per_iter: int = 50
    simulations: int = 50
    selfplay_max_ply: int = 150
    selfplay_move_limit_value_mode: str = "penalty"
    selfplay_move_limit_value_scale: float = 4.0

    # Network architecture
    res_blocks: int = 3
    channels: int = 64

    # Training
    batch_size: int = 256
    train_epochs: int = 1
    buffer_capacity_states: int = 50_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Evaluation
    eval_games: int = 20
    eval_interval: int = 5        # Eval every N iters (0 = every iter)
    eval_record_games: int = 2
    eval_simulations: int = 200   # MCTS sims for eval / recording (decoupled from self-play)

    # Self-play exploration
    temp_cutoff: int = 20         # T=1.0 for first N plies, then T=0
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25

    # Resign (self-play only)
    resign_enabled: bool = True
    resign_threshold: float = -0.95
    resign_min_ply: int = 40
    resign_patience: int = 3
    # Draw adjudication (self-play only)
    draw_adjudicate_enabled: bool = True
    draw_adjudicate_min_ply: int = 60
    draw_adjudicate_patience: int = 15
    draw_adjudicate_value_abs_thr: float = 0.08

    # Gating (adaptive Wilson CI)
    gating_min_games: int = 10
    gating_max_games: int = 80
    gating_step_games: int = 10
    gating_threshold: float = 0.55
    gating_confidence: float = 0.95
    gating_simulations: int = 20
    gating_use_score: bool = True  # True=score_ci(W,D,L), False=wilson_ci(W,L)

    # Endgame curriculum
    endgame_ratio: float = 0.0   # fraction of self-play games starting from endgame positions

    # Curriculum annealing
    curriculum_schedule: str = "none"   # "none" or "3phase"
    disable_gating: bool = False        # unconditionally accept new model (skip gating)

    # Environment
    device: str = "auto"
    seed: int = 0
    ablation: str = "extra_cannon"

    # C++ engine
    use_cpp: bool = False

    # Parallel self-play
    num_workers: int = 1
    use_inference_server: bool = False
    inference_batch_size: int = 32
    inference_timeout_ms: float = 5.0
    inference_device: str = "auto"

    # Optional logging (lazy import, no hard dependency)
    use_wandb: bool = False
    use_tensorboard: bool = False

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}
# Helpers

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _split_games_evenly(total_games: int, num_workers: int) -> List[int]:
    """Distribute total_games evenly across num_workers."""
    if total_games < 0:
        raise ValueError("total_games must be >= 0")
    if num_workers <= 0:
        raise ValueError("num_workers must be > 0")
    base, rem = divmod(total_games, num_workers)
    return [base + (1 if i < rem else 0) for i in range(num_workers)]


def _apply_ablation(ablation: str) -> "VariantConfig":
    """Return a VariantConfig from an ablation string (comma-separated presets)."""
    from hybrid.core.config import VariantConfig

    if ablation == "none":
        return VariantConfig()

    _PRESET_TO_FIELD = {
        'extra_cannon':      {'extra_cannon': True},
        'no_queen':          {'no_queen': True},
        'no_bishop':         {'no_bishop': True},
        'extra_soldier':     {'extra_soldier': True},
        'one_rook':          {'one_rook': True},
        'no_flying_general': {'flying_general': False},
        'remove_pawn':       {'remove_extra_pawn': True},
        'no_queen_promo':    {'no_queen_promotion': True},
        # Rule reforms
        'no_promotion':      {'no_promotion': True},
        'chess_palace':      {'chess_palace': True},
        'knight_block':      {'knight_block': True},
        # Piece additions
        'xq_queen':          {'xq_queen': True},
    }

    variant_fields: dict = {}
    for part in (p.strip() for p in ablation.split(',')):
        if part in _PRESET_TO_FIELD:
            variant_fields.update(_PRESET_TO_FIELD[part])
        else:
            print(f"[WARNING] Unknown ablation: {part!r}, skipping")

    return VariantConfig(**variant_fields)


def _save_checkpoint(
    net: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    cfg: AZIterConfig,
    iteration: int,
    global_step: int,
    path: str,
) -> None:
    """Save checkpoint with model, optimizer, config, and training progress."""
    torch.save({
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.to_dict(),
        "iteration": iteration,
        "global_step": global_step,
        "arch": {"res_blocks": cfg.res_blocks, "channels": cfg.channels},
    }, path)


def _load_model_weights(net: PolicyValueNet, path: str, device: torch.device) -> None:
    """Load model weights from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(ckpt["model"])


def build_net_from_checkpoint(
    path: str, device: str = "cpu",
    fallback_res_blocks: int = 3, fallback_channels: int = 64,
) -> PolicyValueNet:
    """Build a PolicyValueNet with architecture matching the checkpoint.

    Reads 'arch' key from the checkpoint if present; otherwise falls back
    to the given defaults (backward-compatible with older checkpoints).
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    arch = ckpt.get("arch", {})
    res_blocks = arch.get("res_blocks", fallback_res_blocks)
    channels = arch.get("channels", fallback_channels)
    net = PolicyValueNet(num_res_blocks=res_blocks, channels=channels)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net
# Self-play diagnostics

def _aggregate_game_records(records: List[GameRecord]) -> Dict[str, Any]:
    """Aggregate GameRecords from one iteration into summary statistics."""
    if not records:
        return {}

    n = len(records)
    plies = [r.ply_count for r in records]
    plies_sorted = sorted(plies)

    draw_move_limit = sum(1 for r in records if r.termination_reason == "Max plies reached")
    draw_threefold = sum(1 for r in records if r.termination_reason == "Threefold repetition")
    draw_stalemate = sum(1 for r in records if r.termination_reason == "Stalemate (draw by rule)")
    draw_adjudicated = sum(1 for r in records if r.termination_reason == "Adjudicated draw")
    decisive = sum(1 for r in records if r.result != "draw")
    resign_count = sum(1 for r in records if r.resigned)

    # --- Faction outcome telemetry ---
    chess_wins = sum(1 for r in records if r.winner_side == "chess")
    xiangqi_wins = sum(1 for r in records if r.winner_side == "xiangqi")
    draws = n - chess_wins - xiangqi_wins

    # --- Branching factor telemetry ---
    # Even plies (0,2,4,...) = Chess moves, odd plies (1,3,5,...) = Xiangqi moves
    chess_legal, xiangqi_legal = [], []
    for r in records:
        for ply_i, count in enumerate(r.legal_move_counts):
            if ply_i % 2 == 0:
                chess_legal.append(count)
            else:
                xiangqi_legal.append(count)

    avg_legal_chess = round(sum(chess_legal) / max(len(chess_legal), 1), 2)
    avg_legal_xiangqi = round(sum(xiangqi_legal) / max(len(xiangqi_legal), 1), 2)

    mat_diffs = [r.material_diff for r in records]
    rootv_mins = [r.rootv_min for r in records]
    rootv_mins_sorted = sorted(rootv_mins)
    total_low_rootv_steps = sum(r.low_rootv_steps for r in records)
    total_rootv_steps = sum(r.rootv_steps for r in records)

    adjudicated_plies = [r.ply_count for r in records if r.termination_reason == "Adjudicated draw"]

    # --- Per-piece survival diagnostics ---
    from hybrid.rl.az_selfplay import INITIAL_PIECES
    piece_keys = sorted(INITIAL_PIECES.keys())
    piece_survived = {}
    piece_lost = {}
    for k in piece_keys:
        init = INITIAL_PIECES[k]
        survived_total = sum(r.piece_census.get(k, 0) for r in records if r.piece_census)
        n_with_census = sum(1 for r in records if r.piece_census)
        if n_with_census > 0:
            avg_survived = survived_total / n_with_census
            survival_rate = avg_survived / init if init > 0 else 0.0
            piece_survived[f"surv_{k}"] = round(survival_rate, 3)
            piece_lost[f"lost_{k}"] = round(init - avg_survived, 2)

    # Chess vs XQ total material at game end
    chess_mat_keys = [k for k in piece_keys if k.startswith('chess_')]
    xq_mat_keys = [k for k in piece_keys if k.startswith('xiangqi_')]
    MAT_VAL = {'KING':0,'QUEEN':9,'ROOK':5,'BISHOP':3,'KNIGHT':3,'PAWN':1,
               'GENERAL':0,'ADVISOR':2,'ELEPHANT':2,'HORSE':3,'CHARIOT':5,'CANNON':4.5,'SOLDIER':1}
    chess_end_mat = []
    xq_end_mat = []
    for r in records:
        if not r.piece_census:
            continue
        cm = sum(r.piece_census.get(k, 0) * MAT_VAL.get(k.split('_',1)[1], 0) for k in chess_mat_keys)
        xm = sum(r.piece_census.get(k, 0) * MAT_VAL.get(k.split('_',1)[1], 0) for k in xq_mat_keys)
        chess_end_mat.append(cm)
        xq_end_mat.append(xm)

    result = {
        "sp_games": n,
        "sp_decisive": decisive,
        "sp_draw_move_limit": draw_move_limit,
        "sp_draw_threefold": draw_threefold,
        "sp_draw_stalemate": draw_stalemate,
        "sp_draw_adjudicated": draw_adjudicated,
        "sp_draw_adjudicated_rate": round(draw_adjudicated / n, 4),
        "sp_adjudicate_avg_ply": round(sum(adjudicated_plies) / len(adjudicated_plies), 1)
        if adjudicated_plies else 0.0,
        "sp_resign_count": resign_count,
        "sp_avg_ply": round(sum(plies) / n, 1),
        "sp_p50_ply": plies_sorted[n // 2],
        "sp_p90_ply": plies_sorted[min(int(n * 0.9), n - 1)],
        "sp_avg_mat_diff": round(sum(mat_diffs) / n, 2),
        "sp_rootv_min_mean": round(sum(rootv_mins) / n, 3),
        "sp_rootv_min_p10": round(rootv_mins_sorted[min(int(n * 0.1), n - 1)], 3),
        "sp_low_rootv_steps_sum": total_low_rootv_steps,
        "sp_low_rootv_steps_rate": round(
            total_low_rootv_steps / max(total_rootv_steps, 1), 4
        ),
        # --- Faction / branching telemetry ---
        "sp_chess_wins": chess_wins,
        "sp_xiangqi_wins": xiangqi_wins,
        "sp_draws": draws,
        "sp_avg_legal_chess": avg_legal_chess,
        "sp_avg_legal_xiangqi": avg_legal_xiangqi,
        # --- Per-piece survival ---
        "sp_chess_end_mat": round(sum(chess_end_mat) / max(len(chess_end_mat), 1), 2),
        "sp_xq_end_mat": round(sum(xq_end_mat) / max(len(xq_end_mat), 1), 2),
    }
    result.update(piece_survived)
    result.update(piece_lost)
    return result
# CSV logging

CSV_COLUMNS = [
    "iter", "samples_added", "buffer_size",
    "policy_loss", "value_loss", "total_loss",
    "eval_random_w", "eval_random_d", "eval_random_l",
    "eval_ab_w", "eval_ab_d", "eval_ab_l",
    "gating_games_used", "gating_w", "gating_d", "gating_l",
    "gating_p_hat", "gating_ci_low", "gating_ci_high",
    "gating_score",
    "gate",
    "lr", "simulations", "selfplay_max_ply",
    "selfplay_move_limit_value_mode", "selfplay_move_limit_value_scale",
    "draw_adjudicate_enabled", "draw_adjudicate_min_ply",
    "draw_adjudicate_patience", "draw_adjudicate_value_abs_thr",
    "endgame_ratio",
    "selfplay_seconds", "samples_per_sec",
    "sp_games", "sp_decisive", "sp_draw_move_limit",
    "sp_draw_threefold", "sp_draw_stalemate", "sp_draw_adjudicated",
    "sp_draw_adjudicated_rate", "sp_adjudicate_avg_ply",
    "sp_resign_count",
    "sp_avg_ply", "sp_p50_ply", "sp_p90_ply",
    "sp_avg_mat_diff",
    "sp_rootv_min_mean", "sp_rootv_min_p10",
    "sp_low_rootv_steps_sum", "sp_low_rootv_steps_rate",
    # --- Faction / branching telemetry ---
    "sp_chess_wins", "sp_xiangqi_wins", "sp_draws",
    "sp_avg_legal_chess", "sp_avg_legal_xiangqi",
    # --- Per-piece end material ---
    "sp_chess_end_mat", "sp_xq_end_mat",
    # --- Survival rates ---
    "surv_chess_QUEEN", "surv_chess_ROOK", "surv_chess_BISHOP",
    "surv_chess_KNIGHT", "surv_chess_PAWN",
    "surv_xiangqi_CHARIOT", "surv_xiangqi_CANNON", "surv_xiangqi_HORSE",
    "surv_xiangqi_ELEPHANT", "surv_xiangqi_ADVISOR", "surv_xiangqi_SOLDIER",
    # --- Avg pieces lost ---
    "lost_chess_QUEEN", "lost_chess_ROOK", "lost_chess_BISHOP",
    "lost_chess_KNIGHT", "lost_chess_PAWN",
    "lost_xiangqi_CHARIOT", "lost_xiangqi_CANNON", "lost_xiangqi_HORSE",
    "lost_xiangqi_ELEPHANT", "lost_xiangqi_ADVISOR", "lost_xiangqi_SOLDIER",
]


def _init_csv(path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def _append_csv(path: str, row: dict) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)

_wandb_run = None          # cached wandb.run
_tb_writer = None          # cached SummaryWriter

def _log_metrics(cfg: AZIterConfig, row: dict, step: int) -> None:
    """Log metrics to WandB / TensorBoard if enabled."""
    global _wandb_run, _tb_writer

    if cfg.use_wandb:
        if _wandb_run is None:
            try:
                import wandb
                _wandb_run = wandb.run or wandb.init(
                    project="hybrid-chess", config=cfg.to_dict()
                )
            except ImportError:
                print("[Runner] WARNING: wandb not installed, disabling --use-wandb")
                cfg.use_wandb = False
                return
        # Log only numeric values
        _wandb_run.log({k: v for k, v in row.items()
                        if isinstance(v, (int, float))}, step=step)

    if cfg.use_tensorboard:
        if _tb_writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                _tb_writer = SummaryWriter()
            except ImportError:
                print("[Runner] WARNING: tensorboard not installed, disabling")
                cfg.use_tensorboard = False
                return
        for k, v in row.items():
            if isinstance(v, (int, float)):
                _tb_writer.add_scalar(k, v, step)

def _save_game_recordings(recordings: List[dict], outdir: Path,
                          iteration: int, label: str) -> None:
    """Save recorded game transcripts as JSON."""
    if not recordings:
        return
    rec_dir = outdir / "game_records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    for i, rec in enumerate(recordings):
        path = rec_dir / f"iter{iteration}_{label}_game{i}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
# Curriculum annealing

def _get_curriculum_params(
    iteration: int, schedule: str, cfg: AZIterConfig,
) -> tuple:
    """Return (endgame_ratio, max_ply, disable_gating) for this iteration."""
    if schedule == "3phase":
        if iteration <= 4:      # Phase 1: Endgame Consolidation
            return 0.6, 80, True
        elif iteration <= 11:   # Phase 2: Middlegame Bridge
            return 0.2, 120, False
        else:                   # Phase 3: Full Game
            return 0.0, 150, False
    if schedule == "3phase_v2":
        if iteration <= 4:      # Phase 1: Heavy endgame bootstrap
            return 0.8, 80, True
        elif iteration <= 11:   # Phase 2: Midgame bridge + endgame anchor
            return 0.4, 120, True
        else:                   # Phase 3: Full game + 15% endgame anchor
            return 0.15, 150, True
    # No schedule — use static config values
    return cfg.endgame_ratio, cfg.selfplay_max_ply, cfg.disable_gating
# Main iteration loop

def run_iterations(cfg: AZIterConfig, outdir: Path) -> None:
    """Run iterative AlphaZero training: self-play → train → gate → eval."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(cfg.device)
    print(f"[Runner] Device: {device}")
    print(f"[Runner] Output: {outdir}")

    variant_cfg = _apply_ablation(cfg.ablation)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    config_path = outdir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"[Runner] Config saved: {config_path}")

    csv_path = outdir / "metrics.csv"

    net = PolicyValueNet(
        num_res_blocks=cfg.res_blocks, channels=cfg.channels,
    ).to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_model_path = outdir / "best_model.pt"
    buffer = ReplayBuffer(max_size=cfg.buffer_capacity_states)
    global_step = 0
    start_iteration = 0

    # ── Resume from checkpoint ──
    existing_ckpts = sorted(outdir.glob("ckpt_iter*.pt"),
                            key=lambda p: int(p.stem.split("iter")[1]))
    if existing_ckpts:
        last_ckpt = existing_ckpts[-1]
        last_iter = int(last_ckpt.stem.split("iter")[1])
        print(f"[Runner] Found checkpoint: {last_ckpt.name} (iteration {last_iter})")

        ckpt_data = torch.load(str(last_ckpt), map_location=device, weights_only=True)
        net.load_state_dict(ckpt_data["model"])
        if "optimizer" in ckpt_data:
            optimizer.load_state_dict(ckpt_data["optimizer"])
        global_step = ckpt_data.get("global_step", 0)
        start_iteration = last_iter + 1

        # Reload replay buffer from saved replay files
        replay_files = sorted(outdir.glob("replay_iter*.npz"),
                              key=lambda p: int(p.stem.split("iter")[1]))
        if replay_files:
            loaded_samples = 0
            for rf in replay_files:
                try:
                    rb = ReplayBuffer.load_npz(str(rf))
                    buffer.append(rb.examples)
                    loaded_samples += len(rb.examples)
                except Exception as e:
                    print(f"[Runner] WARNING: failed to load {rf.name}: {e}")
            print(f"[Runner] Loaded replay buffer: {loaded_samples} samples "
                  f"from {len(replay_files)} files, buffer={len(buffer)}")

        if start_iteration >= cfg.iterations:
            print(f"[Runner] All {cfg.iterations} iterations already complete!")
            return

        print(f"[Runner] RESUMING from iteration {start_iteration}")
    elif best_model_path.exists():
        _load_model_weights(net, str(best_model_path), device)
        print(f"[Runner] Loaded best_model: {best_model_path}")

    # ── CSV: append if resuming, init if fresh ──
    if start_iteration > 0 and csv_path.exists():
        print(f"[Runner] Appending to existing metrics.csv "
              f"({start_iteration} rows preserved)")
    else:
        if csv_path.exists():
            import shutil
            backup = outdir / "metrics_prev.csv"
            shutil.copy2(csv_path, backup)
            print(f"[Runner] Backed up stale metrics.csv -> {backup.name}")
        _init_csv(str(csv_path))

    print(f"[Runner] Parameters: {sum(p.numel() for p in net.parameters()):,}")
    print(f"[Runner] Resign: {'ON' if cfg.resign_enabled else 'OFF'}"
          f" (threshold={cfg.resign_threshold}, min_ply={cfg.resign_min_ply},"
          f" patience={cfg.resign_patience})")
    print(f"[Runner] Draw adjudication: {'ON' if cfg.draw_adjudicate_enabled else 'OFF'}"
          f" (min_ply={cfg.draw_adjudicate_min_ply},"
          f" patience={cfg.draw_adjudicate_patience},"
          f" |v|<={cfg.draw_adjudicate_value_abs_thr})")
    print(f"[Runner] Self-play max_ply={cfg.selfplay_max_ply}, "
          f"move_limit_value={cfg.selfplay_move_limit_value_mode}"
          f"(k={cfg.selfplay_move_limit_value_scale})")
    print(f"[Runner] MCTS sims: self-play={cfg.simulations}, "
          f"eval={cfg.eval_simulations}, gating={cfg.gating_simulations}")
    if cfg.use_cpp:
        print(f"[Runner] C++ engine: ENABLED")
    print(f"[Runner] Starting {cfg.iterations} iterations "
          f"(from iter {start_iteration})\n")

    for iteration in range(start_iteration, cfg.iterations):
        iter_start = time.time()
        print(f"{'='*60}")
        print(f"  Iteration {iteration}/{cfg.iterations - 1}")
        print(f"{'='*60}")

        # Per-iteration curriculum overrides
        iter_endgame_ratio, iter_max_ply, iter_disable_gating = \
            _get_curriculum_params(iteration, cfg.curriculum_schedule, cfg)

        if cfg.curriculum_schedule != "none":
            phase = ("Phase 1: Endgame" if iteration <= 4
                     else "Phase 2: Middlegame" if iteration <= 11
                     else "Phase 3: Full Game")
            print(f"  [Curriculum] {phase}: endgame={iter_endgame_ratio}, "
                  f"max_ply={iter_max_ply}, "
                  f"gating={'OFF' if iter_disable_gating else 'ON'}")
        # 1. Self-play

        selfplay_cfg = SelfPlayConfig(
            temperature=1.0,
            temp_cutoff_ply=cfg.temp_cutoff,
            simulations=cfg.simulations,
            max_ply=iter_max_ply,
            move_limit_value_mode=cfg.selfplay_move_limit_value_mode,
            move_limit_value_scale=cfg.selfplay_move_limit_value_scale,
            resign_enabled=cfg.resign_enabled,
            resign_threshold=cfg.resign_threshold,
            resign_min_ply=cfg.resign_min_ply,
            resign_patience=cfg.resign_patience,
            draw_adjudicate_enabled=cfg.draw_adjudicate_enabled,
            draw_adjudicate_min_ply=cfg.draw_adjudicate_min_ply,
            draw_adjudicate_patience=cfg.draw_adjudicate_patience,
            draw_adjudicate_value_abs_thr=cfg.draw_adjudicate_value_abs_thr,
        )

        game_records: List[GameRecord] = []

        if cfg.num_workers <= 1:
            # Sequential self-play (single process)
            print(f"\n  [Self-play] {cfg.selfplay_games_per_iter} games, "
                  f"{cfg.simulations} sim/step, "
                  f"max_ply={iter_max_ply}, "
                  f"resign={'ON' if cfg.resign_enabled else 'OFF'} ...")

            model = TorchPolicyValueModel(net, device=str(device))
            selfplay_agent = AlphaZeroMiniAgent(
                model=model,
                cfg=MCTSConfig(
                    simulations=cfg.simulations,
                    dirichlet_alpha=cfg.dirichlet_alpha,
                    dirichlet_eps=cfg.dirichlet_eps,
                ),
                seed=cfg.seed + iteration * 1000,
                use_cpp=cfg.use_cpp,
            )

            env = HybridChessEnv(max_plies=iter_max_ply, use_cpp=cfg.use_cpp, variant=variant_cfg)
            iter_examples = []
            sp_start = time.time()
            endgame_rng = __import__('random').Random(cfg.seed + iteration * 7777)

            for game_i in range(cfg.selfplay_games_per_iter):
                # Endgame curriculum: with probability endgame_ratio,
                # start from a generated endgame position
                initial_state = None
                if iter_endgame_ratio > 0 and endgame_rng.random() < iter_endgame_ratio:
                    from hybrid.core.env import GameState
                    eg_board, eg_side = generate_endgame_board(endgame_rng)
                    initial_state = GameState(
                        board=eg_board, side_to_move=eg_side, ply=0, repetition={}
                    )

                examples, record = self_play_game(
                    env, selfplay_agent, selfplay_cfg,
                    initial_state=initial_state,
                )
                iter_examples.extend(examples)
                buffer.append(examples)
                game_records.append(record)

                if (game_i + 1) % max(1, cfg.selfplay_games_per_iter // 5) == 0 or \
                   game_i == cfg.selfplay_games_per_iter - 1:
                    print(f"    game {game_i+1}/{cfg.selfplay_games_per_iter}  "
                          f"new_samples={len(iter_examples)}  "
                          f"buffer={len(buffer)}  "
                          f"elapsed={time.time()-sp_start:.1f}s")

            samples_added = len(iter_examples)
            selfplay_seconds = time.time() - sp_start
            samples_per_sec = samples_added / max(selfplay_seconds, 0.001)

            replay_path = outdir / f"replay_iter{iteration}.npz"
            iter_buffer = ReplayBuffer()
            iter_buffer.examples = iter_examples
            iter_buffer.save_npz(str(replay_path))
            print(f"    Saved: {replay_path} ({samples_added} samples)")

        else:
            # Parallel self-play (multi-process)
            from hybrid.rl.az_selfplay_parallel import generate_selfplay_parallel

            print(f"\n  [Self-play] {cfg.selfplay_games_per_iter} games, "
                  f"{cfg.simulations} sim/step, "
                  f"max_ply={iter_max_ply}, "
                  f"{cfg.num_workers} workers, "
                  f"server={'ON' if cfg.use_inference_server else 'OFF'}, "
                  f"resign={'ON' if cfg.resign_enabled else 'OFF'} ...")

            tmp_ckpt_path = str(outdir / f"_tmp_selfplay_iter{iteration}.pt")
            torch.save({
                "model": net.state_dict(),
                "arch": {"res_blocks": cfg.res_blocks, "channels": cfg.channels},
            }, tmp_ckpt_path)

            actual_workers = min(cfg.num_workers, cfg.selfplay_games_per_iter)
            games_per_worker = _split_games_evenly(
                cfg.selfplay_games_per_iter, actual_workers
            )

            inf_device = cfg.inference_device
            if inf_device == "auto":
                inf_device = "cuda" if torch.cuda.is_available() else "cpu"

            sp_out_dir = str(outdir / f"selfplay_iter{iteration}")
            stats = generate_selfplay_parallel(
                num_workers=actual_workers,
                games_per_worker=games_per_worker,
                selfplay_cfg=selfplay_cfg,
                mcts_cfg=MCTSConfig(
                    simulations=cfg.simulations,
                    dirichlet_alpha=cfg.dirichlet_alpha,
                    dirichlet_eps=cfg.dirichlet_eps,
                ),
                model_ckpt_path=tmp_ckpt_path,
                out_dir=sp_out_dir,
                seed=cfg.seed + iteration * 1000,
                ablation=cfg.ablation,
                use_inference_server=cfg.use_inference_server,
                inference_batch_size=cfg.inference_batch_size,
                inference_timeout_ms=cfg.inference_timeout_ms,
                inference_device=inf_device,
                endgame_ratio=iter_endgame_ratio,
                use_cpp=cfg.use_cpp,
            )

            iter_examples = []
            for wid in range(actual_workers):
                npz_path = os.path.join(sp_out_dir, f"worker_{wid}.npz")
                if os.path.exists(npz_path):
                    worker_buf = ReplayBuffer.load_npz(npz_path)
                    iter_examples.extend(worker_buf.examples)
                    buffer.append(worker_buf.examples)

                records_path = os.path.join(sp_out_dir, f"worker_{wid}_records.json")
                if os.path.exists(records_path):
                    with open(records_path, "r", encoding="utf-8") as f:
                        recs_raw = json.load(f)
                    for rd in recs_raw:
                        game_records.append(GameRecord(**rd))

            samples_added = len(iter_examples)
            selfplay_seconds = stats["elapsed_seconds"]
            samples_per_sec = stats["samples_per_sec"]

            replay_path = outdir / f"replay_iter{iteration}.npz"
            iter_buffer = ReplayBuffer()
            iter_buffer.examples = iter_examples
            iter_buffer.save_npz(str(replay_path))

            if os.path.exists(tmp_ckpt_path):
                os.remove(tmp_ckpt_path)

            print(f"    Parallel: {stats['total_games']} games, "
                  f"{samples_added} samples, {selfplay_seconds:.1f}s, "
                  f"{samples_per_sec:.1f} samples/s")

        # Self-play diagnostics
        sp_diag = _aggregate_game_records(game_records)
        if sp_diag:
            print(f"    [Diagnostics] decisive={sp_diag['sp_decisive']}/"
                  f"{sp_diag['sp_games']}  "
                  f"draw_limit={sp_diag['sp_draw_move_limit']}  "
                  f"draw_3fold={sp_diag['sp_draw_threefold']}  "
                  f"draw_stale={sp_diag['sp_draw_stalemate']}  "
                  f"draw_adj={sp_diag['sp_draw_adjudicated']}  "
                  f"resign={sp_diag['sp_resign_count']}")
            print(f"    [Diagnostics] avg_ply={sp_diag['sp_avg_ply']}  "
                  f"P50={sp_diag['sp_p50_ply']}  "
                  f"P90={sp_diag['sp_p90_ply']}  "
                  f"avg_mat_diff={sp_diag['sp_avg_mat_diff']}")
            print(f"    [Diagnostics] rootv_min_mean={sp_diag['sp_rootv_min_mean']}  "
                  f"rootv_min_p10={sp_diag['sp_rootv_min_p10']}  "
                  f"low_rootv_steps={sp_diag['sp_low_rootv_steps_sum']}  "
                  f"low_rootv_rate={sp_diag['sp_low_rootv_steps_rate']}")
        # 2. Train

        train_start = time.time()
        print(f"\n  [Train] {cfg.train_epochs} epochs, "
              f"batch_size={cfg.batch_size}, buffer={len(buffer)} ...")

        net.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_total_loss = 0.0

        for epoch in range(cfg.train_epochs):
            stats = train_one_epoch(
                net=net,
                buffer=buffer,
                optimizer=optimizer,
                device=device,
                batch_size=cfg.batch_size,
                grad_clip=cfg.grad_clip,
            )
            total_policy_loss += stats["policy_loss"]
            total_value_loss += stats["value_loss"]
            total_total_loss += stats["total_loss"]
            global_step += stats["steps"]

            print(f"    epoch {epoch+1}/{cfg.train_epochs}  "
                  f"p_loss={stats['policy_loss']:.4f}  "
                  f"v_loss={stats['value_loss']:.4f}  "
                  f"total={stats['total_loss']:.4f}  "
                  f"steps={stats['steps']}")

        avg_policy_loss = total_policy_loss / cfg.train_epochs
        avg_value_loss = total_value_loss / cfg.train_epochs
        avg_total_loss = total_total_loss / cfg.train_epochs

        ckpt_path = outdir / f"ckpt_iter{iteration}.pt"
        _save_checkpoint(net, optimizer, cfg, iteration, global_step, str(ckpt_path))
        train_seconds = time.time() - train_start
        print(f"    Saved: {ckpt_path}  (train: {train_seconds:.1f}s)")
        # 3. Gating (adaptive CI)

        from hybrid.rl.az_eval_parallel import gating_match_parallel, play_match_parallel

        gate_accepted = False
        gating_w = gating_d = gating_l = 0
        gating_games_used = 0
        gating_p_hat = 0.5
        gating_ci_low = 0.0
        gating_ci_high = 1.0

        candidate_ckpt_path = str(ckpt_path)

        if best_model_path.exists() and not iter_disable_gating:
            gating_sims = cfg.gating_simulations
            eval_workers = max(1, cfg.num_workers)
            ci_mode = "score" if cfg.gating_use_score else "wilson"
            print(f"\n  [Gating] adaptive {ci_mode} CI ({gating_sims} sims, "
                  f"{eval_workers} workers): "
                  f"min={cfg.gating_min_games}, max={cfg.gating_max_games}, "
                  f"step={cfg.gating_step_games}, "
                  f"threshold={cfg.gating_threshold}, "
                  f"confidence={cfg.gating_confidence}")

            games_played = 0
            decision = None

            while games_played < cfg.gating_max_games:
                if games_played == 0:
                    batch = min(cfg.gating_min_games, cfg.gating_max_games)
                else:
                    batch = min(cfg.gating_step_games,
                                cfg.gating_max_games - games_played)
                if batch <= 0:
                    break

                result = gating_match_parallel(
                    candidate_ckpt=candidate_ckpt_path,
                    best_ckpt=str(best_model_path),
                    games=batch,
                    num_workers=eval_workers,
                    simulations=gating_sims,
                    seed=cfg.seed + iteration * 5000 + games_played,
                    ablation=cfg.ablation,
                    swap_sides=True,
                    use_cpp=cfg.use_cpp,
                )
                gating_w += result.win_a
                gating_d += result.draw
                gating_l += result.win_b
                games_played += batch

                if cfg.gating_use_score:
                    gating_p_hat, gating_ci_low, gating_ci_high = score_ci(
                        gating_w, gating_d, gating_l, cfg.gating_confidence
                    )
                else:
                    gating_p_hat, gating_ci_low, gating_ci_high = wilson_ci(
                        gating_w, gating_l, cfg.gating_confidence
                    )

                print(f"    batch +{batch} -> total {games_played} games: "
                      f"W={gating_w} D={gating_d} L={gating_l}  "
                      f"{ci_mode}_score={gating_p_hat:.3f}  "
                      f"CI=[{gating_ci_low:.3f}, {gating_ci_high:.3f}]")

                if gating_ci_low > cfg.gating_threshold:
                    decision = "ACCEPT"
                    break
                elif gating_ci_high < cfg.gating_threshold:
                    decision = "REJECT"
                    break

            gating_games_used = games_played

            if decision is None:
                decision = "REJECT"
                print(f"    Reached max {cfg.gating_max_games} games, "
                      f"CI still spans threshold -> REJECT (conservative)")

            gate_accepted = (decision == "ACCEPT")
            print(f"    Decision: {decision}  "
                  f"(W={gating_w} D={gating_d} L={gating_l}, "
                  f"{gating_games_used} games used)")
        elif iter_disable_gating:
            gate_accepted = True
            gating_games_used = 0
            print(f"\n  [Gating] DISABLED (curriculum phase) → auto-accept")
        else:
            gate_accepted = True
            gating_games_used = 0
            print(f"\n  [Gating] No best_model yet, auto-accept")

        if gate_accepted:
            torch.save({
                "model": net.state_dict(),
                "arch": {"res_blocks": cfg.res_blocks, "channels": cfg.channels},
            }, str(best_model_path))
            print(f"    Updated best_model: {best_model_path}")
        # 4. Evaluate

        is_last_iter = (iteration == cfg.iterations - 1)
        eval_interval = cfg.eval_interval if cfg.eval_interval > 0 else 1
        should_eval = (
            iteration == 0 or
            is_last_iter or
            gate_accepted or
            (iteration % eval_interval == 0)
        )

        eval_random = MatchStats()
        eval_ab = MatchStats()

        eval_seconds = 0.0
        if should_eval:
            eval_start = time.time()
            eval_workers = max(1, cfg.num_workers)
            eval_sims = cfg.eval_simulations
            print(f"\n  [Eval] vs Random & AB, {cfg.eval_games} games each, "
                  f"{eval_sims} sims, {eval_workers} workers ...")

            eval_ckpt = str(best_model_path) if best_model_path.exists() \
                else candidate_ckpt_path

            eval_random = play_match_parallel(
                model_ckpt_path=eval_ckpt,
                opponent_type="random",
                games=cfg.eval_games,
                num_workers=eval_workers,
                simulations=eval_sims,
                seed=cfg.seed + iteration * 2000,
                ablation=cfg.ablation,
                use_cpp=cfg.use_cpp,
            )
            print(f"    vs Random: W={eval_random.win_a} D={eval_random.draw} "
                  f"L={eval_random.win_b}  avg_plies={eval_random.avg_plies:.1f}")

            eval_ab = play_match_parallel(
                model_ckpt_path=eval_ckpt,
                opponent_type="ab_d1",
                games=cfg.eval_games,
                num_workers=eval_workers,
                simulations=eval_sims,
                seed=cfg.seed + iteration * 4000,
                ablation=cfg.ablation,
                use_cpp=cfg.use_cpp,
            )
            print(f"    vs AB(d=1): W={eval_ab.win_a} D={eval_ab.draw} "
                  f"L={eval_ab.win_b}  avg_plies={eval_ab.avg_plies:.1f}")

            if cfg.eval_record_games > 0:
                _record_eval_games(eval_ckpt, cfg, iteration, outdir)
            eval_seconds = time.time() - eval_start
        else:
            print(f"\n  [Eval] skipped (next eval at iter "
                  f"{((iteration // eval_interval) + 1) * eval_interval})")
        # 5. Log to CSV

        row = {
            "iter": iteration,
            "samples_added": samples_added,
            "buffer_size": len(buffer),
            "policy_loss": round(avg_policy_loss, 6),
            "value_loss": round(avg_value_loss, 6),
            "total_loss": round(avg_total_loss, 6),
            "eval_random_w": eval_random.win_a if should_eval else "",
            "eval_random_d": eval_random.draw if should_eval else "",
            "eval_random_l": eval_random.win_b if should_eval else "",
            "eval_ab_w": eval_ab.win_a if should_eval else "",
            "eval_ab_d": eval_ab.draw if should_eval else "",
            "eval_ab_l": eval_ab.win_b if should_eval else "",
            "gating_games_used": gating_games_used,
            "gating_w": gating_w,
            "gating_d": gating_d,
            "gating_l": gating_l,
            "gating_p_hat": round(gating_p_hat, 4),
            "gating_ci_low": round(gating_ci_low, 4),
            "gating_ci_high": round(gating_ci_high, 4),
            "gating_score": round(
                (gating_w + 0.5 * gating_d) / max(gating_w + gating_d + gating_l, 1), 4
            ),
            "gate": "Y" if gate_accepted else "N",
            "lr": cfg.lr,
            "simulations": cfg.simulations,
            "selfplay_max_ply": iter_max_ply,
            "selfplay_move_limit_value_mode": cfg.selfplay_move_limit_value_mode,
            "selfplay_move_limit_value_scale": cfg.selfplay_move_limit_value_scale,
            "draw_adjudicate_enabled": int(cfg.draw_adjudicate_enabled),
            "draw_adjudicate_min_ply": cfg.draw_adjudicate_min_ply,
            "draw_adjudicate_patience": cfg.draw_adjudicate_patience,
            "draw_adjudicate_value_abs_thr": cfg.draw_adjudicate_value_abs_thr,
            "endgame_ratio": iter_endgame_ratio,
            "selfplay_seconds": round(selfplay_seconds, 2),
            "samples_per_sec": round(samples_per_sec, 1),
        }
        row.update(sp_diag)

        _append_csv(str(csv_path), row)
        _log_metrics(cfg, row, iteration)

        elapsed = time.time() - iter_start
        print(f"\n  Iteration {iteration} done in {elapsed:.1f}s "
              f"(selfplay={selfplay_seconds:.1f}s, train={train_seconds:.1f}s, "
              f"eval={eval_seconds:.1f}s)")
        print()

    print(f"{'='*60}")
    print(f"  All {cfg.iterations} iterations complete!")
    print(f"  Output: {outdir}")
    print(f"  Metrics: {csv_path}")
    print(f"{'='*60}")


def _record_eval_games(
    eval_ckpt: str,
    cfg: AZIterConfig,
    iteration: int,
    outdir: Path,
) -> None:
    """Record a few eval games (single-process, CPU inference)."""
    from hybrid.agents.alphazero_stub import TorchPolicyValueModel as _TPVM

    net = build_net_from_checkpoint(eval_ckpt, device="cpu")
    model = _TPVM(net, device="cpu")
    az = make_eval_az_agent(model, simulations=cfg.eval_simulations, seed=cfg.seed,
                            use_cpp=cfg.use_cpp)

    n = cfg.eval_record_games

    opp_random = RandomAgent(seed=cfg.seed + 9999)
    _, recs_random = play_match(
        az, opp_random, games=n, swap_sides=True, seed=cfg.seed,
        record_first_n=n,
    )
    _save_game_recordings(recs_random, outdir, iteration, "vs_random")

    opp_ab = AlphaBetaAgent(SearchConfig(depth=1))
    _, recs_ab = play_match(
        az, opp_ab, games=n, swap_sides=True, seed=cfg.seed,
        record_first_n=n,
    )
    _save_game_recordings(recs_ab, outdir, iteration, "vs_ab")

    print(f"    [Record] Saved {len(recs_random)} vs_random + "
          f"{len(recs_ab)} vs_ab game recordings")
