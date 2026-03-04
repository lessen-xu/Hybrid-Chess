# -*- coding: utf-8 -*-
"""Sequential launcher for three-universe experiment runs.

Runs three AlphaZero training instances sequentially (same GPU):
  Run 0: Vanilla         (--ablation none)
  Run 1: Extra Cannon    (--ablation extra_cannon)
  Run 2: No Queen        (--ablation no_queen)

All hyper-parameters are identical — only the rule variant differs.
Designed to be launched with Start-Process for background execution.

Usage:
  python -m scripts.run_experiment
  python -m scripts.run_experiment --iterations 10 --simulations 100
"""

from __future__ import annotations

import scripts._fix_encoding  # noqa: F401
import os
import sys
import time
import json
from pathlib import Path

from hybrid.rl.az_runner import AZIterConfig, run_iterations


# ====================================================================
# Experiment configurations
# ====================================================================

RUNS = [
    {"label": "Vanilla (control)",         "ablation": "none",          "outdir": "runs/experiment_vanilla"},
    {"label": "Extra Cannon (material)",    "ablation": "extra_cannon",  "outdir": "runs/experiment_extra_cannon"},
    {"label": "No Queen (topological)",     "ablation": "no_queen",      "outdir": "runs/experiment_no_queen"},
]

PROGRESS_FILE = "runs/experiment_progress.json"


def _write_progress(data: dict) -> None:
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, PROGRESS_FILE)


def main():
    # Shared hyper-parameters (can override via command line later; hardcoded for now)
    shared = dict(
        iterations=20,
        selfplay_games_per_iter=50,
        simulations=200,
        selfplay_max_ply=150,
        selfplay_move_limit_value_mode="penalty",
        batch_size=256,
        train_epochs=1,
        buffer_capacity_states=50_000,
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        eval_games=20,
        eval_simulations=400,
        eval_interval=0,        # eval every iteration (for telemetry)
        eval_record_games=2,
        temp_cutoff=20,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        resign_enabled=True,
        resign_threshold=-0.95,
        resign_min_ply=40,
        resign_patience=3,
        draw_adjudicate_enabled=True,
        draw_adjudicate_min_ply=60,
        draw_adjudicate_patience=15,
        draw_adjudicate_value_abs_thr=0.08,
        device="auto",
        seed=0,
        use_cpp=True,
        num_workers=8,
        use_inference_server=True,
        inference_batch_size=128,
        inference_timeout_ms=5.0,
        inference_device="cuda",
        curriculum_schedule="3phase_v2",
        disable_gating=False,
    )

    total_runs = len(RUNS)
    progress = {
        "status": "running",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runs": total_runs,
        "completed_runs": 0,
        "current_run": None,
        "runs": {},
    }
    _write_progress(progress)

    print("=" * 70)
    print("  THREE-UNIVERSE EXPERIMENT LAUNCHER")
    print("=" * 70)
    print(f"  Total runs: {total_runs}")
    print(f"  Shared config: {shared['iterations']} iter, "
          f"{shared['simulations']} sims, "
          f"{shared['num_workers']} workers, "
          f"C++={'ON' if shared['use_cpp'] else 'OFF'}")
    print(f"  Progress file: {PROGRESS_FILE}")
    print("=" * 70)
    print()

    for run_idx, run_spec in enumerate(RUNS):
        label = run_spec["label"]
        ablation = run_spec["ablation"]
        outdir = run_spec["outdir"]

        progress["current_run"] = {
            "index": run_idx,
            "label": label,
            "ablation": ablation,
            "outdir": outdir,
            "status": "running",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_progress(progress)

        print(f"\n{'='*70}")
        print(f"  RUN {run_idx}/{total_runs - 1}: {label}")
        print(f"  Ablation: {ablation}")
        print(f"  Output: {outdir}")
        print(f"{'='*70}\n")

        cfg = AZIterConfig(
            ablation=ablation,
            **shared,
        )

        t0 = time.time()
        try:
            run_iterations(cfg, Path(outdir))
            elapsed = time.time() - t0
            status = "done"
        except Exception as e:
            elapsed = time.time() - t0
            status = f"error: {e}"
            print(f"\n  [ERROR] Run {run_idx} failed after {elapsed:.0f}s: {e}")
            import traceback
            traceback.print_exc()

        progress["runs"][label] = {
            "index": run_idx,
            "ablation": ablation,
            "outdir": outdir,
            "status": status,
            "elapsed_seconds": round(elapsed, 1),
            "elapsed_hours": round(elapsed / 3600, 2),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        progress["completed_runs"] = run_idx + 1
        _write_progress(progress)

        print(f"\n  Run {run_idx} ({label}): {status} in {elapsed/3600:.1f}h")

    progress["status"] = "done"
    progress["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    progress["current_run"] = None
    _write_progress(progress)

    print(f"\n{'='*70}")
    print(f"  ALL {total_runs} RUNS COMPLETE")
    print(f"{'='*70}")
    for label, info in progress["runs"].items():
        print(f"  {label}: {info['status']} ({info['elapsed_hours']}h)")
    print()


if __name__ == "__main__":
    main()
