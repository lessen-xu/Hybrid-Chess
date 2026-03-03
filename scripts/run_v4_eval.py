#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V4 Champion Evaluation: run all evaluations and save results."""

import scripts._fix_encoding  # noqa: F401
import subprocess
import sys
import time

PYTHON = sys.executable
RUN_DIR = "runs/az_grand_run_v4"

jobs = [
    # Group 1: vs AB-d1 @ 400 sims, no_queen, 40 games
    {
        "label": "vs AB-d1 @ 400 sims",
        "script": "scripts.eval_champions",
        "args": [
            "--run-dir", RUN_DIR,
            "--iters", "5", "15", "19",
            "--eval-simulations", "400",
            "--games", "40",
            "--num-workers", "8",
            "--ablation", "no_queen",
            "--opponents", "ab_d1",
            "--use-cpp",
        ],
    },
    # Group 2: vs AB-d2 @ 800 sims (showdown), no_queen, 40 games — iter 15 only
    {
        "label": "vs AB-d2 @ 800 sims (iter 15)",
        "script": "scripts.eval_az_vs_ab",
        "args": [
            "--ckpt", f"{RUN_DIR}/ckpt_iter15.pt",
            "--eval-simulations", "800",
            "--ab-depth", "2",
            "--games", "40",
            "--ablation", "no_queen",
            "--tag", "v4_iter15_vs_abd2",
            "--use-cpp",
        ],
    },
]

print("=" * 70)
print("  V4 Champion Evaluation — All Jobs")
print("=" * 70)

for i, job in enumerate(jobs):
    print(f"\n[{i+1}/{len(jobs)}] {job['label']}")
    print("-" * 70)
    t0 = time.time()
    cmd = [PYTHON, "-m", job["script"]] + job["args"]
    result = subprocess.run(cmd, cwd=".")
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"\n  [{status}] {job['label']} — {elapsed:.0f}s")

print("\n" + "=" * 70)
print("  All evaluations complete.")
print("=" * 70)
