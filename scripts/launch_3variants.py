"""Launch 3 intermediate variant training runs sequentially."""
import subprocess, sys

RUNS = [
    ("runs/rq4_az_pk_nopromo", "chess_palace,knight_block,no_promotion"),
    ("runs/rq4_az_nq_pk",     "no_queen,chess_palace,knight_block"),
    ("runs/rq4_az_nq_nopromo", "no_queen,no_promotion"),
]

BASE_ARGS = [
    sys.executable, "scripts/train_az_iter.py",
    "--iterations", "50",
    "--selfplay-games-per-iter", "100",
    "--simulations", "50",
    "--selfplay-max-ply", "150",
    "--batch-size", "256",
    "--train-epochs", "2",
    "--eval-games", "20",
    "--eval-interval", "2",
    "--eval-simulations", "100",
    "--disable-gating", "1",
    "--resign-enabled", "1",
    "--temp-cutoff", "20",
    "--device", "auto",
    "--seed", "42",
    "--use-cpp",
    "--num-workers", "4",
]

for i, (outdir, ablation) in enumerate(RUNS):
    print(f"\n{'='*60}")
    print(f"  Run {i+1}/3: {ablation}")
    print(f"  Output: {outdir}")
    print(f"{'='*60}\n")
    cmd = BASE_ARGS + ["--ablation", ablation, "--outdir", outdir]
    result = subprocess.run(cmd, cwd=".")
    if result.returncode != 0:
        print(f"ERROR: Run {i+1} failed with code {result.returncode}")
        sys.exit(1)
    print(f"\n  Run {i+1}/3 complete!\n")

print("\n" + "="*60)
print("  All 3 runs complete!")
print("="*60)
