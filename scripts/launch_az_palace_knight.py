"""Launch AZ training for palace+knight_blk variant with speed maximized."""
import subprocess, sys, os, json, time
from pathlib import Path
from datetime import datetime

OUTDIR = Path("runs/rq4_az_palace_knight")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Write a marker so dashboard knows the run name
(OUTDIR / "run_info.json").write_text(json.dumps({
    "variant": "chess_palace,knight_block",
    "description": "Rule reform: Chess palace + Knight block",
    "started": datetime.now().isoformat(),
}))

cmd = [
    sys.executable, "-m", "scripts.train_az_iter",
    "--iterations", "10",
    "--selfplay-games-per-iter", "100",
    "--simulations", "50",
    "--selfplay-max-ply", "150",
    "--batch-size", "256",
    "--train-epochs", "2",
    "--eval-games", "20",
    "--eval-interval", "2",
    "--eval-simulations", "100",
    "--gating-min-games", "10",
    "--gating-max-games", "40",
    "--gating-step-games", "10",
    "--gating-simulations", "20",
    "--disable-gating", "1",
    "--resign-enabled", "1",
    "--resign-threshold", "-0.95",
    "--temp-cutoff", "20",
    "--device", "auto",
    "--seed", "42",
    "--ablation", "chess_palace,knight_block",
    "--use-cpp",
    "--num-workers", "4",
    "--outdir", str(OUTDIR),
]

print("=" * 70)
print("  AZ Training: palace + knight_block")
print(f"  Output: {OUTDIR}")
print(f"  Command: {' '.join(cmd)}")
print("=" * 70)

# Run directly (foreground)
os.execv(sys.executable, cmd)
