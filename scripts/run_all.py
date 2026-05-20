"""Sequentially train all 9 AlphaZero variants in runs/.

Usage::

    python -m scripts.run_all

Behavior:
  * Variants run one-at-a-time in a fixed order.
  * Each variant invokes ``python -m scripts.train_az_iter`` as a subprocess
    so a per-variant Python crash cannot bring down the orchestrator.
  * If a variant's outdir already contains ckpt_iter*.pt, ``run_iterations``
    auto-resumes from the last completed iteration. So re-running this
    orchestrator after a crash safely continues where it left off.
  * Per-variant stdout is tee'd to ``runs/<name>/train.log``.
  * After each iteration of each variant, ``runs/orchestrator_status.json``
    is updated; the dashboard script reads that + per-variant metrics.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path


# Variant manifest: (run_name, --ablation arg).
# Names match the audit/report mapping for runs/.
VARIANTS: "OrderedDict[str, str]" = OrderedDict([
    ("rq4_az_default",        "none"),
    ("rq4_az_xqqueen_only",   "xq_queen"),
    ("rq4_az_pk_xqqueen",     "chess_palace,knight_block,xq_queen"),
    ("rq4_az_noq_only",       "no_queen"),
    ("rq4_az_palace_knight",  "chess_palace,knight_block"),
    ("rq4_az_pk_nopromo",     "chess_palace,knight_block,no_promotion"),
    ("rq4_az_nq_nopromo",     "no_queen,no_promotion"),
    ("rq4_az_nq_pk",          "no_queen,chess_palace,knight_block"),
    ("rq4_az_nq_allrules",    "no_queen,chess_palace,knight_block,no_promotion"),
])


# Default training config. Mirrors the audit's recommended full-rerun template
# and matches the per-variant compute used in the original course-project runs
# (50 iter × 100 games × 50 sims × 150 ply × 4 workers, ~6-7 h/variant).
DEFAULT_CONFIG = dict(
    iterations=50,
    selfplay_games_per_iter=100,
    simulations=50,
    selfplay_max_ply=150,
    batch_size=256,
    train_epochs=2,
    eval_games=20,
    eval_interval=2,
    eval_simulations=100,
    disable_gating=1,
    resign_enabled=1,
    device="auto",
    seed=42,
    num_workers=4,
)


def status_path(root: Path) -> Path:
    return root / "orchestrator_status.json"


def write_status(root: Path, status: dict) -> None:
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(status_path(root), "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)


def variant_done(variant_dir: Path, target_iters: int) -> bool:
    """A variant is 'done' iff metrics.csv has >= target_iters rows AND a
    best_model.pt exists. Robust against partial runs."""
    csv = variant_dir / "metrics.csv"
    best = variant_dir / "best_model.pt"
    if not csv.exists() or not best.exists():
        return False
    try:
        with open(csv, "r", encoding="utf-8") as f:
            n_rows = sum(1 for _ in f) - 1  # minus header
    except Exception:
        return False
    return n_rows >= target_iters


def build_command(run_name: str, ablation: str, outdir: Path,
                  cfg: dict) -> list[str]:
    cmd = [
        sys.executable, "-m", "scripts.train_az_iter",
        "--ablation", ablation,
        "--outdir", str(outdir),
        "--use-cpp",
        "--iterations", str(cfg["iterations"]),
        "--selfplay-games-per-iter", str(cfg["selfplay_games_per_iter"]),
        "--simulations", str(cfg["simulations"]),
        "--selfplay-max-ply", str(cfg["selfplay_max_ply"]),
        "--batch-size", str(cfg["batch_size"]),
        "--train-epochs", str(cfg["train_epochs"]),
        "--eval-games", str(cfg["eval_games"]),
        "--eval-interval", str(cfg["eval_interval"]),
        "--eval-simulations", str(cfg["eval_simulations"]),
        "--disable-gating", str(cfg["disable_gating"]),
        "--resign-enabled", str(cfg["resign_enabled"]),
        "--device", cfg["device"],
        "--seed", str(cfg["seed"]),
        "--num-workers", str(cfg["num_workers"]),
    ]
    return cmd


def run_one(run_name: str, ablation: str, root: Path, cfg: dict,
            status: dict) -> int:
    variant_dir = root / run_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    log_path = variant_dir / "train.log"
    cmd = build_command(run_name, ablation, variant_dir, cfg)

    status["current"] = {
        "name": run_name,
        "ablation": ablation,
        "outdir": str(variant_dir),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cmd": " ".join(cmd),
    }
    write_status(root, status)

    print(f"\n{'='*72}\n[orch] Running {run_name}  ablation={ablation}")
    print(f"[orch] outdir: {variant_dir}")
    print(f"[orch] log:    {log_path}")
    print(f"[orch] cmd:    {' '.join(cmd)}\n")

    # Tee subprocess stdout/stderr to BOTH parent terminal and per-variant log.
    with open(log_path, "ab") as logf:
        logf.write(
            f"\n\n=== orchestrator launch @ "
            f"{datetime.now(timezone.utc).isoformat()} ===\n".encode("utf-8")
        )
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=False,
        )
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.read(4096), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            logf.write(chunk)
            logf.flush()
        rc = proc.wait()
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="runs",
                    help="root directory for all 9 variant runs")
    ap.add_argument("--variants", type=str, default="",
                    help="comma-separated list of run_names to include "
                         "(default: all 9). Example: rq4_az_default,rq4_az_xqqueen_only")
    ap.add_argument("--retries", type=int, default=2,
                    help="retry a crashed variant this many extra times "
                         "(default: 2; resume is automatic)")
    # Compute knobs (override DEFAULT_CONFIG)
    for k, v in DEFAULT_CONFIG.items():
        argname = "--" + k.replace("_", "-")
        if isinstance(v, int):
            ap.add_argument(argname, type=int, default=v)
        elif isinstance(v, float):
            ap.add_argument(argname, type=float, default=v)
        else:
            ap.add_argument(argname, type=str, default=v)
    args = ap.parse_args()

    cfg = {k: getattr(args, k) for k in DEFAULT_CONFIG.keys()}
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    # Filter manifest if --variants given.
    if args.variants:
        wanted = [s.strip() for s in args.variants.split(",") if s.strip()]
        manifest = OrderedDict(
            (k, v) for k, v in VARIANTS.items() if k in wanted
        )
        unknown = [w for w in wanted if w not in VARIANTS]
        if unknown:
            print(f"[orch] ERROR: unknown variant names: {unknown}", file=sys.stderr)
            return 2
    else:
        manifest = VARIANTS

    status = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "config": cfg,
        "variants": [
            {"name": n, "ablation": a, "outdir": str(root / n), "state": "pending"}
            for n, a in manifest.items()
        ],
        "current": None,
    }
    write_status(root, status)

    n_total = len(manifest)
    n_done = 0
    n_skipped = 0
    n_failed = 0

    for idx, (run_name, ablation) in enumerate(manifest.items()):
        variant_dir = root / run_name

        if variant_done(variant_dir, cfg["iterations"]):
            print(f"\n[orch] {run_name}: already complete "
                  f"({cfg['iterations']} iters in metrics.csv) — skipping.")
            for v in status["variants"]:
                if v["name"] == run_name:
                    v["state"] = "done"
            n_skipped += 1
            n_done += 1
            write_status(root, status)
            continue

        for v in status["variants"]:
            if v["name"] == run_name:
                v["state"] = "running"
        write_status(root, status)

        attempts = 0
        rc = -1
        while attempts <= args.retries:
            attempts += 1
            rc = run_one(run_name, ablation, root, cfg, status)
            if rc == 0:
                break
            print(f"[orch] {run_name} exited rc={rc}, attempt {attempts}/"
                  f"{args.retries + 1} — will retry (resume is automatic).")
            time.sleep(5)

        if rc == 0:
            for v in status["variants"]:
                if v["name"] == run_name:
                    v["state"] = "done"
                    v["finished_at"] = datetime.now(timezone.utc).isoformat()
            n_done += 1
        else:
            for v in status["variants"]:
                if v["name"] == run_name:
                    v["state"] = "failed"
                    v["finished_at"] = datetime.now(timezone.utc).isoformat()
            n_failed += 1
            print(f"[orch] {run_name} FAILED after {attempts} attempts — "
                  f"continuing to next variant.")
        write_status(root, status)

    status["current"] = None
    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    status["summary"] = {
        "total": n_total, "done": n_done,
        "skipped": n_skipped, "failed": n_failed,
    }
    write_status(root, status)

    print(f"\n{'='*72}")
    print(f"[orch] Finished: total={n_total} done={n_done} "
          f"(skipped={n_skipped}) failed={n_failed}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
