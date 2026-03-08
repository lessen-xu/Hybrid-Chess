# -*- coding: utf-8 -*-
"""Gate sweep for 9 checkpoints from 3 seeds, then 9-agent round-robin tournament.

Step 1: Gate all 9 checkpoints (3 seeds Ã— 3 iterations)
Step 2: Run round-robin tournament with gate-pass agents
"""
import sys, io, os, subprocess, time, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT = r"d:\course\FS2026\Reinforcement learning\hybrid chess"
OUTDIR = os.path.join(PROJECT, "runs", "multi_seed_egta")
os.makedirs(OUTDIR, exist_ok=True)

LOG = os.path.join(OUTDIR, "pipeline_progress.txt")

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# 9 checkpoints: 3 seeds Ã— 3 stages
CHECKPOINTS = [
    ("S0_E",   "runs/az_grand_run_v4/ckpt_iter2.pt"),
    ("S0_M",   "runs/az_grand_run_v4/ckpt_iter9.pt"),
    ("S0_L",   "runs/az_grand_run_v4/ckpt_iter19.pt"),
    ("S100_E", "runs/az_v4_seed100/ckpt_iter2.pt"),
    ("S100_M", "runs/az_v4_seed100/ckpt_iter9.pt"),
    ("S100_L", "runs/az_v4_seed100/ckpt_iter19.pt"),
    ("S200_E", "runs/az_v4_seed200/ckpt_iter2.pt"),
    ("S200_M", "runs/az_v4_seed200/ckpt_iter9.pt"),
    ("S200_L", "runs/az_v4_seed200/ckpt_iter19.pt"),
]

# ============================================================
# Step 1: Gate sweep (parallel, 8 workers)
# ============================================================
log("=" * 60)
log("STEP 1: Gate Sweep â€?9 checkpoints")
log("=" * 60)

gate_results = {}
for label, ckpt_path in CHECKPOINTS:
    log(f"  Gating {label} ({ckpt_path}) ...")
    t0 = time.time()
    
    cmd = [
        sys.executable, "-u", "-m", "scripts.gate_az_checkpoints",
        "--checkpoint", ckpt_path,
        "--oracle", "paper/data/tier_a_oracle.json",
        "--trials", "5",
        "--workers", "8",
        "--outdir", OUTDIR,
    ]
    result = subprocess.run(cmd, cwd=PROJECT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    
    elapsed = time.time() - t0
    output = result.stdout + result.stderr
    
    # Parse conversion rate from output
    conv_rate = None
    passed = False
    for line in output.split("\n"):
        if "conversion=" in line.lower() or "conv=" in line.lower():
            log(f"    {line.strip()}")
        if "PASS" in line:
            passed = True
            log(f"    {line.strip()}")
        if "FAIL" in line:
            log(f"    {line.strip()}")
        # Try to extract rate
        if "%" in line and ("iter" in line.lower() or "ckpt" in line.lower()):
            log(f"    {line.strip()}")
    
    gate_results[label] = {"path": ckpt_path, "passed": passed, "time": elapsed}
    log(f"  {label}: {'PASS' if passed else 'FAIL'} ({elapsed:.1f}s)")

# Summarize gate results
log("\n" + "=" * 60)
log("GATE SUMMARY")
log("=" * 60)
passed_agents = []
for label, info in gate_results.items():
    status = "PASS âœ? if info["passed"] else "FAIL âœ?
    log(f"  {label:>8s}: {status}")
    if info["passed"]:
        passed_agents.append((label, info["path"]))

log(f"\n  Gate-pass agents: {len(passed_agents)}/{len(CHECKPOINTS)}")

if len(passed_agents) < 2:
    log("  ERROR: Need at least 2 gate-pass agents for tournament. Aborting.")
    sys.exit(1)

# Save gate results
gate_json = os.path.join(OUTDIR, "gate_results.json")
with open(gate_json, "w", encoding="utf-8") as f:
    json.dump({
        "checkpoints": {label: info for label, info in gate_results.items()},
        "passed": [label for label, _ in passed_agents],
        "passed_paths": [path for _, path in passed_agents],
    }, f, indent=2)
log(f"  Gate results saved: {gate_json}")

# ============================================================
# Step 2: Round-robin tournament (parallel, 4 workers per pair)
# ============================================================
log("\n" + "=" * 60)
log(f"STEP 2: Round-Robin Tournament â€?{len(passed_agents)} agents")
log("=" * 60)

# Build agent specs for tournament
agent_specs = ",".join(path for _, path in passed_agents)
agent_labels = [label for label, _ in passed_agents]
log(f"  Agents: {agent_labels}")
log(f"  Pairs: {len(passed_agents) * (len(passed_agents)-1) // 2}")
log(f"  Games per pair: 100")

tournament_outdir = os.path.join(OUTDIR, "tournament")
os.makedirs(tournament_outdir, exist_ok=True)

t0 = time.time()
cmd = [
    sys.executable, "-u", "-m", "scripts.egta_tournament",
    "--preset", "custom",
    "--agents", agent_specs,
    "--ablation", "extra_cannon",
    "--games-per-pair", "100",
    "--simulations", "400",
    "--use-cpp",
    "--seed", "42",
    "--outdir", tournament_outdir,
    "--workers", "4",
]

log(f"  Launching tournament...")
result = subprocess.run(cmd, cwd=PROJECT, capture_output=True, text=True, encoding="utf-8", errors="replace")
elapsed = time.time() - t0

# Save tournament output
tourn_log = os.path.join(OUTDIR, "tournament_output.txt")
with open(tourn_log, "w", encoding="utf-8") as f:
    f.write(result.stdout)
    if result.stderr:
        f.write("\n\nSTDERR:\n" + result.stderr)

# Print key results
log(f"  Tournament completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
for line in result.stdout.split("\n"):
    if any(k in line for k in ["Nash", "matrix", "Matrix", "payoff", "Payoff", "value", "RESULT", "winner", "Winner"]):
        log(f"  {line.strip()}")
    if "â”? in line or "â”€" in line or "|" in line:
        log(f"  {line.strip()}")

log(f"\n  Full output: {tourn_log}")
log(f"  Tournament dir: {tournament_outdir}")

log("\n" + "=" * 60)
log("PIPELINE COMPLETE")
log("=" * 60)
