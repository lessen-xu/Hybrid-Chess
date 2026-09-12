# Round 02 Status

## Actions Taken
- **Symmetric Zero-Sum Evaluation**: Removed `-8.0` anti-stalemate penalty and enforced exact antisymmetry $v(\text{Chess}) = -v(\text{Xiangqi})$ in both Python (`hybrid/agents/eval.py`) and C++ (`cpp/src/ab_search.cpp`).
- **Balanced Replay Sampling**: Implemented `BalancedBuffer` in `hybrid/rl/az_replay.py` to sample uniformly across the 6 Side × Outcome strata. Enabled in `configs/general-ai.json`.
- **Codebase Clean-Up**: Removed obsolete course-project files (RQ1–RQ4 scripts, legacy diagnostics, and outdated report dumps).
- **Cluster Deployment**: Packaged source archive `round02-r01.tar.gz` and deployed to `/storage/homefs/lx24y045/hybrid-chess/releases/round02-r01`.
- **Cluster Verification**: Ran verification suite on compute node `gnode25` with C++ engine (82 passed, 1 skipped).
- **Teacher Job Submission**: Submitted `hc2-teacher01` via `cluster_submit.py` to generate 50,000 clean teacher samples on partition `epyc2`.

## Current Results
- **Test Suite**: 82 passed, 1 skipped on UBELIX compute node.
- **Active Job**: `hc2-teacher01` (JobID: `14958307`) queued on partition `epyc2` (14 CPUs, 32GB RAM).
- **Budget Ledger**: 0 / 28,800 GPU-seconds used; 100,800 / 230,400 CPU core-seconds reserved under account `gratis`.
- **Git Status**: Clean working tree on `main` (commit `5cd30a9`).
