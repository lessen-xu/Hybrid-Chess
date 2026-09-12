# Round 02 Status

## Actions Taken
- **Symmetric Zero-Sum Evaluation**: Removed `-8.0` anti-stalemate penalty and enforced exact antisymmetry $v(\text{Chess}) = -v(\text{Xiangqi})$ in both Python (`hybrid/agents/eval.py`) and C++ (`cpp/src/ab_search.cpp`).
- **Balanced Replay Sampling**: Implemented `BalancedBuffer` in `hybrid/rl/az_replay.py` to sample uniformly across the 6 Side × Outcome strata. Enabled in `configs/general-ai.json`.
- **Codebase Clean-Up**: Removed obsolete course-project files (RQ1–RQ4 scripts, legacy diagnostics, and outdated report dumps).
- **Cluster Deployment**: Packaged source archive `round02-r01.tar.gz` and deployed to `/storage/homefs/lx24y045/hybrid-chess/releases/round02-r01`.
- **Cluster Verification**: Ran verification suite on compute node `gnode25` with C++ engine (82 passed, 1 skipped).
- **Teacher Dataset Generation**: Executed `hc2-teacher01` on compute node `bnode048`, generating 49,980 samples.
- **Teacher Dataset Audit**: Ran `audit_teacher.py` to verify game-disjointness and win/loss balance across splits.
- **GPU Training Submission**: Submitted Stage 2 job `hc2-train01` via `cluster_submit.py` requesting 1x H100 GPU and 16 CPUs for 10 supervised epochs + self-play.

## Current Results
- **Teacher Dataset**: 49,980 samples across 727 games generated in 1,544 seconds.
- **Dataset Symmetry**: Decisive games in training split: 35 Chess wins vs 34 Xiangqi wins (1.03:1 balance, resolving legacy 3.5:1 skew).
- **Active Job**: `hc2-train01` (JobID: `14959962`) queued as the top runnable job on partition `gpu` (1x H100, 16 CPUs, 64GB RAM).
- **Budget Ledger**: 0 / 28,800 GPU-seconds used; 22,442 CPU core-seconds used (100,800 reserved) under account `gratis`.
- **Git Status**: Clean working tree on `main`.
