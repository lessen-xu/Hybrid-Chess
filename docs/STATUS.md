# Round 02 Status

## Actions Taken
- **Symmetric Zero-Sum Evaluation**: Removed `-8.0` anti-stalemate penalty and enforced exact antisymmetry $v(\text{Chess}) = -v(\text{Xiangqi})$ in both Python (`hybrid/agents/eval.py`) and C++ (`cpp/src/ab_search.cpp`).
- **Balanced Replay Sampling**: Implemented `BalancedBuffer` in `hybrid/rl/az_replay.py` to sample uniformly across the 6 Side × Outcome strata. Enabled in `configs/general-ai.json`.
- **Codebase Clean-Up**: Removed obsolete course-project files (RQ1–RQ4 scripts, legacy diagnostics, and outdated report dumps).
- **Cluster Deployment**: Packaged source archive `round02-r01.tar.gz` and deployed to `/storage/homefs/lx24y045/hybrid-chess/releases/round02-r01`.
- **Cluster Verification**: Ran verification suite on compute node `gnode25` with C++ engine (82 passed, 1 skipped).
- **Teacher Dataset Generation**: Executed `hc2-teacher01` on compute node `bnode048`, generating 49,980 samples (35 Chess wins vs 34 Xiangqi wins in decisive training games).
- **GPU Supervised Pretraining**: Executed on `gnode26` (H100 GPU), converging to validation loss 1.219; exported `supervised.pt` and initial `candidate.pt`.
- **RL Self-Play Iterations**: Completed Iterations 0–6 (896 games, 63,061 buffer samples, simulations 128, loss 3.056 with `BalancedBuffer`); currently executing Iteration 7.

## Current Results
- **Active Job**: `hc2-train01` (JobID: `14959962`) running on node `gnode26` (1x H100 GPU, 16 CPUs, 64GB RAM).
- **Model Checkpoints**: `supervised.pt`, `candidate-0001.pt` through `candidate-0007.pt`, and `candidate.pt` generated.
- **Self-Play Progress**: 896 games completed (466 Chess wins, 125 Xiangqi wins, 305 draws); Iteration 7 in progress.
- **Budget Ledger**: ~1,450 / 28,800 GPU-seconds used; 22,442 / 230,400 CPU core-seconds used under account `gratis`.
- **Git Status**: Clean working tree on `main`.
