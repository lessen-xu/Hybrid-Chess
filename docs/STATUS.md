# Round 02 Status

## Actions Taken
- **Symmetric Zero-Sum Evaluation**: Removed `-8.0` anti-stalemate penalty and enforced exact antisymmetry $v(\text{Chess}) = -v(\text{Xiangqi})$ in both Python (`hybrid/agents/eval.py`) and C++ (`cpp/src/ab_search.cpp`).
- **Balanced Replay Sampling**: Implemented `BalancedBuffer` in `hybrid/rl/az_replay.py` to sample uniformly across the 6 Side × Outcome strata. Enabled in `configs/general-ai.json`.
- **Codebase Clean-Up**: Removed obsolete course-project files (RQ1–RQ4 scripts, legacy diagnostics, and outdated report dumps).
- **Cluster Deployment**: Packaged source archive `round02-r01.tar.gz` and deployed to `/storage/homefs/lx24y045/hybrid-chess/releases/round02-r01`.
- **Cluster Verification**: Ran verification suite on compute node `gnode25` with C++ engine (82 passed, 1 skipped).
- **Teacher Dataset Generation**: Executed `hc2-teacher01` on compute node `bnode048`, generating 49,980 samples (35 Chess wins vs 34 Xiangqi wins in decisive training games).
- **GPU Supervised Pretraining**: Executed on `gnode26` (H100 GPU), converging to validation loss 1.219; exported `supervised.pt` and initial `candidate.pt`.
- **RL Self-Play Iterations**: Completed Iterations 0–14 (1,920 self-play games; replay buffer saturated at 100,000 balanced samples; loss converged from 3.332 to 2.843); finalized `candidate.pt`.
- **Comprehensive Evaluation Arena**: Executed 360 paired games (`hc2-eval01`, 14 CPUs on `bnode023`) measuring the model across 6 rule presets against `random`, `greedy`, and `ab_fast` (AlphaBeta depth 1).

## Final Results & Balance Analysis

### Model Convergence
- **Teacher Phase**: 49,980 samples from 727 games, exact 1:1 decisive balance (35:34).
- **Pretraining**: Train loss 0.322, validation loss 1.219 (`supervised.pt`).
- **Self-Play RL**: 1,920 games across 15 iterations (0–14). Loss decreased monotonically from 3.332 to 2.843 with 100,000 balanced samples.

### Rule Variant Balance Benchmarks (vs AlphaBeta Minimax `ab_fast`)
Evaluated across 20 paired games (side-swapping with identical opening seeds):

| Preset | Rules / Modifications | Chess Score | Xiangqi Score | Balance Delta | Record (W/D/L) |
|---|---|---|---|---|---|
| `none` | Standard Rules (Original) | 1.00 (100%) | 0.45 (45%) | 55% | 10W / 9D / 1L |
| `no_queen` | Chess without Queen | 1.00 (100%) | 0.50 (50%) | 50% | 12W / 6D / 2L |
| `extra_cannon` | Xiangqi with 3rd Cannon | 0.95 (95%) | 0.70 (70%) | 25% | 13W / 7D / 0L |
| `xq_queen` | Xiangqi with Queen | 0.85 (85%) | 0.65 (65%) | 20% | 10W / 10D / 0L |
| `pk` | Palace & Blocked Knights | 0.85 (85%) | 0.65 (65%) | 20% | 12W / 6D / 2L |
| `pk_xq_queen` | Palace, Blocked Knights & Xiangqi Queen | **0.70 (70%)** | **0.60 (60%)** | **10%** | 7W / 12D / 1L |

### Key Observations & Methodological Notes
1. **Empirical Asymmetry**: Under standard rules (`none`), the model demonstrates significant performance asymmetry against AlphaBeta (1.00 as Chess vs 0.45 as Xiangqi).
2. **Preset Comparison vs. Equilibrium**: In this preliminary test, `pk_xq_queen` exhibited the smallest disparity (0.70 as Chess vs 0.60 as Xiangqi) among the six tested presets. However, per `docs/BALANCE_PROTOCOL.md`, these numbers reflect $P(M_{\text{Chess}} \text{ beats } AB_{\text{XQ}})$ vs $P(M_{\text{XQ}} \text{ beats } AB_{\text{Chess}})$, which convolve army advantage with agent competence and heuristic bias rather than proving game-theoretic equilibrium.
3. **Causal Attribution Next Step**: Because `pk` and `pk_xq_queen` bundle multiple modifications simultaneously (palace, knight-blocking, xiangqi-queen), isolating the exact main effect of each rule requires systematic fractional factorial screening.

### Transition to Asymmetric Balance Laboratory
The repository has transitioned from heuristic variant benchmarking to a systematic balance laboratory governed by [BALANCE_PROTOCOL.md](file:///d:/Project/Hybrid%20Chess/docs/BALANCE_PROTOCOL.md).

### Resource & Budget Ledger
- **GPU Budget**: 3,741 / 28,800 seconds (13.0% used) on NVIDIA H100.
- **CPU Budget**: 32,242 / 230,400 core-seconds (14.0% used) on AMD EPYC.
- **All Slurm Jobs**: `hc2-teacher01`, `hc2-train01`, `hc2-eval01` completed successfully with exit code 0.


