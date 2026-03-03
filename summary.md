# Hybrid Chess ♔♚ — Project Summary

**International Chess vs Chinese Chess: Reinforcement Learning on an Asymmetric Board Game**

> Course project for *Reinforcement Learning* (FS2026).
> A two-player zero-sum game where one side plays with Chess pieces and the other with Xiangqi pieces on a shared 9×10 board. We built agents from random baselines up to a mini AlphaZero to study rule balance and learn strategies through self-play.

---

## The Game

**Board:** 9 columns × 10 rows (Xiangqi dimensions). Chess starts at the bottom (y=0–1), Xiangqi at the top (y=6–9).

**Pieces & Rules:**
- **Chess side:** King (all 8 dirs), Queen (orth+diag slide), Rook (orth slide), Bishop (diag slide), Knight (L-shape, NO leg block), Pawn (forward, double-step from y=1, diagonal capture, promotes at y=9 to Q/R/B/N).
- **Xiangqi side:** General (orth 1-step, palace only 3≤x≤5, 7≤y≤9), Advisor (diag 1-step, palace only), Elephant (diag 2-step, eye block, cannot cross river y<5), Horse (L-shape, WITH leg block), Chariot (orth slide = Rook), Cannon (orth slide non-capture; capture requires jumping over exactly 1 screen piece), Soldier (forward only before river; forward+sideways after crossing y≤4).
- **Flying General:** If General and King are on the same column with no pieces between, General can capture King directly.
- **Termination:** King/General captured, checkmate, stalemate (draw), threefold repetition (draw), move limit 400 ply (draw).
- **Rule variants (ablation flags in `config.py`):** `no_queen` (remove Chess Queen), `extra_cannon` (add 3rd Cannon at (4,7)), `no_queen_promotion` (Pawn cannot promote to Queen), `remove_extra_pawn` (remove 9th-file Pawn).

---

## Project Structure

```
hybrid-chess/
├── summary.md                          # This file — full project context
├── hybrid/                             # Core library
│   ├── core/                           # Game engine (pure Python)
│   │   ├── types.py                    #   Side, PieceKind, Piece, Move dataclasses
│   │   ├── board.py                    #   Board class (9x10 grid), initial_board()
│   │   ├── config.py                   #   Constants: BOARD_W/H, MAX_PLIES, ablation flags
│   │   ├── rules.py                    #   Move generation, check/checkmate, terminal detection
│   │   │                               #     Key functions: generate_legal_moves(), terminal_info(),
│   │   │                               #     _xiangqi_cannon_moves(), _xiangqi_horse_moves(),
│   │   │                               #     _xiangqi_elephant_moves(), _xiangqi_general_moves(),
│   │   │                               #     is_in_check(), is_square_attacked(), board_hash()
│   │   ├── env.py                      #   HybridChessEnv (gym-like): reset(), step(), legal_moves()
│   │   ├── coords.py                   #   Coordinate utilities
│   │   └── render.py                   #   ASCII board rendering
│   ├── agents/                         # All agent implementations
│   │   ├── base.py                     #   BaseAgent abstract class
│   │   ├── random_agent.py             #   RandomAgent (uniform random legal move)
│   │   ├── alphabeta_agent.py          #   AlphaBetaAgent (minimax + alpha-beta pruning)
│   │   ├── td_agent.py                 #   TDAgent (linear value function + TD(0))
│   │   ├── alphazero_stub.py           #   AlphaZeroMiniAgent (MCTS + neural network)
│   │   │                               #     _run_mcts_search_cpp: C++ MCTS path (use_cpp=True)
│   │   ├── az_remote_model.py          #   Remote model proxy for inference server
│   │   └── eval.py                     #   Hand-crafted evaluation function for AB agent
│   └── rl/                             # AlphaZero training pipeline
│       ├── az_network.py               #   PolicyValueNet (small ResNet: 4 blocks, 64 filters)
│       ├── az_encoding.py              #   State → tensor (14×10×9), GPU batch encoding, move → policy plane (92×10×9)
│       ├── az_selfplay.py              #   Single-process self-play with MCTS
│       ├── az_selfplay_parallel.py     #   Multi-worker parallel self-play
│       ├── az_inference_server.py      #   Centralized GPU inference server (batched, server-side encoding)
│       ├── az_replay.py                #   Replay buffer (state/policy/value targets)
│       ├── az_train.py                 #   Network training (MSE value + CE policy loss)
│       ├── az_eval.py                  #   Evaluation: play_one_game(), score_ci()
│       ├── az_eval_parallel.py         #   Parallel evaluation with side-swapping
│       ├── az_runner.py                #   Full iterative loop: selfplay→train→eval→gating
│       ├── endgame_spawner.py          #   Generates random endgame positions (K+pieces vs G+pieces)
│       ├── td_learning.py              #   TD(0) linear value function training
│       └── features.py                 #   Feature extraction for TD agent
├── scripts/                            # CLI tools
│   ├── train_az_iter.py                #   Main AZ training entrypoint (--iterations, --curriculum, --use-cpp)
│   ├── profile_mcts.py                 #   MCTS time distribution profiler (Python vs C++ comparison)
│   ├── train_td.py                     #   TD-learning training script
│   ├── play_match.py                   #   Agent vs agent batch match (Random/AB/TD)
│   ├── ab_tournament.py                #   AB vs AB rule balance tournament (3 conditions × N games)
│   │                                   #     Supports --condition, --tag, --depth; tracks termination reasons
│   ├── eval_champions.py              #   Evaluate AZ checkpoints vs baselines (parallel, side-swap)
│   ├── eval_az_vs_ab.py                #   AZ vs AB-d2 showdown (800 sims, termination tracking)
│   ├── monitor_training.py             #   Live training dashboard (reads metrics.csv → PNG)
│   ├── monitor_tournament.py           #   Live tournament dashboard (4-col: donut, termination, histogram, stats)
│   ├── visualize_game.py               #   Game replay → HTML/GIF visualization
│   ├── analyze_games.py                #   Batch game record analysis
│   ├── overfit_micro.py                #   Sanity check: overfit network on 1 position
│   ├── mcts_sanity_check.py            #   Sanity check: MCTS captures/avoids material
│   ├── action_mask_check.py            #   Sanity check: policy output respects legal moves
│   └── _fix_encoding.py                #   UTF-8 stdout/stderr guard for Windows (auto-imported)
├── cpp/                                # C++ game engine (pybind11)
│   ├── build.ps1                       #   Build script (MSYS2 ucrt64 g++ 15.2.0)
│   └── src/
│       ├── types.h                     #   Side, PieceKind, Piece, Move
│       ├── board.h / board.cpp         #   Board class (9×10 grid, SHA1 hash)
│       ├── rules.h / rules.cpp         #   Move gen + check + terminal (~350 LOC)
│       └── bindings.cpp                #   pybind11 module → hybrid_cpp_engine.pyd
├── hybrid/
│   └── cpp_engine/
│       ├── __init__.py                 #   Python wrapper for C++ module
│       └── hybrid_cpp_engine.*.pyd     #   Compiled extension (built by build.ps1)
├── tests/                              # pytest test suite
│   ├── test_rules.py                   #   ★ Rules oracle: move gen + terminal (40 tests)
│   │                                   #     Cannon(8), FlyingGeneral(4), Horse/Knight(5),
│   │                                   #     Elephant(4), Advisor/General(3), Soldier(3),
│   │                                   #     Pawn(4), SelfCheck(4), Terminal(5)
│   ├── test_rules_cpp.py              #   ★ C++ engine: same 40 oracle tests via pybind11
│   ├── fuzz_dual_engine.py            #   ★ Differential fuzz: Python vs C++ (500 games, 156k pos, 0 mismatch)
│   ├── test_env_cpp.py                #   ★ Env-level C++ vs Python comparison (100 games + 3 sanity)
│   ├── test_basic.py                   #   Board init, turn switching (2 tests)
│   ├── test_az_encoding.py             #   State encoding, GPU batch, in-place, cache pollution (13 tests)
│   ├── test_az_train_step.py           #   Network forward/backward (3 tests)
│   ├── test_az_replay.py               #   Replay buffer add/sample (3 tests)
│   ├── test_az_inference_server.py     #   GPU inference server (2 tests)
│   ├── test_az_runner_smoke.py         #   Full pipeline smoke test (3 tests)
│   ├── test_parallel_selfplay_smoke.py #   Parallel self-play (2 tests)
│   ├── test_endgame_spawner.py         #   Endgame positions + reset_from_board (6 tests)
│   ├── test_resign_and_diagnostics.py  #   Resign, adjudication, diagnostics (12 tests)
│   ├── test_gating_wilson.py           #   Wilson CI gating (5 tests)
│   └── test_runner_game_split.py       #   Game distribution across workers (1 test)
└── runs/                               # Experiment outputs (not in repo)
    ├── az_grand_run/                   #   V1: 20 iter, 3phase, 50 sims, extra_cannon (~21.5h)
    ├── az_grand_run_v2/                #   V2: 20 iter, 3phase_v2, 100 sims, extra_cannon (~27h) ← best models
    │   ├── ckpt_iter{0..19}.pt         #     Model checkpoints (iter 16 = best stable model)
    │   ├── metrics.csv                 #     Per-iteration training metrics
    │   ├── config.json                 #     Run configuration
    │   └── game_records/               #     JSON game records for visualization
    ├── az_no_queen_run/                #   V3: 10 iter, no_queen, 100 sims (~12h)
    ├── az_grand_run_v4/                #   V4: 20 iter, 3phase_v2, 200 sims, extra_cannon, C++ (~24h)
    │   ├── ckpt_iter{0..19}.pt         #     Model checkpoints
    │   └── metrics.csv                 #     Per-iteration training metrics
    ├── ab_tournaments/                 #   AB vs AB dashboards + termination analysis
    │   ├── ab_tournament_d{1,2}_*.png/json  # d1/d2 × 3 conditions
    │   └── ab_termination_d2_*.png/json     # no_queen termination breakdown
    └── az_grand_run_v2/
        └── az_vs_ab_showdown_*.png/json #   AZ Iter16@800sims vs AB-d2 (no_queen, 40 games)
```

---

## Agents

| Agent | Description | Strength |
|---|---|---|
| **Random** | Uniform random legal move | Baseline |
| **AlphaBeta** | Minimax + alpha-beta + hand-crafted eval (material + mobility). Configurable depth (d1/d2). | d1: tactical but shallow. d2: sees 2-ply ahead, extremely defensive. |
| **TD** | Linear value function trained via TD(0) self-play. Requires ε-greedy exploration (ε=0.2) to learn anything. | Slightly above Random |
| **AlphaZero-Mini** | Small ResNet (4 blocks, 64 filters) + MCTS. Configurable simulations (50–800). Dirichlet noise for exploration. | Best agent. Beats AB-d2 at 800 sims **no_queen** (13W/27D/0L). |

---

## Training Pipeline

**Iterative AlphaZero loop:** self-play → train network → evaluate vs baselines → (optional gating) → update model.

**Key mechanisms:**
- **Parallel self-play:** multi-process workers + centralized GPU inference server for batched neural net queries.
- **Hard truncation:** 150-ply limit during self-play (saves compute); 400-ply in evaluation/tournaments.
- **Truncation penalty:** flat −0.1 reward for truncated games (replaced `tanh(material_diff/4)` which caused reward hacking).
- **Draw adjudication:** if `|root_value| ≤ 0.08` for 15 consecutive moves → draw.
- **Endgame curriculum:** configurable fraction of self-play games start from generated endgame positions (K+Q vs bare General, etc.) for dense checkmate signals.
- **MCTS value discounting (γ=0.99):** shorter wins are strictly preferred, breaking perpetual-check loops.
- **Score-based gating:** Wilson CI on score (wins + 0.5×draws) instead of pure win rate, handles high-draw regimes.
- **Curriculum schedules:**
  - `3phase`: endgame 60%→20%→0%, gating on. Failed: gating rejected 13/20 iterations, endgame knowledge forgotten.
  - `3phase_v2`: endgame 80%→40%→15%, gating always off. Better: permanent endgame anchor + unrestricted model evolution.
- **C++ game engine (`--use-cpp`):** pybind11-wrapped C++ rules engine replaces Python in MCTS inner loop. Profile: 21× raw playout speedup → **3.2× end-to-end training speedup** (selfplay 2.5×, eval 4.6×). NN inference now dominates at 63% of MCTS time.
- **GPU server-side encoding:** `encode_state` moved from per-worker CPU Python loops to GPU batch `scatter_` inside InferenceServer. Workers send compact `(10,9) int8` board IDs (13.8× smaller than old `(14,10,9) uint8`), server encodes on GPU in batch.
- **Zero-allocation inference pipeline:** pre-allocated pinned CPU + GPU-resident buffers eliminate all dynamic allocation from the hot loop. Async DMA via `non_blocking=True` on pinned memory. AMP autocast (FP16) for forward pass on CUDA.
- **Virtual-loss leaf batching:** MCTS gathers K=8 leaves per round via virtual loss diversion before making one batched IPC call. DFS assertion verifies zero VL leakage after every search.
- **Zero-copy shared memory IPC:** `SharedMemoryPool` holds cross-process tensors (`share_memory_()`). Queue carries only `(wid, K)` tuples (~15 bytes). Server reads from pool, writes full policy flat back; workers slice locally. `mp.Event` for wake signaling.

**GPU engineering profiling** (`scripts/profile_server_path.py`, C++ engine, 200 sims, 2 plies/worker):

| Metric | Baseline (8W) | +VL (8W) | +SHM (8W) |
|---|---|---|---|
| Throughput | 221 states/s | 443 states/s | **452 states/s** |
| Avg batch size | 7.8 (6%) | 45.9 (36%) | **50.2 (39%)** |
| Max batch size | 8 | 64 | **64** |
| GPU duty cycle | 41% | 38% | **43%** |
| Worker IPC wait | 91% | 80% | **77%** |
| Worker MCTS CPU | 9% | 20% | **23%** |

**Diagnosis:** Virtual-loss leaf batching was the decisive win (2× throughput). SHM added incremental improvement (+2% throughput, +5pp GPU duty) by eliminating pickle overhead. Remaining IPC time is now OS Event latency + actual GPU compute wait. Further scaling requires more workers (linear scaling confirmed).

---

## Key Results

### AB Rule Balance Tournament (2×3 matrix, 100 games each)

AB vs AB at two depths × three rule variants. 4 random opening plies to break determinism.

| Rule variant | AB-d1 (Chess/XQ/Draw) | AB-d2 (Chess/XQ/Draw) |
|---|---|---|
| **Vanilla** | **33**/5/62 | 0/10/90 |
| **Extra Cannon** | **28**/6/66 | 0/10/90 |
| **No Queen** | **24**/7/69 | 0/5/95 |

**Conclusions:**
1. d1→d2 eliminates ALL Chess wins (33%→0%). Search depth >> rule variant.
2. Queen amplifies weak defense — at d2 it's perfectly neutralized.
3. Chess has a natural edge even without Queen (24% at d1 from Bishop diagonals + Pawn promotion).

### AB-d2 Termination Analysis (**no_queen**, 100 games)

| Termination Reason | Count | Avg Ply |
|---|---|---|
| **Threefold repetition** | 95 (95%) | 36.3 |
| Checkmate (Xiangqi wins) | 5 (5%) | 11.2 |
| Move limit (400 ply) | **0 (0%)** | — |

**100% of AB-d2 draws are true deadlocks (threefold repetition), 0% time pressure.** AB-d2 enters deterministic cowardice loops ~36 plies in — every forward move looks like material loss, so both sides retreat and repeat until repetition triggers. The 400-ply limit is never approached (max observed: 98 ply).

### AlphaZero Training Runs

| Run | Config | vs Random (final) | vs AlphaBeta (best) | Key Issue |
|---|---|---|---|---|
| **V1** | 20 iter, 3phase, 50 sims, **extra_cannon** | 75% | 10W (iter 2 only) | Gating killed 13/20 iters, draw trap from iter 6 |
| **V2** | 20 iter, 3phase_v2, 100 sims, **extra_cannon** | 75% | 10W (iter 11, 16) | Breakthrough oscillation (not sustained) |
| **V3** | 10 iter, **no_queen**, 100 sims | 55% | 10W (iter 6) | Same oscillation, 5 iters earlier without Queen |
| **V4** | 20 iter, 3phase_v2, **200 sims**, **extra_cannon**, C++ | 80% (avg 12.8W) | 10W (8 of 20 iters) | 40% breakthrough rate, 3× more frequent than V2 |

**V4 vs V2 comparison:** V4 (200 sims) achieves 40% breakthrough frequency vs AB-d1, compared to V2's ~10% (2/20). However, the breakthrough pattern remains oscillatory—none of the 8 breakthrough iters are consecutive. The overall vs-AB score of 0.438 confirms the fundamental pipeline limitation: the small network (4 blocks, 64 filters) cannot sustainably retain tactical knowledge across training iterations. 200 sims produces sharper policy targets, making breakthroughs more frequent but not more durable.

### Champion Evaluation: Side-Aware Analysis (V2, extra_cannon)

Iter 16 (V2, trained with **extra_cannon**) evaluated with side-swapping (games 0–9 AZ=Chess, 10–19 AZ=Xiangqi):

| Condition | AZ as Chess | AZ as Xiangqi | Interpretation |
|---|---|---|---|
| With Queen (400 sims) | 10 Draw | 10 Loss | Queen defense is impenetrable; Queen attack crushes AZ |
| **No Queen (400 sims)** | **10 Win** | 0 Win / 10 Draw | Removing Queen unlocks wins on Chess side |

The Queen imbalance is **side-deterministic**: when AZ holds the Queen, it survives; when it faces the Queen, it collapses in ~18 moves. This implies **AZ's tactical defense level ≈ AB-d1**: the Queen is devastating to AZ (just as it is to AB-d1, which loses 33% of games to it) but harmless against AB-d2 (which neutralizes it completely). The policy network has not yet learned the deep defensive patterns that exhaustive 2-ply search discovers automatically.

### AZ vs AB-d2 Showdown (V2 Iter 16 @ 800 sims, **no_queen**, 40 games)

| | W | D | L |
|---|---|---|---|
| **AZ total** | **13** | 27 | **0** |
| AZ as Chess | 8 | 12 | 0 |
| AZ as Xiangqi | 5 | 15 | 0 |

Score: **0.662** | Termination: 13 checkmate (32%), 27 threefold repetition (68%), 0 move limit.

AZ is **undefeated** against AB-d2. It breaks the cowardice lock (95%→68% draws, 5%→32% checkmates) and wins from both sides, confirming genuine learned strategy.

### Simulation Scaling: Non-Linear Breakthrough (V2 Iter 16, no_queen)

Across evaluations (all **no_queen** ablation), MCTS simulations show a surprising non-linear effect:

| Sims | Opponent | Result | Score |
|---|---|---|---|
| 200 | AB-d1 | 10W/10D/0L | 0.750 |
| 400 | AB-d1 | 0W/20D/0L | 0.500 |
| **800** | **AB-d2** (stronger) | **13W/27D/0L** | **0.662** |

400 sims vs AB-d1 *regressed* relative to 200 sims (likely seed variance + different game trajectories), yet 800 sims vs the *stronger* AB-d2 produced the best result of all. This suggests a **phase transition** in MCTS search quality: below a threshold, deeper search exposes the network's evaluation inaccuracies (more accurate search surface + inaccurate value → worse moves); above a threshold, the search becomes deep enough to find genuinely winning tactical sequences that the network alone cannot see.

---

## Challenges & Solutions

1. **Self-play too slow** — 400 ply games × MCTS = hours per iteration. Fix: hard truncation at 150 ply + truncation penalty (−0.1).
2. **Draw adjudication never triggered** — 3-way AND condition too strict (0 triggers in 100s of games). Fix: simplified to root_value threshold only.
3. **TD-learning learned nothing** — 100% draws without exploration → zero signal. Fix: ε-greedy (ε=0.2).
4. **Gating broke in high-draw regimes** — W=0, L=0 → CI=[0,1] → no decision possible. Fix: score-based CI (draws = 0.5).
5. **Reward hacking** — `tanh(material_diff/4)` taught model to hoard material and stall (≈+0.98 without checkmating). Fix: flat −0.1 penalty + endgame curriculum.
6. **No sense of urgency** — mate-in-3 and mate-in-20 both propagate +1.0. Fix: MCTS value discounting (γ=0.99) makes shorter wins strictly better.
7. **Gating killed curriculum progress** — new model tested against old model on full-game starts → looks "weaker" → rejected. Fix: disable gating during curriculum.
8. **Endgame knowledge forgetting** — Phase 3 dropped endgame ratio to 0% → model forgot how to checkmate. Fix: permanent 15% endgame anchor.
9. **C++ env speedup didn't help end-to-end** — C++ gave 21× raw playout speedup, but MCTS called Python rules directly (72% of time), bypassing the C++ env entirely. Fix: integrate C++ into MCTS internals (`_run_mcts_search_cpp`), achieving 3.2× real speedup.
10. **Crash-restart corrupted metrics CSV** — restarting a run in the same `outdir` appended new data after stale rows from the crashed run. Fix: `az_runner.py` now backs up existing `metrics.csv` to `metrics_prev.csv` before overwriting.
11. **Windows GBK encoding crash** — `Start-Process -RedirectStandardOutput` uses system ANSI codepage (GBK on Chinese Windows); any emoji or em-dash in `print()` crashes the script. Fix: `scripts/_fix_encoding.py` forces `stdout/stderr` to UTF-8 with `errors="replace"`, imported by all 15 scripts.

---

## Known Limitations

- **~~GPU severely starved (3–6% batch fill)~~** — ~~Workers had 1 in-flight request each~~ → **Resolved**: Virtual-loss leaf batching (K=8) raises batch fill to 22–36%, throughput 2×. Remaining idle time is IPC serialization.
- **~~NN inference dominates MCTS time~~** — ~~63% of single-process self-play time~~ → In multi-worker server mode, NN inference is amortized; the real bottleneck is Python IPC serialization.
- **~~encode_state CPU bottleneck~~** — ~~Workers ran Python loops to build (14,10,9) tensor per leaf~~ → **Resolved**: `encode_batch_gpu` does GPU scatter on server side; workers send (10,9) int8 IDs.
- **Breakthrough oscillation** — single breakthrough iteration followed by regression. Caused by small network + catastrophic forgetting + high-variance 20-game evals.
- **AZ still draws 68% vs AB-d2** — even at 800 sims, AZ cannot always break through AB-d2's repetition lock.
- **AB-d2 draws are deterministic cowardice** — 100% threefold repetition at avg 36 ply, 0% move limit. Classical AI refuses any move that looks like material loss.
- **Small sample evaluations are high-variance** — same checkpoint produces wildly different results under different seeds (±10W in 20 games).
- **~~Test coverage gap~~** — ~~`rules.py` has zero unit tests on piece movement~~ → **Resolved**: `test_rules.py` now provides 40 oracle tests covering all piece types, flying general, self-check filtering, and terminal detection.

---

## Test Coverage Assessment

**Well-tested:** AZ encoding, network training, replay buffer, inference server, self-play pipeline, resign/adjudication, gating, endgame spawner.

**Rules engine (`test_rules.py` — 40 tests, Python oracle for C++ rewrite):**

| Function | Tests | Status |
|---|---|---|
| `_xiangqi_cannon_moves` | 8 (slide, jump, screen rules, vertical, friendly) | ✅ Covered |
| `_xiangqi_general_moves` + flying general | 4 (open file, blocked, one-directional, diff column) | ✅ Covered |
| `_xiangqi_horse_moves` vs `KNIGHT` | 5 (unblocked, leg block, all blocked, Knight comparison, edge) | ✅ Covered |
| `_xiangqi_elephant_moves` | 4 (unblocked, eye block, river, edge) | ✅ Covered |
| `_xiangqi_advisor_moves` | 2 (center, corner) | ✅ Covered |
| `_xiangqi_soldier_moves` | 3 (before river, after river, edge) | ✅ Covered |
| `_chess_pawn_moves` | 4 (double step, blocked, diagonal capture, promotion) | ✅ Covered |
| `generate_legal_moves` (self-check filter) | 4 (absolute pin, check response, king safety, cannon pin) | ✅ Covered |
| `terminal_info` | 5 (checkmate, stalemate, repetition, max ply, ongoing) | ✅ Covered |

**Key discovery during testing:** Flying general is **one-directional** — only General→King, not King→General. `is_square_attacked` by Chess does NOT include flying general. A Xiangqi piece on the General-King column is NOT pinned to that column.

---

## Development Timeline

| Phase | Description |
|---|---|
| 0–1 | Rule design + environment + validation |
| 2–3 | Baseline balance analysis + `extra_cannon` compensation |
| 3 | TD-learning (discovered exploration necessity) |
| 4–6 | AlphaZero inference / encoding / training loop |
| 7–8 | Parallel self-play + GPU inference server |
| 9 | CI gating + eval speedup |
| 10 | Score-based CI + resign / diagnostics / visualization |
| 11 | Game-count distribution fix + root_value diagnostics |
| 12 | Draw adjudication A/B test |
| 13 | Hard truncation (max_ply=150) + simplified adjudication |
| 14 | Pipeline sanity checks (overfit, alignment, MCTS, masking) |
| 15 | Reward purification + endgame curriculum learning |
| 16 | MCTS value discounting + asymmetric eval scaling + 3-phase curriculum |
| 17 | 3phase_v2 curriculum (endgame anchor + gating off) + grand runs V1/V2 |
| 18 | Champion evaluation (Iter 11/16 @ 400 sims) + side-aware analysis |
| 19 | No-Queen ablation: Iter 16 goes from 0W to 10W when Queen removed |
| 20 | AB-d2 rule balance tournament + Grand Run V3 (no_queen) |
| 21 | AB-d1 tournament: 2×3 matrix proves search depth >> Queen rule variant |
| 22 | Termination analysis: 100% of AB-d2 draws = threefold repetition |
| 23 | AZ vs AB-d2 showdown: Iter 16 @ 800 sims goes 13W/27D/0L undefeated |
| 24 | `test_rules.py` oracle suite: 40 tests covering all piece move gen + terminal detection |
| 25 | C++ game engine (pybind11): `types.h`, `board.h/cpp`, `rules.h/cpp`, `bindings.cpp` — 40/40 oracle tests pass |
| 26 | Dual-engine differential fuzz: 500 games, 156k positions, 0 mismatches — C++ engine bit-perfect |
| 27 | Hot-swap `env.py` with `use_cpp=True`: 103 env tests pass, **21× speedup** (85→1,802 plies/s) |
| 28 | `--use-cpp` pipeline integration + profiling: 1.0× end-to-end speedup — MCTS calls Python rules directly (72%), C++ only touches env (0.2%) |
| 29 | C++ MCTS integration: `_run_mcts_search_cpp` in `alphazero_stub.py` — **3.2× end-to-end speedup** (selfplay 2.5×, eval 4.6×), 248/248 tests pass |
| 30 | Grand Run V4: 200 sims + C++ engine, 20 iter, ~24h — 8/20 iters hit 10W vs AB (40% breakthrough, 3× V2) |
| 31 | GPU encode_state migration: `encode_batch_gpu` (scatter), workers send (10,9) int8 → 13.8× IPC shrink, 13/13 tests pass |
| 32 | Zero-alloc inference server: pinned memory + GPU buffers + AMP (FP16), in-place `encode_batch_gpu`, 15/15 tests pass |
| 33 | Server-path profiler: GPU 3-6% batch fill, 38-41% duty; workers 91-92% IPC-bound → GPU severely starved |
| 34 | Virtual-loss leaf batching (K=8): avg batch 3.9→27.7 (4W), 7.8→45.9 (8W); throughput 109→202 (4W), 221→443 (8W); 16/16 tests pass |
| 35 | Zero-copy shared memory IPC: `SharedMemoryPool` + `(wid,K)` signal (~15B Queue payload), throughput 443→452 (8W), GPU duty 38→43%, 16/16 tests pass |

---

## Quick Start

```bash
pip install -r requirements.txt

# Run AlphaZero training
python -m scripts.train_az_iter --iterations 20 \
    --curriculum-schedule 3phase_v2 \
    --simulations 200 --eval-simulations 400 \
    --num-workers 8 --use-inference-server --inference-device cuda \
    --use-cpp \
    --outdir runs/az_run

# Monitor training
python -m scripts.monitor_training --run-dir runs/az_run

# AB tournament with termination tracking
python -m scripts.ab_tournament --depth 2 --games 100 --condition no_queen --tag my_test

# Monitor tournament
python -m scripts.monitor_tournament --tag my_test --watch 60

# AZ vs AB showdown
python -m scripts.eval_az_vs_ab \
    --ckpt runs/az_grand_run_v2/ckpt_iter16.pt \
    --eval-simulations 800 --games 40 --ablation no_queen

# Run tests
pytest -q
```
