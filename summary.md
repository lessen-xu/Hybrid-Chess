# Hybrid Chess ♔♚ — Project Summary

**Self-Play Reinforcement Learning for Asymmetric Cross-Species Board Games**

> A two-player zero-sum game where one side plays with International Chess pieces and the other with Chinese Chess (Xiangqi) pieces on a shared 9×10 board. We train AlphaZero-style agents via self-play to study the game-theoretic implications of asymmetric rule sets, diagnose systematic faction imbalance, and evaluate balance-restoration mechanisms.

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
│   │   ├── env.py                      #   HybridChessEnv (gym-like): reset(), step(), legal_moves()
│   │   ├── coords.py                   #   Coordinate utilities
│   │   └── render.py                   #   ASCII board rendering
│   ├── agents/                         # All agent implementations
│   │   ├── base.py                     #   BaseAgent abstract class
│   │   ├── random_agent.py             #   RandomAgent (uniform random legal move)
│   │   ├── alphabeta_agent.py          #   AlphaBetaAgent (minimax + alpha-beta pruning, Python rules)
│   │   ├── alphazero_stub.py           #   AlphaZeroMiniAgent (MCTS + neural network)
│   │   ├── az_remote_model.py          #   Remote model proxy for inference server
│   │   └── eval.py                     #   Hand-crafted evaluation function for AB agent
│   ├── rl/                             # AlphaZero training pipeline
│   │   ├── az_network.py               #   PolicyValueNet (small ResNet: 4 blocks, 64 filters)
│   │   ├── az_encoding.py              #   State → tensor (14×10×9), GPU batch encoding
│   │   ├── az_selfplay.py              #   Single-process self-play with MCTS + telemetry
│   │   ├── az_selfplay_parallel.py     #   Multi-worker parallel self-play
│   │   ├── az_inference_server.py      #   Centralized GPU inference server (batched)
│   │   ├── az_replay.py                #   Replay buffer (state/policy/value targets)
│   │   ├── az_train.py                 #   Network training (MSE value + CE policy loss)
│   │   ├── az_eval.py                  #   Evaluation: play_one_game(), score_ci()
│   │   ├── az_eval_parallel.py         #   Parallel evaluation with side-swapping
│   │   ├── az_runner.py                #   Full iterative loop: selfplay→train→eval→gating
│   │   └── endgame_spawner.py          #   Random endgame position generator
│   └── cpp_engine/
│       ├── __init__.py                 #   Python wrapper for C++ module
│       └── hybrid_cpp_engine.*.pyd     #   Compiled extension (built by build.ps1)
├── cpp/                                # C++ game engine (pybind11, MSYS2 ucrt64 g++ 15.2.0)
│   ├── build.ps1                       #   Build script → hybrid_cpp_engine.pyd
│   └── src/
│       ├── types.h                     #   Side, PieceKind, Piece, Move
│       ├── board.h / board.cpp         #   Board class (9×10 grid, SHA1 hash, Zobrist 128-bit)
│       ├── zobrist.h                   #   ZKey128 type, deterministic Zobrist random table
│       ├── rules.h / rules.cpp         #   Move gen, check, terminal, make/unmake (~450 LOC)
│       │                               #     make_move / unmake_move: in-place board mutation
│       │                               #     generate_legal_moves_inplace: zero-clone legal filter
│       ├── ab_search.h / ab_search.cpp #   Full C++ negamax α-β search (~530 LOC)
│       │                               #     best_move(): single-call entry, SearchResult return
│       │                               #     Dual-mode: Zobrist fast path (zero SHA1) + SHA1 legacy
│       │                               #     Zero Board clones, zero heap alloc in recursive search
│       │                               #     Per-ply pre-allocated buffers, leaf eval 1× opp movegen
│       └── bindings.cpp                #   pybind11 module: Board, Move, best_move, etc.
├── scripts/                            # CLI tools
│   ├── train_az_iter.py                #   Main AZ training entrypoint
│   ├── eval_arena.py                   #   Side-switching evaluation arena + _CppABAgent wrapper
│   ├── egta_tournament.py              #   EGTA dual-matrix round-robin + Nash equilibrium
│   ├── ab_tournament.py                #   AB vs AB rule balance tournament
│   ├── analyze_experiment.py           #   Generate protocol figures
│   ├── eval_champions.py               #   Evaluate AZ checkpoints vs baselines
│   ├── eval_az_vs_ab.py                #   AZ vs AB showdown
│   ├── visualize_game.py               #   Game replay → HTML/ASCII visualization
│   ├── monitor_training.py             #   Live training dashboard
│   └── _fix_encoding.py                #   UTF-8 stdout/stderr guard for Windows
├── tests/                              # pytest test suite (~150 tests)
│   ├── test_rules.py                   #   Rules oracle: move gen + terminal (40 tests)
│   ├── test_rules_cpp.py              #   C++ engine: same 40 oracle tests via pybind11
│   ├── test_env_cpp.py                #   Env-level C++ vs Python (100 games + 3 sanity)
│   ├── test_ab_cpp.py                 #   AB search: legality, determinism, mutation (14 tests)
│   ├── fuzz_dual_engine.py            #   Differential fuzz: Python vs C++ (500 games, 0 mismatch)
│   ├── test_az_encoding.py             #   State encoding, GPU batch (13 tests)
│   ├── test_az_train_step.py           #   Network forward/backward (3 tests)
│   └── ...                             #   Replay, inference server, runner, gating, etc.
└── runs/                               # Experiment outputs (not in repo)
```

---

## Agents

| Agent | Description | Strength |
|---|---|---|
| **Random** | Uniform random legal move | Baseline anchor |
| **AlphaBeta** | Minimax + α-β + hand-crafted eval (material + mobility + check bonus). Configurable depth (d1/d2/d4). | d1: tactical. d2: defensive. d4: strong. |
| **AlphaBeta C++** | Full C++ negamax via `best_move()` — zero Python overhead. `_CppABAgent` wrapper in `eval_arena.py`. | Same logic, orders of magnitude faster. |
| **AlphaZero-Mini** | Small ResNet (4 blocks, 64 filters) + MCTS. Configurable simulations (50–800). | Best agent. Beats AB-d2 at 800 sims (13W/27D/0L). |

---

## C++ Engine Architecture

The C++ engine (`hybrid_cpp_engine.pyd`) provides both a rules engine and a full AB search engine, accessed via pybind11.

### Rules Engine (`rules.h/cpp`)

| Function | Description |
|---|---|
| `generate_pseudo_legal_moves` | Direct grid scan (no `iter_pieces` allocation), all piece types |
| `generate_legal_moves` | Single clone + make/unmake filter (was per-move clone) |
| `generate_legal_moves_inplace` | Zero-clone: make/unmake on mutable board ref |
| `make_move` / `unmake_move` | In-place board mutation + reversal (captures, pawn promotion) |
| `apply_move` | Clone-based wrapper (for Python API compatibility) |
| `is_in_check` / `is_square_attacked` | Direct grid scan for attackers |
| `terminal_info` | Full terminal detection (royal, ply, repetition, checkmate) |

### AB Search Engine (`ab_search.h/cpp`)

**Entry point:** `best_move(board, side, depth, rep_table, ply, max_plies) → SearchResult{move, score, nodes}`

**Performance optimizations (Steps 1–3, 5):**

| Optimization | Before | After |
|---|---|---|
| Board clones / search node | 3+ (`terminal_info` + `legal_moves` + `apply_move`) | **0** (single clone at `best_move` entry) |
| `generate_legal_moves` / node | 2× (`terminal_info` + search) | **1×** (`generate_legal_moves_inplace`) |
| Board hash / node | 2× SHA1 (RepGuard + `terminal_info`) | **O(1) Zobrist XOR** (incremental, zero SHA1 in fast path) |
| Terminal detection | `terminal_info()` call (clones board) | Inline `has_royal` + ply/rep/moves-empty check |
| Move ordering check detection | `apply_move` clone per move | `make_move` / `unmake_move` per move |
| `iter_pieces` vector alloc | Every `find_royal`, `pseudo_legal`, `is_square_attacked` | Direct `board.grid[y][x]` scan |
| Heap allocs in recursion | `vector<Move>` per node (movegen + ordering) | **0** — per-ply pre-allocated `PlyBuffers` |
| Leaf eval movegen | 2× (`generate_legal_moves` for both sides) | **1×** (reuses stm count, generates opp only) |

**Search features (Steps 4–7):**
- Negamax + alpha-beta pruning
- **PVS (Negascout):** null-window scout for non-PV moves, re-search on fail-high within true window
- **Root Aspiration Windows:** narrow [center±0.75] from d≥2, exponential widening on fail, full-window fallback after 5 retries
- **Zobrist 128-bit hashing** (incremental XOR in `Board::set`/`move_piece`, splitmix64 deterministic table, 32-hex key format)
- **Transposition Table** (512K entries, Zobrist 128-bit key, `rep_bucket` prevents repetition pollution, generation-based isolation for determinism, mate-score pack/unpack for path-independent TT storage)
- **Dual-mode repetition:** Zobrist fast path (32-hex keys, zero SHA1) + SHA1 legacy (40-hex keys, backward compatible)
- **Iterative Deepening** (depth 1..D, TT PV reuse across iterations, cumulative nodes)
- **Move ordering:** TT PV move → captures (by victim value) + checks → killer moves (2 slots/ply) → history heuristic (`depth²` bonus) → deterministic tie-break
- Evaluation: material + mobility (0.05×) + check bonus (0.3)
- Win score with ply correction (prefers faster wins)

---

## Training Pipeline

**Iterative AlphaZero loop:** self-play → train network → evaluate vs baselines → (optional gating) → update model.

**Key mechanisms:**
- **Parallel self-play:** multi-process workers + centralized GPU inference server for batched NN queries.
- **Hard truncation:** 150-ply limit during self-play; 400-ply in evaluation/tournaments.
- **Truncation penalty:** flat −0.1 reward (replaced `tanh(material_diff/4)` which caused reward hacking).
- **MCTS value discounting (γ=0.99):** shorter wins strictly preferred.
- **Endgame curriculum:** permanent 15% anchor (80%→40%→15% schedule).
- **C++ MCTS integration:** `_run_mcts_search_cpp` in `alphazero_stub.py` — **3.2× end-to-end speedup**.
- **GPU server-side encoding:** workers send compact `(10,9) int8` board IDs → server does GPU scatter.
- **Virtual-loss leaf batching (K=8):** avg batch 7.8→45.9, throughput 221→443 states/s (8W).
- **Static batching + TF32:** 16W: **667 states/s** (3× baseline).

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

### AB-d2 Termination Analysis (no_queen, 100 games)

| Termination Reason | Count | Avg Ply |
|---|---|---|
| **Threefold repetition** | 95 (95%) | 36.3 |
| Checkmate (Xiangqi wins) | 5 (5%) | 11.2 |
| Move limit (400 ply) | **0 (0%)** | — |

**100% of AB-d2 draws are true deadlocks (threefold repetition), 0% time pressure.** AB-d2 enters deterministic cowardice loops ~36 plies in. The 400-ply limit is never approached (max observed: 98 ply).

### AlphaZero Training Runs

| Run | Config | vs Random (final) | vs AlphaBeta (best) | Key Issue |
|---|---|---|---|---|
| **V1** | 20 iter, 3phase, 50 sims, **extra_cannon** | 75% | 10W (iter 2 only) | Gating killed 13/20 iters, draw trap from iter 6 |
| **V2** | 20 iter, 3phase_v2, 100 sims, **extra_cannon** | 75% | 10W (iter 11, 16) | Breakthrough oscillation (not sustained) |
| **V3** | 10 iter, **no_queen**, 100 sims | 55% | 10W (iter 6) | Same oscillation, 5 iters earlier without Queen |
| **V4** | 20 iter, 3phase_v2, **200 sims**, **extra_cannon**, C++ | 80% (avg 12.8W) | 10W (8 of 20 iters) | 40% breakthrough rate, 3× more frequent than V2 |

**V4 vs V2 comparison:** V4 (200 sims) achieves 40% breakthrough frequency vs AB-d1, compared to V2's ~10% (2/20). However, the breakthrough pattern remains oscillatory—none of the 8 breakthrough iters are consecutive. The overall vs-AB score of 0.438 confirms the fundamental pipeline limitation: the small network (4 blocks, 64 filters) cannot sustainably retain tactical knowledge across training iterations.

### Champion Evaluation: Side-Aware Analysis (V2, extra_cannon)

Iter 16 (V2, trained with **extra_cannon**) evaluated with side-swapping (games 0–9 AZ=Chess, 10–19 AZ=Xiangqi):

| Condition | AZ as Chess | AZ as Xiangqi | Interpretation |
|---|---|---|---|
| With Queen (400 sims) | 10 Draw | 10 Loss | Queen defense is impenetrable; Queen attack crushes AZ |
| **No Queen (400 sims)** | **10 Win** | 0 Win / 10 Draw | Removing Queen unlocks wins on Chess side |

The Queen imbalance is **side-deterministic**: when AZ holds the Queen, it survives; when it faces the Queen, it collapses in ~18 moves.

### AZ vs AB-d2 Showdown (V2 Iter 16 @ 800 sims, no_queen, 40 games)

| | W | D | L |
|---|---|---|---|
| **AZ total** | **13** | 27 | **0** |
| AZ as Chess | 8 | 12 | 0 |
| AZ as Xiangqi | 5 | 15 | 0 |

Score: **0.662** | Termination: 13 checkmate (32%), 27 threefold repetition (68%), 0 move limit.

AZ is **undefeated** against AB-d2. It breaks the cowardice lock (95%→68% draws, 5%→32% checkmates) and wins from both sides, confirming genuine learned strategy.

### Simulation Scaling: Non-Linear Breakthrough (V2 Iter 16, no_queen)

| Sims | Opponent | Result | Score |
|---|---|---|---|
| 200 | AB-d1 | 10W/10D/0L | 0.750 |
| 400 | AB-d1 | 0W/20D/0L | 0.500 |
| **800** | **AB-d2** (stronger) | **13W/27D/0L** | **0.662** |

400 sims vs AB-d1 *regressed* relative to 200 sims, yet 800 sims vs the *stronger* AB-d2 produced the best result — suggesting a **phase transition** in MCTS search quality.

### EGTA Dual-Matrix Ablation (Pending)

**Method:** Empirical Game-Theoretic Analysis. 7×7 round-robin payoff matrix, Nash Equilibrium via LP.

| | Universe A: V4 (extra_cannon) | Universe B: V3 (no_queen) |
|---|---|---|
| **Hypothesis** | Queen + Cannon firepower → transitive | Without Queen → non-transitive |
| **Agents** | Random, AB-d2, AB-d4, AZ V4 iter 0/6/13/19 | Random, AB-d2, AB-d4, AZ V3 iter 0/3/6/9 |

**Key metrics:** game value (v), Nash support size (support ≥ 2 = non-transitive / cyclic dynamics).

### GPU Engineering Profiling (C++ engine, 200 sims)

| Metric | Baseline (8W) | +VL (8W) | +SHM (8W) | +Static (16W) |
|---|---|---|---|---|
| Throughput | 221 states/s | 443 states/s | 452 states/s | **667 states/s** |
| Avg batch size | 7.8 (6%) | 45.9 (36%) | 50.2 (39%) | 83.5 (65%) |
| GPU duty cycle | 41% | 38% | 43% | **57%** |

---

## Development Timeline

| Phase | Description |
|---|---|
| 0–8 | Game design, environment, baselines, AlphaZero pipeline, parallel self-play |
| 9–14 | Pipeline hardening: gating, diagnostics, sanity checks, hard truncation |
| 15–17 | Reward purification, endgame curriculum, MCTS discounting, grand runs V1/V2 |
| 18–23 | Champion evaluation, Queen ablation analysis, AB tournament 2×3 matrix, AZ vs AB-d2 showdown |
| 24–27 | C++ game engine: pybind11 rules, 40-test oracle, differential fuzz (500 games × 0 mismatch), env integration (21× speedup) |
| 28–29 | C++ MCTS integration: `_run_mcts_search_cpp` → **3.2× end-to-end** (selfplay 2.5×, eval 4.6×) |
| 30 | Grand Run V4: 200 sims + C++ engine → 40% breakthrough rate (8/20 iters hit 10W vs AB) |
| 31–37 | GPU pipeline engineering: server-side encoding, zero-alloc, VL batching, SHM IPC, static batching → **667 states/s** |
| 38–39 | Evaluation protocol telemetry, EGTA tournament framework |
| 40 | **Pure C++ negamax search:** `ab_search.h/cpp` (~230 LOC), `best_move()` single-call API, `_CppABAgent` wrapper |
| 41 | **Make/unmake refactor:** `make_move`/`unmake_move` in-place mutation, `generate_legal_moves_inplace` (zero-clone), eliminated `iter_pieces` from hot paths |
| 42 | **Inline terminal detection:** negamax bypasses `terminal_info()`, 1× movegen + 1× board_hash per node, `stm_hash_key` parameter threading |
| 43 | **TT + Iterative Deepening + Killer/History:** 512K-entry transposition table (128-bit key, rep_bucket, generation isolation, mate-score pack/unpack), iterative deepening (depth 1..D), killer moves (2 slots/ply), history heuristic (depth² bonus), move ordering (TT PV > captures+checks > killers > history > tie-break), ~420 LOC, 14 tests pass |
| 44 | **Zobrist 128-bit hashing:** incremental ZKey128 in Board (`set`/`move_piece` XOR), dual-mode AB search (`negamax_z` zero SHA1 + `negamax_sha1` legacy), TT switched to Zobrist keys, 24 tests pass |
| 45 | **Per-ply buffers + leaf eval opt:** `PlyBuffers` pre-allocation (zero heap alloc in recursion), `ScoredMove` sort, `evaluate_leaf` reuses stm move count (1× opp movegen vs 2×), all 24+40+env tests pass |
| 46 | **PVS + Aspiration Windows:** Negascout null-window scout + re-search in both negamax paths, root aspiration windows (0.75 initial, 5 retries, exp widening), test_ab_cpp 0.52s (was 1.49s), all tests green |

---

## Quick Start

```bash
pip install -r requirements.txt

# Build C++ engine
.\cpp\build.ps1

# Run tests
pytest -q

# Single AlphaZero training run
python -m scripts.train_az_iter --iterations 20 \
    --curriculum-schedule 3phase_v2 \
    --simulations 200 --eval-simulations 400 \
    --num-workers 8 --use-inference-server --inference-device cuda \
    --use-cpp --ablation extra_cannon \
    --outdir runs/az_run

# Side-switching evaluation arena (C++ AB search)
python -m scripts.eval_arena --model-a ab_d4 --model-b ab_d1 \
    --games 20 --use-cpp

# EGTA dual-matrix tournament
python -m scripts.egta_tournament --preset both --outdir runs/egta
```
