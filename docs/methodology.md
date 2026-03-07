# Methodology

## Agents

| Agent | Description | Strength |
|---|---|---|
| **Random** | Uniform random legal move | Baseline anchor |
| **Greedy** | 1-ply capture maximizer (highest-value target, random tiebreak) | Short-sighted rational |
| **Pure MCTS** | AlphaZero MCTS with uniform policy + random rollout (no NN) | Brute-force search |
| **AlphaBeta** | Negamax + α-β + hand-crafted eval (material + mobility + check bonus). Configurable depth (d1/d2/d4). | d1: tactical. d2: defensive. d4: strong. **⚠ Demoted (Sprint 4): 0% endgame conversion.** |
| **AlphaBeta C++** | Full C++ negamax via `best_move()` — zero Python overhead. `_CppABAgent` wrapper in `eval_arena.py`. Includes conversion mode (auto D8 boost + lexicographic eval). | Same logic, orders of magnitude faster. **⚠ Demoted: heuristic eval cannot encode tactical mating patterns.** |
| **AlphaZero-Mini** | Small ResNet (4 blocks, 64 filters) + MCTS. Configurable simulations (50–800). | Best agent. Beats AB-d2 at 800 sims. |

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
| `terminal_info` | Full terminal detection. Priority: royal capture → no legal moves (stalemate=loss) → ply limit → threefold repetition |

### AB Search Engine (`ab_search.h/cpp`)

**Entry point:** `best_move(board, side, depth, rep_table, ply, max_plies) → SearchResult{move, score, nodes}`

**Performance optimizations:**

| Optimization | Before | After |
|---|---|---|
| Board clones / search node | 3+ | **0** (single clone at `best_move` entry) |
| `generate_legal_moves` / node | 2× | **1×** (`generate_legal_moves_inplace`) |
| Board hash / node | 2× SHA1 | **O(1) Zobrist XOR** (incremental) |
| Terminal detection | `terminal_info()` call | Inline `has_royal` + ply/rep/moves check |
| Move ordering check detection | `apply_move` clone/move | `make_move` / `unmake_move` |
| `iter_pieces` vector alloc | Every hot function | Direct `board.grid[y][x]` scan |
| Heap allocs in recursion | `vector<Move>` per node | **0** — per-ply pre-allocated `PlyBuffers` |
| Leaf eval movegen | 2× (both sides) | **1×** (reuses stm count) |

**Search features:**
- Negamax + alpha-beta pruning
- **PVS (Negascout):** null-window scout for non-PV moves, re-search on fail-high
- **Root Aspiration Windows:** narrow [center±0.75], exponential widening, full-window fallback
- **Zobrist 128-bit hashing** (incremental XOR, splitmix64 deterministic table)
- **Transposition Table** (512K entries, rep_bucket, generation isolation, mate-score pack/unpack)
- **Iterative Deepening** (depth 1..D, TT PV reuse)
- **Move ordering:** TT PV → captures+checks → killer moves → history heuristic → tie-break
- Evaluation: material + mobility (0.05×) + check bonus (0.3). **V2 endgame:** 3× material amp, Chebyshev piece→enemy king, own-king proximity, mobility squeeze, check bonus 5.0 when winning
- **Conversion mode (Sprint 4):** `detect_conversion()` triggers on defender ≤1 piece / attacker mat lead ≥5. Lexicographic `evaluate_conversion()` (mobility×50 > check×30 > approach×2 > king prox×3 > confine×5). Check extension (+2 ply), low-mobility extension (+1 ply), twofold rep penalty (−500), auto depth boost D4→D8. **Still insufficient for ≥80% conversion target.**
- **Stalemate = loss** for the stalemated side (Xiangqi convention)

---

## Training Pipeline

**Iterative AlphaZero loop:** self-play → train network → evaluate vs baselines → (optional gating) → update model.

**Key mechanisms:**
- **Parallel self-play:** multi-process workers + centralized GPU inference server for batched NN queries
- **Hard truncation:** 150-ply limit during self-play; 400-ply in evaluation/tournaments
- **Truncation penalty:** flat −0.1 reward (replaced `tanh(material_diff/4)` which caused reward hacking)
- **MCTS value discounting (γ=0.99):** shorter wins strictly preferred
- **Endgame curriculum:** permanent 15% anchor (80%→40%→15% schedule)
- **C++ MCTS integration:** `_run_mcts_search_cpp` — **3.2× end-to-end speedup**
- **GPU server-side encoding:** workers send compact `(10,9) int8` board IDs → server does GPU scatter
- **Virtual-loss leaf batching (K=8):** avg batch 7.8→45.9, throughput 221→443 states/s (8W)
- **Static batching + TF32:** 16W: **667 states/s** (3× baseline)

### GPU Throughput Scaling

| Metric | Baseline (8W) | +VL (8W) | +SHM (8W) | +Static (16W) |
|---|---|---|---|---|
| Throughput | 221 states/s | 443 states/s | 452 states/s | **667 states/s** |
| Avg batch size | 7.8 (6%) | 45.9 (36%) | 50.2 (39%) | 83.5 (65%) |
| GPU duty cycle | 41% | 38% | 43% | **57%** |
