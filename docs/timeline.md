# Development Timeline

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
| 47 | **Royal cache:** O(1) king/general lookup via `royal_sq[2]` in Board, incremental maintenance in `set`/`move_piece`, `is_in_check`/`has_royal`/`terminal_info` use cache (zero grid scan), test_ab_cpp 0.16s (was 0.52s), 28+40+103 tests green |
| 48 | **Fast is_square_attacked:** reverse-ray/offset attack detection (~230 LOC) replaces O(W×H) grid scan, covers all 12 piece types (Knight L-offset, Horse reverse-L+leg, Cannon slide+screen, Elephant eye+river, Advisor/General palace, flying general King-only), deep equivalence test (20 seeds × 300 plies = 19.4M checks), 29+40+103 tests green |
| 49 | **In-place movegen scratch buffers:** `generate_pseudo_legal_moves_inplace` + 4-arg `generate_legal_moves_inplace(board, side, out, pseudo_scratch)`, `PlyBuffers.pseudo` added to AB search, all 5 call sites use scratch version (zero hidden heap alloc in movegen hot path), `xiangqi_soldier_moves` static array, 80+40+103 tests green |
| 50 | **Perft + frozen regression:** `perft_nodes(board, stm, depth)` in rules.cpp (per-depth scratch buffers, depth-1 bulk-count optimization), `test_perft_cpp.py` freezes d1–d3 for 3 positions, 90+1s tests green |
| 51 | **Docs restructure:** `summary.md` (328 LOC) → modular `docs/` layout: `game_rules.md`, `methodology.md`, `results/ab_tournament.md`, `results/az_training.md`, `timeline.md`; lightweight `README.md` (65 LOC); added `paper/`, `notebooks/` placeholders |
| 52 | **EGTA Sprint 1:** Greedy agent (1-ply capture maximizer) + Pure MCTS (C++ random rollout, `PolicyValueModel` interface) + frozen 9-agent pool (Random, Greedy, Pure MCTS 100, AB d1/d2/d4, AZ iter 2/9/19) + pilot tournament |
| 53 | **EGTA Sprint 2 tooling:** `predict_batch()` for AZ VL leaf batching, `analyze_topology.py` (alpha-rank + cycle detection), V3 9-agent pool frozen (iter 1/6/9) |
| 54 | **Parallel tournament:** `ProcessPoolExecutor` + `--workers N` in `egta_tournament.py`, 6 workers per universe (12 total on 16-core CPU), ~6× wall-clock speedup for round-robin |
| 55 | **N=100 dual-universe tournament complete:** V3 + V4 each 3,600 games (9 agents × 36 pairs × 100 games), ~11h runtime. Both TRANSITIVE (Nash support=1, 0 cycles). V3 Nash=iter9, V4 Nash=iter19 |
| 56 | **Draw disease fix:** N=100 matrix exposed AB/GREEDY vs RANDOM scoring 0.50 (100% draws). Root causes: (1) stalemate rule wrong (Western chess = draw), changed to Xiangqi convention (stalemate = loss), modified `negamax_z`/`terminal_info`/`rules.py`; (2) V2 endgame heuristics in `evaluate_leaf` (Chebyshev distance to enemy king, own-king proximity, anti-stalemate penalty, 3x material amplification, mobility squeeze). GREEDY vs RANDOM 0.55→**0.775**, AB_D4 vs AB_D1 0.50→**0.75**. Discovered inherent Chess vs Xiangqi asymmetry (Chess side 100% win rate, Xiangqi side struggles to convert) |
| 57 | **Sprint 3 — Protocol Lock + Asymmetric EGTA:** (1) legacy results isolated to `legacy_pre_v2/` + `INVALIDATED.md` (terminal semantics changed + symmetric averaging retired); (2) `game_spec_v2.md` protocol lock (stalemate=loss, threefold=draw, 400 ply=draw, V2 eval, determinism contract); (3) `egta_tournament.py` refactored to role-separated matrix `M_chess[i,j]` (row=Chess player, col=Xiangqi player), dual CSV + dual heatmap output; (4) terminal semantics gate 15/15 all pass; (5) endgame gate **FAIL** (C++ AB_D4 0% conversion on both sides, all 12 advantage endgames end in 200-ply draws), confirms V2 heuristics still insufficient to guide checkmate |

## Frozen Perft Values

Any rule change that alters these = test failure:

| Position | Side | d1 | d2 | d3 | d4 |
|---|---|---|---|---|---|
| Initial | CHESS | 23 | 1,048 | 26,311 | 1,130,358 |
| Initial | XIANGQI | 46 | 1,052 | 45,840 | 1,140,360 |
| Mid-game (seed=42, ply=10) | CHESS | 26 | 1,266 | 33,477 | 1,549,389 |

d4 速度参考：初始局面 CHESS ~0.25s, XIANGQI ~0.27s (O2, GCC ucrt64)。
