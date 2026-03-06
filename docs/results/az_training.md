# AlphaZero Training Results

## Training Runs Overview

| Run | Config | vs Random (final) | vs AlphaBeta (best) | Key Issue |
|---|---|---|---|---|
| **V1** | 20 iter, 3phase, 50 sims, **extra_cannon** | 75% | 10W (iter 2 only) | Gating killed 13/20 iters, draw trap from iter 6 |
| **V2** | 20 iter, 3phase_v2, 100 sims, **extra_cannon** | 75% | 10W (iter 11, 16) | Breakthrough oscillation (not sustained) |
| **V3** | 10 iter, **no_queen**, 100 sims | 55% | 10W (iter 6) | Same oscillation, 5 iters earlier without Queen |
| **V4** | 20 iter, 3phase_v2, **200 sims**, **extra_cannon**, C++ | 80% (avg 12.8W) | 10W (8 of 20 iters) | 40% breakthrough rate, 3× more frequent than V2 |

**V4 vs V2 comparison:** V4 (200 sims) achieves 40% breakthrough frequency vs AB-d1, compared to V2's ~10% (2/20). However, the breakthrough pattern remains oscillatory—none of the 8 breakthrough iters are consecutive. The overall vs-AB score of 0.438 confirms the fundamental pipeline limitation: the small network (4 blocks, 64 filters) cannot sustainably retain tactical knowledge across training iterations.

---

## Champion Evaluation: Side-Aware Analysis (V2, extra_cannon)

Iter 16 (V2, trained with **extra_cannon**) evaluated with side-swapping (games 0–9 AZ=Chess, 10–19 AZ=Xiangqi):

| Condition | AZ as Chess | AZ as Xiangqi | Interpretation |
|---|---|---|---|
| With Queen (400 sims) | 10 Draw | 10 Loss | Queen defense is impenetrable; Queen attack crushes AZ |
| **No Queen (400 sims)** | **10 Win** | 0 Win / 10 Draw | Removing Queen unlocks wins on Chess side |

The Queen imbalance is **side-deterministic**: when AZ holds the Queen, it survives; when it faces the Queen, it collapses in ~18 moves.

---

## AZ vs AB-d2 Showdown (V2 Iter 16 @ 800 sims, no_queen, 40 games)

| | W | D | L |
|---|---|---|---|
| **AZ total** | **13** | 27 | **0** |
| AZ as Chess | 8 | 12 | 0 |
| AZ as Xiangqi | 5 | 15 | 0 |

Score: **0.662** | Termination: 13 checkmate (32%), 27 threefold repetition (68%), 0 move limit.

AZ is **undefeated** against AB-d2. It breaks the cowardice lock (95%→68% draws, 5%→32% checkmates) and wins from both sides, confirming genuine learned strategy.

---

## Simulation Scaling: Non-Linear Breakthrough (V2 Iter 16, no_queen)

| Sims | Opponent | Result | Score |
|---|---|---|---|
| 200 | AB-d1 | 10W/10D/0L | 0.750 |
| 400 | AB-d1 | 0W/20D/0L | 0.500 |
| **800** | **AB-d2** (stronger) | **13W/27D/0L** | **0.662** |

400 sims vs AB-d1 *regressed* relative to 200 sims, yet 800 sims vs the *stronger* AB-d2 produced the best result — suggesting a **phase transition** in MCTS search quality.

---

## EGTA Dual-Matrix Ablation (In Progress)

**Method:** Empirical Game-Theoretic Analysis. 9×9 round-robin payoff matrix, Nash Equilibrium via LP.

### Frozen Agent Pool (V4, extra_cannon)

| # | Agent | Spec |
|---|---|---|
| 1 | Random | `random` |
| 2 | Greedy | `greedy` |
| 3 | Pure MCTS | `pure_mcts_100` (100 sims, no NN) |
| 4 | AB-d1 | `ab_d1` (C++) |
| 5 | AB-d2 | `ab_d2` (C++) |
| 6 | AB-d4 | `ab_d4` (C++) |
| 7 | AZ-Early | V4 `ckpt_iter2.pt` |
| 8 | AZ-Mid | V4 `ckpt_iter9.pt` |
| 9 | AZ-Best | V4 `ckpt_iter19.pt` |

**Key metrics:** game value (v), Nash support size (support ≥ 2 = non-transitive / cyclic dynamics).
