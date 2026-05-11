# Hybrid Chess — Experiment Results

> Last updated: 2026-05-12 (fixed_v1 retrain)

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Overview](#overview)
3. [RQ4 — Early Exploration](#rq4--early-exploration)
4. [AB D2 Rule Reform Scan](#ab-d2-rule-reform-scan)
5. [Rule Reform Implementation](#rule-reform-implementation)
6. [AlphaZero Nine-Variant Training](#alphazero-nine-variant-training)
7. [Factor Analysis](#factor-analysis)
8. [Cross-Variant Tournament (RQ3)](#cross-variant-tournament-rq3)
9. [Recommended Variant](#recommended-variant)
10. [Training Command](#training-command)
11. [TODO](#todo)

---

## Project Structure

```
hybrid chess/
├── cpp/                   # C++ engine (move gen, AB search, pybind11)
│   └── src/
├── hybrid/
│   ├── core/              # Game engine (types, board, rules, config, env, fen)
│   ├── agents/            # AI agents (Random, Greedy, AlphaBeta, AlphaZero)
│   └── rl/                # AlphaZero pipeline (network, encoding, selfplay, train, eval, runner)
├── scripts/
│   ├── train_az_iter.py                       # AZ training CLI entry
│   ├── run_fixed_v1_all.py                    # Orchestrator: trains all 9 variants in sequence
│   ├── dashboard_fixed_v1.py                  # Live HTML progress dashboard
│   ├── cross_variant_tournament_fixed_v1.py   # Cross-variant tournament with temperature sampling
│   ├── rq4_rule_reform_ab_fixed_v1.py         # AB D2 rule reform scan
│   └── eval_arena.py                          # Side-swapped evaluation
├── tests/                 # Test suite (340+ tests, including conftest.py state reset)
├── ui/                    # Browser game UI
├── runs/fixed_v1/         # Experiment outputs (gitignored)
│   ├── rq4_rule_reform_ab/         # AB scan results
│   ├── rq4_az_default/             # Default 50 iters
│   ├── rq4_az_noq_only/            # Q only 50 iters
│   ├── rq4_az_xqqueen_only/        # X only 50 iters
│   ├── rq4_az_palace_knight/       # PK 50 iters
│   ├── rq4_az_pk_nopromo/          # PK+noPromo 50 iters
│   ├── rq4_az_pk_xqqueen/          # PK+xqQueen 50 iters ⭐
│   ├── rq4_az_nq_nopromo/          # noQ+noPromo 50 iters
│   ├── rq4_az_nq_pk/               # noQ+PK 50 iters
│   ├── rq4_az_nq_allrules/         # noQ+ALL 50 iters
│   └── cross_variant_tournament/   # 3,600-game tournament (T=0.5)
└── docs/
    ├── ARCHITECTURE.md
    ├── EXPERIMENTS_EN.md  # This file (English)
    └── EXPERIMENTS_ZH.md  # Chinese version
```

---

## Overview

| Phase | Goal | Status | Output |
|-------|------|--------|--------|
| AB D2 Rule Reform Scan | Fast screening of 23 variants | ✅ Done | `runs/fixed_v1/rq4_rule_reform_ab/` |
| AZ Nine-Variant Comparison (50 iters each) | Find optimal balance | ✅ Done | `runs/fixed_v1/rq4_az_*` |
| Cross-Variant Tournament | Meta-strategy analysis | ✅ Done | `runs/fixed_v1/cross_variant_tournament/` |

- **AZ Training**: 9 variants × 50 iters = 450 iters, 45,000 self-play games total
- **AB Scan**: 23 variants × 40 games = 920 games
- **Tournament**: 36 pairs × 100 games per pair = 3,600 games (temperature-sampled action selection so each game is an independent sample)

---

## RQ4 — Early Exploration

Tested piece-reduction variants (no_queen, no_bishop, extra_soldier, etc.) using AB D2:
- Default rules: mat_diff ≈ +19 (Chess dominates)
- Piece reduction can approach 0 but draw rate too high (AB D2 too shallow; "balance" was actually ineffective play)
- Introduced `mat_diff` as material-difference metric to distinguish "real balance" from "dead draws"

**Conclusion**: Piece reduction alone cannot eliminate Chess's structural advantage; rule-level reform is needed.

---

## AB D2 Rule Reform Scan

- **Script**: `scripts/rq4_rule_reform_ab_fixed_v1.py`
- **Output**: `runs/fixed_v1/rq4_rule_reform_ab/results.json` + `progress.log`
- **Scale**: 23 variants × 40 games, Alpha-Beta depth=2, C++ accelerated, 8 workers
- **Three reform rules**:
  - `no_promotion`: Pawns do not promote upon reaching the back rank
  - `chess_palace`: Chess King confined to a 3×3 palace (x=3–5, y=0–2)
  - `knight_block`: Chess Knight uses Xiangqi horse blocking rules

Ranked by `|avg_mat_diff|` (closest to 0 = best). `mtb*` = material tiebreak among drawn games.

| Rank | Variant | matdiff | C | X | D | mtbC | mtbX | mtbE | avg ply |
|------|---------|---------|---|---|---|------|------|------|---------|
| 1 | palace+knight_blk | +0.0 | 0 | 0 | 40 | 0 | 0 | 40 | 85 |
| 2 | ALL_RULES | +0.0 | 0 | 0 | 40 | 0 | 0 | 40 | 85 |
| 3 | nq+ec | +1.0 | 0 | 0 | 40 | 40 | 0 | 0 | 64 |
| 4 | nq+ec+no_promo | +1.0 | 0 | 0 | 40 | 40 | 0 | 0 | 64 |
| 5 | nq+ec+palace | +1.0 | 0 | 0 | 40 | 40 | 0 | 0 | 64 |
| 6 | nq+nb | −2.0 | 0 | 0 | 40 | 0 | 40 | 0 | 45 |
| 7 | nq+nb+no_promo | −2.0 | 0 | 0 | 40 | 0 | 40 | 0 | 45 |
| 8 | nq+nb+palace | −2.0 | 0 | 0 | 40 | 0 | 40 | 0 | 45 |
| 9 | no_queen+ALL_RULES | +3.0 | 0 | 0 | 40 | 40 | 0 | 0 | 101 |
| 10 | nq+nb+knight_blk | −5.0 | 0 | 0 | 40 | 0 | 40 | 0 | 27 |
| 11 | nq+nb+es+ALL_RULES | +7.0 | 0 | 0 | 40 | 40 | 0 | 0 | 108 |
| 12 | no_queen | +9.0 | 0 | 0 | 40 | 40 | 0 | 0 | 150 |
| 13 | no_queen+no_promo | +9.0 | 0 | 0 | 40 | 40 | 0 | 0 | 150 |
| 14 | no_queen+palace | +9.0 | 0 | 0 | 40 | 40 | 0 | 0 | 150 |
| 15 | nq+nb+ALL_RULES | +9.0 | 0 | 0 | 40 | 40 | 0 | 0 | 88 |
| 16 | default | +11.0 | 0 | 0 | 40 | 40 | 0 | 0 | 150 |
| 17 | no_promo | +11.0 | 0 | 0 | 40 | 40 | 0 | 0 | 150 |
| 18 | palace | +11.0 | 0 | 0 | 40 | 40 | 0 | 0 | 150 |
| 19 | no_promo+palace | +11.0 | 0 | 0 | 40 | 40 | 0 | 0 | 150 |
| 20 | no_queen+knight_blk | +16.0 | 0 | 0 | 40 | 40 | 0 | 0 | 146 |
| 21 | knight_blk | +17.0 | 0 | 0 | 40 | 40 | 0 | 0 | 150 |
| 22 | no_promo+knight_blk | +17.0 | 0 | 0 | 40 | 40 | 0 | 0 | 150 |
| 23 | nq+ec+ALL_RULES | +23.0 | 0 | 0 | 40 | 40 | 0 | 0 | 149 |

**Conclusion**: `palace + knight_block` (and the all-rules combination) achieves perfect material balance (matdiff = 0.0) under shallow AB search — the optimal structural intervention identified at the screening stage. Default rules show a strong Chess material advantage (matdiff ≈ +11). Knight-block alone is strictly worse than knight-block + palace, because palace by itself does nothing decisive at depth 2.

---

## Rule Reform Implementation

**C++ side** (`cpp/src/`):
- `types.h`: `RuleFlags` struct + `thread_local g_rule_flags`; `PieceKind::XQ_QUEEN` enum value for the Xiangqi-side queen-like piece.
- `rules.cpp`: All three reforms integrated in move generation, attack detection, and the fast `is_square_attacked_fast` path (which handles `XQ_QUEEN` orthogonal and diagonal rays on the Xiangqi side).
- `bindings.cpp`: Exposes `RuleFlags`, `set_rule_flags`, and `XQ_QUEEN` to Python.
- `zobrist.h`: Zobrist table extended to 14 piece kinds; `board.cpp` repetition hash uses unique per-kind tokens (full enum name, not first-letter) so KING/KNIGHT and CHARIOT/CANNON cannot collide.

**Python side** (`hybrid/core/`):
- `types.py`: `PieceKind` gains `XQ_QUEEN`.
- `board.py` / `rules.py`: `xq_queen=True` places a `PieceKind.XQ_QUEEN` at the left-Advisor square; move generation treats `QUEEN` and `XQ_QUEEN` as queen-like sliders.
- `config.py`: `no_promotion`, `chess_palace`, `knight_block`, `xq_queen` fields on `VariantConfig`.
- `env.py` `_set_active_variant()`: Auto-syncs C++ rule flags on environment reset.

**Ablation mapping** (`hybrid/rl/az_runner.py`):
```python
'no_promotion':  {'no_promotion': True},
'chess_palace':  {'chess_palace': True},
'knight_block':  {'knight_block': True},
'xq_queen':      {'xq_queen': True},
```

**State encoding**: 15-channel binary planes (one per piece kind, with `XQ_QUEEN` getting its own channel so the Xiangqi-side queen-like piece is unambiguously distinguished from a Chess Queen at the same square) + 1 side-to-move plane.

---

## AlphaZero Nine-Variant Training

### Configuration

All AZ runs use a uniform config (50 iters × 100 games/iter = 5,000 self-play games/variant):
- Self-play: 100 games/iter, 50 sims, max_ply=150, 4 workers
- Training: 2 epochs, batch=256, buffer=50000
- Evaluation: 20 games vs Random + 20 games vs AB(d1), every 2 iters
- Total: **9 variants × 50 iters = 45,000 self-play games**

> **PK** = chess_palace + knight_block, **Q** = no_queen, **X** = xq_queen, **ALL** = PK + no_promotion

### Nine-Variant Comparison (last-10-iter averages)

| Variant | Iters | Chess% | XQ% | Draw% | C:X | MatDiff |
|---------|-------|--------|-----|-------|-----|---------|
| Default | 50 | 35.6 | 4.0 | 60.4 | 8.9× | −6.40 |
| Q only | 50 | 0.9 | 1.6 | 97.5 | 0.6× | −11.72 |
| X only | 50 | 22.8 | 7.8 | 69.4 | 2.9× | −11.27 |
| PK | 50 | 30.9 | 9.3 | 59.8 | 3.3× | −6.77 |
| PK+noPromo | 50 | 31.1 | 9.1 | 59.8 | 3.4× | −6.25 |
| **PK+xqQueen** ⭐ | 50 | **21.2** | **18.0** | **60.8** | **1.2×** | **−10.68** |
| noQ+noPromo | 50 | 2.2 | 1.4 | 96.4 | 1.6× | −11.32 |
| noQ+PK | 50 | 1.2 | 3.6 | 95.2 | 0.3× | −11.57 |
| noQ+ALL | 50 | 1.5 | 4.6 | 93.9 | 0.3× | −11.58 |

Among interventions that keep a meaningful decisive rate (draw % below ~70%), **PK+xqQueen is the closest to parity at C:X = 1.2×**. Variants without the Chess Queen (Q only, noQ+*) push the C:X ratio near 1 but only by inflating the draw rate above 95%, which is not strategic balance but draw-degeneration.

---

## Factor Analysis

### Queen Configuration × Structural Reform (computed from last-10-iter averages)

| | Without PK | With PK |
|--|-----------|---------|
| **Chess Q / XQ no Q** | Default 8.9× (60% draw) | PK 3.3× (60% draw) |
| **Chess Q / XQ has Q** | X only 2.9× (69% draw) | **PK+xqQueen 1.2× (61% draw)** ⭐ |
| **No Chess Q / XQ no Q** | Q only 0.6× (98% draw) | noQ+PK 0.3× (95% draw) |

> A single-axis intervention is not enough. Adding `xq_queen` alone (X only) leaves a residual ~3× Chess advantage; adding `PK` alone leaves ~3.3×. **Combining `PK` and `xq_queen` is what moves the ratio into the 1.x band while keeping the draw rate comparable to Default.**
> Removing the Chess Queen pushes the ratio below 1 but at the cost of >95% draws — symptomatic of a degenerate, decision-poor game, not of strategic balance.

### xq_queen Stability (PK+xqQueen per-10-iter trend)

PK+xqQueen reaches its 1.2× steady state by iteration ~20 and stays in the 1.0–1.5× band thereafter (see `runs/fixed_v1/rq4_az_pk_xqqueen/metrics.csv`).

### Piece Survival Rate (PK+xqQueen variant, last 10 iters avg)

Survival denominators are variant-aware (xq_queen variants have 1 left-side XQ_QUEEN and 1 right-side Advisor at game start). See `surv_*` columns in `metrics.csv`.

---

## Cross-Variant Tournament (RQ3)

### Purpose

AZ agents trained under different rule variants compete against each other under **Default rules**, revealing how training conditions shape strategy.

### Configuration

- **Agent pool**: 9 variant `best_model.pt` (all 50-iter trained)
- **Play rules**: Default (standard Hybrid Chess, no reforms)
- **Games**: 36 pairs × 100 games per pair (50 games per color assignment) = **3,600 games**
- **Search**: 50 sims MCTS, C++ engine, 4 parallel workers
- **Action selection**: temperature-sampled visit counts (`temperature=0.5`) so games with the same (pair, color) but different seeds genuinely diverge — each of the 3,600 games is an independent sample.
- **Seeds**: deterministic `hashlib.sha256` per `(name_a, name_b, half, gi)` (reproducible across processes and sessions).
- **Duration**: ≈ 264 min
- **Output**: `runs/fixed_v1/cross_variant_tournament/` — `game_records.json`, `payoff_matrix.csv`, `wdl_matrix.csv`, `pairwise_ci.csv`, `summary.json`.

### Payoff Matrix

| | Default | Q_only | X_only | PK | PK_noPromo | PK_xqQueen | noQ_noPromo | noQ_PK | noQ_ALL |
|--|------|------|------|------|------|------|------|------|------|
| **Default** | 0.500 | 0.480 | 0.510 | 0.485 | 0.500 | 0.555 | 0.470 | 0.535 | 0.525 |
| **Q_only** | 0.520 | 0.500 | 0.510 | 0.515 | 0.545 | 0.505 | 0.500 | 0.595 | 0.560 |
| **X_only** | 0.490 | 0.490 | 0.500 | 0.425 | 0.505 | 0.520 | 0.465 | 0.550 | 0.520 |
| **PK** | 0.515 | 0.485 | 0.575 | 0.500 | 0.525 | 0.485 | 0.460 | 0.575 | 0.490 |
| **PK_noPromo** | 0.500 | 0.455 | 0.495 | 0.475 | 0.500 | 0.525 | 0.470 | 0.590 | 0.485 |
| **PK_xqQueen** | 0.445 | 0.495 | 0.480 | 0.515 | 0.475 | 0.500 | 0.455 | 0.505 | 0.540 |
| **noQ_noPromo** | 0.530 | 0.500 | 0.535 | 0.540 | 0.530 | 0.545 | 0.500 | 0.540 | 0.505 |
| **noQ_PK** | 0.465 | 0.405 | 0.450 | 0.425 | 0.410 | 0.495 | 0.460 | 0.500 | 0.480 |
| **noQ_ALL** | 0.475 | 0.440 | 0.480 | 0.510 | 0.515 | 0.460 | 0.495 | 0.520 | 0.500 |

### Agent Ranking

| Rank | Agent | Avg Score | Training Rules |
|------|-------|-----------|----------------|
| 1 | **Q_only** | 0.531 | Remove Chess Queen |
| 2 | **noQ_noPromo** | 0.528 | noQ + No Promotion |
| 3 | **PK** | 0.514 | Palace + Knight Block |
| 4 | Default | 0.508 | Standard Rules |
| 5 | PK_noPromo | 0.499 | PK + No Promotion |
| 6 | X_only | 0.496 | Give XQ a Queen |
| 7 | PK_xqQueen | 0.489 | PK + XQ Queen |
| 8 | noQ_ALL | 0.487 | All Restrictions |
| 9 | noQ_PK | 0.449 | noQ + PK |

> All 9 agents fall in a tight 0.449–0.531 band — under Default rules, **no agent strictly dominates**.

### Key Findings

#### 1. In-variant balance ≠ out-of-variant transfer

The variant with the best in-training balance (PK+xqQueen, in-variant C:X = 1.2×) is **not** the strongest agent under Default rules (rank 7). Conversely, agents trained under restrictive Chess variants (Q_only, noQ_noPromo) — which produce degenerate, draw-heavy self-play — transfer **best** to Default rules, presumably because those constrained training conditions force the network to learn stronger positional play that compensates for the missing Chess Queen at evaluation time.

#### 2. Strict non-transitive cycle

The tournament exhibits a strict rock-paper-scissors cycle among `PK`, `X_only`, and `PK_xqQueen`:

| Edge | Score | Direction |
|------|-------|-----------|
| PK vs X_only | 0.575 | PK > X_only |
| X_only vs PK_xqQueen | 0.520 | X_only > PK_xqQueen |
| PK_xqQueen vs PK | 0.515 | PK_xqQueen > PK |

All three pairwise scores exceed 0.50, forming a closed 3-cycle. This supports the RQ3 hypothesis that asymmetric rule design induces a multi-niche strategic landscape rather than a single linear strength ranking.

#### 3. Game-level diversity

With `temperature=0.5` action selection the tournament produces 19.4% draws and 80.6% decisive games (78.6% checkmate, 1.3% threefold repetition, 18.1% max-plies). The per-(pair, color) bucket shows on average 38 distinct (outcome, ply-count) combinations out of 50 games — confirming that the 100-game pair sample is composed of independent draws from the policy, not deterministic replicas.

---

## Recommended Variant

**`chess_palace + knight_block + xq_queen` (PK+xqQueen)** — the cleanest in-variant balance:
- In-variant C:X ≈ **1.2×** (closest to 1:1 among non-degenerate variants)
- Draw rate ~61% (comparable to Default, much lower than queen-removal variants at 95%+)
- Combines a structural restriction on Chess (palace + knight leg block) with a tactical Xiangqi resource (queen-like piece), instead of relying on a single-axis intervention.

A single-axis change (only `xq_queen`, only `PK`, or only `no_queen`) leaves either a noticeable residual Chess advantage or a near-100% draw rate.

---

## Training Command

```bash
# Single variant
python scripts/train_az_iter.py \
  --iterations 50 --selfplay-games-per-iter 100 --simulations 50 \
  --selfplay-max-ply 150 --batch-size 256 --train-epochs 2 \
  --eval-games 20 --eval-interval 2 --eval-simulations 100 \
  --disable-gating 1 --resign-enabled 1 --device auto --seed 42 \
  --ablation "chess_palace,knight_block,xq_queen" --use-cpp --num-workers 4 \
  --outdir runs/fixed_v1/rq4_az_pk_xqqueen

# All 9 variants sequentially with auto-resume + retry
python -m scripts.run_fixed_v1_all

# Live HTML progress dashboard (in another terminal)
python -m scripts.dashboard_fixed_v1
# then open runs/fixed_v1/progress.html in a browser (auto-refresh every 30s)

# Cross-variant tournament on the 9 best_model.pt
python -m scripts.cross_variant_tournament_fixed_v1 \
  --games 50 --sims 50 --workers 4 --temperature 0.5 --seed 42
```

---

## TODO

- [x] AB D2 rule-reform scan (23 variants)
- [x] AZ 9-variant training (50 iters × 100 games × 50 sims × 150 ply each)
- [x] Cross-variant tournament (3,600 games, temperature-sampled, deterministic seeds)
- [x] Factor analysis (Queen × PK)
- [x] Non-transitive cycle detection (PK > X_only > PK_xqQueen > PK)
- [x] All figures regenerated from data (`course_project/plot_figures_fixed_v1.R`)
- [ ] Final course report rewrite
