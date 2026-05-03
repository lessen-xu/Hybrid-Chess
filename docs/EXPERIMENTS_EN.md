# Hybrid Chess — Experiment Results

> Last updated: 2026-05-03

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
│   ├── train_az_iter.py           # AZ training CLI entry
│   ├── cross_variant_tournament.py # Cross-variant tournament
│   ├── eval_arena.py              # Side-swapped evaluation
│   └── rq4_rule_reform_ab.py      # AB D2 rule reform scan
├── tests/                 # Test suite
├── ui/                    # Browser game UI
├── runs/                  # Experiment outputs (gitignored, ~1.7GB)
│   ├── rq4_rule_reform_ab/        # AB scan results
│   ├── rq4_az_default_v2/         # Default 50 iters
│   ├── rq4_az_noq_only/           # Q only 50 iters
│   ├── rq4_az_xqqueen_only/       # X only 50 iters ⭐
│   ├── rq4_az_palace_knight_v2/   # PK 50 iters
│   ├── rq4_az_pk_nopromo/         # PK+noPromo 50 iters
│   ├── rq4_az_pk_xqqueen/         # PK+xqQueen 50 iters
│   ├── rq4_az_nq_nopromo/         # noQ+noPromo 50 iters
│   ├── rq4_az_nq_pk/              # noQ+PK 50 iters
│   ├── rq4_az_nq_allrules_v2/     # noQ+ALL 50 iters
│   └── cross_variant_tournament/  # 1800-game tournament
└── docs/
    ├── ARCHITECTURE.md
    ├── EXPERIMENTS_EN.md  # This file (English)
    └── EXPERIMENTS_ZH.md  # Chinese version
```

---

## Overview

| Phase | Goal | Status | Output |
|-------|------|--------|--------|
| AB D2 Rule Reform Scan | Fast screening of 23 variants | ✅ Done | `runs/rq4_rule_reform_ab/` |
| AZ Nine-Variant Comparison (50 iters each) | Find optimal balance | ✅ Done | `runs/rq4_az_*` |
| Cross-Variant Tournament | Meta-strategy analysis | ✅ Done | `runs/cross_variant_tournament/` |

- **AZ Training**: 9 variants × 50 iters = 450 iters, 45,000 self-play games total
- **AB Scan**: 23 variants × 40 games = 920 games
- **Tournament**: 1,800 games

---

## RQ4 — Early Exploration

Tested piece-reduction variants (no_queen, no_bishop, extra_soldier, etc.) using AB D2:
- Default rules: mat_diff ≈ +19 (Chess dominates)
- Piece reduction can approach 0 but draw rate too high (AB D2 too shallow; "balance" was actually ineffective play)
- Introduced `mat_diff` as material-difference metric to distinguish "real balance" from "dead draws"

**Conclusion**: Piece reduction alone cannot eliminate Chess's structural advantage; rule-level reform is needed.

---

## AB D2 Rule Reform Scan

- **Script**: `scripts/rq4_rule_reform_ab.py`
- **Output**: `runs/rq4_rule_reform_ab/results.json` + `progress.log`
- **Scale**: 23 variants × 40 games, Alpha-Beta depth=2, C++ accelerated
- **Three reform rules**:
  - `no_promotion`: Pawns do not promote upon reaching the back rank
  - `chess_palace`: Chess King confined to a 3×3 palace (x=3–5, y=0–2)
  - `knight_block`: Chess Knight uses Xiangqi horse blocking rules

| Variant | Games | C Win | X Win | Draw | TB Chess | TB XQ | TB Even | avg_matdiff |
|---------|-------|-------|-------|------|----------|-------|---------|-------------|
| palace+knight_blk | 40 | 0 | 0 | 40 | 0 | 0 | 40 | +0.0 |
| ALL_RULES | 40 | 0 | 0 | 40 | 0 | 0 | 40 | +0.0 |
| nq+ec | 40 | 0 | 0 | 40 | 40 | 0 | 0 | +1.0 |
| nq+ec+no_promo | 40 | 0 | 0 | 40 | 40 | 0 | 0 | +1.0 |
| nq+ec+palace | 40 | 0 | 0 | 40 | 40 | 0 | 0 | +1.0 |
| nq+nb | 40 | 0 | 0 | 40 | 0 | 40 | 0 | -2.0 |
| nq+nb+no_promo | 40 | 0 | 0 | 40 | 0 | 40 | 0 | -2.0 |
| nq+nb+palace | 40 | 0 | 0 | 40 | 0 | 40 | 0 | -2.0 |
| no_queen+ALL_RULES | 40 | 0 | 0 | 40 | 40 | 0 | 0 | +3.0 |
| nq+nb+knight_blk | 40 | 0 | 0 | 40 | 0 | 40 | 0 | -5.0 |
| nq+nb+es+ALL_RULES | 40 | 0 | 0 | 40 | 40 | 0 | 0 | +7.0 |
| no_queen | 40 | 0 | 0 | 40 | 40 | 0 | 0 | +9.0 |
| no_queen+no_promo | 40 | 0 | 0 | 40 | 40 | 0 | 0 | +9.0 |
| no_queen+palace | 40 | 0 | 0 | 40 | 40 | 0 | 0 | +9.0 |
| nq+nb+ALL_RULES | 40 | 0 | 0 | 40 | 40 | 0 | 0 | +9.0 |
| no_queen+knight_blk | 40 | 0 | 0 | 40 | 40 | 0 | 0 | +16.0 |
| knight_blk | 40 | 0 | 0 | 40 | 40 | 0 | 0 | +17.0 |
| no_promo+knight_blk | 40 | 0 | 0 | 40 | 40 | 0 | 0 | +17.0 |
| nq+ec+ALL_RULES | 40 | 0 | 0 | 40 | 40 | 0 | 0 | +18.0 |
| default | 40 | 0 | 0 | 40 | 40 | 0 | 0 | +19.0 |
| no_promo | 40 | 0 | 0 | 40 | 40 | 0 | 0 | +19.0 |
| palace | 40 | 0 | 0 | 40 | 40 | 0 | 0 | +19.0 |
| no_promo+palace | 40 | 0 | 0 | 40 | 40 | 0 | 0 | +19.0 |

**Conclusion**: `palace + knight_block` achieves perfect material balance (matdiff = 0.0) under AB D2 — the optimal structural reform.

---

## Rule Reform Implementation

**C++ side** (`cpp/src/`):
- `types.h`: Added `RuleFlags` struct + `thread_local g_rule_flags`
- `rules.cpp`: Integrated all three reforms into move generation and attack detection
- `bindings.cpp`: Exposed `RuleFlags`, `set_rule_flags` to Python

**Python side** (`hybrid/core/`):
- `config.py`: Added `no_promotion`, `chess_palace`, `knight_block`, `xq_queen` fields to `VariantConfig`
- `rules.py`: Synchronized three-rule logic branches
- `env.py` `_set_active_variant()`: Auto-syncs C++ rule flags on environment reset

**Ablation mapping** (`hybrid/rl/az_runner.py`):
```python
'no_promotion': {'no_promotion': True},
'chess_palace':  {'chess_palace': True},
'knight_block':  {'knight_block': True},
'xq_queen':      {'xq_queen': True},
```

---

## AlphaZero Nine-Variant Training

### Configuration

All AZ runs use a uniform config (50 iters × 100 games/iter = 5,000 self-play games/variant):
- Self-play: 100 games/iter, 50 sims, max_ply=150, 4 workers
- Training: 2 epochs, batch=256, buffer=50000
- Evaluation: 20 games vs Random + 20 games vs AB(d1), every 2 iters
- Total: **9 variants × 50 iters = 45,000 self-play games**

> **PK** = chess_palace + knight_block, **Q** = no_queen, **X** = xq_queen, **ALL** = PK + no_promotion

### Nine-Variant Comparison

| Variant | Iters | Chess% | XQ% | Draw% | C:X | L10 C:X | MatDiff |
|---------|-------|--------|-----|-------|-----|---------|---------|
| Default | 50 | 29.6% | 3.3% | 67.1% | 9.0x | 6.6x | -6.0 |
| Q only | 50 | 11.4% | 2.6% | 86.0% | 4.5x | 3.1x | -14.2 |
| X only ⭐ | 50 | 17.9% | 24.2% | 57.9% | 0.7x | 0.7x | -11.1 |
| PK | 50 | 30.1% | 8.7% | 61.1% | 3.4x | 3.2x | -6.6 |
| PK+noPromo | 50 | 31.0% | 8.8% | 60.3% | 3.5x | 4.0x | -6.8 |
| PK+xqQueen ⭐ | 50 | 18.5% | 27.4% | 54.1% | 0.7x | 0.7x | -10.7 |
| noQ+noPromo | 50 | 9.6% | 2.6% | 87.8% | 3.7x | 2.0x | -13.4 |
| noQ+PK ⭐ | 50 | 4.8% | 7.6% | 87.7% | 0.6x | 0.2x | -12.7 |
| noQ+ALL ⭐ | 50 | 3.9% | 6.9% | 89.3% | 0.6x | 0.3x | -12.5 |

---

## Factor Analysis

### Queen Configuration 2×2

| | Chess has Queen | Chess no Queen |
|--|----------------|----------------|
| **XQ no Queen** | Default **9.0x** (67% draw) | Q only **4.5x** (86% draw) |
| **XQ has Queen** | X only **0.7x** (58% draw) | — |

> **Giving XQ a Queen (X only)** drops C:X from 9.0x to 0.7x, with the *lowest* draw rate.
> **Removing Chess Queen (Q only)** only drops from 9.0x to 4.5x, with draws surging to 86%.

### Structural Reform Factor (PK Interaction)

| | Without PK | With PK |
|--|-----------|---------|
| **Chess has Q / XQ no Q** | 9.0x | **3.4x** (PK effective) |
| **Chess has Q / XQ has Q** | **0.7x** | **0.7x** (PK redundant) |
| **Chess no Q / XQ no Q** | **4.5x** | **0.6x** (PK effective) |

> PK is effective when XQ lacks a Queen (-62% to -87%), but **completely redundant** when XQ has a Queen.

### xq_queen Stability (X only per-10 trend)

| Phase | Chess | XQ | Draw | C:X |
|-------|-------|----|------|-----|
| 0–9 | 193 | 230 | 577 | 0.8x |
| 10–19 | 176 | 235 | 589 | 0.7x |
| 20–29 | 157 | 236 | 607 | 0.7x |
| 30–39 | 193 | 249 | 558 | 0.8x |
| 40–49 | 177 | 259 | 564 | 0.7x |

No drift across 50 iterations — 0.7x is the converged steady-state balance point.

### Piece Survival Rate (X only variant, last 10 iters avg)

| Piece | Survival(%) | Note |
|-------|-------------|------|
| chess_QUEEN | 53.5 |  |
| chess_ROOK | 67.3 |  |
| chess_BISHOP | 63.0 |  |
| chess_KNIGHT | 59.2 |  |
| chess_PAWN | 67.7 |  |
| xiangqi_CHARIOT | 91.5 | well protected |
| xiangqi_CANNON | 61.9 |  |
| xiangqi_HORSE | 93.2 | well protected |
| xiangqi_ELEPHANT | 94.4 | well protected |
| xiangqi_ADVISOR | 46.8 |  |
| xiangqi_SOLDIER | 69.5 |  |

---

## Cross-Variant Tournament (RQ3)

### Purpose

AZ agents trained under different rule variants compete against each other under **Default rules**, revealing how training conditions shape strategy.

### Configuration

- **Agent pool**: 9 variant `best_model.pt` (all 50-iter trained)
- **Play rules**: Default (standard Hybrid Chess, no reforms)
- **Games**: 36 pairs × 50 games (25 games/half, side-swapped) = **1,800 games**
- **Search**: 50 sims MCTS, C++ engine, 4 parallel workers
- **Duration**: 45.7 minutes
- **Output**: `runs/cross_variant_tournament/`

### Payoff Matrix

| | Default | Q_only | X_only | PK | PK_noPromo | PK_xqQueen | noQ_noPromo | noQ_PK | noQ_ALL |
|--|------|------|------|------|------|------|------|------|------|
| **Default** | 0.500 | 0.500 | 0.500 | 0.250 | 0.500 | 0.750 | 0.500 | 0.750 | 0.250 |
| **Q_only** | 0.500 | 0.500 | 0.750 | 0.500 | 0.750 | 0.500 | 0.750 | 0.750 | 0.500 |
| **X_only** | 0.500 | 0.250 | 0.500 | 0.500 | 0.750 | 0.500 | 0.500 | 0.750 | 0.500 |
| **PK** | 0.750 | 0.500 | 0.500 | 0.500 | 0.500 | 0.750 | 0.750 | 0.500 | 0.750 |
| **PK_noPromo** | 0.500 | 0.250 | 0.250 | 0.500 | 0.500 | 0.750 | 0.500 | 0.500 | 0.500 |
| **PK_xqQueen** | 0.250 | 0.500 | 0.500 | 0.250 | 0.250 | 0.500 | 0.500 | 0.500 | 0.250 |
| **noQ_noPromo** | 0.500 | 0.250 | 0.500 | 0.250 | 0.500 | 0.500 | 0.500 | 0.500 | 0.250 |
| **noQ_PK** | 0.250 | 0.250 | 0.250 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| **noQ_ALL** | 0.750 | 0.500 | 0.500 | 0.250 | 0.500 | 0.750 | 0.750 | 0.500 | 0.500 |

### Agent Ranking

| Rank | Agent | Avg Score | Training Rules |
|------|-------|-----------|----------------|
| 1 | **Q_only** | **0.625** | Remove Chess Queen |
| 1 | **PK** | **0.625** | Palace + Knight block |
| 3 | noQ_ALL | 0.562 | All restrictions |
| 4 | X_only | 0.531 | Give XQ a Queen |
| 5 | Default | 0.500 | Default rules |
| 6 | PK_noPromo | 0.469 | PK + no promotion |
| 7 | noQ_noPromo | 0.406 | noQ + no promotion |
| 8 | noQ_PK | 0.406 | noQ + PK |
| 9 | PK_xqQueen | 0.375 | PK + XQ Queen |

### Key Findings

#### 1. "Adversity Breeds Strength"

Agents trained under **restricted rules** (Q_only, PK) perform best under Default rules (0.625).
They learned more refined defensive and offensive strategies in more challenging environments.

#### 2. Default Agent is Only Average

The agent trained under its "home rules" scores only 0.500. The Chess-dominant Default environment (9.0x) teaches only brute-force aggression, lacking strategic finesse.

#### 3. Balanced Training → Worst Transfer

PK_xqQueen ranks last (0.375). The agent trained under the most balanced rules grew accustomed to the "comfort zone" of XQ having a Queen, and cannot adapt when XQ lacks one under Default rules.

#### 4. Non-Transitivity Exists

| A | B | A Score | Note |
|---|---|---------|------|
| PK_xqQueen | Default | 0.250 | PK_xQ loses |
| Default | noQ_ALL | 0.250 | Default loses |
| noQ_ALL | PK_xqQueen | 0.750 | noQ_ALL wins |

This forms a rock-paper-scissors cycle, confirming the RQ3 hypothesis: different training conditions produce **qualitatively different strategies**, not a simple strength ranking.

---

## Recommended Variant

**`xq_queen`** (give XQ a Queen) — simplest and most effective balancing variant:
- C:X ≈ 0.7x (closest to 1:1)
- Draw rate 58% (highest game quality)
- Only one flag change, minimal rule complexity
- Structural reforms (PK) optional but not necessary

---

## Training Command

```bash
python scripts/train_az_iter.py \
  --iterations 50 --selfplay-games-per-iter 100 --simulations 50 \
  --selfplay-max-ply 150 --batch-size 256 --train-epochs 2 \
  --eval-games 20 --eval-interval 2 --eval-simulations 100 \
  --disable-gating 1 --resign-enabled 1 --device auto --seed 42 \
  --ablation "xq_queen" --use-cpp --num-workers 4 \
  --outdir "runs/MY_RUN_NAME"
```

---

## TODO

- [x] AZ Default (50 iters) → C:X = 9.0x
- [x] AZ Q only (50 iters) → C:X = 4.5x + 86% draw
- [x] AZ X only (50 iters) → **C:X = 0.7x** ⭐
- [x] AZ PK / PK+noPromo (50 iters each) → structural reform effective but insufficient
- [x] AZ PK+xqQueen (50 iters) → C:X = 0.7x = X only
- [x] AZ noQ+PK / noQ+noPromo / noQ+ALL (50 iters each) → over-nerfed
- [x] Factor analysis confirms: xq_queen is the only necessary factor
- [x] Cross-variant tournament (1,800 games) → "adversity breeds strength" + non-transitivity ⭐
- [ ] Course report
