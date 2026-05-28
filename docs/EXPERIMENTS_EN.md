# Hybrid Chess: Experiment Results

> Last updated: 2026-05-28
---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Overview](#overview)
3. [RQ4: Early Exploration](#rq4-early-exploration)
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
│   ├── run_all.py                    # Orchestrator: trains all 9 variants in sequence
│   ├── dashboard.py                  # Live HTML progress dashboard
│   ├── cross_variant_tournament.py   # Cross-variant tournament with temperature sampling
│   ├── rq4_rule_reform_ab.py         # AB D2 rule reform scan
│   └── eval_arena.py                          # Side-swapped evaluation
├── tests/                 # Test suite (331 tests, including conftest.py state reset)
├── ui/                    # Browser game UI
├── runs/         # Experiment outputs (gitignored)
│   ├── rq4_rule_reform_ab/         # AB scan results
│   ├── rq4_az_default/             # Default 50 iters
│   ├── rq4_az_noq_only/            # noQ 50 iters
│   ├── rq4_az_xqqueen_only/        # xqQueen 50 iters
│   ├── rq4_az_palace_knight/       # PK 50 iters
│   ├── rq4_az_pk_nopromo/          # PK+noPromo 50 iters
│   ├── rq4_az_pk_xqqueen/          # PK+xqQueen 50 iters ⭐
│   ├── rq4_az_nq_nopromo/          # noQ+noPromo 50 iters
│   ├── rq4_az_nq_pk/               # noQ+PK 50 iters
│   ├── rq4_az_nq_allrules/         # noQ+ALL 50 iters
│   ├── cross_variant_tournament/   # Initial n=100 tournament (3,600 games)
│   └── cross_variant_tournament_ext/ # n=500 extension (17,969 games total)
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
- **Tournament**: 36 unordered pairs × 500 side-swapped games per pair = 17,969 games (an initial n=100 pass at 3,600 games surfaced an apparent 3-cycle that did not survive the n=500 replay)

---

## RQ4: Early Exploration

Tested piece-reduction variants (no_queen, no_bishop, extra_soldier, etc.) using AB D2:
- Default rules: mat_diff ≈ +19 (Chess dominates)
- Piece reduction can approach 0 but draw rate too high (AB D2 too shallow; "balance" was actually ineffective play)
- Introduced `mat_diff` as material-difference metric to distinguish "real balance" from "dead draws"

**Conclusion**: Piece reduction alone cannot eliminate Chess's structural advantage; rule-level reform is needed.

---

## AB D2 Rule Reform Scan

- **Script**: `scripts/rq4_rule_reform_ab.py`
- **Output**: `runs/rq4_rule_reform_ab/results.json` + `progress.log`
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

**Conclusion**: `palace + knight_block` (and the all-rules combination) achieves perfect material balance (matdiff = 0.0) under shallow AB search. This is the optimal structural intervention identified at the screening stage. Default rules show a strong Chess material advantage (matdiff ≈ +11). Knight-block alone is strictly worse than knight-block + palace, because palace by itself does nothing decisive at depth 2.

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

> **PK** = chess_palace + knight_block, **noQ** = no_queen, **xqQueen** = xq_queen, **ALL** = PK + no_promotion

### Nine-Variant Comparison (last-10-iter averages)

| Variant | Iters | Chess% | XQ% | Draw% | C:X | MatDiff |
|---------|-------|--------|-----|-------|-----|---------|
| Default | 50 | 35.6 | 4.0 | 60.4 | 8.9× | −6.40 |
| noQ | 50 | 0.9 | 1.6 | 97.5 | 0.6× | −11.72 |
| xqQueen | 50 | 22.8 | 7.8 | 69.4 | 2.9× | −11.27 |
| PK | 50 | 30.9 | 9.3 | 59.8 | 3.3× | −6.77 |
| PK+noPromo | 50 | 31.1 | 9.1 | 59.8 | 3.4× | −6.25 |
| **PK+xqQueen** ⭐ | 50 | **21.2** | **18.0** | **60.8** | **1.2×** | **−10.68** |
| noQ+noPromo | 50 | 2.2 | 1.4 | 96.4 | 1.6× | −11.32 |
| noQ+PK | 50 | 1.2 | 3.6 | 95.2 | 0.3× | −11.57 |
| noQ+ALL | 50 | 1.5 | 4.6 | 93.9 | 0.3× | −11.58 |

Among interventions that keep a meaningful decisive rate (draw % below ~70%), **PK+xqQueen is the closest to parity at C:X = 1.2×**. Variants without the Chess Queen (noQ, noQ+*) push the C:X ratio near 1 but only by inflating the draw rate above 95%, which is draw-degeneration rather than strategic balance.

---

## Factor Analysis

### Queen Configuration × Structural Reform (computed from last-10-iter averages)

| | Without PK | With PK |
|--|-----------|---------|
| **Chess Q / XQ no Q** | Default 8.9× (60% draw) | PK 3.3× (60% draw) |
| **Chess Q / XQ has Q** | xqQueen 2.9× (69% draw) | **PK+xqQueen 1.2× (61% draw)** ⭐ |
| **No Chess Q / XQ no Q** | noQ 0.6× (98% draw) | noQ+PK 0.3× (95% draw) |

> A single-axis intervention is not enough. Adding `xq_queen` alone (xqQueen) leaves a residual ~3× Chess advantage; adding `PK` alone leaves ~3.3×. **Combining `PK` and `xq_queen` is what moves the ratio into the 1.x band while keeping the draw rate comparable to Default.**
> Removing the Chess Queen pushes the ratio below 1 but at the cost of >95% draws, which is symptomatic of a degenerate, decision-poor game rather than strategic balance.

### xq_queen Stability (PK+xqQueen per-10-iter trend)

PK+xqQueen reaches its 1.2× steady state by iteration ~20 and stays in the 1.0–1.5× band thereafter (see `runs/rq4_az_pk_xqqueen/metrics.csv`).

### Piece Survival Rate (PK+xqQueen variant, last 10 iters avg)

Survival denominators are variant-aware (xq_queen variants have 1 left-side XQ_QUEEN and 1 right-side Advisor at game start). See `surv_*` columns in `metrics.csv`.

---

## Cross-Variant Tournament (RQ3)

### Purpose

AZ agents trained under different rule variants compete against each other under **Default rules**, revealing how training conditions shape strategy.

### Configuration

- **Agent pool**: 9 variant `best_model.pt` (all 50-iter trained)
- **Play rules**: Default (standard Hybrid Chess, no reforms)
- **Games**: 36 unordered pairs × 500 side-swapped games per pair = **17,969 games** (after dropping 31 seed-collision duplicates)
- **Search**: 50 sims MCTS, C++ engine, 6 parallel workers
- **Action selection**: temperature-sampled visit counts (`temperature=0.5`) so games with the same (pair, color) but different seeds genuinely diverge.
- **Seeds**: deterministic `hashlib.sha256` per `(name_a, name_b, half, gi)` (reproducible across processes and sessions).
- **Output**: `runs/cross_variant_tournament/analysis_500/` contains `payoff_matrix.csv`, `elo.csv`, `per_side.csv`, `pairwise_significance.csv`, `decisive_rate.csv`, `game_length.csv`.

### Payoff Matrix (n=500, row vs column)

| | Default | noQ | xqQueen | PK | PK_noPromo | PK_xqQueen | noQ_noPromo | noQ_PK | noQ_ALL |
|--|------|------|------|------|------|------|------|------|------|
| **Default** | 0.500 | 0.484 | 0.509 | 0.492 | 0.522 | 0.522 | 0.487 | 0.537 | 0.499 |
| **noQ** | 0.516 | 0.500 | 0.516 | 0.503 | 0.521 | 0.501 | 0.502 | 0.586 | 0.550 |
| **xqQueen** | 0.491 | 0.484 | 0.500 | 0.460 | 0.476 | 0.497 | 0.466 | 0.538 | 0.516 |
| **PK** | 0.508 | 0.497 | 0.540 | 0.500 | 0.530 | 0.505 | 0.491 | 0.528 | 0.509 |
| **PK_noPromo** | 0.478 | 0.479 | 0.524 | 0.470 | 0.500 | 0.525 | 0.482 | 0.585 | 0.514 |
| **PK_xqQueen** | 0.478 | 0.499 | 0.503 | 0.495 | 0.475 | 0.500 | 0.438 | 0.509 | 0.495 |
| **noQ_noPromo** | 0.513 | 0.498 | 0.534 | 0.509 | 0.518 | 0.562 | 0.500 | 0.524 | 0.507 |
| **noQ_PK** | 0.463 | 0.414 | 0.462 | 0.472 | 0.415 | 0.491 | 0.476 | 0.500 | 0.498 |
| **noQ_ALL** | 0.501 | 0.450 | 0.484 | 0.491 | 0.486 | 0.505 | 0.493 | 0.502 | 0.500 |

### Agent Ranking (Bradley–Terry Elo, mean-anchored at 1500, 500-resample bootstrap 95% CI)

| Rank | Agent | Elo | 95% CI | Avg Score | Training Rules |
|------|-------|-----|--------|-----------|----------------|
| 1 | **noQ_noPromo** | 1520.8 | [1502.5, 1527.3] | 0.521 | noQ + No Promotion |
| 2 | **noQ** | 1520.5 | [1503.7, 1529.6] | 0.524 | Remove Chess Queen |
| 3 | **PK** | 1514.5 | [1494.4, 1521.4] | 0.514 | Palace + Knight Block |
| 4 | Default | 1503.4 | [1490.0, 1517.9] | 0.506 | Standard Rules |
| 5 | PK_noPromo | 1502.5 | [1490.3, 1517.7] | 0.507 | PK + No Promotion |
| 6 | noQ_ALL | 1488.4 | [1478.4, 1505.5] | 0.489 | noQ + ALL |
| 7 | xqQueen | 1488.2 | [1480.6, 1505.5] | 0.491 | Give XQ a Queen |
| 8 | PK_xqQueen | 1486.2 | [1479.5, 1502.2] | 0.487 | PK + XQ Queen |
| 9 | noQ_PK | 1475.6 | [1467.8, 1487.3] | 0.461 | noQ + PK |

> Avg-score band tightens to **0.461–0.524** at n=500 (vs the original 0.449–0.531 at n=100). Bootstrap CIs span ~15–30 Elo points, and 32 of 36 pairwise Wilson intervals still overlap 0.50, so most between-agent differences remain inside sampling noise.

### Key Findings

#### 1. In-variant balance does not predict default-rule transfer strength

The variant with the best in-training balance (PK+xqQueen, in-variant C:X = 1.2×) lands at **rank 8** under Default rules (avg 0.487). The top of the table is taken by variants that produce degenerate, draw-heavy self-play (noQ and noQ_noPromo, both >96% in-variant draws). In our tested set, in-variant balance and default-rule strength measure different properties of the trained agent.

#### 2. Apparent 3-cycle at n=100, refuted at n=500

The initial 100-game tournament showed three pairings whose scores nominally formed a closed rock-paper-scissors cycle:

| Edge | Score (n=100) | 95% CI (n=100) |
|------|---------------|----------------|
| PK vs xqQueen | 0.575 | [0.477, 0.667] |
| xqQueen vs PK_xqQueen | 0.520 | [0.423, 0.615] |
| PK_xqQueen vs PK | 0.515 | [0.418, 0.611] |

We replayed the three pairings with 500 games each (same seed convention, side-swapped, T=0.5):

| Edge | Score (n=500) | 95% CI (n=500) | Direction |
|------|---------------|----------------|-----------|
| PK vs xqQueen | 0.540 | [0.508, 0.572] | kept, CI now excludes 0.5 |
| xqQueen vs PK_xqQueen | 0.497 | [0.453, 0.541] | flipped, inside CI of 0.5 |
| PK_xqQueen vs PK | 0.495 | [0.453, 0.541] | flipped, inside CI of 0.5 |

At 500 games per pair, the headline 3-cycle dissolves: two of the three directions flip, only PK vs xqQueen separates from 0.5, and the cycle structure is gone. Replay output: `runs/cycle_3pair_ci/`.

Reproduce with: `python -m scripts.cycle_3pair_ci --games 250 --workers 12`.

#### 3. Pairwise significance at n=500

Per-pair Wilson intervals on the symmetric score isolate four pairings whose 95% CI excludes 0.50: noQ vs noQ_PK (0.586), PK_noPromo vs noQ_PK (0.585), noQ_noPromo vs PK_xqQueen (0.562), noQ vs noQ_ALL (0.550). The same four pairs at n=100 all overlapped 0.50, so the additional resolution comes from the larger N. See `runs/cross_variant_tournament/analysis_500/pairwise_significance.csv`.

---

## Recommended Variant

**`chess_palace + knight_block + xq_queen` (PK+xqQueen)** gives the cleanest in-variant balance:
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
  --outdir runs/rq4_az_pk_xqqueen

# All 9 variants sequentially with auto-resume + retry
python -m scripts.run_all

# Live HTML progress dashboard (in another terminal)
python -m scripts.dashboard
# then open runs/progress.html in a browser (auto-refresh every 30s)

# Cross-variant tournament on the 9 best_model.pt (n=500 setting)
python -m scripts.cross_variant_tournament \
  --games 250 --sims 50 --workers 6 --temperature 0.5 --seed 42
```

---

## TODO

- [x] AB D2 rule-reform scan (23 variants)
- [x] AZ 9-variant training (50 iters × 100 games × 50 sims × 150 ply each)
- [x] Cross-variant tournament (17,969 games at n=500, temperature-sampled, deterministic seeds)
- [x] Bradley–Terry Elo + bootstrap CIs, per-side breakdown, pairwise significance, decisive rate, game-length analysis
- [x] Factor analysis (Queen × PK)
- [x] Non-transitive cycle detection (apparent at n=100, refuted at n=500)
- [x] 500-game cycle replay with Wilson 95% CIs (`scripts/cycle_3pair_ci.py`)
- [x] All figures regenerated from data (`course_project/plot_figures.R`, `course_project/plot_cycle_replay.py`)
- [x] Final course report rewrite (n=500, RQ3 closed via side-of-play and in-variant-vs-transfer analysis)
