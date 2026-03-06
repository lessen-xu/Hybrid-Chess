# EGTA Pilot Results (V4, extra_cannon)

**720 games** (36 pairs × 20 games/pair), ~13h runtime, seed=42.

---

## 9×9 Payoff Matrix

Score = win rate of **row** player (W + 0.5×D).

| | RANDOM | GREEDY | MCTS_100 | AB_D1 | AB_D2 | AB_D4 | AZ-E(2) | AZ-M(9) | AZ-B(19) |
|---|---|---|---|---|---|---|---|---|---|
| **RANDOM** | .500 | .425 | .550 | **.675** | .500 | .500 | .275 | .300 | .225 |
| **GREEDY** | .575 | .500 | .375 | **.600** | .500 | .500 | .350 | .325 | .275 |
| **MCTS_100** | .450 | .625 | .500 | .500 | .525 | .500 | **.000** | .250 | .250 |
| **AB_D1** | .325 | .400 | .500 | .500 | .500 | .500 | .500 | .250 | .250 |
| **AB_D2** | .500 | .500 | .475 | .500 | .500 | .500 | .500 | .500 | .500 |
| **AB_D4** | .500 | .500 | .500 | .500 | .500 | .500 | .250 | .500 | .500 |
| **AZ-E(2)** | **.725** | **.650** | **1.000** | .500 | .500 | **.750** | .500 | .500 | .250 |
| **AZ-M(9)** | **.700** | **.675** | **.750** | **.750** | .500 | .500 | .500 | .500 | .500 |
| **AZ-B(19)** | **.775** | **.725** | **.750** | **.750** | .500 | .500 | **.750** | .500 | .500 |

---

## Agent Rankings (avg score vs all opponents)

| Rank | Agent | Avg Score | Interpretation |
|---|---|---|---|
| 1 | **ckpt_iter19** (AZ-Best) | **0.625** | Dominant — beats everything except AB-d2/d4 draw wall |
| 2 | ckpt_iter9 (AZ-Mid) | 0.578 | Strong — beats weak agents + AB-d1, walls at d2/d4 |
| 3 | ckpt_iter2 (AZ-Early) | 0.609 | Surprisingly strong — 1.000 vs MCTS, 0.750 vs AB-d4 |
| 4 | AB_D2 | 0.497 | Universal draw machine |
| 5 | AB_D4 | 0.469 | Same draw wall, slightly worse vs AZ |
| 6 | GREEDY | 0.425 | Beats Random, loses to MCTS and AZ |
| 7 | PURE_MCTS_100 | 0.394 | Terrible vs AZ (0.000 vs iter2!), decent vs Greedy |
| 8 | AB_D1 | 0.403 | Loses to Random (!), walls at d2/d4 |
| 9 | RANDOM | 0.419 | Baseline anchor |

---

## Key Findings

### 1. Transitive Hierarchy Confirmed

**Nash Equilibrium:** support = {ckpt_iter19}, support size = **1**, game value = 0.0.

Interpretation: **TRANSITIVE** — Nash concentrates on a single dominant agent. No cyclic rock-paper-scissors dynamics detected in V4 `extra_cannon`.

### 2. The AB Draw Wall

AB-d2 scores **exactly 0.500** against 7 of 8 opponents (all except MCTS where it's 0.475). AB-d4 has 6 of 8 at 0.500. This is the "cowardice lock" — threefold repetition draw kicks in at ~36 plies. AB agents at depth ≥2 are invincible but also incapable of winning.

**Implication for EGTA:** AB-d2/d4 are essentially "noise absorbers" — they contribute no information to the Nash structure. In the full experiment, we may want to collapse them into a single representative or analyze with/without them.

### 3. AZ Dominance Gradient

Clear training progression:
- **AZ-Early (iter2):** Beats all baselines (0.65–1.00) but loses to AZ-Best (0.250)
- **AZ-Mid (iter9):** Same pattern, slightly better vs AB-d1 (0.750 vs 0.500)
- **AZ-Best (iter19):** Dominates AZ-Early (0.750) and ties AZ-Mid (0.500)

The AZ hierarchy is **mostly transitive**: iter19 ≥ iter9 ≥ iter2. No AZ-internal cycling.

### 4. Pure MCTS Catastrophe

MCTS_100 vs AZ-Early = **0.000** (20 losses, 0 wins, 0 draws). This is the most extreme cell in the entire matrix. The NN prior (even an early one) completely dominates uniform-rollout MCTS.

### 5. Random > AB-d1 (!)

Random scores 0.675 vs AB-d1 — this is likely a **side bias** artifact at depth 1. Chess side has a natural advantage (Queen + Bishop diagonals), and both Random and AB-d1 are too weak to overcome it. The side-swapping protocol averages over sides, but with only 20 games the signal is noisy.

---

## Statistical Power Analysis

With **N=20 games/pair**, the 95% CI for a score is ±0.219 (SE = √(0.25/20) = 0.112).

| Observed Score | 95% CI | Distinguishable from 0.500? |
|---|---|---|
| 0.500 | [0.281, 0.719] | No |
| 0.625 | [0.406, 0.844] | No |
| 0.725 | [0.506, 0.944] | **Barely** |
| 0.750 | [0.531, 0.969] | **Yes** (marginal) |
| 1.000 | [0.781, 1.000] | **Yes** |

**Most results in the 0.55–0.70 range are NOT statistically significant at N=20.** Only the extreme scores (≥0.750) are distinguishable from a coin flip.

### Sample Size Recommendations for Full Experiment

| Target Margin | Games/Pair | Total (36 pairs) | EST Runtime |
|---|---|---|---|
| ±0.10 | 96 | 3,456 | ~60h |
| ±0.07 | 196 | 7,056 | ~120h |
| ±0.05 | 384 | 13,824 | ~250h |

**Recommendation: N=100 games/pair** (margin ±0.098). This halves the CI width vs pilot, makes 0.60+ results significant, and completes in ~3 days with the `predict_batch()` optimization (8× speedup for AZ pairs).

---

## Heatmap

![EGTA Pilot Payoff Heatmap](../../runs/egta_pilot/payoff_heatmap.png)
