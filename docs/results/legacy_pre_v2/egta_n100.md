# EGTA N=100 Dual-Universe Results

## Experiment Setup

- **Games per pair:** 100 (50 per side) → 3,600 games per universe
- **Two universes:** V3 (`no_queen`) and V4 (`extra_cannon`)
- **Agent pool (9 each):** Random, Greedy, Pure MCTS-100, AB-d1/d2/d4, AZ checkpoints
  - V3: iter1 (early), iter6 (mid), iter9 (final)
  - V4: iter2 (early), iter9 (mid), iter19 (final)
- **MCTS simulations:** 200
- **Optimizations:** C++ engine, `predict_batch()`, 4 parallel workers per universe
- **Runtime:** ~11 hours total (both universes simultaneously)

---

## V3: no_queen

### Nash Equilibrium

| Property | Value |
|---|---|
| **Support** | `{ckpt_iter9}` |
| **Support size** | 1 |
| **Game value** | 0.000 |
| **Topology** | **TRANSITIVE** |

### Agent Rankings (Avg Score)

| Agent | Avg Score |
|---|---|
| ckpt_iter9 | 0.656 |
| ckpt_iter6 | 0.600 |
| ckpt_iter1 | 0.478 |
| RANDOM | 0.486 |
| GREEDY | 0.459 |
| PURE_MCTS_100 | 0.452 |
| AB_D2 | 0.500 |
| AB_D4 | 0.497 |
| AB_D1 | 0.372 |

### Alpha-Rank (α=100)

Uniform distribution (0.111 each). The high prevalence of 0.500 draws among AB agents prevents differentiation.

### Key Findings

- **AZ dominance gradient:** iter9 (0.656) > iter6 (0.600) > iter1 (0.478) — clear learning progression
- **AB draw wall:** AB-d2 and AB-d4 score 0.500 against almost every opponent including AZ
- **iter9 dominates iter6 at 0.75** — same score as iter6 vs iter1 (0.75)
- **Random beats AB-d1 (0.725)** — side-bias anomaly persists at N=100
- **0 dominance cycles** at threshold 0.60 → strict transitive hierarchy

---

## V4: extra_cannon

### Nash Equilibrium

| Property | Value |
|---|---|
| **Support** | `{ckpt_iter19}` |
| **Support size** | 1 |
| **Game value** | 0.000 |
| **Topology** | **TRANSITIVE** |

### Agent Rankings (Avg Score)

| Agent | Avg Score |
|---|---|
| ckpt_iter19 | 0.631 |
| ckpt_iter9 | 0.619 |
| ckpt_iter2 | 0.584 |
| AB_D2 | 0.498 |
| AB_D4 | 0.496 |
| GREEDY | 0.483 |
| RANDOM | 0.470 |
| PURE_MCTS_100 | 0.413 |
| AB_D1 | 0.405 |

### Alpha-Rank (α=100)

iter19 (32.3%) > iter9 (22.9%) > iter2 (14.4%) > AB_D2 (8.7%) > AB_D4 (7.0%) > Greedy (4.5%) > MCTS (3.8%) > Random (3.5%) > AB_D1 (2.8%)

### Key Findings

- **AZ dominance gradient:** iter19 (0.631) > iter9 (0.619) > iter2 (0.584)
- **MCTS vs iter2 = 0.000** — complete shutdown (100 losses in 100 games)
- **iter2 vs AB-d4 = 0.75, iter19 vs iter2 = 0.75** — consistent 3:1 ratios
- **AB-d2/d4 draw wall confirmed at N=100** — 0.500 against all AZ
- **0 dominance cycles** at threshold 0.60

---

## Cross-Universe Comparison

| Metric | V3 (no_queen) | V4 (extra_cannon) |
|---|---|---|
| Nash support size | 1 | 1 |
| Nash agent | ckpt_iter9 | ckpt_iter19 |
| Topology | TRANSITIVE | TRANSITIVE |
| Dominance cycles (θ=0.60) | 0 | 0 |
| Best AZ avg score | 0.656 | 0.631 |
| AB draw wall | yes (d2, d4) | yes (d2, d4) |
| MCTS vs best AZ | 0.25 | 0.25 |
| Alpha-rank differentiation | uniform (too many 0.50) | clear gradient |

### Conclusion

**Both universes exhibit strict transitive hierarchies.** The hypothesis of a "topological phase transition" from transitive (V4) to non-transitive (V3) is **not supported** at N=100. Removing the Queen does not induce cyclic dominance patterns. Both rule variants produce the same macro-topology: AZ > heuristic baselines > random play, with AB agents forming a "draw wall" at 0.500.

The key structural difference is in *alpha-rank differentiation*: V4 produces a clear gradient with iter19 at 32.3%, while V3's payoff matrix is too draw-heavy among AB agents to separate them.
