# Hybrid Chess Balance Protocol: Methodological Charter

## 1. Philosophical Foundations: Operational vs. Theoretical Balance

In deterministic, perfect-information board games like Hybrid Chess, game-theoretic balance is formalised by the minimax value of the initial state $s_0$:
$$V(s_0) \in \{-1, 0, +1\}$$
Computing $V(s_0)$ requires solving the entire game tree, which is intractable for games of this complexity.

Therefore, this project formalises and studies **Operational (Empirical) Balance Under Bounded Rationality**:
> A rule configuration $r$ is operationally balanced if the army advantage between Chess and Xiangqi is statistically indistinguishable from zero across a diverse family of agents with varying heuristic structures and search horizons, and does not exhibit systematic drift toward one army as search budget increases.

---

## 2. Mathematical Definition: The Balance Curve

Instead of reporting a single, scalar "balance score" under a single arbitrary agent matchup, every rule configuration $r$ is characterised by a **Balance Curve** across a population of computational budgets and algorithms $\mathcal{K}$:
$$\mathcal{K} = \{\text{Random}, \text{Greedy}, \text{AB}_{d=1}, \text{AB}_{d=2}, \text{AB}_{d=4}, \text{NN-Policy}, \text{MCTS}_{16}, \text{MCTS}_{64}, \text{MCTS}_{256}\}$$

For each agent $k \in \mathcal{K}$, we measure the estimated army advantage $\hat{A}_k(r) \in [-1, +1]$:
$$\hat{A}_k(r) = P(\text{Chess wins} \mid k, r) - P(\text{Xiangqi wins} \mid k, r)$$
under paired side-switching matches with identical opening seeds.

### The Balance Objective Function
We evaluate a rule configuration by jointly optimising for low overall bias and low sensitivity to agent strength:
$$J(r) = |\hat{A}_{\text{pop}}(r)| + \lambda \operatorname{Var}_{k \in \mathcal{K}}\big(\hat{A}_k(r)\big)$$
1. **Bias Term ($|\hat{A}_{\text{pop}}(r)|$)**: The average army disparity across the entire population.
2. **Sensitivity Term ($\operatorname{Var}_k(\hat{A}_k(r))$)**: The variance across search depths. A game that appears 50/50 under shallow search but drifts to 80/20 under deep MCTS is fundamentally unbalanced; a truly robust rule exhibits low drift across all $k$.

---

## 3. Disentangling Confounders: The Population Bradley-Terry Model

Observing that a neural model $M$ scores 70% playing Chess against opponent $O$ and 60% playing Xiangqi against $O$ does **not** indicate a 10% rule imbalance. The observed score conflates four distinct effects:
$$\text{Observed Score} = \text{Army Advantage} + \text{Agent Competence} + \text{Opponent Competence} + \text{Agent} \times \text{Army Interaction}$$

To rigorously isolate the true army advantage $\beta_r$, paired games across an agent population $\mathcal{A} = \{a_1, a_2, \dots, a_m\}$ are fitted to a generalized Bradley-Terry logistic regression:
$$\operatorname{logit} P(a_i \text{ beats } a_j \text{ in game } g) = (s_i - s_j) + \beta_r \cdot \mathbb{I}(a_i \text{ plays Chess}) + \gamma_{\text{opening}}$$
- $s_i, s_j$: Intrinsic agent skill parameters.
- $\beta_r$: The true marginal army advantage of Chess over Xiangqi under rule $r$, strictly de-confounded from individual agent competence.
- $\gamma_{\text{opening}}$: Nuisance opening variance absorbed by paired opening seeds.

---

## 4. Decomposing Observed Weakness

When an army underperforms in empirical benchmarks, the failure must be dissected into three orthogonal hypotheses:
$$\text{Observed Weakness} = \text{Rule Weakness} + \text{Search Weakness} + \text{Representation Weakness}$$

1. **Rule Weakness**: Intrinsic structural/material asymmetry of the game rules. Isolated by symmetrical self-play ($a_i$ vs $a_i$) with army swapping.
2. **Search Weakness**: Horizon and tactical blind-spots (e.g. Xiangqi mating nets requiring deeper tactical calculations than direct Chess queen attacks). Isolated by comparing depth-1 vs depth-4 AlphaBeta and 16-sim vs 256-sim MCTS.
3. **Representation Weakness**: Inability of the neural network architecture (e.g., standard CNN receptive fields) to represent non-local patterns such as cannon screens or horse leg-blocking. Isolated by probing linear value probes $V(s) \approx \mathbf{w}^\top \mathbf{c}(s)$.

---

## 5. Termination Distribution Standard

Every benchmark report must record the complete categorical distribution of game terminations:
$$\mathcal{T} = [P(\text{Checkmate}), P(\text{Stalemate-Loss}), P(\text{Threefold-Draw}), P(\text{400-Ply-Draw}), P(\text{Perpetual-Check-Loss})]$$

- **400-Ply Safety vs Mechanic**: If $P(\text{400-Ply-Draw}) < 1\%$, the ply cutoff acts as a benign safety net. If $P(\text{400-Ply-Draw}) > 5\%$, the truncation has mutated into an active game mechanic that agents exploit to force draws, signaling endgame conversion failure.
- **Stalemate Symmetry**: Tracking $P(\text{Stalemate-Loss})$ reveals whether the palace constraint causes Xiangqi generals to suffer asymmetric stalemate penalties.

---

## 6. Multi-Fidelity Screening Architecture

To maximize compute efficiency on available allocations, candidate rules are screened across four multi-fidelity tiers:

```
[4096 Combinations]  ─── Tier 0: Static Structural Metrics (Zero Compute)
                              │  Filtered (~500 Candidates)
                              ▼
[~500 Candidates]   ─── Tier 1: Cheap Agent Tournament with SPRT Early-Stopping
                              │  Filtered (~30 Candidates)
                              ▼
[~30 Candidates]    ─── Tier 2: Active Surrogate Modeling (Bayesian Optimization)
                              │  Filtered (Top 2-3 Candidates)
                              ▼
[Top 2-3 Candidates] ─── Tier 3: Deep AlphaZero Self-Play & Multi-Seed H100 Validation
```

### Tier 0: Zero-Cost Structural Metrics
Static graph and heuristic analysis of the starting position:
- Initial legal move count per army.
- Branching factor proxy across plies 1–10.
- Royal escape freedom: safe squares accessible within 2 plies.
- Horse leg-blocking frequency: geometric ratio of blocked vs unobstructed moves.
- Effective active material ratio (excluding restricted backline palace pieces).

### Tier 1: Cheap Agent Tournament & SPRT
- Rapid paired matches with Random, Greedy, shallow AlphaBeta, and fast policy rollouts.
- Sequential Probability Ratio Test (SPRT): rules with $|A(r)| > 0.30$ are rejected after 8–16 games, conserving >90% of compute.

### Tier 2: Active Surrogate Learning
- A surrogate model $f(r) \rightarrow (\hat{A}(r), \sigma(r))$ guides rule selection using an upper confidence / balance acquisition function:
  $$\alpha(r) = -|\hat{A}(r)| + \lambda \sigma(r)$$
- Directly explores rule combinations with either high estimated balance or high epistemic uncertainty.

### Tier 3: Targeted Deep RL Validation
- Full self-play reinforcement learning and deep MCTS evaluation reserved exclusively for the Pareto-optimal frontier.

---

## 7. The Multi-Objective Pareto Frontier

The goal of the laboratory is not to enforce an arbitrary single "patched" rule, but to discover and present the **Pareto Optimal Frontier**:
$$\min_r \Big( |\hat{A}(r)|, \;\; \operatorname{Dist}(r, r_{\text{native}}), \;\; 1 - P(\text{Decisive}) \Big)$$

We maintain four canonical archetypes along this frontier:
1. **Original Hybrid**: Historical unmodified rules. Serves as the scientific control for measuring asymmetry.
2. **Minimal-Intervention Balance**: Minimal edit distance from native rules (e.g. 8 Chess pawns + Chess king palace) that achieves empirical equilibrium while preserving piece identities.
3. **Competitive Balance**: Optimized for deep agent tournament play with maximum draw-resilience and tactical richness.
4. **Experimental / Free Variants**: Exploring radical piece substitutions (e.g. Xiangqi Queen, extra cannons).
