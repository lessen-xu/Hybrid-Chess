# Legacy Results — INVALIDATED

These results were produced under the **pre-V2 rule set** (stalemate = draw) and using **symmetric side-swap averaging**.

## Reasons for Invalidation

1. **Terminal semantics changed**: Stalemate was changed from draw to loss for the stalemated side (Xiangqi convention) in commit `f2d2640`. This is a game-definition-level change that invalidates all prior payoff matrices.

2. **Symmetric averaging retired**: Prior matrices averaged `A@Chess vs B@Xiangqi` and `B@Chess vs A@Xiangqi` into a single scalar. The game is fundamentally asymmetric (Chess side has stronger mobility), so primary analysis now uses role-separated matrices.

## Contents

| File | Original Purpose |
|---|---|
| `egta_pilot.md` | 720-game pilot tournament (N=20) |
| `egta_n100.md` | N=100 dual-universe tournament (3,600 games each) |
| `v3_heatmap.png` | V3 (no_queen) payoff heatmap |
| `v4_heatmap.png` | V4 (extra_cannon) payoff heatmap |

These may be cited as appendix/sensitivity analysis but **must not** appear as primary evidence.
