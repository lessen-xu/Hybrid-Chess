# Hybrid Chess — Canonical Game Specification (V2)

> **Protocol lock**: No rule changes permitted while this version is active.
> Any modification requires a new version tag and re-validation of all results.

## Game Identity

| Property | Value |
|---|---|
| Name | Hybrid Chess |
| Type | Two-player, deterministic, perfect-information, zero-sum, **asymmetric** |
| Board | 9 columns × 10 rows (Xiangqi dimensions) |
| Sides | **Chess** (bottom, y=0–1) vs **Xiangqi** (top, y=6–9) |
| Engine | C++ (`hybrid_cpp_engine.pyd`) via pybind11 + Python fallback |

## Piece Set

### Chess Side (y=0–1)
- **King**: all 8 directions, 1 step. No castling.
- **Queen**: orthogonal + diagonal slide. No limit.
- **Rook** (×2): orthogonal slide.
- **Bishop** (×2): diagonal slide.
- **Knight** (×2): L-shape, **no** leg block.
- **Pawn** (×8+1): forward 1 (or 2 from y=1), diagonal capture, promotes at y=9 to Q/R/B/N. 9 pawns (extra at file 8).

### Xiangqi Side (y=6–9)
- **General**: orthogonal 1-step, palace only (3≤x≤5, 7≤y≤9).
- **Advisor** (×2): diagonal 1-step, palace only.
- **Elephant** (×2): diagonal 2-step, eye block, cannot cross river (y<5).
- **Horse** (×2): L-shape, **with** leg block.
- **Chariot** (×2): orthogonal slide (= Rook).
- **Cannon** (×2): orthogonal slide for non-capture; capture requires jumping over exactly 1 screen piece.
- **Soldier** (×5): forward only before river; forward + sideways after crossing y≤4.
- **Flying General**: if General and King share a column with no pieces between, General can capture King directly.

## Termination Rules

| Condition | Result | Implementation |
|---|---|---|
| Royal captured | Win for capturing side | Inline `has_royal()` check in negamax |
| Checkmate (no legal moves + in check) | **Loss** for checkmated side | `terminal_info()` in rules.cpp/rules.py |
| **Stalemate** (no legal moves + not in check) | **Loss for stalemated side** | Xiangqi convention. Same code path as checkmate. |
| Threefold repetition | Draw | Board hash + side-to-move, counter ≥ 3 |
| Max plies reached (400) | Draw | Ply counter ≥ `MAX_PLIES` |

## Evaluation Function (V2)

Base: material sum + mobility (0.05×) + check bonus (0.3).

Endgame heuristics (activated when `abs(material_diff) > 3.0`):
- Material amplification: 3.0× multiplier
- Chebyshev distance: all friendly pieces → enemy king (weight 0.15)
- Own-king proximity: winning king approaches enemy king (weight 0.5)
- Mobility squeeze: per-move opponent restriction bonus (0.3)
- Anti-stalemate penalty: -8.0 if opponent has ≤2 moves and not in check
- Check bonus: 5.0 when winning

Checkmate score in search: `1e6 - ply` (prefer shorter mates).

## Determinism Contract

- Same `seed` + same `agent_specs` + same `ablation` → identical game outcome
- Seeds are derived: `base_seed + pair_index * 10000 + game_index + side_offset`
- C++ and Python engines produce identical legal moves (verified by differential fuzz)

## Ablation Variants

| Flag | Effect |
|---|---|
| `ABLATION_NO_QUEEN` | Remove Chess Queen from initial board |
| `ABLATION_EXTRA_CANNON` | Add 3rd Cannon for Xiangqi at (4,7) |
| `ABLATION_NO_QUEEN_PROMOTION` | Pawn cannot promote to Queen |
| `ABLATION_REMOVE_EXTRA_PAWN` | Remove Chess 9th-file Pawn |

## Version History

| Version | Date | Changes |
|---|---|---|
| V1 | pre-2026-03-07 | Stalemate = draw, flat eval, symmetric averaging |
| **V2** | 2026-03-07 | Stalemate = loss, V2 endgame heuristics, role-separated EGTA |
