# Hybrid Chess — Game Rules

**Board:** 9 columns × 10 rows (Xiangqi dimensions).
Chess starts at the bottom (y=0–1), Xiangqi at the top (y=6–9).

---

## Pieces

### Chess Side (bottom)

| Piece | Movement |
|-------|----------|
| **King** | All 8 directions, 1 step. No castling. |
| **Queen** | Orthogonal + diagonal slide, unlimited range. |
| **Rook** (×2) | Orthogonal slide. |
| **Bishop** (×2) | Diagonal slide. |
| **Knight** (×2) | L-shape, **no** leg block. |
| **Pawn** (×9) | Forward 1 (or 2 from y=1), diagonal capture. Promotes at y=9 to Q/R/B/N. |

### Xiangqi Side (top)

| Piece | Movement |
|-------|----------|
| **General** | Orthogonal 1-step, palace only (3≤x≤5, 7≤y≤9). |
| **Advisor** (×2) | Diagonal 1-step, palace only. |
| **Elephant** (×2) | Diagonal 2-step, eye block, cannot cross river (y<5). |
| **Horse** (×2) | L-shape, **with** leg block. |
| **Chariot** (×2) | Orthogonal slide (same as Rook). |
| **Cannon** (×2) | Orthogonal slide; capture requires jumping over exactly 1 piece. |
| **Soldier** (×5) | Forward only before river; forward + sideways after crossing y≤4. |

### Special: Flying General

If General and King share a column with no pieces between them, the General can capture the King directly.

---

## Termination

| Condition | Result |
|-----------|--------|
| Royal captured | Win for capturing side |
| Checkmate (no legal moves + in check) | Loss for checkmated side |
| Stalemate (no legal moves + not in check) | **Loss** for stalemated side (Xiangqi convention) |
| Threefold repetition | Draw |
| Max plies (400) | Draw |

---

## Rule Variants (Ablation Flags)

Controlled via `hybrid/core/config.py` or the `--ablation` CLI flag:

### Piece Modifications

| Flag | Effect |
|------|--------|
| `no_queen` | Remove Chess Queen |
| `no_bishop` | Remove Chess Bishops |
| `one_rook` | Remove one Chess Rook |
| `remove_pawn` | Remove Chess 9th-file Pawn |
| `extra_cannon` | Add 3rd Cannon for Xiangqi at (4,7) |
| `extra_soldier` | Add extra Soldier for Xiangqi |
| `xq_queen` | Give Xiangqi a Queen (replaces left Advisor) |

### Rule Reforms

| Flag | Effect |
|------|--------|
| `chess_palace` | Confine Chess King to 3×3 palace (x=3–5, y=0–2) |
| `knight_block` | Chess Knight uses Xiangqi Horse leg-blocking rules |
| `no_promotion` | Disable pawn promotion (pawn stays pawn at y=9) |
| `no_queen_promo` | Pawn cannot promote to Queen (R/B/N only) |
| `no_flying_general` | Disable Flying General rule |

Multiple variants can be combined: `--ablation chess_palace,knight_block,xq_queen`

---

## Evaluation Function

- **Material**: piece value sum
- **Mobility**: 0.05 × legal move count
- **Check bonus**: 0.3
- **Endgame** (when |material_diff| > 3.0): 3× material amplification, piece→enemy king distance, mobility squeeze, check bonus 5.0
- **Checkmate score**: 1e6 − ply (prefer shorter mates)
