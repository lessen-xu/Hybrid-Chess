# Game Rules

**Board:** 9 columns × 10 rows (Xiangqi dimensions). Chess starts at the bottom (y=0–1), Xiangqi at the top (y=6–9).

## Pieces & Rules

- **Chess side:** King (all 8 dirs), Queen (orth+diag slide), Rook (orth slide), Bishop (diag slide), Knight (L-shape, NO leg block), Pawn (forward, double-step from y=1, diagonal capture, promotes at y=9 to Q/R/B/N).
- **Xiangqi side:** General (orth 1-step, palace only 3≤x≤5, 7≤y≤9), Advisor (diag 1-step, palace only), Elephant (diag 2-step, eye block, cannot cross river y<5), Horse (L-shape, WITH leg block), Chariot (orth slide = Rook), Cannon (orth slide non-capture; capture requires jumping over exactly 1 screen piece), Soldier (forward only before river; forward+sideways after crossing y≤4).
- **Flying General:** If General and King are on the same column with no pieces between, General can capture King directly.

## Termination

- King/General captured
- Checkmate
- Stalemate (loss for stalemated side — Xiangqi convention)
- Threefold repetition (draw)
- Move limit: 400 ply (draw)

## Rule Variants (Ablation Flags)

Controlled via `hybrid/core/config.py`:

| Flag | Effect |
|---|---|
| `ABLATION_NO_QUEEN` | Remove Chess Queen from initial board |
| `ABLATION_NO_QUEEN_PROMOTION` | Pawn cannot promote to Queen (only R/B/N) |
| `ABLATION_EXTRA_CANNON` | Add 3rd Cannon for Xiangqi at (4,7) |
| `ABLATION_REMOVE_EXTRA_PAWN` | Remove Chess 9th-file Pawn |
