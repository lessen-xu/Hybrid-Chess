# AB Rule Balance Tournament

AB vs AB at two search depths × three rule variants. 4 random opening plies to break determinism. 100 games per cell.

## 2×3 Balance Matrix

| Rule variant | AB-d1 (Chess/XQ/Draw) | AB-d2 (Chess/XQ/Draw) |
|---|---|---|
| **Vanilla** | **33**/5/62 | 0/10/90 |
| **Extra Cannon** | **28**/6/66 | 0/10/90 |
| **No Queen** | **24**/7/69 | 0/5/95 |

### Conclusions

1. d1→d2 eliminates ALL Chess wins (33%→0%). Search depth >> rule variant.
2. Queen amplifies weak defense — at d2 it's perfectly neutralized.
3. Chess has a natural edge even without Queen (24% at d1 from Bishop diagonals + Pawn promotion).

---

## AB-d2 Termination Analysis (no_queen, 100 games)

| Termination Reason | Count | Avg Ply |
|---|---|---|
| **Threefold repetition** | 95 (95%) | 36.3 |
| Checkmate (Xiangqi wins) | 5 (5%) | 11.2 |
| Move limit (400 ply) | **0 (0%)** | — |

**100% of AB-d2 draws are true deadlocks (threefold repetition), 0% time pressure.** AB-d2 enters deterministic cowardice loops ~36 plies in. The 400-ply limit is never approached (max observed: 98 ply).
