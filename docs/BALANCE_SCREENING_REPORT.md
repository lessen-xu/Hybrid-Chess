# Rule Balance Causal Attribution Report

**Variants Analyzed**: 7
**Baseline Intercept (Standard Chess Score)**: 25.0%

## Marginal Main Effects (OLS)
| Rule Factor | Delta (% Chess Score) | Direction | Impact Description |
|---|---|---|---|
| `chess_mirror` | +0.00% | Favor Chess | Increases Chess advantage |
| `chess_palace` | +0.00% | Favor Chess | Increases Chess advantage |
| `extra_cannon` | +0.00% | Favor Chess | Increases Chess advantage |
| `extra_pawn_i_file` | +25.00% | Favor Chess | Increases Chess advantage |
| `first_side` | +0.00% | Favor Chess | Increases Chess advantage |
| `knight_block` | +0.00% | Favor Chess | Increases Chess advantage |
| `no_queen` | -12.50% | Favor Xiangqi | Reduces Chess advantage |
| `no_queen_promotion` | +0.00% | Favor Chess | Increases Chess advantage |
| `repetition_rule` | +0.00% | Favor Chess | Increases Chess advantage |
| `stalemate_rule` | +0.00% | Favor Chess | Increases Chess advantage |
| `xq_queen` | +18.75% | Favor Chess | Increases Chess advantage |

## Key Two-Factor Interactions
| Interaction Pair | Delta (%) |
|---|---|
| `chess_palace * knight_block` | +0.00% |
| `chess_palace * stalemate_rule` | +0.00% |
| `xq_queen * chess_palace` | +18.75% |
| `first_side * chess_palace` | +0.00% |

## Top 5 Balanced Candidates
| Variant | Chess Score | XQ Score | Disparity | Mean Plies |
|---|---|---|---|---|
| `standard_original` | 50.0% | 50.0% | 0.0% | 60.8 |
| `pk_palace_knight` | 50.0% | 50.0% | 0.0% | 80.0 |
| `extra_cannon` | 50.0% | 50.0% | 0.0% | 80.0 |
| `fide_draw_stalemate` | 50.0% | 50.0% | 0.0% | 60.8 |
| `tempo_swapped_xq_first` | 50.0% | 50.0% | 0.0% | 80.0 |