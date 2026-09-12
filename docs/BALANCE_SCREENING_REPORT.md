# Rule Balance Causal Attribution Report

**Variants Analyzed**: 32
**Baseline Intercept (Standard Chess Score)**: 63.5%

## Marginal Main Effects (OLS)
| Rule Factor | Delta (% Chess Score) | Direction | Impact Description |
|---|---|---|---|
| `chess_mirror` | +4.69% | Favor Chess | Increases Chess advantage |
| `chess_palace` | -8.85% | Favor Xiangqi | Reduces Chess advantage |
| `extra_cannon` | -2.61% | Favor Xiangqi | Reduces Chess advantage |
| `extra_pawn_i_file` | -0.52% | Favor Xiangqi | Reduces Chess advantage |
| `first_side` | +1.04% | Favor Chess | Increases Chess advantage |
| `knight_block` | -3.13% | Favor Xiangqi | Reduces Chess advantage |
| `no_queen` | -14.06% | Favor Xiangqi | Reduces Chess advantage |
| `no_queen_promotion` | +0.52% | Favor Chess | Increases Chess advantage |
| `repetition_rule` | -1.56% | Favor Xiangqi | Reduces Chess advantage |
| `stalemate_rule` | -7.29% | Favor Xiangqi | Reduces Chess advantage |
| `xq_queen` | +5.21% | Favor Chess | Increases Chess advantage |

## Key Two-Factor Interactions
| Interaction Pair | Delta (%) |
|---|---|
| `chess_palace * knight_block` | +3.13% |
| `chess_palace * stalemate_rule` | +5.21% |
| `xq_queen * chess_palace` | -5.21% |
| `first_side * chess_palace` | +1.04% |

## Top 5 Balanced Candidates
| Variant | Chess Score | XQ Score | Disparity | Mean Plies |
|---|---|---|---|---|
| `run_03` | 50.0% | 50.0% | 0.0% | 93.7 |
| `run_08` | 50.0% | 50.0% | 0.0% | 100.0 |
| `run_09` | 50.0% | 50.0% | 0.0% | 100.0 |
| `run_11` | 50.0% | 50.0% | 0.0% | 100.0 |
| `run_12` | 50.0% | 50.0% | 0.0% | 100.0 |