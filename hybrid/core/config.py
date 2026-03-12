# -*- coding: utf-8 -*-
"""Centralized rule switches and ablation flags for experiments."""

# Board dimensions (9x10 is assumed throughout the codebase)
BOARD_W = 9
BOARD_H = 10

# ----- Rule switches -----

# 1) Extra Chess pawn on the i-file (9th column)
CHESS_EXTRA_PAWN_ON_I_FILE = True

# 2) Flying-general capture: General can capture King on an open file
ENABLE_FLYING_GENERAL_CAPTURE = True

# 3) Threefold repetition draw (simplified; no perpetual-check/chase rules)
ENABLE_THREEFOLD_REPETITION_DRAW = True

# 4) Maximum plies before forced draw (400 half-moves ≈ 200 full moves)
MAX_PLIES = 400

# ----- Ablation switches -----

# 5) Remove Queen from Chess side
ABLATION_NO_QUEEN = False

# 6) Restrict pawn promotion: no Queen promotion (only R/B/N)
ABLATION_NO_QUEEN_PROMOTION = False

# 7) Extra Cannon for Xiangqi side at (4, 7)
ABLATION_EXTRA_CANNON = False

# 8) Remove Chess extra pawn on i-file (overrides CHESS_EXTRA_PAWN_ON_I_FILE)
ABLATION_REMOVE_EXTRA_PAWN = False

# 9) Remove one Chess Bishop (left-side, (2,0))
ABLATION_CHESS_NO_BISHOP = False

# 10) Xiangqi gets a 6th Soldier at center column (4, 5)
ABLATION_XIANGQI_EXTRA_SOLDIER = False

# 11) Remove Chess second Rook (right-side Rook at (7,0))
ABLATION_CHESS_ONE_ROOK = False

# 12) Disable Flying-General capture (weakens Xiangqi General's threat)
ABLATION_NO_FLYING_GENERAL = False
