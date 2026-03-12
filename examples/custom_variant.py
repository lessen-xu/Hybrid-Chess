#!/usr/bin/env python3
"""Example: Creating and playing a custom variant.

This script shows how to:
1. Define a custom rule variant using VariantConfig
2. Initialize the environment with that variant
3. Generate / parse FEN positions
4. Run a quick random self-play game

Usage:
    python examples/custom_variant.py
"""

from hybrid.core.config import VariantConfig
from hybrid.core.env import HybridChessEnv
from hybrid.core.fen import board_to_fen
import random

# ──────────────────────────────────────────────
# 1. Define a custom variant
# ──────────────────────────────────────────────
# Remove the Chess Queen, add an extra Cannon for Xiangqi,
# and disable the flying-general capture rule.
my_variant = VariantConfig(
    no_queen=True,
    extra_cannon=True,
    flying_general=False,
)
print(f"Variant: {my_variant}")
print(f"Serialized: {my_variant.to_dict()}")

# ──────────────────────────────────────────────
# 2. Create environment with variant
# ──────────────────────────────────────────────
env = HybridChessEnv(variant=my_variant)
state = env.reset()

# Show FEN of starting position
fen = board_to_fen(state.board, state.side_to_move)
print(f"\nStarting FEN: {fen}")
print(f"Legal moves: {len(env.legal_moves())}")

# ──────────────────────────────────────────────
# 3. Quick random self-play game
# ──────────────────────────────────────────────
print("\n--- Random self-play ---")
random.seed(42)
state = env.reset()

for ply in range(200):
    legal = env.legal_moves()
    if not legal:
        break
    move = random.choice(legal)
    state, reward, done, info = env.step(move)
    if done:
        print(f"  Game over at ply {state.ply}: {info}")
        break

if not done:
    print(f"  Game still running after 200 plies (ply={state.ply})")

final_fen = board_to_fen(state.board, state.side_to_move)
print(f"  Final FEN: {final_fen}")

# ──────────────────────────────────────────────
# 4. Load from FEN
# ──────────────────────────────────────────────
print("\n--- Loading from FEN ---")
env2 = HybridChessEnv()
state2 = env2.reset_from_fen(fen)
print(f"  Loaded FEN: {fen}")
print(f"  Legal moves: {len(env2.legal_moves())}")
print(f"  Side to move: {state2.side_to_move.name}")

print("\nDone!")
