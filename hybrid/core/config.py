"""Centralized rule switches, ablation flags, and VariantConfig for experiments.

VariantConfig is the recommended way to configure game variants.
The legacy global flags are kept for backwards compatibility and are
used as defaults when no VariantConfig is explicitly passed.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict

# Board dimensions (true constants, never change)
BOARD_W = 9
BOARD_H = 10

# Maximum plies (half-moves) before forced draw
MAX_PLIES = 400

# Threefold repetition draw (simplified; no perpetual-check/chase rules)
ENABLE_THREEFOLD_REPETITION_DRAW = True
# VariantConfig — the clean, composable way to configure game variants

@dataclass(frozen=True)
class VariantConfig:
    """Game variant configuration.

    Pass this to ``HybridChessEnv`` to create custom rule variants
    without modifying source code.

    Example::

        from hybrid.core.config import VariantConfig
        from hybrid.core.env import HybridChessEnv

        variant = VariantConfig(no_queen=True, extra_cannon=True)
        env = HybridChessEnv(variant=variant)
        state = env.reset()
    """
    # --- Board setup flags ---
    extra_pawn_i_file: bool = True     # Chess 9th pawn
    no_queen: bool = False             # Remove Chess Queen
    no_bishop: bool = False            # Remove Chess left Bishop
    one_rook: bool = False             # Remove Chess right Rook
    remove_extra_pawn: bool = False    # Remove Chess 9th pawn (overrides extra_pawn_i_file)
    extra_cannon: bool = False         # Extra Cannon for Xiangqi at (4,7)
    extra_soldier: bool = False        # Extra Soldier for Xiangqi at (4,5)

    # --- Rule flags ---
    flying_general: bool = True        # Enable flying-general capture
    no_queen_promotion: bool = False   # Pawn can only promote to R/B/N

    def to_dict(self) -> dict:
        """Serialize for checkpoints / JSON configs."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VariantConfig":
        """Reconstruct from a dict (e.g. loaded from checkpoint)."""
        # Only pass keys that are valid fields
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})


# Default variant — matches standard Hybrid Chess rules
DEFAULT_VARIANT = VariantConfig()
# Legacy global flags — kept for backwards compatibility

# Rule switches
CHESS_EXTRA_PAWN_ON_I_FILE = True
ENABLE_FLYING_GENERAL_CAPTURE = True

# Ablation switches (all default False)
ABLATION_NO_QUEEN = False
ABLATION_NO_QUEEN_PROMOTION = False
ABLATION_EXTRA_CANNON = False
ABLATION_REMOVE_EXTRA_PAWN = False
ABLATION_CHESS_NO_BISHOP = False
ABLATION_XIANGQI_EXTRA_SOLDIER = False
ABLATION_CHESS_ONE_ROOK = False
ABLATION_NO_FLYING_GENERAL = False
