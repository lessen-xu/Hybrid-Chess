"""Factorial experimental design for asymmetric rule balance search.

Generates orthogonal / fractional factorial design matrices (Resolution IV, 32 runs)
across 11 causal rule factors:
1. first_side: tempo de-confounding ("chess" vs "xiangqi")
2. chess_palace: King confined to 3x3 palace (False vs True)
3. knight_block: Knight leg blocked like Xiangqi horse (False vs True)
4. no_queen: Chess starts without Queen (False vs True)
5. stalemate_rule: Stalemate outcome ("loss" vs "draw")
6. no_queen_promotion: Pawn cannot promote to Queen (False vs True)
7. repetition_rule: Threefold repetition ("draw" vs "perpetual_check_loss")
8. xq_queen: Xiangqi starts with Queen (False vs True)
9. extra_cannon: Xiangqi starts with 3rd Cannon (False vs True)
10. chess_mirror: Chess pieces mirrored horizontally (False vs True)
11. extra_pawn_i_file: Chess has 9th pawn on i-file (False vs True)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import json

from hybrid.core.config import VariantConfig


FACTORS = [
    "first_side",
    "chess_palace",
    "knight_block",
    "no_queen",
    "stalemate_rule",
    "no_queen_promotion",
    "repetition_rule",
    "xq_queen",
    "extra_cannon",
    "chess_mirror",
    "extra_pawn_i_file",
]


@dataclass
class DesignPoint:
    """A single configuration point in the screening design."""
    id: int
    name: str
    factors: Dict[str, Any]
    variant: VariantConfig

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "factors": self.factors,
            "variant": self.variant.to_dict(),
        }


def point_to_variant(factors: Dict[str, Any]) -> VariantConfig:
    """Convert factor dictionary to VariantConfig."""
    first_side = "xiangqi" if factors.get("first_side", 0) == 1 else "chess"
    stalemate_rule = "draw" if factors.get("stalemate_rule", 0) == 1 else "loss"
    repetition_rule = "perpetual_check_loss" if factors.get("repetition_rule", 0) == 1 else "draw"

    return VariantConfig(
        chess_palace=bool(factors.get("chess_palace", 0)),
        knight_block=bool(factors.get("knight_block", 0)),
        no_queen=bool(factors.get("no_queen", 0)),
        no_queen_promotion=bool(factors.get("no_queen_promotion", 0)),
        extra_pawn_i_file=bool(factors.get("extra_pawn_i_file", 1)),
        remove_extra_pawn=not bool(factors.get("extra_pawn_i_file", 1)),
        xq_queen=bool(factors.get("xq_queen", 0)),
        extra_cannon=bool(factors.get("extra_cannon", 0)),
        first_side=first_side,
        stalemate_rule=stalemate_rule,
        repetition_rule=repetition_rule,
        chess_mirror=bool(factors.get("chess_mirror", 0)),
    )


def generate_screening_matrix(num_runs: int = 32) -> List[DesignPoint]:
    """Generate a balanced Resolution IV fractional factorial design matrix.

    Uses 5 independent base columns (A, B, C, D, E) producing 2^5 = 32 runs,
    with generators for the remaining 6 factors:
      F = A ^ B ^ C
      G = A ^ B ^ D
      H = A ^ C ^ E
      I = B ^ C ^ D
      J = B ^ D ^ E
      K = C ^ D ^ E
    """
    points: List[DesignPoint] = []

    for i in range(num_runs):
        # 5 base factors from binary representation of index i
        a = (i >> 0) & 1  # first_side
        b = (i >> 1) & 1  # chess_palace
        c = (i >> 2) & 1  # knight_block
        d = (i >> 3) & 1  # no_queen
        e = (i >> 4) & 1  # stalemate_rule

        # 6 generator factors
        f = a ^ b ^ c     # no_queen_promotion
        g = a ^ b ^ d     # repetition_rule
        h = a ^ c ^ e     # xq_queen
        j = b ^ c ^ d     # extra_cannon
        k = b ^ d ^ e     # chess_mirror
        m = c ^ d ^ e     # extra_pawn_i_file

        factors = {
            "first_side": a,
            "chess_palace": b,
            "knight_block": c,
            "no_queen": d,
            "stalemate_rule": e,
            "no_queen_promotion": f,
            "repetition_rule": g,
            "xq_queen": h,
            "extra_cannon": j,
            "chess_mirror": k,
            "extra_pawn_i_file": m,
        }

        var = point_to_variant(factors)
        name = f"run_{i:02d}"
        points.append(DesignPoint(id=i, name=name, factors=factors, variant=var))

    return points


def get_standard_reference_points() -> List[DesignPoint]:
    """Return key historical presets for benchmarking and sanity checks."""
    presets = [
        ("standard_original", {}),
        ("pk_palace_knight", {"chess_palace": 1, "knight_block": 1}),
        ("pk_xq_queen", {"chess_palace": 1, "knight_block": 1, "xq_queen": 1}),
        ("no_queen", {"no_queen": 1}),
        ("extra_cannon", {"extra_cannon": 1}),
        ("fide_draw_stalemate", {"stalemate_rule": 1}),
        ("tempo_swapped_xq_first", {"first_side": 1}),
    ]

    ref_points: List[DesignPoint] = []
    for idx, (name, factor_overrides) in enumerate(presets):
        base_factors = {f: 0 for f in FACTORS}
        base_factors["extra_pawn_i_file"] = 1  # default is 9 pawns
        base_factors.update(factor_overrides)
        var = point_to_variant(base_factors)
        ref_points.append(DesignPoint(id=1000 + idx, name=name, factors=base_factors, variant=var))

    return ref_points
