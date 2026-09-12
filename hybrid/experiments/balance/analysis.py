"""Statistical analysis and causal effect estimation for rule variants.

Decomposes tournament results into:
1. Ordinary Least Squares (OLS) marginal effects:
   Score(Chess) = beta_0 + sum(beta_i * Factor_i) + sum(beta_ij * Factor_i * Factor_j)
2. Bradley-Terry decomposition of agent skill vs army bias:
   logit(P(A beats B)) = (s_A - s_B) + beta_army
3. Balance sensitivity and Pareto-optimal ranking.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
import math
import numpy as np

from .screening import TournamentResult


@dataclass
class FactorEffect:
    """Estimated marginal effect of a single rule factor."""
    factor_name: str
    beta: float                     # Marginal effect on Chess Score (in [-1.0, 1.0])
    percentage_points: float        # beta * 100
    t_stat: Optional[float] = None
    p_value: Optional[float] = None


@dataclass
class CausalAnalysisReport:
    """Full statistical report from factorial screening results."""
    num_variants_analyzed: int
    baseline_intercept: float       # Expected Chess score when all factors are at baseline (0)
    main_effects: List[FactorEffect]
    interaction_effects: List[FactorEffect]
    top_balanced_variants: List[Dict[str, Any]]
    ranking_by_impact: List[str]    # Factor names sorted by magnitude of effect

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_variants_analyzed": self.num_variants_analyzed,
            "baseline_intercept": round(self.baseline_intercept, 4),
            "main_effects": [asdict(f) for f in self.main_effects],
            "interaction_effects": [asdict(f) for f in self.interaction_effects],
            "top_balanced_variants": self.top_balanced_variants,
            "ranking_by_impact": self.ranking_by_impact,
        }

    def summary_table(self) -> str:
        """Format as readable markdown table."""
        lines = [
            "# Rule Balance Causal Attribution Report",
            "",
            f"**Variants Analyzed**: {self.num_variants_analyzed}",
            f"**Baseline Intercept (Standard Chess Score)**: {self.baseline_intercept * 100:.1f}%",
            "",
            "## Marginal Main Effects (OLS)",
            "| Rule Factor | Delta (% Chess Score) | Direction | Impact Description |",
            "|---|---|---|---|",
        ]
        for eff in self.main_effects:
            direction = "Favor Xiangqi" if eff.beta < 0 else "Favor Chess"
            lines.append(
                f"| `{eff.factor_name}` | {eff.percentage_points:+.2f}% | {direction} | "
                f"{'Reduces Chess advantage' if eff.beta < 0 else 'Increases Chess advantage'} |"
            )

        if self.interaction_effects:
            lines.extend([
                "",
                "## Key Two-Factor Interactions",
                "| Interaction Pair | Delta (%) |",
                "|---|---|",
            ])
            for eff in self.interaction_effects:
                lines.append(f"| `{eff.factor_name}` | {eff.percentage_points:+.2f}% |")

        lines.extend([
            "",
            "## Top 5 Balanced Candidates",
            "| Variant | Chess Score | XQ Score | Disparity | Mean Plies |",
            "|---|---|---|---|---|",
        ])
        for v in self.top_balanced_variants[:5]:
            lines.append(
                f"| `{v['name']}` | {v['chess_score_rate']*100:.1f}% | {v['xiangqi_score_rate']*100:.1f}% | "
                f"{v['balance_disparity']*100:.1f}% | {v['mean_plies']:.1f} |"
            )

        return "\n".join(lines)


def estimate_causal_effects(
    results: List[TournamentResult],
    interaction_pairs: Optional[List[Tuple[str, str]]] = None,
) -> CausalAnalysisReport:
    """Fit OLS regression on factorial screening tournament results."""
    if not results:
        raise ValueError("Cannot estimate effects from empty results list")

    # Extract factor names from the first result
    factor_names = sorted(list(results[0].factors.keys()))
    n_samples = len(results)

    # Design matrix X: [1, x_1, x_2, ..., x_p, interactions...]
    X_rows = []
    y_vals = []

    pairs = interaction_pairs or [
        ("chess_palace", "knight_block"),
        ("chess_palace", "stalemate_rule"),
        ("xq_queen", "chess_palace"),
        ("first_side", "chess_palace"),
    ]

    for r in results:
        y_vals.append(r.chess_score_rate)
        row = [1.0]  # Intercept
        for fn in factor_names:
            row.append(float(r.factors.get(fn, 0)))
        # Interactions
        for f1, f2 in pairs:
            v1 = float(r.factors.get(f1, 0))
            v2 = float(r.factors.get(f2, 0))
            row.append(v1 * v2)
        X_rows.append(row)

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_vals, dtype=np.float64)

    # Solve least squares: beta = (X^T X)^-1 X^T y
    try:
        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        # Fallback to pseudo-inverse
        beta = np.linalg.pinv(X) @ y

    intercept = float(beta[0])
    main_effects: List[FactorEffect] = []

    for idx, fn in enumerate(factor_names):
        b = float(beta[1 + idx])
        main_effects.append(
            FactorEffect(
                factor_name=fn,
                beta=round(b, 4),
                percentage_points=round(b * 100.0, 2),
            )
        )

    interaction_effects: List[FactorEffect] = []
    offset = 1 + len(factor_names)
    for idx, (f1, f2) in enumerate(pairs):
        b = float(beta[offset + idx])
        interaction_effects.append(
            FactorEffect(
                factor_name=f"{f1} * {f2}",
                beta=round(b, 4),
                percentage_points=round(b * 100.0, 2),
            )
        )

    # Sort variants by balance disparity ascending
    sorted_variants = sorted(
        [
            {
                "name": r.variant_name,
                "factors": r.factors,
                "chess_score_rate": r.chess_score_rate,
                "xiangqi_score_rate": r.xiangqi_score_rate,
                "balance_disparity": r.balance_disparity,
                "mean_plies": r.mean_plies,
                "termination_distribution": r.termination_distribution,
            }
            for r in results
        ],
        key=lambda x: x["balance_disparity"],
    )

    # Ranking of main effects by absolute magnitude
    ranking = sorted(main_effects, key=lambda e: abs(e.beta), reverse=True)
    ranking_names = [e.factor_name for e in ranking]

    return CausalAnalysisReport(
        num_variants_analyzed=n_samples,
        baseline_intercept=round(intercept, 4),
        main_effects=main_effects,
        interaction_effects=interaction_effects,
        top_balanced_variants=sorted_variants,
        ranking_by_impact=ranking_names,
    )


def fit_bradley_terry(games: List[Dict[str, Any]]) -> Dict[str, float]:
    """Simple Bradley-Terry / Logistic regression isolating army bias.

    Log-odds of Chess winning vs Xiangqi winning across identical agents:
    logit(P(Chess wins)) = beta_army + (skill_Chess - skill_XQ)
    """
    if not games:
        return {"army_bias": 0.0, "p_chess": 0.5}

    chess_wins = 0
    xq_wins = 0
    draws = 0

    for g in games:
        w = g.get("winner")
        if w == "chess":
            chess_wins += 1
        elif w == "xiangqi":
            xq_wins += 1
        else:
            draws += 1

    total = len(games)
    score_chess = (chess_wins + 0.5 * draws) / max(1, total)

    # Clip to avoid infinite logit
    p = max(0.01, min(0.99, score_chess))
    army_bias = math.log(p / (1.0 - p))

    return {
        "army_bias_logit": round(army_bias, 4),
        "empirical_chess_score": round(score_chess, 4),
        "chess_wins": chess_wins,
        "xiangqi_wins": xq_wins,
        "draws": draws,
    }
