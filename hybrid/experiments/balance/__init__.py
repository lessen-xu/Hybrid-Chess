"""Balance Laboratory: Systematic rule design, multi-agent screening, and causal balance diagnosis."""

from .metrics import compute_tier0_metrics, Tier0Metrics
from .design import generate_screening_matrix, FACTORS, DesignPoint
from .screening import run_paired_tournament, TournamentConfig, TournamentResult
from .analysis import estimate_causal_effects, fit_bradley_terry

__all__ = [
    "compute_tier0_metrics",
    "Tier0Metrics",
    "generate_screening_matrix",
    "FACTORS",
    "DesignPoint",
    "run_paired_tournament",
    "TournamentConfig",
    "TournamentResult",
    "estimate_causal_effects",
    "fit_bradley_terry",
]
