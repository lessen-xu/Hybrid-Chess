# -*- coding: utf-8 -*-
"""Statistical utilities for paper tables: Wilson CI, bootstrap CI, side masking."""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, List


def wilson_ci(successes: int, trials: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials == 0:
        return (0.0, 0.0)
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p_hat = successes / trials
    denom = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_ci(scores: List[float], n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 42) -> Tuple[float, float]:
    """Bootstrap percentile interval for the mean of scores."""
    arr = np.array(scores)
    if len(arr) == 0:
        return (0.0, 0.0)
    rng = np.random.RandomState(seed)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)
