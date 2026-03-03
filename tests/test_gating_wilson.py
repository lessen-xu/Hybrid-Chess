# -*- coding: utf-8 -*-
"""Unit tests for Wilson CI adaptive gating."""

import pytest
from hybrid.rl.az_eval import wilson_ci


class TestWilsonCI:
    """Wilson confidence interval test cases."""

    def test_zero_games(self):
        """wins + losses == 0 -> returns (0.5, 0.0, 1.0)."""
        p_hat, ci_low, ci_high = wilson_ci(0, 0)
        assert p_hat == 0.5
        assert ci_low == 0.0
        assert ci_high == 1.0

    def test_strong_win_signal(self):
        """(9, 1) -> p_hat=0.9, ci_low should > 0.55 (ACCEPT)."""
        p_hat, ci_low, ci_high = wilson_ci(9, 1)
        assert abs(p_hat - 0.9) < 1e-6
        assert ci_low > 0.55, f"ci_low={ci_low} should be > 0.55 for (9,1)"
        assert ci_high <= 1.0

    def test_weak_win_signal(self):
        """(6, 4) -> p_hat=0.6, CI should span 0.55 (inconclusive)."""
        p_hat, ci_low, ci_high = wilson_ci(6, 4)
        assert abs(p_hat - 0.6) < 1e-6
        assert ci_low < 0.55, f"ci_low={ci_low} should be < 0.55 for (6,4)"
        assert ci_high > 0.55, f"ci_high={ci_high} should be > 0.55 for (6,4)"

    def test_strong_loss_signal(self):
        """(1, 9) -> p_hat=0.1, ci_high should < 0.55 (REJECT)."""
        p_hat, ci_low, ci_high = wilson_ci(1, 9)
        assert abs(p_hat - 0.1) < 1e-6
        assert ci_high < 0.55, f"ci_high={ci_high} should be < 0.55 for (1,9)"
        assert ci_low >= 0.0

    def test_monotonicity_ci_low(self):
        """Fixed losses=2, increasing wins -> ci_low should be monotonically increasing."""
        prev_ci_low = -1.0
        for wins in [3, 5, 8, 15]:
            _, ci_low, _ = wilson_ci(wins, 2)
            assert ci_low > prev_ci_low, \
                f"ci_low should increase: wins={wins}, ci_low={ci_low}, prev={prev_ci_low}"
            prev_ci_low = ci_low

    def test_ci_bounds_valid(self):
        """CI bounds: 0 <= ci_low <= p_hat <= ci_high <= 1."""
        test_cases = [(1, 0), (0, 1), (5, 5), (10, 0), (0, 10), (50, 30)]
        for w, l in test_cases:
            p_hat, ci_low, ci_high = wilson_ci(w, l)
            assert 0.0 <= ci_low <= p_hat <= ci_high <= 1.0, \
                f"Invalid bounds for ({w},{l}): ci_low={ci_low}, p_hat={p_hat}, ci_high={ci_high}"

    def test_all_wins(self):
        """All wins (10, 0) -> p_hat=1.0, ci_low should be high but < 1.0."""
        p_hat, ci_low, ci_high = wilson_ci(10, 0)
        assert p_hat == 1.0
        assert ci_low > 0.55
        assert ci_high == 1.0

    def test_all_losses(self):
        """All losses (0, 10) -> p_hat=0.0, ci_high should be low but > 0.0."""
        p_hat, ci_low, ci_high = wilson_ci(0, 10)
        assert p_hat == 0.0
        assert ci_low == 0.0
        assert ci_high < 0.55

    def test_large_sample_narrows_ci(self):
        """Larger sample size should produce narrower CI."""
        _, ci_low_small, ci_high_small = wilson_ci(6, 4)        # n=10
        _, ci_low_large, ci_high_large = wilson_ci(60, 40)      # n=100
        width_small = ci_high_small - ci_low_small
        width_large = ci_high_large - ci_low_large
        assert width_large < width_small, \
            f"Large sample CI should be narrower: {width_large} vs {width_small}"

    def test_confidence_99_wider_than_95(self):
        """99% CI should be wider than 95% CI."""
        _, low95, high95 = wilson_ci(7, 3, confidence=0.95)
        _, low99, high99 = wilson_ci(7, 3, confidence=0.99)
        assert (high99 - low99) > (high95 - low95), \
            "99% CI should be wider than 95% CI"
