# -*- coding: utf-8 -*-
"""TD(0) self-play training with a linear value function V_theta(s) = theta · phi(s).

Update rule:  delta_t = target - V(s_t),  theta += alpha * delta_t * phi(s_t)
Note: in two-player zero-sum games, V(s_{t+1}) from the current player's view is -V(s_{t+1}).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from hybrid.core.env import GameState
from hybrid.core.types import Side
from .features import extract_features, feature_dim


@dataclass
class TDConfig:
    alpha: float = 0.05
    gamma: float = 1.0  # finite-horizon game, gamma=1 is standard
    l2: float = 1e-4    # small L2 regularization


class LinearValueFunction:
    """Linear value function V_theta(s)."""

    def __init__(self, dim: int):
        self.theta = np.zeros((dim,), dtype=np.float32)

    def value(self, state: GameState, perspective: Side) -> float:
        phi = extract_features(state, perspective)
        return float(np.dot(self.theta, phi))

    def update_td0(self, trajectory: List[GameState], final_outcome: float, cfg: TDConfig) -> None:
        """Run TD(0) updates over a full game trajectory.

        Args:
            trajectory: states s_0..s_{T-1} in chronological order.
            final_outcome: terminal reward from the last mover's perspective (+1/-1/0).
        """
        for t in range(len(trajectory)):
            s_t = trajectory[t]
            p = s_t.side_to_move

            v_t = self.value(s_t, p)

            if t == len(trajectory) - 1:
                target = final_outcome
            else:
                s_tp1 = trajectory[t + 1]
                # Next state is opponent's turn: negate value
                v_tp1 = self.value(s_tp1, s_tp1.side_to_move)
                target = 0.0 + cfg.gamma * (-v_tp1)

            delta = target - v_t
            phi = extract_features(s_t, p)
            # Gradient clipping
            grad = delta * phi
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1.0:
                grad = grad / grad_norm
            # L2 regularization + update
            self.theta *= (1.0 - cfg.alpha * cfg.l2)
            self.theta += cfg.alpha * grad

    def as_dict(self):
        return {"theta": self.theta.tolist()}

    @staticmethod
    def from_dict(d):
        v = LinearValueFunction(len(d["theta"]))
        v.theta = np.array(d["theta"], dtype=np.float32)
        return v
