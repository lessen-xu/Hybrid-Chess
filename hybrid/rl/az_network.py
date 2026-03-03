# -*- coding: utf-8 -*-
"""AlphaZero-Mini policy/value network (dual-head ResNet).

Architecture:
  - Backbone: Conv3x3 (14→64 ch) + 3 residual blocks (64 ch each).
  - Policy head: 1x1 conv → (B, 92, 10, 9) raw logits.
  - Value head: 1x1 conv → BN → ReLU → flatten(90) → FC(64) → ReLU → FC(1) → tanh.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .az_encoding import NUM_STATE_CHANNELS, TOTAL_POLICY_PLANES
from hybrid.core.config import BOARD_W, BOARD_H


class ResidualBlock(nn.Module):
    """Standard residual block: Conv3x3 → BN → ReLU → Conv3x3 → BN → skip → ReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out


class PolicyValueNet(nn.Module):
    """Dual-head network: shared backbone → policy logits + value scalar.

    Input:  (B, 14, 10, 9)
    Output: policy_logits (B, 92, 10, 9), value (B, 1) in [-1, 1].
    """

    def __init__(self, in_channels: int = NUM_STATE_CHANNELS,
                 num_res_blocks: int = 3, channels: int = 64):
        super().__init__()

        # Shared backbone
        self.initial_conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        # Policy head: 1x1 conv → 92 planes
        self.policy_conv = nn.Conv2d(channels, TOTAL_POLICY_PLANES, kernel_size=1)

        # Value head: 1x1 conv → flatten → MLP → tanh
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_H * BOARD_W, channels)
        self.value_fc2 = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor):
        """Forward pass. Returns (policy_logits, value)."""
        out = F.relu(self.initial_bn(self.initial_conv(x)))
        out = self.res_blocks(out)

        policy_logits = self.policy_conv(out)  # (B, 92, 10, 9)

        v = F.relu(self.value_bn(self.value_conv(out)))  # (B, 1, 10, 9)
        v = v.view(v.size(0), -1)                        # (B, 90)
        v = F.relu(self.value_fc1(v))                     # (B, 64)
        v = torch.tanh(self.value_fc2(v))                 # (B, 1)

        return policy_logits, v
