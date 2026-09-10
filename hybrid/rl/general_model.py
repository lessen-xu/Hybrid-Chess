"""Versioned rule-aware policy/value model. Legacy 15-plane models stay v1."""
from __future__ import annotations

import numpy as np
import torch

from hybrid.core.config import VariantConfig
from hybrid.core.rules import board_hash
from hybrid.rl.az_encoding import encode_state
from hybrid.rl.az_network import PolicyValueNet
from hybrid.web_variants import PRESETS, parse_variant

ENCODING_VERSION = 2
# Stable checkpoint contract; never derive this order from UI display order.
RULE_FIELDS = (
    "no_queen", "no_bishop", "one_rook", "remove_extra_pawn", "xq_queen",
    "extra_cannon", "extra_soldier", "chess_palace", "knight_block",
    "no_promotion", "no_queen_promotion", "flying_general",
)
CONTEXT_SIZE = len(RULE_FIELDS) + 2
INPUT_CHANNELS = 15 + CONTEXT_SIZE


def context_features(state):
    variant = parse_variant(state.variant.to_dict())
    remaining = max(0, state.max_plies - state.ply) / max(1, state.max_plies)
    repetitions = min(3, state.repetition.get(board_hash(state.board, state.side_to_move), 0)) / 3
    return np.asarray([float(getattr(variant, k)) for k in RULE_FIELDS] +
                      [remaining, repetitions], dtype=np.float32)


def encode_general(state):
    context = torch.from_numpy(context_features(state))[:, None, None].expand(-1, 10, 9)
    return torch.cat((encode_state(state), context), dim=0)


def encode_for_model(state, net):
    return encode_general(state) if getattr(net, "encoding_version", 1) == 2 else encode_state(state)


def new_model(channels=96, res_blocks=4):
    net = PolicyValueNet(in_channels=INPUT_CHANNELS, channels=channels, num_res_blocks=res_blocks)
    net.encoding_version = ENCODING_VERSION
    return net


def model_payload(net, **metadata):
    return {
        "encoding_version": ENCODING_VERSION, "rule_fields": list(RULE_FIELDS),
        "arch": {"in_channels": INPUT_CHANNELS,
                 "channels": net.initial_conv.out_channels,
                 "res_blocks": len(net.res_blocks)},
        "model": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
        **metadata,
    }


def validate_payload(payload):
    if payload.get("encoding_version") != ENCODING_VERSION:
        raise ValueError("Expected a version 2 rule-aware model")
    if payload.get("rule_fields") != list(RULE_FIELDS):
        raise ValueError("Incompatible rule feature order")
    if payload.get("arch", {}).get("in_channels") != INPUT_CHANNELS:
        raise ValueError("Incompatible state encoding")


def load_general_model(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_payload(payload)
    arch = payload["arch"]
    net = new_model(arch["channels"], arch["res_blocks"])
    net.load_state_dict(payload["model"])
    return net.to(device).eval()


def sample_variant(game_id, seed):
    """Every 30 games: four games per preset and six reproducible custom games."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, game_id, 731]))
    slot = game_id % 30
    preset = PRESETS[slot % 6]
    flags = dict(preset["variant"])
    if slot >= 24:
        for key in rng.choice(RULE_FIELDS, size=int(rng.integers(1, 4)), replace=False):
            flags[str(key)] = not flags[str(key)]
    return parse_variant(flags), (preset["id"] if slot < 24 else "custom")
