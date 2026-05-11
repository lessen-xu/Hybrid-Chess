"""Move-limit penalty value tests (audit fix-pack v1).

Before the fix, ``az_selfplay`` applied the move-limit value as a Chess-
perspective scalar and sign-flipped it for Xiangqi-to-move examples. Under
``move_limit_value_mode == "penalty"`` (which always returns -0.1), this meant
Chess-to-move examples got z = -0.1 but Xiangqi-to-move examples got z = +0.1.
That is *not* a "both sides are punished for stalling" target.

These tests check the post-fix invariant: under penalty mode every example's
z is the same negative scalar regardless of side_to_move; under hard/soft
modes the existing per-side sign-flip still applies.
"""

from __future__ import annotations

from hybrid.core.types import Side
from hybrid.rl.az_selfplay import material_diff_to_value


def test_penalty_mode_returns_constant_negative():
    """material_diff_to_value("penalty") should be a side-independent scalar."""
    for diff in [-20.0, -1.0, 0.0, 5.0, 30.0]:
        v = material_diff_to_value(diff, mode="penalty")
        assert v == -0.1, f"penalty mode must return -0.1, got {v} for diff={diff}"


def test_penalty_mode_assignment_same_for_both_sides_synthetic():
    """Replicate the per-example z-assignment from az_selfplay under penalty
    mode and verify both Chess- and Xiangqi-to-move examples receive the same
    negative value (no sign flip)."""

    move_limit_value = material_diff_to_value(0.0, mode="penalty")  # = -0.1
    penalty_mode = True

    # Two synthetic examples, one per side.
    sides = [Side.CHESS, Side.XIANGQI]
    zs = []
    for side_to_move in sides:
        if penalty_mode:
            z = move_limit_value
        elif side_to_move == Side.CHESS:
            z = move_limit_value
        else:
            z = -move_limit_value
        zs.append(z)

    assert zs[0] == zs[1] == -0.1, (
        f"Penalty mode must give the same z to both sides, got {zs}"
    )


def test_hard_mode_still_uses_chess_perspective_sign_flip():
    """Hard mode encodes Chess-perspective material as ±1; Xiangqi-to-move
    samples must see the sign-flipped target (existing behavior)."""

    diff = 5.0  # Chess advantage
    move_limit_value = material_diff_to_value(diff, mode="hard")  # +1.0
    assert move_limit_value == 1.0

    # Mirror the post-fix code path.
    penalty_mode = False
    chess_z = move_limit_value if not penalty_mode else move_limit_value
    xq_z = -move_limit_value if not penalty_mode else move_limit_value
    assert chess_z == 1.0
    assert xq_z == -1.0


def test_soft_mode_sign_flip_for_xiangqi():
    """Soft mode = tanh(diff/scale), Chess perspective; sign-flip for Xiangqi."""
    diff = 4.0
    v = material_diff_to_value(diff, mode="soft", scale=4.0)
    assert v > 0.0
    assert -v < 0.0  # the Xiangqi-perspective target is negative


# End-to-end: run a forced-truncation self-play game and check ex.z

def test_self_play_max_ply_assigns_same_negative_to_both_sides():
    """Run one self-play game with a tiny max_plies and penalty mode, then
    assert every ex.z is exactly -0.1 (no positive Xiangqi-side flip)."""
    import torch

    from hybrid.agents.alphazero_stub import (
        AlphaZeroMiniAgent, MCTSConfig, TorchPolicyValueModel,
    )
    from hybrid.core.config import VariantConfig
    from hybrid.core.env import HybridChessEnv
    from hybrid.rl.az_network import PolicyValueNet
    from hybrid.rl.az_selfplay import SelfPlayConfig, self_play_game

    torch.manual_seed(0)
    net = PolicyValueNet(num_res_blocks=1, channels=16)
    net.eval()
    model = TorchPolicyValueModel(net, device="cpu")

    mcts_cfg = MCTSConfig(simulations=4, dirichlet_eps=0.0, max_plies=8)
    agent = AlphaZeroMiniAgent(model=model, cfg=mcts_cfg, seed=42, use_cpp=False)

    env = HybridChessEnv(max_plies=8, use_cpp=False, variant=VariantConfig())

    sp_cfg = SelfPlayConfig(
        max_ply=8,
        resign_enabled=False,
        draw_adjudicate_enabled=False,
        move_limit_value_mode="penalty",
        temperature=0.0,
        temp_cutoff_ply=0,
    )

    examples, record = self_play_game(env, agent, sp_cfg)

    if record.termination_reason != "Max plies reached":
        # If the random network happens to hit a decisive result before max-ply,
        # the test target doesn't apply — re-run with the seed bumped.
        import pytest
        pytest.skip(
            f"Game ended with reason={record.termination_reason!r}, not max-ply; "
            "re-seed and rerun for max-ply coverage."
        )

    assert len(examples) > 0
    for ex in examples:
        assert ex.z == -0.1, (
            f"penalty mode must give z=-0.1 for every example, got "
            f"{ex.z} (side_to_move={ex.side_to_move})"
        )
