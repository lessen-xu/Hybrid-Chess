# -*- coding: utf-8 -*-
"""Smoke test for the iterative training runner.

Runs 2 iterations with minimal config and verifies:
1. No errors during execution
2. metrics.csv is produced with correct row count
3. At least 1 checkpoint file
4. At least 1 replay npz file
5. config.json and best_model.pt exist
"""

import os
import csv
import tempfile
from pathlib import Path

import pytest

from hybrid.rl.az_runner import AZIterConfig, run_iterations


@pytest.fixture
def tiny_config():
    """Minimal config for fast smoke testing."""
    return AZIterConfig(
        iterations=2,
        selfplay_games_per_iter=1,
        simulations=5,
        selfplay_max_ply=40,
        batch_size=8,
        train_epochs=1,
        buffer_capacity_states=1000,
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        eval_games=2,
        eval_simulations=5,
        eval_interval=0,
        gating_min_games=2,
        gating_max_games=4,
        gating_step_games=2,
        gating_threshold=0.55,
        gating_confidence=0.95,
        gating_simulations=5,
        temp_cutoff=5,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        device="cpu",
        seed=42,
        ablation="none",
    )


def test_runner_smoke(tiny_config, tmp_path):
    """Smoke test: 2 iterations complete without errors."""
    outdir = tmp_path / "test_run"

    run_iterations(tiny_config, outdir)

    # metrics.csv exists with correct row count
    csv_path = outdir / "metrics.csv"
    assert csv_path.exists(), "metrics.csv not found"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2, f"Expected 2 rows in metrics.csv, got {len(rows)}"

    expected_cols = {"iter", "policy_loss", "value_loss", "total_loss",
                     "eval_random_w", "eval_random_d", "eval_random_l",
                     "gating_games_used", "gating_w", "gating_d", "gating_l",
                     "gating_p_hat", "gating_ci_low", "gating_ci_high", "gate",
                     "selfplay_max_ply", "draw_adjudicate_enabled",
                     "sp_rootv_min_mean", "sp_draw_adjudicated"}
    assert expected_cols.issubset(set(rows[0].keys())), \
        f"Missing columns: {expected_cols - set(rows[0].keys())}"

    # At least 1 checkpoint
    ckpt_files = list(outdir.glob("ckpt_iter*.pt"))
    assert len(ckpt_files) >= 1, "No checkpoint files found"

    # At least 1 replay file
    replay_files = list(outdir.glob("replay_iter*.npz"))
    assert len(replay_files) >= 1, "No replay files found"

    assert (outdir / "config.json").exists(), "config.json not found"
    assert (outdir / "best_model.pt").exists(), "best_model.pt not found"


def test_runner_losses_finite(tiny_config, tmp_path):
    """Training losses should be finite (not NaN/inf)."""
    outdir = tmp_path / "test_finite"
    run_iterations(tiny_config, outdir)

    csv_path = outdir / "metrics.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["policy_loss", "value_loss", "total_loss"]:
                val = float(row[key])
                assert val == val, f"{key} is NaN in iter {row['iter']}"
                assert abs(val) < 1e10, f"{key} is too large in iter {row['iter']}"


def test_runner_endgame_mode(tmp_path):
    """Smoke test: 2 iterations with 100% endgame starts + penalty reward."""
    cfg = AZIterConfig(
        iterations=2,
        selfplay_games_per_iter=2,
        simulations=5,
        selfplay_max_ply=40,
        selfplay_move_limit_value_mode="penalty",
        batch_size=8,
        train_epochs=1,
        buffer_capacity_states=1000,
        eval_games=2,
        eval_simulations=5,
        eval_interval=0,
        gating_min_games=2,
        gating_max_games=4,
        gating_step_games=2,
        gating_simulations=5,
        temp_cutoff=5,
        device="cpu",
        seed=42,
        ablation="none",
        endgame_ratio=1.0,
    )
    outdir = tmp_path / "test_endgame"
    run_iterations(cfg, outdir)

    csv_path = outdir / "metrics.csv"
    assert csv_path.exists()
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2

    # Endgame games should be short (< 40 ply on average)
    for row in rows:
        avg_ply = float(row["sp_avg_ply"])
        assert avg_ply <= 40, f"Expected avg_ply <= 40, got {avg_ply}"


def test_runner_curriculum_3phase(tmp_path):
    """Smoke test: 3phase curriculum with 3 iters (covers Phase 1 only)."""
    cfg = AZIterConfig(
        iterations=3,
        selfplay_games_per_iter=1,
        simulations=5,
        eval_simulations=5,
        selfplay_max_ply=40,       # base value, overridden by curriculum
        batch_size=8,
        train_epochs=1,
        buffer_capacity_states=1000,
        eval_games=2,
        eval_interval=0,
        gating_min_games=2,
        gating_max_games=4,
        gating_step_games=2,
        gating_simulations=5,
        temp_cutoff=5,
        device="cpu",
        seed=42,
        ablation="none",
        curriculum_schedule="3phase",
    )
    outdir = tmp_path / "test_curriculum"
    run_iterations(cfg, outdir)

    csv_path = outdir / "metrics.csv"
    assert csv_path.exists()
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3

    # Phase 1 (iter 0-4) should override max_ply to 80
    for row in rows:
        assert int(row["selfplay_max_ply"]) == 80, (
            f"Expected max_ply=80 (Phase 1), got {row['selfplay_max_ply']}"
        )
    # Gating disabled in Phase 1 → all iterations auto-accepted
    for row in rows:
        assert row["gate"] == "Y", "Phase 1 should auto-accept (gating disabled)"


def test_runner_curriculum_3phase_v2(tmp_path):
    """Smoke test: 3phase_v2 curriculum — endgame anchor + gating always off."""
    cfg = AZIterConfig(
        iterations=3,
        selfplay_games_per_iter=1,
        simulations=5,
        eval_simulations=5,
        selfplay_max_ply=40,       # base value, overridden by curriculum
        batch_size=8,
        train_epochs=1,
        buffer_capacity_states=1000,
        eval_games=2,
        eval_interval=0,
        gating_min_games=2,
        gating_max_games=4,
        gating_step_games=2,
        gating_simulations=5,
        temp_cutoff=5,
        device="cpu",
        seed=42,
        ablation="none",
        curriculum_schedule="3phase_v2",
    )
    outdir = tmp_path / "test_curriculum_v2"
    run_iterations(cfg, outdir)

    csv_path = outdir / "metrics.csv"
    assert csv_path.exists()
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3

    # Phase 1 (iter 0-4) should override max_ply to 80
    for row in rows:
        assert int(row["selfplay_max_ply"]) == 80, (
            f"Expected max_ply=80 (Phase 1), got {row['selfplay_max_ply']}"
        )
    # v2: endgame_ratio should be 0.8 in Phase 1
    for row in rows:
        assert float(row["endgame_ratio"]) == 0.8, (
            f"Expected endgame_ratio=0.8 (Phase 1 v2), got {row['endgame_ratio']}"
        )
    # Gating disabled in ALL phases → all iterations auto-accepted
    for row in rows:
        assert row["gate"] == "Y", "3phase_v2 should auto-accept (gating always off)"
