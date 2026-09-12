"""Unit tests for the Robust Balance Curve engine and Bradley-Terry population modeling."""
import tempfile
from pathlib import Path
import pytest

from hybrid.rl.balance_curve import (
    create_agent,
    play_balance_game,
    summarize_balance_records,
    AGENT_BUDGETS,
)
from scripts.fit_balance_population import fit_bradley_terry_population


def test_agent_factory_and_budgets():
    for name in ("random", "greedy", "ab_d1", "ab_d2", "pure_mcts_32"):
        agent = create_agent(name, seed=42)
        assert agent is not None
        assert name in AGENT_BUDGETS


def test_play_balance_game_instrumentation():
    job = {
        "id": "test_game_01",
        "variant": "golden_palace_draw",
        "chess_agent": "random",
        "xiangqi_agent": "random",
        "seed": 12345,
        "max_plies": 20,
    }
    record = play_balance_game(job)
    assert record is not None
    assert record["id"] == "test_game_01"
    assert record["variant"] == "golden_palace_draw"
    assert record["plies"] > 0
    assert record["seconds"] > 0
    assert "captures" in record
    assert "chess_checks" in record
    assert "xiangqi_checks" in record
    assert record["winner"] in ("chess", "xiangqi", None)


def test_summarize_balance_records():
    dummy_records = [
        {
            "id": "g1", "variant": "golden_palace_draw", "chess_agent": "random", "xiangqi_agent": "random",
            "is_symmetric": True, "winner": "chess", "plies": 40, "chess_checks": 2, "xiangqi_checks": 1, "captures": 5, "reason": "Checkmate"
        },
        {
            "id": "g2", "variant": "golden_palace_draw", "chess_agent": "random", "xiangqi_agent": "random",
            "is_symmetric": True, "winner": "xiangqi", "plies": 60, "chess_checks": 0, "xiangqi_checks": 3, "captures": 6, "reason": "Checkmate"
        },
        {
            "id": "g3", "variant": "golden_palace_draw", "chess_agent": "random", "xiangqi_agent": "random",
            "is_symmetric": True, "winner": None, "plies": 80, "chess_checks": 1, "xiangqi_checks": 1, "captures": 2, "reason": "Threefold repetition"
        },
    ]
    summary = summarize_balance_records(dummy_records, expected_games=3)
    assert summary["complete"] is True
    assert "golden_palace_draw" in summary["summary_by_variant"]
    
    g_summary = summary["summary_by_variant"]["golden_palace_draw"]
    curve = g_summary["balance_curve"]
    assert len(curve) == 1
    assert curve[0]["agent"] == "random"
    assert curve[0]["chess_score"] == 0.5
    assert curve[0]["xiangqi_score"] == 0.5
    assert curve[0]["army_advantage"] == 0.0

    quality = g_summary["game_quality"]
    assert quality["decisiveness"] == pytest.approx(2 / 3, 0.01)
    assert quality["draw_rate"] == pytest.approx(1 / 3, 0.01)
    assert quality["plies_median"] == 60.0


def test_fit_bradley_terry_population():
    # Construct synthetic game records where Chess has an advantage under 'none'
    # and near 50/50 under 'golden'
    games = []
    # Variant 'none': Chess wins 8 out of 10
    for i in range(10):
        winner = "chess" if i < 8 else "xiangqi"
        games.append({
            "variant": "none",
            "chess_agent": "greedy",
            "xiangqi_agent": "greedy",
            "winner": winner,
            "chess_budget": 1.0,
            "xiangqi_budget": 1.0,
        })
    # Variant 'golden': Chess wins 5, XQ wins 5
    for i in range(10):
        winner = "chess" if i < 5 else "xiangqi"
        games.append({
            "variant": "golden",
            "chess_agent": "greedy",
            "xiangqi_agent": "greedy",
            "winner": winner,
            "chess_budget": 1.0,
            "xiangqi_budget": 1.0,
        })

    results = fit_bradley_terry_population(games, ["none", "golden"], ["greedy"])
    assert results["n_games"] == 20
    assert "none" in results["variants"]
    assert "golden" in results["variants"]
    # 'none' should have higher Chess bias than 'golden'
    beta_none = results["variants"]["none"]["army_bias_beta"]
    beta_golden = results["variants"]["golden"]["army_bias_beta"]
    assert beta_none > beta_golden
