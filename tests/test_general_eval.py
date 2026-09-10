"""Evaluation protocol and durable CLI recovery checks."""
import json
import sys

import pytest
import torch

from hybrid.rl import general_eval
from hybrid.rl.general_model import model_payload, new_model
from hybrid.web_variants import PRESETS


def test_summary_keeps_army_scores_and_incomplete_pairs_separate():
    records = [dict(preset="none", opponent="random", seed=seed,
        side=side, score=score, move_seconds=[.2]) for seed, side, score in
        [(1, "chess", 1.), (1, "xiangqi", 0.), (2, "chess", 1.)]]
    summary = general_eval.summarize(records, 4)
    group = summary["groups"][0]
    assert not summary["complete"] and summary["completed_games"] == 3
    assert group["chess_score"] == 1. and group["xiangqi_score"] == 0.
    assert group["complete_pairs"] == 1
    assert group["wins"] == 2 and group["losses"] == 1
    assert group["score_95_ci"][0] <= .5 <= group["score_95_ci"][1]


def test_evaluation_cli_resumes_completed_games_without_replaying(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(5)
    model = tmp_path / "fixture.pt"
    torch.save(model_payload(new_model(8, 1)), model)
    variants = [PRESETS[0], {"id": "custom", "variant": {
        "chess_palace": True, "knight_block": True, "xq_queen": True,
        "no_promotion": True, "extra_cannon": True}}]
    config = tmp_path / "variants.json"
    config.write_text(json.dumps(variants))
    output = tmp_path / "evaluation"
    monkeypatch.setattr(general_eval, "require_compute", lambda: None)
    monkeypatch.setattr(sys, "argv", ["general_eval", "--model", str(model),
        "--output", str(output), "--variants", str(config), "--games", "2",
        "--workers", "2", "--seconds", "0.001", "--seed", "24680", "--wall-seconds", "120"])
    general_eval.main()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete"] and summary["completed_games"] == 12
    assert json.loads((output / "identity.json").read_text())["opening_seed"] == 24680
    paths = sorted((output / "games").glob("*.json"))
    stamps = {p.name: p.stat().st_mtime_ns for p in paths}
    for path in paths:
        game = json.loads(path.read_text())
        assert game["reason"] and game["plies"] <= 400
        assert game["seed"] == (24680 if game["preset"] == "none" else 24780)
        assert len(game["states_ascii"]) == len(game["moves"]) + 1
    general_eval.main()
    assert stamps == {p.name: p.stat().st_mtime_ns for p in paths}
    assert json.loads((output / "summary.json").read_text()) == summary


@pytest.mark.parametrize("seconds", ["0", "-1", "nan"])
def test_invalid_budget_is_rejected_before_loading_model(seconds, monkeypatch):
    monkeypatch.setattr(general_eval, "require_compute", lambda: None)
    monkeypatch.setattr(sys, "argv", ["general_eval", "--model", "missing.pt",
        "--output", "unused", "--seconds", seconds])
    with pytest.raises(ValueError, match="per-move seconds"):
        general_eval.main()
