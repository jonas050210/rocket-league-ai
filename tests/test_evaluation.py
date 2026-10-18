"""Evaluation is deterministic for a seed and reports honest numbers."""

from __future__ import annotations

from rlai.evaluation import evaluate_model


def test_evaluation_reports_complete_record(model_path):
    result = evaluate_model(
        model_path,
        mode="soccer-1v1",
        opponent="idle",
        games=2,
        seed=11,
        match_seconds=6,
        episode_seconds=6,
        device="cpu",
    )
    assert result["games"] == 2
    assert result["wins"] + result["losses"] + result["draws"] == 2
    assert len(result["details"]["games"]) == 2
    assert result["goals_for"] >= 0 and result["goals_against"] >= 0
    for game in result["details"]["games"]:
        assert game["outcome"] in {"win", "loss", "draw"}
        assert game["ticks"] >= 0


def test_same_seed_same_result(model_path):
    kwargs = dict(
        mode="soccer-1v1", opponent="idle", games=1, seed=99, match_seconds=5, episode_seconds=5
    )
    first = evaluate_model(model_path, **kwargs)
    second = evaluate_model(model_path, **kwargs)
    assert first["wins"] == second["wins"]
    assert first["goals_for"] == second["goals_for"]
    assert first["goals_against"] == second["goals_against"]
    assert first["details"]["games"][0]["ticks"] == second["details"]["games"][0]["ticks"]
