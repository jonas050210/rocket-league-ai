"""Match analysis and the honest limits of the replay support.

The analysis must work on the data the project really records (the
per-game rows of the evaluations and the episode rows) and it must say
plainly that Rocket League replay files cannot be parsed here.  Nothing in
this module may invent a number.
"""

from __future__ import annotations

import pytest

from rlai.replay import (
    AnalysisRequest,
    analyze_episodes,
    analyze_evaluations,
    analyze_matches,
    any_analysis,
    describe_sources,
    export_analysis,
    load_replay,
    merge_analyses,
    replay_support,
)
from rlai.store import Store


@pytest.fixture()
def store(tmp_path):
    return Store(tmp_path / "db.sqlite3")


def _evaluation(store, run_id, *, opponent="difficulty:pro", games=2, wins=1, losses=1):
    per_game = []
    for index in range(games):
        won = index < wins
        per_game.append(
            {
                "game": index,
                "side": "blue" if index % 2 == 0 else "orange",
                "mirrored": index % 2 == 1,
                "goals_for": 3 if won else 0,
                "goals_against": 0 if won else 2,
                "outcome": "win" if won else "loss",
                "ticks": 7200,
                "touches": 12 + index,
                "seed": 1000 + index,
            }
        )
    store.add_evaluation(
        {
            "mode": "soccer-1v1",
            "opponent": opponent,
            "games": games,
            "wins": wins,
            "losses": losses,
            "draws": 0,
            "goals_for": sum(game["goals_for"] for game in per_game),
            "goals_against": sum(game["goals_against"] for game in per_game),
            "run_id": run_id,
            "model_id": "m1",
            "details": {"games": per_game, "protocol": "soccer-1v1"},
        }
    )


# ------------------------------------------------------------------ replays
def test_the_replay_support_is_explicitly_unavailable():
    support = replay_support()
    assert support["available"] is False
    assert "no replay parser" in support["reason"]
    assert "evaluation game rows" in support["supported_inputs"]


def test_loading_a_replay_reports_the_limitation(tmp_path):
    path = tmp_path / "match.replay"
    path.write_bytes(b"not a real replay")
    result = load_replay(path)
    assert result["available"] is False
    assert "not available" in result["reason"]
    assert result["size_bytes"] == len(b"not a real replay")


def test_loading_a_missing_replay_says_so(tmp_path):
    result = load_replay(tmp_path / "missing.replay")
    assert result["available"] is False
    assert "does not exist" in result["reason"]


# -------------------------------------------------------------- evaluations
def test_the_evaluation_analysis_needs_per_game_rows(store):
    run_id = store.create_run("r", "soccer-1v1", {}, 100)
    store.add_evaluation(
        {
            "mode": "soccer-1v1",
            "opponent": "difficulty:pro",
            "games": 2,
            "wins": 1,
            "losses": 1,
            "draws": 0,
            "goals_for": 3,
            "goals_against": 2,
            "run_id": run_id,
            "model_id": "m1",
        }
    )
    analysis = analyze_evaluations(store, run_id)
    assert analysis["available"] is False
    assert "per-game details" in analysis["reason"]


def test_the_evaluation_analysis_summarises_the_measured_games(store):
    run_id = store.create_run("r", "soccer-1v1", {}, 100)
    _evaluation(store, run_id, games=4, wins=3, losses=1)
    analysis = analyze_evaluations(store, run_id)
    assert analysis["available"] is True
    assert analysis["games"] == 4
    summary = analysis["summary"]
    assert summary["wins"] == 3
    assert summary["losses"] == 1
    assert summary["win_rate"] == pytest.approx(0.75)
    assert summary["goals_for"] == 9
    assert summary["goals_against"] == 2
    assert summary["goal_diff"] == 7
    assert summary["mean_touches"] == pytest.approx(13.5)
    assert summary["mean_match_seconds"] == pytest.approx(60.0)


def test_the_evaluation_analysis_groups_by_opponent_and_side(store):
    run_id = store.create_run("r", "soccer-1v1", {}, 100)
    _evaluation(store, run_id, opponent="difficulty:pro", games=2, wins=2, losses=0)
    _evaluation(store, run_id, opponent="difficulty:rock", games=2, wins=0, losses=2)
    analysis = analyze_evaluations(store, run_id)
    opponents = {row["opponent"]: row for row in analysis["by_opponent"]}
    assert set(opponents) == {"difficulty:pro", "difficulty:rock"}
    assert opponents["difficulty:pro"]["win_rate"] == pytest.approx(1.0)
    assert opponents["difficulty:rock"]["win_rate"] == pytest.approx(0.0)
    sides = {row["side"] for row in analysis["by_side"]}
    assert sides == {"blue", "orange"}


def test_the_evaluation_analysis_builds_a_timeline(store):
    run_id = store.create_run("r", "soccer-1v1", {}, 100)
    _evaluation(store, run_id, games=2, wins=1, losses=1)
    analysis = analyze_evaluations(store, run_id)
    assert len(analysis["timeline"]) == 2
    assert all(row["outcome"] in {"win", "loss"} for row in analysis["timeline"])


# ----------------------------------------------------------------- episodes
def test_the_episode_analysis_needs_finished_episodes(store):
    run_id = store.create_run("r", "soccer-1v1", {}, 100)
    analysis = analyze_episodes(store, run_id)
    assert analysis["available"] is False
    assert "no episode" in analysis["reason"]


def test_the_episode_analysis_separates_optimisation_from_strength(store):
    run_id = store.create_run("r", "soccer-1v1", {}, 100)
    store.log_episode(
        run_id, 0, reward=1.0, length=100, goals_for=1, goals_against=0, result="win", opponent="pro"
    )
    store.log_episode(
        run_id, 1, reward=-1.0, length=80, goals_for=0, goals_against=1, result="loss", opponent="pro"
    )
    analysis = analyze_episodes(store, run_id)
    assert analysis["available"] is True
    assert analysis["summary"]["count"] == 2
    assert analysis["summary"]["wins"] == 1
    assert "strength comes from the evaluation" in analysis["note"]


# ------------------------------------------------------------------- whole
def test_the_match_analysis_states_its_sources(store):
    run_id = store.create_run("r", "soccer-1v1", {}, 100)
    _evaluation(store, run_id, games=2, wins=1, losses=1)
    store.log_episode(run_id, 0, reward=1.0, length=100, goals_for=1, goals_against=0, result="win")
    analysis = analyze_matches(store, run_id)
    assert analysis["schema"] == "rlai-match-analysis-v1"
    assert analysis["replays"]["available"] is False
    assert analysis["evaluations"]["available"] is True
    assert analysis["episodes"]["available"] is True
    assert "stored evaluations and episodes" in analysis["generated_from"]


def test_the_sources_are_described(store):
    sources = describe_sources()
    assert sources["evaluation_rows"]["available"] is True
    assert sources["episode_rows"]["available"] is True
    assert sources["replay_files"]["available"] is False


def test_the_analysis_can_be_exported(store, tmp_path):
    run_id = store.create_run("r", "soccer-1v1", {}, 100)
    _evaluation(store, run_id, games=2, wins=1, losses=1)
    analysis = analyze_matches(store, run_id)
    path = export_analysis(analysis, tmp_path / "out" / "analysis.json")
    assert path.exists()
    assert "rlai-match-analysis-v1" in path.read_text(encoding="utf-8")


def test_several_analyses_can_be_merged(store):
    run_id = store.create_run("r", "soccer-1v1", {}, 100)
    _evaluation(store, run_id, games=2, wins=1, losses=1)
    analysis = analyze_matches(store, run_id)
    merged = merge_analyses([analysis, analysis])
    assert merged["available"] is True
    assert merged["games"] == 4
    assert merged["sources"] == 2


def test_merging_nothing_is_unavailable():
    merged = merge_analyses([])
    assert merged["available"] is False
    assert "no analysis" in merged["reason"]


def test_the_helper_finds_the_per_game_rows(store):
    run_id = store.create_run("r", "soccer-1v1", {}, 100)
    _evaluation(store, run_id, games=2, wins=1, losses=1)
    rows = store.list_evaluations(run_id=run_id)
    assert any_analysis(rows)["available"] is True
    assert any_analysis([])["available"] is False


def test_the_request_object_describes_itself():
    request = AnalysisRequest(run_id="abc", include_episodes=False)
    assert request.to_dict() == {
        "run_id": "abc",
        "include_episodes": False,
        "include_timeline": True,
    }
