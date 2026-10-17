"""Evaluation counting, statistics, game-loop and API tests.

The game loop is exercised against a scripted fake environment (counting logic
only); genuine RocketSim evaluation is covered by the bounded integration test.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from rocket_league_ai.evaluation import (
    EvaluationSpec,
    GameRecord,
    compare_evaluations,
    summarize,
    wilson_interval,
)
from rocket_league_ai.training.evaluate_core import opponent_action, play_game


def spec(games=4):
    return EvaluationSpec(run_id="a" * 32, checkpoint_id="step-000000000512-abcdef012345",
                          opponent_id="random-v1", games=games, seed=5)


def game(index, side, result):
    return GameRecord(index=index, candidate_side=side, result=result,
                      goals_for=1 if result == "win" else 0, goals_against=1 if result == "loss" else 0,
                      decisions=10, duration_seconds=10 / 15)


def test_wilson_interval_is_sane():
    low, high = wilson_interval(50, 100)
    assert 0.40 < low < 0.5 < high < 0.60
    low, high = wilson_interval(2, 4)
    assert low < 0.5 < high  # tiny samples stay visibly uncertain
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_summarize_requires_side_balance_and_complete_games():
    games = [game(0, "blue", "win"), game(1, "orange", "loss"), game(2, "blue", "draw"), game(3, "orange", "win")]
    result = summarize(spec(), games, candidate_id="ckpt", candidate_config_hash="h" * 64, environment_version="kickoff-v1")
    assert (result["wins"], result["losses"], result["draws"]) == (2, 1, 1)
    assert result["goals_for"] == 2 and result["goals_against"] == 1
    assert result["win_rate"] == 0.5 and result["side_balanced"] is True
    with pytest.raises(ValueError):
        summarize(spec(), games[:3], candidate_id="c", candidate_config_hash="h" * 64, environment_version="v")
    with pytest.raises(ValueError):  # not side balanced
        summarize(spec(), [game(0, "blue", "win"), game(1, "blue", "win"), game(2, "blue", "win"), game(3, "blue", "win")],
                  candidate_id="c", candidate_config_hash="h" * 64, environment_version="v")


def test_spec_rejects_odd_game_counts():
    with pytest.raises(ValidationError):
        EvaluationSpec(run_id="a" * 32, checkpoint_id="step-000000000512-abcdef012345", games=5)


def test_compare_ranks_by_lower_confidence_bound():
    def entry(eid, wins, games):
        low, high = wilson_interval(wins, games)
        return {"id": eid, "status": "completed", "created": 1.0,
                "result": {"candidate_checkpoint_id": eid, "candidate_run_id": "r", "opponent_id": "random-v1",
                           "games": games, "win_rate": wins / games, "win_rate_ci95": [low, high],
                           "goals_for": wins, "goals_against": games - wins}}
    ranked = compare_evaluations([entry("lucky-small", 4, 4), entry("proven", 70, 100),
                                  {"id": "x", "status": "failed", "created": 1.0, "result": None}])
    assert [item["candidate_checkpoint_id"] for item in ranked] == ["proven", "lucky-small"]


def _fake_physics():
    from types import SimpleNamespace

    return SimpleNamespace(position=np.zeros(3), linear_velocity=np.zeros(3), yaw=0.0)


class FakeEnv:
    """Counting fixture only: candidate scores on step 3, else truncates."""

    def __init__(self, candidate_scores):
        from types import SimpleNamespace

        self.candidate_scores = candidate_scores
        self.steps = 0
        car = lambda blue: SimpleNamespace(physics=_fake_physics(), is_blue=blue, boost_amount=33.0)  # noqa: E731
        self.state = SimpleNamespace(
            ball=SimpleNamespace(position=np.array([0.0, 0.0, 93.0]), linear_velocity=np.zeros(3)),
            cars={"blue-0": car(True), "orange-0": car(False)})

    def reset(self):
        self.steps = 0
        return {"blue-0": np.zeros(92, dtype=np.float32), "orange-0": np.zeros(92, dtype=np.float32)}

    def step(self, actions):
        self.steps += 1
        terminated = {"blue-0": False, "orange-0": False}
        truncated = {"blue-0": False, "orange-0": False}
        rewards = {"blue-0": 0.0, "orange-0": 0.0}
        if self.candidate_scores and self.steps == 3:
            terminated = {"blue-0": True, "orange-0": True}
            rewards = {"blue-0": 10.0, "orange-0": -10.0}
        elif self.steps >= 5:
            truncated = {"blue-0": True, "orange-0": True}
        return ({"blue-0": np.zeros(92, dtype=np.float32), "orange-0": np.zeros(92, dtype=np.float32)},
                rewards, terminated, truncated)


def test_play_game_counts_goal_and_draw_from_terminal_rewards():
    win = play_game(FakeEnv(True), lambda agent, obs: 8, "idle-v1", index=0,
                    candidate_side="blue", seed=1, max_decisions=100, record=False)
    assert win.result == "win" and win.goals_for == 1 and win.decisions == 3
    draw = play_game(FakeEnv(False), lambda agent, obs: 8, "idle-v1", index=1,
                     candidate_side="orange", seed=1, max_decisions=100, record=False)
    assert draw.result == "draw" and draw.goals_for == 0


def test_opponents_are_seeded_and_idle_is_no_input():
    rng = np.random.default_rng(3)
    sequence = [opponent_action("random-v1", rng) for _ in range(5)]
    rng = np.random.default_rng(3)
    assert [opponent_action("random-v1", rng) for _ in range(5)] == sequence
    assert all(0 <= action < 90 for action in sequence)
    assert opponent_action("idle-v1", rng) == 8


def test_evaluation_api_persists_and_exposes_trajectory(client, manager, tmp_path):
    from conftest import unit_checkpoint_writer

    from rocket_league_ai.config import TrainingConfig
    from rocket_league_ai.metadata import experiment_metadata

    run = manager.store.create_run(TrainingConfig(), experiment_metadata())
    manager.store.update_run(run["id"], status="stopped")
    checkpoint = manager.checkpoints(run["id"]).publish(512, run["config_hash"], unit_checkpoint_writer, {})
    payload = {"run_id": run["id"], "checkpoint_id": checkpoint["id"], "games": 4, "seed": 9}
    # No evaluation may start while the (fake) checkpoint cannot be verified as a real model:
    # the API validates existence/integrity through the orchestrator target check.
    response = client.post("/api/evaluations", json=payload)
    assert response.status_code == 200, response.text
    evaluation_id = response.json()["id"]
    # The real subprocess will fail on fake weights; the persisted record must say so honestly.
    import time
    deadline = time.time() + 60
    record = manager.store.evaluation(evaluation_id)
    while record["status"] == "running" and time.time() < deadline:
        time.sleep(0.1)
        record = manager.store.evaluation(evaluation_id)
    assert record["status"] == "failed" and record["error"]
    assert client.get(f"/api/evaluations/{evaluation_id}/trajectory").status_code in (404, 409)
    listing = client.get("/api/evaluations").json()
    assert listing["evaluations"][0]["id"] == evaluation_id and listing["comparison"] == []


def test_checkpoint_opponent_requires_both_ids_and_is_labelled():
    with pytest.raises(ValidationError):
        EvaluationSpec(run_id="a" * 32, checkpoint_id="step-000000000512-abcdef012345",
                       opponent_run_id="b" * 32)
    checkpoint_spec = EvaluationSpec(
        run_id="a" * 32, checkpoint_id="step-000000000512-abcdef012345",
        opponent_run_id="b" * 32, opponent_checkpoint_id="step-000000000128-012345abcdef", games=2)
    games = [game(0, "blue", "draw"), game(1, "orange", "draw")]
    result = summarize(checkpoint_spec, games, candidate_id="c", candidate_config_hash="h" * 64,
                       environment_version="kickoff-v1")
    assert result["opponent_id"] == "checkpoint:step-000000000128-012345abcdef"
    assert "Frozen checkpoint" in result["opponent_description"]


def test_store_migration_adds_progress_to_existing_evaluations_table(tmp_path):
    import sqlite3

    from rocket_league_ai.store import Store

    db_path = tmp_path / "control.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE evaluations (id TEXT PRIMARY KEY, spec TEXT NOT NULL, status TEXT NOT NULL,"
        " result TEXT, trajectory TEXT, error TEXT, created REAL NOT NULL, updated REAL NOT NULL)")
    connection.commit()
    connection.close()
    store = Store(tmp_path)  # opening an old workspace must migrate, never fail or delete
    columns = {row[1] for row in store.db.execute("PRAGMA table_info(evaluations)")}
    assert "progress" in columns
    store.close()


def test_multi_seed_total_games_and_seed_schedule():
    multi = EvaluationSpec(run_id="a" * 32, checkpoint_id="step-000000000512-abcdef012345",
                           games=4, seed=100, extra_seeds=[500, 900])
    assert multi.total_games == 12
    assert [multi.seed_for_game(i) for i in (0, 3, 4, 7, 8, 11)] == [100, 103, 500, 503, 900, 903]
    games = []
    for i in range(12):
        record = game(i, "blue" if i % 2 == 0 else "orange", "draw")
        games.append(record.model_copy(update={"seed": multi.seed_for_game(i)}))
    result = summarize(multi, games, candidate_id="c", candidate_config_hash="h" * 64, environment_version="kickoff-v1")
    assert result["games"] == 12 and result["extra_seeds"] == [500, 900]


def test_play_game_records_chosen_actions_in_frames():
    frames = []
    table = [[0.0] * 8 for _ in range(90)]
    table[8] = [0, 0, 0, 0, 0, 0, 0, 0]
    play_game(FakeEnv(True), lambda agent, obs: (8, 0.42), "idle-v1", index=0,
              candidate_side="blue", seed=1, max_decisions=100, record=True, frames=frames,
              action_table=table)
    assert frames and frames[0]["candidate_action"] == 8
    assert frames[0]["candidate_action_prob"] == 0.42
    assert frames[0]["candidate_action_vector"] == [0.0] * 8
