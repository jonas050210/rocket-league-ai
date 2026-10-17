"""Evidence-based insights and next-experiment suggestions (advisory only)."""

from rocket_league_ai.monitoring import next_experiment, recommendations

SYSTEM = {"gpu_name": "RTX", "memory_percent": 10}
PLAN = {"device": "cuda:0", "n_proc": 1, "reward_version": "goal10-touch0.1-transition-v1"}


def evaluation(win_rate, low, high, *, opponent="random-v1", games=100):
    return {"status": "completed", "result": {
        "candidate_checkpoint_id": "step-000000000512-abcdef012345", "opponent_id": opponent,
        "games": games, "win_rate": win_rate, "win_rate_ci95": [low, high],
        "goals_for": 3, "goals_against": 9}}


def test_next_experiment_starts_with_baseline_evaluation():
    suggestion = next_experiment([], [], None, PLAN)
    assert suggestion["kind"] == "evaluation"


def test_next_experiment_suggests_one_shaping_variable_after_weak_random_result():
    suggestion = next_experiment([], [evaluation(0.2, 0.05, 0.4)], None, PLAN)
    assert suggestion["kind"] == "reward_shaping"
    assert "only reward_velocity_to_ball_weight" in suggestion["action"]


def test_next_experiment_moves_to_checkpoint_opponents_after_baseline_success():
    suggestion = next_experiment([], [evaluation(0.9, 0.7, 0.97)], None, PLAN)
    assert suggestion["kind"] == "checkpoint_opponent"


def test_bottleneck_insight_uses_measured_time_shares():
    runs = [{"status": "running", "metrics": {"collection_seconds": 0.9, "learning_seconds": 0.1, "reward": 0.1}}]
    titles = [item["title"] for item in recommendations(runs, SYSTEM)]
    assert "Simulation collection is the bottleneck" in titles


def test_snapshot_evaluations_are_light_but_api_keeps_detail(client, manager):
    from rocket_league_ai.config import TrainingConfig
    from rocket_league_ai.metadata import experiment_metadata

    manager.store.create_run(TrainingConfig(), experiment_metadata())
    evaluation = manager.store.create_evaluation({"games": 2})
    manager.store.update_evaluation(evaluation["id"], status="completed", result={
        "wins": 1, "games_detail": [{"index": 0}]})
    snapshot = client.get("/api/snapshot").json()
    assert "games_detail" not in snapshot["evaluations"][0]["result"]
    full = client.get("/api/evaluations").json()
    assert full["evaluations"][0]["result"]["games_detail"] == [{"index": 0}]
    assert snapshot["next_experiment"] is not None
