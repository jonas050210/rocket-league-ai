"""Genuine bounded CPU benchmark + evaluation against a real trained checkpoint.

These execute the actual worker and evaluator subprocesses; nothing is faked.
Kept deliberately tiny (hidden 32, batch 128) so they stay bounded on CPU.
"""

import time

import pytest

from rocket_league_ai.benchmark import BenchmarkSpec
from rocket_league_ai.config import TrainingConfig
from rocket_league_ai.evaluation import EvaluationSpec

pytestmark = pytest.mark.integration


def wait_for(predicate, timeout=1200):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.5)
    return False


def small_plan(manager, **overrides):
    config = TrainingConfig(
        name="Integration CPU test", device="cpu", hidden_sizes=(32, 32),
        total_timesteps=512, timesteps_per_iteration=128, batch_size=128, buffer_size=384,
        n_minibatches=2, checkpoint_every_iterations=2,
        episode_timeout_seconds=5, no_touch_timeout_seconds=5, **overrides,
    )
    manager.store.set_settings({"plan": config.model_dump(mode="json"), "plan_revision": 99})
    return config


def test_real_benchmark_measures_a_genuine_cpu_run(manager):
    small_plan(manager)
    spec = BenchmarkSpec(devices=["cpu"], n_proc_values=[1], batch_size=128,
                         warmup_iterations=1, measure_iterations=2, repeats=1, timeout_seconds=900)
    session = manager.start_benchmark(spec)
    assert wait_for(lambda: manager.store.benchmark(session["id"])["status"] != "running")
    session = manager.store.benchmark(session["id"])
    assert session["status"] == "completed", session
    results = manager.store.benchmark_results(session["id"])
    assert len(results) == 1 and results[0]["status"] == "completed", results
    aggregate = results[0]["aggregate"]
    assert aggregate["agent_steps_per_second"]["median"] > 0
    assert aggregate["simulation_steps_per_second"]["median"] > 0
    assert aggregate["ppo_steps_per_second"]["median"] > 0
    assert aggregate["iterations_measured"] == 2
    assert session["recommendation"]["candidate"] == {"device": "cpu", "n_proc": 1}
    # The measurement was a real run in the experiment history.
    assert manager.store.run(results[0]["run_id"])["status"] == "completed"


def test_real_evaluation_of_trained_checkpoint_with_replay(manager):
    config = small_plan(manager)
    launched = manager._launch(config)
    run_id = launched["run_id"]
    assert wait_for(lambda: manager.store.run(run_id)["status"] == "completed" and manager.active is None)
    checkpoint = manager.checkpoints(run_id).latest_valid(config.digest())
    assert checkpoint is not None
    spec = EvaluationSpec(run_id=run_id, checkpoint_id=checkpoint["id"], opponent_id="idle-v1",
                          games=2, seed=42, max_game_seconds=10, record_games=2)
    evaluation = manager.start_evaluation(spec)
    assert wait_for(lambda: manager.store.evaluation(evaluation["id"])["status"] != "running")
    record = manager.store.evaluation(evaluation["id"])
    assert record["status"] == "completed", record
    result = record["result"]
    assert result["games"] == 2
    assert result["wins"] + result["losses"] + result["draws"] == 2
    assert result["side_balanced"] is True
    trajectory = manager.trajectory(evaluation["id"])
    assert trajectory["candidate_side_by_game"] == ["blue", "orange"]
    assert len(trajectory["frames"]) > 10
    first = trajectory["frames"][0]
    assert abs(first["ball"][0]) < 1 and abs(first["ball"][1]) < 1  # kickoff centre
    assert len(first["cars"]) == 2
