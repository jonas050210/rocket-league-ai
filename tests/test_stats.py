"""The statistics behind the GUI's Stats tab.

The Stats tab shows one run: its training time, its progress, its learning
metrics, its performance, its checkpoints, its evaluations and its history.
Everything is computed by :mod:`rlai.stats` so the GUI never derives a
number itself and the headless report cannot disagree with the GUI.

The rules the tests protect:

* a value that was never measured is reported as *unavailable*, never as 0;
* the training time comes from the measured sessions;
* a checkpoint without duration metadata shows "unknown";
* the curves are downsampled, so a long run does not produce a huge payload.
"""

from __future__ import annotations

import pytest

from rlai import checkpoints
from rlai.policy import PolicyConfig
from rlai.stats import (
    DEFAULT_CURVE_METRICS,
    MAX_POINTS,
    build_stats,
    checkpoint_block,
    checkpoint_comparison,
    curves_block,
    evaluation_block,
    learning_block,
    performance_block,
    progress_block,
    timing_block,
)
from rlai.store import Store


@pytest.fixture()
def store(tmp_path):
    return Store(tmp_path / "db.sqlite3")


@pytest.fixture()
def run_id(store):
    return store.create_run("stats-run", "soccer-1v1", {"mode": "soccer-1v1"}, 10_000)


def _state():
    model = PolicyConfig(hidden=(16,)).build()
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


# ------------------------------------------------------------------- timing
def test_the_timing_block_is_unavailable_without_sessions(store, run_id):
    block = timing_block(store, store.get_run(run_id))
    assert block["available"] is False
    assert block["accumulated_s"] is None
    assert block["accumulated_human"] == "unknown"


def test_the_timing_block_sums_the_measured_sessions(store, run_id):
    first = store.start_session(run_id, started_at="2026-01-01T00:00:00+00:00")
    store.finish_session(
        first,
        ended_at="2026-01-01T00:00:10+00:00",
        duration_s=10.0,
        status="completed",
        end_step=100,
    )
    second = store.start_session(run_id, started_at="2026-01-01T01:00:00+00:00")
    store.finish_session(
        second,
        ended_at="2026-01-01T01:00:05+00:00",
        duration_s=5.0,
        status="stopped",
        end_step=200,
    )
    block = timing_block(store, store.get_run(run_id))
    assert block["available"] is True
    assert block["accumulated_s"] == pytest.approx(15.0)
    assert block["accumulated_human"] == "15 s"
    assert block["session_count"] == 2
    assert len(block["sessions"]) == 2
    assert block["sessions"][0]["status"] == "stopped"
    assert block["sessions"][0]["duration_human"] == "5 s"


def test_the_timing_block_reports_an_interrupted_session(store, run_id):
    session = store.start_session(run_id, started_at="2026-01-01T00:00:00+00:00")
    store.recover_interrupted_sessions(run_id)
    block = timing_block(store, store.get_run(run_id))
    assert block["interrupted_count"] == 1
    # the duration of a crashed session was never measured
    assert block["sessions"][0]["duration_s"] is None
    assert block["sessions"][0]["duration_human"] == "unknown"
    assert session


def test_the_timestamps_are_shown_in_local_time(store, run_id):
    session = store.start_session(run_id, started_at="2026-01-01T12:00:00+00:00")
    store.finish_session(
        session,
        ended_at="2026-01-01T12:00:10+00:00",
        duration_s=10.0,
        status="completed",
    )
    block = timing_block(store, store.get_run(run_id))
    assert block["sessions"][0]["started_at"] == "2026-01-01T12:00:00+00:00"
    # the local timestamp exists and is a string; the UTC one stays untouched
    assert isinstance(block["sessions"][0]["started_at_local"], str)


# ----------------------------------------------------------------- progress
def test_the_progress_block_reports_the_steps_and_the_episodes(store, run_id):
    store.update_run(run_id, status="running", step=512, total_steps=10_000)
    store.log_episode(run_id, 0, reward=1.0, length=100, goals_for=1, goals_against=0, result="win")
    store.log_episode(run_id, 1, reward=-1.0, length=90, goals_for=0, goals_against=2, result="loss")
    block = progress_block(store, store.get_run(run_id))
    assert block["status"] == "running"
    assert block["step"] == 512
    assert block["total_steps"] == 10_000
    assert block["progress"] == pytest.approx(0.0512)
    assert block["episode_count"] == 2


def test_the_progress_block_has_no_progress_without_a_target(store, run_id):
    store.update_run(run_id, total_steps=0)
    block = progress_block(store, store.get_run(run_id))
    assert block["progress"] is None


# ----------------------------------------------------------------- learning
def test_the_learning_block_summarises_the_measured_metrics(store, run_id):
    store.log_metric(run_id, 10, "charts/episodic_return", 1.0)
    store.log_metric(run_id, 20, "charts/episodic_return", 3.0)
    store.log_metric(run_id, 30, "losses/policy", 0.5)
    block = learning_block(store, store.get_run(run_id))
    assert block["charts/episodic_return"]["last"] == pytest.approx(3.0)
    assert block["charts/episodic_return"]["mean"] == pytest.approx(2.0)
    assert block["charts/episodic_return"]["count"] == 2
    assert block["losses/policy"]["last"] == pytest.approx(0.5)


def test_the_learning_block_says_when_nothing_was_measured(store, run_id):
    block = learning_block(store, store.get_run(run_id))
    assert block["available"] is False
    assert "no metrics" in block["reason"]


def test_the_learning_block_can_start_at_a_step(store, run_id):
    store.log_metric(run_id, 10, "charts/episodic_return", 1.0)
    store.log_metric(run_id, 500, "charts/episodic_return", 9.0)
    block = learning_block(store, store.get_run(run_id), since_step=100)
    assert block["charts/episodic_return"]["count"] == 1
    assert block["charts/episodic_return"]["last"] == pytest.approx(9.0)


# ------------------------------------------------------------------- curves
def test_the_curves_contain_the_measured_points(store, run_id):
    for step in range(10):
        store.log_metric(run_id, step * 10, "charts/win_rate", step / 10)
    block = curves_block(store, store.get_run(run_id), metrics=["charts/win_rate"])
    assert len(block["charts/win_rate"]) == 10
    assert block["charts/win_rate"][0]["value"] == pytest.approx(0.0)


def test_the_curves_are_downsampled(store, run_id):
    for step in range(MAX_POINTS * 3):
        store.log_metric(run_id, step, "charts/episodic_return", float(step))
    block = curves_block(store, store.get_run(run_id), metrics=["charts/episodic_return"])
    assert len(block["charts/episodic_return"]) <= MAX_POINTS + 1


def test_a_metric_without_points_is_not_in_the_curves(store, run_id):
    store.log_metric(run_id, 1, "charts/win_rate", 0.5)
    block = curves_block(store, store.get_run(run_id))
    assert "losses/policy" not in block


def test_the_default_curve_metrics_are_the_interesting_ones():
    assert "charts/episodic_return" in DEFAULT_CURVE_METRICS
    assert "losses/policy" in DEFAULT_CURVE_METRICS


# -------------------------------------------------------------- performance
def test_the_performance_block_reports_the_throughput(store, run_id):
    store.log_metric(run_id, 100, "time/sps", 120.0)
    store.log_metric(run_id, 200, "time/sps", 130.0)
    block = performance_block(store.get_run(run_id), store=store)
    assert block["available"] is True
    assert block["steps_per_second"] == pytest.approx(130.0)
    assert block["mean_steps_per_second"] == pytest.approx(125.0)
    assert "measured" in block["basis"]


def test_the_performance_block_has_no_throughput_when_it_was_not_measured(store, run_id):
    block = performance_block(store.get_run(run_id), store=store)
    assert block["available"] is False
    assert block["steps_per_second"] is None


def test_the_performance_block_divides_the_steps_by_the_measured_time(store, run_id):
    store.update_run(run_id, step=1000, duration_s=100.0)
    block = performance_block(store.get_run(run_id), store=store)
    assert block["run_steps_per_training_second"] == pytest.approx(10.0)
    assert "stopped" in block["run_steps_per_training_second_basis"]


# -------------------------------------------------------------- checkpoints
def test_the_checkpoint_block_carries_the_duration(store, tmp_path, run_id):
    models = tmp_path / "models"
    checkpoints.save_checkpoint(
        models,
        model_state=_state(),
        policy_config=PolicyConfig(hidden=(16,)).to_dict(),
        step=100,
        run_id=run_id,
        extra={"timing": {"accumulated_s": 42.5, "session_s": 2.5}},
    )
    rows = checkpoint_block(store, models, store.get_run(run_id))
    assert len(rows) == 1
    assert rows[0]["duration_s"] == pytest.approx(42.5)
    assert rows[0]["duration_human"] == "42 s"
    assert rows[0]["duration_available"] is True
    assert rows[0]["session_duration_s"] == pytest.approx(2.5)


def test_an_old_checkpoint_without_a_duration_shows_unknown(store, tmp_path, run_id):
    models = tmp_path / "models"
    checkpoints.save_checkpoint(
        models,
        model_state=_state(),
        policy_config=PolicyConfig(hidden=(16,)).to_dict(),
        step=100,
        run_id=run_id,
    )
    rows = checkpoint_block(store, models, store.get_run(run_id))
    assert rows[0]["duration_s"] is None
    assert rows[0]["duration_human"] == "unknown"
    assert rows[0]["duration_available"] is False


def test_the_checkpoints_of_another_run_are_not_shown(store, tmp_path, run_id):
    models = tmp_path / "models"
    checkpoints.save_checkpoint(
        models,
        model_state=_state(),
        policy_config=PolicyConfig(hidden=(16,)).to_dict(),
        step=100,
        run_id="other-run",
    )
    assert checkpoint_block(store, models, store.get_run(run_id)) == []


def test_the_checkpoint_comparison_needs_a_measured_score(store, tmp_path, run_id):
    models = tmp_path / "models"
    manifest = checkpoints.save_checkpoint(
        models,
        model_state=_state(),
        policy_config=PolicyConfig(hidden=(16,)).to_dict(),
        step=100,
        run_id=run_id,
    )
    comparison = checkpoint_comparison(store, models, store.get_run(run_id))
    assert comparison["available"] is False
    assert "no checkpoint" in comparison["reason"]

    store.register_model(manifest)
    store.set_model_eval_score(manifest["id"], 0.75)
    comparison = checkpoint_comparison(store, models, store.get_run(run_id))
    assert comparison["available"] is True
    assert comparison["best"]["id"] == manifest["id"]
    assert comparison["ranking"][0]["eval_score"] == pytest.approx(0.75)


def test_a_checkpoint_score_is_found_in_the_stored_evaluations(store, tmp_path, run_id):
    """A checkpoint the database never registered still has a measured score."""
    models = tmp_path / "models"
    manifest = checkpoints.save_checkpoint(
        models,
        model_state=_state(),
        policy_config=PolicyConfig(hidden=(16,)).to_dict(),
        step=100,
        run_id=run_id,
    )
    store.add_evaluation(
        {
            "mode": "soccer-1v1",
            "opponent": "difficulty:pro",
            "games": 4,
            "wins": 3,
            "losses": 1,
            "draws": 0,
            "goals_for": 6,
            "goals_against": 2,
            "run_id": run_id,
            "model_id": manifest["id"],
            "win_rate": 0.75,
        }
    )
    comparison = checkpoint_comparison(store, models, store.get_run(run_id))
    assert comparison["available"] is True
    assert comparison["best"]["eval_score"] == pytest.approx(0.75)


# --------------------------------------------------------------- evaluation
def test_the_evaluation_block_groups_by_opponent(store, run_id):
    for _ in range(2):
        store.add_evaluation(
            {
                "mode": "soccer-1v1",
                "opponent": "difficulty:pro",
                "games": 10,
                "wins": 7,
                "losses": 3,
                "draws": 0,
                "goals_for": 15,
                "goals_against": 9,
                "run_id": run_id,
                "model_id": "m1",
            }
        )
    block = evaluation_block(store, store.get_run(run_id))
    assert block["available"] is True
    assert block["count"] == 2
    entry = block["per_opponent"][0]
    assert entry["opponent"] == "difficulty:pro"
    assert entry["games"] == 20
    assert entry["wins"] == 14
    assert entry["win_rate"] == pytest.approx(0.7)


def test_the_evaluation_block_says_when_nothing_was_measured(store, run_id):
    block = evaluation_block(store, store.get_run(run_id))
    assert block["available"] is False
    assert "no evaluation" in block["reason"]


# -------------------------------------------------------------------- whole
def test_the_stats_of_an_unknown_run_are_unavailable(store):
    stats = build_stats(store, "does-not-exist")
    assert stats["available"] is False
    assert "unknown run" in stats["reason"]


def test_the_stats_view_is_complete(store, tmp_path, run_id):
    session = store.start_session(run_id, started_at="2026-01-01T00:00:00+00:00")
    store.finish_session(
        session,
        ended_at="2026-01-01T00:00:30+00:00",
        duration_s=30.0,
        status="completed",
        end_step=300,
    )
    store.update_run(run_id, status="running", step=300, total_steps=1000, duration_s=30.0)
    store.log_metric(run_id, 300, "time/sps", 42.0)
    store.log_metric(run_id, 300, "charts/episodic_return", 5.0)
    store.log_episode(run_id, 0, reward=5.0, length=100, goals_for=1, goals_against=0, result="win")
    store.add_evaluation(
        {
            "mode": "soccer-1v1",
            "opponent": "difficulty:medium",
            "games": 4,
            "wins": 2,
            "losses": 1,
            "draws": 1,
            "goals_for": 5,
            "goals_against": 4,
            "run_id": run_id,
            "model_id": "m1",
        }
    )
    models = tmp_path / "models"
    checkpoints.save_checkpoint(
        models,
        model_state=_state(),
        policy_config=PolicyConfig(hidden=(16,)).to_dict(),
        step=300,
        run_id=run_id,
        extra={"timing": {"accumulated_s": 30.0, "session_s": 30.0}},
    )

    stats = build_stats(store, run_id, models_dir=models)
    assert stats["available"] is True
    assert stats["timing"]["accumulated_s"] == pytest.approx(30.0)
    assert stats["progress"]["step"] == 300
    assert stats["learning"]["charts/episodic_return"]["last"] == pytest.approx(5.0)
    assert stats["performance"]["steps_per_second"] == pytest.approx(42.0)
    assert len(stats["checkpoints"]) == 1
    assert stats["checkpoints"][0]["duration_s"] == pytest.approx(30.0)
    assert stats["evaluation"]["available"] is True
    assert stats["history"] == stats["timing"]["sessions"]
    assert stats["retention"]["rows"] >= 1


def test_the_stats_view_works_without_a_models_directory(store, run_id):
    stats = build_stats(store, run_id)
    assert stats["available"] is True
    assert stats["checkpoints"] == []
    assert stats["comparison"]["available"] is False
