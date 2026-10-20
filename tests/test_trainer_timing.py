"""The training time of a *real* run, end to end.

The unit tests of :mod:`rlai.timing` prove the arithmetic; these tests
prove the wiring: a trainer run opens a session, measures it with the
monotonic clock, writes it into the database, into the checkpoint and into
the run row - and a continued run keeps the total instead of restarting it.

The runs are deliberately tiny (a few hundred steps): they verify the
bookkeeping, not the learning.
"""

from __future__ import annotations

import pytest

from rlai import checkpoints
from rlai.config import TrainingConfig
from rlai.store import Store
from rlai.trainer import Trainer


def _config(**overrides) -> TrainingConfig:
    base = dict(
        run_name="timing",
        total_steps=2000,
        rollout_steps=64,
        n_workers=1,
        minibatch_size=64,
        eval_interval=0,
        policy={"hidden": [16]},
        device="cpu",
    )
    base.update(overrides)
    return TrainingConfig(**base)


def _trainer(config, store, run_id, **kwargs) -> Trainer:
    return Trainer(config, run_id, store, **kwargs)


def test_a_run_measures_its_own_training_time(data_root):
    store = Store(data_root / "rlai.sqlite3")
    config = _config()
    trainer = _trainer(config, store, "run-a")
    result = trainer.run()
    assert result["status"] == "completed"

    sessions = store.list_sessions("run-a")
    assert len(sessions) == 1
    session = sessions[0]
    assert session["status"] == "completed"
    # the duration was measured with the monotonic clock
    assert session["duration_s"] is not None
    assert session["duration_s"] > 0.0
    assert session["ended_at"] is not None
    # the session ends where the run ends (the last rollout can overshoot the
    # configured total a little, the recorded step is the real one)
    assert session["end_step"] == store.get_run("run-a")["step"]
    assert session["end_step"] >= config.total_steps

    run = store.get_run("run-a")
    assert run["duration_s"] == pytest.approx(session["duration_s"])
    assert run["session_duration_s"] == pytest.approx(session["duration_s"])
    assert run["started_at"] is not None
    assert run["ended_at"] is not None


def test_the_checkpoint_carries_the_accumulated_training_time(data_root):
    store = Store(data_root / "rlai.sqlite3")
    config = _config()
    trainer = _trainer(config, store, "run-b")
    trainer.run()

    manifests = checkpoints.list_models(data_root / "models")
    assert manifests
    for manifest in manifests:
        assert manifest["duration_s"] is not None
        assert manifest["duration_s"] > 0.0
        assert manifest["session_duration_s"] is not None

    payload = checkpoints.load_checkpoint(
        data_root / "models" / manifests[0]["path"]
    )
    timing = payload["extra"]["timing"]
    assert timing["source"] == "monotonic clock"
    assert timing["accumulated_s"] > 0.0
    assert timing["session_started_at"]
    assert timing["measured_at"]


def test_a_continued_run_keeps_the_accumulated_time(data_root):
    store = Store(data_root / "rlai.sqlite3")
    config = _config(total_steps=2000)
    first = _trainer(config, store, "run-c")
    first.run()
    first_total = store.session_totals("run-c")["accumulated_s"]
    assert first_total > 0.0

    # continue the same run from the checkpoint it just wrote
    manifests = checkpoints.list_models(data_root / "models")
    final = [m for m in manifests if m["label"] == "final"]
    assert final
    second = _trainer(
        _config(total_steps=4000),
        store,
        "run-c2",
        resume_from=str(data_root / "models" / final[0]["path"]),
        resume_mode="continue",
    )
    second.run()

    totals = store.session_totals("run-c2")
    # the inherited time of the parent run is part of the new total
    assert totals["inherited_count"] == 1
    assert totals["accumulated_s"] >= first_total
    assert totals["own_s"] > 0.0
    statuses = {row["status"] for row in totals["sessions"]}
    assert "inherited" in statuses
    assert "completed" in statuses


def test_a_continued_run_does_not_count_the_stopped_time(data_root):
    """The time between the two runs is not training time."""
    import time

    store = Store(data_root / "rlai.sqlite3")
    config = _config(total_steps=2000)
    started_wall = time.perf_counter()
    first = _trainer(config, store, "run-d")
    first.run()

    # "wait" between the two runs - the process was not training
    pause = 2.0
    time.sleep(pause)

    manifests = checkpoints.list_models(data_root / "models")
    final = [m for m in manifests if m["label"] == "final"][0]
    second = _trainer(
        _config(total_steps=4000),
        store,
        "run-d2",
        resume_from=str(data_root / "models" / final["path"]),
        resume_mode="continue",
    )
    second.run()
    wall_clock = time.perf_counter() - started_wall

    totals = store.session_totals("run-d2")
    # the total is the sum of the two *measured* sessions
    measured = sum(row["duration_s"] for row in totals["sessions"] if row["duration_s"])
    assert totals["accumulated_s"] == pytest.approx(measured)
    # ... and the pause is not part of it.  Comparing against the wall clock
    # of the whole test is load-independent: a busy machine inflates the
    # measured time and the wall clock alike, while an implementation that
    # charged "time since the run started" would leave no gap at all.
    # (The old assertion used a fixed 5 s bound and failed on a loaded
    # machine with 26.4 s of accumulated time - all of it real training.)
    assert wall_clock - totals["accumulated_s"] >= pause - 0.05


def test_a_resume_from_an_old_checkpoint_reports_an_unknown_duration(data_root):
    """A checkpoint without time metadata stays loadable - and honest."""
    store = Store(data_root / "rlai.sqlite3")
    config = _config()
    policy = config.policy_config()
    model = policy.build()
    old = checkpoints.save_checkpoint(
        data_root / "models",
        model_state=model.state_dict(),
        policy_config=policy.to_dict(),
        step=0,
        run_id="run-e",
        mode=config.mode,
    )
    trainer = _trainer(
        _config(total_steps=2000),
        store,
        "run-e",
        resume_from=str(data_root / "models" / old["path"]),
        resume_mode="continue",
    )
    trainer.run()
    totals = store.session_totals("run-e")
    # the historical duration is unknown, so it is not counted
    assert totals["inherited_count"] == 0
    assert totals["accumulated_s"] > 0.0
    messages = [row["message"] for row in store.tail_logs("run-e", limit=200)]
    assert any("no training-time metadata" in message for message in messages)


def test_an_interrupted_session_is_recovered_and_marked(data_root):
    """A session left open by a crashed process is closed as *interrupted*."""
    store = Store(data_root / "rlai.sqlite3")
    store.create_run("run-f", "soccer-1v1", {}, 128, run_id="run-f")
    store.start_session("run-f", started_at="2026-01-01T00:00:00+00:00", start_step=0)

    config = _config(total_steps=2000)
    trainer = _trainer(config, store, "run-f")
    trainer.run()

    sessions = store.list_sessions("run-f")
    statuses = [row["status"] for row in sessions]
    assert "interrupted" in statuses
    interrupted = [row for row in sessions if row["status"] == "interrupted"][0]
    # the duration of a crashed process was never measured
    assert interrupted["duration_s"] is None
    totals = store.session_totals("run-f")
    assert totals["unfinished_count"] == 0
    # the interrupted session is not part of the measured total
    assert totals["accumulated_s"] == pytest.approx(
        sum(row["duration_s"] for row in sessions if row["duration_s"])
    )


def test_a_stopped_run_keeps_its_measured_time(data_root):
    """A safe stop closes the session with the time measured so far."""
    store = Store(data_root / "rlai.sqlite3")
    config = _config(total_steps=100_000)
    trainer = _trainer(config, store, "run-g")
    trainer.run_dir.mkdir(parents=True, exist_ok=True)
    # ask the trainer to stop before the first iteration
    trainer.control_path.write_text('{"stop": true}', encoding="utf-8")
    result = trainer.run()
    assert result["status"] == "stopped"

    sessions = store.list_sessions("run-g")
    assert len(sessions) == 1
    assert sessions[0]["status"] == "stopped"
    assert sessions[0]["duration_s"] is not None
    assert sessions[0]["duration_s"] > 0.0
    assert store.get_run("run-g")["status"] == "stopped"


def test_the_headless_summary_contains_the_measured_time(data_root):
    from rlai.cli import _headless_summary

    store = Store(data_root / "rlai.sqlite3")
    config = _config(total_steps=2000)
    trainer = _trainer(config, store, "run-h")
    result = trainer.run()
    summary = _headless_summary(store, "run-h", result)
    assert summary["schema"] == "rlai-headless-summary-v1"
    assert summary["timing"]["accumulated_s"] > 0.0
    assert summary["timing"]["accumulated_human"]
    assert summary["status"] == "completed"
