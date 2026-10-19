"""Training-time accounting.

The rules the whole project depends on:

* a duration is measured with a monotonic clock, never by subtracting two
  wall-clock timestamps;
* the accumulated total survives a resume;
* a checkpoint never claims more training time than was measured;
* an old checkpoint without time metadata stays loadable and reports
  "unknown" instead of a made-up number;
* time between sessions is not training time.
"""

from __future__ import annotations

import time

import pytest

from rlai.checkpoints import load_checkpoint, save_checkpoint, validate_checkpoint
from rlai.store import Store
from rlai.timing import (
    COMPLETED,
    INTERRUPTED,
    SessionTimer,
    TrainingClock,
    duration_summary,
    format_duration,
    merge_durations,
)


POLICY_CONFIG = {"obs_size": 198, "hidden": [16], "activation": "tanh"}


def _tiny_state():
    """A real (tiny) parameter state.

    The checkpoint validator compares the stored tensors against the declared
    architecture, so the test has to use genuine weights - a synthetic dict
    would be rejected for the right reason but for the wrong purpose.
    """
    from rlai.policy import PolicyConfig

    model = PolicyConfig.from_dict(POLICY_CONFIG).build()
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


class FakeClock:
    """A monotonic clock the test controls."""

    def __init__(self, start: float = 1000.0):
        self.value = float(start)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


# ---------------------------------------------------------------- session timer
def test_session_timer_measures_with_the_monotonic_clock():
    clock = FakeClock()
    timer = SessionTimer("run-1", monotonic=clock)
    clock.advance(12.5)
    assert timer.elapsed() == pytest.approx(12.5)
    record = timer.stop(COMPLETED)
    assert record.duration_s == pytest.approx(12.5)
    assert record.status == COMPLETED
    assert record.ended_at is not None


def test_stopping_twice_does_not_change_the_duration():
    clock = FakeClock()
    timer = SessionTimer("run-1", monotonic=clock)
    clock.advance(5.0)
    first = timer.stop(COMPLETED)
    clock.advance(100.0)
    second = timer.stop(COMPLETED)
    assert first.duration_s == second.duration_s == pytest.approx(5.0)


def test_interrupted_session_keeps_its_duration():
    clock = FakeClock()
    timer = SessionTimer("run-1", monotonic=clock)
    clock.advance(3.0)
    record = timer.mark_interrupted("the process died")
    assert record.status == INTERRUPTED
    assert record.duration_s == pytest.approx(3.0)
    assert record.note == "the process died"


def test_running_session_has_no_final_duration():
    clock = FakeClock()
    timer = SessionTimer("run-1", monotonic=clock)
    clock.advance(2.0)
    # while the session runs the duration is live, but the record has none
    assert timer.duration_s == pytest.approx(2.0)
    assert timer.record().duration_s is None


# ---------------------------------------------------------------- training clock
def test_training_clock_accumulates_over_sessions():
    clock = FakeClock()
    training = TrainingClock("run-1", monotonic=clock)
    training.start_session()
    clock.advance(10.0)
    training.stop_session(COMPLETED)
    assert training.accumulated_s == pytest.approx(10.0)

    training.start_session()
    clock.advance(5.0)
    training.stop_session(COMPLETED)
    assert training.accumulated_s == pytest.approx(15.0)
    assert len(training.sessions) == 2


def test_training_clock_refuses_two_running_sessions():
    training = TrainingClock("run-1", monotonic=FakeClock())
    training.start_session()
    with pytest.raises(RuntimeError):
        training.start_session()


def test_restore_sets_the_accumulated_total():
    training = TrainingClock("run-1", monotonic=FakeClock())
    training.restore(120.0)
    assert training.accumulated_s == pytest.approx(120.0)
    # an unknown historical duration starts at zero instead of guessing
    training.restore(None)
    assert training.accumulated_s == 0.0


def test_current_session_is_separate_from_the_total():
    clock = FakeClock()
    training = TrainingClock("run-1", accumulated_s=100.0, monotonic=clock)
    training.start_session()
    clock.advance(7.0)
    assert training.current_session_s == pytest.approx(7.0)
    assert training.accumulated_s == pytest.approx(100.0)
    assert training.total_s == pytest.approx(107.0)


# ---------------------------------------------------------------- helpers
def test_merge_durations_keeps_unknown_as_unknown():
    assert merge_durations(None, None) is None
    assert merge_durations(None, 3.0) == pytest.approx(3.0)
    assert merge_durations(1.0, 2.0) == pytest.approx(3.0)


def test_format_duration_reads_naturally():
    assert format_duration(None) == "unknown"
    assert format_duration(30) == "30 s"
    assert format_duration(90) == "1m 30s"
    assert format_duration(3725) == "1h 02m 05s"


def test_duration_summary_marks_missing_values():
    summary = duration_summary(accumulated_s=None, current_session_s=5.0)
    assert summary["accumulated_available"] is False
    assert summary["current_available"] is True
    assert summary["total_s"] == pytest.approx(5.0)


# ---------------------------------------------------------------- store
def test_store_records_and_closes_sessions(tmp_path):
    store = Store(tmp_path / "db.sqlite3")
    session_id = store.start_session("run-1", started_at="2026-01-01T00:00:00+00:00", start_step=0)
    store.finish_session(
        session_id,
        ended_at="2026-01-01T00:00:10+00:00",
        duration_s=10.0,
        status=COMPLETED,
        end_step=512,
    )
    totals = store.session_totals("run-1")
    assert totals["accumulated_s"] == pytest.approx(10.0)
    assert totals["finished_count"] == 1
    assert totals["sessions"][0]["end_step"] == 512


def test_store_marks_left_open_sessions_as_interrupted(tmp_path):
    store = Store(tmp_path / "db.sqlite3")
    store.start_session("run-1", started_at="2026-01-01T00:00:00+00:00")
    recovered = store.recover_interrupted_sessions("run-1")
    assert recovered == 1
    totals = store.session_totals("run-1")
    # the duration was never measured, so it stays unknown
    assert totals["sessions"][0]["status"] == INTERRUPTED
    assert totals["sessions"][0]["duration_s"] is None
    assert totals["unfinished_count"] == 0


def test_store_separates_inherited_from_own_time(tmp_path):
    store = Store(tmp_path / "db.sqlite3")
    first = store.start_session("run-2", started_at="2026-01-01T00:00:00+00:00")
    store.finish_session(
        first, ended_at="2026-01-01T00:00:10+00:00", duration_s=10.0, status="inherited"
    )
    second = store.start_session("run-2", started_at="2026-01-01T01:00:00+00:00")
    store.finish_session(
        second, ended_at="2026-01-01T01:00:05+00:00", duration_s=5.0, status=COMPLETED
    )
    totals = store.session_totals("run-2")
    assert totals["accumulated_s"] == pytest.approx(15.0)
    assert totals["own_s"] == pytest.approx(5.0)
    assert totals["inherited_count"] == 1


# ---------------------------------------------------------------- checkpoints
def test_checkpoint_stores_the_accumulated_duration(tmp_path):
    models = tmp_path / "models"
    manifest = save_checkpoint(
        models,
        model_state=_tiny_state(),
        policy_config=POLICY_CONFIG,
        step=10,
        run_id="run-1",
        extra={"timing": {"accumulated_s": 42.5, "session_s": 2.5}},
    )
    assert manifest["duration_s"] == pytest.approx(42.5)
    assert manifest["session_duration_s"] == pytest.approx(2.5)

    validation = validate_checkpoint(models / manifest["path"])
    assert validation.ok
    assert validation.has_timing is True
    assert validation.duration_s == pytest.approx(42.5)


def test_checkpoint_without_timing_stays_loadable(tmp_path):
    models = tmp_path / "models"
    manifest = save_checkpoint(
        models,
        model_state=_tiny_state(),
        policy_config=POLICY_CONFIG,
        step=10,
        run_id="run-1",
    )
    assert manifest["duration_s"] is None
    validation = validate_checkpoint(models / manifest["path"])
    assert validation.ok
    assert validation.has_timing is False
    assert validation.duration_s is None
    payload = load_checkpoint(models / manifest["path"])
    assert payload["step"] == 10


def test_manifest_duration_matches_the_payload(tmp_path):
    models = tmp_path / "models"
    manifest = save_checkpoint(
        models,
        model_state=_tiny_state(),
        policy_config=POLICY_CONFIG,
        step=10,
        extra={"timing": {"accumulated_s": 7.0}},
    )
    payload = load_checkpoint(models / manifest["path"])
    assert payload["extra"]["timing"]["accumulated_s"] == pytest.approx(7.0)
    assert manifest["duration_s"] == pytest.approx(7.0)


# ---------------------------------------------------------------- real clock
def test_real_monotonic_clock_is_used_by_default():
    timer = SessionTimer("run-1")
    time.sleep(0.01)
    assert timer.elapsed() > 0.0
