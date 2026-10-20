"""Scheduled task changes ("the ladder"): config validation and application.

The sample budget, not the wall clock, decides when a task has been learned.
An escalation point changes the environment (horizon, resets, opponent) and/or
the reward definition at a given step; weights, optimizer, step counter and
RNG stay untouched.
"""

from __future__ import annotations

import pytest

from rlai.config import EscalationPoint, TrainingConfig
from rlai.store import Store
from rlai.trainer import Trainer


def _config(**overrides) -> TrainingConfig:
    base = {
        "run_name": "escalation unit",
        "total_steps": 20000,
        "rollout_steps": 256,
        "n_workers": 1,
        "minibatch_size": 256,
        "eval_interval": 0,
        "policy": {"hidden": [16]},
        "device": "cpu",
    }
    base.update(overrides)
    return TrainingConfig(**base)


def _trainer(data_root, config: TrainingConfig) -> Trainer:
    store = Store(data_root / "rlai.sqlite3")
    run_id = store.create_run("escalation", config.mode, config.to_dict(), config.total_steps)
    return Trainer(config, run_id, store)


def test_point_only_carries_what_it_changes():
    point = EscalationPoint(at_step=5000, episode_seconds=30.0)
    assert point.to_dict() == {"at_step": 5000, "episode_seconds": 30.0}
    # empty strings and None are not part of the payload
    assert "label" not in point.to_dict() and "reward" not in point.to_dict()


def test_points_must_be_ordered_unique_and_reachable():
    with pytest.raises(ValueError, match="ordered by at_step"):
        _config(
            escalation=[
                EscalationPoint(at_step=9000),
                EscalationPoint(at_step=2000),
            ]
        )
    with pytest.raises(ValueError, match="same at_step"):
        _config(
            escalation=[
                EscalationPoint(at_step=2000),
                EscalationPoint(at_step=2000),
            ]
        )
    with pytest.raises(ValueError, match="never apply"):
        _config(escalation=[EscalationPoint(at_step=20000)])
    with pytest.raises(ValueError, match="no-touch timeout must be shorter"):
        _config(
            escalation=[
                EscalationPoint(at_step=2000, episode_seconds=20.0, no_touch_timeout_seconds=25.0)
            ]
        )


def test_invalid_components_are_rejected_at_load_time():
    with pytest.raises(ValueError, match="reset_mode"):
        EscalationPoint(at_step=1000, reset_mode="nope")
    with pytest.raises(ValueError, match="unknown reward components"):
        EscalationPoint(at_step=1000, reward={"not_a_component": 1.0})
    with pytest.raises(ValueError):
        EscalationPoint(at_step=1000, opponent="not-an-opponent")


def test_apply_changes_task_and_is_idempotent(data_root):
    config = _config(
        episode_seconds=20.0,
        no_touch_timeout_seconds=10.0,
        reset_mode="reach",
        reward={"goal": 10.0, "touch": 0.4},
        escalation=[
            EscalationPoint(
                at_step=4096,
                label="wider world",
                episode_seconds=45.0,
                no_touch_timeout_seconds=25.0,
                reset_mode="drill",
                reward={"goal": 25.0, "touch": 0.15},
            ),
            EscalationPoint(at_step=8192, label="full match", reset_mode="mixed"),
        ],
    )
    trainer = _trainer(data_root, config)

    # before the point nothing changes
    trainer.step = 2048
    trainer._apply_escalation()
    assert trainer.escalation_applied == 0
    assert trainer.config.episode_seconds == 20.0

    trainer.step = 4096
    trainer._apply_escalation()
    assert trainer.escalation_applied == 1
    assert trainer.config.episode_seconds == 45.0
    assert trainer.config.no_touch_timeout_seconds == 25.0
    assert trainer.config.reset_mode == "drill"
    assert trainer.config.reward.goal == 25.0
    # untouched components keep their value (a reward point is a delta)
    assert trainer.config.reward.touch == 0.15
    assert trainer.config.reward.concede == 10.0

    # applying twice must not double-apply
    trainer._apply_escalation()
    assert trainer.escalation_applied == 1

    # weights are not touched by an escalation
    assert trainer.model is not None

    trainer.step = 9000
    trainer._apply_escalation()
    assert trainer.escalation_applied == 2
    assert trainer.config.reset_mode == "mixed"
    # the horizon from the first point survives the second one
    assert trainer.config.episode_seconds == 45.0


def test_resumed_run_applies_points_that_are_already_behind_it(data_root):
    config = _config(
        escalation=[
            EscalationPoint(at_step=4096, episode_seconds=45.0),
            EscalationPoint(at_step=8192, reset_mode="drill"),
        ]
    )
    trainer = _trainer(data_root, config)
    # a continue-resume starts at the restored step: everything at or below it
    # is the task the checkpoint was trained on
    trainer.step = 8192
    trainer._apply_escalation()
    assert trainer.escalation_applied == 2
    assert trainer.config.episode_seconds == 45.0
    assert trainer.config.reset_mode == "drill"


def test_curriculum_owns_the_opponent_and_says_so(data_root):
    config = _config(
        curriculum=True,
        escalation=[EscalationPoint(at_step=2048, opponent="chase", label="harder")],
    )
    trainer = _trainer(data_root, config)
    trainer.step = 2048
    trainer._apply_escalation()
    assert trainer.config.opponent == config.opponent  # unchanged
    assert any("curriculum owns the opponent" in warning for warning in trainer.run_warnings)


def test_collect_forwards_reward_weights_to_the_worker():
    from rlai.workers import WorkerPool

    class FakeParent:
        def __init__(self):
            self.messages = []

        def send(self, message):
            self.messages.append(message)

        def poll(self, _timeout):
            return True

        def recv(self):
            return {"type": "rollout", "worker": 0}

    class FakeProcess:
        exitcode = None

    pool = WorkerPool.__new__(WorkerPool)
    pool.parents = [FakeParent()]
    pool.workers = [FakeProcess()]
    pool.collect_timeout = 1.0
    weights = {"goal": 25.0, "touch": 0.15}
    pool.collect({}, 10, 7, reward_weights=weights)
    assert pool.parents[0].messages[0]["reward"] == weights


def test_report_scopes_reward_components_to_the_current_task(data_root):
    from rlai.report import build_run_report

    config = _config(
        reward={"goal": 10.0, "touch": 0.4},
        escalation=[EscalationPoint(at_step=1000, label="harder", reward={"goal": 25.0})],
    )
    store = Store(data_root / "rlai.sqlite3")
    run_id = store.create_run("escalated", config.mode, config.to_dict(), config.total_steps)
    store.update_run(run_id, status="completed", step=2000)
    store.log_metric(run_id, 500, "reward/goal", 0.5)
    store.log_metric(run_id, 1500, "reward/goal", 4.0)

    report = build_run_report(store, run_id)
    assert report["reward_components"]["reward/goal"]["mean"] == 4.0
    assert any("changed at step 1000" in note for note in report["notes"])
