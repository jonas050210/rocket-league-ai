"""Trainer lifecycle: run registration, resume state and best-model rule."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rlai import checkpoints
from rlai.config import TrainingConfig
from rlai.curriculum import Curriculum, stages_for_mode
from rlai.league import OpponentPool
from rlai.store import Store
from rlai.trainer import Trainer, is_best_candidate


def _config() -> TrainingConfig:
    return TrainingConfig(
        run_name="unit",
        total_steps=2000,
        rollout_steps=256,
        n_workers=1,
        minibatch_size=256,
        eval_interval=0,
        policy={"hidden": [16]},
        device="cpu",
    )


def test_best_candidate_requires_an_actual_win():
    assert is_best_candidate({"wins": 0, "win_rate": 0.0}, float("-inf")) is False
    assert is_best_candidate({"wins": 0, "win_rate": 0.0}, -1.0) is False
    assert is_best_candidate({"wins": 1, "win_rate": 0.25}, -1.0) is True
    assert is_best_candidate({"wins": 1, "win_rate": 0.25}, 0.5) is False


def test_run_creates_missing_run_row_and_final_checkpoint(data_root, tmp_path):
    config = _config()
    policy = config.policy_config()
    model = policy.build()
    checkpoint = checkpoints.save_checkpoint(
        data_root / "models",
        model_state=model.state_dict(),
        policy_config=policy.to_dict(),
        step=config.total_steps,
        mode=config.mode,
        label="resume",
    )
    store = Store(data_root / "rlai.sqlite3")
    assert store.get_run("cli-run") is None
    trainer = Trainer(
        config,
        "cli-run",
        store,
        resume_from=str(data_root / "models" / checkpoint["path"]),
    )
    result = trainer.run()
    assert result["status"] == "completed"
    assert result["step"] == config.total_steps
    assert result["best_win_rate"] is None
    assert store.get_run("cli-run")["status"] == "completed"
    final = [m for m in checkpoints.list_models(data_root / "models") if m["label"] == "final"]
    assert final and final[0]["step"] == config.total_steps


def test_resume_restores_league_and_curriculum(data_root):
    config = _config()
    policy = config.policy_config()
    model = policy.build()
    league = OpponentPool(capacity=4)
    league.add("snap", 512, {"w": 1})
    league.agent_elo = 1042.0
    league.record("snap", 3, 1, 0)
    curriculum = Curriculum(
        stages_for_mode(config.mode),
        start_index=1,
        stage_iterations=7,
        total_iterations=3,
    )
    checkpoint = checkpoints.save_checkpoint(
        data_root / "models",
        model_state=model.state_dict(),
        policy_config=policy.to_dict(),
        step=768,
        mode=config.mode,
        extra={
            "league": league.to_dict(),
            "curriculum": curriculum.describe(),
            "obs_version": 2,
        },
    )
    store = Store(data_root / "rlai.sqlite3")
    trainer = Trainer(
        config.model_copy(update={"curriculum": True}),
        "resume-run",
        store,
        resume_from=str(data_root / "models" / checkpoint["path"]),
    )
    trainer._load_resume()
    assert trainer.step == 768
    assert trainer.league.agent_elo == pytest.approx(league.agent_elo)
    assert len(trainer.league) == 1
    assert trainer.curriculum is not None
    assert trainer.curriculum.index == 1
    assert trainer.curriculum.stage_iterations == 7
    assert trainer.curriculum.total_iterations == 3


def test_resume_rejects_a_different_architecture(data_root):
    config = _config()
    from rlai.policy import PolicyConfig

    other = PolicyConfig(
        obs_size=config.policy_config().obs_size,
        hidden=(32,),
        activation=config.policy_config().activation,
    )
    model = other.build()
    checkpoint = checkpoints.save_checkpoint(
        data_root / "models",
        model_state=model.state_dict(),
        policy_config=other.to_dict(),
        step=128,
        mode=config.mode,
    )
    trainer = Trainer(
        config,
        "mismatch",
        Store(data_root / "rlai.sqlite3"),
        resume_from=str(data_root / "models" / checkpoint["path"]),
    )
    with pytest.raises(ValueError, match="architecture"):
        trainer._load_resume()
    # the loaded tensors are untouched by the failed resume
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())


def _trainer(config: TrainingConfig, store: Store, name: str) -> Trainer:
    run_id = store.create_run(name, config.mode, config.to_dict(), config.total_steps)
    return Trainer(config, run_id, store)


def test_self_play_checkpoint_keeps_a_playable_pool(data_root):
    """A resumed self-play run must face its real former selves, not only ratings."""
    config = _config().model_copy(update={"self_play": True, "snapshot_interval": 1})
    store = Store(data_root / "rlai.sqlite3")
    parent = _trainer(config, store, "parent")
    parent.step = 512
    parent.iteration = 2
    parent.league.add("snapshot-a", 256, parent._snapshot_state())
    parent.league.add("snapshot-b", 512, parent._snapshot_state())
    parent.league.record("snapshot-b", wins=3, losses=1, draws=0)
    elo_before = parent.league.agent_elo
    manifest = parent._save(label="test", force=True)
    assert manifest is not None

    payload = checkpoints.load_checkpoint(data_root / "models" / manifest["path"])
    league_data = payload["extra"]["league"]
    assert league_data["states_included"] is True
    assert all(entry["has_state"] for entry in league_data["entries"])

    child = _trainer(config, store, "child")
    child.resume_from = str(data_root / "models" / manifest["path"])
    child._load_resume()
    assert child.step == 512
    assert len(child.league) == 2
    assert child.league.usable == 2, "restored snapshots must be playable"
    assert child.league.agent_elo == pytest.approx(elo_before)
    entry = child.league.sample(child.rng)
    assert entry.id in {"snapshot-a", "snapshot-b"}
    # the restored weights are the ones the parent froze, and they are copies
    original = parent.league.entries[-1].state
    restored = child.league.entries[-1].state
    assert torch.equal(next(iter(original.values())), next(iter(restored.values())))
    assert next(iter(original.values())).data_ptr() != next(iter(restored.values())).data_ptr()


def test_league_states_can_be_turned_off(data_root):
    config = _config().model_copy(
        update={"self_play": True, "snapshot_interval": 1, "league_states": False}
    )
    store = Store(data_root / "rlai.sqlite3")
    parent = _trainer(config, store, "parent")
    parent.step = 256
    parent.league.add("snapshot-a", 256, parent._snapshot_state())
    manifest = parent._save(label="test", force=True)
    payload = checkpoints.load_checkpoint(data_root / "models" / manifest["path"])
    assert payload["extra"]["league"]["states_included"] is False

    child = _trainer(config, store, "child")
    child.resume_from = str(data_root / "models" / manifest["path"])
    child._load_resume()
    assert len(child.league) == 1 and child.league.usable == 0
    assert any(
        "ratings but no weights" in record["message"]
        for record in store.tail_logs(child.run_id, 0, 100)
    )


def _write_resumable_checkpoint(models_dir, config, *, step=512, hidden=None):
    from rlai.policy import PolicyConfig

    base = config.policy_config()
    policy = PolicyConfig(
        obs_size=base.obs_size,
        hidden=tuple(hidden or base.hidden),
        activation=base.activation,
    )
    model = policy.build()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    loss = model(torch.zeros(1, policy.obs_size))[0].pow(2).sum()
    loss.backward()
    optimizer.step()
    return checkpoints.save_checkpoint(
        models_dir,
        model_state=model.state_dict(),
        policy_config=policy.to_dict(),
        optimizer_state=optimizer.state_dict(),
        step=step,
        run_id="parent-run",
        mode=config.mode,
        training_config=config.to_dict(),
        extra={
            "iteration": step // 64,
            "obs_version": config.obs_version,
            "rng": {
                "numpy": np.random.default_rng(step).bit_generator.state,
                "torch_cpu": torch.get_rng_state(),
                "python": [1, 2, 3],
            },
            "curriculum": {"index": 1, "iterations": 1, "total_iterations": 2},
            "league": {"capacity": 8, "agent_elo": 1000.0, "entries": []},
        },
        label="final",
    )


def test_automatic_resume_without_an_explicit_reference(data_root):
    config = _config()
    manifest = _write_resumable_checkpoint(data_root / "models", config, step=512)
    store = Store(data_root / "rlai.sqlite3")
    trainer = Trainer(config, "child-run", store)
    trainer._load_resume()
    assert trainer.step == 512
    assert trainer.start_step == 512
    assert trainer.resume_plan is not None
    assert trainer.resume_plan.model_id == manifest["id"]
    assert trainer.resume_mode == "continue"


def test_no_checkpoint_starts_fresh_and_says_so(data_root):
    store = Store(data_root / "rlai.sqlite3")
    trainer = Trainer(_config(), "first-run", store)
    trainer._load_resume()
    assert trainer.step == 0
    assert trainer.resume_mode == "fresh"
    messages = [row["message"] for row in store.tail_logs("first-run", limit=100)]
    assert any("no checkpoint found" in message for message in messages)
    assert any("fresh training" in message for message in messages)


def test_automatic_resume_refuses_to_replace_unusable_progress(data_root):
    config = _config()
    _write_resumable_checkpoint(data_root / "models", config, step=512, hidden=(32,))
    trainer = Trainer(config, "cannot-resume", Store(data_root / "rlai.sqlite3"))
    with pytest.raises(ValueError, match="Refusing to start a fresh model"):
        trainer._load_resume()
    # the trainer did not pretend to have restored anything
    assert trainer.step == 0
    assert trainer.start_step == 0


def test_a_final_save_does_not_duplicate_the_checkpoint_at_the_same_step(data_root):
    store = Store(data_root / "rlai.sqlite3")
    config = _config()
    trainer = Trainer(config, "dup-run", store)
    trainer.step = 64
    first = trainer._save(label="candidate")
    second = trainer._save(label="final", force=True)
    assert first is not None and second is not None
    assert second["id"] == first["id"]
    assert len(checkpoints.list_models(data_root / "models")) == 1
    manifest = checkpoints.get_model(data_root / "models", first["id"])
    assert manifest["label"] == "final"
    assert checkpoints.latest_model(data_root / "models")["id"] == first["id"]


def test_normal_completion_writes_a_discoverable_final_checkpoint(data_root):
    config = _config()
    policy = config.policy_config()
    checkpoint = checkpoints.save_checkpoint(
        data_root / "models",
        model_state=policy.build().state_dict(),
        policy_config=policy.to_dict(),
        step=config.total_steps,
        mode=config.mode,
        label="resume",
    )
    store = Store(data_root / "rlai.sqlite3")
    trainer = Trainer(
        config,
        "complete-run",
        store,
        resume_from=str(data_root / "models" / checkpoint["path"]),
    )
    result = trainer.run()
    assert result["status"] == "completed"
    assert result["resumable"] is True
    assert result["interrupted"] is False

    latest = checkpoints.latest_model(data_root / "models")
    assert latest is not None
    assert latest["label"] == "final"
    assert latest["step"] == config.total_steps
    # the restored checkpoint was relabelled, not written a second time
    assert len(checkpoints.list_models(data_root / "models")) == 1

    from rlai.resume import plan_resume

    plan = plan_resume(data_root / "models", config, mode="continue", run_id="next-run")
    assert plan.will_resume is True
    assert plan.model_id == latest["id"]
    assert plan.restored_step == config.total_steps


class _InterruptingTrainer(Trainer):
    """Raise KeyboardInterrupt once, as Ctrl+C would during the loop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fired = False

    def _log_iteration(self, *args, **kwargs):
        super()._log_iteration(*args, **kwargs)
        if not self.fired:
            self.fired = True
            raise KeyboardInterrupt


def _interrupt_config() -> TrainingConfig:
    return TrainingConfig(
        run_name="unit",
        total_steps=100_000,
        rollout_steps=64,
        n_workers=1,
        minibatch_size=64,
        checkpoint_interval=1000,
        eval_interval=0,
        policy={"hidden": [16]},
        device="cpu",
    )


def test_ctrl_c_saves_a_resumable_checkpoint_and_closes_the_run(data_root):
    config = _interrupt_config()
    store = Store(data_root / "rlai.sqlite3")
    trainer = _InterruptingTrainer(config, "interrupted-run", store)

    result = trainer.run()

    assert result["status"] == "stopped"
    assert result["interrupted"] is True
    assert result["resumable"] is True
    assert result["step"] == 64
    assert store.get_run("interrupted-run")["status"] == "stopped"
    sessions = store.list_sessions("interrupted-run")
    assert sessions[0]["status"] == "stopped"
    assert sessions[0]["duration_s"] > 0.0

    latest = checkpoints.latest_model(data_root / "models")
    assert latest is not None
    assert latest["step"] == 64
    assert latest["label"] == "interrupt"
    assert latest["id"] == result["checkpoint"]
    report = store.get_report("interrupted-run")["report"]
    assert report["status"] == "stopped"
    assert report["final_checkpoint"]["step"] == 64
    assert report["final_checkpoint"]["valid"] is True

    # the next process discovers the interrupted checkpoint without being told
    resumed = Trainer(config, "next-run", store)
    resumed._load_resume()
    assert resumed.step == 64
    assert resumed.resume_plan.model_id == latest["id"]


class _FailingInterruptTrainer(_InterruptingTrainer):
    def _save(self, *, label: str = "", force: bool = False):
        if label == "interrupt":
            raise OSError("disk full while writing shutdown checkpoint")
        return super()._save(label=label, force=force)


def test_a_failed_shutdown_save_does_not_claim_the_run_is_resumable(data_root):
    store = Store(data_root / "rlai.sqlite3")
    trainer = _FailingInterruptTrainer(_interrupt_config(), "failed-save", store)

    result = trainer.run()

    assert result["status"] == "stopped"
    assert result["interrupted"] is True
    assert result["resumable"] is False
    assert "could NOT be saved" in result["message"]
    assert result["checkpoint"] is None
    assert checkpoints.list_models(data_root / "models") == []
    report = store.get_report("failed-save")["report"]
    assert report["final_checkpoint"]["available"] is False
    assert "no checkpoint" in report["final_checkpoint"]["reason"]
    assert report["resume"]["resumable"] is False


def test_the_interrupt_guard_defers_one_signal_and_ignores_later_ones():
    from rlai.trainer import _InterruptGuard

    guard = _InterruptGuard()
    with pytest.raises(KeyboardInterrupt):
        with guard.deferred():
            guard._handle(None, None)
            guard._handle(None, None)  # a second Ctrl+C must not abort cleanup
    assert guard.count == 2

    # once shutdown has begun, no signal can interrupt the save any more
    guard.begin_shutdown()
    guard._handle(None, None)
    guard._handle(None, None)
    assert guard.count == 4


class _SignalDuringSaveTrainer(Trainer):
    """Simulate Ctrl+C arriving while the final checkpoint is written."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signalled = False

    def _save(self, *, label: str = "", force: bool = False):
        if not self.signalled:
            self.signalled = True
            self._interrupt_guard._handle(None, None)
        return super()._save(label=label, force=force)


def test_ctrl_c_during_a_save_finishes_the_save_then_relabels_it(data_root):
    config = _interrupt_config().model_copy(
        update={"total_steps": 2048, "rollout_steps": 256, "minibatch_size": 256}
    )
    store = Store(data_root / "rlai.sqlite3")
    trainer = _SignalDuringSaveTrainer(config, "mid-save", store)

    result = trainer.run()

    assert result["status"] == "stopped"
    assert result["resumable"] is True
    models = checkpoints.list_models(data_root / "models")
    assert len(models) == 1
    assert models[0]["label"] == "interrupt"
    assert models[0]["step"] == 2048
    assert checkpoints.latest_model(data_root / "models")["id"] == models[0]["id"]


def test_routine_training_writes_no_periodic_checkpoints(data_root):
    """Only a finished run publishes a checkpoint, not every interval."""
    config = _config().model_copy(
        update={"total_steps": 2048, "checkpoint_interval": 1}
    )
    store = Store(data_root / "rlai.sqlite3")
    trainer = Trainer(config, "one-checkpoint", store)

    result = trainer.run()

    assert result["status"] == "completed"
    models = checkpoints.list_models(data_root / "models")
    assert len(models) == 1, [m["label"] for m in models]
    assert models[0]["label"] == "final"
    assert models[0]["step"] == config.total_steps
    assert result["checkpoint"] == models[0]["id"]


def test_a_stop_request_writes_an_interrupt_checkpoint(data_root):
    config = _config().model_copy(update={"total_steps": 100_000})
    store = Store(data_root / "rlai.sqlite3")
    trainer = Trainer(config, "cancelled-run", store)
    trainer.control_path.write_text('{"stop": true}', encoding="utf-8")

    result = trainer.run()

    assert result["status"] == "stopped"
    assert result["interrupted"] is True
    assert result["resumable"] is True
    models = checkpoints.list_models(data_root / "models")
    assert len(models) == 1
    assert models[0]["label"] == "interrupt"
    assert models[0]["step"] == 0
    assert result["checkpoint"] == models[0]["id"]
    report = store.get_report("cancelled-run")["report"]
    assert report["final_checkpoint"]["valid"] is True


def test_the_interrupt_guard_also_catches_sigterm():
    import signal

    from rlai.trainer import _InterruptGuard

    guard = _InterruptGuard()
    guard.begin_shutdown()  # no exception may escape during cleanup
    guard._handle(signal.SIGTERM, None)
    assert "SIGTERM" in guard.describe()

    ctrl_c = _InterruptGuard()
    ctrl_c.begin_shutdown()
    ctrl_c._handle(signal.SIGINT, None)
    assert "Ctrl+C" in ctrl_c.describe()
