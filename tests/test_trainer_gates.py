"""Trainer + mastery curriculum: gate evaluations, blocking, checkpoint safety.

These tests run the *real* trainer on a tiny budget so the whole path is
exercised: worker rollout -> PPO update -> independent gate evaluation ->
curriculum decision -> report.
"""

from __future__ import annotations

import json

import pytest

from rlai import checkpoints
from rlai.config import TrainingConfig
from rlai.curriculum import MASTERY_STAGES
from rlai.store import Store
from rlai.trainer import Trainer


def _config(**overrides) -> TrainingConfig:
    base = dict(
        run_name="gate-unit",
        total_steps=2048,
        rollout_steps=128,
        n_workers=1,
        minibatch_size=128,
        epochs=1,
        eval_interval=0,
        eval_before=False,
        policy={"hidden": [16]},
        device="cpu",
        curriculum=True,
        curriculum_gate_interval=1,
        curriculum_gate_trials=4,
        player_memory=False,
    )
    base.update(overrides)
    return TrainingConfig(**base)


def _logs(store: Store, run_id: str) -> str:
    return "\n".join(row["message"] for row in store.tail_logs(run_id, limit=2000))


def test_training_runs_the_independent_gate_evaluation(data_root):
    store = Store(data_root / "rlai.sqlite3")
    trainer = Trainer(_config(), "gate-run", store)
    result = trainer.run()

    assert result["status"] == "completed"
    logs = _logs(store, "gate-run")
    assert "gate[approach]" in logs
    # the gate is measured, it just is not passed by a barely trained policy
    assert "confirmations" in logs

    metrics = {
        row["name"]: row["value"]
        for row in store.get_metrics("gate-run", ["curriculum/gate_evaluations"])
    }
    assert metrics["curriculum/gate_evaluations"] >= 1.0

    report = json.loads((data_root / "runs" / "gate-run" / "report.json").read_text())
    curriculum = report["curriculum"]
    assert curriculum["stage"]["name"] in {stage.name for stage in MASTERY_STAGES}
    assert curriculum["diagnostics"]["confirmations_required"] >= 2
    # every measured gate row carries a sample size and an interval, or says
    # why it has none (the confirmation counter is a counter, not a probe)
    measured = [
        row for row in curriculum["criteria"] if row.get("kind") == "gate" and "skill" in row
    ]
    assert measured, curriculum["criteria"]
    for row in measured:
        assert row["observed"] is None or row.get("ci95")


def test_an_untrained_policy_does_not_leave_the_first_stage(data_root):
    """The gate must actually block: no silent promotion on elapsed time."""
    store = Store(data_root / "rlai.sqlite3")
    trainer = Trainer(_config(), "blocked-run", store)
    result = trainer.run()
    assert result["status"] == "completed"
    assert trainer.curriculum.index == 0
    assert trainer.curriculum.stage.name == "approach"
    reason = trainer.curriculum.blocked_reason()
    assert reason and "approach/rate" in reason
    assert "gate[approach]" in _logs(store, "blocked-run")


def test_gate_evaluation_cost_is_bounded(data_root):
    """A gate evaluation must stay cheap enough to run inside training."""
    store = Store(data_root / "rlai.sqlite3")
    trainer = Trainer(_config(), "cost-run", store)
    trainer.run()
    # 5 probes x 4 trials of at most 10 s game time on one core
    assert trainer.gate_seconds < 120.0
    assert trainer.gate_probe_seconds > 0.0


def test_restored_step_beyond_total_steps_is_reported_not_silent(data_root):
    """A target below the checkpoint step must not look like a normal run."""
    config = _config(total_steps=2048)
    policy = config.policy_config()
    model = policy.build()
    checkpoint = checkpoints.save_checkpoint(
        data_root / "models",
        model_state=model.state_dict(),
        policy_config=policy.to_dict(),
        step=4096,  # beyond the configured target
        mode=config.mode,
        label="old",
    )
    store = Store(data_root / "rlai.sqlite3")
    trainer = Trainer(
        config,
        "zero-steps",
        store,
        resume_from=str(data_root / "models" / checkpoint["path"]),
    )
    result = trainer.run()

    assert result["status"] == "completed"
    assert result["steps_completed"] == 0
    assert result["step"] == 4096
    assert any("already reached total_steps" in warning for warning in result["warnings"])
    assert "already reached total_steps" in _logs(store, "zero-steps")
    # and no second checkpoint was written for a run that trained nothing
    manifests = [
        path
        for path in (data_root / "models").iterdir()
        if path.is_dir() and path.name != checkpoint["id"]
    ]
    assert manifests == []


def test_gate_state_is_restored_from_the_checkpoint(data_root):
    config = _config(total_steps=2048, rollout_steps=128)
    store = Store(data_root / "rlai.sqlite3")
    trainer = Trainer(config, "persist-run", store)
    trainer.curriculum.record_gate(
        {"approach": {"rate": 1.0, "ci95": [0.9, 1.0], "trials": 24}}
    )
    assert trainer.curriculum.gates.passes_in_a_row == 1
    manifest = trainer._save(label="interrupt", force=True)
    assert manifest is not None

    payload = checkpoints.load_checkpoint(data_root / "models" / manifest["path"])
    saved = payload["extra"]["curriculum"]
    assert saved["gate_state"]["passes_in_a_row"] == 1
    assert saved["gate_state"]["last"]["approach"]["rate"] == 1.0

    restored = Trainer(
        config,
        "persist-run-2",
        Store(data_root / "rlai.sqlite3"),
        resume_from=str(data_root / "models" / manifest["path"]),
    )
    restored._load_resume()
    assert restored.curriculum.gates.passes_in_a_row == 1
    assert restored.curriculum.gates.last["approach"]["rate"] == 1.0


def test_reaching_the_configured_target_still_writes_exactly_one_checkpoint(data_root):
    """Gate evaluations must not reintroduce routine checkpoints."""
    store = Store(data_root / "rlai.sqlite3")
    trainer = Trainer(_config(), "final-run", store)
    result = trainer.run()
    assert result["status"] == "completed"
    written = [
        path.name
        for path in (data_root / "models").iterdir()
        if path.is_dir() and (path / "manifest.json").exists()
    ]
    assert len(written) == 1
    manifest = json.loads(
        (data_root / "models" / written[0] / "manifest.json").read_text()
    )
    assert manifest["label"] == "final"
    assert manifest["step"] == trainer.config.total_steps


def test_legacy_preset_keeps_the_old_window_behaviour(data_root):
    config = _config(curriculum_preset="legacy", curriculum_gate_interval=0)
    store = Store(data_root / "rlai.sqlite3")
    trainer = Trainer(config, "legacy-run", store)
    assert trainer.curriculum.stage.name == "first-touch"
    result = trainer.run()
    assert result["status"] == "completed"
    assert "gate[" not in _logs(store, "legacy-run")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
