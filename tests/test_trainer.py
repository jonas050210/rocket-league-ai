"""Trainer lifecycle: run registration, resume state and best-model rule."""

from __future__ import annotations

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
