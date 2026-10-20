"""Behaviour cloning: collect expert actions, imitate them, save a checkpoint."""

from __future__ import annotations

import numpy as np
import pytest

from rlai import checkpoints
from rlai.config import TrainingConfig
from rlai.imitation import (
    DemonstrationSet,
    collect_demonstrations,
    run_pretraining,
    train_imitation,
)
from rlai.policy import PolicyConfig
from rlai.resume import plan_resume


def _config(**overrides) -> TrainingConfig:
    data = {
        "run_name": "imitation unit",
        "total_steps": 4000,
        "rollout_steps": 256,
        "n_workers": 1,
        "minibatch_size": 256,
        "episode_seconds": 5.0,
        "policy": {"hidden": [32, 32]},
        "device": "cpu",
        "opponent": "idle",
        "eval_interval": 0,
        "no_touch_timeout_seconds": 4.0,
    }
    data.update(overrides)
    return TrainingConfig(**data)


def test_train_imitation_learns_a_simple_rule():
    rng = np.random.default_rng(0)
    n = 400
    observations = rng.normal(size=(n, 12)).astype(np.float32)
    # the expert does action 3 when the first feature is positive, else 7
    actions = np.where(observations[:, 0] > 0, 3, 7).astype(np.int64)
    data = DemonstrationSet(observations=observations, actions=actions, episodes=10)

    state, metrics = train_imitation(
        data,
        PolicyConfig(obs_size=12, n_actions=90, hidden=(32, 32)),
        epochs=8,
        batch_size=64,
        learning_rate=3e-3,
        device="cpu",
    )
    assert metrics["samples"] == n
    assert metrics["final_loss"] < metrics["history"][0]["loss"]
    # far above the 1/90 a uniform policy would reach
    assert metrics["final_accuracy"] > 0.9
    assert "actor.0.weight" in state or any(key.startswith("actor") for key in state)


def test_collect_demonstrations_from_the_chaser():
    config = _config()
    data = collect_demonstrations(config, episodes=2, expert="chase", max_decisions=80)
    assert data.samples > 0
    assert data.episodes == 2
    assert data.observations.shape[1] == config.observation_size
    assert data.observations.dtype == np.float32
    assert data.actions.min() >= 0 and data.actions.max() < 90
    # the chaser reaches the ball; that is the whole point of cloning it
    assert data.details["expert"] == "chase"
    assert data.touches_per_episode >= 0.0


def test_pretrain_writes_a_resumable_checkpoint(data_root):
    config = _config()
    summary = run_pretraining(
        config,
        episodes=1,
        epochs=1,
        expert="chase",
        max_decisions=60,
        models_dir=data_root / "models",
    )
    assert summary["step"] == 0
    assert summary["demonstrations"]["samples"] > 0
    assert summary["training"]["final_accuracy"] >= 0.0

    manifest = checkpoints.latest_record(data_root / "models")
    assert manifest["id"] == summary["checkpoint"]
    assert manifest["label"] == "pretrain"

    validation = checkpoints.validate_checkpoint(
        summary["path"], policy_config=config.policy_config().to_dict(), load=False
    )
    assert validation.ok, validation.reasons

    # a normal resume treats it like any other checkpoint
    plan = plan_resume(data_root / "models", config, mode="continue")
    assert plan.will_resume
    assert plan.restored_step == 0
    assert plan.model_id == summary["checkpoint"]


def test_pretrain_rejects_an_empty_request(data_root):
    with pytest.raises(ValueError):
        collect_demonstrations(_config(), episodes=0)
    data = DemonstrationSet(
        observations=np.zeros((2, 12), dtype=np.float32),
        actions=np.zeros(2, dtype=np.int64),
    )
    with pytest.raises(ValueError):
        train_imitation(data, PolicyConfig(obs_size=12, n_actions=90), epochs=0)


def test_cli_and_main_expose_pretrain():
    import importlib.util
    from pathlib import Path as _Path

    from rlai.cli import build_parser

    args = build_parser().parse_args(
        ["pretrain", "--config-file", "c.json", "--episodes", "5", "--epochs", "2"]
    )
    assert args.episodes == 5 and args.epochs == 2 and args.expert == "chase"

    root = _Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("rlai_main_entry_bc", root / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = module._prepare_argv(["pretrain", "--episodes", "5"])
    assert argv[0] == "pretrain"
    assert "--config-file" in argv
