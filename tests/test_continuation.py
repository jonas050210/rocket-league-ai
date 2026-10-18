"""End-to-end continuation: interrupt a real run, then resume it exactly.

These tests run the real trainer with a real RocketSim worker process, so they
cover the whole chain: checkpoint contents written by ``Trainer._save``, the
automatic resume plan, and the statistics report of an interrupted run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from rlai import checkpoints
from rlai.config import TrainingConfig, ensure_dirs
from rlai.resume import plan_resume
from rlai.store import Store
from rlai.trainer import Trainer

pytestmark = pytest.mark.slow


class StoppingTrainer(Trainer):
    """Trainer that behaves as if the user pressed Stop after one iteration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._control_calls = 0

    def _control(self):
        self._control_calls += 1
        return "stop" if self._control_calls > 1 else None


def _config(**overrides) -> TrainingConfig:
    base = dict(
        run_name="phase-a",
        total_steps=2000,
        rollout_steps=1000,
        n_workers=1,
        minibatch_size=1000,
        episode_seconds=5,
        eval_interval=0,
        checkpoint_interval=1,
        snapshot_interval=1,
        curriculum=True,
        policy={"hidden": [16]},
        device="cpu",
    )
    base.update(overrides)
    return TrainingConfig(**base)


@pytest.fixture(scope="module")
def interrupted(tmp_path_factory) -> dict:
    """One real, interrupted training run shared by the tests in this module."""
    root = tmp_path_factory.mktemp("continuation") / "data"
    previous = os.environ.get("RLAI_DATA_DIR")
    os.environ["RLAI_DATA_DIR"] = str(root)
    try:
        paths = ensure_dirs()
        store = Store(paths["root"] / "rlai.sqlite3")
        config = _config()
        run_id = store.create_run(
            config.run_name, config.mode, config.to_dict(), config.total_steps
        )
        trainer = StoppingTrainer(config, run_id, store)
        result = trainer.run()
        yield {
            "root": root,
            "paths": paths,
            "store": store,
            "config": config,
            "run_id": run_id,
            "result": result,
            "models_dir": paths["models"],
        }
    finally:
        if previous is None:
            os.environ.pop("RLAI_DATA_DIR", None)
        else:
            os.environ["RLAI_DATA_DIR"] = previous


def test_interrupted_run_stops_and_reports(interrupted):
    result = interrupted["result"]
    store = interrupted["store"]
    run_id = interrupted["run_id"]
    assert result["status"] == "stopped"
    assert result["step"] == 1000
    assert result["steps_completed"] == 1000
    assert store.get_run(run_id)["status"] == "stopped"

    report = store.get_report(run_id)["report"]
    assert report["status"] == "stopped"
    assert report["steps"]["start"] == 0
    assert report["steps"]["completed"] == 1000
    assert report["duration_s"] is not None and report["duration_s"] > 0
    assert report["steps_per_second"] == pytest.approx(
        1000 / report["duration_s"], rel=1e-6
    )
    assert report["final_checkpoint"]["valid"] is True
    assert Path(report["final_checkpoint"]["path"]).exists()
    # the optimisation really ran: losses and timings were measured
    assert report["optimisation"]["losses/policy"]["available"] is True
    assert report["optimisation"]["losses/approx_kl"]["count"] == 1
    assert report["timings"]["time/collect_seconds"]["available"] is True
    assert report["timings"]["time/sim_seconds"]["available"] is True
    assert report["timings"]["time/infer_seconds"]["available"] is True
    assert report["timings"]["time/checkpoint_seconds"]["available"] is True
    # episodes and their reward components were persisted
    episodes = report["episodes"]["this_run"]
    assert episodes["available"] is True and episodes["count"] > 0
    assert isinstance(report["reward_components"], dict)
    assert "reward/goal" in report["reward_components"]
    assert report["hardware"]["workers"] == 1
    assert report["hardware"]["device"] == "cpu"
    assert report["curves"]["charts/episodic_return"]
    # no evaluation was configured, and the report says so instead of faking one
    assert report["evaluation"]["before"]["available"] is False
    assert report["evaluation"]["after"]["available"] is False


def test_resume_plan_finds_the_interrupted_checkpoint(interrupted):
    config = _config(run_name="phase-b")
    plan = plan_resume(
        interrupted["models_dir"], config, mode="continue", run_id="run-b"
    )
    assert plan.will_resume is True
    assert plan.restored_step == 1000
    assert plan.parent_run == interrupted["run_id"]
    assert plan.validation["has_optimizer"] is True
    assert plan.validation["has_rng"] is True
    assert plan.validation["has_curriculum"] is True
    assert plan.config["curriculum"] is True


def test_resume_restores_weights_optimizer_step_rng_and_curriculum(interrupted):
    store = interrupted["store"]
    config = _config(run_name="phase-b")
    plan = plan_resume(
        interrupted["models_dir"], config, mode="continue", run_id="run-restore"
    )
    run_id = store.create_run(
        config.run_name, config.mode, config.to_dict(), config.total_steps, run_id="run-restore"
    )
    trainer = Trainer(config, run_id, store, resume_plan=plan)
    trainer._load_resume()

    payload = checkpoints.load_checkpoint(plan.checkpoint_path)
    for name, tensor in payload["model"].items():
        assert torch.equal(trainer.model.state_dict()[name].cpu(), tensor), name
    assert trainer.optimizer.state, "optimizer state must be restored"
    assert trainer.step == 1000
    assert trainer.iteration == payload["extra"]["iteration"] == 1
    assert trainer.rng.bit_generator.state == payload["extra"]["rng"]["numpy"]
    assert trainer.curriculum is not None
    assert trainer.curriculum.total_iterations == payload["extra"]["curriculum"]["total_iterations"]
    assert trainer.league.agent_elo == pytest.approx(payload["extra"]["league"]["agent_elo"])
    row = store.get_run(run_id)
    assert row["parent_run"] == interrupted["run_id"]
    assert row["resumed_step"] == 1000


def test_snapshots_do_not_change_while_the_learner_trains(interrupted):
    """The self-play pool must hold frozen weights, not aliases."""
    config = _config(run_name="snapshot-check", curriculum=False, self_play=True)
    store = interrupted["store"]
    run_id = store.create_run(
        config.run_name, config.mode, config.to_dict(), config.total_steps, run_id="snap-check"
    )
    trainer = Trainer(config, run_id, store)
    trainer.league.add("snap", trainer.step, trainer._snapshot_state())
    key = next(iter(trainer.league.entries[0].state))
    frozen = trainer.league.entries[0].state[key].clone()
    assert trainer.league.entries[0].state[key].data_ptr() != trainer.model.state_dict()[
        key
    ].data_ptr(), "a snapshot must not alias the live parameters"

    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.5)
    observation_size = trainer.config.policy_config().obs_size
    for _ in range(3):
        logits, values = trainer.model(torch.randn(4, observation_size))
        loss = logits.pow(2).mean() + values.pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert torch.equal(trainer.league.entries[0].state[key], frozen)


def test_continuation_run_finishes_and_documents_the_lineage(interrupted):
    store = interrupted["store"]
    config = _config(run_name="phase-b", total_steps=2000)
    run_id = store.create_run(
        config.run_name, config.mode, config.to_dict(), config.total_steps, run_id="run-phase-b"
    )
    plan = plan_resume(
        interrupted["models_dir"], config, mode="continue", run_id=run_id
    )
    trainer = Trainer(config, run_id, store, resume_plan=plan)
    result = trainer.run()
    assert result["status"] == "completed"
    assert result["start_step"] == 1000
    assert result["step"] == 2000
    assert result["steps_completed"] == 1000

    report = store.get_report(run_id)["report"]
    assert report["steps"]["start"] == 1000
    assert report["steps"]["end"] == 2000
    assert report["steps"]["completed"] == 1000
    assert report["resume"]["mode"] == "continue"
    assert report["resume"]["parent_run"] == interrupted["run_id"]
    assert report["resume"]["restored_step"] == 1000
    assert report["resume"]["model_id"] == plan.model_id
    assert report["final_checkpoint"]["valid"] is True
    assert report["final_checkpoint"]["step"] == 2000
    # the continued run is listed as comparable history for its parent
    assert any(item["run_id"] == interrupted["run_id"] for item in report["comparisons"])
