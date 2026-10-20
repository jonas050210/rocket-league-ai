"""``--steps N`` means N steps *on top of* the restored checkpoint step.

The historical ``--max-steps`` is an absolute target; continuing a 2M-step
checkpoint with ``--max-steps 500000`` therefore trained zero steps and the run
was reported as completed.  These tests pin the incremental semantics, the
resolving helper, the learning-rate window it starts, and the small safety
features that go with it (retention of the run's first episodes, recovery
checkpoints, no-touch truncation).
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlai.cli import build_parser
from rlai.config import TrainingConfig
from rlai.ppo import learning_rate_at
from rlai.resume import plan_resume, with_additional_steps
from rlai.store import Store

ROOT = Path(__file__).resolve().parents[1]


def _config(**overrides) -> TrainingConfig:
    data = {
        "total_steps": 200_000,
        "rollout_steps": 256,
        "minibatch_size": 256,
        "n_workers": 1,
        "obs_version": 2,
        "policy": {"hidden": [32, 32]},
        "opponent": "idle",
        "eval_interval": 0,
    }
    data.update(overrides)
    return TrainingConfig(**data)


@pytest.fixture()
def resume_models(tmp_path):
    """A real checkpoint at step 150,000 written through the production path."""
    import torch

    from rlai import checkpoints
    from rlai.policy import PolicyConfig

    models = tmp_path / "models"
    models.mkdir()
    config = _config()
    policy = PolicyConfig(
        obs_size=config.observation_size,
        hidden=tuple(config.hidden_tuple),
    )
    model = policy.build()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.01)
    checkpoints.save_checkpoint(
        models,
        model_state=model.state_dict(),
        policy_config=policy.to_dict(),
        step=150_000,
        mode="soccer-1v1",
        training_config=config.to_dict(),
        label="test",
    )
    return models, config


# -- the resolving helper --------------------------------------------------


def test_additional_steps_make_an_absolute_target():
    config = _config()
    resolved = with_additional_steps(config, restored_step=2_000_000, additional_steps=500_000)
    # 2,500,000 is not a multiple of the 256-step rollout batch: the run can
    # only stop at a batch boundary and rounds up by at most one batch
    assert resolved.total_steps == 2_500_096
    # the anneal covers the added chunk, not a 50M-step history
    assert resolved.lr_anneal_start == 2_000_000
    # the original model object is not mutated
    assert config.total_steps == 200_000
    assert config.lr_anneal_start == 0


def test_additional_steps_validate_the_request():
    config = _config()
    with pytest.raises(ValueError):
        with_additional_steps(config, 0, 0)
    with pytest.raises(ValueError):
        with_additional_steps(config, 0, -5)
    # fewer steps than one rollout batch can never run an iteration
    with pytest.raises(ValueError):
        with_additional_steps(config, 0, 10)


def test_additional_steps_round_up_to_whole_rollout_batches():
    config = _config()
    resolved = with_additional_steps(config, 1_000, 300)  # batch = 256 * 1
    assert resolved.total_steps == 1_536
    assert resolved.total_steps % 256 == 0
    # already aligned: unchanged
    assert with_additional_steps(config, 0, 512).total_steps == 512


def test_learning_rate_anneals_over_the_added_window():
    config = _config(
        total_steps=10_000,
        lr_anneal_start=8_000,
        learning_rate=3e-4,
        lr_anneal=True,
    )
    assert learning_rate_at(config, 8_000) == pytest.approx(3e-4)
    assert learning_rate_at(config, 9_000) == pytest.approx(3e-4 * 0.55)
    assert learning_rate_at(config, 10_000) == pytest.approx(3e-4 * 0.1)


def test_learning_rate_defaults_to_the_whole_run():
    config = _config(total_steps=10_000, learning_rate=3e-4, lr_anneal=True)
    assert learning_rate_at(config, 5_000) == pytest.approx(3e-4 * 0.55)


def test_config_rejects_an_empty_anneal_window():
    with pytest.raises(ValueError):
        _config(total_steps=10_000, lr_anneal_start=10_000)


# -- the resume plan -------------------------------------------------------


def test_plan_reports_the_target_and_does_not_complain(resume_models):
    models, config = resume_models
    plan = plan_resume(models, config, mode="continue", additional_steps=500_000)
    assert plan.will_resume
    assert plan.restored_step == 150_000
    assert plan.additional_steps == 500_000
    # 650,000 rounds up to the next 256-step batch boundary
    assert plan.target_total_steps == 650_240
    assert plan.effective_config["total_steps"] == 650_240
    assert plan.effective_config["lr_anneal_start"] == 150_000
    assert any("adding 500000 steps" in note for note in plan.notes)
    assert any("rounded up from 650000" in note for note in plan.notes)
    assert not any("already reached total_steps" in warning for warning in plan.warnings)
    assert "adding 500000 steps" in plan.summary()


def test_plan_without_additional_steps_keeps_the_absolute_semantics(resume_models):
    models, _ = resume_models
    # the requested absolute target is below the checkpoint: the honest warning
    plan = plan_resume(models, _config(total_steps=100_000), mode="continue")
    assert plan.restored_step == 150_000
    assert any("already reached total_steps" in warning for warning in plan.warnings)
    # ... and the same request with --steps resolves to a reachable target
    plan = plan_resume(
        models, _config(total_steps=100_000), mode="continue", additional_steps=500_000
    )
    assert not any("already reached total_steps" in warning for warning in plan.warnings)


def test_plan_without_any_checkpoint_still_reports_the_target(tmp_path):
    empty = tmp_path / "models"
    empty.mkdir()
    plan = plan_resume(empty, _config(), mode="continue", additional_steps=50_000)
    assert not plan.will_resume
    assert plan.additional_steps == 50_000
    assert plan.target_total_steps % 256 == 0
    assert 50_000 <= plan.target_total_steps < 50_256


def test_plan_rejects_a_non_positive_step_request(tmp_path):
    with pytest.raises(ValueError):
        plan_resume(tmp_path, _config(), mode="continue", additional_steps=0)
    with pytest.raises(ValueError):
        plan_resume(tmp_path, _config(), mode="continue", additional_steps="many")


# -- CLI wiring ------------------------------------------------------------


def test_cli_accepts_the_step_aliases():
    parser = build_parser()
    for flag in ("--steps", "--maxsteps", "--Maxsteps"):
        args = parser.parse_args(["train", "--config-file", "config.json", flag, "500000"])
        assert args.steps == 500_000
        args = parser.parse_args(["headless", "--config-file", "config.json", flag, "7"])
        assert args.steps == 7
        args = parser.parse_args(["resume-plan", "--config-file", "config.json", flag, "9"])
        assert args.steps == 9
    absolute = parser.parse_args(["train", "--config-file", "config.json", "--max-steps", "5"])
    assert absolute.max_steps == 5 and absolute.steps is None
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["train", "--config-file", "config.json", "--steps", "5", "--max-steps", "5"]
        )


def test_main_py_wrapper_defaults_to_train_and_the_example_config():
    spec = importlib.util.spec_from_file_location("rlai_main_entry", ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    argv = module._prepare_argv(["--Maxsteps", "500000"])
    assert argv[0] == "train"
    assert "--Maxsteps" in argv
    assert "--config-file" in argv
    assert argv[argv.index("--config-file") + 1].endswith("configs/1v1-mastery-1v1.json")

    # an explicit sub-command is not rewritten, and an explicit config wins
    argv = module._prepare_argv(["train", "--config-file", "other.json", "--steps", "10"])
    assert argv == ["train", "--config-file", "other.json", "--steps", "10"]

    argv = module._prepare_argv(["serve", "--port", "8000"])
    assert argv == ["serve", "--port", "8000"]

    # --help must not claim a configuration file it will never read
    argv = module._prepare_argv(["--help"])
    assert argv == ["train", "--help"]


def test_main_py_targets_exist():
    spec = importlib.util.spec_from_file_location("rlai_main_entry", ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.DEFAULT_CONFIG.is_file()
    assert module.DEFAULT_CONFIG.name == "1v1-mastery-1v1.json"


# -- retention and recovery checkpoints ------------------------------------


def test_prune_episodes_can_protect_the_beginning_of_a_run(tmp_path):
    store = Store(tmp_path / "rlai.sqlite3")
    run_id = store.create_run("t", "soccer-1v1", {}, 1000)
    for index in range(12):
        store.log_episode(
            run_id,
            episode=index,
            step=index,
            reward=float(index),
            length=10,
            goals_for=0,
            goals_against=0,
            result="draw",
            truncated=True,
        )
    removed = store.prune_episodes(run_id, keep=4, protect_first=3)
    assert removed == 5
    with sqlite3.connect(store.path) as conn:
        kept = sorted(row[0] for row in conn.execute("SELECT reward FROM episodes"))
    # the first three and the newest four rows survived
    assert kept == [0.0, 1.0, 2.0, 8.0, 9.0, 10.0, 11.0]
    first = store.episode_window_stats(run_id, which="first", window=3)
    last = store.episode_window_stats(run_id, which="last", window=4)
    assert (first["count"], first["mean_reward"]) == (3, 1.0)
    assert (last["count"], last["mean_reward"]) == (4, 9.5)


def test_recovery_checkpoints_are_opt_in_and_time_based():
    from rlai.trainer import Trainer

    trainer = object.__new__(Trainer)
    calls: list[str] = []
    trainer._save = lambda *, label="", force=False: (  # type: ignore[method-assign]
        calls.append(label) or {"id": "recovery-1"}
    )
    trainer._log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    trainer.step = 1234

    trainer.config = SimpleNamespace(recovery_checkpoint_minutes=0)
    assert trainer._maybe_recovery_checkpoint() is None
    assert calls == []

    trainer.config = SimpleNamespace(recovery_checkpoint_minutes=30)
    trainer._last_recovery_at = time.monotonic()
    assert trainer._maybe_recovery_checkpoint() is None
    assert calls == []

    # relative to the monotonic clock, not to 0: a short-lived container may
    # not have been up for the configured interval yet
    trainer._last_recovery_at = time.monotonic() - 31 * 60.0
    assert trainer._maybe_recovery_checkpoint() == {"id": "recovery-1"}
    assert calls == ["recovery"]


def test_recommended_configs_use_the_new_learning_settings():
    mastery = json.loads((ROOT / "configs" / "1v1-mastery-1v1.json").read_text())
    # more steps per update: the measured gradient cosine between two batches
    # was 0.14 at 3k samples, i.e. close to noise
    assert mastery["rollout_steps"] >= 1024
    assert mastery["no_touch_timeout_seconds"] > 0
    assert mastery["recovery_checkpoint_minutes"] > 0
    TrainingConfig.model_validate(mastery)


def test_no_touch_timeout_is_wired_into_the_env():
    from rlgym.rocket_league.done_conditions import TimeoutCondition

    from rlai.envs import MatchEnv
    from rlai.modes import get_mode
    from rlai.opponents import build_opponent

    spec = get_mode("soccer-1v1")
    off = MatchEnv(spec, build_opponent("idle"), seed=1, episode_seconds=10)
    assert type(off.env.truncation_cond) is TimeoutCondition

    limited = MatchEnv(
        spec,
        build_opponent("idle"),
        seed=1,
        episode_seconds=10,
        no_touch_timeout_seconds=7,
    )
    condition = limited.env.truncation_cond
    # a truncation, not a termination: GAE bootstraps from the final state
    assert type(condition).__name__ == "AnyCondition"
    assert [type(part).__name__ for part in condition.conditions] == [
        "NoTouchTimeoutCondition",
        "TimeoutCondition",
    ]


def test_new_safety_settings_default_to_off_and_time_based():
    config = TrainingConfig()
    # historical behaviour unless a config opts in
    assert config.no_touch_timeout_seconds == 0.0
    assert config.lr_anneal_start == 0
    # crash protection is on by default, but only every half hour
    assert config.recovery_checkpoint_minutes == 30.0


def test_collect_forwards_the_no_touch_timeout_to_the_worker():
    """The stage timeout travels with the collect command, like reset_mode."""
    from rlai.workers import WorkerPool

    class FakeParent:
        def __init__(self):
            self.messages = []
            self.reply = {"type": "rollout", "worker": 0, "steps": 10}

        def send(self, message):
            self.messages.append(message)

        def poll(self, _timeout):
            return True

        def recv(self):
            return self.reply

    class FakeProcess:
        exitcode = None

    pool = WorkerPool.__new__(WorkerPool)
    pool.parents = [FakeParent()]
    pool.workers = [FakeProcess()]
    pool.collect_timeout = 1.0
    pool.results = pool.collect(
        {}, 10, 7, reset_mode="drill", episode_seconds=30.0, no_touch_timeout=25.0
    )
    message = pool.parents[0].messages[0]
    assert message["cmd"] == "collect"
    assert message["no_touch_timeout"] == 25.0
    assert message["reset_mode"] == "drill" and message["episode_seconds"] == 30.0
    assert pool.results[0]["type"] == "rollout"


def test_mastery_shooting_stage_uses_attacking_starts_and_a_longer_timeout():
    """The measured bottleneck: kickoff starts plus a 10 s timeout starved it."""
    from rlai.curriculum import MASTERY_STAGES

    stages = {stage.name: stage for stage in MASTERY_STAGES}
    shooting = stages["shooting"]
    # ``drill`` keeps every start in an attack/defense/race situation; kickoff
    # starts (``mixed``) put the ball 2500+ uu away, which is a reach problem.
    assert shooting.reset_mode == "drill"
    assert shooting.no_touch_timeout == 25.0
    # the early stages keep the run-wide timeout: their starts are close range
    assert stages["approach"].no_touch_timeout is None
    assert stages["control"].no_touch_timeout is None
