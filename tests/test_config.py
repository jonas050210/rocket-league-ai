import math

import pytest
from pydantic import ValidationError

from rocket_league_ai.config import TrainingConfig, changes
from rocket_league_ai.domain import AppError, EvaluationResult, control_target


def test_default_cuda_is_explicit_and_hash_reproducible():
    config = TrainingConfig()
    assert config.device == "cuda:0"
    assert TrainingConfig.model_validate_json(config.model_dump_json()).digest() == config.digest()
    with pytest.raises(ValidationError):
        config.seed = 12


@pytest.mark.parametrize("values, message", [
    ({"n_procc": 2}, "Extra inputs"),
    ({"n_proc": 0}, "greater than or equal"),
    ({"device": "auto"}, "Input should be"),
    ({"team_size": 3}, "Input should be"),
    ({"batch_size": 256}, "must equal"),
    ({"buffer_size": 100}, "greater than or equal"),
    ({"buffer_size": 8192}, "one and three"),
    ({"n_minibatches": 3}, "must divide"),
    ({"total_timesteps": 8193}, "multiple"),
    ({"n_proc": 3}, "divisible"),
    ({"learning_rate": math.nan}, "finite"),
    ({"gamma": math.inf}, "finite"),
    ({"hidden_sizes": []}, "1–4"),
    ({"hidden_sizes": [1024]}, "16 and 512"),
    ({"no_touch_timeout_seconds": 40}, "must not exceed"),
    ({"device": "cpu", "pin_memory": True}, "requires"),
])
def test_actionable_validation(values, message):
    with pytest.raises(ValidationError, match=message):
        TrainingConfig(**values)


def test_diff_is_explicit():
    before = TrainingConfig()
    after = TrainingConfig(device="cpu", seed=999)
    assert changes(before, after) == {
        "device": {"before": "cuda:0", "after": "cpu"},
        "seed": {"before": 123, "after": 999},
    }


@pytest.mark.parametrize("status, command, target", [
    ("running", "pause", "pausing"), ("paused", "resume", "resuming"),
    ("paused", "stop", "stopping"), ("starting", "stop", "stopping"),
    ("pausing", "stop", "stopping"),
])
def test_control_transition(status, command, target):
    assert control_target(status, command) == target


@pytest.mark.parametrize("status, command", [("failed", "resume"), ("paused", "pause"),
                                                ("completed", "stop"), ("running", "resume")])
def test_reject_invalid_controls(status, command):
    with pytest.raises(AppError, match="Cannot"):
        control_target(status, command)


def test_evaluation_is_a_validated_contract_not_an_inferred_metric():
    kwargs = dict(candidate_id="candidate", opponent_id="frozen", wins=4, losses=4, draws=2,
                  games=10, seed=123, side_balanced=True, environment_version="kickoff-v1",
                  goals_for=4, goals_against=4)
    assert EvaluationResult(**kwargs).games == 10
    with pytest.raises(ValidationError, match="must equal"):
        EvaluationResult(**{**kwargs, "wins": 8})


def test_ppo_knobs_and_shaping_validation():
    config = TrainingConfig(clip_range=0.3, max_grad_norm=1.0)
    assert config.clip_range == 0.3 and config.max_grad_norm == 1.0
    with pytest.raises(ValidationError):
        TrainingConfig(reward_velocity_to_ball_weight=0.5)  # shaping requires v2 or v3
    shaped = TrainingConfig(reward_version="goal10-touch0.1-shaped-v2",
                            reward_velocity_to_ball_weight=0.5, reward_ball_to_goal_weight=0.25)
    assert shaped.reward_ball_to_goal_weight == 0.25
    with pytest.raises(ValidationError):
        TrainingConfig(reward_version="goal10-touch0.1-shaped-v2")  # v2 without weights is meaningless
    assert TrainingConfig(environment_version="mixed-scenarios-v1").environment_version == "mixed-scenarios-v1"


def test_pro_v3_high_scale_presets_and_milestones():
    from rocket_league_ai.config import ROADMAP_MILESTONES, TRAINING_PRESETS, align_timesteps

    pro = TrainingConfig(
        n_proc=32,
        batch_size=16384,
        timesteps_per_iteration=16384,
        buffer_size=49152,
        total_timesteps=500_006_912,
        n_minibatches=8,
        reward_version="goal10-touch0.1-pro-v3",
        reward_velocity_to_ball_weight=0.2,
        reward_ball_to_goal_weight=0.35,
        reward_boost_weight=0.15,
        reward_defensive_weight=0.25,
        reward_aerial_shot_weight=0.3,
        self_play_enabled=True,
    )
    assert pro.n_proc == 32
    assert pro.self_play_enabled is True
    assert align_timesteps(1_000_000, 2048) % 2048 == 0
    assert len(TRAINING_PRESETS) == 8
    assert "all_in_one_complete" in TRAINING_PRESETS
    assert "mechanics_speedflip" in TRAINING_PRESETS
    assert "team_2v2_pro" in TRAINING_PRESETS
    assert [m["target_steps"] for m in ROADMAP_MILESTONES] == [
        1_000_000, 10_000_000, 50_000_000, 150_000_000, 500_000_000
    ]

