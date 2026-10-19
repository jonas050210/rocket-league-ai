"""Evaluation payload: honest confidence intervals and checkpoint compat."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rlai import checkpoints
from rlai.evaluation import evaluate_model, evaluate_policy, policy_obs_version, wilson_interval
from rlai.policy import PolicyConfig, TorchPolicy


def test_wilson_interval_brackets_the_estimate():
    low, high = wilson_interval(0, 0)
    assert (low, high) == (0.0, 1.0)
    low, high = wilson_interval(0, 20)
    assert low == 0.0 and 0.0 < high < 0.2
    low, high = wilson_interval(10, 20)
    assert low < 0.5 < high
    low, high = wilson_interval(20, 20)
    assert high == pytest.approx(1.0) and low > 0.8
    # more games narrow the interval around the same rate
    narrow = wilson_interval(50, 100)
    wide = wilson_interval(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_policy_obs_version_from_checkpoint(models_dir):
    policy = TorchPolicy.from_config(PolicyConfig())
    assert policy_obs_version(policy) == 2
    legacy = TorchPolicy.from_config(PolicyConfig(obs_size=190))
    assert policy_obs_version(legacy) == 1
    with pytest.raises(ValueError):
        policy_obs_version(TorchPolicy.from_config(PolicyConfig(obs_size=191)))


def test_evaluate_policy_reports_ci_and_version(model_path):
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    policy = TorchPolicy.from_config(
        PolicyConfig.from_dict(payload.get("policy_config"))
    )
    policy.model.load_state_dict(payload["model"])
    result = evaluate_policy(
        policy,
        mode="soccer-1v1",
        opponent="idle",
        games=1,
        match_seconds=1.0,
        episode_seconds=5.0,
    )
    assert result["games"] == 1
    assert result["wins"] + result["losses"] + result["draws"] == 1
    assert len(result["win_rate_ci95"]) == 2
    low, high = result["win_rate_ci95"]
    assert 0.0 <= low <= high <= 1.0
    assert result["obs_version"] == 2
    assert result["details"]["games"][0]["ticks"] > 0


def test_evaluate_model_uses_the_checkpoint_layout(data_root, model_path):
    model_id = checkpoints.list_models(model_path.parent.parent)[0]["id"]
    result = evaluate_model(
        model_path,
        mode="soccer-1v1",
        opponent="idle",
        games=1,
        match_seconds=1.0,
        episode_seconds=5.0,
    )
    assert result["model"] == str(model_path)
    assert result["obs_version"] == 2
    assert model_id in str(model_path)
    assert result["win_rate_ci95"][1] >= result["win_rate"]


def test_legacy_190_observation_policy_evaluates(models_dir):
    config = PolicyConfig(obs_size=190, hidden=(8,))
    policy = TorchPolicy.from_config(config)
    result = evaluate_policy(
        policy,
        mode="soccer-1v1",
        opponent="random",
        games=1,
        match_seconds=0.5,
        episode_seconds=5.0,
    )
    assert result["obs_version"] == 1
    assert result["stochastic"] is False
    assert np.isfinite(result["win_rate"])

    sampled = evaluate_policy(
        policy,
        mode="soccer-1v1",
        opponent="idle",
        games=1,
        match_seconds=0.5,
        episode_seconds=5.0,
        stochastic=True,
    )
    assert sampled["stochastic"] is True
    assert sampled["games"] == 1


def test_touch_count_is_a_per_step_delta():
    """``ball_touches`` counts contacts since the last state was sent.

    The engine zeroes it after every state read, so a contact on two
    consecutive decision steps is two contacts - not one, which is what the
    old "only add the difference when the counter grew" logic reported.
    """
    from types import SimpleNamespace

    from rlai.evaluation import count_touches

    class Counter:
        def __init__(self, value):
            self.ball_touches = value

    state = SimpleNamespace(cars={"blue-0": Counter(1), "orange-0": Counter(0)})
    env = SimpleNamespace(state=state)
    agents = ["blue-0"]

    assert count_touches(env, agents) == 1
    state.cars["blue-0"].ball_touches = 1  # another contact in the next step
    assert count_touches(env, agents) == 1
    state.cars["blue-0"].ball_touches = 2  # two callbacks in one step
    assert count_touches(env, agents) == 2
    state.cars["blue-0"].ball_touches = 0  # engine reset, no contact
    assert count_touches(env, agents) == 0
