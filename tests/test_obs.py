"""Observation layout: v1 compatibility, v2 field frame, deterministic ordering."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from rlai import obs as obs_mod
from rlai.envs import MatchEnv
from rlai.modes import get_mode
from rlai.obs import OBS_SIZE, OBS_SIZE_BY_VERSION, build_observation, obs_size
from rlai.opponents import IdleOpponent


def _env(mode="soccer-1v1", obs_version=None):
    kwargs = {} if obs_version is None else {"obs_version": obs_version}
    env = MatchEnv(get_mode(mode), IdleOpponent(), seed=3, episode_seconds=10, **kwargs)
    env.reset()
    return env


def test_declared_sizes_are_consistent():
    assert OBS_SIZE_BY_VERSION == {1: 190, 2: 198, 3: 204, 4: 222}
    assert obs_size(2) == OBS_SIZE == 198
    assert obs_size() == obs_size(obs_mod.OBS_VERSION)
    with pytest.raises(ValueError):
        obs_size(99)


def test_training_config_exposes_matching_sizes():
    from rlai.config import TrainingConfig

    assert TrainingConfig().policy_config().obs_size == 198
    assert TrainingConfig(obs_version=1).policy_config().obs_size == 190
    assert TrainingConfig().observation_size == 198


def test_version_is_inferred_from_size():
    assert obs_mod.version_for_obs_size(190) == 1
    assert obs_mod.version_for_obs_size(198) == 2
    with pytest.raises(ValueError):
        obs_mod.version_for_obs_size(191)


@pytest.mark.parametrize("mode", ["soccer-1v1", "soccer-2v2", "soccer-3v3", "heatseeker-2v2"])
def test_every_mode_produces_the_declared_size(mode):
    env = _env(mode)
    observations = env.reset()
    for value in observations.values():
        assert value.shape == (OBS_SIZE,)
        assert np.isfinite(value).all()


def test_v1_observation_has_no_field_block():
    env = _env(obs_version=1)
    result = env.reset()
    for value in result.values():
        assert value.shape == (190,)
        assert not value[obs_mod.FIELD_START : obs_mod.FIELD_START + 8].any()


def test_v2_field_block_tracks_ball_and_goals():
    env = _env()
    result = env.reset()
    state = env.env.state
    for agent, values in result.items():
        car = state.cars[agent]
        team = int(car.team_num)
        block = values[obs_mod.FIELD_START : obs_mod.FIELD_START + 8]
        assert block[0] == pytest.approx(float(car.physics.position[0]) / obs_mod.FIELD_X, abs=1e-5)
        assert block[1] == pytest.approx(float(car.physics.position[1]) / obs_mod.FIELD_Y, abs=1e-5)
        forward = car.physics.forward
        assert block[2] == pytest.approx(float(forward[0]), abs=1e-6)
        assert block[3] == pytest.approx(float(forward[1]), abs=1e-6)
        assert block[4] == pytest.approx(
            float(state.ball.position[0]) / obs_mod.FIELD_X, abs=1e-5
        )
        assert block[5] == pytest.approx(
            float(state.ball.position[1]) / obs_mod.FIELD_Y, abs=1e-5
        )
        goal_x, goal_y = obs_mod.GOAL_XY[team]
        dx, dy = goal_x - float(state.ball.position[0]), goal_y - float(state.ball.position[1])
        expected = (forward[0] * dx + forward[1] * dy) / np.hypot(dx, dy)
        assert block[6] == pytest.approx(float(expected), abs=1e-5)
        assert block[7] == pytest.approx(np.hypot(dx, dy) / 8000.0, abs=1e-5)
        assert 0.0 <= block[7] <= 1.5


def test_opponent_goal_differs_by_team():
    env = _env("soccer-2v2")
    env.reset()
    state = env.env.state
    state.ball.position = np.array([0.0, 3000.0, 100.0], dtype=np.float32)
    teams = {int(car.team_num) for car in state.cars.values()}
    assert teams == {0, 1}
    # "distance to the enemy goal" must depend on the team, otherwise the
    # feature carries no side information.
    distances = {
        int(state.cars[agent].team_num): build_observation(
            state, agent, env.spec, obs_version=2
        )[obs_mod.FIELD_START + 7]
        for agent in state.cars
    }
    assert abs(distances[0] - distances[1]) > 0.1


def test_observation_is_deterministic_and_order_independent():
    env = _env("soccer-3v3")
    assert env.reset()
    state = env.env.state
    shuffled = dict(reversed(list(state.cars.items())))
    shuffled_state = copy.deepcopy(state)
    shuffled_state.cars = shuffled
    for agent in state.cars:
        first = build_observation(state, agent, env.spec, obs_version=2)
        second = build_observation(shuffled_state, agent, env.spec, obs_version=2)
        np.testing.assert_allclose(first, second, atol=1e-6)


def test_padding_slots_are_zero_for_1v1():
    env = _env("soccer-1v1")
    result = env.reset()
    for values in result.values():
        teammates = values[obs_mod.TEAMMATES_START : obs_mod.TEAMMATES_START + 2 * 9]
        assert not teammates.any()
        opponents = values[obs_mod.OPPONENTS_START : obs_mod.OPPONENTS_START + 3 * 9]
        assert opponents[:9].any()  # exactly one opponent present
        assert not opponents[9:].any()


def test_v2_adds_exactly_eight_values_before_the_action_block():
    env = _env()
    result = env.reset()
    assert result
    agent = next(iter(result))
    env.spec
    v1 = build_observation(env.env.state, agent, env.spec, last_action=7, obs_version=1)
    v2 = build_observation(env.env.state, agent, env.spec, last_action=7, obs_version=2)
    np.testing.assert_allclose(v1[: obs_mod.ACTION_START_V1], v2[: obs_mod.ACTION_START_V1])
    # the action one-hot moves by exactly the field block width
    assert v1[obs_mod.ACTION_START_V1 + 7] == 1.0
    assert v2[obs_mod.ACTION_START_V2 + 7] == 1.0
    assert obs_mod.ACTION_START_V2 - obs_mod.ACTION_START_V1 == 8
