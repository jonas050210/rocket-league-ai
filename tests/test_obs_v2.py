"""Observation v2: field frame, per-team next-goal features, pad order."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from rlai.actions import ACTION_COUNT
from rlai.envs import MatchEnv
from rlai.modes import get_mode
from rlai.obs import (
    ACTION_START_V1,
    ACTION_START_V2,
    FIELD_START,
    OBS_SIZE,
    OBS_VERSION,
    build_observation,
    obs_size,
    version_for_obs_size,
)
from rlai.opponents import IdleOpponent


def _env(mode: str = "soccer-2v2"):
    env = MatchEnv(get_mode(mode), IdleOpponent(), seed=11, episode_seconds=40)
    env.reset()
    env.step({agent: 0 for agent in env.blue_agents})
    return env


def test_sizes_match_the_documented_versions():
    assert OBS_VERSION == 2
    assert obs_size(1) == 190
    assert obs_size(2) == OBS_SIZE == 198
    assert version_for_obs_size(190) == 1
    assert version_for_obs_size(198) == 2
    assert version_for_obs_size(OBS_SIZE) == 2
    with pytest.raises(ValueError):
        obs_size(3)


def test_version_one_payload_is_the_legacy_prefix():
    env = _env()
    agent = env.blue_agents[0]
    spec = get_mode("soccer-2v2")
    v1 = build_observation(
        env.state, agent, spec, 40, time_remaining=180.0, score_diff=1, obs_version=1
    )
    v2 = build_observation(
        env.state, agent, spec, 40, time_remaining=180.0, score_diff=1, obs_version=2
    )
    assert v1.shape == (190,)
    assert v2.shape == (198,)
    # field features are inserted before the action block in v2
    assert np.allclose(v1, np.concatenate([v2[:FIELD_START], v2[ACTION_START_V2:]]))
    # 90 actions in both versions, the field block sits between them
    assert ACTION_START_V1 == FIELD_START
    assert ACTION_START_V2 == ACTION_START_V1 + 8


def test_field_frame_uses_absolute_coordinates_and_attacking_goal():
    env = _env()
    spec = get_mode("soccer-2v2")
    blue_agent = next(agent for agent in env.state.cars if agent.startswith("blue"))
    orange_agent = next(agent for agent in env.state.cars if agent.startswith("orange"))
    # asymmetric ball so the two frames cannot coincide accidentally
    env.state.ball.position = np.array([500.0, 3000.0, 120.0], dtype=np.float32)
    blue_obs = build_observation(
        env.state, blue_agent, spec, 0, time_remaining=200.0, score_diff=0, obs_version=2
    )
    orange_obs = build_observation(
        env.state, orange_agent, spec, 0, time_remaining=200.0, score_diff=0, obs_version=2
    )
    blue_field = blue_obs[FIELD_START : FIELD_START + 8]
    orange_field = orange_obs[FIELD_START : FIELD_START + 8]
    # ball x/y in the own attacking frame are mirrored between the teams
    # positions are absolute field coordinates, normalised by half field size
    for field, agent in ((blue_field, blue_agent), (orange_field, orange_agent)):
        car_pos = env.state.cars[agent].physics.position
        assert float(field[0]) == pytest.approx(float(car_pos[0]) / 4096.0, abs=1e-4)
        assert float(field[1]) == pytest.approx(float(car_pos[1]) / 5120.0, abs=1e-4)
    assert float(blue_field[4]) == pytest.approx(500.0 / 4096.0, abs=1e-4)
    assert float(blue_field[5]) == pytest.approx(3000.0 / 5120.0, abs=1e-4)
    # the encoded goal is the one the agent attacks: blue +5120, orange -5120
    assert float(blue_field[7]) == pytest.approx(
        (500.0**2 + (5120.0 - 3000.0) ** 2) ** 0.5 / 8000.0, abs=1e-4
    )
    assert float(orange_field[7]) == pytest.approx(
        (500.0**2 + (5120.0 + 3000.0) ** 2) ** 0.5 / 8000.0, abs=1e-4
    )
    assert np.all(np.isfinite(blue_field)) and np.all(np.isfinite(orange_field))
    # no pad information leaked into the field block
    assert np.all(np.abs(blue_field) < 30.0)


def test_observation_is_deterministic_and_order_independent():
    env = _env()
    agent = env.blue_agents[1]
    spec = get_mode("soccer-2v2")
    first = build_observation(
        env.state, agent, spec, 12, time_remaining=123.0, score_diff=-2, obs_version=2
    )
    second = build_observation(
        env.state, agent, spec, 12, time_remaining=123.0, score_diff=-2, obs_version=2
    )
    assert np.array_equal(first, second)
    # car dictionary order must not leak into the features
    shuffled = copy.deepcopy(env.state)
    shuffled.cars = dict(list(shuffled.cars.items())[::-1])
    third = build_observation(
        shuffled, agent, spec, 12, time_remaining=123.0, score_diff=-2, obs_version=2
    )
    assert np.allclose(first, third, atol=1e-6)


def test_terminal_observation_is_finite_and_bounded():
    env = _env()
    spec = get_mode("soccer-2v2")
    env.state.ball.position = np.array([0.0, 5120.0, 0.0], dtype=np.float32)
    obs = build_observation(
        env.state,
        env.blue_agents[0],
        spec,
        0,
        time_remaining=0.0,
        score_diff=0,
        obs_version=2,
    )
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    field = obs[FIELD_START : FIELD_START + 8]
    assert np.all(np.abs(field) < 1e4)
    assert len(obs) == OBS_SIZE
    # action block is a one-hot vector
    assert float(np.sum(obs[ACTION_START_V2:])) == pytest.approx(1.0)
    assert int(np.argmax(obs[ACTION_START_V2:])) == 0
    assert ACTION_COUNT == 90
