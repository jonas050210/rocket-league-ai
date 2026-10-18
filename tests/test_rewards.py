"""Reward shaping: goals dominate, padding/opponents do not receive rewards."""

from __future__ import annotations

import copy

import numpy as np

from rlai.envs import MatchEnv
from rlai.modes import get_mode
from rlai.opponents import IdleOpponent
from rlai.rewards import RewardWeights, SoccerReward


def _env(mode="soccer-1v1"):
    env = MatchEnv(get_mode(mode), IdleOpponent(), seed=2, episode_seconds=10)
    env.reset()
    return env


def test_goal_and_concede_are_symmetric():
    env = _env()
    spec = env.spec
    reward = SoccerReward(spec, RewardWeights(goal=10.0, concede=10.0))
    state = copy.deepcopy(env.env.state)
    agents = list(state.cars)
    shared = {}
    reward.reset(agents, state, shared)

    state.goal_scored = True
    state.ball.position = np.array([0.0, 6000.0, 100.0], dtype=np.float32)
    values = reward.get_rewards(agents, state, {}, {}, shared)
    blue = [a for a in agents if state.cars[a].team_num == 0]
    orange = [a for a in agents if state.cars[a].team_num == 1]
    assert all(values[a] > 9.0 for a in blue)
    assert all(values[a] < -9.0 for a in orange)


def test_potential_shaping_has_no_goal_bonus_without_goal():
    env = _env()
    reward = SoccerReward(env.spec, RewardWeights(goal=10.0, concede=10.0))
    state = copy.deepcopy(env.env.state)
    agents = list(state.cars)
    shared = {}
    reward.reset(agents, state, shared)
    values = reward.get_rewards(agents, state, {}, {}, shared)
    assert all(abs(v) < 5.0 for v in values.values())


def test_unknown_weight_rejected():
    import pytest

    with pytest.raises(ValueError):
        RewardWeights.from_dict({"goal": 1.0, "magic": 2.0})
    with pytest.raises(ValueError):
        RewardWeights.from_dict({"goal": -1.0})
