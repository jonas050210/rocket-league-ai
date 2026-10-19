"""Reward shaping: goals dominate, padding/opponents do not receive rewards."""

from __future__ import annotations

import copy

import numpy as np
import pytest

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


def test_legacy_demo_key_maps_to_bump():
    assert RewardWeights.from_dict({"bump": 0.4}).bump == 0.4
    assert RewardWeights.from_dict({"demo": 0.4}).bump == 0.4
    with pytest.raises(ValueError, match="not both"):
        RewardWeights.from_dict({"bump": 0.4, "demo": 0.4})


def test_training_config_accepts_legacy_demo_key():
    from rlai.config import TrainingConfig

    config = TrainingConfig.model_validate({"reward": {"demo": 0.5}})
    assert config.reward.bump == 0.5
    assert "bump" in config.to_dict()["reward"]


def test_unknown_weight_rejected():
    with pytest.raises(ValueError):
        RewardWeights.from_dict({"goal": 1.0, "magic": 2.0})
    with pytest.raises(ValueError):
        RewardWeights.from_dict({"goal": -1.0})


def test_touch_reward_is_an_event_not_a_constant():
    env = _env()
    reward = SoccerReward(env.spec, RewardWeights(touch=1.0))
    state = copy.deepcopy(env.env.state)
    agents = list(state.cars)
    shared = {}
    reward.reset(agents, state, shared)
    for agent in agents:
        state.cars[agent].ball_touches = 0

    before = reward.get_rewards(agents, state, {}, {}, shared)
    assert all(abs(value) < 1.0 for value in before.values())

    for agent in agents:
        state.cars[agent].ball_touches = 1
    touching = reward.get_rewards(agents, state, {}, {}, shared)
    # touches are directional now: with the ball at rest the base factor pays
    # 0.3 * weight; a touch that drives the ball at the goal pays in full
    assert all(value >= 0.25 for value in touching.values())
    # the engine zeroes the counter after every state read, so the next step
    # starts at zero and must not pay the same contact again
    for agent in agents:
        state.cars[agent].ball_touches = 0
    after = reward.get_rewards(agents, state, {}, {}, shared)
    assert all(abs(value) < 0.25 for value in after.values())


def test_reward_components_match_the_documented_order():
    from rlai.rewards import _COMPONENTS

    env = _env()
    reward = SoccerReward(env.spec)
    state = copy.deepcopy(env.env.state)
    agent = next(iter(state.cars))
    value, parts = reward._reward(agent, state)
    assert len(parts) == len(_COMPONENTS)
    assert value == np.float32(value) or isinstance(value, float)
    assert abs(sum(parts) - value) < 1e-6

def test_consecutive_steps_count_every_contact():
    """``ball_touches`` is a per-step count, not a running total.

    The simulator zeroes the counter after each state read (rlgym's own
    ``Car.ball_touches`` docstring: "number of ball touches since last state
    was sent").  Treating it as cumulative dropped the second of two contacts
    that happened on consecutive decision steps.
    """
    env = _env()
    reward = SoccerReward(env.spec, RewardWeights(touch=1.0, progress=0.0, ball_progress=0.0))
    state = copy.deepcopy(env.env.state)
    agents = list(state.cars)
    shared = {}
    reward.reset(agents, state, shared)
    for agent in agents:
        state.cars[agent].ball_touches = 0
    for _ in range(2):
        for agent in agents:
            state.cars[agent].ball_touches = 1  # one contact this step
        reward.get_rewards(agents, state, {}, {}, shared)
        for agent in agents:
            state.cars[agent].ball_touches = 0  # what the engine does next
    for agent in agents:
        assert shared["reward_stats_by_agent"][agent]["touches"] == 2

