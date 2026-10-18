"""Heatseeker rules are verified against the simulator, not assumed.

RocketSim implements the mode natively; these tests prove that our env
activates it and that the documented rules (homing after a touch, speed-up
per touch, unlimited boost, goals counted) actually occur.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from rlai.envs import MatchEnv
from rlai.modes import get_mode
from rlai.opponents import IdleOpponent

BLUE, ORANGE = 0, 1


@pytest.fixture()
def hs_env():
    spec = get_mode("heatseeker-2v2")
    env = MatchEnv(spec, IdleOpponent(), seed=5, episode_seconds=30)
    env.reset()
    assert env.env.transition_engine._arena.game_mode == 2  # rsim.GameMode.HEATSEEKER
    return env


def _force_touch(env: MatchEnv):
    state = copy.deepcopy(env.env.state)
    blue = env.blue_agents[0]
    car = state.cars[blue]
    state.ball.position = np.array([0.0, -2220.0, 92.75], dtype=np.float32)
    state.ball.linear_velocity = np.zeros(3, dtype=np.float32)
    state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
    car.physics.position = np.array([0.0, -2370.0, 17.0], dtype=np.float32)
    car.physics.linear_velocity = np.array([0.0, 800.0, 0.0], dtype=np.float32)
    car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
    car.physics.euler_angles = np.array([0.0, np.pi / 2, 0.0], dtype=np.float32)
    car.boost_amount = 100.0
    env.env.set_state(state)
    return blue


def test_ball_starts_without_target(hs_env):
    ball = hs_env.env.transition_engine._arena.ball.get_state()
    assert ball.heatseeker_target_dir == 0.0
    assert ball.heatseeker_target_speed == pytest.approx(2900.0)


def test_touch_engages_homing_and_speedup(hs_env):
    blue = _force_touch(hs_env)
    arena = hs_env.env.transition_engine._arena
    touched = False
    for _ in range(4):
        hs_env.step({agent: 0 for agent in hs_env.blue_agents})
        ball = arena.ball.get_state()
        if ball.heatseeker_target_dir != 0.0:
            touched = True
            break
    assert touched, "the scripted car should have touched the ball"
    assert ball.heatseeker_target_dir == 1.0  # touched by blue -> homes to orange goal
    assert ball.heatseeker_target_speed == pytest.approx(2985.0)
    assert np.linalg.norm(ball.vel.as_numpy()) > 2000.0
    assert hs_env.env.state.cars[blue].ball_touches > 0


def test_goal_is_scored_in_orange_net(hs_env):
    _force_touch(hs_env)
    scored = None
    for _ in range(80):
        result = hs_env.step({agent: 0 for agent in hs_env.blue_agents})
        if result.done:
            scored = result
            break
    assert scored is not None, "ball did not reach the goal"
    assert scored.info.goal_scored
    assert scored.info.scoring_team == BLUE
    assert hs_env.score_blue == 1


def test_unlimited_boost(hs_env):
    blue = hs_env.blue_agents[0]
    car = hs_env.env.state.cars[blue]
    car.boost_amount = 100.0
    hs_env.env.set_state(copy.deepcopy(hs_env.env.state))
    # boost continuously for 120 ticks and confirm fuel is not consumed
    before = hs_env.env.state.cars[blue].boost_amount
    for _ in range(15):
        hs_env.step({agent: 89 for agent in hs_env.blue_agents})
    after = hs_env.env.state.cars[blue].boost_amount
    assert after >= before - 1e-3
    assert hs_env.env.state.config.boost_consumption == 0.0
