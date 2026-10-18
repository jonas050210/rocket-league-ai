"""All four modes build, reset and step through the real simulator."""

from __future__ import annotations

import numpy as np
import pytest

from rlai.envs import MatchEnv
from rlai.modes import all_modes
from rlai.obs import (
    BALL_FEATURES,
    CAR_FEATURES,
    OBS_SIZE,
    OPPONENT_SLOTS,
    SELF_FEATURES,
    TEAMMATE_SLOTS,
)
from rlai.opponents import build_opponent

SELF = slice(0, SELF_FEATURES)
BALL = slice(SELF_FEATURES, SELF_FEATURES + BALL_FEATURES)
TEAMMATES = slice(
    BALL.start + BALL_FEATURES, BALL.start + BALL_FEATURES + TEAMMATE_SLOTS * CAR_FEATURES
)
OPPONENTS = slice(TEAMMATES.stop, TEAMMATES.stop + OPPONENT_SLOTS * CAR_FEATURES)


@pytest.mark.parametrize("mode_id", [spec.id for spec in all_modes()])
def test_mode_reset_and_step(mode_id):
    from rlai.modes import get_mode

    spec = get_mode(mode_id)
    env = MatchEnv(spec, build_opponent("idle"), seed=7, episode_seconds=10)
    obs = env.reset()
    agents = env.blue_agents
    assert len(agents) == spec.team_size
    assert len(env.orange_agents) == spec.team_size
    for agent in agents:
        assert obs[agent].shape == (OBS_SIZE,)
        assert np.isfinite(obs[agent]).all()

    # teammate padding: absent slots stay zero
    teammate_block = obs[agents[0]][TEAMMATES]
    if spec.team_size == 1:
        assert not teammate_block.any()
    else:
        used = (spec.team_size - 1) * CAR_FEATURES
        assert np.abs(teammate_block[used:]).sum() == 0.0

    result = env.step({agent: 0 for agent in agents})
    assert set(result.rewards) == set(agents)
    assert result.info.tick_count > 0
    for agent in agents:
        assert np.isfinite(result.obs[agent]).all()


def test_observation_is_bounded():
    from rlai.modes import get_mode

    env = MatchEnv(
        get_mode("soccer-1v1"), build_opponent("chase", seed=1), seed=1, episode_seconds=10
    )
    env.reset()
    agent = env.blue_agents[0]
    env.step({agent: 0})
    values = env.step({agent: 0}).obs[agent]
    assert values.min() >= -5.0 and values.max() <= 5.0


def test_episode_truncates_on_time():
    from rlai.modes import get_mode

    env = MatchEnv(get_mode("soccer-1v1"), build_opponent("idle"), seed=1, episode_seconds=2)
    env.reset()
    truncated = False
    for _ in range(80):
        result = env.step({agent: 0 for agent in env.blue_agents})
        if result.truncated:
            truncated = True
            break
    assert truncated


def test_seed_reproducibility():
    from rlai.modes import get_mode

    def rollout(seed):
        env = MatchEnv(
            get_mode("soccer-1v1"),
            build_opponent("chase", seed=seed),
            seed=seed,
            episode_seconds=10,
        )
        env.reset()
        trace = []
        for _ in range(6):
            result = env.step({agent: 0 for agent in env.blue_agents})
            trace.append(result.obs[env.blue_agents[0]].copy())
        return np.stack(trace)

    assert np.allclose(rollout(3), rollout(3), atol=1e-5)
