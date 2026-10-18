"""Side symmetry: the observation mirror must match the simulator's own.

A checkpoint that only ever played blue can only be evaluated - or deployed -
on orange if the observation can be rewritten into the side it was trained
on.  These tests check that rewrite against a mirrored *real* game state
built with RLGym's inversion primitives, so the convention cannot drift apart
from the simulator's.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlai.envs import MatchEnv
from rlai.modes import get_mode
from rlai.obs import (
    FIELD_START,
    OBS_SIZE_V1,
    OBS_SIZE_V2,
    PAD_FEATURES,
    PADS_START,
    mirror_observation,
    build_observation,
    pad_mirror_permutation,
)
from rlai.opponents import ChaseOpponent, IdleOpponent

from tests.mirror_utils import invert_state


def _live_state(mode: str = "soccer-2v2", steps: int = 5):
    spec = get_mode(mode)
    env = MatchEnv(spec, ChaseOpponent(), seed=17, episode_seconds=20)
    obs = env.reset(episode_seed=101)
    for index in range(steps):
        actions = {agent: (index * 7 + 3) % 90 for agent in env.policy_agents}
        result = env.step(actions)
        obs = result.obs
    return env, spec, obs


def test_pad_permutation_is_the_simulator_reversal():
    permutation = pad_mirror_permutation()
    assert len(permutation) == PAD_FEATURES
    assert sorted(permutation) == list(range(PAD_FEATURES))
    # reversing twice is the identity
    assert [permutation[permutation[i]] for i in range(PAD_FEATURES)] == list(range(PAD_FEATURES))
    from rlgym.rocket_league.common_values import BOOST_LOCATIONS

    locations = np.asarray(BOOST_LOCATIONS, dtype=float)
    for index in range(PAD_FEATURES):
        target = np.array([-locations[index, 0], -locations[index, 1], locations[index, 2]])
        # the reference table rounds one pad 2 uu differently than RocketSim
        assert np.abs(locations[permutation[index]] - target).max() <= 2.5, index


def test_mirror_observation_matches_a_mirrored_real_state():
    env, spec, obs = _live_state()
    state = env.state
    time_remaining = float(env.env.shared_info["time_remaining"])
    score_diff = int(env.env.shared_info["score_diff"])
    mirrored_state = invert_state(state)

    for agent in env.env.agents:
        original = build_observation(
            state,
            agent,
            spec,
            13,
            time_remaining=time_remaining,
            score_diff=score_diff,
            obs_version=2,
        )
        # In the mirrored scene the agent plays for the other team, so the same
        # builder must produce exactly what mirror_observation promises.
        expected = build_observation(
            mirrored_state,
            agent,
            spec,
            13,
            time_remaining=time_remaining,
            score_diff=score_diff,
            obs_version=2,
        )
        mirrored = mirror_observation(original, 2)
        assert mirrored == pytest.approx(expected, abs=1e-5), agent


def test_mirror_is_an_involution_and_keeps_actions():
    _, _, obs = _live_state(steps=3)
    agent_obs = obs[sorted(obs)[0]]
    twice = mirror_observation(mirror_observation(agent_obs, 2), 2)
    assert np.allclose(twice, agent_obs, atol=1e-6)
    mirrored = mirror_observation(agent_obs, 2)
    # the last-action one-hot block is untouched: every control is car-relative
    from rlai.obs import ACTION_START_V2

    assert np.array_equal(mirrored[ACTION_START_V2:], agent_obs[ACTION_START_V2:])
    # everything before the absolute field block is car-relative and untouched
    assert np.array_equal(mirrored[:FIELD_START], agent_obs[:FIELD_START])


def test_mirror_reverses_pads_and_flips_the_field_block():
    observation = np.zeros(OBS_SIZE_V2, dtype=np.float32)
    observation[PADS_START : PADS_START + PAD_FEATURES] = np.arange(PAD_FEATURES, dtype=np.float32)
    observation[FIELD_START : FIELD_START + 8] = np.arange(8, dtype=np.float32) + 1.0
    mirrored = mirror_observation(observation, 2)
    assert np.array_equal(
        mirrored[PADS_START : PADS_START + PAD_FEATURES],
        np.arange(PAD_FEATURES, dtype=np.float32)[::-1],
    )
    # x/y of car, forward and ball flip; goal alignment and distance do not
    assert np.allclose(mirrored[FIELD_START : FIELD_START + 6], -np.arange(6, dtype=np.float32) - 1.0)
    assert np.allclose(mirrored[FIELD_START + 6 : FIELD_START + 8], np.array([7.0, 8.0]))


def test_mirror_supports_the_legacy_layout():
    observation = np.arange(OBS_SIZE_V1, dtype=np.float32)
    mirrored = mirror_observation(observation, 1)
    assert mirrored.shape == (OBS_SIZE_V1,)
    assert np.array_equal(mirrored[PADS_START : PADS_START + PAD_FEATURES], observation[PADS_START : PADS_START + PAD_FEATURES][::-1])
    with pytest.raises(ValueError):
        mirror_observation(observation, 2)


def test_mirrored_env_play_is_symmetric_for_a_scripted_opponent():
    """The env on orange must behave like the env on blue, mirrored."""
    blue_env = MatchEnv(get_mode("soccer-1v1"), IdleOpponent(), seed=3, episode_seconds=10)
    orange_env = MatchEnv(
        get_mode("soccer-1v1"), IdleOpponent(), seed=3, episode_seconds=10, policy_team=1
    )
    assert blue_env.policy_agents == blue_env.blue_agents
    assert orange_env.policy_agents == orange_env.orange_agents
    assert orange_env.opponent_agents == orange_env.blue_agents
    blue_obs = blue_env.reset(episode_seed=11)
    orange_obs = orange_env.reset(episode_seed=11)
    blue_agent = blue_env.policy_agents[0]
    orange_agent = orange_env.policy_agents[0]
    mirrored = mirror_observation(orange_obs[orange_agent], 2)
    # Kickoff spawns are mirrored between the teams, so the mirrored orange
    # observation has to agree with the blue one on every side-dependent slot.
    assert mirrored[FIELD_START : FIELD_START + 6] == pytest.approx(
        blue_obs[blue_agent][FIELD_START : FIELD_START + 6], abs=1e-4
    )
