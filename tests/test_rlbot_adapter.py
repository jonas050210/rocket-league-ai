"""Simulator -> GamePacket -> observation parity for the RLBot adapter.

Building a fake GamePacket from a real simulator state is the closest we can
get to the live game in this environment.  If this drifts, the deployed bot
would receive different features than the policy was trained on.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlai.envs import MatchEnv
from rlai.modes import get_mode
from rlai.obs import build_observation, obs_size
from rlai.opponents import IdleOpponent
from rlai.rlbot.obs_adapter import packet_to_observation
from tests.packet_utils import packet_from_env


@pytest.mark.parametrize("mode_id", ["soccer-1v1", "soccer-3v3", "heatseeker-2v2"])
@pytest.mark.parametrize("version", [1, 2])
def test_packet_observation_matches_training_observation(mode_id, version):
    spec = get_mode(mode_id)
    env = MatchEnv(spec, IdleOpponent(), seed=3, episode_seconds=30, obs_version=version)
    env.reset()
    for _ in range(3):
        env.step({agent: 0 for agent in env.blue_agents})
    state = env.state

    packet = packet_from_env(env, scores=(0, 0), time_remaining=210.0)

    last_action = 40
    obs_packet, meta = packet_to_observation(packet, 0, last_action, spec, obs_version=version)
    reference_agent = list(state.cars)[0]
    obs_reference = build_observation(
        state,
        reference_agent,
        spec,
        last_action,
        time_remaining=210.0,
        score_diff=0,
        obs_version=version,
    )
    assert obs_packet.shape == obs_reference.shape == (obs_size(version),)
    assert np.allclose(obs_packet, obs_reference, atol=1e-3)
    assert meta["team"] == int(state.cars[reference_agent].team_num)
