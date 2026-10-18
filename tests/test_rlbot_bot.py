"""Model loading and action mapping used by the deployed bot."""

from __future__ import annotations

import numpy as np
import pytest
from rlbot import flat

from rlai.actions import ACTION_COUNT, controller_dict
from rlai.envs import MatchEnv
from rlai.modes import get_mode
from rlai.obs import OBS_SIZE
from rlai.opponents import IdleOpponent
from rlai.rlbot.bot import RocketLeagueAIBot, decide_action, load_policy
from tests.packet_utils import packet_from_env

CONTROLLER_KEYS = {
    "throttle",
    "steer",
    "pitch",
    "yaw",
    "roll",
    "jump",
    "boost",
    "handbrake",
}


def test_load_policy_from_checkpoint(model_path):
    policy = load_policy(model_path, device="cpu")
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    action, logprob, value = policy.act(obs, deterministic=True)
    assert 0 <= action < ACTION_COUNT
    assert np.isfinite(logprob) and np.isfinite(value)
    assert set(controller_dict(action)) == CONTROLLER_KEYS


def test_bot_action_loop_is_deterministic(model_path):
    policy = load_policy(model_path, device="cpu")
    obs = np.linspace(-1, 1, OBS_SIZE, dtype=np.float32)
    first = policy.act(obs, deterministic=True)[0]
    for _ in range(5):
        assert policy.act(obs, deterministic=True)[0] == first


def test_bot_decision_and_controller_output(model_path):
    spec = get_mode("soccer-1v1")
    env = MatchEnv(spec, IdleOpponent(), seed=5, episode_seconds=10)
    env.reset()
    packet = packet_from_env(env)

    policy = load_policy(model_path, device="cpu")
    action, payload = decide_action(policy, packet, 0, 0, spec)
    assert 0 <= action < ACTION_COUNT
    assert set(payload) == CONTROLLER_KEYS

    bot = object.__new__(RocketLeagueAIBot)
    bot.policy = policy
    bot.spec = spec
    bot.index = 0
    bot.tick = 0
    bot.tick_skip = 1
    bot.last_action = 0
    bot.decisions = 0
    bot.obs_version = 2
    output = bot.get_output(packet)
    assert isinstance(output, flat.ControllerState)
    assert output.throttle == pytest.approx(payload["throttle"])
    assert bool(output.jump) == payload["jump"]
    assert bot.decisions == 1


def test_missing_model_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("RLAI_DATA_DIR", str(tmp_path))
    from rlai.rlbot.bot import _resolve_model

    with pytest.raises(FileNotFoundError, match="No model configured"):
        _resolve_model()
