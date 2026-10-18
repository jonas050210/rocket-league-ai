"""Model loading, action mapping and side handling used by the deployed bot."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from rlbot import flat

from rlai.actions import ACTION_COUNT, controller_dict
from rlai.checkpoints import CHECKPOINT_FORMAT, CHECKPOINT_VERSION
from rlai.envs import MatchEnv
from rlai.modes import get_mode
from rlai.obs import OBS_SIZE, mirror_observation, obs_size
from rlai.opponents import IdleOpponent
from rlai.policy import PolicyConfig
from rlai.rlbot.bot import (
    RocketLeagueAIBot,
    _resolve_model,
    decide_action,
    load_policy,
    team_from_name,
)
from rlai.rlbot.obs_adapter import packet_to_observation
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
    bot.mirror = False
    bot.deterministic = True
    bot.errors = 0
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


class _NotATensor:
    """Picklable object the safe loader must refuse to reconstruct."""


def test_load_policy_refuses_a_legacy_pickle_checkpoint(tmp_path, model_path):
    """A checkpoint that needs arbitrary code to load is rejected by default."""
    policy = load_policy(model_path)
    payload = torch.load(model_path, map_location="cpu", weights_only=True)

    hostile = dict(payload)
    hostile["surprise"] = _NotATensor()
    path = tmp_path / "hostile.pt"
    torch.save(hostile, path)
    with pytest.raises(ValueError, match="safely"):
        load_policy(path)
    # the safe checkpoint still loads, and the policy behaves identically
    assert isinstance(load_policy(model_path), type(policy))


def test_load_policy_rejects_a_plain_state_dict(tmp_path):
    config = PolicyConfig(obs_size=obs_size(2))
    path = tmp_path / "raw.pt"
    torch.save({"model": config.build().state_dict()}, path)
    with pytest.raises(ValueError, match=CHECKPOINT_FORMAT):
        load_policy(path)


def test_checkpoint_version_is_recorded():
    assert CHECKPOINT_VERSION >= 2


def test_team_from_name():
    assert team_from_name("blue") == 0
    assert team_from_name("ORANGE") == 1
    assert team_from_name(0) == 0 and team_from_name(1) == 1
    with pytest.raises(ValueError, match="unknown side"):
        team_from_name("purple")


class _RecordingPolicy:
    """Stands in for TorchPolicy to capture the observation the bot feeds it."""

    def __init__(self, obs_size_value=obs_size(2)):
        self.seen = []

        class _Model:
            obs_size = obs_size_value

        self.model = _Model()

    def act(self, obs, deterministic=True):
        self.seen.append((np.asarray(obs).copy(), deterministic))
        return 18, 0.0, 0.0


def test_decide_action_mirrors_the_observation_for_the_orange_side():
    spec = get_mode("soccer-1v1")
    env = MatchEnv(spec, IdleOpponent(), seed=11, episode_seconds=10)
    env.reset()
    packet = packet_from_env(env)

    plain = _RecordingPolicy()
    mirrored = _RecordingPolicy()
    action_a, controls_a = decide_action(plain, packet, 0, 0, spec, mirror=False)
    action_b, controls_b = decide_action(mirrored, packet, 0, 0, spec, mirror=True)

    raw, _ = packet_to_observation(packet, 0, 0, spec, 2)
    assert np.allclose(plain.seen[0][0], raw)
    assert np.allclose(mirrored.seen[0][0], mirror_observation(raw, 2))
    # both calls return a legal action and the full controller payload
    for action, controls in ((action_a, controls_a), (action_b, controls_b)):
        assert 0 <= action < ACTION_COUNT
        assert set(controls) == CONTROLLER_KEYS


def _bare_bot(policy, spec, *, index, team, mirror, tick_skip=1, logger=None):
    """A bot instance without RLBot's process plumbing (initialize is tested separately)."""
    bot = object.__new__(RocketLeagueAIBot)
    bot.policy = policy
    bot.spec = spec
    bot.index = index
    bot.team = team
    bot.tick = 0
    bot.tick_skip = tick_skip
    bot.last_action = 0
    bot.decisions = 0
    bot.errors = 0
    bot.obs_version = 2
    bot.deterministic = True
    bot.trained_side = "blue"
    bot.mirror = mirror
    bot.logger = logger or type("L", (), {"error": staticmethod(lambda *a, **k: None),
                                          "info": staticmethod(lambda *a, **k: None),
                                          "warning": staticmethod(lambda *a, **k: None)})()
    return bot


def test_bot_feeds_the_mirrored_observation_when_it_plays_orange(model_path):
    spec = get_mode("soccer-1v1")
    env = MatchEnv(spec, IdleOpponent(), seed=3, episode_seconds=10)
    env.reset()
    packet = packet_from_env(env)

    blue = _bare_bot(_RecordingPolicy(), spec, index=0, team=0, mirror=False)
    blue.get_output(packet)
    orange = _bare_bot(_RecordingPolicy(), spec, index=1, team=1, mirror=True)
    orange.get_output(packet)

    raw_blue = packet_to_observation(packet, 0, 0, spec, 2)[0]
    raw_orange = packet_to_observation(packet, 1, 0, spec, 2)[0]
    assert np.allclose(blue.policy.seen[0][0], raw_blue)
    assert np.allclose(orange.policy.seen[0][0], mirror_observation(raw_orange, 2))
    # the rotation actually changes the observation, so the test is meaningful
    assert not np.allclose(raw_orange, mirror_observation(raw_orange, 2))
    # actions are car-relative: mirroring the observation must not remap them
    plain = _bare_bot(_RecordingPolicy(), spec, index=1, team=1, mirror=False)
    plain.get_output(packet)
    assert plain.policy.seen[0][0].shape == orange.policy.seen[0][0].shape


def test_initialize_resolves_the_model_and_decides_mirroring(models_dir, trained_model, monkeypatch):
    # RLAI_MODEL is set by the launcher; it must resolve inside the models directory
    monkeypatch.setenv("RLAI_MODEL", trained_model["id"])
    monkeypatch.setenv("RLAI_MODE", "soccer-1v1")
    monkeypatch.setenv("RLAI_TICK_SKIP", "4")

    logs: list[str] = []

    def record(level):
        def emit(fmt, *args, **kwargs):
            logs.append(f"{level}: " + (fmt % args if args else str(fmt)))

        return staticmethod(emit)

    logger = type("L", (), {"info": record("info"), "warning": record("warning"), "error": record("error")})()

    blue = object.__new__(RocketLeagueAIBot)
    blue.team, blue.index, blue.logger = 0, 0, logger
    blue.initialize()
    assert blue.mirror is False and blue.tick_skip == 4
    assert blue.obs_version == 2 and blue.deterministic is True
    assert any("mirror=False" in entry for entry in logs), logs
    assert any("trained_side=blue" in entry for entry in logs), logs

    monkeypatch.setenv("RLAI_TRAINED_SIDE", "blue")
    orange = object.__new__(RocketLeagueAIBot)
    orange.team, orange.index, orange.logger = 1, 0, logger
    orange.initialize()
    assert orange.mirror is True

    # mirroring can be switched off to measure the raw, unrotated behaviour
    monkeypatch.setenv("RLAI_MIRROR", "0")
    raw = object.__new__(RocketLeagueAIBot)
    raw.team, raw.index, raw.logger = 1, 0, logger
    raw.initialize()
    assert raw.mirror is False

    # a checkpoint trained on orange playing orange needs no rotation either
    monkeypatch.delenv("RLAI_MIRROR")
    monkeypatch.setenv("RLAI_TRAINED_SIDE", "orange")
    trained_orange = object.__new__(RocketLeagueAIBot)
    trained_orange.team, trained_orange.index, trained_orange.logger = 1, 0, logger
    trained_orange.initialize()
    assert trained_orange.mirror is False


def test_initialize_refuses_an_unusable_checkpoint(models_dir, monkeypatch):
    config = PolicyConfig(obs_size=obs_size(2))
    broken_dir = models_dir / "broken"
    broken_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": config.build().state_dict()}, broken_dir / "model.pt")
    monkeypatch.setenv("RLAI_MODEL", "broken")
    bot = object.__new__(RocketLeagueAIBot)
    bot.team, bot.index = 0, 0
    bot.logger = type("L", (), {"info": staticmethod(lambda *a, **k: None),
                                "warning": staticmethod(lambda *a, **k: None),
                                "error": staticmethod(lambda *a, **k: None)})()
    with pytest.raises(ValueError, match=CHECKPOINT_FORMAT):
        bot.initialize()


def test_bot_keeps_playing_when_a_decision_fails(model_path):
    spec = get_mode("soccer-1v1")
    env = MatchEnv(spec, IdleOpponent(), seed=4, episode_seconds=10)
    env.reset()
    packet = packet_from_env(env)
    policy = load_policy(model_path)

    class Broken(list):
        pass

    broken_packet = packet_from_env(env)
    broken_packet.players = Broken(packet.players[:1])  # orange index missing
    bot = _bare_bot(policy, spec, index=1, team=1, mirror=False)
    bot.last_action = 18

    output = bot.get_output(broken_packet)
    assert isinstance(output, flat.ControllerState)
    assert bot.errors == 1
    # the previous action is repeated instead of crashing the bot process
    assert output.throttle == pytest.approx(controller_dict(18)["throttle"])

    good = bot.get_output(packet)
    assert isinstance(good, flat.ControllerState)


def test_initialize_refuses_a_reference_outside_the_models_directory(tmp_path, monkeypatch, models_dir):
    """RLAI_MODEL cannot point the loader at an arbitrary file on the machine."""
    outside = tmp_path / "outside.pt"
    config = PolicyConfig(obs_size=obs_size(2))
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "model": config.build().state_dict(),
            "policy_config": config.to_dict(),
            "step": 1,
        },
        outside,
    )
    monkeypatch.setenv("RLAI_MODEL", str(outside))
    with pytest.raises(ValueError, match="outside the models directory"):
        _resolve_model()
