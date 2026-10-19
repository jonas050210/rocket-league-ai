"""Skill probes: structure, determinism, side balance and discrimination.

The discrimination tests are the important ones: a probe that a car doing
nothing passes is not a gate.  Every threshold in the mastery curriculum was
derived from these reference rates (docs/MASTERY_CURRICULUM.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from rlai.actions import ACTION_TABLE
from rlai.envs import MatchEnv
from rlai.modes import get_mode
from rlai.opponents import ChaseOpponent, IdleOpponent
from rlai.policy import PolicyConfig, TorchPolicy
from rlai.scenarios import apply_scenario
from rlai.skills import MATCH_SKILLS, PROBES, SKILL_NAMES, run_probe

POLICY_CONFIG = PolicyConfig(obs_size=222, hidden=(16, 16))


def _action_index(throttle: int, steer: int, boost: int = 0) -> int:
    target = np.array([throttle, steer, 0, steer, 0, 0, boost, 0], dtype=np.float32)
    return int(np.where((ACTION_TABLE == target).all(axis=1))[0][0])


def constant_policy(index: int) -> TorchPolicy:
    """A real TorchPolicy that always chooses one action (no test doubles)."""
    policy = TorchPolicy.from_config(POLICY_CONFIG, device="cpu")
    with __import__("torch").no_grad():
        layer = [m for m in policy.model.actor.net if hasattr(m, "weight")][-1]
        layer.weight.zero_()
        layer.bias.zero_()
        layer.bias[index] = 10.0
    return policy


class ChasePolicy:
    """The scripted chase driver, exposed through the policy interface."""

    def __init__(self):
        self.chase = ChaseOpponent()
        self.model = POLICY_CONFIG.build()
        self.device = "cpu"

    def act(self, obs, *, deterministic=True, generator=None):
        keys = list(obs)
        actions = self.chase.act(_ENV["env"], obs, keys)
        return actions, {k: 0.0 for k in keys}, {k: 0.0 for k in keys}


class DriveAtBallPolicy:
    """Steers at the ball and *always* keeps the throttle on.

    :class:`rlai.opponents.ChaseOpponent` lifts the throttle when the ball is
    more than ~109 deg off the nose, so a car facing away from the ball never
    turns (measured: 0.00 on the ``reengage`` probe).  A driver that keeps
    rolling does turn, which is what the recovery skill actually needs.
    """

    def __init__(self):
        self.model = POLICY_CONFIG.build()
        self.device = "cpu"

    def act(self, obs, *, deterministic=True, generator=None):
        env = _ENV["env"]
        state = env.state
        ball = np.asarray(state.ball.position, dtype=np.float32)
        actions = {}
        for agent in obs:
            car = state.cars[agent]
            flat = (ball - np.asarray(car.physics.position, dtype=np.float32))[:2]
            forward = np.asarray(car.physics.forward, dtype=np.float32)
            angle = float(
                np.arctan2(
                    forward[0] * flat[1] - forward[1] * flat[0],
                    forward[0] * flat[0] + forward[1] * flat[1],
                )
            )
            steer = 0 if abs(angle) < 0.06 else (1 if angle > 0 else -1)
            boost = int(abs(angle) < 0.5 and float(car.boost_amount) > 12)
            actions[agent] = _action_index(1, steer, boost)
        return actions, {k: 0.0 for k in obs}, {k: 0.0 for k in obs}


_ENV: dict = {"env": None}


@pytest.fixture()
def chase_reference(monkeypatch):
    """Let a scripted driver see the simulator state during a probe."""
    import rlai.skills as skills

    original = skills._play

    def play(env, view, trial, team):
        _ENV["env"] = env
        return original(env, view, trial, team)

    monkeypatch.setattr(skills, "_play", play)
    return ChasePolicy()


@pytest.fixture()
def driving_reference(chase_reference):
    """A driver that steers at the ball and never lifts (reuses the patch)."""
    return DriveAtBallPolicy()


# ---------------------------------------------------------------------------
# structure, determinism, side balance
# ---------------------------------------------------------------------------
def test_every_probe_runs_on_both_sides_and_reports_uncertainty():
    policy = TorchPolicy.from_config(POLICY_CONFIG, device="cpu")
    for skill in SKILL_NAMES:
        result = run_probe(policy, skill, trials=4, obs_version=4)
        assert result["skill"] == skill
        assert result["trials"] == 4
        assert result["rate"] == pytest.approx(
            sum(row["success"] for row in result["details"]) / 4
        )
        low, high = result["ci95"]
        assert 0.0 <= low <= result["rate"] <= high <= 1.0
        assert sorted(result["side_results"]) == ["blue", "orange"]
        assert sum(side["trials"] for side in result["side_results"].values()) == 4
        assert result["duration_s"] > 0.0


def test_probe_results_are_reproducible():
    policy = TorchPolicy.from_config(POLICY_CONFIG, device="cpu")
    first = run_probe(policy, "approach", trials=6, seed=99, obs_version=4)
    second = run_probe(policy, "approach", trials=6, seed=99, obs_version=4)
    assert first["details"] == second["details"]
    assert first["rate"] == second["rate"]


def test_unknown_skill_or_side_is_refused():
    policy = TorchPolicy.from_config(POLICY_CONFIG, device="cpu")
    with pytest.raises(ValueError, match="unknown skill probe"):
        run_probe(policy, "kickflip", trials=2)
    with pytest.raises(ValueError, match="unknown side"):
        run_probe(policy, "approach", trials=2, sides=("purple",))


def test_match_skills_name_their_opponent():
    assert MATCH_SKILLS == {"match_chase": "chase", "match_mixed": "mixed"}
    assert set(MATCH_SKILLS).isdisjoint(PROBES)


# ---------------------------------------------------------------------------
# the scenarios really are applied (and mirrored for the orange side)
# ---------------------------------------------------------------------------
def test_scenario_places_ball_and_cars_exactly():
    env = MatchEnv(get_mode("soccer-1v1"), IdleOpponent(), seed=3, episode_seconds=6.0,
                   obs_version=4, policy_team=0)
    try:
        env.scenario = {
            "name": "unit",
            "ball": {"position": (500.0, -1000.0, 93.0), "velocity": (0.0, 300.0, 0.0)},
            "policy": [{"position": (500.0, -1800.0, 17.0), "aim": (500.0, -1000.0),
                        "boost": 42.0}],
            "opponent": [{"position": (0.0, 4600.0, 17.0), "yaw": -np.pi / 2}],
        }
        env.reset()
        ball = np.asarray(env.state.ball.position)
        assert ball[0] == pytest.approx(500.0, abs=1.0)
        assert ball[1] == pytest.approx(-1000.0, abs=1.0)
        car = env.state.cars[env.policy_agents[0]]
        assert float(car.physics.position[1]) == pytest.approx(-1800.0, abs=1.0)
        assert float(car.boost_amount) == pytest.approx(42.0, abs=0.5)
        # facing the ball: forward points along +y
        assert float(np.asarray(car.physics.forward)[1]) > 0.9
    finally:
        env.close()


def test_scenario_is_mirrored_for_the_orange_side():
    """One scenario definition measures both sides of the field."""
    spec = get_mode("soccer-1v1")
    scenario = {
        "name": "unit",
        "ball": {"position": (400.0, 1000.0, 93.0), "velocity": (100.0, 200.0, 0.0)},
        "policy": [{"position": (400.0, 200.0, 17.0), "aim": (400.0, 1000.0)}],
        "opponent": [{"position": (0.0, 4600.0, 17.0), "yaw": -np.pi / 2}],
    }
    env = MatchEnv(spec, IdleOpponent(), seed=4, episode_seconds=6.0, obs_version=4,
                   policy_team=1)
    try:
        env.reset()  # no scenario yet: a plain kickoff
        env.scenario = scenario
        env.reset()
        ball = np.asarray(env.state.ball.position)
        assert ball[0] == pytest.approx(-400.0, abs=1.0)
        assert ball[1] == pytest.approx(-1000.0, abs=1.0)
        velocity = np.asarray(env.state.ball.linear_velocity)
        assert velocity[0] == pytest.approx(-100.0, abs=1.0)
        assert velocity[1] == pytest.approx(-200.0, abs=1.0)
        car = env.state.cars[env.policy_agents[0]]
        assert float(car.physics.position[1]) == pytest.approx(-200.0, abs=1.0)
        # orange attacks -y, so "aim at the ball" still faces the ball
        assert float(np.asarray(car.physics.forward)[1]) < -0.9
    finally:
        env.close()


def test_apply_scenario_reports_the_ball_position():
    class State:
        class _Ball:
            position = np.zeros(3, dtype=np.float32)
            linear_velocity = np.zeros(3, dtype=np.float32)
            angular_velocity = np.zeros(3, dtype=np.float32)

        class _Config:
            boost_consumption = 1.0

        def __init__(self):
            self.ball = State._Ball()
            self.config = State._Config()
            self.cars = {}

    state = State()
    shared = {"drill_team": 0}
    apply_scenario(state, shared, {"name": "x", "ball": {"position": (10.0, 20.0, 93.0)},
                                   "unlimited_boost": True})
    assert shared["drill_scenario"] == "x"
    assert state.config.boost_consumption == 0.0


# ---------------------------------------------------------------------------
# discrimination: the reason these probes can gate anything
# ---------------------------------------------------------------------------
def test_a_car_that_never_drives_forward_does_not_reach_the_ball():
    policy = constant_policy(_action_index(-1, 0))  # reverse, no steering
    result = run_probe(policy, "approach", trials=6, obs_version=4)
    assert result["rate"] == 0.0
    assert result["touches"] == 0.0


def test_the_scripted_chase_driver_reaches_the_ball_every_time(chase_reference):
    result = run_probe(chase_reference, "approach", trials=8, obs_version=4)
    assert result["rate"] == 1.0
    assert result["ci95"][0] > 0.6
    assert result["touches"] > 1.0


def test_a_fresh_network_is_far_from_the_approach_gate():
    """Stage 0 must start blocked, otherwise the gate measures nothing."""
    policy = TorchPolicy.from_config(POLICY_CONFIG, device="cpu")
    result = run_probe(policy, "approach", trials=16, obs_version=4)
    assert result["rate"] < 0.5


def test_reengage_needs_movement_and_a_turn(driving_reference):
    """Facing away from the ball: a passive car never gets back to it.

    (Reversing *does* work - backing into the ball is a legitimate recovery -
    so the passive reference here is a car with no throttle at all.)
    """
    passive = run_probe(constant_policy(_action_index(0, 0)), "reengage", trials=6,
                        obs_version=4)
    assert passive["rate"] == 0.0
    # driving straight never turns, so it cannot re-engage either
    straight = run_probe(constant_policy(_action_index(1, 0)), "reengage", trials=6,
                         obs_version=4)
    assert straight["rate"] == 0.0
    recovered = run_probe(driving_reference, "reengage", trials=8, obs_version=4)
    assert recovered["rate"] >= 0.75
