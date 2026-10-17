"""Published RLGym v2 / RocketSim APIs, fixed 1v1 kickoff-only headless environment."""

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rlgym.api import RewardFunction, RLGym, StateMutator
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import (
    AnyCondition,
    GoalCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import GoalReward, TouchReward
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)

from ..config import TrainingConfig

_SIMULATOR_INITIALIZED = False
_SIMULATOR_LOCK = threading.Lock()


class Float32DefaultObs(DefaultObs):
    def build_obs(self, agents, state, shared_info):
        return {agent: np.ascontiguousarray(obs, dtype=np.float32)
                for agent, obs in super().build_obs(agents, state, shared_info).items()}


MAX_SPEED = 2300.0  # Rocket League supersonic threshold scale for normalization.


def shaping_rewards(agents, state, *, velocity_to_ball_weight: float, ball_to_goal_weight: float) -> dict:
    """Opt-in dense shaping from genuine physics state, normalized to [-1, 1] per component.

    velocity-to-ball rewards moving toward the ball; ball-to-goal rewards ball
    velocity toward the opponent's goal. Both are potential-style signals that
    must prove themselves in evaluation; with zero weights this returns zeros
    and the foundation reward is bit-identical.
    """
    result = {}
    ball_position = np.asarray(state.ball.position, dtype=np.float64)
    ball_velocity = np.asarray(state.ball.linear_velocity, dtype=np.float64)
    for agent in agents:
        car = state.cars[agent]
        position = np.asarray(car.physics.position, dtype=np.float64)
        velocity = np.asarray(car.physics.linear_velocity, dtype=np.float64)
        value = 0.0
        if velocity_to_ball_weight:
            direction = ball_position - position
            norm = float(np.linalg.norm(direction))
            if norm > 1e-9:
                value += velocity_to_ball_weight * float(np.dot(velocity, direction / norm)) / MAX_SPEED
        if ball_to_goal_weight:
            goal = np.array([0.0, 5120.0 if agent.startswith("blue") else -5120.0, 0.0])
            direction = goal - ball_position
            norm = float(np.linalg.norm(direction))
            if norm > 1e-9:
                value += ball_to_goal_weight * float(np.dot(ball_velocity, direction / norm)) / MAX_SPEED
        result[agent] = value
    return result


class AuditedReward(RewardFunction):
    """Use real per-transition touch semantics. Exploit risk: repeated low-value contacts."""

    def __init__(self, *, velocity_to_ball_weight: float = 0.0, ball_to_goal_weight: float = 0.0):
        self.goal, self.touch = GoalReward(), TouchReward()
        self.velocity_to_ball_weight = velocity_to_ball_weight
        self.ball_to_goal_weight = ball_to_goal_weight

    def reset(self, agents, initial_state, shared_info):
        self.goal.reset(agents, initial_state, shared_info)
        self.touch.reset(agents, initial_state, shared_info)
        shared_info.update(reward_goal_abs=0.0, reward_touch=0.0, reward_shaping=0.0)

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        goals = self.goal.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        touches = self.touch.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        shaping = shaping_rewards(agents, state, velocity_to_ball_weight=self.velocity_to_ball_weight,
                                  ball_to_goal_weight=self.ball_to_goal_weight)
        shared_info["reward_goal_abs"] = float(sum(abs(value) * 10 for value in goals.values()))
        shared_info["reward_touch"] = float(sum(touches.values()) * 0.1)
        shared_info["reward_shaping"] = float(sum(shaping.values()))
        return {agent: 10.0 * goals[agent] + 0.1 * touches[agent] + shaping[agent] for agent in agents}


class ScenarioMutator(StateMutator):
    """Varied genuine starting situations for ``mixed-scenarios-v1``.

    Duck-typed StateMutator (apply(state, shared_info)): per reset, one of
    kickoff / attacking / defensive / open-play setups, mirrored for both
    teams with seeded jitter. Kickoff-only remains the default environment.
    """

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.kickoff = KickoffMutator()

    def _place(self, state, ball, setups):
        state.ball.position = np.array(ball, dtype=np.float32)
        state.ball.linear_velocity = np.zeros(3, dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        for car in state.cars.values():
            position, yaw = setups["blue" if car.is_blue else "orange"]
            car.physics.position = np.array(position, dtype=np.float32)
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.physics.euler_angles = np.array([0, yaw, 0], dtype=np.float32)
            car.boost_amount = 33.3

    def apply(self, state, shared_info):
        choice = int(self.rng.integers(0, 4))
        if choice == 0:
            self.kickoff.apply(state, shared_info)
            return
        jitter = float(self.rng.uniform(-600, 600))
        if choice == 1:  # blue attacking: ball ahead of blue, orange retreating; mirrored roles via sides
            ball = [jitter, -1200.0, 93.15]
            setups = {"blue": ([jitter, -2200.0, 17], np.pi / 2), "orange": ([-jitter, 1800.0, 17], -np.pi / 2)}
        elif choice == 2:  # blue defensive: ball in blue half, blue between ball and own goal
            ball = [jitter, -2600.0, 93.15]
            setups = {"blue": ([jitter, -3400.0, 17], np.pi / 2), "orange": ([-jitter, -600.0, 17], -np.pi / 2)}
        else:  # open play: scattered mid-field possession
            ball = [jitter, float(self.rng.uniform(-1500, 1500)), 93.15]
            setups = {"blue": ([jitter - 900.0, ball[1] - 1400.0, 17], np.pi / 2),
                      "orange": ([-jitter + 900.0, ball[1] + 1400.0, 17], -np.pi / 2)}
        self._place(state, ball, setups)


class NonGoalTruncation(AnyCondition):
    def is_done(self, agents, state, shared_info):
        # At a simultaneous time limit and goal, true termination takes precedence.
        if state.goal_scored:
            return dict.fromkeys(agents, False)
        return super().is_done(agents, state, shared_info)


class EpisodeRLGym(RLGym):
    """Make the reset-after-done invariant explicit rather than stepping a terminal state."""

    def __init__(self, response_fence=None, **kwargs):
        super().__init__(**kwargs)
        self.needs_reset = True
        self.response_fence = response_fence
        self._first_reset = True

    def reset(self):
        obs = super().reset()
        if not self._first_reset and self.response_fence is not None:
            self.response_fence.acquire()
        self._first_reset = False
        self.needs_reset = False
        return obs

    def step(self, actions):
        if self.needs_reset:
            raise RuntimeError("Episode ended or has not started. Reset before stepping again.")
        result = super().step(actions)
        self.needs_reset = all(result[2][agent] or result[3][agent] for agent in self.agents)
        if any(not np.isfinite(obs).all() for obs in result[0].values()):
            raise RuntimeError("Non-finite simulation observation; stopping rather than training on invalid data.")
        if self.response_fence is not None:
            # Learn 2.0.0 signals the environment BEFORE draining its UDP response socket.
            # A fast response is otherwise discarded; the collector then stalls until a 5 s
            # keepalive. Do not publish this response until the real send call has returned.
            self.response_fence.acquire()
        return result


def initialize_simulator() -> Path:
    import rlgym.rocket_league.sim.rocketsim_engine as module
    import RocketSim as rsim

    meshes = Path(module.__file__).parent / "collision_meshes"
    if len(list((meshes / "soccar").glob("*.cmf"))) != 16:
        raise RuntimeError("Expected 16 packaged soccar meshes. Reinstall the locked rlgym-rocket-league 2.0.1 package.")
    # Do not depend on RocketSimEngine's blanket except/pass. RocketSim init is
    # process-global and rejects a second call; make our own successful path idempotent.
    global _SIMULATOR_INITIALIZED
    with _SIMULATOR_LOCK:
        if not _SIMULATOR_INITIALIZED:
            rsim.init(str(meshes))
            _SIMULATOR_INITIALIZED = True
    return meshes


def build_environment(config: TrainingConfig, response_fence=None) -> EpisodeRLGym:
    initialize_simulator()
    from rlgym.rocket_league.sim import RocketSimEngine

    scenario = (ScenarioMutator(config.seed) if config.environment_version == "mixed-scenarios-v1"
                else KickoffMutator())
    return EpisodeRLGym(
        response_fence=response_fence,
        state_mutator=MutatorSequence(FixedTeamSizeMutator(blue_size=1, orange_size=1), scenario),
        obs_builder=Float32DefaultObs(zero_padding=1),
        action_parser=RepeatAction(LookupTableAction(), repeats=config.tick_skip),
        reward_fn=AuditedReward(velocity_to_ball_weight=config.reward_velocity_to_ball_weight,
                                ball_to_goal_weight=config.reward_ball_to_goal_weight),
        termination_cond=GoalCondition(),
        truncation_cond=NonGoalTruncation(
            NoTouchTimeoutCondition(timeout_seconds=config.no_touch_timeout_seconds),
            TimeoutCondition(timeout_seconds=config.episode_timeout_seconds),
        ),
        transition_engine=RocketSimEngine(rlbot_delay=True),
        renderer=None,
    )


@dataclass
class EnvironmentFactory:
    config: TrainingConfig
    response_fence: object = None

    def __call__(self):
        return build_environment(self.config, self.response_fence)
