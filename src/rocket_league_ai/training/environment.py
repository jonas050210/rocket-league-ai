"""Published RLGym v2 / RocketSim APIs, fixed 1v1 headless environment with pro reward shaping."""

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
    def get_obs_space(self, agent):
        return "real", 92

    def build_obs(self, agents, state, shared_info):
        raw = super().build_obs(agents, state, shared_info)
        out = {}
        for agent, obs in raw.items():
            arr = np.ascontiguousarray(obs, dtype=np.float32)
            if arr.shape[0] > 92:
                # In 2v2 (team_size=2), DefaultObs produces [global(52), self(20), ally(20), enemy0(20), enemy1(20)] = 132.
                # Project to [global(52), self(20), nearest_enemy(20)] = 92 so 1v1 and 2v2 share checkpoints seamlessly.
                self_pos = arr[52:55]
                e0 = arr[92:112]
                e1 = arr[112:132] if arr.shape[0] >= 132 else e0
                d0 = float(np.sum((e0[0:3] - self_pos) ** 2))
                d1 = float(np.sum((e1[0:3] - self_pos) ** 2))
                nearest_enemy = e0 if d0 <= d1 else e1
                arr = np.ascontiguousarray(np.concatenate([arr[:72], nearest_enemy]), dtype=np.float32)
            out[agent] = arr
        return out


MAX_SPEED = 2300.0  # Rocket League supersonic threshold scale for normalization.


def shaping_rewards(
    agents,
    state,
    *,
    velocity_to_ball_weight: float = 0.0,
    ball_to_goal_weight: float = 0.0,
    boost_weight: float = 0.0,
    defensive_weight: float = 0.0,
    aerial_shot_weight: float = 0.0,
    mechanics_weight: float = 0.0,
) -> dict:
    """Opt-in dense shaping from genuine physics state, normalized to [-1, 1] per component.

    - velocity-to-ball: rewards moving toward the ball
    - ball-to-goal: rewards ball velocity toward the opponent's goal
    - boost: rewards collecting and maintaining boost reserves
    - defensive: rewards positioning between ball and own goal when under pressure
    - aerial_shot: rewards high-altitude ball contacts and fast shots on target
    - mechanics: rewards Speedflips (flip-cancel + supersonic approach), Fast Aerials, and clean 4-wheel recoveries
    With zero weights this returns zeros and the foundation reward is bit-identical.
    """
    result = {}
    ball_position = np.asarray(state.ball.position, dtype=np.float64)
    ball_velocity = np.asarray(state.ball.linear_velocity, dtype=np.float64)
    for agent in agents:
        car = state.cars[agent]
        is_blue = getattr(car, "is_blue", str(agent).startswith("blue"))
        position = np.asarray(car.physics.position, dtype=np.float64)
        velocity = np.asarray(car.physics.linear_velocity, dtype=np.float64)
        value = 0.0
        opp_goal = np.array([0.0, 5120.0 if is_blue else -5120.0, 0.0])
        own_goal = np.array([0.0, -5120.0 if is_blue else 5120.0, 0.0])

        if velocity_to_ball_weight:
            direction = ball_position - position
            norm = float(np.linalg.norm(direction))
            if norm > 1e-9:
                value += velocity_to_ball_weight * float(np.dot(velocity, direction / norm)) / MAX_SPEED

        if ball_to_goal_weight:
            direction = opp_goal - ball_position
            norm = float(np.linalg.norm(direction))
            if norm > 1e-9:
                value += ball_to_goal_weight * float(np.dot(ball_velocity, direction / norm)) / MAX_SPEED

        if boost_weight:
            raw_boost = float(getattr(car, "boost_amount", 0.0) or 0.0)
            norm_boost = raw_boost / 100.0 if raw_boost > 1.0 else raw_boost
            value += boost_weight * max(0.0, min(norm_boost, 1.0)) * 0.25

        if defensive_weight:
            # Reward staying between own goal and the ball when ball is in own half or moving toward own goal
            goal_to_ball = ball_position - own_goal
            goal_to_car = position - own_goal
            norm_gb = float(np.linalg.norm(goal_to_ball))
            norm_gc = float(np.linalg.norm(goal_to_car))
            if norm_gb > 1e-9 and norm_gc > 1e-9 and norm_gc <= norm_gb + 400.0:
                alignment = float(np.dot(goal_to_ball / norm_gb, goal_to_car / norm_gc))
                if alignment > 0:
                    value += defensive_weight * alignment * 0.3

        if aerial_shot_weight:
            touches = int(getattr(car, "ball_touches", 0) or 0)
            height_factor = max(0.0, min((float(ball_position[2]) - 93.0) / 800.0, 1.0))
            to_opp = opp_goal - ball_position
            norm_opp = float(np.linalg.norm(to_opp))
            shot_speed = (
                max(0.0, float(np.dot(ball_velocity, to_opp / norm_opp)) / MAX_SPEED)
                if norm_opp > 1e-9
                else 0.0
            )
            if touches > 0:
                value += aerial_shot_weight * (0.5 * height_factor + 0.5 * shot_speed)
            else:
                value += aerial_shot_weight * 0.15 * shot_speed

        if mechanics_weight:
            speed = float(np.linalg.norm(velocity))
            to_ball = ball_position - position
            dist_ball = float(np.linalg.norm(to_ball))
            speed_to_ball = float(np.dot(velocity, to_ball / dist_ball)) / MAX_SPEED if dist_ball > 1e-9 else 0.0
            is_flipping = bool(getattr(car, "is_flipping", False) or getattr(car, "has_flipped", False))
            is_supersonic = bool(getattr(car, "is_supersonic", False) or speed >= 2150.0)
            on_ground = bool(getattr(car, "on_ground", True))
            fwd = getattr(car.physics, "forward", np.array([0.0, 1.0, 0.0]))
            up = getattr(car.physics, "up", np.array([0.0, 0.0, 1.0]))
            mech_bonus = 0.0
            # 1) Speedflip signature: flipping + high forward speed toward ball + flat pitch cancel
            if is_flipping and speed_to_ball > 0.55:
                pitch_flatness = max(0.0, 1.0 - abs(float(fwd[2])))
                mech_bonus += 0.45 * speed_to_ball * pitch_flatness
            if is_supersonic and speed_to_ball > 0.65:
                mech_bonus += 0.25 * speed_to_ball
            # 2) Fast Aerial climb toward high ball
            if not on_ground and float(ball_position[2]) > 240.0 and float(velocity[2]) > 220.0 and speed_to_ball > 0.3:
                mech_bonus += 0.35 * min(1.0, float(velocity[2]) / 900.0)
            # 3) Clean 4-wheel recovery / wave-dash landing readiness
            if not on_ground and float(position[2]) < 110.0 and float(up[2]) > 0.85 and speed > 800.0:
                mech_bonus += 0.2 * float(up[2])
            value += mechanics_weight * mech_bonus

        result[agent] = value
    return result


class AuditedReward(RewardFunction):
    """Use real per-transition touch semantics plus opt-in v2/v3 pro shaping."""

    def __init__(
        self,
        *,
        velocity_to_ball_weight: float = 0.0,
        ball_to_goal_weight: float = 0.0,
        boost_weight: float = 0.0,
        defensive_weight: float = 0.0,
        aerial_shot_weight: float = 0.0,
        mechanics_weight: float = 0.0,
    ):
        self.goal, self.touch = GoalReward(), TouchReward()
        self.velocity_to_ball_weight = velocity_to_ball_weight
        self.ball_to_goal_weight = ball_to_goal_weight
        self.boost_weight = boost_weight
        self.defensive_weight = defensive_weight
        self.aerial_shot_weight = aerial_shot_weight
        self.mechanics_weight = mechanics_weight

    def reset(self, agents, initial_state, shared_info):
        self.goal.reset(agents, initial_state, shared_info)
        self.touch.reset(agents, initial_state, shared_info)
        shared_info.update(reward_goal_abs=0.0, reward_touch=0.0, reward_shaping=0.0)

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        goals = self.goal.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        touches = self.touch.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
        shaping = shaping_rewards(
            agents,
            state,
            velocity_to_ball_weight=self.velocity_to_ball_weight,
            ball_to_goal_weight=self.ball_to_goal_weight,
            boost_weight=self.boost_weight,
            defensive_weight=self.defensive_weight,
            aerial_shot_weight=self.aerial_shot_weight,
            mechanics_weight=self.mechanics_weight,
        )
        shared_info["reward_goal_abs"] = float(sum(abs(value) * 10 for value in goals.values()))
        shared_info["reward_touch"] = float(sum(touches.values()) * 0.1)
        shared_info["reward_shaping"] = float(sum(shaping.values()))
        return {agent: 10.0 * goals[agent] + 0.1 * touches[agent] + shaping[agent] for agent in agents}


class ScenarioMutator(StateMutator):
    """Varied genuine starting situations for ``mixed-scenarios-v1``.

    Supports 10 game & mechanics drills (including Speedflip kickoff, Fast Aerial,
    Wall-to-Air Dribble, 180 Half-Flip Recovery, Defensive Save, Shadow Defense)
    and supports both 1v1 (2 bots) and 2v2 (4 bots) plus Boost modes.
    """

    def __init__(self, seed: int, *, randomize_boost: bool = False, boost_mode: str = "standard"):
        self.rng = np.random.default_rng(seed)
        self.kickoff = KickoffMutator()
        self.randomize_boost = randomize_boost
        self.boost_mode = boost_mode

    def _initial_boost(self) -> float:
        if self.boost_mode == "unlimited":
            return 100.0
        if self.boost_mode == "starved":
            return float(self.rng.uniform(0.0, 22.0))
        if self.randomize_boost:
            return float(self.rng.uniform(12.0, 100.0))
        return 33.3

    def _place(self, state, ball, setups, ball_vel=(0.0, 0.0, 0.0), car_vels=None):
        state.ball.position = np.array(ball, dtype=np.float32)
        state.ball.linear_velocity = np.array(ball_vel, dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        blue_idx = 0
        orange_idx = 0
        for car in state.cars.values():
            team_key = "blue" if car.is_blue else "orange"
            idx = blue_idx if car.is_blue else orange_idx
            if car.is_blue:
                blue_idx += 1
            else:
                orange_idx += 1
            position, yaw = setups[team_key]
            pos = np.array(position, dtype=np.float32)
            if idx > 0:
                pos[0] = float(np.clip(pos[0] + (600.0 if pos[0] <= 0 else -600.0), -3400.0, 3400.0))
                pos[1] = float(np.clip(pos[1] + (-500.0 if car.is_blue else 500.0), -4600.0, 4600.0))
            car.physics.position = pos
            vel = (car_vels or {}).get(team_key, (0.0, 0.0, 0.0))
            car.physics.linear_velocity = np.array(vel, dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.physics.euler_angles = np.array([0, yaw, 0], dtype=np.float32)
            car.boost_amount = self._initial_boost()

    def apply(self, state, shared_info):
        choice = int(self.rng.integers(0, 10))
        if choice == 0:
            self.kickoff.apply(state, shared_info)
            if self.boost_mode != "standard":
                for car in state.cars.values():
                    car.boost_amount = self._initial_boost()
            return
        jitter = float(self.rng.uniform(-600, 600))
        if choice == 1:  # attacking rolling shot setup
            ball = [jitter, -1200.0, 93.15]
            setups = {"blue": ([jitter, -2200.0, 17], np.pi / 2), "orange": ([-jitter, 1800.0, 17], -np.pi / 2)}
            self._place(state, ball, setups, ball_vel=(0.0, 450.0, 0.0),
                        car_vels={"blue": (0.0, 650.0, 0.0), "orange": (0.0, -300.0, 0.0)})
        elif choice == 2:  # high-pressure defensive save setup
            ball = [jitter, -2600.0, 120.0]
            setups = {"blue": ([jitter * 0.5, -3600.0, 17], np.pi / 2), "orange": ([-jitter, -900.0, 17], -np.pi / 2)}
            self._place(state, ball, setups, ball_vel=(-jitter * 0.2, -950.0, 120.0),
                        car_vels={"blue": (0.0, 350.0, 0.0), "orange": (0.0, -600.0, 0.0)})
        elif choice == 3:  # open play mid-field 50/50 possession
            ball = [jitter, float(self.rng.uniform(-1200, 1200)), 93.15]
            setups = {"blue": ([jitter - 800.0, ball[1] - 1300.0, 17], np.pi / 2),
                      "orange": ([-jitter + 800.0, ball[1] + 1300.0, 17], -np.pi / 2)}
            self._place(state, ball, setups, ball_vel=(float(self.rng.uniform(-300, 300)), 0.0, 0.0),
                        car_vels={"blue": (200.0, 600.0, 0.0), "orange": (-200.0, -600.0, 0.0)})
        elif choice == 4:  # high bouncing ball / Fast Aerial & power-shot opportunity
            ball = [jitter, float(self.rng.uniform(-900, 900)), 460.0]
            setups = {"blue": ([jitter, -1800.0, 17], np.pi / 2),
                      "orange": ([-jitter, 1800.0, 17], -np.pi / 2)}
            self._place(state, ball, setups, ball_vel=(0.0, 0.0, 320.0),
                        car_vels={"blue": (0.0, 650.0, 0.0), "orange": (0.0, -650.0, 0.0)})
        elif choice == 5:  # corner / side-wall bounce read
            side = 1.0 if self.rng.random() > 0.5 else -1.0
            ball = [side * 3200.0, float(self.rng.uniform(-1500, 1500)), 180.0]
            setups = {"blue": ([side * 2200.0, ball[1] - 1200.0, 17], np.pi / 2),
                      "orange": ([side * 1800.0, ball[1] + 1400.0, 17], -np.pi / 2)}
            self._place(state, ball, setups, ball_vel=(-side * 900.0, 500.0, 180.0),
                        car_vels={"blue": (0.0, 600.0, 0.0), "orange": (0.0, -500.0, 0.0)})
        elif choice == 6:  # shadow defense 1v1 breakaway
            ball = [jitter * 0.5, -800.0, 93.15]
            setups = {"blue": ([jitter * 0.3, -2300.0, 17], -np.pi / 2),
                      "orange": ([jitter * 0.5, -200.0, 17], -np.pi / 2)}
            self._place(state, ball, setups, ball_vel=(0.0, -850.0, 0.0),
                        car_vels={"blue": (0.0, -750.0, 0.0), "orange": (0.0, -900.0, 0.0)})
        elif choice == 7:  # dribble / catch & flick setup
            ball = [jitter, -600.0, 240.0]
            setups = {"blue": ([jitter, -950.0, 17], np.pi / 2),
                      "orange": ([-jitter * 0.5, 2200.0, 17], -np.pi / 2)}
            self._place(state, ball, setups, ball_vel=(0.0, 550.0, -80.0),
                        car_vels={"blue": (0.0, 550.0, 0.0), "orange": (0.0, -300.0, 0.0)})
        elif choice == 8:  # Speedflip diagonal kickoff drill
            side = 1.0 if self.rng.random() > 0.5 else -1.0
            ball = [0.0, 0.0, 93.15]
            diag_yaw_blue = np.pi / 2 + (-side * np.pi / 4)
            diag_yaw_orange = -np.pi / 2 + (side * np.pi / 4)
            setups = {
                "blue": ([side * 2048.0, -2560.0, 17], diag_yaw_blue),
                "orange": ([-side * 2048.0, 2560.0, 17], diag_yaw_orange),
            }
            self._place(state, ball, setups, ball_vel=(0.0, 0.0, 0.0),
                        car_vels={"blue": (-side * 450.0, 550.0, 0.0), "orange": (side * 450.0, -550.0, 0.0)})
        else:  # Wall-to-air dribble / Half-flip recovery drill
            side = 1.0 if self.rng.random() > 0.5 else -1.0
            ball = [side * 3750.0, 0.0, 220.0]
            setups = {
                "blue": ([side * 3300.0, -500.0, 17], np.pi / 4 if side > 0 else 3 * np.pi / 4),
                "orange": ([0.0, 2400.0, 17], np.pi / 2),  # facing away -> needs half-flip / turn recovery!
            }
            self._place(state, ball, setups, ball_vel=(side * 350.0, 400.0, 320.0),
                        car_vels={"blue": (side * 500.0, 500.0, 0.0), "orange": (0.0, 200.0, 0.0)})


class NonGoalTruncation(AnyCondition):
    def is_done(self, agents, state, shared_info):
        # At a simultaneous time limit and goal, true termination takes precedence.
        if state.goal_scored:
            return dict.fromkeys(agents, False)
        return super().is_done(agents, state, shared_info)


class EpisodeRLGym(RLGym):
    """Make the reset-after-done invariant explicit rather than stepping a terminal state."""

    def __init__(self, response_fence=None, boost_mode: str = "standard", **kwargs):
        super().__init__(**kwargs)
        self.needs_reset = True
        self.response_fence = response_fence
        self.boost_mode = boost_mode
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
        if self.boost_mode == "unlimited":
            state = self.state
            for car in state.cars.values():
                car.boost_amount = 100.0
            self.set_state(state)
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

    scenario = (
        ScenarioMutator(config.seed, randomize_boost=config.self_play_enabled, boost_mode=config.boost_mode)
        if config.environment_version == "mixed-scenarios-v1"
        else KickoffMutator()
    )
    return EpisodeRLGym(
        response_fence=response_fence,
        boost_mode=config.boost_mode,
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=config.team_size, orange_size=config.team_size),
            scenario,
        ),
        obs_builder=Float32DefaultObs(zero_padding=1),
        action_parser=RepeatAction(LookupTableAction(), repeats=config.tick_skip),
        reward_fn=AuditedReward(
            velocity_to_ball_weight=config.reward_velocity_to_ball_weight,
            ball_to_goal_weight=config.reward_ball_to_goal_weight,
            boost_weight=config.reward_boost_weight,
            defensive_weight=config.reward_defensive_weight,
            aerial_shot_weight=config.reward_aerial_shot_weight,
            mechanics_weight=config.reward_mechanics_weight,
        ),
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
