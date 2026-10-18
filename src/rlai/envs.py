"""Headless simulation environments for all supported modes.

RocketSim implements Soccar *and* Heatseeker natively
(``RocketSim::GameMode::HEATSEEKER`` with ball homing, speed-up per touch,
wall bounce redirection and goal scoring).  We only need to:

* select the arena game mode,
* reproduce Heatseeker's kickoff state and unlimited boost, and
* correct RLGym's coarse goal check (it only looks at ``|y|`` and would count
  balls outside the goal mouth).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import RocketSim as rsim
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.sim.rocketsim_engine import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence

from .modes import ModeSpec
from .obs import OBS_VERSION, SoccerObsBuilder
from .rewards import RewardWeights, SoccerReward

GOAL_HALF_WIDTH = 892.755
GOAL_HEIGHT = 642.775
BALL_RADIUS = 91.25

#: Heatseeker kickoff constants (RocketSim RLConst, mirrored for orange).
HEATSEEKER_BALL_POS = np.array([-1000.0, -2220.0, 92.75], dtype=np.float32)
HEATSEEKER_BALL_VEL = np.array([0.0, -65.0, 650.0], dtype=np.float32)
HEATSEEKER_SPAWNS = (
    np.array([-1000.0, -4620.0, 17.0], dtype=np.float32),
    np.array([1000.0, -4620.0, 17.0], dtype=np.float32),
    np.array([-2000.0, -4620.0, 17.0], dtype=np.float32),
    np.array([2000.0, -4620.0, 17.0], dtype=np.float32),
)


class ModeKickoffMutator(KickoffMutator):
    """Heatseeker kickoff: off-centre ball with velocity, 4 spawn points."""

    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        state.config.boost_consumption = 0.0  # unlimited boost
        state.config.dodge_deadzone = 0.5
        state.ball.position = HEATSEEKER_BALL_POS.copy()
        state.ball.linear_velocity = HEATSEEKER_BALL_VEL.copy()
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)

        indices = list(range(4))
        random.shuffle(indices)
        blue_idx = 0
        orange_idx = 0
        for car in state.cars.values():
            if car.team_num == BLUE_TEAM:
                pos = HEATSEEKER_SPAWNS[indices[blue_idx]].copy()
                yaw = np.pi / 2
                blue_idx += 1
            else:
                pos = HEATSEEKER_SPAWNS[indices[orange_idx]].copy()
                pos[1] *= -1
                yaw = -np.pi / 2
                orange_idx += 1
            car.physics.position = pos
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.physics.euler_angles = np.array([0.0, yaw, 0.0], dtype=np.float32)
            car.boost_amount = 100.0


class ModeRocketSimEngine(RocketSimEngine):
    """RocketSimEngine with an explicit game mode and accurate goals."""

    def __init__(self, spec: ModeSpec, rlbot_delay: bool = True):
        self.spec = spec
        self._rsim_mode = rsim.GameMode.HEATSEEKER if spec.heatseeker else rsim.GameMode.SOCCAR
        super().__init__(rlbot_delay)
        self._arena = rsim.Arena(self._rsim_mode)
        self._arena.set_ball_touch_callback(self._ball_touch_callback)

    def _get_state(self) -> GameState:
        state = super()._get_state()
        ball = state.ball.position
        inside_mouth = abs(float(ball[0])) <= GOAL_HALF_WIDTH and float(ball[2]) <= GOAL_HEIGHT
        state.goal_scored = bool(abs(float(ball[1])) > 5215.5 and inside_mouth)
        return state


def build_rlgym_env(
    spec: ModeSpec,
    *,
    reward_weights: RewardWeights | None = None,
    episode_seconds: float | None = None,
    tick_skip: int = 8,
    rlbot_delay: bool = True,
    obs_version: int = OBS_VERSION,
) -> RLGym:
    kickoff = ModeKickoffMutator() if spec.heatseeker else KickoffMutator()
    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(spec.team_size, spec.team_size), kickoff
        ),
        obs_builder=SoccerObsBuilder(spec, obs_version=obs_version),
        action_parser=RepeatAction(LookupTableAction(), tick_skip),
        reward_fn=SoccerReward(spec, reward_weights),
        transition_engine=ModeRocketSimEngine(spec, rlbot_delay=rlbot_delay),
        termination_cond=GoalCondition(),
        truncation_cond=TimeoutCondition(
            episode_seconds if episode_seconds is not None else spec.default_episode_seconds
        ),
    )


@dataclass(slots=True)
class StepInfo:
    tick_count: int
    time_remaining: float
    score_blue: int
    score_orange: int
    goal_scored: bool
    scoring_team: int | None
    touches: dict[str, int]
    #: reward components accumulated for the current episode (see rewards.py)
    reward_components: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class StepResult:
    obs: dict[str, np.ndarray]
    rewards: dict[str, float]
    done: bool
    truncated: bool
    info: StepInfo


class MatchEnv:
    """Single-process environment wrapper.

    The learning team is always BLUE (index 0); the opponent controls ORANGE
    through a policy object.  The wrapper owns score keeping, the episode
    clock exposed to the observation and the per-episode statistics.
    """

    def __init__(
        self,
        spec: ModeSpec,
        opponent,
        *,
        seed: int = 0,
        tick_skip: int = 8,
        reward_weights: RewardWeights | None = None,
        episode_seconds: float | None = None,
        rlbot_delay: bool = True,
        obs_version: int = OBS_VERSION,
        policy_team: int = BLUE_TEAM,
    ):
        self.spec = spec
        self.obs_version = int(obs_version)
        self.opponent = opponent
        #: Team the *learning* policy controls.  Evaluation uses this to play
        #: a side-balanced series; training normally stays on blue.
        self.policy_team = int(policy_team)
        self.seed = int(seed)
        self.tick_skip = tick_skip
        self.episode_seconds = (
            episode_seconds if episode_seconds is not None else spec.default_episode_seconds
        )
        self.env = build_rlgym_env(
            spec,
            reward_weights=reward_weights,
            episode_seconds=self.episode_seconds,
            tick_skip=tick_skip,
            rlbot_delay=rlbot_delay,
            obs_version=self.obs_version,
        )
        self.episode_index = 0
        self.start_tick = 0
        self.score_blue = 0
        self.score_orange = 0
        self.episode_goals_for = 0
        self.episode_goals_against = 0
        self.last_goal_team: int | None = None
        self._observations: dict[str, np.ndarray] = {}
        self._touch_total = -1
        self._touches: dict[str, int] = {}

    # -- helper properties -------------------------------------------------
    @property
    def state(self) -> GameState:
        return self.env.state

    @property
    def blue_agents(self) -> list[str]:
        return [a for a in self.env.agents if self.env.state.cars[a].team_num == BLUE_TEAM]

    @property
    def orange_agents(self) -> list[str]:
        return [a for a in self.env.agents if self.env.state.cars[a].team_num == ORANGE_TEAM]

    @property
    def policy_agents(self) -> list[str]:
        """Cars controlled by the learning policy (``policy_team``)."""
        return self.blue_agents if self.policy_team == BLUE_TEAM else self.orange_agents

    @property
    def opponent_agents(self) -> list[str]:
        """Cars controlled by the opponent policy."""
        return self.orange_agents if self.policy_team == BLUE_TEAM else self.blue_agents

    @property
    def score_for(self) -> int:
        """Goals scored by the policy team in the current episode."""
        return self.score_blue if self.policy_team == BLUE_TEAM else self.score_orange

    @property
    def score_against(self) -> int:
        """Goals conceded by the policy team in the current episode."""
        return self.score_orange if self.policy_team == BLUE_TEAM else self.score_blue

    @property
    def reward_stats(self) -> dict[str, float]:
        """Per-episode reward components accumulated by the reward function."""
        return dict(self.env.shared_info.get("reward_stats", {}) or {})

    # -- lifecycle ---------------------------------------------------------
    def reset(self, *, episode_seed: int | None = None) -> dict[str, np.ndarray]:
        self.episode_index += 1
        seed = self.seed + self.episode_index if episode_seed is None else episode_seed
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        obs = self.env.reset()
        self.start_tick = self.env.state.tick_count
        self.score_blue = 0
        self.score_orange = 0
        self.episode_goals_for = 0
        self.episode_goals_against = 0
        self.last_goal_team = None
        self._touch_total = -1
        self._observations = obs
        self.opponent.reset(self)
        self._sync_clock()
        return obs

    def _touch_cache(self, state) -> dict[str, int]:
        """Reuse the touch dict unless a ball touch actually happened."""
        total = sum(int(state.cars[a].ball_touches) for a in state.cars)
        if total != getattr(self, "_touch_total", -1):
            self._touch_total = total
            self._touches = {a: int(state.cars[a].ball_touches) for a in state.cars}
        return self._touches

    def _sync_clock(self) -> None:
        elapsed = (self.env.state.tick_count - self.start_tick) / 120.0
        remaining = max(self.episode_seconds - elapsed, 0.0)
        self.env.shared_info["time_remaining"] = remaining
        # always from the learning policy's point of view, whatever side it is
        # playing on: a positive value means "we are ahead".
        self.env.shared_info["score_diff"] = self.score_for - self.score_against

    def step(self, actions: dict[str, int]) -> StepResult:
        if not self._observations:
            raise RuntimeError("reset() must be called before step()")
        all_actions: dict[str, np.ndarray] = {
            agent: np.asarray([int(action)], dtype=np.int64) for agent, action in actions.items()
        }
        opponent_actions = self.opponent.act(self, self._observations, self.opponent_agents)
        all_actions.update(
            {
                agent: np.asarray([int(action)], dtype=np.int64)
                for agent, action in opponent_actions.items()
            }
        )

        obs, rewards, terminated, truncated = self.env.step(all_actions)
        self._observations = obs

        for agent, action in all_actions.items():
            self.env.shared_info["last_action"][agent] = int(np.asarray(action).reshape(-1)[0])

        state = self.env.state
        goal = bool(state.goal_scored)
        scoring_team = state.scoring_team if goal else None
        if goal:
            self.last_goal_team = scoring_team
            if scoring_team == BLUE_TEAM:
                self.score_blue += 1
                self.episode_goals_for += 1
            elif scoring_team == ORANGE_TEAM:
                self.score_orange += 1
                self.episode_goals_against += 1

        done = any(bool(terminated.get(a, False)) for a in self.policy_agents)
        trunc = any(bool(truncated.get(a, False)) for a in self.policy_agents)
        self._sync_clock()

        info = StepInfo(
            tick_count=state.tick_count,
            time_remaining=float(self.env.shared_info["time_remaining"]),
            score_blue=self.score_blue,
            score_orange=self.score_orange,
            goal_scored=goal,
            scoring_team=scoring_team,
            touches=self._touch_cache(state),
            reward_components=self.reward_stats,
        )
        return StepResult(
            obs=obs,
            rewards={a: float(rewards.get(a, 0.0)) for a in self.policy_agents},
            done=done,
            truncated=trunc,
            info=info,
        )
