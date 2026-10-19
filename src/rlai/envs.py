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


#: episode starting-state distributions (see :class:`ResetMutator`)
RESET_MODES = ("kickoff", "drill", "mixed")


class ResetMutator(KickoffMutator):
    """Kickoff or randomized "drill" starts, chosen per reset.

    Every pre-overhaul episode started from a kickoff.  A weak policy then
    spends its life in the small set of states reachable from kickoffs and
    almost never experiences the states that matter (ball close, ball near
    the goal, shooting angles), so goal credit has nothing to attach to.

    ``drill`` starts place the ball at a random field position (sometimes
    rolling) and every car on its own defensive side of the ball, 700-2600 uu
    away and facing it: the agent practices the last two seconds of an attack
    thousands of times instead of hoping to stumble into them.  ``mixed``
    flips a coin per episode.  The mode is read from
    ``shared_info['reset_mode']`` on every reset, so a curriculum stage can
    change it without rebuilding the arena.  Randomness comes from the
    ``random`` module, which :meth:`MatchEnv.reset` seeds deterministically.
    """

    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        mode = str(shared_info.get("reset_mode", "kickoff"))
        if mode == "mixed":
            mode = "drill" if random.random() < 0.5 else "kickoff"
        if mode != "drill":
            super().apply(state, shared_info)
            return
        ball_x = random.uniform(-2500.0, 2500.0)
        ball_y = random.uniform(-3500.0, 3500.0)
        state.ball.position = np.array([ball_x, ball_y, 93.0], dtype=np.float32)
        if random.random() < 0.5:
            angle = random.uniform(0.0, 2.0 * np.pi)
            speed = random.uniform(0.0, 600.0)
            state.ball.linear_velocity = np.array(
                [speed * np.cos(angle), speed * np.sin(angle), 0.0], dtype=np.float32
            )
        else:
            state.ball.linear_velocity = np.zeros(3, dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        for car in state.cars.values():
            # Blue attacks +y (orange goal), orange attacks -y.  "Behind the
            # ball" = on the side of the car's own goal, so driving at the
            # ball also drives it toward the target goal.
            attack = 1.0 if car.team_num == BLUE_TEAM else -1.0
            distance = random.uniform(700.0, 2600.0)
            car_x = float(np.clip(ball_x + random.uniform(-900.0, 900.0), -3800.0, 3800.0))
            car_y = float(np.clip(ball_y - attack * distance, -4800.0, 4800.0))
            car.physics.position = np.array([car_x, car_y, 17.0], dtype=np.float32)
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            yaw = float(np.arctan2(ball_y - car_y, ball_x - car_x))
            car.physics.euler_angles = np.array([0.0, yaw, 0.0], dtype=np.float32)
            car.boost_amount = random.uniform(20.0, 100.0)


def build_rlgym_env(
    spec: ModeSpec,
    *,
    reward_weights: RewardWeights | None = None,
    episode_seconds: float | None = None,
    tick_skip: int = 8,
    rlbot_delay: bool = True,
    obs_version: int = OBS_VERSION,
) -> RLGym:
    kickoff = ModeKickoffMutator() if spec.heatseeker else ResetMutator()
    seconds = episode_seconds if episode_seconds is not None else spec.default_episode_seconds
    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(spec.team_size, spec.team_size), kickoff
        ),
        obs_builder=SoccerObsBuilder(spec, obs_version=obs_version, episode_seconds=seconds),
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
    #: reward components accumulated for the current episode (see rewards.py);
    #: these are the *learning team's* components only
    reward_components: dict[str, float] = field(default_factory=dict)
    #: event counts (touches, shots, defensive touches, goals) of the
    #: learning team for the current episode
    reward_events: dict[str, int] = field(default_factory=dict)


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
        reset_mode: str = "kickoff",
    ):
        if reset_mode not in RESET_MODES:
            raise ValueError(f"reset_mode must be one of {RESET_MODES}, got {reset_mode!r}")
        #: starting-state distribution, read by the reset mutator per episode
        self.reset_mode = reset_mode
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
        #: per-episode accumulators owned by the env (not by a rollout), so an
        #: episode that spans two rollout collections still reports its full
        #: return and length instead of being silently dropped.
        self.episode_reward_total: dict[str, float] = {}
        self.episode_decisions = 0

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
    def current_obs(self) -> dict[str, np.ndarray]:
        """Observations of the current (possibly unfinished) episode."""
        return self._observations

    @property
    def reward_stats(self) -> dict[str, float]:
        """Per-episode reward components of the *learning team only*.

        The reward function tracks every agent; summing all of them (the old
        behaviour) mixed the opponent's shaping into the policy's reported
        components, so the numbers described neither team.
        """
        by_agent = self.env.shared_info.get("reward_stats_by_agent") or {}
        if not by_agent:
            return dict(self.env.shared_info.get("reward_stats", {}) or {})
        totals: dict[str, float] = {}
        for agent in self.policy_agents:
            for key, value in (by_agent.get(agent) or {}).items():
                if isinstance(value, float):
                    totals[key] = totals.get(key, 0.0) + value
        return totals

    @property
    def reward_events(self) -> dict[str, int]:
        """Per-episode event counts (touches/shots/…) of the learning team."""
        by_agent = self.env.shared_info.get("reward_stats_by_agent") or {}
        totals: dict[str, int] = {}
        for agent in self.policy_agents:
            for key, value in (by_agent.get(agent) or {}).items():
                if isinstance(value, int):
                    totals[key] = totals.get(key, 0) + value
        return totals

    # -- lifecycle ---------------------------------------------------------
    def reset(self, *, episode_seed: int | None = None) -> dict[str, np.ndarray]:
        self.episode_index += 1
        seed = self.seed + self.episode_index if episode_seed is None else episode_seed
        # RLGym's kickoff mutator draws from the *global* ``random`` module, so
        # a deterministic kickoff needs the global seed set.  The global state
        # is restored right afterwards: the trainer keeps its own Python RNG
        # state in the checkpoint, and an evaluation in the trainer process
        # must not silently rewrite it.
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            # the reset mutator reads the starting-state distribution here
            self.env.shared_info["reset_mode"] = self.reset_mode
            obs = self.env.reset()
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        self.start_tick = self.env.state.tick_count
        self.score_blue = 0
        self.score_orange = 0
        self.episode_goals_for = 0
        self.episode_goals_against = 0
        self.episode_reward_total = {}
        self.episode_decisions = 0
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

    def close(self) -> None:
        """Release the RocketSim arena.

        A reused arena keeps its memory footprint between games, so the
        evaluation loop closes them explicitly instead of waiting for the
        garbage collector.
        """
        close = getattr(self.env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - defensive
                pass

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

        policy_rewards = {a: float(rewards.get(a, 0.0)) for a in self.policy_agents}
        for agent, value in policy_rewards.items():
            self.episode_reward_total[agent] = self.episode_reward_total.get(agent, 0.0) + value
        self.episode_decisions += 1

        info = StepInfo(
            tick_count=state.tick_count,
            time_remaining=float(self.env.shared_info["time_remaining"]),
            score_blue=self.score_blue,
            score_orange=self.score_orange,
            goal_scored=goal,
            scoring_team=scoring_team,
            touches=self._touch_cache(state),
            reward_components=self.reward_stats,
            reward_events=self.reward_events,
        )
        return StepResult(
            obs=obs,
            rewards=policy_rewards,
            done=done,
            truncated=trunc,
            info=info,
        )
