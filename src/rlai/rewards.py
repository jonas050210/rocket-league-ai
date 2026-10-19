"""Configurable reward shaping.

Design (rewritten after the 2M-step stall analysis, see
``docs/LEARNING_OVERHAUL.md``):

* **Task first.**  Goals (±10) dominate, and the strongest shaping signal is
  *ball progress toward the opponent goal* (``ball_progress``) - the actual
  objective of football.  The previous implementation had no ball-to-goal
  signal at all; its dominant terms paid the car for *facing* the ball and
  *being near* it, so a near-optimal policy parked, faced the ball and
  almost never touched it (~0.06 touches/episode over 6 816 episodes).
* **Potentials, not states.**  Every dense term is a difference of a
  potential (car→ball distance, ball→goal distance).  A potential telescopes
  over an episode, so oscillation, circling and parking net ~zero and -
  crucially - scoring early no longer forfeits a stream of per-step bonuses
  (with the old state bonuses a goal at step 50 gave up to ~8.75 of
  remaining "alignment/approach" income for a +10 goal).
* **Events are directional.**  A touch pays most when it sends the ball
  toward the opponent goal; a hard touch in the attacking half pays an
  additional ``shot`` bonus; a touch that stops a ball heading for the own
  goal pays ``defense``.  Touches toward the own goal pay almost nothing.
* **Legacy terms.**  ``alignment`` and ``approach`` are kept (default **0**)
  so a pre-overhaul configuration still reproduces its old behaviour on an
  exact ``continue`` resume.  They are per-step state bonuses and are
  farmable; do not raise them.

Every component *and* event count is tracked per agent in
``shared_info['reward_stats_by_agent']``; :class:`rlai.envs.MatchEnv`
aggregates the learning team's share only.  (The old code summed both teams
into one dict, so the reported component means described nobody's reward.)
``shared_info['reward_stats']`` still holds the all-agent sum for
compatibility.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_GOAL_CENTER, BLUE_GOAL_CENTER

from .modes import ModeSpec

_COMPONENTS = (
    "goal",
    "concede",
    "touch",
    "shot",
    "ball_progress",
    "progress",
    "alignment",
    "approach",
    "boost",
    "defense",
    "bump",
)

#: event counters tracked next to the (monetary) components.  ``shots`` are
#: on-target attempts (trajectory projected inside the posts);
#: ``shots_off_target`` are hard goal-ward touches that would miss - the
#: ratio is the policy's shooting accuracy.
_EVENT_KEYS = ("touches", "shots", "shots_off_target", "defensive_touches", "goals")

#: ball speed toward the goal (uu/s) above which a touch counts as a shot
SHOT_SPEED_THRESHOLD = 800.0
#: ball must be at most this far (uu) from the opponent goal for a shot bonus
SHOT_DISTANCE_THRESHOLD = 6500.0
#: goal mouth (uu); mirrored from envs so "on target" means the same here
GOAL_HALF_WIDTH = 892.755
GOAL_HEIGHT = 642.775
#: Rocket League gravity (uu/s^2), for the shot trajectory projection
BALL_GRAVITY = 650.0
#: ball speed toward the own goal (uu/s) that makes a touch "defensive"
DEFENSE_SPEED_THRESHOLD = 500.0
#: ball must be at most this far (uu) from the own goal for a defense bonus
DEFENSE_DISTANCE_THRESHOLD = 5500.0
#: decision steps between two *paid* touch events per car.  Contacts are
#: always counted; paying every micro-contact would reward pinning or
#: tapping the ball in place instead of playing it somewhere useful.
TOUCH_COOLDOWN_STEPS = 4

SUPERSONIC_SPEED = 2300.0


def shot_on_target(ball_pos, ball_vel, goal) -> bool:
    """True when the ball's straight-line flight crosses inside the posts.

    Projects to the goal line (with gravity on z, ignoring bounces): a hard
    touch "toward the goal centre" that is actually heading two metres wide
    must not earn a shooting bonus - accuracy is the skill being taught.
    """
    vy = float(ball_vel[1])
    dy = float(goal[1]) - float(ball_pos[1])
    if vy == 0.0 or (vy > 0.0) != (dy > 0.0):
        return False
    t = dy / vy
    if t <= 0.0 or t > 4.0:
        return False
    x = float(ball_pos[0]) + float(ball_vel[0]) * t
    z = float(ball_pos[2]) + float(ball_vel[2]) * t - 0.5 * BALL_GRAVITY * t * t
    return abs(x) <= GOAL_HALF_WIDTH and z <= GOAL_HEIGHT


def _toward(velocity, target, origin) -> float:
    """Component of ``velocity`` pointing from ``origin`` toward ``target`` (uu/s)."""
    dx = float(target[0]) - float(origin[0])
    dy = float(target[1]) - float(origin[1])
    dz = float(target[2]) - float(origin[2])
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm < 1e-6:
        return 0.0
    return (float(velocity[0]) * dx + float(velocity[1]) * dy + float(velocity[2]) * dz) / norm


def _distance(a, b) -> float:
    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    )


@dataclass(slots=True)
class RewardWeights:
    goal: float = 10.0
    concede: float = 10.0
    touch: float = 0.4
    #: potential difference of the ball→opponent-goal distance (per 1000 uu).
    #: Bringing the ball the full field is worth roughly one goal; ordinary
    #: episodes collect a small fraction of that.
    ball_progress: float = 1.0
    #: potential difference of the car→ball distance (per 1000 uu).  Early
    #: "get to the ball" gradient only; deliberately weaker than the task.
    progress: float = 0.35
    #: hard touch toward the goal in the attacking half (event).
    shot: float = 1.0
    #: touch that stops/reverses a ball heading for the own goal (event).
    defense: float = 0.5
    # --- legacy state bonuses (farmable; kept for exact-resume compat) ----
    alignment: float = 0.0
    approach: float = 0.0
    boost: float = 0.0
    # RLGym's ``bump_victim_id`` names the car this car had contact with.
    # Default 0: chasing the opponent is a distraction while learning 1v1.
    bump: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RewardWeights":
        if not data:
            return cls()
        # ``demo`` was the old (incorrect) name.  Accept it for configuration
        # files written before the rename, but write ``bump`` from now on.
        aliases = {"demo": "bump"}
        if "demo" in data and "bump" in data:
            raise ValueError("Use either 'bump' or the legacy 'demo', not both")
        normalized = {aliases.get(key, key): value for key, value in data.items()}
        unknown = set(normalized) - set(cls.__slots__)
        if unknown:
            raise ValueError(f"Unknown reward weights: {sorted(unknown)}")
        values = {k: float(v) for k, v in normalized.items()}
        for key, value in values.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Reward weight {key} must be finite and >= 0")
        return cls(**values)


@dataclass
class _AgentMemory:
    prev_ball_offset: np.ndarray | None = None
    prev_ball_pos: np.ndarray | None = None
    prev_ball_vel: np.ndarray | None = None
    prev_boost: float = 0.0
    prev_bump_victim: object | None = None
    steps: int = 0
    last_paid_touch_step: int = -(10**9)


class SoccerReward(RewardFunction[AgentID, GameState, float]):
    def __init__(
        self,
        spec: ModeSpec,
        weights: RewardWeights | None = None,
        potential_gamma: float = 1.0,
    ):
        """``potential_gamma`` optionally makes the potentials discount-exact.

        Exact shaping pays ``F = gamma*Phi(s') - Phi(s)``.  That is
        policy-invariant for the *discounted* objective, but at rest it
        streams ``(1-gamma)*|Phi|`` per step, so undiscounted episode
        returns and component means stop being interpretable (a parked car
        "earns" ball_progress every step; only the discounted sum stays
        bounded).  The plain difference (``gamma=1``, default) leaves a
        residual cycle profit of order ``(1-gamma)`` per round trip, which
        at gamma=0.995 is negligible next to the interpretability win.
        Training only passes its ``gamma`` here when the opt-in
        ``discount_shaping`` config flag is set.
        """
        super().__init__()
        if not 0.0 < float(potential_gamma) <= 1.0:
            raise ValueError("potential_gamma must be in (0, 1]")
        self.spec = spec
        self.weights = weights or RewardWeights()
        self.potential_gamma = float(potential_gamma)
        self._memory: dict[AgentID, _AgentMemory] = {}

    # -- lifecycle ---------------------------------------------------------
    def reset(self, agents, initial_state, shared_info) -> None:
        self._memory = {agent: _AgentMemory() for agent in agents}
        shared_info["reward_stats"] = {key: 0.0 for key in _COMPONENTS}
        shared_info["reward_stats_by_agent"] = {
            agent: {**{key: 0.0 for key in _COMPONENTS}, **{key: 0 for key in _EVENT_KEYS}}
            for agent in agents
        }
        for agent in agents:
            car = initial_state.cars[agent]
            mem = self._memory[agent]
            mem.prev_ball_offset = initial_state.ball.position - car.physics.position
            mem.prev_ball_pos = initial_state.ball.position.copy()
            mem.prev_ball_vel = initial_state.ball.linear_velocity.copy()
            mem.prev_boost = float(car.boost_amount)
            mem.prev_bump_victim = car.bump_victim_id

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards: dict[AgentID, float] = {}
        stats = shared_info.setdefault("reward_stats", {key: 0.0 for key in _COMPONENTS})
        by_agent = shared_info.setdefault("reward_stats_by_agent", {})
        for agent in agents:
            value, parts, events = self._reward_full(agent, state)
            rewards[agent] = value
            own = by_agent.setdefault(
                agent, {**{key: 0.0 for key in _COMPONENTS}, **{key: 0 for key in _EVENT_KEYS}}
            )
            for key, amount in zip(_COMPONENTS, parts):
                if amount:
                    stats[key] = stats.get(key, 0.0) + amount
                    own[key] = own.get(key, 0.0) + amount
            for key, count in events.items():
                if count:
                    own[key] = own.get(key, 0) + count
        return rewards

    # -- core --------------------------------------------------------------
    def _opponent_goal(self, team: int) -> np.ndarray:
        return (
            np.asarray(ORANGE_GOAL_CENTER, dtype=np.float32)
            if team == BLUE_TEAM
            else np.asarray(BLUE_GOAL_CENTER, dtype=np.float32)
        )

    def _own_goal(self, team: int) -> np.ndarray:
        return (
            np.asarray(BLUE_GOAL_CENTER, dtype=np.float32)
            if team == BLUE_TEAM
            else np.asarray(ORANGE_GOAL_CENTER, dtype=np.float32)
        )

    def _reward(self, agent: AgentID, state: GameState) -> tuple[float, tuple[float, ...]]:
        """Backwards-compatible (value, parts) view of :meth:`_reward_full`."""
        value, parts, _events = self._reward_full(agent, state)
        return value, parts

    def _reward_full(
        self, agent: AgentID, state: GameState
    ) -> tuple[float, tuple[float, ...], dict[str, int]]:
        car = state.cars[agent]
        team = car.team_num
        pos = car.physics.position
        ball = state.ball
        w = self.weights
        mem = self._memory.setdefault(agent, _AgentMemory())
        goal_value = 0.0
        concede = 0.0
        touch = 0.0
        shot = 0.0
        ball_progress = 0.0
        progress = 0.0
        alignment = 0.0
        approach = 0.0
        boost = 0.0
        defense = 0.0
        bump = 0.0
        events: dict[str, int] = {}

        goal = self._opponent_goal(team)
        own_goal = self._own_goal(team)

        if state.goal_scored:
            if state.scoring_team == team:
                goal_value = w.goal
                events["goals"] = 1
            else:
                concede = -w.concede

        # -- touch / shot / defense events ---------------------------------
        # Touch rewards are *events*, not states (paying per tick after the
        # first contact would turn the bonus into a constant offset), and
        # they are directional: what matters is where the ball goes *after*
        # the touch, not that contact happened.
        mem.steps += 1
        # RLGym defines ``Car.ball_touches`` as the number of contacts *since
        # the last state was sent*, and the simulator engine zeroes the
        # counter after every state read.  It is therefore already a per-step
        # delta.  The old code treated it as a cumulative counter and
        # subtracted the previous step's value, which silently dropped the
        # second of two contacts that happened on consecutive decision steps.
        new_touches = int(car.ball_touches)
        paid = bool(new_touches) and (
            mem.steps - mem.last_paid_touch_step >= TOUCH_COOLDOWN_STEPS
        )
        if new_touches:
            # contacts are always *counted* (curriculum evidence), but only
            # paid outside the cooldown (anti tap-farming; see constants)
            events["touches"] = new_touches
        if paid:
            mem.last_paid_touch_step = mem.steps
            toward_attack = _toward(ball.linear_velocity, goal, ball.position)
            direction = max(-0.3, min(1.0, toward_attack / 1500.0))
            # 0.3 base so *any* contact is discoverable early, up to 1.0 for
            # a touch that drives the ball at the goal, ~0.09x for knocking
            # it toward the own net.
            touch = w.touch * (0.3 + 0.7 * direction)
            hard_shot = (
                toward_attack > SHOT_SPEED_THRESHOLD
                and _distance(ball.position, goal) < SHOT_DISTANCE_THRESHOLD
            )
            if hard_shot and shot_on_target(ball.position, ball.linear_velocity, goal):
                shot = w.shot * min(toward_attack / SUPERSONIC_SPEED, 1.0)
                events["shots"] = 1
            elif hard_shot:
                # power without accuracy: counted for diagnostics, unpaid -
                # the touch term above already rewarded the goal-ward drive
                events["shots_off_target"] = 1
            elif mem.prev_ball_vel is not None:
                was_threat = _toward(mem.prev_ball_vel, own_goal, ball.position)
                still_threat = _toward(ball.linear_velocity, own_goal, ball.position)
                if (
                    was_threat > DEFENSE_SPEED_THRESHOLD
                    and still_threat < was_threat
                    and _distance(ball.position, own_goal) < DEFENSE_DISTANCE_THRESHOLD
                ):
                    stopped = min((was_threat - still_threat) / SUPERSONIC_SPEED, 1.0)
                    defense = w.defense * stopped
                    events["defensive_touches"] = 1

        offset = ball.position - pos

        # Potential-based shaping must not fire on the terminal goal event:
        # the ball has just travelled into the net, and charging/paying that
        # distance again would distort the scoring reward itself.
        if not state.goal_scored:
            # Task potential: ball closer to the opponent goal is better,
            # whoever moved it.  With Phi = -distance/1000 the discounted
            # shaping F = gamma*Phi' - Phi is (d_prev - gamma*d_new)/1000,
            # which is policy-invariant under the learner's discount and
            # still telescopes (pushing upfield and letting it roll back
            # nets the discounted round trip, never a profit).
            if mem.prev_ball_pos is not None:
                gamma = self.potential_gamma
                ball_progress = (
                    w.ball_progress
                    * (
                        _distance(mem.prev_ball_pos, goal)
                        - gamma * _distance(ball.position, goal)
                    )
                    / 1000.0
                )
                prev_offset = mem.prev_ball_offset
                if prev_offset is not None:
                    progress = (
                        w.progress
                        * (
                            _distance(prev_offset, (0.0, 0.0, 0.0))
                            - gamma * _distance(offset, (0.0, 0.0, 0.0))
                        )
                        / 1000.0
                    )

            # Legacy state bonuses (weights default to 0; see module docs).
            if w.alignment or w.approach:
                dist_to_ball = _distance(offset, (0.0, 0.0, 0.0))
                if dist_to_ball > 1e-3:
                    inv = 1.0 / dist_to_ball
                    to_ball = (
                        float(offset[0]) * inv,
                        float(offset[1]) * inv,
                        float(offset[2]) * inv,
                    )
                    forward = car.physics.forward
                    alignment = w.alignment * (
                        float(forward[0]) * to_ball[0]
                        + float(forward[1]) * to_ball[1]
                        + float(forward[2]) * to_ball[2]
                    )
                    velocity = car.physics.linear_velocity
                    approach_speed = (
                        float(velocity[0]) * to_ball[0]
                        + float(velocity[1]) * to_ball[1]
                        + float(velocity[2]) * to_ball[2]
                    ) / SUPERSONIC_SPEED
                    approach = w.approach * max(-1.0, min(1.0, approach_speed))

        boost_used = max(mem.prev_boost - float(car.boost_amount), 0.0)
        boost = -w.boost * boost_used / 100.0
        mem.prev_boost = float(car.boost_amount)
        mem.prev_ball_offset = offset.copy()
        mem.prev_ball_pos = ball.position.copy()
        mem.prev_ball_vel = ball.linear_velocity.copy()

        if car.bump_victim_id is not None and car.bump_victim_id != mem.prev_bump_victim:
            bump = w.bump
        mem.prev_bump_victim = car.bump_victim_id

        parts = (
            goal_value,
            concede,
            touch,
            shot,
            ball_progress,
            progress,
            alignment,
            approach,
            boost,
            defense,
            bump,
        )
        return float(sum(parts)), parts, events
