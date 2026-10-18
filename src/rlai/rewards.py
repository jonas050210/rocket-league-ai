"""Configurable reward shaping.

The reward is deliberately potential-based where it shapes behaviour
(ball progress, alignment) so that it cannot create infinite reward cycles.
Goals dominate; touches, progress and boost efficiency provide gradient.

Every component is tracked in ``shared_info['reward_stats']`` so the trainer
can log why the agent is being rewarded, not just the total.
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
    "progress",
    "alignment",
    "approach",
    "boost",
    "defense",
    "bump",
)


@dataclass(slots=True)
class RewardWeights:
    goal: float = 10.0
    concede: float = 10.0
    touch: float = 0.35
    progress: float = 0.9
    alignment: float = 0.02
    approach: float = 0.03
    boost: float = 0.02
    defense: float = 0.15
    # RLGym exposes car-to-car contact as ``bump_victim_id``; it does not
    # identify the car that caused a demolition.  The component is therefore
    # named and documented as a bump, not as a demo.
    bump: float = 0.3

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
    prev_ball_vel: np.ndarray | None = None
    prev_boost: float = 0.0
    prev_touches: int = 0
    prev_bump_victim: object | None = None


class SoccerReward(RewardFunction[AgentID, GameState, float]):
    def __init__(self, spec: ModeSpec, weights: RewardWeights | None = None):
        super().__init__()
        self.spec = spec
        self.weights = weights or RewardWeights()
        self._memory: dict[AgentID, _AgentMemory] = {}

    # -- lifecycle ---------------------------------------------------------
    def reset(self, agents, initial_state, shared_info) -> None:
        self._memory = {agent: _AgentMemory() for agent in agents}
        shared_info["reward_stats"] = {key: 0.0 for key in _COMPONENTS}
        for agent in agents:
            car = initial_state.cars[agent]
            mem = self._memory[agent]
            mem.prev_ball_offset = initial_state.ball.position - car.physics.position
            mem.prev_ball_vel = initial_state.ball.linear_velocity.copy()
            mem.prev_boost = float(car.boost_amount)
            mem.prev_touches = int(car.ball_touches)
            mem.prev_bump_victim = car.bump_victim_id

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards: dict[AgentID, float] = {}
        stats = shared_info.setdefault("reward_stats", {key: 0.0 for key in _COMPONENTS})
        for agent in agents:
            value, parts = self._reward(agent, state)
            rewards[agent] = value
            for key, amount in zip(_COMPONENTS, parts):
                if amount:
                    stats[key] = stats.get(key, 0.0) + amount
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
        car = state.cars[agent]
        team = car.team_num
        pos = car.physics.position
        ball = state.ball
        w = self.weights
        mem = self._memory.setdefault(agent, _AgentMemory())
        goal_value = 0.0
        concede = 0.0
        touch = 0.0
        progress = 0.0
        alignment = 0.0
        approach = 0.0
        boost = 0.0
        defense = 0.0
        bump = 0.0

        if state.goal_scored:
            if state.scoring_team == team:
                goal_value = w.goal
            else:
                concede = -w.concede

        # Touch and bump rewards are *events*, not states.  The previous
        # implementation paid the touch bonus on every tick after the first
        # contact, which turned it into a constant offset.
        touches = int(car.ball_touches)
        if touches > mem.prev_touches:
            velocity = ball.linear_velocity
            speed = math.sqrt(
                float(velocity[0]) ** 2 + float(velocity[1]) ** 2 + float(velocity[2]) ** 2
            )
            speed_factor = min(speed / 2300.0, 1.5)
            touch = w.touch * (0.5 + speed_factor) * (touches - mem.prev_touches)
        mem.prev_touches = touches

        goal = self._opponent_goal(team)
        offset = ball.position - pos

        # Potential-based shaping must not fire on the terminal goal event:
        # the ball has just travelled into the net, and penalising that
        # distance would cancel part of the scoring reward.
        if not state.goal_scored:
            prev_offset = mem.prev_ball_offset
            if prev_offset is not None:
                prev_dist = math.sqrt(
                    float(prev_offset[0]) ** 2
                    + float(prev_offset[1]) ** 2
                    + float(prev_offset[2]) ** 2
                )
                new_dist = math.sqrt(
                    float(offset[0]) ** 2 + float(offset[1]) ** 2 + float(offset[2]) ** 2
                )
                progress = w.progress * (prev_dist - new_dist) / 1000.0

            dist_to_ball = math.sqrt(
                float(offset[0]) ** 2 + float(offset[1]) ** 2 + float(offset[2]) ** 2
            )
            if dist_to_ball > 1e-3:
                inv = 1.0 / dist_to_ball
                to_ball_x = float(offset[0]) * inv
                to_ball_y = float(offset[1]) * inv
                to_ball_z = float(offset[2]) * inv
                forward = car.physics.forward
                alignment = w.alignment * (
                    float(forward[0]) * to_ball_x
                    + float(forward[1]) * to_ball_y
                    + float(forward[2]) * to_ball_z
                )
                velocity = car.physics.linear_velocity
                approach_speed = (
                    float(velocity[0]) * to_ball_x
                    + float(velocity[1]) * to_ball_y
                    + float(velocity[2]) * to_ball_z
                ) / 2300.0
                approach = w.approach * max(-1.0, min(1.0, approach_speed))

                own_goal = self._own_goal(team)
                car_behind_ball = (
                    (float(ball.position[0]) - float(pos[0])) * (goal[0] - own_goal[0])
                    + (float(ball.position[1]) - float(pos[1])) * (goal[1] - own_goal[1])
                    + (float(ball.position[2]) - float(pos[2])) * (goal[2] - own_goal[2])
                ) < 0
                if car_behind_ball:
                    to_own_x = float(own_goal[0]) - float(ball.position[0])
                    to_own_y = float(own_goal[1]) - float(ball.position[1])
                    toward_own = (
                        float(ball.linear_velocity[0]) * to_own_x
                        + float(ball.linear_velocity[1]) * to_own_y
                    ) > 0
                    if toward_own:
                        distance_to_goal = math.hypot(
                            float(goal[0]) - float(ball.position[0]),
                            float(goal[1]) - float(ball.position[1]),
                        )
                        defense = w.defense * min(distance_to_goal / 6000.0, 0.2)

        boost_used = max(mem.prev_boost - float(car.boost_amount), 0.0)
        boost = -w.boost * boost_used / 100.0
        mem.prev_boost = float(car.boost_amount)
        mem.prev_ball_offset = offset.copy()

        if car.bump_victim_id is not None and car.bump_victim_id != mem.prev_bump_victim:
            bump = w.bump
        mem.prev_bump_victim = car.bump_victim_id

        parts = (
            goal_value,
            concede,
            touch,
            progress,
            alignment,
            approach,
            boost,
            defense,
            bump,
        )
        return float(sum(parts)), parts
