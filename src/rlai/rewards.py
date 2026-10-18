"""Configurable reward shaping.

The reward is deliberately potential-based where it shapes behaviour
(ball progress, alignment) so that it cannot create infinite reward cycles.
Goals dominate; touches, progress and boost efficiency provide gradient.

Every component is tracked in ``shared_info['reward_stats']`` so the trainer
can log why the agent is being rewarded, not just the total.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
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
    "demo",
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
    demo: float = 0.3

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RewardWeights":
        if not data:
            return cls()
        unknown = set(data) - set(cls.__slots__)
        if unknown:
            raise ValueError(f"Unknown reward weights: {sorted(unknown)}")
        values = {k: float(v) for k, v in data.items()}
        for key, value in values.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Reward weight {key} must be finite and >= 0")
        return cls(**values)


@dataclass
class _AgentMemory:
    prev_ball_offset: np.ndarray | None = None
    prev_ball_vel: np.ndarray | None = None
    prev_boost: float = 0.0


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

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards: dict[AgentID, float] = {}
        stats = shared_info.setdefault("reward_stats", {key: 0.0 for key in _COMPONENTS})
        for agent in agents:
            value, breakdown = self._reward(agent, state)
            rewards[agent] = value
            for key, amount in breakdown.items():
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

    def _reward(self, agent: AgentID, state: GameState) -> tuple[float, dict[str, float]]:
        car = state.cars[agent]
        team = car.team_num
        pos = car.physics.position
        ball = state.ball
        w = self.weights
        mem = self._memory.setdefault(agent, _AgentMemory())
        breakdown = {key: 0.0 for key in _COMPONENTS}

        if state.goal_scored:
            if state.scoring_team == team:
                breakdown["goal"] += w.goal
            else:
                breakdown["concede"] += -w.concede

        if car.ball_touches > 0:
            speed_factor = min(float(np.linalg.norm(ball.linear_velocity)) / 2300.0, 1.5)
            breakdown["touch"] += w.touch * (0.5 + speed_factor)

        goal = self._opponent_goal(team)
        offset = ball.position - pos

        # Potential-based shaping must not fire on the terminal goal event:
        # the ball has just travelled into the net, and penalising that
        # distance would cancel part of the scoring reward.
        if not state.goal_scored:
            prev_offset = mem.prev_ball_offset
            if prev_offset is not None:
                prev_dist = float(np.linalg.norm(prev_offset))
                new_dist = float(np.linalg.norm(offset))
                breakdown["progress"] += w.progress * (prev_dist - new_dist) / 1000.0

            dist_to_ball = float(np.linalg.norm(offset))
            if dist_to_ball > 1e-3:
                to_ball = offset / dist_to_ball
                forward = car.physics.forward
                alignment = float(np.dot(forward, to_ball))
                breakdown["alignment"] += w.alignment * alignment
                approach = float(np.dot(car.physics.linear_velocity, to_ball)) / 2300.0
                breakdown["approach"] += w.approach * float(np.clip(approach, -1, 1))

                own_goal = self._own_goal(team)
                car_behind_ball = float(np.dot(ball.position - pos, goal - own_goal)) < 0
                if car_behind_ball:
                    toward_own = float(np.dot(ball.linear_velocity, (own_goal - ball.position))) > 0
                    if toward_own:
                        breakdown["defense"] += w.defense * min(
                            float(np.linalg.norm(goal - ball.position)) / 6000.0, 0.2
                        )

        boost_used = max(mem.prev_boost - float(car.boost_amount), 0.0)
        breakdown["boost"] -= w.boost * boost_used / 100.0
        mem.prev_boost = float(car.boost_amount)
        mem.prev_ball_offset = offset.copy()

        if car.bump_victim_id is not None:
            breakdown["demo"] += w.demo

        total = float(sum(breakdown.values()))
        return total, breakdown
