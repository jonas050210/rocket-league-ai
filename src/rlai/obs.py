"""Canonical observation layout.

Every mode uses the *same* fixed-size vector.  Teams are padded to three cars
so a 1v1 checkpoint and a 3v3 checkpoint have identical input shapes and a
policy can be fine-tuned across modes.  Features are expressed in the agent's
own car frame, which makes them invariant to team side and spawn rotation.

The exact same function is used by the RLGym environment and by the RLBot
adapter (which first converts a ``GamePacket`` into an RLGym ``GameState``).
That parity is what makes simulation-to-game transfer possible at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rlgym.api import AgentID, ObsBuilder
from rlgym.rocket_league.api import Car, GameState

from .actions import ACTION_COUNT
from .modes import ModeSpec

POS_SCALE = 4000.0
VEL_SCALE = 2300.0
PAD_SCALE = 10.0
TIME_SCALE = 300.0

SELF_FEATURES = 10
BALL_FEATURES = 7
CAR_FEATURES = 9
TEAMMATE_SLOTS = 2
OPPONENT_SLOTS = 3
PAD_FEATURES = 34
GLOBAL_FEATURES = 4
ACTION_FEATURES = ACTION_COUNT

OBS_SIZE = (
    SELF_FEATURES
    + BALL_FEATURES
    + TEAMMATE_SLOTS * CAR_FEATURES
    + OPPONENT_SLOTS * CAR_FEATURES
    + PAD_FEATURES
    + GLOBAL_FEATURES
    + ACTION_FEATURES
)

assert OBS_SIZE == 10 + 7 + 18 + 27 + 34 + 4 + 90


def _local(vec: np.ndarray, car: Car) -> np.ndarray:
    """Rotate a world-space vector into the car's local frame."""
    return car.physics.rotation_mtx.T @ vec


def _car_block(car: Car | None, frame_car: Car) -> list[float]:
    if car is None:
        return [0.0] * CAR_FEATURES
    rel_pos = _local(car.physics.position - frame_car.physics.position, frame_car)
    rel_vel = _local(car.physics.linear_velocity - frame_car.physics.linear_velocity, frame_car)
    return [
        1.0,
        *(rel_pos / POS_SCALE).clip(-4, 4).tolist(),
        *(rel_vel / VEL_SCALE).clip(-3, 3).tolist(),
        float(car.boost_amount) / 100.0,
        1.0 if car.on_ground else 0.0,
    ]


def build_observation(
    state: GameState,
    agent: AgentID,
    spec: ModeSpec,
    last_action: int | None = None,
    *,
    time_remaining: float | None = None,
    score_diff: int = 0,
) -> np.ndarray:
    """Build the observation vector for ``agent``.

    ``time_remaining`` and ``score_diff`` default to values derived from the
    state (episode clock is tracked by the environment wrapper).
    """
    car = state.cars[agent]
    team = car.team_num
    teammates = [
        other
        for other_id, other in sorted(state.cars.items(), key=lambda kv: str(kv[0]))
        if other_id != agent and other.team_num == team
    ]
    opponents = [
        other
        for _, other in sorted(state.cars.items(), key=lambda kv: str(kv[0]))
        if other.team_num != team
    ]

    ball = state.ball
    have_rot = (
        getattr(car.physics, "_rotation_mtx", None) is not None
        or getattr(car.physics, "_quaternion", None) is not None
        or getattr(car.physics, "_euler_angles", None) is not None
    )
    if not have_rot:
        car.physics.rotation_mtx = np.eye(3, dtype=np.float32)

    ball_rel_pos = _local(ball.position - car.physics.position, car)
    ball_rel_vel = _local(ball.linear_velocity - car.physics.linear_velocity, car)

    self_features = [
        float(car.boost_amount) / 100.0,
        1.0 if car.on_ground else 0.0,
        1.0 if car.is_supersonic else 0.0,
        1.0 if car.can_flip else 0.0,
        1.0 if car.is_demoed else 0.0,
        1.0 if car.is_jumping else 0.0,
        1.0 if car.is_holding_jump else 0.0,
        1.0 if car.has_flipped else 0.0,
        float(min(car.air_time_since_jump, 1.25)) / 1.25,
        float(car.handbrake),
    ]

    ball_features = [
        *(ball_rel_pos / POS_SCALE).clip(-4, 4).tolist(),
        *(ball_rel_vel / VEL_SCALE).clip(-3, 3).tolist(),
        float(min(np.linalg.norm(ball.linear_velocity) / 6000.0, 2.0)),
    ]

    teammate_features: list[float] = []
    for i in range(TEAMMATE_SLOTS):
        teammate_features.extend(_car_block(teammates[i] if i < len(teammates) else None, car))

    opponent_features: list[float] = []
    for i in range(OPPONENT_SLOTS):
        opponent_features.extend(_car_block(opponents[i] if i < len(opponents) else None, car))

    pads = np.asarray(state.boost_pad_timers, dtype=np.float32)
    if pads.shape[0] != PAD_FEATURES:
        padded = np.zeros(PAD_FEATURES, dtype=np.float32)
        padded[: min(PAD_FEATURES, pads.shape[0])] = pads[:PAD_FEATURES]
        pads = padded
    pad_features = (pads / PAD_SCALE).clip(0, 1).tolist()

    if time_remaining is None:
        time_remaining = 0.0
    global_features = [
        float(np.clip(time_remaining / TIME_SCALE, 0.0, 1.0)),
        float(np.clip(score_diff / 5.0, -1.0, 1.0)),
        1.0 if spec.heatseeker else 0.0,
        1.0 if spec.team_size > 1 else 0.0,
    ]

    action_features = [0.0] * ACTION_COUNT
    if last_action is not None and 0 <= last_action < ACTION_COUNT:
        action_features[last_action] = 1.0

    obs = np.asarray(
        self_features
        + ball_features
        + teammate_features
        + opponent_features
        + pad_features
        + global_features
        + action_features,
        dtype=np.float32,
    )
    if obs.shape != (OBS_SIZE,):
        raise ValueError(f"observation has shape {obs.shape}, expected {(OBS_SIZE,)}")
    return obs


class SoccerObsBuilder(ObsBuilder[AgentID, np.ndarray, GameState, tuple[str, int]]):
    """RLGym observation builder wrapping :func:`build_observation`."""

    def __init__(self, spec: ModeSpec):
        super().__init__()
        self.spec = spec
        self._state: GameState | None = None
        self._clock: dict[str, Any] = {}

    def get_obs_space(self, agent: AgentID) -> tuple[str, int]:
        return "real", OBS_SIZE

    def reset(self, agents, initial_state, shared_info) -> None:
        self._state = initial_state
        shared_info["last_action"] = {agent: None for agent in agents}
        shared_info["time_remaining"] = float(self.spec.default_episode_seconds)
        shared_info["score_diff"] = 0

    def build_obs(self, agents, state, shared_info) -> dict[AgentID, np.ndarray]:
        self._state = state
        last = shared_info.get("last_action", {})
        remaining = shared_info.get("time_remaining", 0.0)
        score_diff = int(shared_info.get("score_diff", 0))
        return {
            agent: build_observation(
                state,
                agent,
                self.spec,
                last.get(agent),
                time_remaining=remaining,
                score_diff=score_diff,
            )
            for agent in agents
        }
