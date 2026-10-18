"""Canonical observation layout.

Two layouts are supported and both are produced by this module:

``v1`` (190 values, legacy)
    Egocentric features only.  The car knows where the ball and the other
    cars are *relative to itself*, but it has no absolute field anchor, so it
    cannot know where either goal is.  Kept so existing checkpoints stay
    usable for evaluation and deployment.

``v2`` (198 values, default)
    Adds an 8-value field block: own position, own forward vector, ball world
    position, alignment of the ball with the opponent goal and the remaining
    distance to that goal.  Without these features scoring is not learnable;
    a network cannot infer a fixed goal position from relative vectors alone.

Teams are padded to three cars so a 1v1, 2v2 and 3v3 checkpoint share the same
input shape.  Rotated features are expressed in the agent's own car frame,
which makes them invariant to team side and spawn rotation.  The exact same
builder is used by the RLGym environment and by the RLBot adapter, which is
what makes simulation-to-game transfer possible at all.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from rlgym.api import AgentID, ObsBuilder
from rlgym.rocket_league.api import Car, GameState
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_GOAL_CENTER, BLUE_GOAL_CENTER

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
FIELD_FEATURES = 8

SELF_START = 0
BALL_START = SELF_START + SELF_FEATURES
TEAMMATES_START = BALL_START + BALL_FEATURES
OPPONENTS_START = TEAMMATES_START + TEAMMATE_SLOTS * CAR_FEATURES
PADS_START = OPPONENTS_START + OPPONENT_SLOTS * CAR_FEATURES
GLOBAL_START = PADS_START + PAD_FEATURES
FIELD_START = GLOBAL_START + GLOBAL_FEATURES
ACTION_START_V1 = FIELD_START
ACTION_START_V2 = FIELD_START + FIELD_FEATURES

OBS_SIZE_V1 = ACTION_START_V1 + ACTION_COUNT
OBS_SIZE_V2 = ACTION_START_V2 + ACTION_COUNT
OBS_SIZE_BY_VERSION = {1: OBS_SIZE_V1, 2: OBS_SIZE_V2}
OBS_VERSION = 2
OBS_SIZE = OBS_SIZE_BY_VERSION[OBS_VERSION]

assert OBS_SIZE_V1 == 10 + 7 + 18 + 27 + 34 + 4 + 90 == 190
assert OBS_SIZE_V2 == 190 + 8 == 198

GOAL_CENTERS = {
    BLUE_TEAM: np.asarray(ORANGE_GOAL_CENTER, dtype=np.float32),
    1 - BLUE_TEAM: np.asarray(BLUE_GOAL_CENTER, dtype=np.float32),
}
GOAL_XY = {team: (float(center[0]), float(center[1])) for team, center in GOAL_CENTERS.items()}
FIELD_X = 4096.0
FIELD_Y = 5120.0


def obs_size(version: int = OBS_VERSION) -> int:
    try:
        return OBS_SIZE_BY_VERSION[int(version)]
    except KeyError:
        raise ValueError(
            f"unknown observation version {version!r}; available: {sorted(OBS_SIZE_BY_VERSION)}"
        ) from None


def version_for_obs_size(size: int) -> int:
    """Infer the layout from a checkpoint's ``obs_size``."""
    for version, candidate in OBS_SIZE_BY_VERSION.items():
        if int(size) == candidate:
            return version
    raise ValueError(f"no observation layout matches obs_size={size}")


#: RLGym's side-swap convention: rotate the whole scene 180 deg about z.
#: ``PhysicsObject.INV_VEC`` flips x and y, ``GameState.inverted_boost_pad_timers``
#: reverses the pad list, and the rotation matrix is left-multiplied by
#: ``diag(-1, -1, 1)``.  Using the same convention here means the observation
#: mirror agrees with the simulator's own notion of "the other side".
SIDE_SWAP_AXES = (0, 1)
PAD_MIRROR: tuple[int, ...] = tuple(range(PAD_FEATURES - 1, -1, -1))


def pad_mirror_permutation() -> tuple[int, ...]:
    """Pad index a car on the opposite side sees at each slot.

    ``BOOST_LOCATIONS`` is ordered so that reversing the list is exactly the
    180 deg rotation about z; the simulator's own pad order matches that list
    index-for-index (verified against ``RocketSim.Arena.get_boost_pads()``).
    """
    return PAD_MIRROR


def mirror_observation(obs: np.ndarray, obs_version: int = OBS_VERSION) -> np.ndarray:
    """Rewrite an observation as seen from the opposite side of the field.

    The field is symmetric under a 180 deg rotation about the vertical axis.
    Applied to the whole scene (car, ball, pads, both teams) that rotation
    leaves almost the observation untouched, because the observation is
    expressed in the car's own frame and the car rotates with the world:

    * relative ball / teammate / opponent features: **invariant**;
    * the absolute field block (v2): ``pos_x``, ``pos_y``, ``forward_x``,
      ``forward_y``, ``ball_x`` and ``ball_y`` all flip sign, while goal
      alignment and goal distance stay the same;
    * boost pads: the block is reversed (RLGym's ``inverted_boost_pad_timers``);
    * the recorded last action: **invariant** - throttle, steer, yaw, pitch,
      roll, jump, boost and handbrake are all car-relative.

    This is what makes side-balanced evaluation and orange-side deployment
    possible for a checkpoint that only ever trained on blue.
    """
    version = int(obs_version)
    expected = obs_size(version)
    values = np.asarray(obs, dtype=np.float32).reshape(-1)
    if values.shape[0] != expected:
        raise ValueError(
            f"observation has {values.shape[0]} values, expected {expected} for version {version}"
        )
    mirrored = values.copy()
    mirrored[PADS_START : PADS_START + PAD_FEATURES] = values[
        PADS_START + PAD_FEATURES - 1 : PADS_START - 1 : -1
    ]
    if version >= 2:
        for offset in range(6):  # pos x/y, forward x/y, ball x/y
            mirrored[FIELD_START + offset] = -values[FIELD_START + offset]
    return mirrored


def _clamp(value: float, low: float, high: float) -> float:
    # Scalar clamping is ~10x cheaper than calling np.clip on a 3-vector.
    return low if value < low else (high if value > high else value)


def _car_block(
    obs: np.ndarray,
    offset: int,
    car: Car | None,
    frame_pos: np.ndarray,
    frame_rot: np.ndarray,
    frame_vel: np.ndarray,
) -> None:
    """Write a 9-value relative car block.  Absent cars stay zero."""
    if car is None:
        return
    rel_pos = (car.physics.position - frame_pos) @ frame_rot
    rel_vel = (car.physics.linear_velocity - frame_vel) @ frame_rot
    obs[offset] = 1.0
    obs[offset + 1] = _clamp(rel_pos[0] / POS_SCALE, -4.0, 4.0)
    obs[offset + 2] = _clamp(rel_pos[1] / POS_SCALE, -4.0, 4.0)
    obs[offset + 3] = _clamp(rel_pos[2] / POS_SCALE, -4.0, 4.0)
    obs[offset + 4] = _clamp(rel_vel[0] / VEL_SCALE, -3.0, 3.0)
    obs[offset + 5] = _clamp(rel_vel[1] / VEL_SCALE, -3.0, 3.0)
    obs[offset + 6] = _clamp(rel_vel[2] / VEL_SCALE, -3.0, 3.0)
    obs[offset + 7] = float(car.boost_amount) / 100.0
    obs[offset + 8] = 1.0 if car.on_ground else 0.0


def _ordered_cars(state: GameState) -> list[tuple[str, Car]]:
    # Deterministic order regardless of dict insertion order (RLBot packets
    # and the simulator do not have to agree).
    return sorted(state.cars.items(), key=lambda item: str(item[0]))


def _split_teams(
    ordered: list[tuple[str, Car]], agent: AgentID, team: int
) -> tuple[list[Car], list[Car]]:
    """Teammates and opponents in deterministic agent-id order."""
    teammates = [
        car for other_id, car in ordered if int(car.team_num) == team and other_id != agent
    ]
    opponents = [car for _, car in ordered if int(car.team_num) != team]
    return teammates, opponents


def _build_obs(
    state: GameState,
    agent: AgentID,
    spec: ModeSpec,
    last_action: int | None,
    *,
    time_remaining: float,
    score_diff: int,
    obs_version: int,
    ordered: list[tuple[str, Car]],
) -> np.ndarray:
    size = obs_size(obs_version)
    obs = np.zeros(size, dtype=np.float32)

    car = state.cars[agent]
    team = int(car.team_num)
    physics = car.physics
    pos = physics.position
    vel = physics.linear_velocity
    rot = physics.rotation_mtx
    if rot is None:  # defensive: partially initialised states in tests
        rot = np.eye(3, dtype=np.float32)
        physics.rotation_mtx = rot

    # -- self ------------------------------------------------------------
    obs[SELF_START + 0] = float(car.boost_amount) / 100.0
    obs[SELF_START + 1] = 1.0 if car.on_ground else 0.0
    obs[SELF_START + 2] = 1.0 if car.is_supersonic else 0.0
    obs[SELF_START + 3] = 1.0 if car.can_flip else 0.0
    obs[SELF_START + 4] = 1.0 if car.is_demoed else 0.0
    obs[SELF_START + 5] = 1.0 if car.is_jumping else 0.0
    obs[SELF_START + 6] = 1.0 if car.is_holding_jump else 0.0
    obs[SELF_START + 7] = 1.0 if car.has_flipped else 0.0
    obs[SELF_START + 8] = float(min(car.air_time_since_jump, 1.25)) / 1.25
    obs[SELF_START + 9] = float(car.handbrake)

    # -- ball (relative, car frame) --------------------------------------
    ball = state.ball
    ball_pos = ball.position
    ball_vel = ball.linear_velocity
    rel_pos = (ball_pos - pos) @ rot
    rel_vel = (ball_vel - vel) @ rot
    for index in range(3):
        obs[BALL_START + index] = _clamp(rel_pos[index] / POS_SCALE, -4.0, 4.0)
        obs[BALL_START + 3 + index] = _clamp(rel_vel[index] / VEL_SCALE, -3.0, 3.0)
    speed = math.sqrt(
        float(ball_vel[0]) ** 2 + float(ball_vel[1]) ** 2 + float(ball_vel[2]) ** 2
    )
    obs[BALL_START + 6] = min(speed / 6000.0, 2.0)

    # -- teammates / opponents -------------------------------------------
    teammates, opponents = _split_teams(ordered, agent, team)
    for slot in range(TEAMMATE_SLOTS):
        other = teammates[slot] if slot < len(teammates) else None
        _car_block(obs, TEAMMATES_START + slot * CAR_FEATURES, other, pos, rot, vel)
    for slot in range(OPPONENT_SLOTS):
        other = opponents[slot] if slot < len(opponents) else None
        _car_block(obs, OPPONENTS_START + slot * CAR_FEATURES, other, pos, rot, vel)

    # -- boost pads -------------------------------------------------------
    pads = state.boost_pad_timers
    count = min(int(getattr(pads, "shape", (0,))[0]), PAD_FEATURES)
    if count:
        np.clip(pads[:count] / PAD_SCALE, 0.0, 1.0, out=obs[PADS_START : PADS_START + count])

    # -- global -----------------------------------------------------------
    obs[GLOBAL_START + 0] = _clamp(time_remaining / TIME_SCALE, 0.0, 1.0)
    obs[GLOBAL_START + 1] = _clamp(score_diff / 5.0, -1.0, 1.0)
    obs[GLOBAL_START + 2] = 1.0 if spec.heatseeker else 0.0
    obs[GLOBAL_START + 3] = 1.0 if spec.team_size > 1 else 0.0

    # -- field frame (v2) --------------------------------------------------
    if obs_version >= 2:
        pos_x = float(pos[0])
        pos_y = float(pos[1])
        ball_x = float(ball_pos[0])
        ball_y = float(ball_pos[1])
        forward_x = float(rot[0, 0])
        forward_y = float(rot[1, 0])
        obs[FIELD_START + 0] = _clamp(pos_x / FIELD_X, -1.25, 1.25)
        obs[FIELD_START + 1] = _clamp(pos_y / FIELD_Y, -1.25, 1.25)
        obs[FIELD_START + 2] = forward_x
        obs[FIELD_START + 3] = forward_y
        obs[FIELD_START + 4] = _clamp(ball_x / FIELD_X, -1.25, 1.25)
        obs[FIELD_START + 5] = _clamp(ball_y / FIELD_Y, -1.25, 1.25)
        goal_x, goal_y = GOAL_XY.get(team, GOAL_XY[BLUE_TEAM])
        dx = goal_x - ball_x
        dy = goal_y - ball_y
        distance = math.hypot(dx, dy)
        if distance > 1e-6:
            obs[FIELD_START + 6] = _clamp(
                (forward_x * dx + forward_y * dy) / distance, -1.0, 1.0
            )
        obs[FIELD_START + 7] = _clamp(distance / 8000.0, 0.0, 1.5)

    # -- last action one-hot -----------------------------------------------
    action_start = ACTION_START_V2 if obs_version >= 2 else ACTION_START_V1
    if last_action is not None and 0 <= int(last_action) < ACTION_COUNT:
        obs[action_start + int(last_action)] = 1.0
    return obs


def build_observation(
    state: GameState,
    agent: AgentID,
    spec: ModeSpec,
    last_action: int | None = None,
    *,
    time_remaining: float | None = None,
    score_diff: int = 0,
    obs_version: int = OBS_VERSION,
) -> np.ndarray:
    """Build the observation vector for ``agent``.

    ``time_remaining`` and ``score_diff`` are tracked by the environment
    wrapper; standalone callers may pass them explicitly.
    """
    ordered = _ordered_cars(state)
    return _build_obs(
        state,
        agent,
        spec,
        last_action,
        time_remaining=0.0 if time_remaining is None else float(time_remaining),
        score_diff=int(score_diff),
        obs_version=int(obs_version),
        ordered=ordered,
    )


class SoccerObsBuilder(ObsBuilder[AgentID, np.ndarray, GameState, tuple[str, int]]):
    """RLGym observation builder wrapping :func:`build_observation`."""

    def __init__(self, spec: ModeSpec, obs_version: int = OBS_VERSION):
        super().__init__()
        self.spec = spec
        self.obs_version = int(obs_version)
        if self.obs_version not in OBS_SIZE_BY_VERSION:
            raise ValueError(f"unknown observation version {obs_version}")
        self.size = obs_size(self.obs_version)
        self._state: GameState | None = None
        self._clock: dict[str, Any] = {}

    def get_obs_space(self, agent: AgentID) -> tuple[str, int]:
        return "real", self.size

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
        ordered = _ordered_cars(state)
        return {
            agent: _build_obs(
                state,
                agent,
                self.spec,
                last.get(agent),
                time_remaining=remaining,
                score_diff=score_diff,
                obs_version=self.obs_version,
                ordered=ordered,
            )
            for agent in agents
        }
