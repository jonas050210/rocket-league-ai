"""Starting-state generators: the ``reach`` distribution and exact scenarios.

Two different needs are served here, and keeping them in one module is what
makes them comparable:

``reach``
    A randomized starting distribution for early curriculum stages.  It is
    the ``drill`` attack scenario with the learning team much closer to the
    ball, because the measured bottleneck of a fresh policy is the length of
    the causal chain between "drive at the ball" and "touch the ball"
    (docs/MASTERY_CURRICULUM.md, experiments 1-3).

scenarios
    *Exact* starting states used by the independent skill probes in
    :mod:`rlai.skills`.  A gate that measures "can the policy reach the ball
    from 800 uu?" must be able to place the ball at 800 uu, on both sides of
    the field, with a fixed seed - a randomized distribution cannot answer
    that question with a small number of trials.

Both are expressed in the *learning team's attacking frame* (+y is the
direction the learning team attacks).  :func:`apply_scenario` mirrors the
whole scene when the learning team plays orange, so one scenario definition
measures both sides of the field.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from rlgym.rocket_league.common_values import BLUE_TEAM

#: how far behind the ball the learning team spawns in ``reach`` starts (uu)
REACH_DISTANCE = (250.0, 900.0)
#: the opponent spawns further away, so the learner normally gets first touch
REACH_OPPONENT_DISTANCE = (1800.0, 3200.0)
#: lateral scatter of the spawn point around the ball→goal line (uu)
REACH_LATERAL = 250.0
#: ball placement area of a ``reach`` start (uu)
REACH_BALL_X = (-2000.0, 2000.0)
REACH_BALL_Y = (-2500.0, 2500.0)

#: ``recover`` starts: the learning car is 700-1500 uu from the ball and faces
#: its own goal - the ``reengage`` probe's distance band
RECOVER_DISTANCE = (700.0, 1500.0)
RECOVER_OPPONENT_DISTANCE = (1800.0, 3000.0)
RECOVER_LATERAL = 400.0
RECOVER_BALL_X = (-1500.0, 1500.0)
RECOVER_BALL_Y = (-1000.0, 2000.0)

FIELD_X_LIMIT = 3800.0
FIELD_Y_LIMIT = 4800.0
BALL_RADIUS_Z = 93.0


def _mirror(value: float) -> float:
    return -float(value)


def place_car(
    car, position, *, yaw: float | None = None, aim=None, velocity=None, boost=None, pitch=0.0
):
    """Put one car at ``position`` (policy-attacking frame already applied)."""
    car.physics.position = np.array(
        [float(position[0]), float(position[1]), float(position[2])], dtype=np.float32
    )
    car.physics.linear_velocity = (
        np.zeros(3, dtype=np.float32)
        if velocity is None
        else np.array([float(v) for v in velocity], dtype=np.float32)
    )
    car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
    if yaw is None and aim is not None:
        yaw = float(
            np.arctan2(
                float(aim[1]) - float(position[1]), float(aim[0]) - float(position[0])
            )
        )
    car.physics.euler_angles = np.array(
        [float(pitch), float(yaw or 0.0), 0.0], dtype=np.float32
    )
    if boost is not None:
        car.boost_amount = float(boost)


def apply_reach(state, shared_info: dict[str, Any]) -> None:
    """Randomized close-range starts: the ball is a short drive away.

    The learning team (``shared_info['drill_team']``, set by
    :class:`rlai.envs.MatchEnv`) starts between its own goal and the ball,
    facing it; the opponent starts on the far side of the ball.  The ball is
    at rest in three quarters of the episodes and rolls slowly otherwise, so
    the learner has to *drive* to make contact - unlike a scenario where the
    ball is fed into a standing car, which would pay for doing nothing.
    """
    drill_team = int(shared_info.get("drill_team", BLUE_TEAM))
    ball_x = random.uniform(*REACH_BALL_X)
    ball_y = random.uniform(*REACH_BALL_Y)
    if random.random() < 0.25:
        angle = random.uniform(0.0, 2.0 * np.pi)
        speed = random.uniform(0.0, 400.0)
        velocity = np.array([speed * np.cos(angle), speed * np.sin(angle), 0.0], dtype=np.float32)
    else:
        velocity = np.zeros(3, dtype=np.float32)
    state.ball.position = np.array([ball_x, ball_y, BALL_RADIUS_Z], dtype=np.float32)
    state.ball.linear_velocity = velocity
    state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
    shared_info["drill_scenario"] = "reach"
    for car in state.cars.values():
        # +1 for blue (attacks +y), -1 for orange: "behind the ball" is always
        # the own-goal side of the ball.
        attack = 1.0 if int(car.team_num) == BLUE_TEAM else -1.0
        learning = int(car.team_num) == int(drill_team)
        low, high = REACH_DISTANCE if learning else REACH_OPPONENT_DISTANCE
        distance = random.uniform(low, high)
        lateral = REACH_LATERAL if learning else REACH_LATERAL * 3.0
        car_x = float(np.clip(ball_x + random.uniform(-lateral, lateral), -FIELD_X_LIMIT, FIELD_X_LIMIT))
        car_y = float(np.clip(ball_y - attack * distance, -FIELD_Y_LIMIT, FIELD_Y_LIMIT))
        place_car(
            car,
            (car_x, car_y, 17.0),
            aim=(ball_x, ball_y),
            boost=random.uniform(40.0, 100.0),
        )


def apply_recover(state, shared_info: dict[str, Any]) -> None:
    """Recovery starts: the learning car faces its own goal, the ball behind it.

    This is the ``reengage`` probe's situation, isolated: every learning car
    starts 700-1500 uu from a resting ball **facing away from it**, so the
    only way to touch the ball is to turn around first.  No other reset mode
    produces that state - ``reach``/``drill`` spawn cars facing the ball -
    and a policy that never sees it puts ~99 % of its action probability on a
    no-op there (see ``docs/STALL_ANALYSIS.md``).

    Distance band and opponent placement mirror the probe so that a stage
    using this mode trains the measured situation, not an approximation.
    """
    drill_team = int(shared_info.get("drill_team", BLUE_TEAM))
    ball_x = random.uniform(*RECOVER_BALL_X)
    ball_y = random.uniform(*RECOVER_BALL_Y)
    state.ball.position = np.array([ball_x, ball_y, BALL_RADIUS_Z], dtype=np.float32)
    state.ball.linear_velocity = np.zeros(3, dtype=np.float32)
    state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
    shared_info["drill_scenario"] = "recover"
    for car in state.cars.values():
        attack = 1.0 if int(car.team_num) == BLUE_TEAM else -1.0
        learning = int(car.team_num) == int(drill_team)
        # the car starts on its own-goal side of the ball, i.e. "behind" it
        distance = random.uniform(*(RECOVER_DISTANCE if learning else RECOVER_OPPONENT_DISTANCE))
        lateral = RECOVER_LATERAL if learning else RECOVER_LATERAL * 2.0
        car_x = float(
            np.clip(ball_x + random.uniform(-lateral, lateral), -FIELD_X_LIMIT, FIELD_X_LIMIT)
        )
        car_y = float(
            np.clip(ball_y - attack * distance, -FIELD_Y_LIMIT, FIELD_Y_LIMIT)
        )
        if learning:
            # facing away from the ball == facing its own goal
            yaw = float(np.arctan2(ball_y - car_y, ball_x - car_x) + np.pi)
            place_car(car, (car_x, car_y, 17.0), yaw=yaw, boost=random.uniform(30.0, 100.0))
        else:
            place_car(
                car,
                (car_x, car_y, 17.0),
                aim=(ball_x, ball_y),
                boost=random.uniform(30.0, 100.0),
            )


def mirror_point(point) -> tuple[float, float, float]:
    """180 deg rotation about the vertical axis (the field's symmetry)."""
    return (_mirror(point[0]), _mirror(point[1]), float(point[2]))


def apply_scenario(state, shared_info: dict[str, Any], scenario: dict) -> None:
    """Apply an *exact* starting state, mirrored for an orange learning team.

    ``scenario`` keys (all positions in the learning team's attacking frame):

    ``ball``      ``{"position": (x, y, z), "velocity": (vx, vy, vz)}``
    ``policy``    list of car specs for the learning team
    ``opponent``  list of car specs for the other team
    ``unlimited_boost``  optional, disables boost consumption (probes that
                  measure boost behaviour leave it on)

    A car spec is ``{"position": (x, y, z), "yaw": float | None,
    "aim": (x, y) | None, "velocity": (vx, vy, vz) | None, "boost": float}``.
    """
    drill_team = int(shared_info.get("drill_team", BLUE_TEAM))
    flip = drill_team != BLUE_TEAM

    def point(value, default_z: float):
        x, y, z = (float(value[0]), float(value[1]), float(value[2]) if len(value) > 2 else default_z)
        return mirror_point((x, y, z)) if flip else (x, y, z)

    def vector(value):
        out = np.array([float(v) for v in value], dtype=np.float32)
        if flip:
            out[0] *= -1.0
            out[1] *= -1.0
        return out

    ball = scenario.get("ball") or {}
    if ball.get("position") is not None:
        position = point(ball["position"], BALL_RADIUS_Z)
        state.ball.position = np.array(position, dtype=np.float32)
    state.ball.linear_velocity = (
        vector(ball["velocity"]) if ball.get("velocity") is not None else np.zeros(3, dtype=np.float32)
    )
    state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
    if scenario.get("unlimited_boost"):
        state.config.boost_consumption = 0.0

    ball_position = np.asarray(state.ball.position, dtype=np.float32)
    groups = (("policy", drill_team), ("opponent", 1 - drill_team))
    for key, team in groups:
        specs = list(scenario.get(key) or ())
        cars = [car for car in state.cars.values() if int(car.team_num) == int(team)]
        for car, spec in zip(cars, specs):
            position = point(spec["position"], 17.0)
            aim = None
            if spec.get("aim") is not None:
                aim_x, aim_y = float(spec["aim"][0]), float(spec["aim"][1])
                aim = (-aim_x, -aim_y) if flip else (aim_x, aim_y)
            yaw = spec.get("yaw")
            if yaw is not None and flip:
                yaw = float(yaw) + float(np.pi)
            velocity = vector(spec["velocity"]) if spec.get("velocity") is not None else None
            place_car(
                car,
                position,
                yaw=yaw,
                aim=aim if yaw is None else None,
                velocity=velocity,
                boost=spec.get("boost"),
                # an upside-down car is the recovery probe's starting state
                pitch=float(np.pi) if spec.get("flipped") else 0.0,
            )
    shared_info["drill_scenario"] = str(scenario.get("name", "scenario"))
    return ball_position
