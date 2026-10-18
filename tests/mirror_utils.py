"""Helpers to mirror a whole ``GameState`` for side-symmetry tests.

Uses RLGym's own inversion primitives (``PhysicsObject.inverted()``,
``GameState.inverted_ball``, ``inverted_boost_pad_timers``) so the test checks
``rlai.obs.mirror_observation`` against the simulator's convention instead of
against a second, independently written guess.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rlgym.rocket_league.api import Car, GameState


def invert_state(state: Any, swap_teams: bool = True) -> GameState:
    """Return the scene rotated 180 deg about the vertical axis."""
    mirrored = GameState()
    mirrored.tick_count = int(state.tick_count)
    mirrored.goal_scored = bool(state.goal_scored)
    mirrored.config = state.config
    mirrored.cars = {}
    for agent, car in state.cars.items():
        new_car = Car()
        new_car.team_num = (1 - int(car.team_num)) if swap_teams else int(car.team_num)
        new_car.hitbox_type = car.hitbox_type
        new_car.ball_touches = car.ball_touches
        new_car.bump_victim_id = car.bump_victim_id
        new_car.demo_respawn_timer = car.demo_respawn_timer
        new_car.wheels_with_contact = car.wheels_with_contact
        new_car.supersonic_time = car.supersonic_time
        new_car.boost_amount = car.boost_amount
        new_car.boost_active_time = car.boost_active_time
        new_car.handbrake = car.handbrake
        new_car.is_jumping = car.is_jumping
        new_car.has_jumped = car.has_jumped
        new_car.is_holding_jump = car.is_holding_jump
        new_car.jump_time = car.jump_time
        new_car.has_flipped = car.has_flipped
        new_car.has_double_jumped = car.has_double_jumped
        new_car.air_time_since_jump = car.air_time_since_jump
        new_car.flip_time = car.flip_time
        new_car.flip_torque = np.asarray(car.flip_torque, dtype=np.float32) * np.array(
            [-1.0, -1.0, 1.0], dtype=np.float32
        )
        new_car.is_autoflipping = car.is_autoflipping
        new_car.autoflip_timer = car.autoflip_timer
        new_car.autoflip_direction = car.autoflip_direction
        new_car.physics = car.physics.inverted()
        mirrored.cars[agent] = new_car
    mirrored.ball = state.ball.inverted()
    mirrored.boost_pad_timers = np.ascontiguousarray(
        np.asarray(state.boost_pad_timers, dtype=np.float32)[::-1]
    )
    return mirrored
