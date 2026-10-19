"""Convert RLBot v5 ``GamePacket`` data into the training observation.

The converter builds the same RLGym ``GameState`` structures the simulator
produces and then calls the *same* observation builder used in training.
This is the parity contract between simulation and the real game.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.api.game_config import GameConfig
from rlgym.rocket_league.math import euler_to_rotation

from ..modes import ModeSpec, get_mode
from ..obs import OBS_VERSION, build_observation
from .pad_order import CANONICAL_PAD_COUNT, remap_pad_timers


def _vec(v: Any) -> np.ndarray:
    return np.array([float(v.x), float(v.y), float(v.z)], dtype=np.float32)


def _physics(obj: Any) -> PhysicsObject:
    physics = PhysicsObject()
    physics.position = _vec(obj.location)
    physics.linear_velocity = _vec(obj.velocity)
    physics.angular_velocity = _vec(obj.angular_velocity)
    rotation = obj.rotation
    physics.rotation_mtx = euler_to_rotation(
        np.array(
            [float(rotation.pitch), float(rotation.yaw), float(rotation.roll)], dtype=np.float32
        )
    )
    return physics


def _air_state_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    text = str(value)
    return text.split(".")[-1]


def packet_to_game_state(packet: Any, self_index: int) -> tuple[GameState, str, int]:
    """Build a GameState from a packet. Returns (state, own_agent_id, own_team)."""
    state = GameState()
    state.tick_count = int(getattr(packet.match_info, "frame_num", 0))
    state.goal_scored = False
    config = GameConfig()
    config.gravity = 1.0
    config.boost_consumption = 1.0
    config.dodge_deadzone = 0.5
    state.config = config
    state.cars = {}

    own_agent = ""
    own_team = 0
    for index, player in enumerate(packet.players):
        agent = f"player-{player.player_id}"
        car = Car()
        car.team_num = int(player.team)
        car.hitbox_type = 0
        car.ball_touches = 0
        car.physics = _physics(player.physics)
        car.demo_respawn_timer = max(0.0, float(getattr(player, "demolished_timeout", 0.0)))
        air = _air_state_name(getattr(player, "air_state", "OnGround"))
        car.wheels_with_contact = (air == "OnGround",) * 4
        car.supersonic_time = 1.0 if getattr(player, "is_supersonic", False) else 0.0
        car.boost_amount = float(getattr(player, "boost", 0.0))
        car.boost_active_time = (
            1.0 if getattr(getattr(player, "last_input", None), "boost", False) else 0.0
        )
        car.handbrake = (
            1.0 if getattr(getattr(player, "last_input", None), "handbrake", False) else 0.0
        )
        car.is_jumping = air == "Jumping"
        car.has_jumped = bool(getattr(player, "has_jumped", False))
        car.is_holding_jump = bool(getattr(getattr(player, "last_input", None), "jump", False))
        car.jump_time = 0.0
        car.has_flipped = bool(getattr(player, "has_dodged", False))
        car.has_double_jumped = bool(getattr(player, "has_double_jumped", False))
        car.air_time_since_jump = max(0.0, float(getattr(player, "dodge_elapsed", 0.0)))
        car.flip_time = 0.0
        car.flip_torque = np.zeros(3, dtype=np.float32)
        car.is_autoflipping = False
        car.autoflip_timer = 0.0
        car.autoflip_direction = 0.0
        car.bump_victim_id = None
        state.cars[agent] = car
        if index == self_index:
            own_agent = agent
            own_team = int(player.team)

    if not own_agent:
        raise ValueError(
            f"self_index {self_index} not present in packet with {len(packet.players)} players"
        )

    ball = PhysicsObject()
    ball.position = (
        _vec(packet.balls[0].physics.location) if packet.balls else np.zeros(3, dtype=np.float32)
    )
    ball.linear_velocity = (
        _vec(packet.balls[0].physics.velocity) if packet.balls else np.zeros(3, dtype=np.float32)
    )
    ball.angular_velocity = (
        _vec(packet.balls[0].physics.angular_velocity)
        if packet.balls
        else np.zeros(3, dtype=np.float32)
    )
    if packet.balls:
        rotation = packet.balls[0].physics.rotation
        ball.rotation_mtx = euler_to_rotation(
            np.array(
                [float(rotation.pitch), float(rotation.yaw), float(rotation.roll)], dtype=np.float32
            )
        )
    state.ball = ball

    raw = [
        0.0 if getattr(pad, "is_active", False) else float(getattr(pad, "timer", 0.0))
        for pad in packet.boost_pads
    ]
    if len(raw) != CANONICAL_PAD_COUNT:
        raise ValueError(
            f"the game reports {len(raw)} boost pads, but the policy was trained on the "
            f"standard stadium with {CANONICAL_PAD_COUNT}; the observation layout cannot "
            "be matched and the bot must not drive"
        )
    # RLBot orders the pads by y and then x, RocketSim does not: the four
    # slots around the centre circle are swapped.  Without this remap the
    # policy would read the timer of a different pad in those slots.
    state.boost_pad_timers = np.asarray(
        remap_pad_timers(raw), dtype=np.float32
    )
    return state, own_agent, own_team


def packet_to_observation(
    packet: Any,
    self_index: int,
    last_action: int | None,
    mode: str | ModeSpec = "soccer-1v1",
    obs_version: int = OBS_VERSION,
) -> tuple[np.ndarray, dict]:
    spec = get_mode(mode) if isinstance(mode, str) else mode
    state, agent, team = packet_to_game_state(packet, self_index)
    match_info = packet.match_info
    time_remaining = float(getattr(match_info, "game_time_remaining", 300.0))
    score_diff = 0
    teams = list(getattr(packet, "teams", []))
    if len(teams) >= 2:
        own = next((t for t in teams if int(t.team) == team), None)
        other = next((t for t in teams if int(t.team) != team), None)
        if own is not None and other is not None:
            score_diff = int(own.score) - int(other.score)
    obs = build_observation(
        state,
        agent,
        spec,
        last_action,
        time_remaining=max(0.0, min(time_remaining, 300.0)),
        score_diff=score_diff,
        obs_version=obs_version,
    )
    return obs, {"agent": agent, "team": team, "state": state}
