"""Build RLBot-shaped packets from simulator states for adapter tests.

The fake packet has to follow the *real* RLBot conventions, otherwise the
parity test cannot detect a mismatch between the simulator and the game:

* ``boost_pads`` are ordered by y-coordinate and then x-coordinate (RLBot
  documents this for both ``GamePacket`` and ``FieldInfo``), which is **not**
  the order RocketSim uses;
* every other field mirrors the shape of the flatbuffers message.

Ordering the pads the way RLBot does is what makes the adapter's remap
observable: a missing remap now fails the parity test instead of silently
passing.
"""

from __future__ import annotations

from rlgym.rocket_league.math import quat_to_euler, rotation_to_quaternion
from rlbot import flat

from rlai.rlbot.pad_order import (
    CANONICAL_PAD_POSITIONS,
    rlbot_order,
)


class Vec:
    def __init__(self, values):
        self.x, self.y, self.z = map(float, values)


class Rot:
    def __init__(self, values):
        self.pitch, self.yaw, self.roll = values


class Physics:
    def __init__(self, obj):
        self.location = Vec(obj.position)
        self.velocity = Vec(obj.linear_velocity)
        self.angular_velocity = Vec(obj.angular_velocity)
        self.rotation = Rot(quat_to_euler(rotation_to_quaternion(obj.rotation_mtx)))


class LastInput:
    def __init__(self, jump=False, boost=False, handbrake=False):
        self.jump, self.boost, self.handbrake = jump, boost, handbrake


class AirState:
    def __init__(self, name):
        self.name = name


class Player:
    def __init__(self, player_id, car):
        self.player_id = player_id
        self.team = int(car.team_num)
        self.physics = Physics(car.physics)
        self.boost = float(car.boost_amount)
        self.is_supersonic = bool(car.is_supersonic)
        self.has_jumped = bool(car.has_jumped)
        self.has_double_jumped = bool(car.has_double_jumped)
        self.has_dodged = bool(car.has_flipped)
        self.dodge_elapsed = float(car.air_time_since_jump)
        self.demolished_timeout = -1.0 if not car.is_demoed else float(car.demo_respawn_timer)
        air = "OnGround" if car.on_ground else ("Jumping" if car.is_jumping else "InAir")
        self.air_state = AirState(air)
        self.last_input = LastInput(car.is_holding_jump, False, car.handbrake > 0)


class Ball:
    def __init__(self, ball):
        self.physics = Physics(ball)


class Pad:
    def __init__(self, timer):
        self.is_active = timer <= 0
        self.timer = float(timer)


class PadInfo:
    """Static pad information as delivered in RLBot's ``FieldInfo``."""

    def __init__(self, position, is_full_boost):
        self.location = Vec(position)
        self.is_full_boost = bool(is_full_boost)


class Team:
    def __init__(self, team, score):
        self.team, self.score = team, score


class MatchInfo:
    def __init__(self, time_remaining, match_phase=None):
        self.game_time_remaining = time_remaining
        self.frame_num = 0
        self.match_phase = flat.MatchPhase.Active if match_phase is None else match_phase


class Packet:
    pass


def rlbot_pad_order():
    """RLBot's slot order (y, then x) for the standard stadium."""
    return rlbot_order(CANONICAL_PAD_POSITIONS)


def packet_from_env(env, scores=(0, 0), time_remaining=210.0):
    state = env.state
    packet = Packet()
    packet.players = [Player(i, car) for i, car in enumerate(state.cars.values())]
    packet.balls = [Ball(state.ball)]
    # RLBot delivers the pads in its own order, not in the simulator's.
    order = rlbot_pad_order()
    timers = list(state.boost_pad_timers)
    packet.boost_pads = [Pad(timers[index]) for index in order]
    packet.teams = [Team(0, scores[0]), Team(1, scores[1])]
    packet.match_info = MatchInfo(time_remaining)
    return packet


def field_info_from_env(env):
    """Build an RLBot ``FieldInfo``-shaped message for the live arena."""
    info = Packet()
    order = rlbot_pad_order()
    info.boost_pads = [
        PadInfo(CANONICAL_PAD_POSITIONS[index][:3], CANONICAL_PAD_POSITIONS[index][3])
        for index in order
    ]
    return info
