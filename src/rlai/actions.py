"""Discrete action table shared by training, evaluation and the RLBot bot.

The table is the well-known RLGym "lookup table" of 90 actions.  Keeping a
local copy (instead of importing a private RLGym attribute) lets the RLBot
adapter and the tests use exactly the same mapping as the simulator.
"""

from __future__ import annotations

import numpy as np

ACTIONS_PER_TICK = 8

#: index order: throttle, steer, pitch, yaw, roll, jump, boost, handbrake
ACTION_TABLE: np.ndarray = np.array(
    [
        # Ground actions
        *[
            [throttle or boost, steer, 0, steer, 0, 0, boost, handbrake]
            for throttle in (-1, 0, 1)
            for steer in (-1, 0, 1)
            for boost in (0, 1)
            for handbrake in (0, 1)
            if not (boost == 1 and throttle != 1)
        ],
        # Aerial actions (jump / flip)
        *[
            [
                boost,
                yaw,
                pitch,
                yaw,
                roll,
                jump,
                boost,
                (1 if jump == 1 and (pitch != 0 or yaw != 0 or roll != 0) else 0),
            ]
            for pitch in (-1, 0, 1)
            for yaw in (-1, 0, 1)
            for roll in (-1, 0, 1)
            for jump in (0, 1)
            for boost in (0, 1)
            if not (jump == 1 and yaw != 0)
            if not (pitch == 0 and roll == 0 and jump == 0)
        ],
    ],
    dtype=np.float32,
)

ACTION_COUNT = len(ACTION_TABLE)
ACTION_NAMES = (
    "throttle",
    "steer",
    "pitch",
    "yaw",
    "roll",
    "jump",
    "boost",
    "handbrake",
)


def action_for(index: int) -> np.ndarray:
    """Return the 8 float control values for an action index."""
    if not 0 <= index < ACTION_COUNT:
        raise IndexError(f"action index {index} outside [0, {ACTION_COUNT})")
    return ACTION_TABLE[index]


def controller_dict(index: int) -> dict[str, float | bool]:
    """Convert an action index into a controller payload.

    Used by the RLBot adapter and the live GUI preview.  Booleans follow
    RLBot's ``ControllerState`` semantics.
    """
    a = action_for(index)
    return {
        "throttle": float(a[0]),
        "steer": float(a[1]),
        "pitch": float(a[2]),
        "yaw": float(a[3]),
        "roll": float(a[4]),
        "jump": bool(a[5]),
        "boost": bool(a[6]),
        "handbrake": bool(a[7]),
    }


def validate_table() -> None:
    assert ACTION_TABLE.shape == (90, 8), ACTION_TABLE.shape
    assert np.isfinite(ACTION_TABLE).all()
    assert ACTION_TABLE[:, 0].min() >= -1 and ACTION_TABLE[:, 0].max() <= 1
    assert ACTION_TABLE[:, 5:8].max() <= 1
