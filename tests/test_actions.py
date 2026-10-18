"""Action table and controller mapping."""

from __future__ import annotations

import numpy as np
import pytest

from rlai.actions import ACTION_COUNT, ACTION_TABLE, action_for, controller_dict, validate_table


def test_table_shape_and_validation():
    validate_table()
    assert ACTION_COUNT == 90
    assert ACTION_TABLE.shape == (90, 8)


@pytest.mark.parametrize("index", [0, 1, 17, 44, 89])
def test_controller_dict_is_within_limits(index):
    payload = controller_dict(index)
    for key in ("throttle", "steer", "pitch", "yaw", "roll"):
        assert -1.0 <= payload[key] <= 1.0
    for key in ("jump", "boost", "handbrake"):
        assert isinstance(payload[key], bool)
    assert set(payload) == {
        "throttle",
        "steer",
        "pitch",
        "yaw",
        "roll",
        "jump",
        "boost",
        "handbrake",
    }


def test_out_of_range_action():
    with pytest.raises(IndexError):
        action_for(90)
    with pytest.raises(IndexError):
        action_for(-1)


def test_all_90_actions_map_to_controller_state():
    for index in range(ACTION_COUNT):
        payload = controller_dict(index)
        assert -1.0 <= payload["throttle"] <= 1.0
        assert -1.0 <= payload["steer"] <= 1.0
        assert -1.0 <= payload["pitch"] <= 1.0
        assert -1.0 <= payload["yaw"] <= 1.0
        assert -1.0 <= payload["roll"] <= 1.0
        for key in ("jump", "boost", "handbrake"):
            assert isinstance(payload[key], bool)


def test_every_action_is_distinct():
    rows = {tuple(row) for row in ACTION_TABLE.tolist()}
    assert len(rows) == ACTION_COUNT


def test_steering_and_boost_present():
    assert (ACTION_TABLE[:, 1] < 0).any() and (ACTION_TABLE[:, 1] > 0).any()
    assert (ACTION_TABLE[:, 6] == 1).any()
    assert (ACTION_TABLE[:, 5] == 1).any()  # jump exists


def test_controller_types_are_stable():
    payload = controller_dict(45)
    assert isinstance(payload["throttle"], float)
    assert np.isfinite(payload["steer"])
