"""Boost-pad ordering between RocketSim and RLBot.

The observation carries one timer per boost pad.  If the two systems index
the pads differently, the policy reads the timer of a *different* pad - a
silent error that no synthetic-packet test built from the simulator's own
order could ever catch.

These tests pin the mapping down against the live simulator.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlai.envs import MatchEnv
from rlai.modes import get_mode
from rlai.opponents import IdleOpponent
from rlai.rlbot.pad_order import (
    CANONICAL_PAD_COUNT,
    CANONICAL_PAD_POSITIONS,
    ROCKETSIM_TO_RLBOT_INDEX,
    field_info_matches_canonical,
    remap_pad_timers,
    rlbot_order,
    rocketsim_to_rlbot_index,
    verify_constant,
)


def test_canonical_layout_has_the_expected_pad_count():
    assert len(CANONICAL_PAD_POSITIONS) == CANONICAL_PAD_COUNT
    assert CANONICAL_PAD_COUNT == 34


def test_permutation_is_a_bijection():
    assert sorted(ROCKETSIM_TO_RLBOT_INDEX) == list(range(CANONICAL_PAD_COUNT))


def test_permutation_matches_the_measured_layout():
    """The shipped permutation must be the one the layout implies."""
    assert rocketsim_to_rlbot_index(CANONICAL_PAD_POSITIONS) == ROCKETSIM_TO_RLBOT_INDEX


def test_permutation_agrees_with_the_live_simulator():
    """Recompute the permutation from a real RocketSim arena."""
    report = verify_constant()
    if not report.get("available"):
        pytest.skip(report.get("reason", "RocketSim arena unavailable"))
    assert report["ok"], report["differences"]
    assert report["pad_count"] == CANONICAL_PAD_COUNT


def test_rlbot_order_is_by_y_then_x():
    order = rlbot_order(CANONICAL_PAD_POSITIONS)
    keys = [(CANONICAL_PAD_POSITIONS[i][1], CANONICAL_PAD_POSITIONS[i][0]) for i in order]
    assert keys == sorted(keys)


def test_the_two_orders_really_differ():
    """Guard the premise: the remap is not a no-op.

    If a future stadium made the orders identical, the test above would
    still pass - this one makes the difference explicit.
    """
    identity = tuple(range(CANONICAL_PAD_COUNT))
    assert ROCKETSIM_TO_RLBOT_INDEX != identity


def test_remap_moves_exactly_the_central_pads():
    values = list(range(CANONICAL_PAD_COUNT))
    remapped = remap_pad_timers(values)
    changed = [index for index, (a, b) in enumerate(zip(values, remapped)) if a != b]
    assert sorted(changed) == [13, 14, 19, 20]


def test_remap_rejects_a_wrong_number_of_pads():
    with pytest.raises(ValueError, match="boost pads"):
        remap_pad_timers([0.0] * 10)


def test_adapter_preserves_the_simulator_order():
    """The adapter must deliver the simulator's own pad timers."""
    spec = get_mode("soccer-1v1")
    env = MatchEnv(spec, IdleOpponent(), seed=5, episode_seconds=30)
    env.reset()
    for _ in range(2):
        env.step({agent: 0 for agent in env.blue_agents})
    # make the timers distinguishable
    expected = np.linspace(0.0, 3.3, CANONICAL_PAD_COUNT).astype(np.float32)
    env.state.boost_pad_timers = expected.copy()

    from tests.packet_utils import packet_from_env

    packet = packet_from_env(env)
    from rlai.rlbot.obs_adapter import packet_to_game_state

    state, _agent, _team = packet_to_game_state(packet, 0)
    assert np.allclose(state.boost_pad_timers, expected, atol=1e-5)


def test_adapter_rejects_a_foreign_stadium():
    spec = get_mode("soccer-1v1")
    env = MatchEnv(spec, IdleOpponent(), seed=5, episode_seconds=30)
    env.reset()
    from tests.packet_utils import packet_from_env

    packet = packet_from_env(env)
    packet.boost_pads = packet.boost_pads[:20]
    from rlai.rlbot.obs_adapter import packet_to_game_state

    with pytest.raises(ValueError, match="boost pads"):
        packet_to_game_state(packet, 0)


def test_field_info_check_accepts_the_standard_stadium():
    from tests.packet_utils import field_info_from_env

    spec = get_mode("soccer-1v1")
    env = MatchEnv(spec, IdleOpponent(), seed=5, episode_seconds=30)
    env.reset()
    result = field_info_matches_canonical(field_info_from_env(env).boost_pads)
    assert result["available"] is True
    assert result["ok"] is True


def test_field_info_check_rejects_a_foreign_stadium():
    class Foreign:
        location = type("L", (), {"x": 1.0, "y": 2.0, "z": 3.0})()
        is_full_boost = True

    result = field_info_matches_canonical([Foreign()])
    assert result["available"] is True
    assert result["ok"] is False
