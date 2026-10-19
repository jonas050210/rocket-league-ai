"""Boost-pad ordering between RocketSim and RLBot.

The training observation contains one timer per boost pad, indexed the way
RocketSim indexes them.  RLBot delivers the pad states in its own order.
Both orderings are documented and stable, but they are **not** the same:

* RLBot (v5) documents every boost-pad list - ``GamePacket.boost_pads`` and
  ``FieldInfo.boost_pads`` alike - as *"ordered by y-coordinate and then
  x-coordinate"*.
* RocketSim stores the pads in the order of the standard stadium data, which
  is *not* a pure (y, x) sort: the four pads around the centre circle are
  emitted as ``(0, ±1024), (±2048, ±1036)`` while a strict (y, x) sort
  yields ``(±2048, ±1036), (0, ±1024)``.

The difference affects exactly four of the 34 slots (13/14 and 19/20).
Slicing the packet without remapping therefore feeds the policy the timer
of the *wrong* boost pad for those four positions - a silent, hard-to-spot
error that no parity test built from a synthetic packet could detect.

This module owns the permutation and the canonical layout, so the mapping
is data instead of an assumption:

* :data:`CANONICAL_PAD_POSITIONS` is the standard-stadium layout, measured
  from RocketSim itself.
* :data:`ROCKETSIM_TO_RLBOT_INDEX` is the permutation that maps a RocketSim
  slot onto the matching RLBot slot.
* :func:`rocketsim_to_rlbot_index` recomputes the permutation from a live
  arena, so the test suite can prove the constant is still correct.
"""

from __future__ import annotations

from typing import Iterable, Sequence

#: number of boost pads on the standard stadium
CANONICAL_PAD_COUNT = 34

#: Boost pad positions of the standard stadium as RocketSim reports them.
#: (x, y, z, is_full_boost) - the layout the training observation is built
#: against.  Used to remap an RLBot packet onto the training slots and to
#: verify at runtime that the game runs on a compatible stadium.
CANONICAL_PAD_POSITIONS: tuple[tuple[float, float, float, bool], ...] = (
    (0.0, -4240.0, 70.0, False),
    (-1792.0, -4184.0, 70.0, False),
    (1792.0, -4184.0, 70.0, False),
    (-3072.0, -4096.0, 73.0, True),
    (3072.0, -4096.0, 73.0, True),
    (-940.0, -3308.0, 70.0, False),
    (940.0, -3308.0, 70.0, False),
    (0.0, -2816.0, 70.0, False),
    (-3584.0, -2484.0, 70.0, False),
    (3584.0, -2484.0, 70.0, False),
    (-1788.0, -2300.0, 70.0, False),
    (1788.0, -2300.0, 70.0, False),
    (-2048.0, -1036.0, 70.0, False),
    (0.0, -1024.0, 70.0, False),
    (2048.0, -1036.0, 70.0, False),
    (-3584.0, 0.0, 73.0, True),
    (-1024.0, 0.0, 70.0, False),
    (1024.0, 0.0, 70.0, False),
    (3584.0, 0.0, 73.0, True),
    (-2048.0, 1036.0, 70.0, False),
    (0.0, 1024.0, 70.0, False),
    (2048.0, 1036.0, 70.0, False),
    (-1788.0, 2300.0, 70.0, False),
    (1788.0, 2300.0, 70.0, False),
    (-3584.0, 2484.0, 70.0, False),
    (3584.0, 2484.0, 70.0, False),
    (0.0, 2816.0, 70.0, False),
    (-940.0, 3308.0, 70.0, False),
    (940.0, 3308.0, 70.0, False),
    (-3072.0, 4096.0, 73.0, True),
    (3072.0, 4096.0, 73.0, True),
    (-1792.0, 4184.0, 70.0, False),
    (1792.0, 4184.0, 70.0, False),
    (0.0, 4240.0, 70.0, False),
)

#: For every RocketSim slot: the RLBot slot that holds the same pad.
#: Computed from the layout above by sorting it the way RLBot does.
ROCKETSIM_TO_RLBOT_INDEX: tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    14, 13, 15, 16, 17, 18, 20, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
)


def rlbot_order(positions: Sequence[tuple[float, float, float, bool]]) -> list[int]:
    """Indices of ``positions`` in RLBot's (y, then x) order."""
    return sorted(
        range(len(positions)),
        key=lambda index: (positions[index][1], positions[index][0]),
    )


def rocketsim_to_rlbot_index(
    positions: Sequence[tuple[float, float, float, bool]] | None = None,
) -> tuple[int, ...]:
    """Permutation from a RocketSim layout to the matching RLBot slots.

    ``positions`` defaults to the standard stadium layout.  Passing the
    layout of a live arena proves that the shipped constant still matches
    the simulator.
    """
    layout = list(positions or CANONICAL_PAD_POSITIONS)
    order = rlbot_order(layout)
    # ``order[k]`` is the layout index that RLBot puts at slot ``k``; invert
    # it to get "for this RocketSim slot, which RLBot slot holds it?".
    inverse = [0] * len(order)
    for rlbot_slot, layout_index in enumerate(order):
        inverse[layout_index] = rlbot_slot
    return tuple(inverse)


def remap_pad_timers(
    values: Sequence[float],
    *,
    positions: Sequence[tuple[float, float, float, bool]] | None = None,
) -> list[float]:
    """Reorder RLBot pad values into the RocketSim (training) slot order."""
    permutation = rocketsim_to_rlbot_index(positions)
    if len(values) != len(permutation):
        raise ValueError(
            f"expected {len(permutation)} boost pads, got {len(values)}; the game is "
            "running on a stadium whose boost layout differs from the one the "
            "policy was trained on"
        )
    return [float(values[rlbot_slot]) for rlbot_slot in permutation]


def verify_constant() -> dict:
    """Check the shipped permutation against a live RocketSim arena.

    Returns a report; ``ok`` is ``False`` when the simulator's layout no
    longer matches the constant.  The test suite uses this so the mapping
    cannot silently rot.
    """
    positions = _live_arena_positions()
    if positions is None:
        return {
            "available": False,
            "reason": "RocketSim arena could not be created (meshes unavailable)",
        }
    computed = rocketsim_to_rlbot_index(positions)
    matches = computed == ROCKETSIM_TO_RLBOT_INDEX
    differences = [
        {"rocketsim_slot": index, "expected": expected, "computed": actual}
        for index, (expected, actual) in enumerate(
            zip(ROCKETSIM_TO_RLBOT_INDEX, computed)
        )
        if expected != actual
    ]
    return {
        "available": True,
        "ok": matches,
        "pad_count": len(positions),
        "differences": differences,
    }


def _live_arena_positions() -> list[tuple[float, float, float, bool]] | None:
    """Read the pad layout from a real RocketSim arena (may be unavailable)."""
    try:
        import os

        import RocketSim as rsim
        from rlgym.rocket_league.sim import rocketsim_engine as engine
    except Exception:  # pragma: no cover - simulator missing
        return None
    try:
        meshes = os.path.join(
            os.path.dirname(os.path.abspath(engine.__file__)), "collision_meshes"
        )
        try:
            rsim.init(meshes)
        except RuntimeError:
            # already initialised (RLGym does it on import) - fine
            pass
        arena = rsim.Arena(rsim.GameMode.SOCCAR)
        pads = arena.get_boost_pads()
        positions = []
        for pad in pads:
            location = pad.get_pos()
            positions.append(
                (
                    round(float(location.x), 3),
                    round(float(location.y), 3),
                    round(float(location.z), 3),
                    bool(pad.is_big),
                )
            )
        return positions
    except Exception:  # pragma: no cover - simulator missing
        return None


def field_info_positions(field_info: Iterable) -> list[tuple[float, float, float, bool]]:
    """Extract pad positions from an RLBot ``FieldInfo`` message.

    Used at runtime to confirm that the live game uses the stadium the
    policy was trained on.
    """
    positions: list[tuple[float, float, float, bool]] = []
    for pad in field_info:
        location = getattr(pad, "location", None)
        if location is None:
            continue
        positions.append(
            (
                round(float(location.x), 3),
                round(float(location.y), 3),
                round(float(location.z), 3),
                bool(getattr(pad, "is_full_boost", False)),
            )
        )
    return positions


def field_info_matches_canonical(field_info: Iterable) -> dict:
    """Compare the live stadium layout with the canonical training layout.

    The comparison is done on the *sorted* positions, so it checks the
    stadium, not the ordering (the ordering is handled by the permutation).
    """
    positions = field_info_positions(field_info)
    if not positions:
        return {
            "available": False,
            "reason": "the packet carried no boost pad positions",
        }
    if len(positions) != CANONICAL_PAD_COUNT:
        return {
            "available": True,
            "ok": False,
            "reason": (
                f"the game has {len(positions)} boost pads, the policy was trained "
                f"on the standard stadium with {CANONICAL_PAD_COUNT}"
            ),
        }
    canonical = sorted(
        (tuple(round(value, 1) for value in entry[:3]) + (entry[3],))
        for entry in CANONICAL_PAD_POSITIONS
    )
    live = sorted(
        (tuple(round(value, 1) for value in entry[:3]) + (entry[3],)) for entry in positions
    )
    ok = canonical == live
    return {
        "available": True,
        "ok": ok,
        "reason": None if ok else "the live stadium differs from the training stadium",
    }
