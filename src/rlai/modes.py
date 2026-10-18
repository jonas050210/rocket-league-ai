"""Game mode definitions.

All four supported configurations are described declaratively so that the
environment factory, observation layout, reward function, evaluation runner
and the GUI can share a single source of truth.  Nothing here branches on
"1v1" as a special case: team size and heatseeker rules are data.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True, slots=True)
class ModeSpec:
    id: str
    name: str
    team_size: int
    heatseeker: bool
    description: str
    default_match_seconds: int = 300
    default_episode_seconds: int = 60

    @property
    def players(self) -> int:
        return self.team_size * 2

    @property
    def infinite_boost(self) -> bool:
        return self.heatseeker

    @property
    def opponent_slots(self) -> int:
        """Maximum opponents represented in the observation (3v3 padding)."""
        return 3

    @property
    def teammate_slots(self) -> int:
        return 2

    def to_dict(self) -> dict:
        data = asdict(self)
        data.update(
            players=self.players,
            infinite_boost=self.infinite_boost,
            opponent_slots=self.opponent_slots,
            teammate_slots=self.teammate_slots,
        )
        return data


_MODES = {
    "soccer-1v1": ModeSpec(
        id="soccer-1v1",
        name="Standard 1v1",
        team_size=1,
        heatseeker=False,
        description="Soccar, one car per team. Baseline mode for the whole pipeline.",
    ),
    "soccer-2v2": ModeSpec(
        id="soccer-2v2",
        name="Standard 2v2",
        team_size=2,
        heatseeker=False,
        description="Soccar with two cars per team. Shared policy controls both blue cars.",
    ),
    "soccer-3v3": ModeSpec(
        id="soccer-3v3",
        name="Standard 3v3",
        team_size=3,
        heatseeker=False,
        description="Soccar with three cars per team. Observation pads to three slots.",
    ),
    "heatseeker-2v2": ModeSpec(
        id="heatseeker-2v2",
        name="Heatseeker 2v2",
        team_size=2,
        heatseeker=True,
        description=(
            "Heatseeker: the ball homes toward the goal of the team that last "
            "touched it, accelerates on every touch, and boost is unlimited."
        ),
    ),
}


def get_mode(mode_id: str) -> ModeSpec:
    try:
        return _MODES[mode_id]
    except KeyError:
        raise KeyError(
            f"Unknown mode {mode_id!r}. Available: {', '.join(sorted(_MODES))}"
        ) from None


def all_modes() -> list[ModeSpec]:
    return list(_MODES.values())


def mode_ids() -> list[str]:
    return list(_MODES)
