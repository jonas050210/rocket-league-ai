"""Difficulty profiles for the bots of a team.

The product asks for four difficulty levels - *Rock 0 %*, *Tiny 50 %*,
*Medium 75 %* and *Pro 100 %*.  The percentages are **targets that need
validation**, not measured skill levels.  This module makes that explicit:

* every profile carries the target it was designed for,
* every profile also carries the *mechanism* that realises it (how often
  the bot errs, how slowly it reacts, how much of its boost it wastes),
* the measured win rate against a profile is only ever produced by an
  actual evaluation, and the API returns it separately from the target.

So the GUI can say "Medium (target 75 %)" and, once an evaluation exists,
"Medium: 78 % measured over 20 games".  It never claims the target is the
skill.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .botbrain import BrainParameters, PROFILES, PROFILE_DESCRIPTIONS

#: the four shipped difficulty levels
DIFFICULTY_NAMES = ("rock", "tiny", "medium", "pro")


@dataclass
class DifficultyProfile:
    """One difficulty level.

    ``target_win_rate`` is the *design target* of the level.  It is used to
    document the intent and to compare it with a measurement - never
    presented as a measured skill.
    """

    name: str
    label: str
    target_win_rate: float
    #: how often the bot makes a mistake (0..1)
    mistake_rate: float
    #: reaction delay in seconds
    reaction_delay: float
    #: fraction of the boost the bot actually uses (0..1)
    boost_efficiency: float
    #: the behaviour profile this level builds on
    behaviour: str = "balanced"
    description: str = ""
    #: the target is a design goal, not a measurement
    target_is_measured: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DifficultyProfile":
        known = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        return cls(**known)


DIFFICULTY_PROFILES: dict[str, DifficultyProfile] = {
    "rock": DifficultyProfile(
        name="rock",
        label="Rock",
        target_win_rate=0.0,
        mistake_rate=0.75,
        reaction_delay=1.2,
        boost_efficiency=0.15,
        behaviour="defensive",
        description=(
            "Almost never does the right thing: it drives at the ball, wastes "
            "its boost and misses the obvious shots."
        ),
    ),
    "tiny": DifficultyProfile(
        name="tiny",
        label="Tiny",
        target_win_rate=0.5,
        mistake_rate=0.4,
        reaction_delay=0.6,
        boost_efficiency=0.45,
        behaviour="balanced",
        description=(
            "Reaches the ball most of the time and occasionally scores, but "
            "the slow reaction and the wasted boost give away easy goals."
        ),
    ),
    "medium": DifficultyProfile(
        name="medium",
        label="Medium",
        target_win_rate=0.75,
        mistake_rate=0.18,
        reaction_delay=0.25,
        boost_efficiency=0.75,
        behaviour="balanced",
        description=(
            "Competent basics: it rotates, boosts at the right moments and "
            "loses some easy chances."
        ),
    ),
    "pro": DifficultyProfile(
        name="pro",
        label="Pro",
        target_win_rate=1.0,
        mistake_rate=0.02,
        reaction_delay=0.05,
        boost_efficiency=1.0,
        behaviour="aggressive",
        description=(
            "Plays the whole field: fast reactions, efficient boost and "
            "almost no mistakes."
        ),
    ),
}


def get_profile(name: str) -> DifficultyProfile:
    key = str(name or "").strip().lower()
    if key not in DIFFICULTY_PROFILES:
        raise ValueError(
            f"unknown difficulty {name!r}; use one of {sorted(DIFFICULTY_PROFILES)}"
        )
    return DIFFICULTY_PROFILES[key]


def list_profiles() -> list[dict]:
    """Every difficulty level with its target and its mechanism."""
    return [profile.to_dict() for profile in DIFFICULTY_PROFILES.values()]


def brain_parameters(difficulty: str, behaviour: str | None = None) -> BrainParameters:
    """The behaviour parameters a difficulty level implies.

    The difficulty level scales the *execution* (mistakes, reaction time,
    boost use) of a behaviour profile; the profile itself stays readable.
    """
    profile = get_profile(difficulty)
    base = PROFILES.get(behaviour or profile.behaviour, PROFILES["balanced"])
    parameters = BrainParameters(**asdict(base))
    parameters.mistake_rate = profile.mistake_rate
    parameters.reaction_delay = profile.reaction_delay
    parameters.boost_eagerness = round(
        min(1.0, parameters.boost_eagerness * profile.boost_efficiency), 4
    )
    return parameters.clamped()


def skill_estimate(
    difficulty: str,
    *,
    games: int | None = None,
    measured_win_rate: float | None = None,
) -> dict:
    """What may be said about the skill of a difficulty level.

    ``target`` is the design goal, ``measured`` is the (optional) result of
    an evaluation.  The two are never merged, and a level without a
    measurement says so.
    """
    profile = get_profile(difficulty)
    result: dict = {
        "difficulty": profile.name,
        "label": profile.label,
        "target_win_rate": profile.target_win_rate,
        "target_is_measured": False,
        "measured_win_rate": measured_win_rate,
        "measured_available": measured_win_rate is not None,
        "games": games,
        "description": profile.description,
    }
    if measured_win_rate is None:
        result["note"] = (
            "the percentage is a design target; it has not been measured against "
            "this profile yet"
        )
    else:
        delta = measured_win_rate - profile.target_win_rate
        result["delta_to_target"] = round(delta, 4)
        result["note"] = (
            f"measured over {games} game(s)" if games else "measured"
        )
    return result


def custom_profile(
    name: str,
    *,
    label: str | None = None,
    target_win_rate: float | None = None,
    mistake_rate: float = 0.2,
    reaction_delay: float = 0.2,
    boost_efficiency: float = 0.8,
    behaviour: str = "balanced",
) -> DifficultyProfile:
    """Build a custom level (used by the team builder)."""
    return DifficultyProfile(
        name=str(name).strip().lower() or "custom",
        label=label or str(name).strip().title(),
        target_win_rate=(
            float(target_win_rate) if target_win_rate is not None else 0.5
        ),
        mistake_rate=max(0.0, min(1.0, float(mistake_rate))),
        reaction_delay=max(0.0, min(2.0, float(reaction_delay))),
        boost_efficiency=max(0.0, min(1.0, float(boost_efficiency))),
        behaviour=behaviour if behaviour in PROFILES else "balanced",
        description="Custom difficulty level.",
        target_is_measured=False,
    )


def describe_all() -> dict:
    """The whole difficulty catalogue, for the GUI."""
    return {
        "profiles": list_profiles(),
        "targets_are_measured": False,
        "note": (
            "the percentages are design targets that still need validation; "
            "a measured win rate only appears after an evaluation against the "
            "level"
        ),
        "behaviours": {
            name: {
                **asdict(parameters),
                "description": PROFILE_DESCRIPTIONS.get(name, ""),
            }
            for name, parameters in PROFILES.items()
        },
    }
