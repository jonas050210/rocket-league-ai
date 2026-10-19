"""Team Builder: configure a team of bots and their opponents.

A *team* is a list of bot slots.  Every slot has

* a name (shown in the GUI),
* a difficulty level (see :mod:`rlai.difficulty`),
* a behaviour profile (see :mod:`rlai.botbrain`),
* an optional model reference - a trained checkpoint the bot drives.

The module keeps the data model and the validation.  The *behaviour* of a
built team is executed by the opponents module (a scripted opponent with a
difficulty level) and, for model-backed bots, by the policy itself.

Team Builder is deliberately independent of the training configuration: a
team is a description that can be stored, edited, compared and used as the
opponent of a training run or of an evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .botbrain import BotBrain, PROFILE_NAMES
from .difficulty import DIFFICULTY_NAMES, brain_parameters, get_profile

#: a team may not have more bots than the biggest supported mode
MAX_TEAM_SIZE = 3
#: a team needs at least one bot
MIN_TEAM_SIZE = 1


@dataclass
class BotSlot:
    """One bot in a team."""

    name: str
    difficulty: str = "medium"
    behaviour: str = "balanced"
    #: checkpoint id or path of a trained policy (``None`` = scripted bot)
    model: str | None = None
    #: per-bot overrides of the behaviour parameters
    overrides: dict[str, float] = field(default_factory=dict)
    #: the role inside the team (striker, defender, ...)
    role: str = "flex"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BotSlot":
        name = str(data.get("name") or "bot").strip() or "bot"
        difficulty = str(data.get("difficulty") or "medium").strip().lower()
        behaviour = str(data.get("behaviour") or "balanced").strip().lower()
        if difficulty not in DIFFICULTY_NAMES:
            raise ValueError(
                f"unknown difficulty {difficulty!r} for bot {name!r}; "
                f"use one of {sorted(DIFFICULTY_NAMES)}"
            )
        if behaviour not in PROFILE_NAMES:
            raise ValueError(
                f"unknown behaviour {behaviour!r} for bot {name!r}; "
                f"use one of {sorted(PROFILE_NAMES)}"
            )
        overrides = data.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise ValueError(f"overrides of bot {name!r} must be an object")
        return cls(
            name=name,
            difficulty=difficulty,
            behaviour=behaviour,
            model=(str(data["model"]) if data.get("model") else None),
            overrides={str(key): float(value) for key, value in overrides.items()},
            role=str(data.get("role") or "flex"),
        )


@dataclass
class Team:
    """A named team of bots."""

    name: str
    bots: list[BotSlot] = field(default_factory=list)
    mode: str = "soccer-1v1"
    description: str = ""
    #: the team this one was derived from (for the "copy" workflow)
    parent: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mode": self.mode,
            "description": self.description,
            "parent": self.parent,
            "bots": [bot.to_dict() for bot in self.bots],
            "size": len(self.bots),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Team":
        name = str(data.get("name") or "team").strip() or "team"
        bots = [BotSlot.from_dict(item) for item in (data.get("bots") or [])]
        if len(bots) < MIN_TEAM_SIZE:
            raise ValueError(f"team {name!r} needs at least {MIN_TEAM_SIZE} bot")
        if len(bots) > MAX_TEAM_SIZE:
            raise ValueError(
                f"team {name!r} has {len(bots)} bots, at most {MAX_TEAM_SIZE} are supported"
            )
        return cls(
            name=name,
            bots=bots,
            mode=str(data.get("mode") or "soccer-1v1"),
            description=str(data.get("description") or ""),
            parent=(str(data["parent"]) if data.get("parent") else None),
        )


def validate_team(data: dict) -> Team:
    """Build a :class:`Team` from user input, with clear error messages."""
    return Team.from_dict(data)


def describe_team(team: Team) -> dict:
    """A team plus the behaviour each bot actually gets."""
    described = team.to_dict()
    described["details"] = [
        {
            "name": bot.name,
            "difficulty": bot.difficulty,
            "behaviour": bot.behaviour,
            "role": bot.role,
            "model": bot.model,
            "parameters": brain_parameters(bot.difficulty, bot.behaviour).to_dict(),
            "overrides": dict(bot.overrides),
        }
        for bot in team.bots
    ]
    described["average_difficulty_target"] = (
        sum(get_profile(bot.difficulty).target_win_rate for bot in team.bots) / len(team.bots)
        if team.bots
        else None
    )
    described["target_note"] = (
        "the difficulty percentages are design targets, not a measurement"
    )
    return described


def team_brains(team: Team) -> list[BotBrain]:
    """The behaviour brain of every bot in the team."""
    brains = []
    for bot in team.bots:
        parameters = brain_parameters(bot.difficulty, bot.behaviour)
        if bot.overrides:
            from .botbrain import merge_overrides

            parameters = merge_overrides(parameters, bot.overrides)
        brains.append(BotBrain(profile=bot.behaviour, parameters=parameters))
    return brains


def default_team(mode: str = "soccer-1v1") -> Team:
    """A sensible starting team for the given mode."""
    size = 1
    if mode.endswith("2v2"):
        size = 2
    elif mode.endswith("3v3"):
        size = 3
    return Team(
        name=f"{mode} default",
        mode=mode,
        description="Created by the team builder.",
        bots=[
            BotSlot(name=f"bot-{index + 1}", difficulty="medium", behaviour="balanced")
            for index in range(size)
        ],
    )


def opponent_names(team: Team) -> list[str]:
    """The opponent references a team implies.

    Model-backed bots are referenced by their checkpoint; scripted bots by
    their difficulty level, so an evaluation can address them.
    """
    names = []
    for bot in team.bots:
        if bot.model:
            names.append(f"model:{bot.model}")
        else:
            names.append(f"difficulty:{bot.difficulty}")
    return names


def summarise(teams: list[Team]) -> dict:
    """Overview for the GUI."""
    return {
        "count": len(teams),
        "teams": [describe_team(team) for team in teams],
        "difficulties": [get_profile(name).to_dict() for name in DIFFICULTY_NAMES],
    }


def team_from_any(data: Any) -> Team:
    """Accept a dict or a :class:`Team`."""
    if isinstance(data, Team):
        return data
    if isinstance(data, dict):
        return Team.from_dict(data)
    raise ValueError(f"cannot build a team from {type(data).__name__}")
