"""Bot Brain: the behaviour layer of a bot.

A *brain* decides **what** a bot wants to do; the *action mapping* decides
how that is expressed as controller inputs.  Keeping the two apart is what
makes the behaviour configurable without touching the policy, the action
table or the observation layout.

The four shipped profiles are the ones the product asks for:

``aggressive``
    push forward, boost often, shoot whenever the geometry allows.
``defensive``
    stay between the ball and the own goal, clear threats, boost rarely.
``balanced``
    the default: chase, rotate back when the opponent is closer.
``adaptive``
    the balanced profile with a context term: the behaviour changes with
    the score difference and the time left in the match.

Every parameter is a plain number with a documented range, so a profile can
be stored, edited, compared and - most importantly - *measured*.  A profile
is a behaviour description, never a claim about skill; the skill of a bot
is what the evaluation says it is.

The brain is deliberately free of any dependency on the simulator: it reads
a small, explicit :class:`BrainContext` and returns a
:class:`BrainDecision`.  That makes it testable without RocketSim and
reusable by the team builder, the difficulty profiles and the live bot.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterable

#: The behaviour profiles the GUI offers.
PROFILE_NAMES = ("aggressive", "defensive", "balanced", "adaptive")


@dataclass
class BrainParameters:
    """Tunable knobs of a behaviour profile.

    The values are the *balanced* defaults; every profile is a set of
    deviations from them.
    """

    #: 0 = never leave the defensive half, 1 = always attack
    aggression: float = 0.5
    #: how often the bot boosts when the geometry is good (0..1)
    boost_eagerness: float = 0.6
    #: how strongly the bot returns to the defensive position (0..1)
    rotation: float = 0.6
    #: 0 = keep the ball, 1 = shoot whenever a shot is possible
    shot_preference: float = 0.5
    #: how much the bot values a clear over a risky dribble (0..1)
    safety: float = 0.6
    #: how much the score and the clock change the behaviour (0..1); only
    #: the adaptive profile uses this
    context_sensitivity: float = 0.0
    #: reaction delay in seconds applied to the chosen action
    reaction_delay: float = 0.0
    #: probability that a chosen action is replaced by a weaker one
    mistake_rate: float = 0.0

    def clamped(self) -> "BrainParameters":
        """Return a copy with every value inside its documented range."""
        return BrainParameters(
            aggression=_clamp(self.aggression, 0.0, 1.0),
            boost_eagerness=_clamp(self.boost_eagerness, 0.0, 1.0),
            rotation=_clamp(self.rotation, 0.0, 1.0),
            shot_preference=_clamp(self.shot_preference, 0.0, 1.0),
            safety=_clamp(self.safety, 0.0, 1.0),
            context_sensitivity=_clamp(self.context_sensitivity, 0.0, 1.0),
            reaction_delay=_clamp(self.reaction_delay, 0.0, 2.0),
            mistake_rate=_clamp(self.mistake_rate, 0.0, 1.0),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "BrainParameters":
        data = dict(data or {})
        known = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        return cls(**known).clamped()


def _clamp(value: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, number))


#: The four shipped profiles.
PROFILES: dict[str, BrainParameters] = {
    "aggressive": BrainParameters(
        aggression=0.9,
        boost_eagerness=0.9,
        rotation=0.35,
        shot_preference=0.8,
        safety=0.3,
        context_sensitivity=0.0,
    ),
    "defensive": BrainParameters(
        aggression=0.15,
        boost_eagerness=0.3,
        rotation=0.9,
        shot_preference=0.25,
        safety=0.9,
        context_sensitivity=0.0,
    ),
    "balanced": BrainParameters(
        aggression=0.5,
        boost_eagerness=0.6,
        rotation=0.6,
        shot_preference=0.5,
        safety=0.6,
        context_sensitivity=0.0,
    ),
    "adaptive": BrainParameters(
        aggression=0.5,
        boost_eagerness=0.6,
        rotation=0.6,
        shot_preference=0.5,
        safety=0.6,
        context_sensitivity=0.7,
    ),
}

PROFILE_DESCRIPTIONS = {
    "aggressive": "Pushes forward, boosts often, shoots whenever it can.",
    "defensive": "Holds the defensive half, clears threats, rarely boosts.",
    "balanced": "Chases the ball and rotates back when the opponent is closer.",
    "adaptive": "Balanced behaviour that reacts to the score and the clock.",
}


@dataclass
class BrainContext:
    """What the brain knows about the current situation."""

    #: own goal - own position (metres)
    distance_to_ball: float = 1000.0
    #: signed distance from the ball to the own goal (positive = own half)
    ball_ball_to_own_goal: float = 0.0
    #: signed score difference from the bot's point of view
    score_diff: int = 0
    #: seconds left in the match
    time_remaining: float = 300.0
    #: boost in the tank (0..100)
    boost: float = 33.0
    #: is the bot on the ground?
    on_ground: bool = True
    #: can the bot shoot right now?
    can_shoot: bool = False
    #: is the ball in the opponent's half?
    ball_in_attacking_third: bool = False
    #: how many opponents are closer to the ball than the bot
    opponents_closer: int = 0


@dataclass
class BrainDecision:
    """What the brain decided."""

    #: what the bot wants to do
    intent: str
    #: the parameters the decision was based on
    parameters: dict[str, float] = field(default_factory=dict)
    #: a human readable explanation
    reason: str = ""
    #: 0..1, how confident the brain is
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)


class BotBrain:
    """A behaviour profile plus the logic that turns a context into a decision.

    The brain is a pure function: same context and parameters in, same
    decision out.  It never touches the simulator, so it can be unit-tested
    and previewed in the GUI.
    """

    def __init__(
        self,
        profile: str = "balanced",
        parameters: BrainParameters | dict | None = None,
    ):
        key = str(profile or "balanced").strip().lower()
        if key not in PROFILES:
            raise ValueError(
                f"unknown behaviour profile {profile!r}; use one of {sorted(PROFILES)}"
            )
        self.profile = key
        if parameters is None:
            self.parameters = BrainParameters(**asdict(PROFILES[key]))
        elif isinstance(parameters, BrainParameters):
            self.parameters = parameters.clamped()
        else:
            self.parameters = BrainParameters.from_dict(parameters)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict | None) -> "BotBrain":
        data = dict(data or {})
        return cls(
            profile=data.get("profile", "balanced"),
            parameters=data.get("parameters"),
        )

    def to_dict(self) -> dict:
        return {
            "profile": self.profile,
            "parameters": self.parameters.to_dict(),
            "description": PROFILE_DESCRIPTIONS.get(self.profile, ""),
        }

    # -- behaviour ---------------------------------------------------------
    def decide(self, context: BrainContext) -> BrainDecision:
        """Choose an intent for the given situation."""
        params = self.parameters.clamped()
        sensitivity = params.context_sensitivity
        aggression = params.aggression
        safety = params.safety
        reason = ""

        if sensitivity > 0.0:
            aggression, safety, reason = self._apply_context(params, context)
        else:
            reason = f"profile {self.profile!r}"

        # A bot that is out of boost cannot boost; one that is on the ground
        # cannot flip.  The decision reflects what is *possible*.
        if context.boost < 5:
            aggression = min(aggression, 0.35)
            reason += "; low boost"

        if context.can_shoot and params.shot_preference > 0.45 and aggression > 0.4:
            intent = "shoot"
            confidence = min(1.0, params.shot_preference)
            reason += "; shot available"
        elif context.ball_in_attacking_third and aggression > 0.55:
            intent = "attack"
            confidence = 0.6 + 0.3 * aggression
        elif context.opponents_closer > 0 and params.rotation > 0.5 and safety > 0.4:
            intent = "rotate"
            confidence = 0.5 + 0.3 * params.rotation
        elif context.time_remaining < 30 and context.score_diff < 0 and aggression > 0.6:
            intent = "attack"
            confidence = 0.7
            reason += "; behind on time"
        else:
            intent = "chase"
            confidence = 0.5

        return BrainDecision(
            intent=intent,
            parameters={
                "aggression": round(aggression, 4),
                "safety": round(safety, 4),
                "boost_eagerness": params.boost_eagerness,
                "rotation": params.rotation,
                "shot_preference": params.shot_preference,
                "context_sensitivity": sensitivity,
            },
            reason=reason,
            confidence=round(confidence, 3),
        )

    def _apply_context(
        self, params: BrainParameters, context: BrainContext
    ) -> tuple[float, float, str]:
        """Shift aggression/safety with the score and the clock."""
        sensitivity = params.context_sensitivity
        aggression = params.aggression
        safety = params.safety
        reasons = []
        if context.score_diff > 0:
            aggression += 0.15 * sensitivity
            safety -= 0.1 * sensitivity
            reasons.append("ahead")
        elif context.score_diff < 0:
            aggression += 0.25 * sensitivity
            safety -= 0.2 * sensitivity
            reasons.append("behind")
        if context.time_remaining < 60:
            aggression += 0.1 * sensitivity
            reasons.append("late")
        elif context.time_remaining > 240:
            safety += 0.1 * sensitivity
            reasons.append("early")
        return (
            _clamp(aggression, 0.0, 1.0),
            _clamp(safety, 0.0, 1.0),
            "context: " + ", ".join(reasons) if reasons else "context: neutral",
        )

    # -- description -------------------------------------------------------
    def describe(self) -> dict:
        data = self.to_dict()
        data["profiles"] = {
            name: {
                **asdict(parameters),
                "description": PROFILE_DESCRIPTIONS.get(name, ""),
            }
            for name, parameters in PROFILES.items()
        }
        return data


def available_profiles() -> list[dict]:
    """The profiles the GUI offers, with their parameters."""
    return [
        {
            "name": name,
            "description": PROFILE_DESCRIPTIONS.get(name, ""),
            "parameters": asdict(parameters),
        }
        for name, parameters in PROFILES.items()
    ]


def merge_overrides(base: BrainParameters, overrides: dict | None) -> BrainParameters:
    """Apply per-bot overrides on top of a profile."""
    data = base.to_dict()
    for key, value in (overrides or {}).items():
        if key in data:
            data[key] = value
    return BrainParameters.from_dict(data)


def profiles_differ(left: dict | None, right: dict | None) -> list[str]:
    """Which parameters differ between two configurations."""
    left_params = BrainParameters.from_dict(left).to_dict()
    right_params = BrainParameters.from_dict(right).to_dict()
    return [
        key
        for key in left_params
        if not math.isclose(
            float(left_params[key]), float(right_params[key]), rel_tol=1e-9, abs_tol=1e-9
        )
    ]


def describe_intents(decisions: Iterable[BrainDecision]) -> dict:
    """Count the intents of a series of decisions (used by the analysis)."""
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.intent] = counts.get(decision.intent, 0) + 1
    return counts
