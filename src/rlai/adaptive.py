"""Adaptive Match Dynamics: controlled adaptation to the match situation.

"Adaptive" is a dangerous word: an agent that changes its behaviour without
a bound becomes unpredictable and cannot be evaluated.  The implementation
here is deliberately narrow and guarded:

* the adaptation only ever touches the **scripted** opponent's behaviour
  (mistake rate, reaction delay, boost efficiency).  A trained policy is
  never modified at runtime - its weights stay fixed, so an evaluation
  measures what the policy does, not what a wrapper did to it.
* every change is bounded by a configured minimum and maximum.
* the adaptation only reacts to *measured* results (the win rate of the
  last series), never to a single unlucky game.
* a hysteresis and a minimum number of games between two changes prevent
  oscillation.
* every change is logged, so a run's report can explain what happened.

The result is a system that can raise or lower a scripted opponent's
strength within explicit limits - and that can always explain itself.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

from .difficulty import DifficultyProfile, get_profile


@dataclass
class AdaptationLimits:
    """The bounds of the adaptation.  Nothing outside them is ever applied."""

    #: the opponent is never made easier than this
    min_mistake_rate: float = 0.0
    #: nor harder than this
    max_mistake_rate: float = 0.8
    min_reaction_delay: float = 0.0
    max_reaction_delay: float = 1.5
    min_boost_efficiency: float = 0.1
    max_boost_efficiency: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "AdaptationLimits":
        known = {
            key: value
            for key, value in (data or {}).items()
            if key in cls.__dataclass_fields__
        }
        return cls(**known)


@dataclass
class AdaptationPolicy:
    """When and how the opponent may be adapted."""

    #: the target win rate of the agent against the opponent
    target_win_rate: float = 0.6
    #: how far from the target the measured win rate may drift before a
    #: change is considered
    tolerance: float = 0.15
    #: at least this many games between two changes
    min_games_between_changes: int = 10
    #: at least this many seconds between two changes
    min_seconds_between_changes: float = 60.0
    #: how much a single change moves the parameters (0..1)
    step: float = 0.1
    #: the adaptation is off by default
    enabled: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "AdaptationPolicy":
        known = {
            key: value
            for key, value in (data or {}).items()
            if key in cls.__dataclass_fields__
        }
        return cls(**known)


@dataclass
class AdaptationEvent:
    """One change of the opponent's behaviour."""

    at: float
    reason: str
    measured_win_rate: float
    before: dict[str, float]
    after: dict[str, float]
    games: int

    def to_dict(self) -> dict:
        return asdict(self)


class AdaptiveController:
    """Decides whether the scripted opponent should be adapted.

    The controller never touches a policy.  It only produces a new set of
    behaviour parameters for a *scripted* opponent, and only when the
    measured result justifies it.
    """

    def __init__(
        self,
        difficulty: str = "medium",
        *,
        policy: AdaptationPolicy | dict | None = None,
        limits: AdaptationLimits | dict | None = None,
        clock: float = 0.0,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.difficulty = str(difficulty)
        self.policy = (
            policy
            if isinstance(policy, AdaptationPolicy)
            else AdaptationPolicy.from_dict(policy)
        )
        self.limits = (
            limits
            if isinstance(limits, AdaptationLimits)
            else AdaptationLimits.from_dict(limits)
        )
        self.clock = float(clock)
        #: the clock that decides "how long since the last change".  It is
        #: monotonic, so a system clock change cannot shorten the minimum
        #: time between two adaptations.
        self._monotonic = monotonic
        self.events: list[AdaptationEvent] = []
        self._last_change_at: float | None = None
        self._last_games = 0
        # The starting point is clamped as well: the limits describe the
        # opponent the run may use, so even the very first adaptation starts
        # inside them.
        self._current = self._clamp(self._parameters_for(self.difficulty))

    # -- parameters --------------------------------------------------------
    def _parameters_for(self, difficulty: str) -> dict[str, float]:
        profile: DifficultyProfile = get_profile(difficulty)
        return {
            "mistake_rate": profile.mistake_rate,
            "reaction_delay": profile.reaction_delay,
            "boost_efficiency": profile.boost_efficiency,
        }

    def _clamp(self, values: dict[str, float]) -> dict[str, float]:
        limits = self.limits
        return {
            "mistake_rate": max(
                limits.min_mistake_rate, min(limits.max_mistake_rate, values["mistake_rate"])
            ),
            "reaction_delay": max(
                limits.min_reaction_delay,
                min(limits.max_reaction_delay, values["reaction_delay"]),
            ),
            "boost_efficiency": max(
                limits.min_boost_efficiency,
                min(limits.max_boost_efficiency, values["boost_efficiency"]),
            ),
        }

    @property
    def current(self) -> dict[str, float]:
        """The parameters the scripted opponent should use right now."""
        return dict(self._current)

    # -- decision ----------------------------------------------------------
    def observe(
        self,
        *,
        wins: int,
        games: int,
        at: float | None = None,
    ) -> dict:
        """Feed a measured series and decide whether to adapt.

        Returns a result with the decision, the reason and the new
        parameters.  Nothing is changed when the policy is disabled, when
        the sample is too small, or when the win rate is inside the
        tolerance band.
        """
        moment = float(at if at is not None else self._monotonic())
        games = int(games)
        wins = int(wins)
        win_rate = (wins / games) if games else None
        result: dict[str, Any] = {
            "adapted": False,
            "reason": "",
            "measured_win_rate": win_rate,
            "games": games,
            "parameters": self.current,
        }
        if not self.policy.enabled:
            result["reason"] = "the adaptive match dynamics are disabled"
            return result
        if win_rate is None:
            result["reason"] = "no games were played, so nothing can be measured"
            return result
        if self._last_change_at is not None:
            if moment - self._last_change_at < self.policy.min_seconds_between_changes:
                result["reason"] = "too soon after the last change"
                return result
        if games - self._last_games < self.policy.min_games_between_changes:
            result["reason"] = "not enough new games for a reliable decision"
            return result

        target = self.policy.target_win_rate
        delta = win_rate - target
        if abs(delta) <= self.policy.tolerance:
            result["reason"] = (
                f"the measured win rate {win_rate:.0%} is inside the tolerance band "
                f"around the target {target:.0%}"
            )
            self._last_games = games
            return result

        # The agent is too strong -> make the opponent harder (fewer
        # mistakes, faster reaction, more efficient boost).  Too weak ->
        # make it easier.  Both directions are bounded by the limits.
        step = self.policy.step * (1.0 if delta > 0 else -1.0)
        candidate = dict(self._current)
        candidate["mistake_rate"] = candidate["mistake_rate"] - step
        candidate["reaction_delay"] = candidate["reaction_delay"] - step * 0.5
        candidate["boost_efficiency"] = candidate["boost_efficiency"] + step
        clamped = self._clamp(candidate)
        if clamped == self._current:
            result["reason"] = "the opponent is already at the limit of the configuration"
            self._last_games = games
            return result

        before = dict(self._current)
        self._current = clamped
        self._last_change_at = moment
        self._last_games = games
        event = AdaptationEvent(
            at=moment,
            reason=(
                f"measured win rate {win_rate:.0%} vs target {target:.0%}"
            ),
            measured_win_rate=win_rate,
            before=before,
            after=dict(clamped),
            games=games,
        )
        self.events.append(event)
        result.update(
            {
                "adapted": True,
                "reason": event.reason,
                "parameters": dict(clamped),
                "before": before,
                "event": event.to_dict(),
            }
        )
        return result

    # -- reporting ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "enabled": self.policy.enabled,
            "difficulty": self.difficulty,
            "policy": self.policy.to_dict(),
            "limits": self.limits.to_dict(),
            "current": self.current,
            "events": [event.to_dict() for event in self.events],
            "event_count": len(self.events),
            "note": (
                "the adaptation only changes the scripted opponent's behaviour "
                "within the configured limits; a trained policy is never modified"
            ),
        }

    def state(self) -> dict:
        """The state that has to survive a resume."""
        return {
            "difficulty": self.difficulty,
            "policy": self.policy.to_dict(),
            "limits": self.limits.to_dict(),
            "current": self.current,
            "events": [event.to_dict() for event in self.events],
            "last_change_at": self._last_change_at,
            "last_games": self._last_games,
        }

    def restore(self, state: dict | None) -> None:
        """Restore a state saved by :meth:`state`."""
        if not isinstance(state, dict):
            return
        self.difficulty = str(state.get("difficulty") or self.difficulty)
        self.policy = AdaptationPolicy.from_dict(state.get("policy"))
        self.limits = AdaptationLimits.from_dict(state.get("limits"))
        current = state.get("current")
        if isinstance(current, dict) and current:
            self._current = self._clamp(
                {
                    "mistake_rate": float(current.get("mistake_rate", 0.2)),
                    "reaction_delay": float(current.get("reaction_delay", 0.2)),
                    "boost_efficiency": float(current.get("boost_efficiency", 0.8)),
                }
            )
        self.events = [
            AdaptationEvent(**event)
            for event in (state.get("events") or [])
            if isinstance(event, dict)
        ]
        self._last_change_at = state.get("last_change_at")
        self._last_games = int(state.get("last_games") or 0)
