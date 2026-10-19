"""Rival System: named opponents the agent keeps a history with.

A *rival* is an opponent the agent knows by name: a trained checkpoint, a
scripted behaviour or a difficulty level.  The system keeps the match
history against every rival and answers the questions the GUI asks:

* against which rival does the current policy do best?
* how did the performance against a rival change over time?
* which rival is the current bottleneck?

Everything comes from *measured* evaluations.  A rival without a single
finished game is reported as unmeasured - never with a made-up number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class Rival:
    """A named opponent."""

    name: str
    #: what the rival is: a checkpoint, a difficulty level, a behaviour
    kind: str = "checkpoint"
    #: the reference (checkpoint id, difficulty name, ...)
    reference: str = ""
    description: str = ""
    #: the runs that evaluated this rival
    evaluations: int = 0
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    goals_for: int = 0
    goals_against: int = 0
    #: history of the win rate over time (chronological)
    history: list[dict] = field(default_factory=list)

    @property
    def win_rate(self) -> float | None:
        if not self.games:
            return None
        return self.wins / self.games

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "reference": self.reference,
            "description": self.description,
            "evaluations": self.evaluations,
            "games": self.games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "goals_for": self.goals_for,
            "goals_against": self.goals_against,
            "goal_diff": self.goal_diff,
            "win_rate": self.win_rate,
            "history": list(self.history),
        }


class RivalRegistry:
    """The rivals of one workspace, with their match history."""

    def __init__(self) -> None:
        self.rivals: dict[str, Rival] = {}

    # -- registry ----------------------------------------------------------
    def register(
        self,
        name: str,
        *,
        kind: str = "checkpoint",
        reference: str = "",
        description: str = "",
    ) -> Rival:
        key = str(name or "").strip()
        if not key:
            raise ValueError("a rival needs a name")
        rival = self.rivals.get(key)
        if rival is None:
            rival = Rival(
                name=key,
                kind=kind,
                reference=reference or key,
                description=description,
            )
            self.rivals[key] = rival
        else:
            rival.kind = kind or rival.kind
            if reference:
                rival.reference = reference
            if description:
                rival.description = description
        return rival

    def remove(self, name: str) -> bool:
        return self.rivals.pop(str(name or "").strip(), None) is not None

    def get(self, name: str) -> Rival | None:
        return self.rivals.get(str(name or "").strip())

    # -- history -----------------------------------------------------------
    def record(
        self,
        name: str,
        *,
        games: int,
        wins: int,
        losses: int,
        draws: int,
        goals_for: int = 0,
        goals_against: int = 0,
        timestamp: float | None = None,
        model_id: str | None = None,
        step: int | None = None,
    ) -> Rival:
        """Add the result of one evaluation series against a rival."""
        rival = self.register(name)
        rival.evaluations += 1
        rival.games += int(games)
        rival.wins += int(wins)
        rival.losses += int(losses)
        rival.draws += int(draws)
        rival.goals_for += int(goals_for)
        rival.goals_against += int(goals_against)
        rival.history.append(
            {
                "timestamp": timestamp,
                "games": int(games),
                "wins": int(wins),
                "losses": int(losses),
                "draws": int(draws),
                "win_rate": (wins / games) if games else None,
                "goal_diff": int(goals_for) - int(goals_against),
                "model_id": model_id,
                "step": step,
            }
        )
        return rival

    def build_from_evaluations(
        self, evaluations: Iterable[dict], *, opponent_key: str = "opponent"
    ) -> "RivalRegistry":
        """Create the registry from stored evaluation rows.

        The opponent of an evaluation is the reference of the rival, so a
        rival exists for every opponent that was ever measured.
        """
        for row in evaluations:
            name = str(row.get(opponent_key) or "unknown")
            details = row.get("details") or {}
            self.record(
                name,
                games=int(row.get("games") or 0),
                wins=int(row.get("wins") or 0),
                losses=int(row.get("losses") or 0),
                draws=int(row.get("draws") or 0),
                goals_for=int(row.get("goals_for") or 0),
                goals_against=int(row.get("goals_against") or 0),
                timestamp=row.get("created_at"),
                model_id=row.get("model_id"),
            )
            if details.get("protocol"):
                rival = self.rivals.get(name)
                if rival is not None and not rival.description:
                    rival.description = f"protocol: {details['protocol']}"
        return self

    # -- analysis ----------------------------------------------------------
    def ranking(self) -> list[dict]:
        """The rivals, best measured win rate first."""
        rows = [
            {
                "name": rival.name,
                "kind": rival.kind,
                "reference": rival.reference,
                "games": rival.games,
                "win_rate": rival.win_rate,
                "goal_diff": rival.goal_diff,
                "evaluations": rival.evaluations,
                "measured": rival.games > 0,
            }
            for rival in self.rivals.values()
        ]
        rows.sort(
            key=lambda row: (row["win_rate"] is not None, row["win_rate"] or 0.0),
            reverse=True,
        )
        return rows

    def bottleneck(self) -> dict:
        """The rival the agent struggles with most.

        Only rivals with a *measured* win rate count; a rival that was
        never measured is reported as such.
        """
        measured = [rival for rival in self.rivals.values() if rival.win_rate is not None]
        if not measured:
            return {
                "available": False,
                "reason": "no rival has a measured win rate yet",
            }
        worst = min(measured, key=lambda rival: float(rival.win_rate or 0.0))
        return {
            "available": True,
            "name": worst.name,
            "win_rate": worst.win_rate,
            "games": worst.games,
            "goal_diff": worst.goal_diff,
            "note": (
                "a low win rate can also mean the rival is stronger; compare the "
                "goal difference and the number of games before drawing a conclusion"
            ),
        }

    def trend(self, name: str) -> dict:
        """How the performance against one rival developed."""
        rival = self.get(name)
        if rival is None:
            return {"available": False, "reason": f"unknown rival {name!r}"}
        if not rival.history:
            return {
                "available": False,
                "reason": "this rival has no measured evaluation yet",
                "rival": rival.to_dict(),
            }
        points = [
            point
            for point in rival.history
            if point.get("win_rate") is not None
        ]
        first = points[0]["win_rate"] if points else None
        last = points[-1]["win_rate"] if points else None
        return {
            "available": True,
            "rival": rival.to_dict(),
            "first_win_rate": first,
            "last_win_rate": last,
            "delta": (round(last - first, 4) if first is not None and last is not None else None),
            "points": points,
        }

    def to_dict(self) -> dict:
        return {
            "count": len(self.rivals),
            "rivals": [rival.to_dict() for rival in self.rivals.values()],
            "ranking": self.ranking(),
            "bottleneck": self.bottleneck(),
        }


def describe_rival_rows(rows: list[dict]) -> list[dict[str, Any]]:
    """Compact rival rows for the GUI."""
    registry = RivalRegistry().build_from_evaluations(rows)
    return registry.ranking()
