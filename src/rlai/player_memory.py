"""Player Memory: what the agent remembers about the players it meets.

The memory is a *sidecar* of the training data, never part of the model:
the weights stay a pure function of the observation, and the memory is
plain data that can be viewed, exported, reset and deleted.

What is stored per player:

* how often the player was met,
* the result of those meetings (wins/losses/draws from the agent's side),
* the goals for and against,
* the first and last meeting,
* a free-form note.

Two limits keep the memory from growing without bound:

* ``MAX_ENTRIES`` - the number of players that are kept,
* ``MAX_AGE_DAYS`` - how long an untouched entry is kept.

Both are enforced on write, and the module reports what was pruned.  The
memory is stored in its own SQLite table, so deleting it never touches a
checkpoint, a run or an evaluation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

#: how many players are kept
MAX_ENTRIES = 500
#: how long an entry is kept without a new meeting
MAX_AGE_DAYS = 90.0
#: how long a note may be
MAX_NOTE_LENGTH = 500


@dataclass
class PlayerRecord:
    """One remembered player."""

    name: str
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    goals_for: int = 0
    goals_against: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    note: str = ""
    #: which opponent/difficulty the player was met as
    tags: list[str] = field(default_factory=list)

    @property
    def win_rate(self) -> float | None:
        if not self.games:
            return None
        return self.wins / self.games

    def to_dict(self) -> dict:
        data = asdict(self)
        data["win_rate"] = self.win_rate
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PlayerRecord":
        known = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        record = cls(**known)
        record.tags = list(record.tags or [])
        return record


class PlayerMemory:
    """The store of player records.

    The class holds the data and the rules; the persistence lives in
    :meth:`rlai.store.Store` so the memory uses the same database as
    everything else - but in its own table.
    """

    def __init__(self, *, max_entries: int = MAX_ENTRIES, max_age_days: float = MAX_AGE_DAYS):
        self.max_entries = max(1, int(max_entries))
        self.max_age_days = float(max_age_days)
        self.records: dict[str, PlayerRecord] = {}

    # -- mutation ----------------------------------------------------------
    def remember(
        self,
        name: str,
        *,
        won: bool | None = None,
        goals_for: int = 0,
        goals_against: int = 0,
        tag: str | None = None,
        timestamp: float | None = None,
    ) -> PlayerRecord:
        """Record a meeting with a player."""
        key = str(name or "").strip()
        if not key:
            raise ValueError("a player name is required")
        now = float(timestamp if timestamp is not None else time.time())
        record = self.records.get(key)
        if record is None:
            record = PlayerRecord(name=key, first_seen=now)
            self.records[key] = record
        record.games += 1
        if won is True:
            record.wins += 1
        elif won is False:
            record.losses += 1
        else:
            record.draws += 1
        record.goals_for += int(goals_for)
        record.goals_against += int(goals_against)
        record.last_seen = now
        if tag and tag not in record.tags:
            record.tags.append(tag)
        return record

    def set_note(self, name: str, note: str) -> PlayerRecord:
        """Attach a note to a player (the "per-bot info" of the GUI)."""
        key = str(name or "").strip()
        if key not in self.records:
            raise KeyError(f"unknown player {name!r}")
        record = self.records[key]
        record.note = str(note or "")[:MAX_NOTE_LENGTH]
        return record

    def forget(self, name: str) -> bool:
        """Delete one player."""
        return self.records.pop(str(name or "").strip(), None) is not None

    def reset(self) -> int:
        """Delete every player.  Returns how many were removed."""
        count = len(self.records)
        self.records.clear()
        return count

    # -- maintenance -------------------------------------------------------
    def prune(self, *, now: float | None = None) -> dict:
        """Apply the entry and age limits.  Returns what was removed."""
        moment = float(now if now is not None else time.time())
        removed_age = 0
        removed_count = 0
        cutoff = moment - self.max_age_days * 86400.0
        stale = [
            name
            for name, record in self.records.items()
            if record.last_seen < cutoff
        ]
        for name in stale:
            del self.records[name]
            removed_age += 1
        if len(self.records) > self.max_entries:
            oldest = sorted(self.records.items(), key=lambda item: item[1].last_seen)
            for name, _record in oldest[: len(self.records) - self.max_entries]:
                del self.records[name]
                removed_count += 1
        return {"removed_by_age": removed_age, "removed_by_count": removed_count}

    # -- export ------------------------------------------------------------
    def export(self) -> dict:
        """A portable snapshot of the memory."""
        return {
            "schema": "rlai-player-memory-v1",
            "exported_at": time.time(),
            "max_entries": self.max_entries,
            "max_age_days": self.max_age_days,
            "players": [record.to_dict() for record in self.records.values()],
        }

    def import_data(self, data: dict) -> int:
        """Load a snapshot.  Returns the number of restored players."""
        players = data.get("players") if isinstance(data, dict) else None
        if not isinstance(players, list):
            raise ValueError("the export does not contain a player list")
        restored = 0
        for item in players:
            record = PlayerRecord.from_dict(item)
            if not record.name:
                continue
            self.records[record.name] = record
            restored += 1
        self.prune()
        return restored

    def to_dict(self) -> dict:
        return {
            "count": len(self.records),
            "max_entries": self.max_entries,
            "max_age_days": self.max_age_days,
            "players": [
                record.to_dict()
                for record in sorted(
                    self.records.values(), key=lambda item: item.last_seen, reverse=True
                )
            ],
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "PlayerMemory":
        memory = cls()
        if isinstance(data, dict):
            for item in data.get("players") or []:
                record = PlayerRecord.from_dict(item)
                if record.name:
                    memory.records[record.name] = record
        return memory

    # -- statistics --------------------------------------------------------
    def summary(self) -> dict:
        """Aggregated numbers for the GUI."""
        records = list(self.records.values())
        games = sum(record.games for record in records)
        wins = sum(record.wins for record in records)
        losses = sum(record.losses for record in records)
        draws = sum(record.draws for record in records)
        return {
            "players": len(records),
            "games": games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": (wins / games) if games else None,
            "goals_for": sum(record.goals_for for record in records),
            "goals_against": sum(record.goals_against for record in records),
            "max_entries": self.max_entries,
            "max_age_days": self.max_age_days,
        }


def memory_from_rows(rows: list[dict]) -> PlayerMemory:
    """Build a memory from database rows."""
    memory = PlayerMemory()
    for row in rows:
        payload = row.get("data_json")
        if payload:
            try:
                record = PlayerRecord.from_dict(json.loads(payload))
            except (json.JSONDecodeError, TypeError):
                continue
        else:
            record = PlayerRecord(
                name=str(row.get("name") or ""),
                games=int(row.get("games") or 0),
                wins=int(row.get("wins") or 0),
                losses=int(row.get("losses") or 0),
                draws=int(row.get("draws") or 0),
                goals_for=int(row.get("goals_for") or 0),
                goals_against=int(row.get("goals_against") or 0),
                first_seen=float(row.get("first_seen") or 0.0),
                last_seen=float(row.get("last_seen") or 0.0),
            )
        if record.name:
            memory.records[record.name] = record
    return memory


def rows_from_memory(memory: PlayerMemory) -> list[dict[str, Any]]:
    """The database rows for a memory."""
    return [
        {
            "name": record.name,
            "games": record.games,
            "wins": record.wins,
            "losses": record.losses,
            "draws": record.draws,
            "goals_for": record.goals_for,
            "goals_against": record.goals_against,
            "first_seen": record.first_seen,
            "last_seen": record.last_seen,
            "data_json": json.dumps(record.to_dict()),
        }
        for record in memory.records.values()
    ]
