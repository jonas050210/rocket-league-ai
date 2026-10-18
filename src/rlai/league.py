"""Opponent pool with Elo ratings for self-play.

Self-play without a rating system degenerates: the learner is matched against
random old snapshots, including much weaker ones, so half of the training
signal is "beat a bad copy of yourself".  This pool tracks an Elo rating per
snapshot and samples opponents with the probability of a *close* game
(PFSP-style weighting), which keeps the learning signal informative.

The pool deliberately stores at most a handful of snapshots: on a 2-core
machine, memory and pickling cost matter more than a large league.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_ELO = 1000.0


def freeze_state(state: Any) -> Any:
    """Return a genuinely immutable copy of a snapshot state.

    A pool entry must not change when the learner keeps training.  Torch
    tensors taken with ``detach().cpu()`` on a CPU model *alias the live
    parameters*, so the pool would silently degrade into "always play the
    current policy" and its Elo ratings would describe mirror matches.
    Non-tensor states (used by tests and by restored checkpoints) are copied
    shallowly, which is enough for plain data.
    """
    if state is None:
        return None
    if isinstance(state, dict):
        copied: dict[Any, Any] = {}
        for key, value in state.items():
            if hasattr(value, "detach") and hasattr(value, "to"):
                copied[key] = value.detach().to("cpu", copy=True).contiguous()
            elif isinstance(value, np.ndarray):
                copied[key] = value.copy()
            else:
                copied[key] = value
        return copied
    if isinstance(state, np.ndarray):
        return state.copy()
    return state


@dataclass
class PoolEntry:
    id: str
    step: int
    elo: float = DEFAULT_ELO
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    state: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    def summary(self) -> dict:
        return {
            "id": self.id,
            "step": self.step,
            "elo": round(self.elo, 1),
            "games": self.games,
            "win_rate": round(self.win_rate, 3),
        }


class OpponentPool:
    """Bounded pool of frozen policy snapshots."""

    def __init__(
        self,
        capacity: int = 8,
        *,
        initial_elo: float = DEFAULT_ELO,
        k_factor: float = 16.0,
        min_weight: float = 0.05,
    ):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = int(capacity)
        self.agent_elo = float(initial_elo)
        self.k_factor = float(k_factor)
        self.min_weight = float(min_weight)
        self.entries: list[PoolEntry] = []

    # -- maintenance -------------------------------------------------------
    def add(self, entry_id: str, step: int, state: dict[str, Any]) -> PoolEntry:
        entry = PoolEntry(
            id=str(entry_id), step=int(step), elo=self.agent_elo, state=freeze_state(state)
        )
        self.entries.append(entry)
        # newest entries are the most relevant; drop the oldest first
        while len(self.entries) > self.capacity:
            self.entries.pop(0)
        return entry

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def usable(self) -> int:
        """Entries that can actually be played.

        Ratings always survive a resume.  Weights survive too when the
        checkpoint was written with ``include_states=True`` (the default for
        ``league_states`` in the training config); older checkpoints carry
        ratings only, and those entries count for the Elo history until a new
        snapshot is added.
        """
        return sum(1 for entry in self.entries if entry.state is not None)

    # -- sampling ----------------------------------------------------------
    def _weight(self, entry: PoolEntry) -> float:
        # Probability that the current learner beats this snapshot.
        expected = 1.0 / (1.0 + 10 ** ((entry.elo - self.agent_elo) / 400.0))
        # PFSP: games near 50 % are the most informative.
        return max(expected * (1.0 - expected), self.min_weight)

    def sample(self, rng: np.random.Generator | None = None) -> PoolEntry:
        candidates = [entry for entry in self.entries if entry.state is not None]
        if not candidates:
            raise ValueError("opponent pool has no playable snapshots")
        weights = np.asarray([self._weight(entry) for entry in candidates], dtype=np.float64)
        weights /= weights.sum()
        rng = rng or np.random.default_rng()
        index = int(rng.choice(len(candidates), p=weights))
        return candidates[index]

    # -- ratings -----------------------------------------------------------
    def record(self, entry_id: str, wins: int, losses: int, draws: int) -> float:
        """Update Elo after a batch of games.  Returns the new opponent Elo."""
        entry = next((item for item in self.entries if item.id == entry_id), None)
        if entry is None:
            raise KeyError(f"unknown pool entry {entry_id!r}")
        games = wins + losses + draws
        if games <= 0:
            return entry.elo
        score = (wins + 0.5 * draws) / games
        expected = 1.0 / (1.0 + 10 ** ((entry.elo - self.agent_elo) / 400.0))
        delta = self.k_factor * (score - expected)
        self.agent_elo += delta
        entry.elo -= delta
        entry.games += games
        entry.wins += wins
        entry.losses += losses
        entry.draws += draws
        return entry.elo

    def snapshot(self) -> dict:
        return {
            "agent_elo": round(self.agent_elo, 1),
            "size": len(self.entries),
            "entries": [entry.summary() for entry in self.entries],
        }

    # -- serialisation -----------------------------------------------------
    def to_dict(self, *, include_states: bool = False, max_states: int | None = None) -> dict:
        """Serialise the pool.

        Ratings are always stored.  ``include_states=True`` additionally stores
        the frozen weights of the newest ``max_states`` snapshots (all of them by
        default) so a resumed self-play run faces its *actual* former selves
        instead of Elo ratings that point at nothing.  One snapshot costs about
        one policy (roughly 1 MB for a 256x256 network), which is why the
        behaviour is configurable.
        """
        limit = len(self.entries) if max_states is None else max(0, int(max_states))
        first_kept = len(self.entries) - limit if include_states else len(self.entries)
        entries = []
        for index, entry in enumerate(self.entries):
            record = {
                "id": entry.id,
                "step": entry.step,
                "elo": entry.elo,
                "games": entry.games,
                "wins": entry.wins,
                "losses": entry.losses,
                "draws": entry.draws,
                "has_state": bool(index >= first_kept and entry.state is not None),
            }
            if record["has_state"]:
                record["state"] = freeze_state(entry.state)
            entries.append(record)
        return {
            "capacity": self.capacity,
            "agent_elo": self.agent_elo,
            "states_included": any(record["has_state"] for record in entries),
            "entries": entries,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OpponentPool":
        pool = cls(capacity=int(data.get("capacity", 8)))
        pool.agent_elo = float(data.get("agent_elo", DEFAULT_ELO))
        for item in data.get("entries", []):
            state = item.get("state")
            entry = PoolEntry(
                id=str(item["id"]),
                step=int(item.get("step", 0)),
                elo=float(item.get("elo", DEFAULT_ELO)),
                games=int(item.get("games", 0)),
                wins=int(item.get("wins", 0)),
                losses=int(item.get("losses", 0)),
                draws=int(item.get("draws", 0)),
                state=freeze_state(state) if state is not None else None,
            )
            pool.entries.append(entry)
        return pool


def expected_score(elo_a: float, elo_b: float) -> float:
    """Convenience helper used by reports and tests."""
    return 1.0 / (1.0 + math.pow(10.0, (elo_b - elo_a) / 400.0))
