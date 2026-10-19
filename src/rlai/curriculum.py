"""Staged curriculum for Rocket League training.

Each stage fixes the opponent *and* the criteria used to leave it.  Promotion
is driven by measured results over a window of finished episodes, never by
elapsed time: a stage is only left once the policy actually beats what it is
supposed to beat.

The stages are intentionally short.  A curriculum that requires millions of
steps before moving on is not a curriculum on a laptop; it is a single long
run with extra bookkeeping.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    name: str
    opponent: str  # idle | random | chase | mixed | self
    min_iterations: int
    win_rate: float
    min_episodes: int = 30
    window: int = 60

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check the stage's criteria.

        A stage with an impossible threshold would block the run forever,
        and one with a trivial threshold would skip the stage without
        training.  Both are configuration errors, not runtime surprises.
        """
        if not str(self.name).strip():
            raise ValueError("a curriculum stage needs a name")
        if not 0.0 <= float(self.win_rate) <= 1.0:
            raise ValueError(
                f"curriculum stage {self.name!r}: win_rate must be between 0 and 1, "
                f"got {self.win_rate}"
            )
        if int(self.min_iterations) < 0:
            raise ValueError(
                f"curriculum stage {self.name!r}: min_iterations must not be negative"
            )
        if int(self.min_episodes) < 1:
            raise ValueError(
                f"curriculum stage {self.name!r}: min_episodes must be at least 1"
            )
        if int(self.window) < int(self.min_episodes):
            raise ValueError(
                f"curriculum stage {self.name!r}: the window ({self.window}) must hold at "
                f"least min_episodes ({self.min_episodes}), otherwise the target can "
                "never be reached"
            )
        if self.opponent not in {"idle", "random", "chase", "mixed", "self"}:
            raise ValueError(
                f"curriculum stage {self.name!r}: unknown opponent {self.opponent!r}"
            )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "opponent": self.opponent,
            "min_iterations": self.min_iterations,
            "win_rate": self.win_rate,
            "min_episodes": self.min_episodes,
            "window": self.window,
        }


DEFAULT_STAGES: tuple[Stage, ...] = (
    Stage("ball-control", "idle", min_iterations=5, win_rate=0.6, min_episodes=20),
    Stage("chase", "chase", min_iterations=10, win_rate=0.45, min_episodes=40),
    Stage("mixed", "mixed", min_iterations=10, win_rate=0.45, min_episodes=40),
    Stage("self-play", "self", min_iterations=1, win_rate=1.0, min_episodes=1),
)

HEATSEEKER_STAGES: tuple[Stage, ...] = (
    Stage("ball-control", "idle", min_iterations=5, win_rate=0.6, min_episodes=20),
    Stage("chase", "chase", min_iterations=10, win_rate=0.5, min_episodes=40),
    Stage("self-play", "self", min_iterations=1, win_rate=1.0, min_episodes=1),
)

TEAM_STAGES: tuple[Stage, ...] = (
    Stage("chase", "chase", min_iterations=8, win_rate=0.45, min_episodes=30),
    Stage("mixed", "mixed", min_iterations=10, win_rate=0.45, min_episodes=40),
    Stage("self-play", "self", min_iterations=1, win_rate=1.0, min_episodes=1),
)


def stages_for_mode(mode_id: str) -> tuple[Stage, ...]:
    if mode_id.startswith("heatseeker"):
        return HEATSEEKER_STAGES
    if mode_id in {"soccer-2v2", "soccer-3v3"}:
        return TEAM_STAGES
    return DEFAULT_STAGES


class Curriculum:
    """Tracks the active stage and promotes on measured win rate."""

    def __init__(
        self,
        stages: tuple[Stage, ...] | list[Stage],
        *,
        start_index: int = 0,
        stage_iterations: int = 0,
        total_iterations: int = 0,
    ):
        if not stages:
            raise ValueError("curriculum needs at least one stage")
        self.stages = tuple(stages)
        self.index = int(max(0, min(start_index, len(self.stages) - 1)))
        # Restoring the counters keeps ``min_iterations`` meaningful across a
        # resumed run instead of restarting the stage clock at zero.
        self.stage_iterations = max(0, int(stage_iterations))
        self.total_iterations = max(0, int(total_iterations))
        self._window: deque[dict] = deque(maxlen=self.stages[self.index].window)
        self._window_stage = self.index

    # -- state -------------------------------------------------------------
    @property
    def stage(self) -> Stage:
        return self.stages[self.index]

    @property
    def finished(self) -> bool:
        return self.index >= len(self.stages) - 1

    def reset_window(self) -> None:
        self._window.clear()

    # -- training hooks ----------------------------------------------------
    def observe(self, episodes: list[dict]) -> str | None:
        """Feed finished episodes; returns the new stage name on promotion."""
        self.total_iterations += 1
        self.stage_iterations += 1
        if self._window_stage != self.index:
            self._window.clear()
            self._window_stage = self.index
        for episode in episodes:
            self._window.append(episode)

        stage = self.stage
        if (
            self.index >= len(self.stages) - 1
            or self.stage_iterations < stage.min_iterations
            or len(self._window) < stage.min_episodes
        ):
            return None
        wins = sum(1 for item in self._window if item.get("result") == "win")
        win_rate = wins / len(self._window)
        if win_rate < stage.win_rate:
            return None
        self.index += 1
        self.stage_iterations = 0
        self._window.clear()
        self._window_stage = self.index
        return self.stage.name

    def metrics(self) -> dict[str, float]:
        stage = self.stage
        wins = sum(1 for item in self._window if item.get("result") == "win")
        episodes = len(self._window)
        return {
            "curriculum/stage": float(self.index),
            "curriculum/stage_iterations": float(self.stage_iterations),
            "curriculum/window_win_rate": wins / episodes if episodes else 0.0,
            "curriculum/window_episodes": float(episodes),
            "curriculum/target_win_rate": float(stage.win_rate),
        }

    def describe(self) -> dict:
        return {
            "index": self.index,
            "stage": self.stage.to_dict(),
            "finished": self.finished,
            "iterations": self.stage_iterations,
            "total_iterations": self.total_iterations,
        }
