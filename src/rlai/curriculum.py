"""Staged curriculum for Rocket League training.

Each stage fixes the opponent, the starting-state distribution and the
*measurable skill* required to leave it.  Promotion is driven by measured
results over a window of finished episodes, never by elapsed time.

Why this was rewritten: the old first stage ("ball-control" vs ``idle``)
required a 60 % episode *win* rate.  With single-goal episodes a "win" means
scoring inside the time limit, so the ball-control stage actually tested
reliable scoring from a cold start.  Combined with the old reward the policy
reached a 0.07 % win rate and the stage never moved in 2M steps - silently,
because ``observe()`` only ever returned ``None``.  Stages now test the skill
they teach (touches first, then scoring against nobody, then winning against
active opponents), every blocked transition has an exact machine-readable
reason (:meth:`Curriculum.status`), a collapsing stage can demote, and a
stagnating stage says so loudly instead of blocking learning forever without
a trace.  The evidence window is persisted in checkpoints, so a resumed run
does not restart its promotion evidence from zero.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

#: episode fields worth keeping in the (persisted) evidence window
_WINDOW_FIELDS = ("result", "goals_for", "goals_against", "touches", "shots")


@dataclass(frozen=True)
class Stage:
    name: str
    opponent: str  # idle | random | chase | mixed | self
    min_iterations: int
    win_rate: float
    min_episodes: int = 30
    window: int = 60
    #: starting-state distribution for this stage (see envs.ResetMutator)
    reset_mode: str = "kickoff"
    #: per-stage episode horizon; ``None`` uses the run configuration
    episode_seconds: float | None = None
    #: mean learning-team ball touches per episode required to promote
    min_touches: float = 0.0
    #: mean goals scored per episode required to promote
    min_goal_rate: float = 0.0
    #: mean goals conceded per episode that must not be exceeded
    max_conceded_rate: float | None = None
    #: window win rate below which the stage is left *downwards*
    demote_win_rate: float | None = None
    #: iterations after which a stage without promotion reports stagnation
    stagnation_iterations: int = 250

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
        if self.opponent.startswith("difficulty:"):
            from .difficulty import DIFFICULTY_NAMES

            level = self.opponent.split(":", 1)[1]
            if level not in DIFFICULTY_NAMES:
                raise ValueError(
                    f"curriculum stage {self.name!r}: unknown difficulty level "
                    f"{level!r} (use one of {sorted(DIFFICULTY_NAMES)})"
                )
        elif self.opponent not in {"idle", "random", "chase", "mixed", "self"}:
            raise ValueError(
                f"curriculum stage {self.name!r}: unknown opponent {self.opponent!r}"
            )
        if self.reset_mode not in {"kickoff", "drill", "mixed"}:
            raise ValueError(
                f"curriculum stage {self.name!r}: unknown reset_mode {self.reset_mode!r}"
            )
        if self.episode_seconds is not None and not 5 <= float(self.episode_seconds) <= 600:
            raise ValueError(
                f"curriculum stage {self.name!r}: episode_seconds must be between 5 and 600"
            )
        if float(self.min_touches) < 0 or float(self.min_goal_rate) < 0:
            raise ValueError(
                f"curriculum stage {self.name!r}: touch/goal targets must not be negative"
            )
        if self.max_conceded_rate is not None and float(self.max_conceded_rate) < 0:
            raise ValueError(
                f"curriculum stage {self.name!r}: max_conceded_rate must not be negative"
            )
        if self.demote_win_rate is not None and not 0.0 <= float(self.demote_win_rate) < float(
            self.win_rate
        ):
            raise ValueError(
                f"curriculum stage {self.name!r}: demote_win_rate must be below the "
                "promotion win_rate (otherwise the stage would oscillate)"
            )
        if int(self.stagnation_iterations) < 1:
            raise ValueError(
                f"curriculum stage {self.name!r}: stagnation_iterations must be at least 1"
            )

    @property
    def has_criteria(self) -> bool:
        """True when the stage can be left on merit at all."""
        return (
            float(self.win_rate) > 0.0
            or float(self.min_touches) > 0.0
            or float(self.min_goal_rate) > 0.0
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "opponent": self.opponent,
            "min_iterations": self.min_iterations,
            "win_rate": self.win_rate,
            "min_episodes": self.min_episodes,
            "window": self.window,
            "reset_mode": self.reset_mode,
            "episode_seconds": self.episode_seconds,
            "min_touches": self.min_touches,
            "min_goal_rate": self.min_goal_rate,
            "max_conceded_rate": self.max_conceded_rate,
            "demote_win_rate": self.demote_win_rate,
            "stagnation_iterations": self.stagnation_iterations,
        }


DEFAULT_STAGES: tuple[Stage, ...] = (
    # Skill: reach the ball and make contact, from varied attacking setups.
    Stage(
        "first-touch", "idle", min_iterations=3, win_rate=0.0, min_episodes=30,
        reset_mode="drill", episode_seconds=20.0, min_touches=1.5,
    ),
    # Skill: turn contact into goals with nobody defending.
    Stage(
        "shooting", "idle", min_iterations=5, win_rate=0.0, min_episodes=40,
        reset_mode="mixed", episode_seconds=30.0, min_touches=1.0, min_goal_rate=0.2,
    ),
    # Skill: keep scoring (and stop conceding) against an active chaser.
    Stage(
        "chase", "chase", min_iterations=10, win_rate=0.3, min_episodes=40,
        reset_mode="mixed", episode_seconds=30.0, demote_win_rate=0.05,
    ),
    Stage(
        "mixed", "mixed", min_iterations=10, win_rate=0.4, min_episodes=40,
        reset_mode="mixed", episode_seconds=45.0, demote_win_rate=0.1,
    ),
    Stage("self-play", "self", min_iterations=1, win_rate=1.0, min_episodes=1,
          episode_seconds=45.0),
)

HEATSEEKER_STAGES: tuple[Stage, ...] = (
    Stage(
        "first-touch", "idle", min_iterations=3, win_rate=0.0, min_episodes=30,
        min_touches=1.0,
    ),
    Stage("chase", "chase", min_iterations=10, win_rate=0.4, min_episodes=40,
          demote_win_rate=0.1),
    Stage("self-play", "self", min_iterations=1, win_rate=1.0, min_episodes=1),
)

TEAM_STAGES: tuple[Stage, ...] = (
    Stage(
        "first-touch", "idle", min_iterations=3, win_rate=0.0, min_episodes=30,
        reset_mode="drill", episode_seconds=20.0, min_touches=2.0,
    ),
    Stage("chase", "chase", min_iterations=8, win_rate=0.3, min_episodes=30,
          reset_mode="mixed", episode_seconds=30.0, demote_win_rate=0.05),
    Stage("mixed", "mixed", min_iterations=10, win_rate=0.4, min_episodes=40,
          reset_mode="mixed", episode_seconds=45.0, demote_win_rate=0.1),
    Stage("self-play", "self", min_iterations=1, win_rate=1.0, min_episodes=1,
          episode_seconds=45.0),
)


def stages_for_mode(mode_id: str) -> tuple[Stage, ...]:
    if mode_id.startswith("heatseeker"):
        return HEATSEEKER_STAGES
    if mode_id in {"soccer-2v2", "soccer-3v3"}:
        return TEAM_STAGES
    return DEFAULT_STAGES


def _slim(episode: dict) -> dict:
    return {key: episode.get(key) for key in _WINDOW_FIELDS}


class Curriculum:
    """Tracks the active stage and promotes/demotes on measured results."""

    def __init__(
        self,
        stages: tuple[Stage, ...] | list[Stage],
        *,
        start_index: int = 0,
        stage_iterations: int = 0,
        total_iterations: int = 0,
        window_episodes: list[dict] | None = None,
    ):
        if not stages:
            raise ValueError("curriculum needs at least one stage")
        self.stages = tuple(stages)
        for stage in self.stages[:-1]:
            if not stage.has_criteria:
                raise ValueError(
                    f"curriculum stage {stage.name!r} has no promotion criteria "
                    "(win_rate, min_touches and min_goal_rate are all 0); the run "
                    "could never leave it on merit"
                )
        self.index = int(max(0, min(start_index, len(self.stages) - 1)))
        # Restoring the counters keeps ``min_iterations`` meaningful across a
        # resumed run instead of restarting the stage clock at zero.
        self.stage_iterations = max(0, int(stage_iterations))
        self.total_iterations = max(0, int(total_iterations))
        self._window: deque[dict] = deque(maxlen=self.stages[self.index].window)
        self._window_stage = self.index
        #: what happened in the last :meth:`observe` call:
        #: ``promotion`` | ``demotion`` | ``stagnation`` | ``None``
        self.last_event: str | None = None
        self._stagnation_reported_at = -1
        for episode in window_episodes or ():
            if isinstance(episode, dict):
                self._window.append(_slim(episode))

    # -- state -------------------------------------------------------------
    @property
    def stage(self) -> Stage:
        return self.stages[self.index]

    @property
    def finished(self) -> bool:
        return self.index >= len(self.stages) - 1

    def reset_window(self) -> None:
        self._window.clear()

    # -- window statistics ---------------------------------------------------
    def _window_stats(self) -> dict[str, float]:
        episodes = len(self._window)
        if not episodes:
            return {
                "episodes": 0.0, "win_rate": 0.0, "touches": 0.0,
                "goals_for": 0.0, "goals_against": 0.0,
            }
        wins = sum(1 for item in self._window if item.get("result") == "win")
        return {
            "episodes": float(episodes),
            "win_rate": wins / episodes,
            "touches": sum(float(item.get("touches") or 0) for item in self._window) / episodes,
            "goals_for": sum(float(item.get("goals_for") or 0) for item in self._window)
            / episodes,
            "goals_against": sum(
                float(item.get("goals_against") or 0) for item in self._window
            )
            / episodes,
        }

    def criteria(self) -> list[dict]:
        """Every promotion criterion with its observed value (report/GUI)."""
        stage = self.stage
        stats = self._window_stats()
        rows = [
            {
                "name": "stage_iterations", "required": float(stage.min_iterations),
                "observed": float(self.stage_iterations),
                "ok": self.stage_iterations >= stage.min_iterations,
            },
            {
                "name": "window_episodes", "required": float(stage.min_episodes),
                "observed": stats["episodes"],
                "ok": stats["episodes"] >= stage.min_episodes,
            },
        ]
        if stage.win_rate > 0:
            rows.append({
                "name": "win_rate", "required": float(stage.win_rate),
                "observed": stats["win_rate"], "ok": stats["win_rate"] >= stage.win_rate,
            })
        if stage.min_touches > 0:
            rows.append({
                "name": "touches_per_episode", "required": float(stage.min_touches),
                "observed": stats["touches"], "ok": stats["touches"] >= stage.min_touches,
            })
        if stage.min_goal_rate > 0:
            rows.append({
                "name": "goals_per_episode", "required": float(stage.min_goal_rate),
                "observed": stats["goals_for"],
                "ok": stats["goals_for"] >= stage.min_goal_rate,
            })
        if stage.max_conceded_rate is not None:
            rows.append({
                "name": "conceded_per_episode_max",
                "required": float(stage.max_conceded_rate),
                "observed": stats["goals_against"],
                "ok": stats["goals_against"] <= stage.max_conceded_rate,
            })
        return rows

    def blocked_reason(self) -> str | None:
        """Exact reason the active stage cannot be left yet (``None`` = ready)."""
        if self.finished:
            return None
        missing = [row for row in self.criteria() if not row["ok"]]
        if not missing:
            return None
        return "; ".join(
            f"{row['name']} {row['observed']:.3g} < required {row['required']:.3g}"
            if row["name"] != "conceded_per_episode_max"
            else f"{row['name']} {row['observed']:.3g} > allowed {row['required']:.3g}"
            for row in missing
        )

    # -- training hooks ------------------------------------------------------
    def observe(self, episodes: list[dict]) -> str | None:
        """Feed finished episodes; returns the new stage name on a transition."""
        self.last_event = None
        self.total_iterations += 1
        self.stage_iterations += 1
        if self._window_stage != self.index:
            self._window = deque(maxlen=self.stage.window)
            self._window_stage = self.index
        for episode in episodes:
            self._window.append(_slim(episode))

        if self.finished:
            return None
        stage = self.stage
        stats = self._window_stats()

        # Demotion first: a policy that collapsed against this stage's
        # opponent relearns faster one level down than by grinding losses.
        if (
            self.index > 0
            and stage.demote_win_rate is not None
            and self.stage_iterations >= stage.min_iterations
            and stats["episodes"] >= stage.min_episodes
            and stats["win_rate"] < stage.demote_win_rate
        ):
            self.index -= 1
            self.stage_iterations = 0
            self._window = deque(maxlen=self.stage.window)
            self._window_stage = self.index
            self.last_event = "demotion"
            return self.stage.name

        if self.blocked_reason() is None:
            self.index += 1
            self.stage_iterations = 0
            self._window = deque(maxlen=self.stage.window)
            self._window_stage = self.index
            self.last_event = "promotion"
            return self.stage.name

        if (
            self.stage_iterations >= stage.stagnation_iterations
            and self.stage_iterations - self._stagnation_reported_at
            >= stage.stagnation_iterations
        ):
            self._stagnation_reported_at = self.stage_iterations
            self.last_event = "stagnation"
        return None

    def metrics(self) -> dict[str, float]:
        stage = self.stage
        stats = self._window_stats()
        return {
            "curriculum/stage": float(self.index),
            "curriculum/stage_iterations": float(self.stage_iterations),
            "curriculum/window_win_rate": stats["win_rate"],
            "curriculum/window_episodes": stats["episodes"],
            "curriculum/window_touches": stats["touches"],
            "curriculum/window_goals_for": stats["goals_for"],
            "curriculum/window_goals_against": stats["goals_against"],
            "curriculum/target_win_rate": float(stage.win_rate),
            "curriculum/target_touches": float(stage.min_touches),
            "curriculum/target_goal_rate": float(stage.min_goal_rate),
            "curriculum/blocked": 0.0 if self.blocked_reason() is None else 1.0,
        }

    def status(self) -> dict:
        """Full stage diagnostics for logs, the report and the GUI."""
        return {
            "index": self.index,
            "stage": self.stage.to_dict(),
            "finished": self.finished,
            "stage_iterations": self.stage_iterations,
            "total_iterations": self.total_iterations,
            "criteria": self.criteria(),
            "blocked_reason": self.blocked_reason(),
            "last_event": self.last_event,
        }

    def describe(self) -> dict:
        return {
            "index": self.index,
            "stage": self.stage.to_dict(),
            "finished": self.finished,
            "iterations": self.stage_iterations,
            "total_iterations": self.total_iterations,
            # the evidence window survives the checkpoint, so promotion after
            # a resume is judged on the same episodes as without the break
            "window": list(self._window),
            "criteria": self.criteria(),
            "blocked_reason": self.blocked_reason(),
        }
