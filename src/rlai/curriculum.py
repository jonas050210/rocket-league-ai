"""Mastery-based curriculum: stages, independently measured gates, retention.

Every stage fixes four things: the opponent, the starting-state distribution,
the episode horizon and - the important part - the **skill that must be
demonstrated to leave it**.

Promotion is never decided by the training reward.  The reward is the thing
being optimised, so it cannot also be the referee: a policy can raise it
without acquiring the skill (proxy farming), and it can acquire the skill
while the reported reward stays flat.  Instead each stage declares
:class:`Gate` objects that are measured by the independent probes in
:mod:`rlai.skills` (and, for the last stages, by full match evaluations in
:mod:`rlai.evaluation`).  A gate carries its observed value, its sample size
and its 95 % confidence interval, and a stage only promotes when

* every gate of the stage passes, **including the confidence bound** (a lucky
  small sample must not promote anything),
* the same measurement passed ``confirmations`` times in a row (one unusually
  good evaluation is not mastery),
* the stage has run for ``min_iterations`` and seen ``min_episodes``, and
* no skill mastered in an *earlier* stage regressed by more than
  ``retention_tolerance`` (learning to shoot must not destroy ball contact).

Whatever is missing is reported exactly: :meth:`Curriculum.blocked_reason`
names the failing criterion, its observed value, the required value and the
confidence interval, and :meth:`Curriculum.diagnostics` adds what to look at
next.  A stage that does not promote therefore never fails silently - and it
never promotes by running out of patience either.

Thresholds were *measured*, not invented (docs/MASTERY_CURRICULUM.md,
experiment 5, 32 trials per probe, both sides):

=============  ======  ======  =========  =======  =======
probe          idle    random  untrained  shooter  chase
=============  ======  ======  =========  =======  =======
approach       0.062   0.125   0.094      0.750    1.000
progress       0.000   0.094   0.156      0.219    1.000
shooting       0.031   0.031   0.031      0.344    0.438
defense        0.188   0.406   0.125      0.562    0.688
reengage       0.000   0.000   0.000      0.938    0.000
=============  ======  ======  =========  =======  =======

Each threshold sits far above the do-nothing/random/untrained references and
at or below what a competent scripted driver achieves, so a stage is neither
free nor unreachable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

#: episode fields worth keeping in the (persisted) evidence window
_WINDOW_FIELDS = ("result", "goals_for", "goals_against", "touches", "shots")

#: how much an already-mastered skill may drop before promotion is blocked
RETENTION_TOLERANCE = 0.15


@dataclass(frozen=True)
class Gate:
    """One measurable promotion criterion.

    ``skill`` names a probe in :data:`rlai.skills.PROBES`, or ``"match"`` for
    an independent match evaluation.  ``metric`` is the key inside that
    measurement (``rate`` for every probe, so gates stay comparable).
    """

    skill: str
    minimum: float
    metric: str = "rate"
    #: also require the 95 % confidence *lower* bound to clear this value
    ci_low_minimum: float | None = None
    #: how the gate is described in logs and reports
    label: str | None = None

    def __post_init__(self) -> None:
        if not str(self.skill).strip():
            raise ValueError("a gate needs a skill name")
        if not 0.0 <= float(self.minimum) <= 1.0:
            raise ValueError(f"gate {self.name!r}: minimum must be within [0, 1]")
        if self.ci_low_minimum is not None and not 0.0 <= float(self.ci_low_minimum) <= 1.0:
            raise ValueError(f"gate {self.name!r}: ci_low_minimum must be within [0, 1]")
        if self.ci_low_minimum is not None and float(self.ci_low_minimum) > float(self.minimum):
            raise ValueError(
                f"gate {self.name!r}: ci_low_minimum cannot exceed the minimum "
                "(the bound is always below the point estimate)"
            )

    @property
    def name(self) -> str:
        return self.label or f"{self.skill}/{self.metric}"

    def evaluate(self, measurement: dict | None) -> dict:
        """Compare one measurement against the gate (``None`` = not measured)."""
        if not measurement:
            return {
                "name": self.name,
                "kind": "gate",
                "skill": self.skill,
                "required": float(self.minimum),
                "observed": None,
                "ci95": None,
                "ok": False,
                "reason": "not measured yet",
            }
        observed = measurement.get(self.metric)
        ci = measurement.get("ci95") or [None, None]
        if observed is None:
            return {
                "name": self.name, "kind": "gate", "skill": self.skill,
                "required": float(self.minimum), "observed": None, "ci95": ci,
                "ok": False, "reason": f"measurement has no {self.metric!r}",
            }
        observed = float(observed)
        ok = observed >= float(self.minimum)
        reason = "" if ok else f"{observed:.3f} < required {float(self.minimum):.3f}"
        if ok and self.ci_low_minimum is not None:
            low = ci[0]
            if low is None:
                ok = False
                reason = "no confidence interval reported"
            elif float(low) < float(self.ci_low_minimum):
                ok = False
                reason = (
                    f"95% CI lower bound {float(low):.3f} < required "
                    f"{float(self.ci_low_minimum):.3f} ({int(measurement.get('trials') or 0)} "
                    "trials - measure more games before promoting)"
                )
        return {
            "name": self.name,
            "kind": "gate",
            "skill": self.skill,
            "required": float(self.minimum),
            "observed": observed,
            "ci95": ci,
            "trials": int(measurement.get("trials") or 0),
            "ok": ok,
            "reason": reason,
        }

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "metric": self.metric,
            "minimum": float(self.minimum),
            "ci_low_minimum": (
                None if self.ci_low_minimum is None else float(self.ci_low_minimum)
            ),
            "label": self.label,
        }


#: the measured gates (see the module docstring for the reference rates)
GATE_APPROACH = Gate("approach", minimum=0.80, ci_low_minimum=0.60)
GATE_PROGRESS = Gate("progress", minimum=0.70, ci_low_minimum=0.50)
GATE_SHOOTING = Gate("shooting", minimum=0.30, ci_low_minimum=0.15)
GATE_DEFENSE = Gate("defense", minimum=0.60, ci_low_minimum=0.40)
GATE_REENGAGE = Gate("reengage", minimum=0.70, ci_low_minimum=0.50)
#: "match" gates are measured by a real match evaluation, not by a probe; the
#: opponent is part of the skill name (see rlai.skills.MATCH_SKILLS)
GATE_MATCH_CHASE = Gate("match_chase", minimum=0.50, ci_low_minimum=0.30)
GATE_MATCH_MIXED = Gate("match_mixed", minimum=0.40, ci_low_minimum=0.20)


@dataclass(frozen=True)
class Stage:
    name: str
    opponent: str  # idle | random | chase | mixed | self | difficulty:<level>
    min_iterations: int
    win_rate: float
    min_episodes: int = 30
    window: int = 60
    #: starting-state distribution for this stage (see envs.ResetMutator)
    reset_mode: str = "kickoff"
    #: per-stage episode horizon; ``None`` uses the run configuration
    episode_seconds: float | None = None
    #: per-stage no-touch timeout; ``None`` uses the run configuration.  A
    #: stage whose starting states are far from the ball needs a longer
    #: timeout, otherwise more than half of its episodes end before the
    #: learner can touch the ball at all (measured: 55 % zero-touch episodes
    #: in the ``shooting`` stage with the run-wide 10 s timeout).
    no_touch_timeout: float | None = None
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
    #: independently measured promotion criteria
    gates: tuple[Gate, ...] = ()
    #: how many consecutive passing gate evaluations are required
    confirmations: int = 2
    #: how far an earlier stage's mastered skill may drop before promotion is
    #: blocked (``None`` uses :data:`RETENTION_TOLERANCE`)
    retention_tolerance: float | None = None
    #: what the stage is supposed to teach (documentation, reports, GUI)
    purpose: str = ""

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
        from .envs import RESET_MODES

        if self.reset_mode not in RESET_MODES:
            raise ValueError(
                f"curriculum stage {self.name!r}: unknown reset_mode {self.reset_mode!r} "
                f"(use one of {list(RESET_MODES)})"
            )
        if self.episode_seconds is not None and not 5 <= float(self.episode_seconds) <= 600:
            raise ValueError(
                f"curriculum stage {self.name!r}: episode_seconds must be between 5 and 600"
            )
        if self.no_touch_timeout is not None and not (
            0.0 <= float(self.no_touch_timeout) <= 600.0
        ):
            raise ValueError(
                f"curriculum stage {self.name!r}: no_touch_timeout must be between "
                "0 and 600 seconds (0 disables it)"
            )
        if float(self.min_touches) < 0 or float(self.min_goal_rate) < 0:
            raise ValueError(
                f"curriculum stage {self.name!r}: touch/goal targets must not be negative"
            )
        if self.max_conceded_rate is not None and float(self.max_conceded_rate) < 0:
            raise ValueError(
                f"curriculum stage {self.name!r}: max_conceded_rate must not be negative"
            )
        if self.demote_win_rate is not None:
            low = 0.0 <= float(self.demote_win_rate) <= 1.0
            # A stage that promotes on a *window win rate* must demote strictly
            # below it, otherwise it would oscillate.  A stage that promotes on
            # measured gates has no win-rate criterion, so its demotion level is
            # only a collapse detector and may be any rate.
            ordered = float(self.win_rate) <= 0.0 or float(self.demote_win_rate) < float(
                self.win_rate
            )
            if not (low and ordered):
                raise ValueError(
                    f"curriculum stage {self.name!r}: demote_win_rate must be within [0, 1] "
                    "and below the promotion win_rate when one is set (otherwise the "
                    "stage would oscillate)"
                )
        if int(self.stagnation_iterations) < 1:
            raise ValueError(
                f"curriculum stage {self.name!r}: stagnation_iterations must be at least 1"
            )
        if int(self.confirmations) < 1:
            raise ValueError(
                f"curriculum stage {self.name!r}: confirmations must be at least 1 "
                "(a single lucky evaluation is not mastery)"
            )
        for gate in self.gates:
            if not isinstance(gate, Gate):
                raise ValueError(
                    f"curriculum stage {self.name!r}: gates must be rlai.curriculum.Gate "
                    f"objects, got {gate!r}"
                )

    @property
    def has_criteria(self) -> bool:
        """True when the stage can be left on merit at all."""
        return bool(
            self.gates
            or float(self.win_rate) > 0.0
            or float(self.min_touches) > 0.0
            or float(self.min_goal_rate) > 0.0
        )

    @property
    def gate_skills(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(gate.skill for gate in self.gates))

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
            "no_touch_timeout": self.no_touch_timeout,
            "min_touches": self.min_touches,
            "min_goal_rate": self.min_goal_rate,
            "max_conceded_rate": self.max_conceded_rate,
            "demote_win_rate": self.demote_win_rate,
            "stagnation_iterations": self.stagnation_iterations,
            "gates": [gate.to_dict() for gate in self.gates],
            "confirmations": self.confirmations,
            "retention_tolerance": self.retention_tolerance,
            "purpose": self.purpose,
        }


#: Mastery curriculum for 1v1.  Order: reach the ball -> move it where you
#: want -> score -> defend and re-engage -> win against an active opponent ->
#: win against a mixed opponent pool -> self-play.  Every stage re-checks the
#: skills it depends on, so a stage cannot be "passed" by forgetting them.
MASTERY_STAGES: tuple[Stage, ...] = (
    Stage(
        "approach", "idle", min_iterations=20, win_rate=0.0, min_episodes=40,
        reset_mode="reach", episode_seconds=20.0, gates=(GATE_APPROACH,),
        purpose="reach and touch the ball reliably from varied close-range starts",
    ),
    Stage(
        "control", "idle", min_iterations=20, win_rate=0.0, min_episodes=40,
        reset_mode="reach", episode_seconds=20.0,
        gates=(GATE_APPROACH, GATE_PROGRESS),
        purpose="make contacts that move the ball toward the target, not just any contact",
    ),
    Stage(
        "shooting", "idle", min_iterations=30, win_rate=0.0, min_episodes=40,
        # ``drill`` keeps every start in an attacking/defending/race situation
        # within 700-2600 uu of the ball.  ``mixed`` would add kickoffs, whose
        # 2500+ uu approach is a *reach* problem that the two previous stages
        # already solved - and it is where more than half of the episodes ran
        # into the no-touch timeout without a single contact.
        reset_mode="drill", episode_seconds=30.0, no_touch_timeout=25.0,
        gates=(GATE_PROGRESS, GATE_SHOOTING),
        purpose="turn attacking positions into shots and goals",
    ),
    Stage(
        "defense", "chase", min_iterations=40, win_rate=0.0, min_episodes=40,
        reset_mode="mixed", episode_seconds=30.0, no_touch_timeout=25.0,
        gates=(GATE_DEFENSE, GATE_REENGAGE),
        purpose="stop shots on goal and get back to the ball after a missed contact",
    ),
    Stage(
        "pressure", "chase", min_iterations=40, win_rate=0.0, min_episodes=40,
        reset_mode="mixed", episode_seconds=30.0, no_touch_timeout=25.0,
        gates=(GATE_SHOOTING, GATE_DEFENSE, GATE_MATCH_CHASE),
        demote_win_rate=0.02,
        purpose="score and defend in the same game against an active opponent",
    ),
    Stage(
        "integrated", "mixed", min_iterations=60, win_rate=0.0, min_episodes=40,
        reset_mode="mixed", episode_seconds=45.0, no_touch_timeout=25.0,
        gates=(GATE_SHOOTING, GATE_DEFENSE, GATE_MATCH_MIXED),
        demote_win_rate=0.05,
        purpose="one policy that plays a full 1v1 against a mixed opponent pool",
    ),
    Stage(
        "self-play", "self", min_iterations=1, win_rate=1.0, min_episodes=1,
        episode_seconds=45.0, purpose="open-ended improvement against its own snapshots",
    ),
)

#: the pre-mastery stages, kept so an old configuration (or an experiment that
#: wants window-only promotion) can still be run and compared
LEGACY_STAGES: tuple[Stage, ...] = (
    Stage(
        "first-touch", "idle", min_iterations=3, win_rate=0.0, min_episodes=30,
        reset_mode="drill", episode_seconds=20.0, min_touches=1.5,
    ),
    Stage(
        "shooting", "idle", min_iterations=5, win_rate=0.0, min_episodes=40,
        reset_mode="mixed", episode_seconds=30.0, min_touches=1.0, min_goal_rate=0.2,
    ),
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

DEFAULT_STAGES = MASTERY_STAGES

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

CURRICULUM_PRESETS = {"mastery": MASTERY_STAGES, "legacy": LEGACY_STAGES}


def stages_for_mode(mode_id: str, preset: str = "mastery") -> tuple[Stage, ...]:
    """The stage list for a mode.

    ``preset`` selects between the mastery curriculum (default) and the
    pre-mastery window-only stages, which are kept for reproducible
    comparisons with earlier runs.  Heatseeker and the team modes keep their
    own lists: their skills are not measurable with the 1v1 probes.
    """
    if mode_id.startswith("heatseeker"):
        return HEATSEEKER_STAGES
    if mode_id in {"soccer-2v2", "soccer-3v3"}:
        return TEAM_STAGES
    if preset not in CURRICULUM_PRESETS:
        raise ValueError(
            f"unknown curriculum preset {preset!r}; use one of {sorted(CURRICULUM_PRESETS)}"
        )
    return CURRICULUM_PRESETS[preset]


def _slim(episode: dict) -> dict:
    return {key: episode.get(key) for key in _WINDOW_FIELDS}


@dataclass
class GateState:
    """Persisted gate bookkeeping (survives a checkpoint)."""

    evaluations: int = 0
    passes_in_a_row: int = 0
    #: measured value per gate name at the moment its stage was mastered
    baselines: dict[str, float] = field(default_factory=dict)
    #: the most recent measurement per skill
    last: dict[str, dict] = field(default_factory=dict)
    #: stage name in which each baseline was recorded
    baseline_stage: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "evaluations": self.evaluations,
            "passes_in_a_row": self.passes_in_a_row,
            "baselines": dict(self.baselines),
            "baseline_stage": dict(self.baseline_stage),
            "last": {key: dict(value) for key, value in self.last.items()},
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "GateState":
        if not isinstance(data, dict):
            return cls()
        return cls(
            evaluations=int(data.get("evaluations", 0) or 0),
            passes_in_a_row=int(data.get("passes_in_a_row", 0) or 0),
            baselines={str(k): float(v) for k, v in (data.get("baselines") or {}).items()},
            baseline_stage={str(k): str(v) for k, v in (data.get("baseline_stage") or {}).items()},
            last={
                str(key): dict(value)
                for key, value in (data.get("last") or {}).items()
                if isinstance(value, dict)
            },
        )


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
        gate_state: GateState | dict | None = None,
    ):
        if not stages:
            raise ValueError("curriculum needs at least one stage")
        self.stages = tuple(stages)
        for stage in self.stages[:-1]:
            if not stage.has_criteria:
                raise ValueError(
                    f"curriculum stage {stage.name!r} has no promotion criteria "
                    "(gates, win_rate, min_touches and min_goal_rate are all empty); "
                    "the run could never leave it on merit"
                )
        self.index = int(max(0, min(start_index, len(self.stages) - 1)))
        # Restoring the counters keeps ``min_iterations`` meaningful across a
        # resumed run instead of restarting the stage clock at zero.
        self.stage_iterations = max(0, int(stage_iterations))
        self.total_iterations = max(0, int(total_iterations))
        self._window: deque[dict] = deque(maxlen=self.stages[self.index].window)
        self._window_stage = self.index
        #: what happened in the last :meth:`observe`/:meth:`record_gate` call:
        #: ``promotion`` | ``demotion`` | ``stagnation`` | ``regression`` | ``None``
        self.last_event: str | None = None
        self._stagnation_reported_at = -1
        self.gates = (
            gate_state
            if isinstance(gate_state, GateState)
            else GateState.from_dict(gate_state if isinstance(gate_state, dict) else None)
        )
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

    def required_skills(self) -> tuple[str, ...]:
        """Every skill that must be measured right now.

        That is the active stage's own gates *plus* the gates of every earlier
        stage - the earlier ones are the retention check, so they have to be
        measured in the same evaluation to be comparable.
        """
        skills: list[str] = []
        for stage in self.stages[: self.index + 1]:
            for skill in stage.gate_skills:
                if skill not in skills:
                    skills.append(skill)
        return tuple(skills)

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

    # -- criteria ------------------------------------------------------------
    def gate_rows(self) -> list[dict]:
        """Gate criteria of the active stage with their measured values."""
        return [
            gate.evaluate(self.gates.last.get(gate.skill)) for gate in self.stage.gates
        ]

    def retention_rows(self) -> list[dict]:
        """Earlier stages' mastered skills, compared with their baseline."""
        rows: list[dict] = []
        tolerance = (
            RETENTION_TOLERANCE
            if self.stage.retention_tolerance is None
            else float(self.stage.retention_tolerance)
        )
        seen: set[str] = set()
        for stage in self.stages[: self.index]:
            for gate in stage.gates:
                if gate.name in seen:
                    continue
                seen.add(gate.name)
                baseline = self.gates.baselines.get(gate.name)
                measurement = self.gates.last.get(gate.skill)
                if baseline is None or not measurement:
                    continue
                observed = measurement.get(gate.metric)
                if observed is None:
                    continue
                observed = float(observed)
                floor = max(0.0, float(baseline) - tolerance)
                ok = observed >= floor
                rows.append(
                    {
                        "name": f"retention:{gate.name}",
                        "kind": "retention",
                        "skill": gate.skill,
                        "required": floor,
                        "observed": observed,
                        "baseline": float(baseline),
                        "baseline_stage": self.gates.baseline_stage.get(gate.name),
                        "ci95": measurement.get("ci95"),
                        "ok": ok,
                        "reason": (
                            ""
                            if ok
                            else f"{observed:.3f} dropped below {floor:.3f} "
                            f"(mastered at {float(baseline):.3f} in "
                            f"{self.gates.baseline_stage.get(gate.name, 'an earlier stage')})"
                        ),
                    }
                )
        return rows

    def criteria(self) -> list[dict]:
        """Every promotion criterion with its observed value (report/GUI)."""
        stage = self.stage
        stats = self._window_stats()
        rows = [
            {
                "name": "stage_iterations", "kind": "window",
                "required": float(stage.min_iterations),
                "observed": float(self.stage_iterations),
                "ok": self.stage_iterations >= stage.min_iterations,
            },
            {
                "name": "window_episodes", "kind": "window",
                "required": float(stage.min_episodes),
                "observed": stats["episodes"],
                "ok": stats["episodes"] >= stage.min_episodes,
            },
        ]
        if stage.win_rate > 0:
            rows.append({
                "name": "win_rate", "kind": "window", "required": float(stage.win_rate),
                "observed": stats["win_rate"], "ok": stats["win_rate"] >= stage.win_rate,
            })
        if stage.min_touches > 0:
            rows.append({
                "name": "touches_per_episode", "kind": "window",
                "required": float(stage.min_touches),
                "observed": stats["touches"], "ok": stats["touches"] >= stage.min_touches,
            })
        if stage.min_goal_rate > 0:
            rows.append({
                "name": "goals_per_episode", "kind": "window",
                "required": float(stage.min_goal_rate),
                "observed": stats["goals_for"],
                "ok": stats["goals_for"] >= stage.min_goal_rate,
            })
        if stage.max_conceded_rate is not None:
            rows.append({
                "name": "conceded_per_episode_max", "kind": "window",
                "required": float(stage.max_conceded_rate),
                "observed": stats["goals_against"],
                "ok": stats["goals_against"] <= stage.max_conceded_rate,
            })
        rows.extend(self.gate_rows())
        if stage.gates and stage.confirmations > 1:
            rows.append({
                "name": "gate_confirmations", "kind": "gate",
                "required": float(stage.confirmations),
                "observed": float(self.gates.passes_in_a_row),
                "ok": self.gates.passes_in_a_row >= stage.confirmations,
            })
        rows.extend(self.retention_rows())
        return rows

    def blocked_reason(self) -> str | None:
        """Exact reason the active stage cannot be left yet (``None`` = ready)."""
        if self.finished:
            return None
        missing = [row for row in self.criteria() if not row["ok"]]
        if not missing:
            return None
        parts = []
        for row in missing:
            if row.get("reason"):
                parts.append(f"{row['name']}: {row['reason']}")
            elif row["name"] == "conceded_per_episode_max":
                parts.append(
                    f"{row['name']} {row['observed']:.3g} > allowed {row['required']:.3g}"
                )
            else:
                parts.append(
                    f"{row['name']} {row['observed']:.3g} < required {row['required']:.3g}"
                )
        return "; ".join(parts)

    def diagnostics(self) -> dict:
        """What is missing and what to look at next (logs, report, GUI)."""
        failing = [row for row in self.criteria() if not row["ok"]]
        hints: list[str] = []
        for row in failing:
            name = str(row["name"])
            if name.startswith("retention:"):
                hints.append(
                    f"{name}: an earlier skill regressed - re-run that stage's probe and "
                    "check whether the new stage's opponent/resets made it unlearnable"
                )
            elif row.get("kind") == "gate":
                skill = row.get("skill")
                if skill == "match":
                    hints.append(
                        f"{name}: play more evaluation games (win_rate confidence interval "
                        "is still too wide) or improve the underlying probes first"
                    )
                elif skill:
                    hints.append(
                        f"{name}: inspect the probe details (rlai.skills.run_probe "
                        f"'{skill}'), the per-episode touch rate and the policy entropy"
                    )
                else:
                    hints.append(
                        f"{name}: the same measurement must pass every gate twice in a "
                        "row - fix the failing gates first, then let the stage run"
                    )
            elif name == "stage_iterations":
                hints.append(f"{name}: keep training this stage")
            elif name == "window_episodes":
                hints.append(f"{name}: collect more episodes in this stage")
            else:
                hints.append(f"{name}: training-window criterion not met")
        return {
            "stage": self.stage.name,
            "blocked_reason": self.blocked_reason(),
            "failing": [
                {
                    "name": row["name"],
                    "kind": row.get("kind"),
                    "required": row.get("required"),
                    "observed": row.get("observed"),
                    "ci95": row.get("ci95"),
                }
                for row in failing
            ],
            "hints": hints,
            "gate_evaluations": self.gates.evaluations,
            "passes_in_a_row": self.gates.passes_in_a_row,
            "confirmations_required": int(self.stage.confirmations),
        }

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
            return self._demote()

        transition = self._try_promote()
        if transition:
            return transition

        if (
            self.stage_iterations >= stage.stagnation_iterations
            and self.stage_iterations - self._stagnation_reported_at
            >= stage.stagnation_iterations
        ):
            self._stagnation_reported_at = self.stage_iterations
            self.last_event = "stagnation"
        return None

    def record_gate(self, measurements: dict[str, dict]) -> str | None:
        """Feed one independent gate evaluation (see :mod:`rlai.skills`).

        ``measurements`` maps a skill name to its measurement dict (``rate``,
        ``ci95``, ``trials``, …).  Returns the new stage name when this
        evaluation completed a promotion.
        """
        self.last_event = None
        if self.finished:
            return None
        for skill, measurement in (measurements or {}).items():
            if isinstance(measurement, dict):
                self.gates.last[str(skill)] = dict(measurement)
        self.gates.evaluations += 1

        gates_ok = all(row["ok"] for row in self.gate_rows())
        retention = self.retention_rows()
        retention_ok = all(row["ok"] for row in retention)
        if gates_ok and retention_ok:
            self.gates.passes_in_a_row += 1
        else:
            if self.gates.passes_in_a_row:
                self.gates.passes_in_a_row = 0
            if not retention_ok:
                self.last_event = "regression"
        return self._try_promote()

    def _try_promote(self) -> str | None:
        if self.finished or self.blocked_reason() is not None:
            return None
        mastered = self.stage.name
        for gate in self.stage.gates:
            measurement = self.gates.last.get(gate.skill) or {}
            observed = measurement.get(gate.metric)
            if observed is not None:
                self.gates.baselines[gate.name] = float(observed)
                self.gates.baseline_stage[gate.name] = mastered
        self.index += 1
        self.stage_iterations = 0
        self.gates.passes_in_a_row = 0
        self._window = deque(maxlen=self.stage.window)
        self._window_stage = self.index
        self.last_event = "promotion"
        return self.stage.name

    def _demote(self) -> str:
        self.index -= 1
        self.stage_iterations = 0
        self.gates.passes_in_a_row = 0
        self._window = deque(maxlen=self.stage.window)
        self._window_stage = self.index
        self.last_event = "demotion"
        return self.stage.name

    def metrics(self) -> dict[str, float]:
        stage = self.stage
        stats = self._window_stats()
        values = {
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
            "curriculum/gate_evaluations": float(self.gates.evaluations),
            "curriculum/gate_passes_in_a_row": float(self.gates.passes_in_a_row),
            "curriculum/gate_confirmations_required": float(stage.confirmations),
        }
        for row in self.criteria():
            if row.get("kind") != "gate" or row.get("observed") is None:
                continue
            values[f"curriculum/gate/{row['name']}"] = float(row["observed"])
        for row in self.retention_rows():
            if row.get("observed") is None:
                continue
            values[f"curriculum/retention/{row['skill']}"] = float(row["observed"])
        return values

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
            "diagnostics": self.diagnostics(),
            "gates": self.gates.to_dict(),
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
            # ... and so do the gate measurements: a resume must not restart
            # the confirmation count or forget the retention baselines
            "gate_state": self.gates.to_dict(),
        }
