"""Independent skill probes: measurable evidence for curriculum gates.

Why this module exists
----------------------
Promotion in a curriculum must be based on something *other* than the
training reward, because the training reward is the thing being optimised:
a policy can raise it without acquiring the skill (proxy farming), and a
policy can acquire the skill while the reported reward stays flat.  Every
probe here therefore measures the **simulator state** after a scripted,
reproducible starting situation - not the reward.

Each probe answers one question with a sample size and a confidence
interval:

``approach``    does the car reach and touch a ball placed 400-1800 uu away?
``progress``    does a contact move the ball toward the opponent goal?
``shooting``    does it convert a shooting chance into a goal?
``defense``     does it stop a ball that is rolling at its own goal?
``recovery``    does it get back on its wheels and moving after a flip?
``positioning`` does it stay between the ball and its own goal, with boost?
``match``       does it win full games (delegates to :mod:`rlai.evaluation`)?

Probes are played on both sides of the field (the non-trained side through
:class:`rlai.policy.MirroredPolicy`, exactly like the match evaluation) and
with fixed seeds, so two evaluations of the same checkpoint are comparable.

Cost is deliberately small: a probe is a handful of short episodes, so it can
run inside a training loop at curriculum boundaries.  The measured cost is
reported with every result (``duration_s``) - a gate nobody can afford to
measure is not a gate.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .envs import MatchEnv
from .evaluation import wilson_interval
from .modes import ModeSpec, get_mode
from .opponents import build_opponent
from .obs import OBS_VERSION
from .policy import MirroredPolicy, TorchPolicy
from .rewards import shot_on_target

BLUE_TEAM = 0
ORANGE_TEAM = 1
SIDE_TEAM = {"blue": BLUE_TEAM, "orange": ORANGE_TEAM}

#: goal line distance used to decide "the ball reached the goal" (uu)
GOAL_LINE_Y = 5000.0


def _goal_center(team: int) -> tuple[float, float, float]:
    """Target goal of ``team`` in the *policy-attacking* frame (+y)."""
    return (0.0, 5120.0, 0.0) if team == BLUE_TEAM else (0.0, -5120.0, 0.0)


class _Trial:
    """One probe episode: steps the env and records what the probes need."""

    __slots__ = (
        "touches",
        "first_touch_step",
        "steps",
        "ball_start",
        "ball_end",
        "ball_max_progress",
        "goals_for",
        "goals_against",
        "shots_on_target",
        "shots_off_target",
        "on_ground_end",
        "upright_end",
        "speed_end",
        "boost_end",
        "goal_side_steps",
        "sign",
        "outcome",
    )

    def __init__(self) -> None:
        self.touches = 0
        self.first_touch_step: int | None = None
        self.steps = 0
        self.ball_start = np.zeros(3, dtype=np.float32)
        self.ball_end = np.zeros(3, dtype=np.float32)
        self.ball_max_progress = 0.0
        self.goals_for = 0
        self.goals_against = 0
        self.shots_on_target = 0
        self.shots_off_target = 0
        self.on_ground_end = False
        self.upright_end = False
        self.speed_end = 0.0
        self.boost_end = 0.0
        self.goal_side_steps = 0
        self.sign = 1.0
        self.outcome = "draw"


def _play(env: MatchEnv, view, trial: _Trial, team: int) -> _Trial:
    """Play one probe episode, recording state-based (not reward) outcomes."""
    obs = env.reset()
    agents = env.policy_agents
    sign = 1.0 if team == BLUE_TEAM else -1.0
    trial.sign = sign
    trial.ball_start = np.asarray(env.state.ball.position, dtype=np.float32).copy()
    start_y = float(trial.ball_start[1]) * sign
    trial.ball_max_progress = 0.0
    goal = _goal_center(team)
    while True:
        actions, _, _ = view.act({agent: obs[agent] for agent in agents}, deterministic=True)
        result = env.step(actions)
        obs = result.obs
        trial.steps += 1
        state = env.state
        car = state.cars[agents[0]]
        touched = sum(int(result.info.touches.get(agent, 0)) for agent in agents)
        if touched:
            trial.touches += touched
            if trial.first_touch_step is None:
                trial.first_touch_step = trial.steps
            velocity = np.asarray(state.ball.linear_velocity, dtype=np.float32)
            if shot_on_target(state.ball.position, velocity, np.asarray(goal, dtype=np.float32)):
                trial.shots_on_target += 1
            elif float(np.dot(velocity[:2], np.asarray(goal[:2]) - np.asarray(state.ball.position)[:2])) > 0:
                trial.shots_off_target += 1
        ball_y = float(state.ball.position[1]) * sign
        trial.ball_max_progress = max(trial.ball_max_progress, ball_y - start_y)
        # "goal side" = the car is between the ball and its own goal
        own_sign = -sign
        if (float(car.physics.position[1]) - float(state.ball.position[1])) * own_sign > 0:
            trial.goal_side_steps += 1
        if result.info.goal_scored:
            if result.info.scoring_team == team:
                trial.goals_for += 1
            else:
                trial.goals_against += 1
        if result.done or result.truncated:
            break
    state = env.state
    car = state.cars[agents[0]]
    trial.ball_end = np.asarray(state.ball.position, dtype=np.float32).copy()
    trial.on_ground_end = bool(car.on_ground)
    # ``on_ground`` is also true while the car lies on its roof, so recovery
    # is judged on the up vector (world z of the car's local +z axis).
    rotation = np.asarray(car.physics.rotation_mtx, dtype=np.float32)
    trial.upright_end = bool(float(rotation[2, 2]) > 0.5)
    trial.speed_end = float(np.linalg.norm(np.asarray(car.physics.linear_velocity)))
    trial.boost_end = float(car.boost_amount)
    if trial.goals_for > trial.goals_against:
        trial.outcome = "win"
    elif trial.goals_for < trial.goals_against:
        trial.outcome = "loss"
    return trial


def _rate(successes: int, trials: int) -> dict:
    low, high = wilson_interval(successes, trials)
    return {
        "rate": successes / trials if trials else 0.0,
        "successes": int(successes),
        "trials": int(trials),
        "ci95": [round(low, 4), round(high, 4)],
        "ci95_low": round(low, 4),
    }


# -- scenario builders -----------------------------------------------------
# Every distance/speed below was chosen from measured reference rates
# (docs/MASTERY_CURRICULUM.md, experiment 4): a probe is only useful when a
# car that does nothing fails it, a *fresh* network fails it, and a competent
# scripted driver passes it.  The constants are module level so the same
# probes can be re-calibrated with a measurement instead of a guess.

#: approach: car→ball distance and lateral offset (uu)
APPROACH_DISTANCES = (800.0, 1300.0, 1800.0, 2400.0)
APPROACH_LATERAL = (-1200.0, -600.0, 0.0, 600.0, 1200.0)
#: progress: the car starts beside the ball, so driving straight pushes it
#: sideways - useful progress needs aiming, not just contact
PROGRESS_LATERAL = 900.0
PROGRESS_BEHIND = 700.0
#: ball must travel this far toward the target goal to count as useful (uu)
PROGRESS_REQUIRED_UU = 1500.0
#: shooting: lateral offset of the car behind the ball (uu)
SHOOTING_OFFSETS = (-600.0, -300.0, 300.0, 600.0)
SHOOTING_BALL_X = (-1200.0, -400.0, 400.0, 1200.0)
SHOOTING_BALL_Y = (3000.0, 3600.0, 4000.0)
#: defense: ball speed toward the own goal and the car's starting distance
DEFENSE_SPEEDS = (700.0, 1000.0, 1300.0)
DEFENSE_CAR_DISTANCE = 700.0
DEFENSE_CAR_LATERAL = 300.0
#: goal line distance used to decide "the ball reached the goal" (uu)
GOAL_LINE_Y = 5000.0


def _goal_center(team: int) -> tuple[float, float, float]:
    """Target goal of ``team`` in the *policy-attacking* frame (+y)."""
    return (0.0, 5120.0, 0.0) if team == BLUE_TEAM else (0.0, -5120.0, 0.0)


class _Trial:
    """One probe episode: steps the env and records what the probes need."""

    __slots__ = (
        "touches",
        "first_touch_step",
        "steps",
        "ball_start",
        "ball_end",
        "ball_max_progress",
        "goals_for",
        "goals_against",
        "shots_on_target",
        "shots_off_target",
        "on_ground_end",
        "upright_end",
        "speed_end",
        "boost_end",
        "goal_side_steps",
        "sign",
        "outcome",
    )

    def __init__(self) -> None:
        self.touches = 0
        self.first_touch_step: int | None = None
        self.steps = 0
        self.ball_start = np.zeros(3, dtype=np.float32)
        self.ball_end = np.zeros(3, dtype=np.float32)
        self.ball_max_progress = 0.0
        self.goals_for = 0
        self.goals_against = 0
        self.shots_on_target = 0
        self.shots_off_target = 0
        self.on_ground_end = False
        self.upright_end = False
        self.speed_end = 0.0
        self.boost_end = 0.0
        self.goal_side_steps = 0
        self.sign = 1.0
        self.outcome = "draw"


def _play(env: MatchEnv, view, trial: _Trial, team: int) -> _Trial:
    """Play one probe episode, recording state-based (not reward) outcomes."""
    obs = env.reset()
    agents = env.policy_agents
    sign = 1.0 if team == BLUE_TEAM else -1.0
    trial.sign = sign
    trial.ball_start = np.asarray(env.state.ball.position, dtype=np.float32).copy()
    start_y = float(trial.ball_start[1]) * sign
    trial.ball_max_progress = 0.0
    goal = _goal_center(team)
    while True:
        actions, _, _ = view.act({agent: obs[agent] for agent in agents}, deterministic=True)
        result = env.step(actions)
        obs = result.obs
        trial.steps += 1
        state = env.state
        car = state.cars[agents[0]]
        touched = sum(int(result.info.touches.get(agent, 0)) for agent in agents)
        if touched:
            trial.touches += touched
            if trial.first_touch_step is None:
                trial.first_touch_step = trial.steps
            velocity = np.asarray(state.ball.linear_velocity, dtype=np.float32)
            if shot_on_target(state.ball.position, velocity, np.asarray(goal, dtype=np.float32)):
                trial.shots_on_target += 1
            elif float(np.dot(velocity[:2], np.asarray(goal[:2]) - np.asarray(state.ball.position)[:2])) > 0:
                trial.shots_off_target += 1
        ball_y = float(state.ball.position[1]) * sign
        trial.ball_max_progress = max(trial.ball_max_progress, ball_y - start_y)
        # "goal side" = the car is between the ball and its own goal
        own_sign = -sign
        if (float(car.physics.position[1]) - float(state.ball.position[1])) * own_sign > 0:
            trial.goal_side_steps += 1
        if result.info.goal_scored:
            if result.info.scoring_team == team:
                trial.goals_for += 1
            else:
                trial.goals_against += 1
        if result.done or result.truncated:
            break
    state = env.state
    car = state.cars[agents[0]]
    trial.ball_end = np.asarray(state.ball.position, dtype=np.float32).copy()
    trial.on_ground_end = bool(car.on_ground)
    # ``on_ground`` is also true while the car lies on its roof, so recovery
    # is judged on the up vector (world z of the car's local +z axis).
    rotation = np.asarray(car.physics.rotation_mtx, dtype=np.float32)
    trial.upright_end = bool(float(rotation[2, 2]) > 0.5)
    trial.speed_end = float(np.linalg.norm(np.asarray(car.physics.linear_velocity)))
    trial.boost_end = float(car.boost_amount)
    if trial.goals_for > trial.goals_against:
        trial.outcome = "win"
    elif trial.goals_for < trial.goals_against:
        trial.outcome = "loss"
    return trial


def _rate(successes: int, trials: int) -> dict:
    low, high = wilson_interval(successes, trials)
    return {
        "rate": successes / trials if trials else 0.0,
        "successes": int(successes),
        "trials": int(trials),
        "ci95": [round(low, 4), round(high, 4)],
        "ci95_low": round(low, 4),
    }


# -- scenario builders -----------------------------------------------------
# Every distance/speed below was chosen from measured reference rates
# (docs/MASTERY_CURRICULUM.md, experiment 4): a probe is only useful when a
# car that does nothing fails it, a *fresh* network fails it, and a competent
# scripted driver passes it.  The constants are module level so the same
# probes can be re-calibrated with a measurement instead of a guess.

#: approach: car→ball distance and lateral offset (uu)
APPROACH_DISTANCES = (800.0, 1300.0, 1800.0, 2400.0)
APPROACH_LATERAL = (-1200.0, -600.0, 0.0, 600.0, 1200.0)
#: progress: the car starts beside the ball, so driving straight pushes it
#: sideways - useful progress needs aiming, not just contact
PROGRESS_LATERAL = 900.0
PROGRESS_BEHIND = 700.0
#: ball must travel this far toward the target goal to count as useful (uu)
PROGRESS_REQUIRED_UU = 1500.0
#: shooting: lateral offset of the car behind the ball (uu)
SHOOTING_OFFSETS = (-600.0, -300.0, 300.0, 600.0)
SHOOTING_BALL_X = (-1200.0, -400.0, 400.0, 1200.0)
SHOOTING_BALL_Y = (3000.0, 3600.0, 4000.0)
#: defense: ball speed toward the own goal and the car's starting distance
DEFENSE_SPEEDS = (700.0, 1000.0, 1300.0)
DEFENSE_CAR_DISTANCE = 700.0
DEFENSE_CAR_LATERAL = 300.0
#: positioning: the car starts *ahead* of the ball (not goal-side)
POSITIONING_CAR_AHEAD = 1000.0


def _approach_scenario(index: int) -> dict:
    """Ball at rest 800-2400 uu away, with a lateral offset to steer out."""
    distance = APPROACH_DISTANCES[index % len(APPROACH_DISTANCES)]
    lateral = APPROACH_LATERAL[(index // len(APPROACH_DISTANCES)) % len(APPROACH_LATERAL)]
    ball_x = float(np.linspace(-1500.0, 1500.0, 3)[(index // 7) % 3])
    ball_y = float(np.linspace(-1500.0, 2000.0, 3)[(index // 11) % 3])
    return {
        "name": "approach",
        "ball": {"position": (ball_x, ball_y, 93.0)},
        "policy": [
            {
                "position": (ball_x + lateral, ball_y - distance, 17.0),
                "aim": (ball_x, ball_y),
                "boost": 60.0,
            }
        ],
        "opponent": [{"position": (0.0, 4600.0, 17.0), "yaw": -np.pi / 2, "boost": 0.0}],
    }


def _progress_scenario(index: int) -> dict:
    """Ball at rest in the own half; the car starts *beside* it."""
    ball_x = float(np.linspace(-1800.0, 1800.0, 5)[index % 5])
    ball_y = float(np.linspace(-2000.0, 500.0, 4)[(index // 5) % 4])
    side = 1.0 if (index // 20) % 2 == 0 else -1.0
    return {
        "name": "progress",
        "ball": {"position": (ball_x, ball_y, 93.0)},
        "policy": [
            {
                "position": (
                    float(np.clip(ball_x + side * PROGRESS_LATERAL, -3600.0, 3600.0)),
                    ball_y - PROGRESS_BEHIND,
                    17.0,
                ),
                "aim": (ball_x, ball_y),
                "boost": 70.0,
            }
        ],
        "opponent": [{"position": (0.0, 4600.0, 17.0), "yaw": -np.pi / 2, "boost": 0.0}],
    }


def _shooting_scenario(index: int) -> dict:
    """A shooting chance from an angle, with an idle keeper in the goal."""
    ball_x = SHOOTING_BALL_X[index % len(SHOOTING_BALL_X)]
    ball_y = SHOOTING_BALL_Y[(index // len(SHOOTING_BALL_X)) % len(SHOOTING_BALL_Y)]
    offset = SHOOTING_OFFSETS[(index // 7) % len(SHOOTING_OFFSETS)]
    return {
        "name": "shooting",
        "ball": {"position": (ball_x, ball_y, 93.0)},
        "policy": [
            {"position": (ball_x + offset, ball_y - 700.0, 17.0), "aim": (ball_x, ball_y),
             "boost": 80.0}
        ],
        "opponent": [{"position": (0.0, 4900.0, 17.0), "yaw": -np.pi / 2, "boost": 0.0}],
    }


def _defense_scenario(index: int) -> dict:
    """A shot on goal: the ball is rolled *at the goal centre*.

    Aiming the ball at the goal (instead of straight down the field) is what
    makes the probe measurable: with a straight roll, ~50 % of trials miss
    the goal on their own, so a car that does nothing "saved" half of them
    (measured: idle 0.58, random 0.50, scripted chase 0.50 - no signal at
    all).  With the ball on a collision course, not conceding *is* the save.
    """
    ball_x = float(np.linspace(-1200.0, 1200.0, 4)[index % 4])
    ball_y = float(np.linspace(-1500.0, -3500.0, 3)[(index // 4) % 3])
    speed = DEFENSE_SPEEDS[(index // 12) % len(DEFENSE_SPEEDS)]
    side = 1.0 if (index // 3) % 2 == 0 else -1.0
    # unit vector from the ball to the centre of the own goal (policy frame:
    # the own goal is at -y)
    dx, dy = 0.0 - ball_x, -GOAL_LINE_Y - ball_y
    norm = float(np.hypot(dx, dy)) or 1.0
    return {
        "name": "defense",
        "ball": {
            "position": (ball_x, ball_y, 93.0),
            "velocity": (speed * dx / norm, speed * dy / norm, 0.0),
        },
        "policy": [
            {
                "position": (ball_x + side * DEFENSE_CAR_LATERAL,
                             ball_y - DEFENSE_CAR_DISTANCE, 17.0),
                "aim": (ball_x, ball_y),
                "boost": 60.0,
            }
        ],
        "opponent": [{"position": (0.0, 4600.0, 17.0), "yaw": -np.pi / 2, "boost": 0.0}],
    }


def _reengage_scenario(index: int) -> dict:
    """Recovery after a miss: the car faces *away* from the ball.

    The obvious "car upside down" probe turned out to be unmeasurable: in
    RocketSim a car that does nothing rights itself after the drop, so the
    passive reference scored 1.00 while the scripted driver scored 0.00
    (docs/MASTERY_CURRICULUM.md, experiment 5).  What a 1v1 policy actually
    needs after a missed contact is to turn around and get back to the ball,
    and that *is* measurable.
    """
    ball_x = float(np.linspace(-1500.0, 1500.0, 4)[index % 4])
    ball_y = float(np.linspace(-1000.0, 2000.0, 3)[(index // 4) % 3])
    distance = float(np.linspace(700.0, 1500.0, 3)[(index // 12) % 3])
    return {
        "name": "reengage",
        "ball": {"position": (ball_x, ball_y, 93.0)},
        "policy": [
            {
                # behind the ball but facing the own goal: it has to turn
                "position": (ball_x, ball_y - distance, 17.0),
                "yaw": -np.pi / 2,
                "boost": 50.0,
            }
        ],
        "opponent": [{"position": (0.0, 4600.0, 17.0), "yaw": -np.pi / 2, "boost": 0.0}],
    }


def _measure_approach(trial: _Trial) -> dict:
    return {"success": 1.0 if trial.touches > 0 else 0.0, "touches": float(trial.touches),
            "first_touch_steps": float(trial.first_touch_step or -1)}


def _measure_progress(trial: _Trial) -> dict:
    progress = float(trial.ball_max_progress)
    return {
        "success": 1.0 if (trial.touches > 0 and progress >= PROGRESS_REQUIRED_UU) else 0.0,
        "touched": 1.0 if trial.touches > 0 else 0.0,
        "ball_progress_uu": progress,
    }


def _measure_shooting(trial: _Trial) -> dict:
    return {
        "success": 1.0 if trial.goals_for > 0 else 0.0,
        "on_target": 1.0 if trial.shots_on_target > 0 else 0.0,
        "touched": 1.0 if trial.touches > 0 else 0.0,
        "goals": float(trial.goals_for),
    }


def _measure_defense(trial: _Trial) -> dict:
    return {
        "success": 1.0 if trial.goals_against == 0 else 0.0,
        "touched": 1.0 if trial.touches > 0 else 0.0,
        # "cleared" = the ball is further from the own goal than it started
        "cleared": 1.0
        if float(trial.ball_end[1]) * trial.sign > float(trial.ball_start[1]) * trial.sign
        else 0.0,
    }


def _measure_reengage(trial: _Trial) -> dict:
    return {
        "success": 1.0 if trial.touches > 0 else 0.0,
        "touches": float(trial.touches),
        "first_touch_steps": float(trial.first_touch_step or -1),
    }


#: name -> (horizon, scenario builder, measurement, extra reported metrics)
PROBES: dict[str, tuple[float, Callable[[int], dict], Callable[[_Trial], dict], tuple[str, ...]]] = {
    "approach": (6.0, _approach_scenario, _measure_approach, ("touches",)),
    "progress": (10.0, _progress_scenario, _measure_progress, ("ball_progress_uu",)),
    "shooting": (10.0, _shooting_scenario, _measure_shooting, ("on_target", "touched", "goals")),
    "defense": (8.0, _defense_scenario, _measure_defense, ("touched", "cleared")),
    "reengage": (8.0, _reengage_scenario, _measure_reengage, ("touches",)),
}

SKILL_NAMES = tuple(PROBES)

#: gate skills that are measured by a *match* evaluation instead of a probe,
#: mapped to the opponent they are played against
MATCH_SKILLS = {"match_chase": "chase", "match_mixed": "mixed"}


def match_measurement(result: dict) -> dict:
    """Turn an :func:`rlai.evaluation.evaluate_policy` result into a gate value.

    The gate interface is the same for probes and matches (``rate``, ``ci95``,
    ``trials``), so a stage can require either without special cases.
    """
    games = int(result.get("games") or 0)
    return {
        "rate": float(result.get("win_rate") or 0.0),
        "ci95": list(result.get("win_rate_ci95") or [0.0, 1.0]),
        "ci95_low": float((result.get("win_rate_ci95") or [0.0, 1.0])[0]),
        "trials": games,
        "wins": int(result.get("wins") or 0),
        "losses": int(result.get("losses") or 0),
        "draws": int(result.get("draws") or 0),
        "goals_for": int(result.get("goals_for") or 0),
        "goals_against": int(result.get("goals_against") or 0),
        "opponent": result.get("opponent"),
        "duration_s": float(result.get("duration_s") or 0.0),
    }


def run_probe(
    policy: TorchPolicy,
    skill: str,
    *,
    mode: str = "soccer-1v1",
    trials: int = 16,
    seed: int = 4242,
    sides: tuple[str, ...] = ("blue", "orange"),
    mirror: bool = True,
    opponent: str = "idle",
    obs_version: int = OBS_VERSION,
    device: str = "cpu",
) -> dict:
    """Measure one skill on both sides with a fixed seed.

    Returns ``{"skill", "rate", "ci95", "trials", "sides", ...}`` where
    ``rate`` is the share of trials that satisfy the skill's success
    definition and every secondary metric is reported next to it.
    """
    if skill not in PROBES:
        raise ValueError(f"unknown skill probe {skill!r}; available: {list(PROBES)}")
    horizon, build, measure, extra = PROBES[skill]
    spec: ModeSpec = get_mode(mode)
    side_list = [str(side).lower() for side in sides] or ["blue"]
    unknown = [side for side in side_list if side not in SIDE_TEAM]
    if unknown:
        raise ValueError(f"unknown side(s) {unknown}")
    started = time.perf_counter()
    mirrored = MirroredPolicy(policy, obs_version=obs_version) if mirror else None

    rows: list[dict] = []
    envs: dict[int, MatchEnv] = {}
    try:
        for index in range(int(trials)):
            side = side_list[index % len(side_list)]
            team = SIDE_TEAM[side]
            view = policy if (team == BLUE_TEAM or mirrored is None) else mirrored
            env = envs.get(team)
            if env is None:
                env = MatchEnv(
                    spec,
                    build_opponent(opponent, seed=seed + index),
                    seed=seed,
                    episode_seconds=horizon,
                    obs_version=obs_version,
                    policy_team=team,
                )
                envs[team] = env
            env.seed = seed + index * 7919
            env.scenario = build(index)
            trial = _play(env, view, _Trial(), team)
            values = measure(trial)
            rows.append(
                {
                    "trial": index + 1,
                    "side": side,
                    "seed": env.seed,
                    "success": float(values["success"]),
                    **{key: float(values[key]) for key in extra},
                }
            )
    finally:
        for env in envs.values():
            env.close()

    successes = int(sum(row["success"] for row in rows))
    result: dict = {
        "skill": skill,
        "horizon_s": horizon,
        "opponent": opponent,
        "sides": sorted(set(side_list)),
        "mirror": bool(mirror),
        "seed": int(seed),
        **_rate(successes, len(rows)),
        "duration_s": round(time.perf_counter() - started, 3),
        "details": rows,
    }
    for key in extra:
        values = [row[key] for row in rows if row.get(key, -1) >= 0]
        result[key] = round(float(np.mean(values)), 4) if values else 0.0
    per_side = {}
    for side in sorted({row["side"] for row in rows}):
        subset = [row for row in rows if row["side"] == side]
        per_side[side] = _rate(int(sum(row["success"] for row in subset)), len(subset))
    result["side_results"] = per_side
    return result


def evaluate_skills(
    policy: TorchPolicy,
    skills: tuple[str, ...] | list[str],
    *,
    mode: str = "soccer-1v1",
    trials: int = 16,
    seed: int = 4242,
    sides: tuple[str, ...] = ("blue", "orange"),
    mirror: bool = True,
    opponent: str = "idle",
    obs_version: int = OBS_VERSION,
    device: str = "cpu",
    details: bool = False,
) -> dict:
    """Run several probes under one protocol and return them by name."""
    started = time.perf_counter()
    results: dict[str, dict] = {}
    for skill in skills:
        result = run_probe(
            policy,
            skill,
            mode=mode,
            trials=trials,
            seed=seed,
            sides=sides,
            mirror=mirror,
            opponent=opponent,
            obs_version=obs_version,
            device=device,
        )
        if not details:
            result.pop("details", None)
        results[skill] = result
    return {
        "kind": "skills",
        "mode": mode,
        "trials_per_skill": int(trials),
        "sides": sorted({side for side in sides}),
        "seed": int(seed),
        "opponent": opponent,
        "results": results,
        "duration_s": round(time.perf_counter() - started, 3),
        "protocol": {
            "kind": "skills",
            "mode": mode,
            "skills": list(skills),
            "trials": int(trials),
            "seed": int(seed),
            "sides": sorted({str(side).lower() for side in sides}),
            "mirror": bool(mirror),
            "opponent": opponent,
            "obs_version": int(obs_version),
        },
    }
