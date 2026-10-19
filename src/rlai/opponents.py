"""Opponent policies used during training and evaluation.

These are real policies, not scripted replays: they read the live simulation
state every decision tick.  The chaser is intentionally simple but non-trivial
(it drives at the ball and boosts), which makes it a meaningful first
milestone and a stable evaluation baseline.  Trained checkpoints can be used
as opponents through ``model:<path>``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TYPE_CHECKING

import numpy as np

from .actions import ACTION_TABLE, ACTION_COUNT
from .policy import TorchPolicy

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .botbrain import BrainContext


class Opponent(Protocol):
    name: str

    def reset(self, env) -> None: ...

    def act(
        self, env, observations: dict[str, np.ndarray], agents: list[str]
    ) -> dict[str, int]: ...


def _ground_index(throttle: int, steer: int, boost: int) -> int:
    target = np.array([throttle, steer, 0, steer, 0, 0, boost, 0], dtype=np.float32)
    matches = np.where((ACTION_TABLE == target).all(axis=1))[0]
    if len(matches) == 0:
        raise LookupError(f"no ground action for {target}")
    return int(matches[0])


class IdleOpponent:
    name = "idle"

    def reset(self, env) -> None:
        pass

    def act(self, env, observations, agents):
        return {agent: 0 for agent in agents}


class RandomOpponent:
    def __init__(self, seed: int = 0):
        self.name = "random"
        self.rng = np.random.default_rng(seed)

    def reset(self, env) -> None:
        pass

    def act(self, env, observations, agents):
        return {agent: int(self.rng.integers(ACTION_COUNT)) for agent in agents}


class ChaseOpponent:
    """Drives every orange car toward the ball, boosting when aligned."""

    name = "chase"

    def reset(self, env) -> None:
        pass

    def act(self, env, observations, agents):
        state = env.state
        ball = state.ball.position
        actions: dict[str, int] = {}
        for agent in agents:
            car = state.cars[agent]
            if car.is_demoed:
                actions[agent] = 0
                continue
            to_ball = np.asarray(ball, dtype=np.float32) - car.physics.position
            flat = to_ball[:2]
            flat_norm = float(np.linalg.norm(flat))
            if flat_norm < 1e-3:
                actions[agent] = _ground_index(1, 0, 0)
                continue
            forward = np.asarray(car.physics.forward, dtype=np.float32)
            cross = forward[0] * flat[1] - forward[1] * flat[0]
            dot = forward[0] * flat[0] + forward[1] * flat[1]
            angle = np.arctan2(cross, dot)
            steer = 0 if abs(angle) < 0.08 else (1 if angle > 0 else -1)
            aligned = abs(angle) < 0.6
            boost = int(
                aligned
                and flat_norm > 900
                and car.boost_amount > 15
                and np.linalg.norm(car.physics.linear_velocity) < 2100
            )
            throttle = 1 if aligned else (1 if abs(angle) < 1.9 else 0)
            actions[agent] = _ground_index(throttle, steer, boost)
        return actions


class ModelOpponent:
    """Frozen policy used as an opponent (self-play snapshot or evaluation)."""

    def __init__(self, policy: TorchPolicy, name: str = "model", deterministic: bool = True):
        self.policy = policy
        self.name = name
        self.deterministic = deterministic

    @classmethod
    def from_checkpoint(
        cls,
        reference: str | Path,
        device: str = "cpu",
        name: str | None = None,
        *,
        weights_only: bool = True,
        allow_outside: bool = True,
    ) -> "ModelOpponent":
        """Build a frozen opponent from a checkpoint path or a model id.

        Loading goes through :func:`rlai.checkpoints.load_checkpoint` with
        ``weights_only=True`` so a hostile file cannot execute code via pickle
        (every checkpoint this project writes loads safely).  A model id is
        resolved inside the data models directory unless it names an existing
        file, so an id that arrives over HTTP cannot point at an arbitrary path.
        """
        from .checkpoints import load_policy, resolve_model_path

        path = Path(str(reference).strip())
        if not path.exists():
            from .config import ensure_dirs

            path = resolve_model_path(
                ensure_dirs()["models"], str(reference), allow_outside=allow_outside
            )
        policy = load_policy(path, device=device, weights_only=weights_only)
        return cls(policy, name=name or path.parent.name, deterministic=True)

    def reset(self, env) -> None:
        pass

    def act(self, env, observations, agents):
        if not agents:
            return {}
        obs = {agent: observations[agent] for agent in agents}
        actions, _, _ = self.policy.act(obs, deterministic=self.deterministic)
        return actions


#: opponents that can be built without any extra input
SCRIPTED_OPPONENTS = ("idle", "random", "chase")
#: every name the rest of the system accepts
KNOWN_OPPONENTS = SCRIPTED_OPPONENTS + ("self",)


def validate_opponent_name(name: str, *, allow_self: bool = True) -> str:
    """Check an opponent reference *before* a worker process is started.

    Returns the cleaned name or raises :class:`ValueError` with a message
    that names the accepted values.  Used by the config validator, by the
    API and by the trainer so an invalid opponent never reaches a worker
    process and crashes it with an opaque error.
    """
    key = str(name or "").strip()
    if not key:
        raise ValueError("opponent must not be empty")
    if key in SCRIPTED_OPPONENTS:
        return key
    if key == "self":
        if allow_self:
            return key
        raise ValueError(
            "opponent 'self' needs a self-play snapshot and cannot be used here; "
            f"use one of {list(SCRIPTED_OPPONENTS)} or 'model:<checkpoint>'"
        )
    if key.startswith("model:"):
        reference = key.split(":", 1)[1].strip()
        if not reference:
            raise ValueError("opponent 'model:' needs a checkpoint reference after the colon")
        return key
    if key.startswith("difficulty:"):
        level = key.split(":", 1)[1].strip()
        from .difficulty import DIFFICULTY_NAMES

        if level not in DIFFICULTY_NAMES:
            raise ValueError(
                f"unknown difficulty level {level!r}; use one of {sorted(DIFFICULTY_NAMES)}"
            )
        return key
    raise ValueError(
        f"Unknown opponent {name!r}. Use {list(SCRIPTED_OPPONENTS)}, 'self', "
        "'difficulty:<level>' or 'model:<checkpoint.pt>'"
    )


class DifficultyOpponent:
    """A scripted opponent whose execution quality is reduced on purpose.

    This is what makes the four difficulty levels *real*: the opponent
    plays the same way, but it makes mistakes, reacts slowly and wastes
    boost according to the level.  The three knobs are independent, so a
    level can be made easier in one dimension without touching the others.

    The wrapper never changes the *policy* of a trained model - it only
    degrades a scripted behaviour, so a measurement of a trained policy
    against a difficulty level still measures the policy.
    """

    def __init__(
        self,
        base: Opponent,
        *,
        name: str,
        mistake_rate: float = 0.0,
        reaction_delay: float = 0.0,
        boost_efficiency: float = 1.0,
        seed: int = 0,
        execution: dict | None = None,
    ):
        self.base = base
        self.name = name
        self.mistake_rate = max(0.0, min(1.0, float(mistake_rate)))
        self.reaction_delay = max(0.0, min(2.0, float(reaction_delay)))
        self.boost_efficiency = max(0.0, min(1.0, float(boost_efficiency)))
        self.rng = np.random.default_rng(seed)
        self._ticks = 0
        self._last_actions: dict[str, int] = {}
        #: how often the opponent had to repeat an action because it was
        #: still "reacting"; reported by the analysis views
        self.reactions = 0
        self.mistakes = 0
        if execution:
            self.apply_execution(execution)

    def apply_execution(self, execution: dict) -> None:
        """Apply adapted execution parameters.

        This is the only way the opponent changes during a run.  It never
        touches a trained policy and never leaves the documented ranges.
        """
        for key, low, high in (
            ("mistake_rate", 0.0, 1.0),
            ("reaction_delay", 0.0, 2.0),
            ("boost_efficiency", 0.0, 1.0),
        ):
            if key in execution and execution[key] is not None:
                setattr(self, key, max(low, min(high, float(execution[key]))))

    def reset(self, env) -> None:
        self._ticks = 0
        self._last_actions.clear()
        self.reactions = 0
        self.mistakes = 0
        self.base.reset(env)

    def act(self, env, observations, agents):
        self._ticks += 1
        actions = self.base.act(env, observations, agents)
        if not actions:
            return actions
        # A slow opponent repeats its previous action while it "thinks".
        if self.reaction_delay > 0:
            interval = max(1, int(round(1.0 / max(self.reaction_delay, 1e-6) / 8)))
            if self._ticks % interval:
                self.reactions += 1
                return {
                    agent: self._last_actions.get(agent, action) for agent, action in actions.items()
                }
        result: dict[str, int] = {}
        for agent, action in actions.items():
            value = int(action)
            if self.boost_efficiency < 1.0 and _is_boost_action(value):
                if float(self.rng.random()) > self.boost_efficiency:
                    # the boost was missed: the car drives the same way but
                    # without the boost
                    value = _nearest_ground_action(value, boost=0)
            if self.mistake_rate > 0 and float(self.rng.random()) < self.mistake_rate:
                value = _weaker_action(value, self.rng)
                self.mistakes += 1
            result[agent] = value
            self._last_actions[agent] = value
        return result


def _is_boost_action(action: int) -> bool:
    try:
        return bool(ACTION_TABLE[action][6])
    except (IndexError, TypeError):
        return False


def _nearest_ground_action(action: int, boost: int) -> int:
    """The same ground action with a different boost flag."""
    try:
        row = ACTION_TABLE[action]
    except (IndexError, TypeError):
        return action
    target = np.array([row[0], row[1], 0, row[3], 0, 0, boost, 0], dtype=np.float32)
    matches = np.where((ACTION_TABLE == target).all(axis=1))[0]
    if len(matches) == 0:
        return action
    return int(matches[0])


def _weaker_action(action: int, rng: np.random.Generator) -> int:
    """A slightly worse version of the chosen action."""
    try:
        row = ACTION_TABLE[action]
    except (IndexError, TypeError):
        return action
    throttle, steer = int(row[0]), int(row[1])
    # less throttle or less steering - both make the bot slower and less
    # accurate without freezing it completely
    if throttle and rng.random() < 0.5:
        throttle = 0
    elif steer and rng.random() < 0.5:
        steer = 0
    target = np.array([throttle, steer, 0, steer, 0, 0, 0, 0], dtype=np.float32)
    matches = np.where((ACTION_TABLE == target).all(axis=1))[0]
    if len(matches) == 0:
        return action
    return int(matches[0])


class BrainDrivenOpponent:
    """A scripted opponent whose *behaviour* comes from the Bot Brain.

    The brain decides what the bot wants to do (chase, attack, rotate,
    shoot); this class turns that intent into controller inputs.  The
    separation is what makes the behaviour configurable: the profile
    changes the intent, the difficulty level changes how well the intent
    is executed.
    """

    def __init__(self, brain, *, name: str = "brain", seed: int = 0):
        self.brain = brain
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.last_intent = "chase"
        self.intent_counts: dict[str, int] = {}

    def reset(self, env) -> None:
        self.last_intent = "chase"
        self.intent_counts = {}

    def _context(self, env, agent: str) -> "BrainContext":
        from .botbrain import BrainContext

        state = env.state
        car = state.cars[agent]
        ball = np.asarray(state.ball.position, dtype=np.float32)
        position = np.asarray(car.physics.position, dtype=np.float32)
        # the own goal is the one the car defends: blue defends the
        # negative y, orange the positive y
        own_goal = np.array([0.0, -9200.0 if car.team_num == 0 else 9200.0, 0.0], dtype=np.float32)
        to_ball = ball - position
        opponents = [
            other
            for other in state.cars
            if other != agent and state.cars[other].team_num != car.team_num
        ]
        opponents_closer = 0
        for other in opponents:
            other_position = np.asarray(state.cars[other].physics.position, dtype=np.float32)
            if float(np.linalg.norm(other_position - ball)) < float(np.linalg.norm(to_ball)):
                opponents_closer += 1
        return BrainContext(
            distance_to_ball=float(np.linalg.norm(to_ball)),
            ball_ball_to_own_goal=float(np.linalg.norm(ball - own_goal)),
            score_diff=int(getattr(env, "score_for", 0)) - int(getattr(env, "score_against", 0)),
            time_remaining=float(env.env.shared_info.get("time_remaining", 300.0)),
            boost=float(car.boost_amount),
            on_ground=bool(car.on_ground),
            can_shoot=bool(car.on_ground and float(np.linalg.norm(to_ball)) < 2500.0),
            ball_in_attacking_third=bool(
                (ball[1] > 0) if car.team_num == 0 else (ball[1] < 0)
            ),
            opponents_closer=opponents_closer,
        )

    def act(self, env, observations, agents):
        actions: dict[str, int] = {}
        for agent in agents:
            decision = self.brain.decide(self._context(env, agent))
            self.last_intent = decision.intent
            self.intent_counts[decision.intent] = (
                self.intent_counts.get(decision.intent, 0) + 1
            )
            actions[agent] = self._act_for(env, agent, decision)
        return actions

    def _act_for(self, env, agent: str, decision) -> int:
        state = env.state
        car = state.cars[agent]
        if car.is_demoed:
            return 0
        ball = np.asarray(state.ball.position, dtype=np.float32)
        position = np.asarray(car.physics.position, dtype=np.float32)
        if decision.intent == "rotate":
            own_goal = np.array(
                [0.0, -9200.0 if car.team_num == 0 else 9200.0, 0.0], dtype=np.float32
            )
            target = own_goal + (ball - own_goal) * 0.35
        else:
            target = ball
        to_target = np.asarray(target, dtype=np.float32) - position
        flat = to_target[:2]
        distance = float(np.linalg.norm(flat))
        if distance < 1e-3:
            return _ground_index(1, 0, 0)
        forward = np.asarray(car.physics.forward, dtype=np.float32)
        cross = forward[0] * flat[1] - forward[1] * flat[0]
        dot = forward[0] * flat[0] + forward[1] * flat[1]
        angle = float(np.arctan2(cross, dot))
        steer = 0 if abs(angle) < 0.08 else (1 if angle > 0 else -1)
        aligned = abs(angle) < 0.6
        parameters = decision.parameters
        boost_wanted = (
            aligned
            and distance > 900
            and float(car.boost_amount) > 15
            and float(np.linalg.norm(car.physics.linear_velocity)) < 2100
        )
        if decision.intent in ("attack", "shoot"):
            boost = int(boost_wanted and parameters.get("boost_eagerness", 0.6) > 0.5)
        elif decision.intent == "rotate":
            boost = int(boost_wanted and parameters.get("boost_eagerness", 0.6) > 0.8)
        else:
            boost = int(boost_wanted and parameters.get("boost_eagerness", 0.6) > 0.7)
        throttle = 1 if aligned or abs(angle) < 1.9 else 0
        return _ground_index(throttle, steer, boost)


def build_opponent(
    name: str,
    *,
    seed: int = 0,
    device: str = "cpu",
    difficulty: str | None = None,
    execution: dict | None = None,
) -> Opponent:
    """Build an opponent from a name.

    ``difficulty:<level>`` builds a scripted opponent with the behaviour
    *and* the execution quality of that difficulty level; the ``difficulty``
    keyword argument does the same for any other scripted name.

    ``execution`` overrides the *execution* parameters of a scripted
    opponent (``mistake_rate``, ``reaction_delay``, ``boost_efficiency``).
    It is how the adaptive match dynamics reach the opponent without
    changing the policy or the opponent's behaviour profile.
    """
    key = str(name or "").strip()
    if key.startswith("difficulty:"):
        level = key.split(":", 1)[1].strip()
        from .difficulty import brain_parameters, get_profile

        validate_opponent_name(key)
        profile = get_profile(level)
        parameters = brain_parameters(level)
        base = _behaviour_opponent(profile.behaviour, name=f"difficulty:{level}")
        return DifficultyOpponent(
            base,
            name=f"difficulty:{level}",
            mistake_rate=parameters.mistake_rate,
            reaction_delay=parameters.reaction_delay,
            boost_efficiency=profile.boost_efficiency,
            seed=seed,
            execution=execution,
        )
    key = validate_opponent_name(key, allow_self=False)
    if key == "idle":
        base: Opponent = IdleOpponent()
    elif key == "random":
        base = RandomOpponent(seed)
    elif key == "chase":
        base = ChaseOpponent()
    elif key.startswith("model:"):
        return ModelOpponent.from_checkpoint(key.split(":", 1)[1], device=device)
    else:  # pragma: no cover - validate_opponent_name already refused it
        raise ValueError(f"Unknown opponent {name!r}")
    if difficulty:
        from .difficulty import brain_parameters, get_profile

        profile = get_profile(difficulty)
        parameters = brain_parameters(difficulty)
        base = _behaviour_opponent(profile.behaviour, name=f"{key}:{difficulty}")
        return DifficultyOpponent(
            base,
            name=f"{key}:{difficulty}",
            mistake_rate=parameters.mistake_rate,
            reaction_delay=parameters.reaction_delay,
            boost_efficiency=profile.boost_efficiency,
            seed=seed,
        )
    if execution:
        # Only the execution quality changes - the behaviour stays exactly
        # the one the opponent had before.
        return DifficultyOpponent(
            base,
            name=key,
            mistake_rate=float(execution.get("mistake_rate", 0.0) or 0.0),
            reaction_delay=float(execution.get("reaction_delay", 0.0) or 0.0),
            boost_efficiency=float(execution.get("boost_efficiency", 1.0) or 1.0),
            seed=seed,
        )
    return base


def _behaviour_opponent(behaviour: str, *, name: str) -> Opponent:
    """The scripted opponent that implements a behaviour profile."""
    from .botbrain import BotBrain, PROFILES
    from .difficulty import brain_parameters

    if behaviour not in PROFILES:
        behaviour = "balanced"
    brain = BotBrain(profile=behaviour, parameters=brain_parameters("medium", behaviour))
    return BrainDrivenOpponent(brain, name=name)
