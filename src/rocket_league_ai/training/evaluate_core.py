"""Simulator-agnostic evaluation game loop (NumPy only; unit-testable).

The loop counts outcomes from genuine terminal rewards: in this environment a
goal is the only true termination and the scorer receives +10 (conceded -10),
so a terminal candidate reward above +5 is a scored goal. Truncation (time or
no-touch limits) without a goal is a draw. No outcome is ever inferred from
training reward averages.
"""

import numpy as np

from ..evaluation import GameRecord

IDLE_ACTION = 8  # LookupTableAction row 8 is the all-zero (no input) row.
DECISIONS_PER_SECOND = 15  # 120 Hz physics with tick skip 8.


def opponent_action(opponent_id: str, rng: np.random.Generator) -> int:
    if opponent_id == "random-v1":
        return int(rng.integers(0, 90))
    if opponent_id == "idle-v1":
        return IDLE_ACTION
    raise ValueError(f"Unknown opponent: {opponent_id}")


def extract_frame(state, game: int, decisions: int, score: tuple[int, int]) -> dict | None:
    """Best-effort genuine frame from a RocketSim GameState; None if unavailable."""
    try:
        ball = getattr(state.ball, "physics", state.ball)
        cars = []
        for agent_id, car in state.cars.items():
            physics = car.physics
            cars.append({
                "agent": agent_id, "team": "blue" if car.is_blue else "orange",
                "position": [float(value) for value in physics.position],
                "velocity": [float(value) for value in physics.linear_velocity],
                "yaw": float(physics.yaw), "boost": float(car.boost_amount),
                "on_ground": bool(getattr(car, "on_ground", True)),
                "is_boosting": bool(getattr(car, "is_boosting", False)),
                "is_supersonic": bool(getattr(car, "is_supersonic", False)),
            })
        return {
            "game": game, "t": decisions / DECISIONS_PER_SECOND,
            "ball": [float(value) for value in ball.position],
            "ball_velocity": [float(value) for value in ball.linear_velocity],
            "cars": cars, "score": list(score),
        }
    except (AttributeError, TypeError, ValueError):
        return None


def play_game(env, candidate_action, opponent_id: str, *, index: int, candidate_side: str,
              seed: int, max_decisions: int, record: bool, frames: list | None = None,
              opponent_action_fn=None, action_table=None) -> GameRecord:
    """Play one genuine game. ``candidate_action(agent_id, obs) -> int`` is frozen inference.

    ``opponent_action_fn(agent_id, obs, rng) -> int`` overrides the scripted
    opponent (used for frozen checkpoint opponents); both policies are
    inference-only.
    """
    rng = np.random.default_rng(seed)
    blue_agent, orange_agent = "blue-0", "orange-0"
    candidate_agent = blue_agent if candidate_side == "blue" else orange_agent
    opponent_agent = orange_agent if candidate_side == "blue" else blue_agent
    observations = env.reset()
    decisions = 0
    score = (0, 0)
    result, goals_for, goals_against = "draw", 0, 0
    while decisions < max_decisions:
        chosen = candidate_action(candidate_agent, observations[candidate_agent])
        candidate_choice, candidate_prob = chosen if isinstance(chosen, tuple) else (chosen, None)
        opponent = (opponent_action_fn(opponent_agent, observations[opponent_agent], rng)
                    if opponent_action_fn else opponent_action(opponent_id, rng))
        if record and frames is not None:
            frame = extract_frame(env.state, index, decisions, score)
            if frame is not None:
                frame["candidate_action"] = int(candidate_choice)
                frame["candidate_action_prob"] = candidate_prob
                frame["opponent_action"] = int(opponent)
                if action_table is not None:
                    frame["candidate_action_vector"] = [float(v) for v in action_table[int(candidate_choice)]]
                frames.append(frame)
        actions = {
            candidate_agent: np.array([candidate_choice], dtype=np.int64),
            opponent_agent: np.array([opponent], dtype=np.int64),
        }
        observations, rewards, terminated, truncated = env.step(actions)
        decisions += 1
        if all(terminated.values()):
            if rewards[candidate_agent] > 5:
                result, goals_for = "win", 1
                score = (score[0] + (1 if candidate_side == "blue" else 0),
                         score[1] + (1 if candidate_side == "orange" else 0))
            elif rewards[candidate_agent] < -5:
                result, goals_against = "loss", 1
                score = (score[0] + (1 if candidate_side == "orange" else 0),
                         score[1] + (1 if candidate_side == "blue" else 0))
            break
        if all(truncated.values()):
            break
    if record and frames is not None:
        frame = extract_frame(env.state, index, decisions, score)
        if frame is not None:
            frames.append(frame)
    return GameRecord(
        index=index, seed=seed, candidate_side=candidate_side, result=result,
        goals_for=goals_for, goals_against=goals_against,
        decisions=decisions, duration_seconds=decisions / DECISIONS_PER_SECOND,
    )
