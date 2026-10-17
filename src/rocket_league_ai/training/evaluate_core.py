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
              seed: int, max_decisions: int, record: bool, frames: list | None = None) -> GameRecord:
    """Play one genuine game. ``candidate_action(agent_id, obs) -> int`` is frozen inference."""
    rng = np.random.default_rng(seed)
    blue_agent, orange_agent = "blue-0", "orange-0"
    candidate_agent = blue_agent if candidate_side == "blue" else orange_agent
    opponent_agent = orange_agent if candidate_side == "blue" else blue_agent
    observations = env.reset()
    decisions = 0
    score = (0, 0)
    result, goals_for, goals_against = "draw", 0, 0
    while decisions < max_decisions:
        if record and frames is not None:
            frame = extract_frame(env.state, index, decisions, score)
            if frame is not None:
                frames.append(frame)
        actions = {
            candidate_agent: np.array([candidate_action(candidate_agent, observations[candidate_agent])], dtype=np.int64),
            opponent_agent: np.array([opponent_action(opponent_id, rng)], dtype=np.int64),
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
        index=index, candidate_side=candidate_side, result=result,
        goals_for=goals_for, goals_against=goals_against,
        decisions=decisions, duration_seconds=decisions / DECISIONS_PER_SECOND,
    )
