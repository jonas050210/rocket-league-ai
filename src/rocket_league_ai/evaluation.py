"""Reproducible checkpoint evaluation: contracts, counting and statistics.

Control-plane only. The genuine game play happens in
``training.evaluate`` (a separate subprocess using the real simulator and a
frozen checkpoint actor). This module validates manifests, counts outcomes
from recorded per-game records, and computes Wilson score intervals so a win
rate is never displayed without its uncertainty. Training reward is never
converted into a win rate here.
"""

import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

OPPONENTS = {
    "random-v1": "Uniform random choice among the 90 discrete actions, seeded per game.",
    "idle-v1": "Always selects lookup-table action 8, the all-zero (no driving input) row.",
}


class EvaluationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    checkpoint_id: str
    opponent_id: Literal["random-v1", "idle-v1"] = "random-v1"
    opponent_run_id: str | None = None
    opponent_checkpoint_id: str | None = None
    games: int = Field(default=20, ge=2, le=400)
    seed: int = Field(default=1000, ge=0, le=2**31 - 1)
    max_game_seconds: float = Field(default=60.0, ge=5, le=300)
    record_games: int = Field(default=3, ge=0, le=10)
    parallel_workers: int = Field(default=1, ge=1, le=8)

    @model_validator(mode="after")
    def check_spec(self) -> Self:
        if self.games % 2:
            raise ValueError("games must be even so the candidate plays both sides equally")
        if (self.opponent_run_id is None) != (self.opponent_checkpoint_id is None):
            raise ValueError("a checkpoint opponent needs both opponent_run_id and opponent_checkpoint_id")
        return self


def effective_opponent_id(spec: EvaluationSpec) -> str:
    if spec.opponent_checkpoint_id:
        return f"checkpoint:{spec.opponent_checkpoint_id}"
    return spec.opponent_id


class GameRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    candidate_side: Literal["blue", "orange"]
    result: Literal["win", "loss", "draw"]
    goals_for: int = Field(ge=0)
    goals_against: int = Field(ge=0)
    decisions: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)


def wilson_interval(wins: int, games: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a win proportion (draws count as non-wins)."""
    if games <= 0:
        return (0.0, 0.0)
    proportion = wins / games
    denominator = 1 + z * z / games
    centre = (proportion + z * z / (2 * games)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / games + z * z / (4 * games * games)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def summarize(spec: EvaluationSpec, games: list[GameRecord], *, candidate_id: str,
              candidate_config_hash: str, environment_version: str) -> dict:
    """Build the persisted evaluation manifest from genuine per-game records."""
    if len(games) != spec.games:
        raise ValueError(f"expected {spec.games} game records, received {len(games)}")
    if [game.index for game in games] != list(range(spec.games)):
        raise ValueError("game records must be complete and in order")
    blue = sum(1 for game in games if game.candidate_side == "blue")
    orange = spec.games - blue
    if blue != orange:
        raise ValueError("evaluation is not side-balanced")
    wins = sum(1 for game in games if game.result == "win")
    losses = sum(1 for game in games if game.result == "loss")
    draws = spec.games - wins - losses
    lower, upper = wilson_interval(wins, spec.games)
    opponent = effective_opponent_id(spec)
    return {
        "candidate_run_id": spec.run_id, "candidate_checkpoint_id": spec.checkpoint_id,
        "candidate_id": candidate_id, "candidate_config_hash": candidate_config_hash,
        "opponent_id": opponent,
        "opponent_description": OPPONENTS.get(spec.opponent_id) if not spec.opponent_checkpoint_id
        else "Frozen checkpoint actor (greedy inference, weights never updated during evaluation).",
        "games": spec.games, "seed": spec.seed, "inference": "greedy",
        "environment_version": environment_version,
        "wins": wins, "losses": losses, "draws": draws,
        "goals_for": sum(game.goals_for for game in games),
        "goals_against": sum(game.goals_against for game in games),
        "win_rate": wins / spec.games, "win_rate_ci95": [lower, upper],
        "side_balanced": True,
        "games_detail": [game.model_dump() for game in games],
    }


def compare_evaluations(evaluations: list[dict]) -> list[dict]:
    """Rank evaluated checkpoints by the *lower* Wilson bound, honestly labelled.

    Ranking by the lower confidence bound avoids promoting a checkpoint on a
    lucky small sample. Entries against different opponents are never merged.
    """
    ranked = []
    for item in evaluations:
        if item["status"] != "completed" or not item.get("result"):
            continue
        result = item["result"]
        ranked.append({
            "evaluation_id": item["id"], "candidate_checkpoint_id": result["candidate_checkpoint_id"],
            "candidate_run_id": result["candidate_run_id"], "opponent_id": result["opponent_id"],
            "games": result["games"], "win_rate": result["win_rate"],
            "win_rate_ci95": result["win_rate_ci95"],
            "score_lower_bound": result["win_rate_ci95"][0],
            "goals_for": result["goals_for"], "goals_against": result["goals_against"],
            "created": item["created"],
        })
    return sorted(ranked, key=lambda entry: entry["score_lower_bound"], reverse=True)
