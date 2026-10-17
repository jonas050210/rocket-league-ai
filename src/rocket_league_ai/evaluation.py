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
    scenario_environment: Literal["kickoff-v1", "mixed-scenarios-v1"] | None = None
    extra_seeds: list[int] = Field(default_factory=list, max_length=4)

    @property
    def total_games(self) -> int:
        return self.games * (1 + len(self.extra_seeds))

    def seed_for_game(self, index: int) -> int:
        repeat, within = divmod(index, self.games)
        base = self.seed if repeat == 0 else self.extra_seeds[repeat - 1]
        return base + within

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
    seed: int = 0
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
    if len(games) != spec.total_games:
        raise ValueError(f"expected {spec.total_games} game records, received {len(games)}")
    if [game.index for game in games] != list(range(spec.total_games)):
        raise ValueError("game records must be complete and in order")
    blue = sum(1 for game in games if game.candidate_side == "blue")
    orange = spec.total_games - blue
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
        "games": spec.total_games, "seed": spec.seed, "extra_seeds": spec.extra_seeds, "inference": "greedy",
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
    required = {"candidate_checkpoint_id", "candidate_run_id", "opponent_id", "games",
                "win_rate", "win_rate_ci95", "goals_for", "goals_against"}
    for item in evaluations:
        if item["status"] != "completed" or not item.get("result"):
            continue
        if not required <= item["result"].keys():
            continue  # partial/legacy records are listed, never ranked on missing evidence
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


def elo_ratings(evaluations: list[dict], *, k: float = 32.0, initial: float = 1500.0) -> list[dict]:
    """Chronological Elo over genuine checkpoint-vs-checkpoint evaluations only.

    Scripted-baseline results never enter ratings (beating random is a gate,
    not a rating). Each evaluation is one match: score = (W + 0.5·D) / games.
    """
    ratings: dict[str, dict] = {}

    def entry(checkpoint_id, run_id):
        return ratings.setdefault(checkpoint_id, {"checkpoint_id": checkpoint_id, "run_id": run_id,
                                                  "rating": initial, "games": 0, "evaluations": 0})

    matches = [item for item in evaluations
               if item["status"] == "completed" and item.get("result")
               and item["result"].get("candidate_checkpoint_id")
               and str(item["result"].get("opponent_id", "")).startswith("checkpoint:")]
    for item in sorted(matches, key=lambda value: value["created"]):
        result = item["result"]
        candidate = entry(result["candidate_checkpoint_id"], result["candidate_run_id"])
        opponent_id = result["opponent_id"].split(":", 1)[1]
        opponent = ratings.get(opponent_id) or entry(opponent_id, None)
        expected = 1 / (1 + 10 ** ((opponent["rating"] - candidate["rating"]) / 400))
        score = (result["wins"] + 0.5 * result["draws"]) / result["games"]
        delta = k * (score - expected)
        candidate["rating"] += delta
        opponent["rating"] -= delta
        candidate["games"] += result["games"]
        opponent["games"] += result["games"]
        candidate["evaluations"] += 1
        opponent["evaluations"] += 1
    return sorted(ratings.values(), key=lambda value: value["rating"], reverse=True)


def experiment_journal(runs: list[dict], evaluations: list[dict]) -> list[dict]:
    """Chronological experiment record: what changed, how it measured.

    A verdict compares a run's latest evaluation with the previous evaluated
    run *against the same opponent*: improved/regressed only when Wilson
    intervals separate; otherwise unchanged (overlap) — never a reward guess.
    """
    by_checkpoint = {}
    for item in evaluations:
        if item["status"] == "completed" and item.get("result") and item["result"].get("candidate_checkpoint_id"):
            by_checkpoint.setdefault(item["result"]["candidate_checkpoint_id"], []).append(item)
    journal, previous_run, previous_eval = [], None, None
    for run in sorted(runs, key=lambda value: value["created"]):
        changed = []
        if previous_run:
            changed = sorted(key for key in run["config"]
                             if run["config"].get(key) != previous_run["config"].get(key))
        run_evals = [item for items in by_checkpoint.values() for item in items
                     if item["result"].get("candidate_run_id") == run["id"]]
        latest = run_evals[-1]["result"] if run_evals else None
        verdict = "unevaluated"
        if latest and previous_eval and latest["opponent_id"] == previous_eval["opponent_id"]:
            if latest["win_rate_ci95"][0] > previous_eval["win_rate_ci95"][1]:
                verdict = "improved"
            elif latest["win_rate_ci95"][1] < previous_eval["win_rate_ci95"][0]:
                verdict = "regressed"
            else:
                verdict = "unchanged (intervals overlap)"
        elif latest:
            verdict = "baseline"
        journal.append({
            "run_id": run["id"], "name": run["name"], "status": run["status"], "created": run["created"],
            "config_changes_vs_previous": changed, "evaluation": latest, "verdict": verdict,
        })
        previous_run = run
        if latest:
            previous_eval = latest
    return journal
