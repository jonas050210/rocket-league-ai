"""Lifecycle vocabulary and genuine extension contracts (no simulated implementations)."""

from enum import StrEnum
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunStatus(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    RESUMING = "resuming"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


ACTIVE = {
    RunStatus.STARTING, RunStatus.RUNNING, RunStatus.PAUSING,
    RunStatus.PAUSED, RunStatus.RESUMING, RunStatus.STOPPING,
}
TERMINAL = set(RunStatus) - ACTIVE


class AppError(Exception):
    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


def control_target(status: str, command: str) -> RunStatus:
    allowed = {
        "pause": ({RunStatus.RUNNING}, RunStatus.PAUSING),
        "resume": ({RunStatus.PAUSED}, RunStatus.RESUMING),
        "stop": (ACTIVE - {RunStatus.STOPPING}, RunStatus.STOPPING),
    }
    if command not in allowed:
        raise AppError("Unknown control; use pause, resume, or stop.", 422)
    states, target = allowed[command]
    if status not in states:
        raise AppError(f"Cannot {command} a {status} run. Refresh status and retry.")
    return target


class EvaluationResult(BaseModel):
    """Contract for a future real evaluator; no result is synthesized by training."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    candidate_id: str
    opponent_id: str
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    draws: int = Field(ge=0)
    games: int = Field(gt=0)
    seed: int
    side_balanced: bool
    inference: Literal["greedy"] = "greedy"
    environment_version: str
    goals_for: int = Field(ge=0)
    goals_against: int = Field(ge=0)

    @model_validator(mode="after")
    def check_counts(self) -> Self:
        if self.wins + self.losses + self.draws != self.games:
            raise ValueError("wins + losses + draws must equal games")
        if self.side_balanced and self.games % 2:
            raise ValueError("a side-balanced evaluation must contain an even number of games")
        return self


class Evaluator(Protocol):
    def evaluate(self, candidate_id: str, opponent_id: str, games: int, seed: int) -> EvaluationResult: ...


class OpponentPolicy(Protocol):
    def actions(self, agent_ids: list[str], observations: list, *, deterministic: bool) -> list: ...


class ObservationVisualizer(Protocol):
    def display(self, observation: object, metadata: dict) -> None: ...
