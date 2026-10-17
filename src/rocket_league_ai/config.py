"""Small, strict application config. Never import training libraries in the service."""

import hashlib
import json
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    name: str = Field(default="Foundation · 1v1", min_length=1, max_length=80)
    device: Literal["cuda:0", "cpu"] = "cuda:0"
    seed: int = Field(default=123, ge=0, le=2**31 - 1)
    n_proc: int = Field(default=1, ge=1, le=8)
    total_timesteps: int = Field(default=8192, ge=128, le=1_000_000)
    timesteps_per_iteration: int = Field(default=512, ge=128, le=65536)
    batch_size: int = Field(default=512, ge=128, le=65536)
    buffer_size: int = Field(default=1536, ge=128, le=196608)
    n_minibatches: int = Field(default=2, ge=1, le=64)
    n_epochs: int = Field(default=1, ge=1, le=4)
    learning_rate: float = Field(default=0.0002, gt=0, le=0.01)
    entropy_coefficient: float = Field(default=0.01, ge=0, le=0.1)
    gamma: float = Field(default=0.99, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, gt=0, le=1)
    hidden_sizes: tuple[int, ...] = (256, 256, 256)
    checkpoint_every_iterations: int = Field(default=2, ge=1, le=100)
    episode_timeout_seconds: float = Field(default=30, ge=1, le=300)
    no_touch_timeout_seconds: float = Field(default=10, ge=1, le=60)
    pin_memory: bool = False
    # Deliberately fixed compatibility contracts, not speculative curriculum knobs.
    team_size: Literal[1] = 1
    tick_skip: Literal[8] = 8
    environment_version: Literal["kickoff-v1"] = "kickoff-v1"
    reward_version: Literal["goal10-touch0.1-transition-v1"] = "goal10-touch0.1-transition-v1"

    @model_validator(mode="after")
    def validate_relationships(self) -> Self:
        if not 1 <= len(self.hidden_sizes) <= 4 or any(
            type(size) is not int or not 16 <= size <= 512 for size in self.hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain 1–4 integer widths between 16 and 512")
        if self.batch_size != self.timesteps_per_iteration:
            raise ValueError("batch_size must equal timesteps_per_iteration in the foundation adapter")
        if not self.batch_size <= self.buffer_size <= 3 * self.batch_size:
            raise ValueError("buffer_size must be between one and three batches")
        if self.batch_size % self.n_minibatches or self.batch_size // self.n_minibatches < 16:
            raise ValueError("n_minibatches must divide batch_size with at least 16 samples per minibatch")
        if self.total_timesteps % self.timesteps_per_iteration:
            raise ValueError("total_timesteps must be a multiple of timesteps_per_iteration")
        if self.timesteps_per_iteration % (2 * self.n_proc):
            raise ValueError("timesteps_per_iteration must be divisible by 2 * n_proc (1v1 agent steps)")
        if self.no_touch_timeout_seconds > self.episode_timeout_seconds:
            raise ValueError("no_touch_timeout_seconds must not exceed episode_timeout_seconds")
        if self.device == "cpu" and self.pin_memory:
            raise ValueError("pin_memory requires an explicitly selected CUDA device")
        return self

    def digest(self) -> str:
        return digest(self.model_dump(mode="json"))


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def changes(before: TrainingConfig, after: TrainingConfig) -> dict:
    old, new = before.model_dump(mode="json"), after.model_dump(mode="json")
    return {key: {"before": old[key], "after": value} for key, value in new.items() if old[key] != value}
