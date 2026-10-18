"""Configuration models and filesystem layout."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from .modes import mode_ids
from .obs import OBS_SIZE
from .rewards import RewardWeights


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("RLAI_DATA_DIR", project_root() / "data")).resolve()


def ensure_dirs(root: Path | None = None) -> dict[str, Path]:
    root = Path(root) if root else data_dir()
    paths = {
        "root": root,
        "runs": root / "runs",
        "models": root / "models",
        "logs": root / "logs",
        "tmp": root / "tmp",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


class PolicySettings(BaseModel):
    hidden: list[int] = Field(default_factory=lambda: [256, 256], min_length=1, max_length=4)
    activation: str = "tanh"

    @field_validator("hidden")
    @classmethod
    def _hidden_valid(cls, value: list[int]) -> list[int]:
        for x in value:
            if not 16 <= int(x) <= 2048:
                raise ValueError("hidden layer sizes must be between 16 and 2048")
        return [int(x) for x in value]

    @field_validator("activation")
    @classmethod
    def _activation_valid(cls, value: str) -> str:
        if value not in {"tanh", "relu", "elu"}:
            raise ValueError("activation must be tanh, relu or elu")
        return value


class TrainingConfig(BaseModel):
    run_name: str = Field(default="1v1 baseline", min_length=1, max_length=80)
    mode: str = "soccer-1v1"
    seed: int = 42
    total_steps: int = Field(default=200_000, ge=2_000, le=5_000_000_000)
    rollout_steps: int = Field(default=512, ge=32, le=8192)
    n_workers: int = Field(default=2, ge=1, le=32)
    tick_skip: int = Field(default=8, ge=1, le=16)
    episode_seconds: float = Field(default=60.0, ge=5, le=600)
    learning_rate: float = Field(default=3e-4, gt=0, le=0.1)
    gamma: float = Field(default=0.995, gt=0, lt=1)
    gae_lambda: float = Field(default=0.95, gt=0, lt=1)
    clip_coef: float = Field(default=0.2, gt=0, le=1)
    value_coef: float = Field(default=0.5, ge=0, le=10)
    entropy_coef: float = Field(default=0.01, ge=0, le=1)
    epochs: int = Field(default=4, ge=1, le=30)
    minibatch_size: int = Field(default=1024, ge=64, le=65536)
    max_grad_norm: float = Field(default=0.5, gt=0, le=10)
    lr_anneal: bool = True
    device: str = "auto"
    opponent: str = "chase"
    self_play: bool = False
    snapshot_interval: int = Field(default=5, ge=1, le=1000)
    checkpoint_interval: int = Field(default=10, ge=1, le=10000)
    keep_checkpoints: int = Field(default=8, ge=1, le=100)
    reward: RewardWeights = Field(default_factory=RewardWeights)
    policy: PolicySettings = Field(default_factory=PolicySettings)
    episodes_per_eval: int = Field(default=20, ge=1, le=1000)
    eval_opponent: str = "chase"

    @field_validator("mode")
    @classmethod
    def _mode_valid(cls, value: str) -> str:
        if value not in mode_ids():
            raise ValueError(f"mode must be one of {mode_ids()}")
        return value

    @field_validator("device")
    @classmethod
    def _device_valid(cls, value: str) -> str:
        if value not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda or mps")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "TrainingConfig":
        if self.total_steps < self.rollout_steps * self.n_workers:
            raise ValueError(
                "total_steps must be at least rollout_steps * n_workers "
                f"({self.rollout_steps * self.n_workers})"
            )
        if self.minibatch_size > self.rollout_steps * self.n_workers:
            raise ValueError("minibatch_size cannot exceed one rollout batch")
        return self

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def policy_config(self):
        from .policy import PolicyConfig

        return PolicyConfig(
            obs_size=OBS_SIZE,
            hidden=tuple(self.hidden_tuple),
            activation=self.policy.activation,
        )

    @property
    def hidden_tuple(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.policy.hidden)

    def to_dict(self) -> dict:
        data = self.model_dump()
        data["reward"] = self.reward.to_dict()
        return data

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


class AppSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    auth_token: str | None = None
    allow_live_launch: bool = True
    rlbot_python: str | None = None


DEFAULT_TRAINING = TrainingConfig()
