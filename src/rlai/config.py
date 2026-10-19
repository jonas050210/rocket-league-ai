"""Configuration models and filesystem layout."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .modes import mode_ids
from .obs import OBS_SIZE_BY_VERSION, OBS_VERSION, obs_size
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
    clip_value: float = Field(default=0.2, ge=0, le=10)
    target_kl: float = Field(default=0.05, ge=0, le=10)
    lr_anneal: bool = True
    device: str = "auto"
    opponent: str = "chase"
    self_play: bool = False
    snapshot_interval: int = Field(default=5, ge=1, le=1000)
    #: a snapshot that is within this many steps of the newest one is only
    #: kept when its weights actually differ.  Without it the pool fills up
    #: with near-duplicates of the current policy.
    pool_duplicate_step_tolerance: int = Field(default=0, ge=0, le=100000)
    #: mean absolute weight difference below which two snapshots count as
    #: "the same policy"
    pool_duplicate_distance_tolerance: float = Field(default=1e-4, ge=0, le=1.0)
    #: how many snapshots the opponent pool keeps
    pool_capacity: int = Field(default=8, ge=1, le=64)
    #: store the frozen weights of the opponent pool inside checkpoints so a
    #: resumed self-play run keeps its real opponents (about one policy size per
    #: snapshot, at most ``OpponentPool.capacity`` of them).  Ratings are always
    #: stored; turn this off only when checkpoint size matters more.
    league_states: bool = True
    checkpoint_interval: int = Field(default=10, ge=1, le=10000)
    keep_checkpoints: int = Field(default=8, ge=1, le=100)
    reward: RewardWeights = Field(default_factory=RewardWeights)
    policy: PolicySettings = Field(default_factory=PolicySettings)
    obs_version: int = OBS_VERSION
    eval_opponent: str = "chase"
    #: optional multi-opponent suite; when empty ``eval_opponent`` is used
    eval_opponents: list[str] = Field(default_factory=list, max_length=8)
    eval_interval: int = Field(default=0, ge=0, le=100000)
    #: 4-game series made win-rate checkpoint selection mostly noise
    eval_games: int = Field(default=8, ge=1, le=100)
    eval_match_seconds: float = Field(default=30.0, ge=5, le=600)
    #: which side(s) the policy plays during evaluation; both = side balanced
    eval_sides: list[str] = Field(default_factory=lambda: ["blue"], min_length=1, max_length=2)
    #: rewrite the observation of the non-trained side into the trained side
    eval_mirror: bool = True
    #: measure the policy before the first update so the report has a "before"
    eval_before: bool = True
    #: make potential-based shaping discount-exact (F = gamma*Phi' - Phi).
    #: Off by default: exactness streams a small positive reward at rest and
    #: makes returns hard to read, for a negligible gain at gamma=0.995
    #: (see rlai.rewards.SoccerReward).
    discount_shaping: bool = False
    curriculum: bool = False
    #: starting-state distribution when no curriculum stage overrides it:
    #: ``kickoff`` (standard), ``drill`` (randomized attacking setups that
    #: teach ball contact far faster) or ``mixed``
    reset_mode: str = "kickoff"
    #: let workers play on either side so the policy learns both
    side_randomization: bool = False
    #: clamp ``n_workers`` to what this machine can actually run.  Turning
    #: this off reproduces the old behaviour (use exactly ``n_workers``),
    #: which is only sensible when the machine was measured beforehand.
    auto_worker_limit: bool = True
    #: how many rows per metric name are kept in the database.  The newest
    #: rows win, so the charts and the report still describe the end of the
    #: run.  ``0`` disables the pruning.
    metric_retention: int = Field(default=20_000, ge=0, le=5_000_000)
    #: how many log rows are kept per run
    log_retention: int = Field(default=5_000, ge=0, le=1_000_000)
    #: how many episode rows are kept per run
    episode_retention: int = Field(default=20_000, ge=0, le=5_000_000)
    #: write the retained rows to the database every N iterations instead of
    #: after every iteration.  1 keeps the old behaviour; higher values
    #: reduce the write overhead on slow disks.
    metric_flush_interval: int = Field(default=1, ge=1, le=1000)
    #: --- Adaptive Match Dynamics -------------------------------------
    #: adapt the *scripted* opponent to the measured result.  Off by
    #: default; a trained policy is never modified at runtime.
    adaptive_dynamics: bool = False
    #: the win rate the agent should reach against the opponent
    adaptive_target_win_rate: float = Field(default=0.6, ge=0.0, le=1.0)
    #: how far the measured win rate may drift before a change is made
    adaptive_tolerance: float = Field(default=0.15, ge=0.0, le=1.0)
    #: at least this many games between two changes
    adaptive_min_games: int = Field(default=10, ge=1, le=10000)
    #: at least this many seconds between two changes
    adaptive_min_seconds: float = Field(default=60.0, ge=0.0, le=86400.0)
    #: how much a single change moves the opponent (0..1)
    adaptive_step: float = Field(default=0.1, gt=0, le=1.0)
    #: --- Player Memory -------------------------------------------------
    #: remember the players the agent met.  The memory is data, never
    #: part of the model weights.
    player_memory: bool = True

    @field_validator("reward", mode="before")
    @classmethod
    def _reward_legacy_alias(cls, value):
        # Keep old configuration files working after the demo -> bump rename.
        if isinstance(value, dict) and "demo" in value:
            value = dict(value)
            if "bump" in value:
                raise ValueError("reward must use either 'bump' or the legacy 'demo'")
            value["bump"] = value.pop("demo")
        return value

    @field_validator("mode")
    @classmethod
    def _mode_valid(cls, value: str) -> str:
        if value not in mode_ids():
            raise ValueError(f"mode must be one of {mode_ids()}")
        return value

    @field_validator("opponent")
    @classmethod
    def _opponent_valid(cls, value: str) -> str:
        from .opponents import validate_opponent_name

        return validate_opponent_name(value)

    @field_validator("eval_opponent")
    @classmethod
    def _eval_opponent_valid(cls, value: str) -> str:
        from .opponents import validate_opponent_name

        return validate_opponent_name(value)

    @field_validator("eval_opponents")
    @classmethod
    def _eval_opponents_valid(cls, values: list[str]) -> list[str]:
        from .opponents import validate_opponent_name

        cleaned: list[str] = []
        for value in values:
            key = validate_opponent_name(value)
            if key not in cleaned:
                cleaned.append(key)
        return cleaned

    @field_validator("eval_sides")
    @classmethod
    def _eval_sides_valid(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            key = str(value).strip().lower()
            if key not in {"blue", "orange"}:
                raise ValueError("eval_sides entries must be 'blue' or 'orange'")
            if key not in cleaned:
                cleaned.append(key)
        if not cleaned:
            raise ValueError("eval_sides must not be empty")
        return cleaned

    @field_validator("reset_mode")
    @classmethod
    def _reset_mode_valid(cls, value: str) -> str:
        key = str(value or "").strip().lower()
        if key not in {"kickoff", "drill", "mixed"}:
            raise ValueError("reset_mode must be kickoff, drill or mixed")
        return key

    @field_validator("obs_version")
    @classmethod
    def _obs_version_valid(cls, value: int) -> int:
        if int(value) not in OBS_SIZE_BY_VERSION:
            raise ValueError(f"obs_version must be one of {sorted(OBS_SIZE_BY_VERSION)}")
        return int(value)

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
        if self.curriculum:
            # The stages are validated here, so a broken threshold is
            # reported when the configuration is checked and not in the
            # middle of a run.
            from .curriculum import stages_for_mode

            stages_for_mode(self.mode)
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
            obs_size=obs_size(self.obs_version),
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

    @property
    def observation_size(self) -> int:
        return obs_size(self.obs_version)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


#: hosts that only the local machine can reach
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
#: below this length a capability token is reported as weak
MIN_TOKEN_LENGTH = 12


class AppSettings(BaseModel):
    """How the API/GUI process is exposed.

    Same treatment as :class:`TrainingConfig`: validated in one place, used by
    the CLI *and* the FastAPI app, so the security defaults cannot drift apart.
    The default binds to the loopback interface - exposing the training API to
    the network is an explicit decision that also wants a token.
    """

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    auth_token: str | None = None
    cors_origins: list[str] = Field(default_factory=list, max_length=16)
    allow_live_launch: bool = True
    rlbot_python: str | None = None

    @field_validator("host")
    @classmethod
    def _host_valid(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("host must not be empty")
        if any(character in text for character in " /\\?#"):
            raise ValueError(
                f"host must be an interface address or name, got {value!r} "
                "(use 127.0.0.1 for local only, 0.0.0.0 for every interface)"
            )
        return text

    @field_validator("auth_token")
    @classmethod
    def _token_valid(cls, value: str | None) -> str | None:
        # A short token is a bad idea, but refusing to start would break setups
        # that already work; it is reported as a warning instead (see
        # ``weak_token``) so the choice stays with the operator.
        if value is None:
            return None
        return str(value).strip() or None

    @field_validator("cors_origins")
    @classmethod
    def _origins_valid(cls, values: list[str]) -> list[str]:
        cleaned = []
        for value in values:
            text = str(value or "").strip().rstrip("/")
            if not text:
                continue
            if text == "*":
                raise ValueError(
                    "cors_origins must list explicit origins; '*' would let any website "
                    "drive the training API from a visitor's browser"
                )
            if not text.startswith(("http://", "https://")):
                raise ValueError(
                    f"cors origin {value!r} must start with http:// or https://"
                )
            cleaned.append(text)
        return cleaned

    @property
    def is_public(self) -> bool:
        """True when the socket is reachable from other machines."""
        return self.host not in LOOPBACK_HOSTS

    @property
    def exposed_without_token(self) -> bool:
        return self.is_public and not self.auth_token

    @property
    def weak_token(self) -> bool:
        """True when a token is set but short enough to be guessable."""
        return bool(self.auth_token) and len(self.auth_token or "") < MIN_TOKEN_LENGTH

    @classmethod
    def from_env(cls, **overrides: Any) -> "AppSettings":
        """Build settings from the environment, then apply explicit overrides."""
        import os

        data: dict[str, Any] = {}
        if os.environ.get("RLAI_HOST"):
            data["host"] = os.environ["RLAI_HOST"]
        if os.environ.get("RLAI_PORT"):
            data["port"] = int(os.environ["RLAI_PORT"])
        if os.environ.get("RLAI_AUTH_TOKEN"):
            data["auth_token"] = os.environ["RLAI_AUTH_TOKEN"]
        origins = os.environ.get("RLAI_CORS_ORIGINS", "").strip()
        if origins:
            data["cors_origins"] = [item.strip() for item in origins.split(",") if item.strip()]
        data.update({key: value for key, value in overrides.items() if value is not None})
        return cls.model_validate(data)

    def export_env(self) -> dict[str, str]:
        """Environment the worker processes and the GUI need to agree on."""
        env: dict[str, str] = {}
        if self.auth_token:
            env["RLAI_AUTH_TOKEN"] = self.auth_token
        if self.cors_origins:
            env["RLAI_CORS_ORIGINS"] = ",".join(self.cors_origins)
        return env


DEFAULT_TRAINING = TrainingConfig()
