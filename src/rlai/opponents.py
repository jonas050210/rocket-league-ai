"""Opponent policies used during training and evaluation.

These are real policies, not scripted replays: they read the live simulation
state every decision tick.  The chaser is intentionally simple but non-trivial
(it drives at the ball and boosts), which makes it a meaningful first
milestone and a stable evaluation baseline.  Trained checkpoints can be used
as opponents through ``model:<path>``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .actions import ACTION_TABLE, ACTION_COUNT
from .policy import PolicyConfig, TorchPolicy


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
    def from_checkpoint(cls, path: str | Path, device: str = "cpu", name: str | None = None):
        import torch

        payload = torch.load(Path(path), map_location=device, weights_only=False)
        config = PolicyConfig.from_dict(payload.get("policy_config"))
        policy = TorchPolicy.from_config(config, device=device)
        policy.model.load_state_dict(payload["model"])
        policy.model.eval()
        return cls(policy, name=name or Path(path).parent.name, deterministic=True)

    def reset(self, env) -> None:
        pass

    def act(self, env, observations, agents):
        if not agents:
            return {}
        obs = {agent: observations[agent] for agent in agents}
        actions, _, _ = self.policy.act(obs, deterministic=self.deterministic)
        return actions


def build_opponent(name: str, *, seed: int = 0, device: str = "cpu") -> Opponent:
    key = name.strip()
    if key == "idle":
        return IdleOpponent()
    if key == "random":
        return RandomOpponent(seed)
    if key == "chase":
        return ChaseOpponent()
    if key.startswith("model:"):
        return ModelOpponent.from_checkpoint(key.split(":", 1)[1], device=device)
    raise ValueError(f"Unknown opponent {name!r}. Use idle, random, chase or model:<checkpoint.pt>")
