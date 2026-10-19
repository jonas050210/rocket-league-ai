"""Actor-critic policy and action selection.

Separate actor and critic MLPs (as in Necto/rocket-learn) with orthogonal
initialisation.  The architecture is intentionally small: a few hundred
thousand parameters is plenty for a 190-dim observation and trains on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import torch
from torch import nn

from .actions import ACTION_COUNT
from .obs import OBS_SIZE, mirror_observation, version_for_obs_size


def ortho_init(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)


def clone_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Detached, CPU-resident **copy** of a state dict.

    ``tensor.detach().cpu()`` returns the *same* tensor when the model already
    lives on the CPU, so a "frozen" snapshot taken that way silently follows
    every later ``optimizer.step()``.  Self-play opponents and checkpoint
    payloads must be real copies; this helper is the single place that
    guarantees it.
    """
    return {
        str(key): value.detach().to("cpu", copy=True).contiguous()
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }


class MLP(nn.Module):
    def __init__(self, sizes: Iterable[int], activation: type[nn.Module] = nn.Tanh):
        super().__init__()
        layers: list[nn.Module] = []
        sizes = list(sizes)
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_size: int = OBS_SIZE,
        n_actions: int = ACTION_COUNT,
        hidden: tuple[int, ...] = (256, 256),
        activation: str = "tanh",
    ):
        super().__init__()
        act = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[activation]
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.hidden = tuple(hidden)
        self.activation = activation
        self.actor = MLP((obs_size, *hidden, n_actions), act)
        self.critic = MLP((obs_size, *hidden, 1), act)
        self._initialise()

    def _initialise(self) -> None:
        for module in self.actor.net:
            if isinstance(module, nn.Linear):
                ortho_init(module, np.sqrt(2))
        for module in self.critic.net:
            if isinstance(module, nn.Linear):
                ortho_init(module, np.sqrt(2))
        actor_out = [m for m in self.actor.net if isinstance(m, nn.Linear)][-1]
        critic_out = [m for m in self.critic.net if isinstance(m, nn.Linear)][-1]
        ortho_init(actor_out, 0.01)
        ortho_init(critic_out, 1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def config(self) -> dict:
        return {
            "obs_size": self.obs_size,
            "n_actions": self.n_actions,
            "hidden": list(self.hidden),
            "activation": self.activation,
        }


@dataclass(frozen=True)
class PolicyConfig:
    obs_size: int = OBS_SIZE
    n_actions: int = ACTION_COUNT
    hidden: tuple[int, ...] = (256, 256)
    activation: str = "tanh"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["hidden"] = list(self.hidden)
        return data

    @classmethod
    def from_dict(cls, data: dict | None) -> "PolicyConfig":
        if not data:
            return cls()
        return cls(
            obs_size=int(data.get("obs_size", OBS_SIZE)),
            n_actions=int(data.get("n_actions", ACTION_COUNT)),
            hidden=tuple(int(x) for x in data.get("hidden", (256, 256))),
            activation=str(data.get("activation", "tanh")),
        )

    def build(self) -> ActorCritic:
        return ActorCritic(
            obs_size=self.obs_size,
            n_actions=self.n_actions,
            hidden=self.hidden,
            activation=self.activation,
        )


class TorchPolicy:
    """Inference wrapper used by workers, evaluation and the RLBot bot."""

    def __init__(self, model: ActorCritic, device: str | torch.device = "cpu"):
        self.model = model.to(device).eval()
        self.device = torch.device(device)

    @classmethod
    def from_config(cls, config: PolicyConfig, device: str = "cpu") -> "TorchPolicy":
        return cls(config.build(), device)

    @torch.no_grad()
    def act_batch(
        self,
        batch: np.ndarray,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Array in, arrays out: the hot path used by rollout workers.

        Profiling a 1v1 rollout showed ~40 % of ``act`` time inside
        ``torch.distributions.Categorical`` construction/sampling - pure
        Python overhead around a 1-row MLP.  This path does the same math
        with ``log_softmax`` + ``multinomial`` and no dict round-trips.
        ``batch`` is ``(K, obs_size)``; returns int64 actions and float32
        logprobs/values, each ``(K,)``.
        """
        tensor = torch.as_tensor(
            np.asarray(batch, dtype=np.float32), device=self.device
        )
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        logits, values = self.model(tensor)
        log_probs = torch.log_softmax(logits, dim=-1)
        if deterministic:
            actions = torch.argmax(log_probs, dim=-1)
        else:
            actions = torch.multinomial(
                log_probs.exp(), 1, generator=generator
            ).squeeze(-1)
        chosen = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        return (
            actions.cpu().numpy().astype(np.int64),
            chosen.cpu().numpy().astype(np.float32),
            values.cpu().numpy().astype(np.float32),
        )

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray | dict,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[dict[str, int], dict[str, float], dict[str, float]] | tuple[int, float, float]:
        single = isinstance(obs, np.ndarray) and obs.ndim == 1
        batch = (
            np.asarray(obs, dtype=np.float32)
            if single
            else np.stack([np.asarray(obs[k], dtype=np.float32) for k in obs])
        )
        keys = None if single else list(obs)
        actions, logprobs, values = self.act_batch(
            batch, deterministic=deterministic, generator=generator
        )
        if single:
            return int(actions[0]), float(logprobs[0]), float(values[0])
        return (
            {k: int(a) for k, a in zip(keys, actions.tolist())},
            {k: float(lp) for k, lp in zip(keys, logprobs.tolist())},
            {k: float(v) for k, v in zip(keys, values.tolist())},
        )

    def state_dict(self) -> dict:
        return clone_state_dict(self.model.state_dict())


class MirroredPolicy:
    """Same checkpoint, opposite side of the field.

    Rocket League is symmetric under a 180 deg rotation about the vertical
    axis.  ``mirror_observation`` rewrites an observation from the other side
    into the side the checkpoint was trained on, so a blue-trained policy can
    play orange without retraining.  Actions are car-relative and therefore
    pass through unchanged (see ``rlai.obs.mirror_observation``).
    """

    def __init__(self, inner: "TorchPolicy", obs_version: int | None = None):
        self.inner = inner
        self.model = inner.model  # obs_size / device introspection keeps working
        self.device = inner.device
        self.obs_version = (
            int(obs_version)
            if obs_version is not None
            else version_for_obs_size(int(inner.model.obs_size))
        )

    def _mirror(self, obs):
        if isinstance(obs, np.ndarray):
            return mirror_observation(obs, self.obs_version)
        return {key: mirror_observation(value, self.obs_version) for key, value in obs.items()}

    @torch.no_grad()
    def act(self, obs, *, deterministic: bool = False, generator: torch.Generator | None = None):
        return self.inner.act(
            self._mirror(obs), deterministic=deterministic, generator=generator
        )

    def state_dict(self) -> dict:
        return self.inner.state_dict()
