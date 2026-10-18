"""Proximal Policy Optimisation for discrete Rocket League actions.

Implements the standard clipped-surrogate PPO with GAE (Schulman et al.),
advantage normalisation, entropy bonus, global gradient clipping and an
optional linear learning-rate schedule.  The implementation is deliberately
small and auditable; it follows the same recipe as CleanRL/RLGym-PPO.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import TrainingConfig
from .policy import ActorCritic


@dataclass
class Batch:
    obs: np.ndarray
    actions: np.ndarray
    logprobs: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    values: np.ndarray


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE over a (T, K) rollout. ``dones`` marks episode ends."""
    T = rewards.shape[0]
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def flatten_rollout(rollout: dict, last_values: np.ndarray, config: TrainingConfig) -> Batch:
    advantages, returns = compute_gae(
        rollout["rewards"],
        rollout["values"],
        rollout["dones"],
        last_values,
        config.gamma,
        config.gae_lambda,
    )
    return Batch(
        obs=rollout["obs"].reshape(-1, rollout["obs"].shape[-1]),
        actions=rollout["actions"].reshape(-1),
        logprobs=rollout["logprobs"].reshape(-1),
        advantages=advantages.reshape(-1),
        returns=returns.reshape(-1),
        values=rollout["values"].reshape(-1),
    )


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    config: TrainingConfig,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
    progress: float = 0.0,
) -> dict[str, float]:
    obs = torch.as_tensor(batch.obs, device=device)
    actions = torch.as_tensor(batch.actions, device=device, dtype=torch.long)
    old_logprobs = torch.as_tensor(batch.logprobs, device=device)
    advantages = batch.advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages_t = torch.as_tensor(advantages, device=device)
    returns_t = torch.as_tensor(batch.returns, device=device)
    old_values_t = torch.as_tensor(batch.values, device=device)

    n = obs.shape[0]
    batch_size = min(config.minibatch_size, n)
    stats = {
        "losses/policy": 0.0,
        "losses/value": 0.0,
        "losses/entropy": 0.0,
        "losses/approx_kl": 0.0,
        "losses/clipfrac": 0.0,
        "losses/explained_variance": 0.0,
    }
    updates = 0

    model.train()
    for _ in range(config.epochs):
        indices = torch.randperm(n, device=device, generator=generator)
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            mb_obs = obs[idx]
            mb_actions = actions[idx]
            mb_old_logprobs = old_logprobs[idx]
            mb_advantages = advantages_t[idx]
            mb_returns = returns_t[idx]

            logits, values = model(mb_obs)
            dist = torch.distributions.Categorical(logits=logits)
            logprobs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            ratio = (logprobs - mb_old_logprobs).exp()
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (values - mb_returns).pow(2).mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (mb_old_logprobs - logprobs).mean().item()
                clipfrac = ((ratio - 1).abs() > config.clip_coef).float().mean().item()
                var_y = mb_returns.var().item()
                explained = 0.0 if var_y == 0 else 1 - (values - mb_returns).var().item() / var_y
            stats["losses/policy"] += policy_loss.item()
            stats["losses/value"] += value_loss.item()
            stats["losses/entropy"] += entropy.item()
            stats["losses/approx_kl"] += approx_kl
            stats["losses/clipfrac"] += clipfrac
            stats["losses/explained_variance"] += explained
            updates += 1

    if updates:
        stats = {k: v / updates for k, v in stats.items()}
    stats["losses/learning_rate"] = float(optimizer.param_groups[0]["lr"])
    stats["losses/old_values_mean"] = float(old_values_t.mean().item())
    return stats


def learning_rate_at(config: TrainingConfig, step: int) -> float:
    if not config.lr_anneal:
        return config.learning_rate
    fraction = min(max(step / max(config.total_steps, 1), 0.0), 1.0)
    return config.learning_rate * (1.0 - 0.9 * fraction)
