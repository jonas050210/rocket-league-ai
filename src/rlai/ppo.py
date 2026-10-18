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
    *,
    truncated: np.ndarray | None = None,
    bootstrap: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE over a (T, K) rollout.

    ``dones`` marks every episode end.  When ``truncated`` and ``bootstrap``
    are given, a time-limit truncation keeps bootstrapping from the value of
    the final state instead of treating it as a real termination.  Without
    that correction short episodes systematically bias the value function
    downwards: every 60 s timeout looks like a lost game.

    Two different masks are required at an episode boundary:

    * ``terminal`` (goal / real end) decides whether the *value* of the next
      state is bootstrapped.  A truncation is not terminal, so it bootstraps
      from ``bootstrap[t]``.
    * ``dones`` (any end, truncation included) decides whether the advantage
      recursion may continue.  The step after an episode end is the first
      step of a *different* episode, so its advantage must never leak back
      across the boundary.  Using ``terminal`` here instead made every
      timeout copy the next episode's advantage into the last decision of
      the finished one, which corrupts credit assignment on essentially
      every rollout (most Rocket League episodes end on the clock).
    """
    T = rewards.shape[0]
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)
    use_bootstrap = truncated is not None and bootstrap is not None
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        if use_bootstrap:
            is_truncated = truncated[t] > 0.5
            next_value = np.where(is_truncated, bootstrap[t], next_value)
            terminal = np.clip(dones[t] - truncated[t], 0.0, 1.0)
        else:
            terminal = dones[t]
        delta = rewards[t] + gamma * next_value * (1.0 - terminal) - values[t]
        # ``dones`` (not ``terminal``) cuts the recursion: an episode boundary
        # separates two independent trajectories.
        episode_continues = 1.0 - np.clip(dones[t], 0.0, 1.0)
        last_gae = delta + gamma * gae_lambda * episode_continues * last_gae
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
        truncated=rollout.get("truncated"),
        bootstrap=rollout.get("bootstrap"),
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
        "losses/grad_norm": 0.0,
    }
    updates = 0
    epochs_run = 0
    early_stopped = False

    model.train()
    for _ in range(config.epochs):
        epochs_run += 1
        indices = torch.randperm(n, device=device, generator=generator)
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            mb_obs = obs[idx]
            mb_actions = actions[idx]
            mb_old_logprobs = old_logprobs[idx]
            mb_advantages = advantages_t[idx]
            mb_returns = returns_t[idx]
            mb_old_values = old_values_t[idx]

            logits, values = model(mb_obs)
            dist = torch.distributions.Categorical(logits=logits)
            logprobs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            ratio = (logprobs - mb_old_logprobs).exp()
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            if config.clip_value > 0:
                clipped = mb_old_values + torch.clamp(
                    values - mb_old_values, -config.clip_value, config.clip_value
                )
                value_loss = 0.5 * torch.max(
                    (values - mb_returns).pow(2), (clipped - mb_returns).pow(2)
                ).mean()
            else:
                value_loss = 0.5 * (values - mb_returns).pow(2).mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
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
            stats["losses/grad_norm"] += float(grad_norm)
            updates += 1
            if config.target_kl > 0 and approx_kl > config.target_kl:
                early_stopped = True
                break
        if early_stopped:
            break

    if updates:
        stats = {k: v / updates for k, v in stats.items()}
    stats["losses/learning_rate"] = float(optimizer.param_groups[0]["lr"])
    stats["losses/old_values_mean"] = float(old_values_t.mean().item())
    stats["losses/epochs_run"] = float(epochs_run)
    stats["losses/early_stopped"] = 1.0 if early_stopped else 0.0
    return stats


def learning_rate_at(config: TrainingConfig, step: int) -> float:
    if not config.lr_anneal:
        return config.learning_rate
    fraction = min(max(step / max(config.total_steps, 1), 0.0), 1.0)
    return config.learning_rate * (1.0 - 0.9 * fraction)
