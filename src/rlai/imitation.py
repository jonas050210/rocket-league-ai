"""Behaviour cloning from a scripted expert (warm start for PPO).

A fresh policy starts essentially uniform over 90 actions: the measured touch
rate of the first 150k steps is ~1.4 touches per (short) episode, and almost
all of that comes from the *drill* starting positions, not from a learned
approach to the ball.  PPO then has to discover "drive at the ball" from a
reward that barely changes between two uniform decisions.

The scripted ``chase`` opponent already reaches the ball more than ten times
per episode.  Cloning it into the actor gives PPO a policy that touches the
ball from the first iteration, which is exactly the missing prerequisite for
goal rewards to attach to anything.

The module is deliberately small and self-contained: it collects
``(observation, action)`` pairs from an expert that controls the learning
team, trains the actor with cross-entropy, and writes a *normal* checkpoint
(``label="pretrain"``, step 0) through the same code path as training.  A
later ``rlai train`` / ``main.py --steps N`` resumes from it like from any
other checkpoint - no special case anywhere else in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import checkpoints
from .actions import ACTION_COUNT
from .config import TrainingConfig, ensure_dirs
from .envs import MatchEnv
from .modes import get_mode
from .opponents import build_opponent
from .policy import PolicyConfig


@dataclass
class DemonstrationSet:
    """``(obs, action)`` pairs collected from the expert."""

    observations: np.ndarray
    actions: np.ndarray
    episodes: int = 0
    touches: int = 0
    #: per-episode diagnostics, useful in the CLI summary
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def samples(self) -> int:
        return int(self.actions.shape[0])

    @property
    def touches_per_episode(self) -> float:
        return float(self.touches) / self.episodes if self.episodes else 0.0


def _build_env(config: TrainingConfig) -> MatchEnv:
    """Deterministic single-process env for demonstration collection.

    The opponent is idle: the demonstrations should show the expert, not a
    duel whose outcome depends on a second scripted policy.
    """
    return MatchEnv(
        get_mode(config.mode),
        build_opponent("idle"),
        seed=int(config.seed),
        episode_seconds=float(config.episode_seconds),
        tick_skip=int(config.tick_skip),
        obs_version=int(config.obs_version),
        reset_mode=str(config.reset_mode),
        no_touch_timeout_seconds=float(config.no_touch_timeout_seconds),
    )


def collect_demonstrations(
    config: TrainingConfig,
    *,
    episodes: int = 200,
    expert: str = "chase",
    max_decisions: int | None = None,
    progress: Any = None,
) -> DemonstrationSet:
    """Roll out ``expert`` on the learning team and record what it does.

    The expert sees exactly the observations the policy sees (same
    ``obs_version``, ``tick_skip`` and starting-state distribution), so the
    recorded pairs are directly usable as a classification target for the
    actor.
    """
    if int(episodes) <= 0:
        raise ValueError("episodes must be positive")
    expert_policy = build_opponent(expert, seed=config.seed)
    env = _build_env(config)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    touches = 0
    lengths: list[int] = []
    budget = int(max_decisions) if max_decisions else None

    for episode in range(int(episodes)):
        obs = env.reset()
        steps = 0
        while True:
            agents = env.policy_agents
            actions = expert_policy.act(env, obs, agents)
            for agent in agents:
                rows.append(np.asarray(obs[agent], dtype=np.float32))
                labels.append(int(actions[agent]))
            result = env.step(actions)
            obs = result.obs
            steps += 1
            if result.done or result.truncated:
                break
            if budget is not None and len(labels) >= budget:
                break
        episode_touches = int(env.reward_events.get("touches", 0) or 0)
        touches += episode_touches
        lengths.append(steps)
        if progress is not None:
            progress(episode + 1, int(episodes), episode_touches)
        if budget is not None and len(labels) >= budget:
            break

    if not rows:
        raise RuntimeError("the expert produced no demonstrations")
    observations = np.stack(rows).astype(np.float32)
    actions = np.asarray(labels, dtype=np.int64)
    return DemonstrationSet(
        observations=observations,
        actions=actions,
        episodes=len(lengths),
        touches=touches,
        details={
            "expert": str(expert),
            "mean_episode_decisions": float(np.mean(lengths)) if lengths else 0.0,
            "obs_version": int(config.obs_version),
        },
    )


def train_imitation(
    demonstrations: DemonstrationSet,
    policy_config: PolicyConfig,
    *,
    epochs: int = 3,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 0,
    init_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train the actor to predict the expert's action with cross-entropy.

    Returns ``(model_state, metrics)``.  Only the actor head is used for the
    loss; the shared trunk and the value head are trained jointly by PPO
    afterwards, so it is enough that the action distribution starts out
    sensible.
    """
    import torch
    from torch import nn

    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    torch.manual_seed(int(seed))
    model = policy_config.build().to(device)
    if init_state is not None:
        # DAgger fine-tunes the current policy instead of restarting from a
        # random network: the new states are corrections, not a new task
        model.load_state_dict(init_state)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    loss_fn = nn.CrossEntropyLoss()

    obs = torch.as_tensor(demonstrations.observations, device=device)
    actions = torch.as_tensor(demonstrations.actions, device=device, dtype=torch.long)
    n = obs.shape[0]
    batch = max(1, min(int(batch_size), n))
    history: list[dict[str, float]] = []

    model.train()
    for epoch in range(int(epochs)):
        order = torch.randperm(n, device=device)
        total_loss = 0.0
        correct = 0
        seen = 0
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            logits, _ = model(obs[idx])
            loss = loss_fn(logits, actions[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * idx.shape[0]
            correct += int((logits.argmax(dim=-1) == actions[idx]).sum().item())
            seen += int(idx.shape[0])
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": total_loss / max(seen, 1),
                "accuracy": correct / max(seen, 1),
            }
        )

    metrics = {
        "samples": n,
        "actions": ACTION_COUNT,
        # a uniform policy agrees with any expert 1/90 of the time; anything
        # far above that means the network learned something from the expert
        "uniform_accuracy": 1.0 / ACTION_COUNT,
        "final_accuracy": history[-1]["accuracy"] if history else 0.0,
        "final_loss": history[-1]["loss"] if history else 0.0,
        "history": history,
    }
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return state, metrics


def _concat(first: DemonstrationSet, second: DemonstrationSet) -> DemonstrationSet:
    """Aggregate datasets: every DAgger round *adds* states, it never replaces."""
    return DemonstrationSet(
        observations=np.concatenate([first.observations, second.observations], axis=0),
        actions=np.concatenate([first.actions, second.actions], axis=0),
        episodes=int(first.episodes) + int(second.episodes),
        touches=int(first.touches) + int(second.touches),
        details={**first.details, "dagger_rounds": int(first.details.get("dagger_rounds", 0)) + 1},
    )


def _actions_from_model(model: Any, observations: dict, agents: list[str], device: str) -> dict[str, int]:
    import torch

    batch = np.stack([np.asarray(observations[agent], dtype=np.float32) for agent in agents])
    with torch.no_grad():
        logits, _ = model(torch.as_tensor(batch, device=device))
    return {
        agent: int(index)
        for agent, index in zip(agents, logits.argmax(dim=-1).detach().cpu().tolist())
    }


def collect_dagger_states(
    config: TrainingConfig,
    model: Any,
    *,
    episodes: int = 50,
    expert: str = "chase",
    beta: float = 0.5,
    device: str = "cpu",
    max_decisions: int | None = None,
    progress: Any = None,
) -> DemonstrationSet:
    """DAgger round: the *learner* drives, the expert labels what it sees.

    Behaviour cloning only sees states the expert visits.  The learner then
    drifts into states the expert never showed it (the classic distribution
    shift).  Here the learner rolls out its own trajectory, every visited
    state is labelled by the expert, and the dataset grows with exactly the
    mistakes the learner makes.  ``beta`` keeps a fraction of the actions
    under expert control so the rollout does not wander into pure noise.
    """
    if int(episodes) <= 0:
        raise ValueError("episodes must be positive")
    beta = min(max(float(beta), 0.0), 1.0)
    env = _build_env(config)
    expert_policy = build_opponent(expert, seed=config.seed)
    rng = np.random.default_rng(int(config.seed) + 7919)
    budget = int(max_decisions) if max_decisions else None
    rows: list[np.ndarray] = []
    labels: list[int] = []
    touches = 0
    lengths: list[int] = []

    for episode in range(int(episodes)):
        obs = env.reset()
        steps = 0
        while True:
            agents = env.policy_agents
            learner = _actions_from_model(model, obs, agents, device)
            expert_actions = expert_policy.act(env, obs, agents)
            for agent in agents:
                rows.append(np.asarray(obs[agent], dtype=np.float32))
                labels.append(int(expert_actions[agent]))
            executed = {
                agent: (
                    expert_actions[agent] if rng.random() < beta else learner[agent]
                )
                for agent in agents
            }
            result = env.step(executed)
            obs = result.obs
            steps += 1
            if result.done or result.truncated:
                break
            if budget is not None and len(labels) >= budget:
                break
        touches += int(env.reward_events.get("touches", 0) or 0)
        lengths.append(steps)
        if progress is not None:
            progress(episode + 1, int(episodes), touches)
        if budget is not None and len(labels) >= budget:
            break

    if not rows:
        raise RuntimeError("the learner produced no states")
    return DemonstrationSet(
        observations=np.stack(rows).astype(np.float32),
        actions=np.asarray(labels, dtype=np.int64),
        episodes=len(lengths),
        touches=touches,
        details={
            "expert": str(expert),
            "beta": beta,
            "mean_episode_decisions": float(np.mean(lengths)) if lengths else 0.0,
            "obs_version": int(config.obs_version),
        },
    )


def run_pretraining(
    config: TrainingConfig,
    *,
    episodes: int = 200,
    epochs: int = 3,
    expert: str = "chase",
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str | None = None,
    models_dir: str | Path | None = None,
    max_decisions: int | None = None,
    dagger_rounds: int = 0,
    dagger_episodes: int | None = None,
    dagger_beta: float = 0.5,
    dagger_epochs: int = 1,
    dagger_learning_rate: float | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Collect, imitate and save one ``pretrain`` checkpoint.

    With ``dagger_rounds > 0`` the recipe becomes DAgger: after the first
    imitation pass the learner drives its own episodes, the expert labels
    every visited state, and the enlarged dataset is trained again.  This
    fixes the states the learner reaches but the expert never visited.

    Measured caveat: with the built-in ``chase`` expert (79 % imitable)
    DAgger was neutral-to-worse than plain behaviour cloning in our A/B runs,
    and a large fine-tuning rate *degraded* the policy (0W 9L 11D vs 5W 7L 8D
    against idle+chase).  It is off by default and fine-tunes with
    ``learning_rate / 10``; use it with a stronger expert or more data.

    The checkpoint is written with ``step=0``: it is a warm start, not
    training progress, so a later resume that already has real checkpoints
    keeps preferring those (they sort by step).
    """
    paths = ensure_dirs()
    target = Path(models_dir) if models_dir else paths["models"]
    resolved_device = device or config.resolved_device()
    demonstrations = collect_demonstrations(
        config,
        episodes=episodes,
        expert=expert,
        max_decisions=max_decisions,
        progress=progress,
    )
    state, metrics = train_imitation(
        demonstrations,
        config.policy_config(),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=resolved_device,
        seed=int(config.seed),
    )
    rounds: list[dict[str, Any]] = []
    for round_index in range(int(dagger_rounds or 0)):
        model = config.policy_config().build().to(resolved_device)
        model.load_state_dict(state)
        model.eval()
        added = collect_dagger_states(
            config,
            model,
            episodes=dagger_episodes or max(1, int(episodes) // 4),
            expert=expert,
            beta=dagger_beta,
            device=resolved_device,
            max_decisions=max_decisions,
        )
        demonstrations = _concat(demonstrations, added)
        state, metrics = train_imitation(
            demonstrations,
            config.policy_config(),
            # fine-tune rather than restart: the aggregate mixes the easy
            # expert states with the harder learner states
            epochs=max(1, int(dagger_epochs)),
            batch_size=batch_size,
            learning_rate=float(
                dagger_learning_rate if dagger_learning_rate is not None else learning_rate / 10.0
            ),
            device=resolved_device,
            seed=int(config.seed) + round_index + 1,
            init_state=state,
        )
        metrics["dagger_round"] = round_index + 1
        rounds.append(
            {
                "round": round_index + 1,
                "added_samples": added.samples,
                "total_samples": demonstrations.samples,
                "final_accuracy": metrics["final_accuracy"],
                "final_loss": metrics["final_loss"],
            }
        )
    manifest = checkpoints.save_checkpoint(
        target,
        model_state=state,
        policy_config=config.policy_config().to_dict(),
        step=0,
        mode=config.mode,
        training_config=config.to_dict(),
        extra={
            "obs_version": int(config.obs_version),
            "phase": "imitation",
            "expert": str(expert),
            "demonstrations": {
                "episodes": demonstrations.episodes,
                "samples": demonstrations.samples,
                "touches_per_episode": round(demonstrations.touches_per_episode, 4),
                "dagger_rounds": len(rounds),
            },
            "imitation": {
                "final_accuracy": metrics["final_accuracy"],
                "final_loss": metrics["final_loss"],
            },
        },
        label="pretrain",
    )
    return {
        "checkpoint": manifest["id"],
        "path": str(target / manifest["path"]),
        "step": 0,
        "device": resolved_device,
        "obs_version": int(config.obs_version),
        "demonstrations": {
            "expert": str(expert),
            "episodes": demonstrations.episodes,
            "samples": demonstrations.samples,
            "touches": demonstrations.touches,
            "touches_per_episode": round(demonstrations.touches_per_episode, 4),
            "mean_episode_decisions": round(
                float(demonstrations.details.get("mean_episode_decisions") or 0.0), 2
            ),
        },
        "dagger": rounds,
        "training": metrics,
        "next": (
            "continue with: python -m rlai.cli train --config-file <file> "
            "--resume-from <checkpoint> --resume-mode continue --steps N"
        ),
    }
