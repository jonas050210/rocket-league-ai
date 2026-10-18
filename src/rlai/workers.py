"""Parallel environment workers.

One process per environment.  Workers own the RocketSim arena, collect a
fixed number of decision steps with the current policy and return a rollout.
Weights are sent from the trainer each iteration; the policy is tiny
(~250k parameters), so this is cheaper than sharing memory on Windows.
"""

from __future__ import annotations

import time
import traceback
from typing import Any

import numpy as np
import torch

from .envs import MatchEnv
from .modes import get_mode
from .opponents import ModelOpponent, build_opponent
from .policy import PolicyConfig, TorchPolicy
from .rewards import RewardWeights

torch.set_num_threads(1)


def _empty_rollout() -> dict[str, np.ndarray]:
    return {
        "obs": [],
        "actions": [],
        "logprobs": [],
        "values": [],
        "rewards": [],
        "dones": [],
    }


def collect_rollout(env: MatchEnv, policy: TorchPolicy, n_steps: int, seed: int) -> dict[str, Any]:
    obs = env.reset(episode_seed=seed)
    agents = env.blue_agents
    n_agents = len(agents)
    obs = {agent: obs[agent] for agent in agents}
    rollout = _empty_rollout()
    episode_rewards = np.zeros(n_agents, dtype=np.float32)
    episode_start_tick = env.state.tick_count
    episodes: list[dict] = []
    started = time.perf_counter()

    for _ in range(n_steps):
        actions, logprobs, values = policy.act(obs, deterministic=False)
        result = env.step(actions)
        rewards = np.asarray([result.rewards.get(a, 0.0) for a in agents], dtype=np.float32)
        done = bool(result.done or result.truncated)

        rollout["obs"].append(np.stack([obs[a] for a in agents]).astype(np.float32))
        rollout["actions"].append(np.asarray([actions[a] for a in agents], dtype=np.int64))
        rollout["logprobs"].append(np.asarray([logprobs[a] for a in agents], dtype=np.float32))
        rollout["values"].append(np.asarray([values[a] for a in agents], dtype=np.float32))
        rollout["rewards"].append(rewards)
        rollout["dones"].append(np.full(n_agents, float(done), dtype=np.float32))

        episode_rewards += rewards
        if done:
            length = int(result.info.tick_count - episode_start_tick)
            if result.info.score_blue > result.info.score_orange:
                outcome = "win"
            elif result.info.score_blue < result.info.score_orange:
                outcome = "loss"
            else:
                outcome = "draw"
            episodes.append(
                {
                    "episode": int(env.episode_index),
                    "reward": float(episode_rewards.mean()),
                    "length": length,
                    "goals_for": int(result.info.score_blue),
                    "goals_against": int(result.info.score_orange),
                    "result": outcome,
                    "won": outcome == "win",
                }
            )
            obs_full = env.reset()
            obs = {agent: obs_full[agent] for agent in agents}
            episode_rewards[:] = 0
            episode_start_tick = env.state.tick_count
        else:
            obs = {agent: result.obs[agent] for agent in agents}

    _, _, last_values = policy.act(obs, deterministic=True)
    elapsed = time.perf_counter() - started
    return {
        "obs": np.stack(rollout["obs"]),
        "actions": np.stack(rollout["actions"]),
        "logprobs": np.stack(rollout["logprobs"]),
        "values": np.stack(rollout["values"]),
        "rewards": np.stack(rollout["rewards"]),
        "dones": np.stack(rollout["dones"]),
        "last_values": np.asarray([last_values[a] for a in agents], dtype=np.float32),
        "episodes": episodes,
        "steps": n_steps,
        "seconds": elapsed,
    }


def worker_main(remote, config_dict: dict, worker_index: int, seed: int) -> None:
    try:
        torch.set_num_threads(1)
        from .config import TrainingConfig

        config = TrainingConfig.model_validate(config_dict)
        spec = get_mode(config.mode)
        weights = RewardWeights.from_dict(config.reward.to_dict())
        opponent_name = "chase" if config.self_play else config.opponent
        env = MatchEnv(
            spec,
            build_opponent(opponent_name, seed=seed, device="cpu"),
            seed=seed,
            tick_skip=config.tick_skip,
            reward_weights=weights,
            episode_seconds=config.episode_seconds,
        )
        policy_config = PolicyConfig.from_dict(config.policy_config().to_dict())
        policy = TorchPolicy.from_config(policy_config, device="cpu")
        opponent_policy = TorchPolicy.from_config(policy_config, device="cpu")
        remote.send({"type": "ready", "worker": worker_index})
        while True:
            message = remote.recv()
            if message["cmd"] == "close":
                break
            if message["cmd"] != "collect":
                remote.send({"type": "error", "error": f"unknown command {message['cmd']!r}"})
                continue
            policy.model.load_state_dict(message["state_dict"])
            policy.model.eval()
            if message.get("opponent_state") is not None:
                opponent_policy.model.load_state_dict(message["opponent_state"])
                opponent_policy.model.eval()
                env.opponent = ModelOpponent(opponent_policy, name="self-play", deterministic=True)
            rollout = collect_rollout(env, policy, int(message["n_steps"]), int(message["seed"]))
            rollout["type"] = "rollout"
            rollout["worker"] = worker_index
            remote.send(rollout)
    except Exception as exc:  # pragma: no cover - defensive
        try:
            remote.send({"type": "error", "error": str(exc), "traceback": traceback.format_exc()})
        except Exception:
            pass
    finally:
        try:
            remote.close()
        except Exception:
            pass


class WorkerPool:
    def __init__(self, config, seed_offset: int = 0):
        import multiprocessing as mp

        self.ctx = mp.get_context("spawn")
        self.config = config
        self.workers = []
        self.parents = []
        base_seed = int(config.seed) + 100_003 * seed_offset
        for index in range(config.n_workers):
            parent, child = self.ctx.Pipe()
            process = self.ctx.Process(
                target=worker_main,
                args=(child, config.to_dict(), index, base_seed + index * 7_919),
                daemon=True,
                name=f"rlai-env-{index}",
            )
            process.start()
            self.workers.append(process)
            self.parents.append(parent)
        for index, parent in enumerate(self.parents):
            if not parent.poll(120):
                raise RuntimeError(f"worker {index} did not become ready within 120 s")
            message = parent.recv()
            if message.get("type") != "ready":
                raise RuntimeError(f"worker {index} failed to start: {message}")

    def collect(self, state_dict, n_steps: int, seed: int, opponent_state=None) -> list[dict]:
        for parent in self.parents:
            parent.send(
                {
                    "cmd": "collect",
                    "state_dict": state_dict,
                    "n_steps": int(n_steps),
                    "seed": int(seed),
                    "opponent_state": opponent_state,
                }
            )
        results = []
        for index, parent in enumerate(self.parents):
            if not parent.poll(600):
                raise TimeoutError(f"worker {index} timed out during rollout")
            message = parent.recv()
            if message.get("type") == "error":
                raise RuntimeError(
                    f"worker {index} error: {message.get('error')}\n{message.get('traceback', '')}"
                )
            results.append(message)
        return results

    def close(self) -> None:
        for parent in self.parents:
            try:
                parent.send({"cmd": "close"})
            except Exception:
                pass
        for process in self.workers:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
        for parent in self.parents:
            try:
                parent.close()
            except Exception:
                pass
