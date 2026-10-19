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

#: Torch thread count used inside a worker process.  A worker only ever runs
#: one tiny MLP, so more than one thread costs more than it saves.
WORKER_TORCH_THREADS = 1


def limit_worker_threads(threads: int = WORKER_TORCH_THREADS) -> int:
    """Pin Torch to ``threads`` threads *in this process only*.

    This used to run at import time, which meant that importing
    :mod:`rlai.workers` anywhere - including the trainer, which imports
    :class:`WorkerPool` - silently reduced the trainer to a single Torch
    thread.  The limit is now applied where it belongs: inside the worker
    process itself.
    """
    try:
        torch.set_num_threads(max(1, int(threads)))
    except Exception:  # pragma: no cover - platform specific
        pass
    return max(1, int(threads))


def _empty_rollout() -> dict[str, np.ndarray]:
    return {
        "obs": [],
        "actions": [],
        "logprobs": [],
        "values": [],
        "rewards": [],
        "dones": [],
        # time-limit truncations bootstrap from the value of the final state;
        # true terminations (goals) bootstrap with zero.
        "truncated": [],
        "bootstrap": [],
    }


def collect_rollout(
    env: MatchEnv, policy: TorchPolicy, n_steps: int, seed: int, *, fresh: bool = True
) -> dict[str, Any]:
    """Collect ``n_steps`` decision steps.

    ``fresh=True`` starts a new episode (first collection, or after the env
    was rebuilt).  ``fresh=False`` continues the episode the previous
    collection left unfinished: the old code reset on every call, which
    silently discarded one partial episode per worker per iteration and
    oversampled kickoff states.  Episode returns/lengths are accumulated by
    the env itself, so a spanning episode is reported whole.
    """
    if fresh or not env.current_obs:
        obs_full = env.reset(episode_seed=seed)
    else:
        obs_full = env.current_obs
    agents = env.policy_agents
    n_agents = len(agents)
    obs = {agent: obs_full[agent] for agent in agents}
    rollout = _empty_rollout()
    episodes: list[dict] = []
    started = time.perf_counter()
    infer_seconds = 0.0

    for _ in range(n_steps):
        inference_started = time.perf_counter()
        actions, logprobs, values = policy.act(obs, deterministic=False)
        infer_seconds += time.perf_counter() - inference_started
        result = env.step(actions)
        rewards = np.asarray([result.rewards.get(a, 0.0) for a in agents], dtype=np.float32)
        terminated = bool(result.done)
        truncated = bool(result.truncated) and not terminated
        done = terminated or truncated

        rollout["obs"].append(np.stack([obs[a] for a in agents]).astype(np.float32))
        rollout["actions"].append(np.asarray([actions[a] for a in agents], dtype=np.int64))
        rollout["logprobs"].append(np.asarray([logprobs[a] for a in agents], dtype=np.float32))
        rollout["values"].append(np.asarray([values[a] for a in agents], dtype=np.float32))
        rollout["rewards"].append(rewards)
        rollout["dones"].append(np.full(n_agents, float(done), dtype=np.float32))
        rollout["truncated"].append(np.full(n_agents, float(truncated), dtype=np.float32))
        if truncated:
            final_obs = {a: result.obs[a] for a in agents}
            inference_started = time.perf_counter()
            _, _, final_values = policy.act(final_obs, deterministic=True)
            infer_seconds += time.perf_counter() - inference_started
            rollout["bootstrap"].append(
                np.asarray([final_values[a] for a in agents], dtype=np.float32)
            )
        else:
            rollout["bootstrap"].append(np.zeros(n_agents, dtype=np.float32))

        if done:
            goals_for = int(env.score_for)
            goals_against = int(env.score_against)
            if goals_for > goals_against:
                outcome = "win"
            elif goals_for < goals_against:
                outcome = "loss"
            else:
                outcome = "draw"
            totals = [env.episode_reward_total.get(a, 0.0) for a in agents]
            events = result.info.reward_events
            episodes.append(
                {
                    "episode": int(env.episode_index),
                    "reward": float(np.mean(totals)) if totals else 0.0,
                    # length in *decision* steps (the unit PPO and the step
                    # counter use).  The old code stored physics ticks
                    # (120/s), which made a 15 s episode look 1800 "steps"
                    # long next to a 512-step rollout.
                    "length": int(env.episode_decisions),
                    "length_ticks": int(env.episode_decisions * env.tick_skip),
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "result": outcome,
                    "won": outcome == "win",
                    "truncated": bool(truncated),
                    "policy_team": int(env.policy_team),
                    # ball-interaction evidence for the curriculum and the
                    # report: reward alone cannot distinguish progress from
                    # proxy farming.
                    "touches": int(events.get("touches", 0)),
                    "shots": int(events.get("shots", 0)),
                    "defensive_touches": int(events.get("defensive_touches", 0)),
                    # reward components accumulated by the reward function for
                    # this episode (learning team only); persisted so the
                    # report can show *why* the reward moved.
                    "reward_components": {
                        key: float(value)
                        for key, value in result.info.reward_components.items()
                    },
                    "opponent": getattr(env.opponent, "name", "unknown"),
                }
            )
            obs_full = env.reset()
            obs = {agent: obs_full[agent] for agent in agents}
        else:
            obs = {agent: result.obs[agent] for agent in agents}

    inference_started = time.perf_counter()
    _, _, last_values = policy.act(obs, deterministic=True)
    infer_seconds += time.perf_counter() - inference_started
    elapsed = time.perf_counter() - started
    return {
        "obs": np.stack(rollout["obs"]),
        "actions": np.stack(rollout["actions"]),
        "logprobs": np.stack(rollout["logprobs"]),
        "values": np.stack(rollout["values"]),
        "rewards": np.stack(rollout["rewards"]),
        "dones": np.stack(rollout["dones"]),
        "truncated": np.stack(rollout["truncated"]),
        "bootstrap": np.stack(rollout["bootstrap"]),
        "last_values": np.asarray([last_values[a] for a in agents], dtype=np.float32),
        "episodes": episodes,
        "steps": n_steps,
        "seconds": elapsed,
        # measured separately so the report can attribute cost to the
        # simulator, to inference or to the learner instead of guessing.
        "infer_seconds": infer_seconds,
        "sim_seconds": max(elapsed - infer_seconds, 0.0),
        "policy_team": int(env.policy_team),
        "agents": list(agents),
    }


def worker_main(remote, config_dict: dict, worker_index: int, seed: int) -> None:
    try:
        limit_worker_threads()
        from .config import TrainingConfig

        config = TrainingConfig.model_validate(config_dict)
        spec = get_mode(config.mode)
        weights = RewardWeights.from_dict(config.reward.to_dict())
        opponent_name = "chase" if config.self_play else config.opponent
        if opponent_name == "self":
            # "self" needs a pool snapshot, which the worker does not have.
            # The trainer normally resolves this before it starts the workers;
            # falling back here keeps a misconfigured run from dying with an
            # opaque "unknown opponent" error.
            opponent_name = "chase"
        active_opponent = opponent_name
        active_reset_mode = str(config.reset_mode)
        active_episode_seconds = float(config.episode_seconds)

        def build_env() -> MatchEnv:
            return MatchEnv(
                spec,
                build_opponent(active_opponent, seed=seed, device="cpu"),
                seed=seed,
                tick_skip=config.tick_skip,
                reward_weights=weights,
                episode_seconds=active_episode_seconds,
                obs_version=config.obs_version,
                reset_mode=active_reset_mode,
            )

        env = build_env()
        fresh = True
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
            # A curriculum stage may change the reset distribution or the
            # episode horizon.  The horizon is baked into the truncation
            # condition, so the env is rebuilt (rare: once per transition);
            # the reset mode alone could switch live, but rebuilding for
            # both keeps one code path and the cost is one arena setup.
            wanted_reset = message.get("reset_mode") or active_reset_mode
            wanted_seconds = message.get("episode_seconds")
            wanted_seconds = (
                active_episode_seconds if wanted_seconds is None else float(wanted_seconds)
            )
            if wanted_reset != active_reset_mode or wanted_seconds != active_episode_seconds:
                old_opponent, old_team = env.opponent, env.policy_team
                env.close()
                active_reset_mode, active_episode_seconds = wanted_reset, wanted_seconds
                env = build_env()
                env.opponent, env.policy_team = old_opponent, old_team
                fresh = True
            requested_team = message.get("policy_team")
            if requested_team is not None:
                env.policy_team = int(requested_team)
            if message.get("opponent_state") is not None:
                opponent_policy.model.load_state_dict(message["opponent_state"])
                opponent_policy.model.eval()
                env.opponent = ModelOpponent(opponent_policy, name="self-play", deterministic=True)
                active_opponent = "self"
            else:
                requested = message.get("opponent") or active_opponent
                execution = message.get("opponent_execution")
                if requested != active_opponent:
                    # the execution override belongs to a difficulty level;
                    # a plain scripted opponent keeps its own behaviour
                    env.opponent = build_opponent(
                        requested,
                        seed=seed,
                        device="cpu",
                        execution=execution if requested.startswith("difficulty:") else None,
                    )
                    active_opponent = requested
                elif execution:
                    # the adaptive match dynamics changed the opponent's
                    # execution parameters without changing its behaviour
                    apply_execution = getattr(env.opponent, "apply_execution", None)
                    if apply_execution is not None:
                        apply_execution(execution)
            # Every worker gets its own deterministic episode seed; otherwise
            # all workers would replay the identical kickoff sequence.
            rollout = collect_rollout(
                env,
                policy,
                int(message["n_steps"]),
                int(message["seed"]) + worker_index * 104_729,
                fresh=fresh,
            )
            fresh = False
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
    """One environment process per worker, with fail-fast error reporting.

    A dead child used to look like a hang: the parent waited the full timeout
    before reporting anything.  Every wait now also watches the child's exit
    code and raises immediately with the real reason.
    """

    def __init__(
        self,
        config,
        seed_offset: int = 0,
        *,
        startup_timeout: float = 120.0,
        collect_timeout: float = 600.0,
    ):
        import multiprocessing as mp

        self.ctx = mp.get_context("spawn")
        self.config = config
        self.startup_timeout = float(startup_timeout)
        self.collect_timeout = float(collect_timeout)
        self.closed = False
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
        try:
            for index in range(len(self.parents)):
                message = self._await(index, self.startup_timeout, "start-up")
                if message.get("type") != "ready":
                    raise RuntimeError(f"worker {index} failed to start: {message}")
        except Exception:
            self.close()
            raise

    def collect(
        self,
        state_dict,
        n_steps: int,
        seed: int,
        opponent_state=None,
        opponent: str | None = None,
        policy_teams: list[int] | None = None,
        opponent_execution: dict | None = None,
        reset_mode: str | None = None,
        episode_seconds: float | None = None,
    ) -> list[dict]:
        for index, parent in enumerate(self.parents):
            team = None if policy_teams is None else int(policy_teams[index % len(policy_teams)])
            parent.send(
                {
                    "cmd": "collect",
                    "state_dict": state_dict,
                    "n_steps": int(n_steps),
                    "seed": int(seed),
                    "opponent_state": opponent_state,
                    "opponent": opponent,
                    "policy_team": team,
                    "opponent_execution": opponent_execution,
                    "reset_mode": reset_mode,
                    "episode_seconds": episode_seconds,
                }
            )
        results = []
        for index in range(len(self.parents)):
            message = self._await(index, self.collect_timeout, "rollout")
            if message.get("type") == "error":
                raise RuntimeError(
                    f"worker {index} error: {message.get('error')}\n{message.get('traceback', '')}"
                )
            results.append(message)
        return results

    def _await(self, index: int, timeout: float, what: str) -> dict:
        """Wait for one worker message, failing fast if the child died."""
        parent = self.parents[index]
        process = self.workers[index]
        deadline = time.monotonic() + max(0.1, float(timeout))
        while True:
            if parent.poll(0.2):
                return parent.recv()
            if process.exitcode is not None:
                raise RuntimeError(
                    f"worker {index} died during {what} (exit code {process.exitcode}); "
                    "see the run log for the traceback"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"worker {index} did not answer within {timeout:.0f} s during {what}"
                )

    def close(self) -> None:
        """Stop every worker and release every handle.  Safe to call twice."""
        if self.closed:
            return
        self.closed = True
        for parent in self.parents:
            try:
                parent.send({"cmd": "close"})
            except Exception:
                pass
        for process in self.workers:
            try:
                process.join(timeout=10)
            except Exception:
                pass
            if process.is_alive():
                process.terminate()
                try:
                    process.join(timeout=5)
                except Exception:
                    pass
            if process.is_alive():  # pragma: no cover - only on a stuck child
                process.kill()
        for parent in self.parents:
            try:
                parent.close()
            except Exception:
                pass
        self.parents = []
        self.workers = []
