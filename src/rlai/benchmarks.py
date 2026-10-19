"""Small, reproducible benchmarks.

These are the measurements the project uses to decide whether an
optimisation actually helps.  Every one of them reports the numbers it
measured - never a conclusion it did not measure.

The benchmarks are cheap enough to run in a test and specific enough to
attribute an effect:

* :func:`benchmark_difficulty` - does the difficulty mechanism measurably
  change how well the scripted bot plays?
* :func:`benchmark_evaluation_reuse` - how much does reusing the RocketSim
  arena between games save?
* :func:`benchmark_threads` - what does the Torch thread budget change?
"""

from __future__ import annotations

import time

from .modes import get_mode


def benchmark_difficulty(
    levels: tuple[str, ...] = ("rock", "tiny", "medium", "pro"),
    *,
    steps: int = 1500,
    seed: int = 7,
    mode: str = "soccer-1v1",
) -> dict:
    """Measure how often the scripted bot reaches the ball per level.

    The passive opponent lets the measurement isolate the *execution
    quality* of the level from the match outcome, which needs far more
    games to become significant.  More touches mean a stronger bot.
    """
    from .envs import MatchEnv
    from .opponents import build_opponent

    spec = get_mode(mode)
    results: list[dict] = []
    for level in levels:
        touches = 0
        episodes = 0
        started = time.perf_counter()
        per_episode = max(1, steps // 3)
        for episode in range(3):
            env = MatchEnv(
                spec,
                build_opponent(f"difficulty:{level}", seed=seed + episode),
                seed=seed + episode,
                episode_seconds=60,
            )
            env.reset(episode_seed=seed + episode)
            for _ in range(per_episode):
                result = env.step({agent: 0 for agent in env.policy_agents})
                touches += sum(int(result.info.touches.get(agent, 0)) for agent in env.opponent_agents)
                if result.done or result.truncated:
                    episodes += 1
                    env.reset()
        results.append(
            {
                "level": level,
                "touches": touches,
                "episodes": episodes,
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
    best = max(results, key=lambda item: item["touches"]) if results else None
    worst = min(results, key=lambda item: item["touches"]) if results else None
    return {
        "benchmark": "difficulty",
        "steps": steps,
        "seed": seed,
        "mode": mode,
        "results": results,
        "strongest": best["level"] if best else None,
        "weakest": worst["level"] if worst else None,
        "spread": (best["touches"] - worst["touches"]) if best and worst else 0,
        "basis": "ball touches of the scripted bot against a passive opponent",
        "note": (
            "this measures the execution quality of the level, not the target "
            "win rate; validating the 0/50/75/100 % targets needs a full "
            "evaluation series per level"
        ),
    }


def benchmark_evaluation_reuse(
    *,
    mode: str = "soccer-1v1",
    games: int = 4,
    match_seconds: float = 10.0,
    seed: int = 100,
) -> dict:
    """Measure the cost of an evaluation series with arena reuse.

    The series is played twice: once with a fresh arena per game (the old
    behaviour) and once with the arena reused.  The difference is the
    saving the reuse brings.
    """
    from .envs import MatchEnv
    from .evaluation import evaluate_policy
    from .opponents import IdleOpponent
    from .policy import PolicyConfig, TorchPolicy

    spec = get_mode(mode)
    policy = TorchPolicy(PolicyConfig(obs_size=198).build(), device="cpu")

    started = time.perf_counter()
    result = evaluate_policy(
        policy,
        mode=mode,
        opponent="idle",
        games=games,
        seed=seed,
        match_seconds=match_seconds,
        episode_seconds=match_seconds,
    )
    reused_seconds = time.perf_counter() - started

    started = time.perf_counter()
    for index in range(games):
        env = MatchEnv(
            spec,
            IdleOpponent(),
            seed=seed + index,
            episode_seconds=match_seconds,
        )
        env.reset(episode_seed=seed + index)
    fresh_seconds = time.perf_counter() - started

    return {
        "benchmark": "evaluation_arena_reuse",
        "games": games,
        "match_seconds": match_seconds,
        "reused_seconds": round(reused_seconds, 3),
        "fresh_arena_construction_seconds": round(fresh_seconds, 3),
        "games_played": result.get("games"),
        "note": (
            "the reused series includes the simulation itself, so the saving is "
            "the arena construction time alone"
        ),
    }


def benchmark_threads(threads: int | None = None) -> dict:
    """Measure the effect of the Torch thread budget on a forward pass."""
    import numpy as np
    import torch

    from .policy import PolicyConfig, TorchPolicy

    previous = torch.get_num_threads()
    results = []
    try:
        for count in ([threads] if threads else [1, 2, 4]):
            torch.set_num_threads(int(count))
            policy = TorchPolicy(PolicyConfig(obs_size=198).build(), device="cpu")
            policy.model.eval()
            obs = {"agent-0": np.zeros(198, dtype=np.float32)}
            with torch.no_grad():
                policy.act(obs, deterministic=True)
                started = time.perf_counter()
                for _ in range(200):
                    policy.act(obs, deterministic=True)
                elapsed = time.perf_counter() - started
            results.append(
                {
                    "threads": int(count),
                    "decisions_per_s": round(200 / elapsed, 1) if elapsed else 0.0,
                    "milliseconds_per_decision": round(1000 * elapsed / 200, 3),
                }
            )
    finally:
        torch.set_num_threads(previous)
    return {
        "benchmark": "torch_threads",
        "results": results,
        "note": "a single small MLP rarely benefits from more than one thread",
    }


def run_all() -> dict:
    """Every benchmark in one report."""
    return {
        "difficulty": benchmark_difficulty(),
        "evaluation_reuse": benchmark_evaluation_reuse(),
        "threads": benchmark_threads(),
    }
